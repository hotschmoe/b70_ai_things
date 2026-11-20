# DESIGN-A: Flash-Next on two B70s in TensorFold (Lead A, model path)

Author: Lead A (Claude Opus 5.5, claude-opus-5-5), 2026-11-20. Status: M-A1 design, v1.1 (reviewed by claude-fable-5-2).
Tags: [M] measured (cited), [R] read in source, [I] inferred / arithmetic. Inferred numbers are targets, not results.

## 0. Inputs read

- PR #3603 XPU Flash-Next (worktree /mnt/vm_8tb/b70/build/tensorfold/fn-model @ 207aaae): xpu_forward.zig,
  xpu_ops.zig, xpu_load.zig, xpu_store.zig, xpu_state.zig, xpu_ple.zig headers [R].
- Strata EP layout and window walk: concurrency-0216/analyst/fable_decode_window_walk.md, opus_sched_prefill_restore_walk.md,
  opus_zig_engine_design.md (rooflines), fable_tensorfold_review.md, tensorfold-probe/JOURNAL.md [R].
- GGUF header census of the served pack (this session): /mnt/vm_8tb/b70/results/tf-flashnext/census/gguf_census.json [M].
- Host link facts docs/P2P_GPU.md J.x as quoted in the Opus design: peer posted WRITE 11.3 GB/s, peer READ 3.24 GB/s,
  8 B peer latency 9.3 us, 10 KB push 7.9 us, unfused host-synced 2-rank push all-reduce 59.5 us at 10 KB [M, their logs].
- tf-xpu-tp-bench (pr3603 tp.zig collectives) HUNG on our host with a GuC job timeout on card 1 [M probe step f].

## 1. Decision: two-card split = PR3603 TP2 shape (head-split dense + expert-split MoE), new transport

Layout per layer (both cards run every layer concurrently):
- Replicated on both cards: embed (IQ3_S rows gathered), HC readout/writeback for attn and mlp (Q8_0 copies of the BF16
  HC weights, built at load, as Strata HC_Q8), router BF16 (512 + shared gate column), indexer q/k proj BF16 + pooled keys
  (indexer needs all 4 index heads; replicated as in PR3603), final mixer.
- Split by heads (rank r owns half): GDN qkv/z/b/a rows (q,k 8 of 16 key heads; v 24 of 48 value heads), conv, state,
  ssm_out columns; QSA q/gate rows (12 of 24 heads), 1 of 2 KV heads, KV cache, o_proj columns.
- Split by expert: 256 routed experts per card (placement table, section 4); shared expert split by FFN channel (320/320).
- Split by vocab: output head rows (124160 per card); argmax/LSE partials per card, combined on card 0.
- Two all-reduces per layer (after attn/GDN out-proj, after MoE combine) = 96 per window, sum in fixed rank order
  (r0 + r1 on both cards: replicated HC streams stay bit-identical across cards).

Why not Strata's EP (dense+attn+GDN+state on card 0, experts split):
- [I] Critical path per layer, EP = full dense on card 0 + max(card-0 experts + shared, hop + card-1 experts + hop).
  TP2 = half dense + hop + router + half experts + hop. Dense+HC fixed bytes are ~3.1 GB/window [R census]; halving the
  head-split part (~1.7 GB of attn/GDN/shared/head) saves ~2 ms/window at 425 GB/s; it costs 96 hops. Break-even hop
  ~20-40 us. A fused device-side push hop should be ~8-15 us [M 10 KB push 7.9 us + I for the flag]. So TP2 wins at C1
  unless the hop is slow; it also halves GDN state, KV and pooled-key bytes per card (capacity for 4 x 200K) and keeps
  card 1 busy (Strata card 1 spins 40 of 44 ms per C4 window [M-partial]).
- EP stays the documented fallback if Lead B's stable hop measures > 40 us (section 6 decision rule).
- TP2 also reuses PR3603's forward/state/MTP/lanes host code nearly unchanged; EP would be a new forward.

Not chosen: PP2 (re-reads dense per micro-batch), replicated dense (no latency gain over EP).

## 2. Fit (per card, TP2, from the census) [M bytes, I sums]

| Item | MiB per card |
|---|---:|
| Routed experts gate+up+down (40925 MiB total) | 20462 |
| Dense attn/GDN (1666 total, head-split) | ~833 |
| HC attn+ffn as Q8_0 copies + output mixer (replicated) | ~640 |
| Router BF16 + indexer BF16 (replicated) | ~158 |
| Shared expert (115 total, split) | ~58 |
| Head Q5_K (417 total, split) + embed IQ3_S (260, replicated) | ~468 |
| MTP drafter (Strata rt: 116 MB dense q8_0 + 708 MB experts, split) | ~420 |
| Total weights | ~23,000 (~22.5 GiB) |

Free after weights ~8.5-9 GiB per card. Per token per card (int8 KV as Strata, 1 KV head: 528 B + indexer key bf16 256 B +
pooled 64 B) x 12 layers ~10 KB -> 200K stream ~2.0 GB per card. GDN state 56.6 MB per slot per card (+ the same again
for the rollback copy). So 4 x 200K resident is borderline; C8/C16 x 200K needs Lead B's RAM tier (Flash-Next top-2048
sparse attention makes a VRAM block cache viable, Strata runs kv-resident 65536).
BF16 KV (PR3603 today) halves that capacity: int8 KV (Lead B, M-B3) is required for the 200K goal.

## 3. Quant formats (census) and kernel plan

Census of shard 1 (1223 tensors) [M]:
- Routed gate/up [2560 -> 640 x 512]: IQ2_XXS (9 layers), IQ2_XS (10), IQ2_S (10), IQ3_XXS (6), IQ3_S (13).
- Routed down [640 -> 2560 x 512]: Q2_0 type 42 (30 layers, 64-element blocks, 18 B), IQ4_NL (18 layers).
- Shared expert: gate/up IQ3_S/IQ4_XS/Q4_K/Q5_K/Q6_K; down IQ4_NL (40) / Q2_0 (8).
- Dense: attn_qkv, attn_gate, ssm_out, attn_q/k/v/output in IQ3_S, IQ4_XS, Q4_K, Q5_K, Q6_K (mixed per layer).
- BF16: HC down/up/inject, router ffn_gate_inp (+ _shexp), ssm_alpha/beta, indexer q/k proj, PLE key/value.
- F32: norms, ssm_a, ssm_dt.bias, ssm_conv1d [4 x 10240]; F16: ple_conv1d. Head output.weight Q5_K, token_embd IQ3_S.
- Shard 2: per_layer_token_embd IQ4_NL [160 x 320,001,536] (27.5 GiB, PLE, host RAM).
- No MTP tensors in the GGUF: drafter from Strata rt (/mnt/vm_8tb/b70/build/strata-ab/mtp-bf16/rt, q8_0 + bf16 + f32,
  experts.bin) or mtp-q2_0.gguf next to it.
So the device quant set is 11 types: Q4_K Q5_K Q6_K IQ4_XS IQ4_NL IQ3_S IQ3_XXS IQ2_S IQ2_XS IQ2_XXS Q2_0 (+ Q8_0 for the
MTP and HC copies, + BF16).

Kernel plan (OpenCL C via ocloc, TF build path):
1. Correct first: GGUF-native block layout per output row, a sub-group (16 lanes) per output row (or row tile), weights
   dequantized once into registers and applied to all R <= 16 activation rows (bf16 in, fp32 accumulate). One templated
   source, per-type dequant inline functions (ggml/Strata reference, MIT notices kept).
2. Grouped experts: same per-type dequant, driven by PR3603's moe_compact plan (active list, n_act), fused gate+up+SwiGLU,
   then down + weighted combine. Grids from the plan, not 10R.
3. Fast later (M-A5): load-time repack into lane-coalesced tiles (KREG-style, as Strata MMVQ KREG and PR3603's 16-col DPAS
   tiles), int8 activations + integer dot where the format allows (Strata's q8_1 path), DPAS tiles for R > 16 (prefill).
4. Oracles: ggml reference dequant (Strata third_party/strata-ggml ggml-quants.c, strata-gguf-py) for every type
   including Q2_0; GPU kernels compared against a CPU reference with fp64 accumulation over the same dequantized weights.

Activations: bf16 between ops as in PR3603 (Strata quantizes activations to q8_1 for MMVQ). Parity vs Strata is judged at
the logits level (top-1 agreement / KL), not bitwise.

## 4. Expert placement (balance = less waiting at the second all-reduce)

Per layer, a logical->(rank, local index) table replaces first_e/local_e in moe_compact. Default: contiguous halves.
Balanced: greedy assignment by Strata's expert profile (terminal/data/expert-profile-sycl.bin) so expected traffic per card
is equal. [I] At C1 (~27 distinct experts per layer per window) a random half split has sd ~2.6 experts per card ->
the slower card carries ~15% more than balanced on the routed part. Later candidate: replicate the hottest experts on both
cards and let moe_compact pick the less loaded card per window (memory permitting).

## 5. Critical path and waits (what the profiler must attribute)

Additions from the Fable review (claude-fable-5-2, /mnt/vm_8tb/b70/results/tf-flashnext/workers/review_fable.md):
- Device-side kernel boundaries: ~3-4 us x 480-960 kernels = 1.5-3.8 ms per window even with replay; own line item.
- Hop count is ~103 per window (96 + drafter + head), not 96.
- keepRows copies tmp_state -> state for 36 layers (56.6 MB per card per slot) at the next forward: make it a pointer swap.
- tp.zig host work: ~384 zeEventHostReset calls per forward + syncAll per MTP step: no host calls in steady state.
- Device-resident drafts conflict with the host PLE gather (verify rows need host n-gram rows before layer 1, ~250 us
  budget at C1): stream draft ids to a host-mapped ring per draft step; mlock the PLE table.
- Host wake latency for the one readback (event sync vs polled USM flag).
- Prefill exchanges: 2048 rows x 10 KB = 21 MB per exchange, ~1.9 ms x 96 = 0.18 s per chunk: overlap or bf16 wire.
- Overlap inside TP2: after pushing the attn/GDN partial, queue peer-independent work (KV/indexer append, saved_proj /
  state stores) before the wait; MoE hops stay exposed. Imbalance (~8.5 us/layer at C1) is only fixed by hot-expert
  replication with per-window choice or a tunable asymmetric expert share.
- Bit identity across cards: both cards must sum identical operands in the same order (no fp32-own + bf16-peer mix);
  replicated router top-10 and attention top-2048 selections must match - debug cross-card checksum of streams and ids.

Per window at C1 (d=3, 4 rows), target <= 14 ms (Strata 23.17 ms [M]); ceiling ~11.7 ms TP2 [I Opus table].
Wait sources, in expected order of size [I]:
1. Host appends: PR3603 appends ~10-20 kernels per layer from the host per forward (~1000 per window, no graphs). If the
   host append rate (~2-5 us each) is slower than GPU execution of small kernels, the cards wait for work.
   Fix ladder: fewer/fused kernels; record regular command lists once per row bucket and replay them
   (zeCommandListImmediateAppendCommandListsExp) with a device-side window descriptor.
2. All-reduce hops: 96 per window x hop latency, plus imbalance (the faster card waits). Fused push hop (producer writes
   its partial into the peer inbox with posted writes, then a flag; consumer waits and sums in rank order).
3. Host round trips per window: end-of-verify sync + logits readback, and one sync + readback per MTP draft step in
   PR3603 (forwardMtp syncAll). Fix: keep draft tokens on device (argmax writes the next embed ids), confidence stop on
   device with masked rows, one host readback per window.
4. Logits gather: PR3603 card 0 copies card 1's 124160-wide shard logits per row (a peer READ, slow direction, ~77 us/row
   at 3.24 GB/s [I]). Fix: per-card argmax/LSE partials, push 256 partials only.
5. PLE: host gather overlaps layer 0 (PR3603 already queues layer 0 before the gather) - verify it stays hidden.

## 6. Transport (coordination with Lead B)

Interface Lead A codes against: tp.zig TpComm.allReduceF32(gpus, lists, null, dst[], n) and syncAll, unchanged
signatures, so the forward code does not change when the transport does.
Requirements for Lead B's stable transport: push (posted writes) not pull; no host involvement per hop; deterministic
rank-order sum; sizes 10 KB x rows (decode 1-16 rows = 10-160 KB; prefill chunks up to 2048 rows = 20 MB, copy engine
above a threshold); no GuC timeout under sustained 96 hops per forward. Target hop <= 15 us at 10-160 KB.
Decision rule (v1.1, after the Fable review): chained p50 hop <= 35 us -> TP2; > 60 us -> EP; between -> decide on a
measured skeleton window (below). Measured on this host [M P2P_GPU.md via review]: CS cross-device event wait 17-22 us
standalone (K.3/K.5), a full all-reduce inside an XPUGraph 35 us (K.6), Strata ep_wait ~12 us/layer; host-mediated paths
34-90 us. So the hop is NOT 8-15 us: budget ~20-35 us x ~103 hops/window = 2-3.6 ms per C1 window - the single largest
wait term after kernel boundaries, and the first thing to overlap.
Recommended primitive (review + Lead B): per rank, producer writes its partial; AppendMemoryCopy local -> peer inbox on the
SOURCE card (push) signalling an event from a HOST_VISIBLE pool spanning both devices; peer AppendWaitOnEvents; local-only
sum (fusable into the next hcReadout prologue); in-list event reset. Race it against Strata's producer-flag + peer spin
kernel over a 96-hop chain (p50/p99). In-kernel mutual flags hang (J.9-C), pull is 3.24 GB/s, reduce-into-peer 2.36 GB/s.
Skeleton-window bench (asked of Lead B, latbench): 48 layers x [dummy kernels reading the real per-card byte counts of
section 2 + 2 exchanges] at R = 4 and 8 - gives the decision number before M-A3.
Lead A's own correctness fallback until then: XPU_TP_HOST_AR=1 (sync both, D2H both partials, CPU rank-order sum, H2D),
slow but host-mediated only; it is a test path, never a speed number.

## 7. Milestone plan (Lead A)

v1.1 changes: placement is table-driven in the loader (per role: replicated / row-split / col-split / expert map / card k)
so EP stays a configuration, not a rewrite; column splits assert block alignment (256 K/IQ, 64 Q2_0, 32 IQ4_NL);
M-A2 gets a speed gate (one dense type + one expert type at R = 4, GB/s vs 608) before the row layout is final, and the
loader repacks per type through a function the kernel owns (layout changes do not touch the loader); every semantic-map
item (qkv order, norm offsets, A_log sign, conv layout, HC stream order, PLE hash, expert axis) gets a single-op test
against a CPU reference; M-A3 adds a one-window go/no-go (C1 window <= 14 ms at 4 rows, C4 <= 20 ms at 8 rows,
unprofiled) and a cross-card selection checksum; a CPU fp32 per-layer reference is built only if teacher-forced top-1
agreement is ambiguous (garbage = semantic bug, ~97%+ = noise level; in between needs layer-level localization).


- M-A2 (in flight): worker K = quant kernel library + CPU reference + per-type parity test (tf-xpu-fn-gguf-test);
  worker M = GGUF semantic map (GGUF-MAP.md: names, layouts, norm offsets, A_log/ssm_a, qkv order, conv, HC stitching,
  PLE hash vs metadata, MTP rt mapping) + Strata oracle recipe. Then worker L = GGUF loader (GgufStore + loadAllRanksGguf,
  QW descriptors, TP2 head splits, Q8_0 HC copies) wired through xpu_ops.
- M-A3: two-card forward (host AR first, Lead B transport when ready), greedy fixed prompts, oracle Strata
  (teacher-forced top-1 agreement over a few hundred positions + first-token logits KL), deterministic reruns.
- M-A4: MTP drafter (Strata rt), device-resident drafts, PLE from host RAM, lanes multi-stream; concurrent==solo and
  drafted==serial gates.
- M-A5: speed with Lead B's profiler: host-append starvation, hop latency/imbalance, logits gather, kernel speed (repack,
  int8 dot, DPAS prefill), expert placement. C1/C2/C4/C8/C16 + prefill.

## 8. Interfaces (for workers)

    pub const QType = enum(u8) { f32, f16, bf16, q8_0, q4_k, q5_k, q6_k, iq4_xs, iq4_nl, iq3_s, iq3_xxs, iq2_s, iq2_xs, iq2_xxs, q2_0 };
    pub const QW = struct { t: QType, w: ?*anyopaque = null, n: usize = 0, k: usize = 0, row_bytes: usize = 0 };
    // n output rows, each row = k inputs in GGUF-native blocks (row_bytes = k / block_elems * block_bytes)
    pub const QE = struct { t: QType, w: ?*anyopaque = null, n: usize = 0, k: usize = 0, row_bytes: usize = 0, experts: usize = 0 };
    // experts contiguous: expert e at w + e * n * row_bytes
    linearQ(d, qw, x_bf16, y, rows, .bf16 | .f32)    // replaces linearBf16/linearF32 for GGUF weights
    moeUpQ(d, gate: QE, up: QE, mixed_bf16, active, n_act, moe_act_bf16, max_pairs)
    moeDownQ(d, down: QE, moe_act_bf16, active, n_act, moe_down_f32, max_pairs)
Kernel file: zig/kernels/xpu/flashnext/fn_gguf.cl (+ fn_gguf_tables.cl for grids), added to the flashnext module build.


## v1.2 (2026-11-20 08:30 UTC, Lead A) - transport decision taken on measured numbers
- Lead B latbench (TRANSPORT-B.md section 3, PROVISIONAL, card pair, 2000 reps, data verified): PUSH copy + host-scope
  event, producer kernel end -> consumer kernel end, one-way p50/p99 = 12.7/28.9 us (4 KB), 12.2/18.4 (16 KB),
  17.3/22.2 (64 KB), 34.8/37.9 (256 KB). Decode exchanges (R x 10 KB, R <= 16 = 10-160 KB) sit at ~12-28 us p50.
- Decision rule (section 6: <= 35 us -> TP2) is MET at p50 for every decode size. TP2 (PR3603 shape) is confirmed;
  the EP fallback is retired unless the skeleton-window measurement shows exposed hop cost > 4 ms per C1 window.
  Budget [I]: ~100 hops x ~15 us = ~1.5 ms per window if fully exposed (target window 14 ms at C1).
- No record/replay graph extension on this driver. Regular (closed) command lists execute 1024 kernels for ~3 us host
  cost vs ~2.7 us PER kernel for immediate appends. M-A5 plan: record each decode window shape (R rows) once as regular
  command lists per card (2 lists per layer boundary at most), with all per-step values (positions, lengths, slot ids,
  draft ids) read from device buffers written by one small H2D at window start. Host work per window becomes: one H2D,
  one execute per list segment, one sync. This replaces "kernel appends" as the first M-A5 latency item.
- Host round trip inside a window costs ~14 us + ~8 us device restart: the PR3603 MTP syncAll per draft step and the
  host logits peer-read must go (device-side argmax/draft loop, logits gather by push).
