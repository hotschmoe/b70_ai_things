# QuixiCore-XPU / QuixiInfer as a kernel and design reference for Strata concurrency (2026-11-19)

Scope: read-only review, no GPU touched. Clones pulled (both already up to date):
/mnt/cache/b70/refs/QuixiCore-XPU and /mnt/cache/b70/refs/QuixiInfer.
Strata compared: upstream v0.2.16 at /mnt/vm_8tb/b70/build/strata-0216/src/engine/sycl/src and our
night branch at /mnt/vm_8tb/b70/build/strata-ab/src-ep/sycl/src.

## Bottom line

QuixiCore-XPU is a broad, correctness-first library (SYCL + oneDNN, Python/PyTorch bindings, parity tests),
measured on 4x Arc Pro B60 (BMG-G21, 456 GB/s per card), not B70. Almost every kernel that matters for
Flash-Next decode is smaller and slower than what Strata already ships (Strata sycl/src has ~121k lines incl.
kernels/cuda/native_mmvq.dp.cpp 6.0k, iq_kernels.dp.cpp 8.0k, s2_expert_grouped 1.9k, verify_kernels 3.1k,
qsa_* 8k+, sampler 1.8k, prefill/esimd_moe 1.0k). QuixiCore-XPU has NO multi-column IQ3/IQ4/IQ2/Q5_K decode
GEMV, no IQ3_S/IQ2_S, no QSA/lightning-indexer sparse attention, no Strata-style EP exchange. Its own notes call
several kernels "correctness-first" or "honest baseline". Expect near-zero direct kernel-port gain for C2/C4.

QuixiInfer (Rust + CUDA, single RTX 5090, DeepSeek-V4.1-Flash now; Qwen3.8-Flash-Next earlier) is mostly derived
from Strata/Basalt ideas (its docs/strata-rca.md lists what it took FROM Strata) and is NVIDIA-only. Its value
is scheduling and policy ideas, and several are cheap host-side changes. Its own concurrency scaling is weak
(3 clients: 37 tok/s aggregate vs 27-30 single; 8x262k slots: 50 -> 80 aggregate at 4 clients), so it is not a
proof that its scheduler is better than Strata's (C1 141 -> C4 188 aggregate).

## 1. QuixiCore-XPU inventory

Registry: .quixicore/kernels.yaml (1005 lines), quant-formats.yaml (90), kernel-stubs.yaml (1588; planned
contracts, mostly stubs), backend.yaml. Dispatch: src/dispatch/*.cpp (13 files, thin routing). Kernels:
kernels/<family>/<op>/variants/xpu_sycl/*.sycl.cpp. Total xpu_sycl code for the families of interest is only
~4.9k lines. docs/capability-gaps.md is stale (2026-07-27 inventory; says 233 of 263 union ops are gaps).

Real implementations (SYCL, many with oneDNN XMX for GEMM):
- GGUF GEMV (kernels/quantization/gguf_gemv/variants/xpu_sycl/gguf_gemv.sycl.cpp, 695 lines): q8_0, q4_0, q6_K,
  q4_K/q5_K/q2_K/q3_K, iq4_nl/iq4_xs/iq2_xxs/iq2_xs/iq3_xxs/iq1_s (tables in gguf_iq_tables.hpp, 1345 lines).
  Code read: one 32-lane subgroup per row, scalar fp16/bf16 activation x decoded weight, M=1 only, no int8
  dot (no dp4a/DPAS), loads from the on-disk interleaved layout with the file header itself calling it
  "correctness-first ... a repack ... is the optimization". No IQ3_S, no IQ2_S, no multi-column.
- NVFP4/MXFP4/int4/fp8 GEMV and MoE (kernels/moe/nvfp4_moe, 639 lines; quantization/*). Best-tuned area:
  bit-relocation e2m1 decode 114 -> 220 GB/s of 456 (48 percent roofline), paired gate/up and output-row
  tiling. Not our formats.
- grouped_qgemm (kernels/moe/grouped_qgemm/variants/xpu_sycl/grouped_qgemm.sycl.cpp, 186 lines): static
  worst-case grid, per-expert prefix in SLM, DPAS via kernels/common/xmx_tile.hpp, w16/int4/nvfp4 stage lambdas.
  Self-described "honest baseline"; no IQ formats.
- gated_delta_rule varlen (linear_attention/gated_delta_rule, 204 lines): exact per-token recurrence, WG per
  (seq, v head), lane per v dim, state in registers, q/k in SLM. Chunked-DPAS variant explicitly deferred.
  qwen_gdn_decode (237 lines) with conv + recurrent update, state slots unique per batch.
- paged_attention decode/prefill (230/216 lines): one work-item per (seq, head, split), LSE-merge; "subgroup/DPAS
  tiled variant is the recorded throughput lever" (i.e. not done). mqa_logits (115 lines): fp8 MQA indexer
  logits on DPAS (DeepSeek indexer, not Strata's QSA).
- all_reduce (collectives/all_reduce, 230 lines) + src/runtime/ipc.cpp (Level Zero dma_buf export/open):
  capturable one-shot/two-shot P2P all-reduce, fixed rank order for bitwise equality, device-resident
  generation counters (so it records into a SYCL command graph), volatile-store flags plus system-scope
  fences because the B60 pairing reports access but NOT peer atomics. Measured 4x B60: 16 KiB one-shot
  0.069 ms, 64 KiB 0.091, 1 MiB two-shot 0.237 ms.
- sampling argmax (SLM tree, 4.9x), categorical (WG scan, 52x), renorm; dsv4_hc post only (72 lines, no pre/comb);
  moe_route/moe_permute; kv_cache_scatter/gather paged; norm_quant/glu_quant fusions; src/runtime/graph.cpp
  (SYCL command graph capture; XCCL kept between segments).

Stubs/planned only: mxfp8, mxfp6, fp4, awq, bitnet, most of kernel-stubs.yaml, hc pre/comb, chunked GDN,
tiled paged attention, MLA.

Measured numbers worth knowing (B60, not B70; 4x B60 box; oneAPI 2026.1, driver 1.14): oneDNN int8 ~182 TOPS
and bf16 ~90 TFLOP/s ceilings; native submit ~2.4 us/kernel vs ~23 us torch dispatch, graph replay ~1.3 us per
kernel at R>=128; w4a16 DPAS GEMM 1.53x at M=16 and 2.12x at M=32 vs per-row GEMV but 0.72x at M=8 and still
0.37-0.53x of the int8 SLM tile; M-tiled NVFP4 decode GEMV REJECTED for M<=6 (1.72x slower at M4, register
pressure), only wins M>=8. perf/scoreboard.md is empty (no hill-climb number ever recorded).

## 2. Mapping to Flash-Next concurrency needs

| Need | QuixiCore-XPU | Strata today | Verdict |
|---|---|---|---|
| 2-16 col IQ3_XXS/IQ3_S/IQ4_NL/IQ2_S/Q5_K/Q6_K/Q8_0 GEMV/MMVQ | M=1 scalar GEMV, no IQ3_S/IQ2_S, no int8 dot | KREG family 1-8 cols, MMVQ2 up to 16, int8 activations, IQ3S_LO; kernels/cuda/native_mmvq.dp.cpp, iq_kernels.dp.cpp, iq_multi_parity.cpp | Strata far ahead. Nothing to port. One data point: M-tiling loses at M<=6, wins at M>=8, which agrees with Strata's KREG 1-8 / MMVQ2 9-16 split |
| Dense 9-32 col via DPAS (C4 with deeper MTP) | w4a16_gemm DPAS tile pattern + xmx_tile.hpp (SLM-staged dequant lambda, bit-exact) | Own joint_matrix/ESIMD already (prefill/esimd_moe, xmx_moe, esimd_dpas_probe, xmx_int8_bench, qsa_prompt_attn_esimd) | Pattern only; their kernel is slower than int8 and unblocked. Low value |
| Grouped MoE decode+prefill, expert-major | grouped_qgemm (no IQ), nvfp4_moe | s2_expert_grouped.dp.cpp, prefill/esimd_moe.dp.cpp, xmx_moe.dp.cpp, 4-slot batch grouping | Strata ahead. Reusable idea: graph-safe static worst-case grid with in-kernel prefix walk (no host read of rows_per_expert) - check whether Strata's grouped decode path still syncs for counts |
| GDN recurrence / chunked prefill | exact serial recurrence, chunked deferred | gdn kernels, GDN2 deltas, deferred commit; chunked FP32 version was 4.4x slower than scan | No gain. Their register-state layout equals ours |
| QSA / lightning indexer | mqa_logits fp8 on DPAS (different indexer), paged attention naive | qsa_select (2.1k), qsa_decode_attn (2.6k), qsa_prompt_attn + ESIMD union | No gain |
| Hyper-connection reads | dsv4_hc_post only, one work-item per output | HC Q8 v2 (3.28 ms/window, 38 percent roofline) | No gain; DSV4 variant differs anyway |
| Sampling/argmax | SLM-tree argmax, WG-scan categorical | sampler.dp.cpp 1.8k, fused with verify | No gain |
| Cross-card IPC/P2P | all_reduce kernel + IPC layer, device-resident gen counters, fixed order | host-mapped doorbell flags (layer.cpp db.h_flag, session.cpp doorbell_wait), EP plan/psum return, remote_experts.cpp | Only partial relevance: see candidate 4 |
| Command-graph capture | src/runtime/graph.cpp, breakable segments | core/graph.cpp, dflash2.cpp finish_capture | Equivalent. Lesson already known: under capture launches are cheap, so +1 kernel can pay if it shortens the critical path |

## 3. QuixiInfer design ideas that transfer (two all-VRAM cards, 2-4 slots)

Relevant, ranked by fit:
- Variable-length verify: pick the verify prefix per window from acceptance estimates and a measured
  throughput-vs-width curve (DeepInfer schedule_mtp_varlen; QuixiInfer docs/deepseek41-plan.md lines ~189-190:
  draft length by cumulative acceptance vs threshold 0.35, adapted per request, scaled by cost). Direct fit to
  Strata's finding that C4 runs depth 1 and depth 2 / 12-row windows was slower: make depth per slot a function of
  active slot count and rolling acceptance.
- Speculation off beside other decoders (QX_DS_SPEC_MULTI=0; worker_ds.rs:451-453): a lone decoder drafts
  from history; beside others they get one token. Same trade-off as Strata's --lookup-chain (opt-in, generate.cpp
  ~750-761): gate lookup drafts on slot occupancy so they cannot widen the 4-slot window past the KREG/MMVQ2 sweet spot.
- Prefix-aware admission (QX_DS_ADMIT_WINDOW default 4, worker_ds.rs): of the waiting requests, admit first the
  one whose prefix a free slot would reuse. Directly addresses the observed 14-request burst at 2 slots where
  prefix reuse fell from 1000-1500 to 70-140 tok/s. Host-side only.
- Decoders sit a step out rather than run a width not in the captured WIDTHS set (width 5 aborted the sealed
  server): consistent with a fixed captured-graph width set; useful rule for 4-slot batch+MTP graph capture.
- Corpus/suffix speculation from other requests' replies (default on, 262k-token corpus, opt-out because reply
  speed leaks across parties; irrelevant for one household, relevant if exposed). Strata has suffix drafter
  (--suffix-draft); the cross-request corpus is the new piece. Gains are on repeats (essay twice 10.8 -> 6.4 s).
- Ring checkpoint / parked-and-resumed sequences bit-exact, cross-slot copies carry checkpoint: matches our
  conversation cache; their checkpoint-at-prompt-end trick (rewind to within 128 positions free) is a cheap add
  for agent turns that drop reasoning from history.
- Lending prompt buffers from the expert cache tail and returning them after prefill (their strata-rca item 2):
  Strata already does the equivalent (that is where QuixiInfer got it), so only useful as a check that our
  elastic-KV plan (docs/20261016_next_investigations.md item 0) uses the same pool.
- Resident experts grouped by slot so each weight is read once per window (qx_moe_groups): +1 percent only, CPU-bound
  there; Strata already groups. Rejected for us.
- Admit-on-copy cache swaps collapsed hit rate under mixed traffic: irrelevant, our experts are all VRAM-resident.
- Expert-major chunked prefill (8192-row chunks, every needed expert streamed once per layer): already how
  Strata prefill works; their data (500 tok/s on 8k tail, 1.1-1.4k tok/s on 16-25k) is below our 2000.

Not transferable: NVMe/RAM expert tiers, CPU expert pool, DeepInfer SM100 kernels, CUDA-graph Rust/CubeCL code,
single-GPU scheduling.

## 4. Licenses

- QuixiCore-XPU: MIT, Copyright 2026 QuixiAI (LICENSE). QuixiInfer: MIT, Copyright 2026 Eric Hartford.
- Strata: MIT, Copyright 2026 Niko1221 and the Strata contributors (repo LICENSE); engine/sycl also bundles dpct
  headers with their own LICENSE.TXT (Intel/LLVM terms).
- MIT into MIT: code can be copied into Strata PRs with attribution and the MIT notice retained (a short
  "adapted from QuixiAI/QuixiCore-XPU, MIT" comment plus NOTICE line). No copyleft barrier.
- Provenance caveat: many QuixiCore-XPU kernels state they are TRANSLATED from vllm-xpu-kernels (Apache-2.0):
  all_reduce (p2p_all_reduce.cpp commit dffcab7), grouped_qgemm, paged_attention, mqa_logits, dsv4_hc_post,
  gated_delta_rule. Their MIT header does not relieve the Apache-2.0 conditions (notice, state changes). Anything
  taken from those should be re-implemented from the algorithm description, or carry Apache attribution. GGUF
  tables (gguf_iq_tables.hpp) come from ggml (MIT) and Strata already has its own ISA decode.
- Ideas/algorithms (admission, varlen verify, device-flag handshake) are not copyrightable; re-implementing in
  Strata style is the clean route and also keeps upstream PRs small.

## 5. Ranked candidate list (impact on C2/C4 decode per stream, aggregate, prefill under load)

Rank 1. Acceptance- and occupancy-adaptive MTP depth per slot (QuixiInfer varlen verify idea)
- Impact on C2/C4: medium to high for C4 (the window width decides whether KREG 1-8 or MMVQ2 9-16 runs; we measured
  depth 2 slower at C4 but faster at C1), plausible +5-10 percent aggregate; unmeasured.
- Effort: medium (host policy in program/generate.cpp + verify.cpp; windows must stay in captured width set).
  Risk: low-medium (identity unchanged if only draft length changes; acceptance-normalised A/B needed).
- Files: QuixiInfer crates/qx-serve/src/worker_ds.rs (spec_max, SPEC_MULTI, WIDTHS), docs/deepseek41-plan.md ~189;
  Strata engine/sycl/src/program/generate.cpp (--spec, --mtp-max-t, --lookup-chain), core/mtp.cpp, core/verify.cpp.

Rank 2. Prefix-aware admission window across the 2-4 slots (plus gating lookup drafts on slot occupancy)
- Impact: medium for agent traffic (queue bursts, prefix reuse collapse), little for synthetic C-matrix runs.
- Effort: low (scheduler/serve layer; frontdoor or Strata server queue). Risk: low (fairness/starvation cap needed).
- Files: QuixiInfer crates/qx-serve/src/worker_ds.rs (QX_DS_ADMIT_WINDOW), crates/qx-engine/src/mailbox.rs
  (admit_on_release); Strata app/ serve path and engine slot admission (find in program/generate.cpp slot loop).

Rank 3. Graph-safe expert segmentation without host reads, and one-launch grouping across slots (QuixiCore grouped_qgemm pattern)
- Impact: low-medium; only if Strata's grouped decode/verify still has host-visible per-expert counts or extra
  syncs per layer at 4 slots (the verifier has 1,575 launches/window, 3.36 ms of in-order gaps at C1; at C4 this
  grows). Audit first, port the idea only if a host read exists.
- Effort: low for audit, medium to change. Risk: medium (touches the verify graph).
- Files: QuixiCore kernels/moe/grouped_qgemm/variants/xpu_sycl/grouped_qgemm.sycl.cpp (prefix walk in SLM, static
  worst-case grid); Strata sycl/src/kernels/cuda/s2_expert_grouped.dp.cpp, core/verify.cpp, kernels/cuda/verify_kernels.dp.cpp.

Rank 4. Device-resident P2P flag handshake (generation counters, fixed order) for the EP exchange, replacing host-mapped doorbells
- Impact: unknown, potentially medium for card balance (card 1 spins 15.5 of 20 ms; per-hop latency is the cost
  of any layer split or shared-expert-on-card-1 plan). QuixiCore data: 16 KiB one-shot 0.069 ms on B60 x4.
  Strata already moves activations GPU-side; whether the doorbell still crosses host memory in the C2/C4 path
  must be measured before any work.
- Effort: high (needs GPU, lease, multi-GPU safety rules; Strata peer path is its own). Risk: high (vLLM/oneCCL
  queue-handoff failures; B70 pair atomics support unverified).
- Files: QuixiCore kernels/collectives/all_reduce/variants/xpu_sycl/all_reduce.sycl.cpp, src/runtime/ipc.cpp,
  examples/p2p_all_reduce_mp.cpp (Apache-derived: re-implement); Strata core/layer.cpp (db.h_flag),
  core/session.cpp (doorbell_wait ~1221-1410), core/remote_experts.cpp.

Rank 5. DPAS-tiled dequant GEMM for 9-32 columns on dense/shared-expert weights (QuixiCore w4a16 + xmx_tile pattern)
- Impact: low today (C4 batch MTP depth 1 stays within 8 columns on KREG; their own kernel loses to int8 at all M);
  becomes relevant only if C4 runs 12-16+ row windows or 8-slot batching.
- Effort: high. Risk: medium. Files: QuixiCore kernels/common/xmx_tile.hpp, kernels/quantization/w4a16_gemm;
  Strata kernels/cuda/iq_kernels.dp.cpp (MMVQ2 9-16 variant), prefill/gemm.dp.cpp, prefill/xmx_moe.dp.cpp.

Rank 6. Ring-checkpoint at prompt end and cross-request suffix corpus (QuixiInfer)
- Impact: low for C-matrix, medium for agent sessions (cheap rewinds, repeated text). Effort: medium. Risk: low-medium
  (exactness test for rewinds; the corpus leaks timing across users).
- Files: QuixiInfer docs/handoff.md "Resume 2026-11-16"/"2026-11-19" sections; Strata conversation_snapshot.cpp, spec/.

Rank 7 (process, not kernel). Adopt QuixiCore's perf scoreboard idea: geomean-ms single score over a fixed row set
(perf/harness/perf_diff.py score) and append-only notebook with verdict vocabulary (perf/optimization_status.md).
- Impact: none on speed, helps keep concurrency experiments comparable. Effort low. Risk none.

Explicitly NOT recommended: porting QuixiCore GGUF GEMV, gated_delta_rule, paged_attention, mqa_logits, sampling,
HC post, or grouped_qgemm kernels. Each is behind Strata's current code (measured or by the authors' own notes).

## Verdict

CONFIG: read-only source review of QuixiCore-XPU (HEAD current, MIT) and QuixiInfer (HEAD current, MIT) vs Strata
v0.2.16 and night branch. COMMAND: file and code reading only; no GPU, no builds. RESULT: no Quixi kernel
outperforms Strata's for Flash-Next formats; measured numbers are B60, M=1 or NVFP4/fp8, not comparable. Policy ideas
from QuixiInfer (adaptive verify width, prefix-aware admission, occupancy-gated speculation) are the only items with
credible C2/C4 upside and low risk. VERDICT: do not adopt kernels; prototype ranks 1-2 behind default-off flags and
measure with the night harness; audit rank 3 and 4 with traces before writing code. Unmeasured speedup figures above
are hypotheses, not claims.
