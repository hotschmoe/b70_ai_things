# Decode launch census: EP C2 verifier, MTP pair, commit (Strata SYCL, 2x B70)

Analyst: Claude Opus 5.5 (claude-opus-5-5), 2026-10-16. READ-ONLY source audit plus
CPU-only reads of existing logs. No build, no GPU, no source edit.

CONFIG -> Source `/mnt/vm_8tb/b70/build/strata-ab/src-ep` (branch night-20261016,
HEAD 53402da2). Serving config as briefed: `--expert-parallel-device 1 --spec 4
--mtp-max-t 4 --kv int8`, STRATA_BATCH_MTP=1, DEPTH=2, PAIR_HEAD=1, EP prefill
reduce/push, MMVQ_WIDE/A2, GDN_COMMIT_SPLIT=1, other candidates 0. `--native`
is on, so native bf16 shared-expert gate, native router/combine/QSA/rope are on;
PLE is `--ple-io direct` (NVMe), all 24576 experts resident (zero-doorbell graph).

COMMAND -> Read verify.cpp, mtp.cpp, generate.cpp (batch_step), graph.cpp,
dpct graph/device headers, kernel launchers. Re-parsed the host_boundary ring of
S136 (profiled) and S142 (UNprofiled, same binary/flags) engine logs with a
30-line script; cross-checked counts against the sibling trace analysis
(`agents/trace_gap_analysis.md`, `trace_gap_csv/representative_layers_cycle5.txt`).

RESULT -> Per-layer launch lists, per-cycle submission/host-block map, event and
reset origin, and 10 ranked opportunities below. Source counts reconcile with
the trace: card1 337 kernels/verifier exactly; card0 2,209 by source vs 2,226
measured (17 unattributed).

VERDICT -> The verifier replay is ~83% of the unprofiled 46.5 ms C2 window and
is a serial chain of ~46 small kernels per layer on card0. The largest
structural items are: (1) the shared-expert queue fork/join makes the verifier
and MTP graphs non-linear, so every kernel node gets an L0 sync event and the
command buffer ends with a 2,226-command eventReset tail that the host waits
through; (2) C2 batch attention runs one row per launch (24 launches per QSA
layer where the solo window uses 4); (3) the paired MTP draft reads the full
437 MB vocabulary head twice per cycle. Host gaps are small without unitrace
(about 1.3 ms/window), so the 10.7 ms "no GPU activity" in S136 is mostly
profiler inflation.

-----------------------------------------------------------------------------

## 0. Ground truth timing (why the profiled picture misleads)

Same binary/flags, host_boundary ring, median over 22 steady S=6 windows
(`.../current-mtp-profile-s136/run/pair-engine-complete.log` vs
`.../current-unprofiled-ring-s142-prep/run/engine.log`):

| Host span (marker pair) | S136 unitrace | S142 unprofiled | GPU0 state during span |
|---|---:|---:|---|
| stage_ple_begin -> end (PLE gather, Direct I/O) | 0.709 ms | 0.685 ms | idle (both cards) |
| verify_submit_begin -> end (2 graph appends) | 3.241 | 0.073 | idle |
| verify_wait_begin -> end (`cs_->wait()`) | 43.868 | 38.635 | verifier + reset tail |
| run_return -> commit_return (submit+wait) | 2.921 | 1.584 | commit graph |
| slot_copy_begin -> end (x2) | 0.098 each | 0.020 each | ~idle |
| pair_submit_begin -> end (6 appends) | 1.950 | 0.048 | idle until 1st append |
| pair_poll_begin -> pair_wait_begin | 1.100 | 2.706 | MTP |
| pair_wait_begin -> done | 2.925 | 2.615 | MTP |
| inter-window gap | 2.2 us | 0.8 us | idle |
| Window total | ~56.0 | ~46.4 | |

Unprofiled split: verifier ~38.6 ms (83%), MTP ~5.4 ms (12%), commit ~1.6 ms
(3%), PLE 0.7 ms (1.5%), other host ~0.15 ms. Card1 computes only ~9.75 ms per
cycle (17%) and spins in `ep_wait` ~26.8 ms (sibling trace). All estimates
below are against the 46.5 ms unprofiled window and are NOT additive.

-----------------------------------------------------------------------------

## 1. Queues, graphs, submissions per C2 cycle

All queues are in-order: `create_queue(true)` -> `create_in_order_queue`
(dpct/device.hpp:616-627; the `true` is the exception handler, not order).
DPCT_PROFILING_ENABLED is #defined at the top of verify.cpp:2, mtp.cpp,
device.dp.cpp, session.cpp ... but not generate.cpp or ep_kernels.dp.cpp, so
`create_queue_impl` (device.hpp:678-692) adds `enable_profiling` depending on
which inline instance the linker keeps (ODR); S140 already flagged this.

| Queue | Card | Created | Used for |
|---|---|---|---|
| `Verifier::cs_` | 0 | verify.cpp:685 | verifier window graph, commit graph |
| `Verifier::sh_cs_` | 0 | verify.cpp:693 | shared-expert branch (pulled into the window graph) |
| `Verifier::copy_` | 0 | verify.cpp:673 | unused in all-resident EP mode |
| `Verifier::ep_q_` | 1 | verify.cpp:1822 | peer half of the window (own graph) |
| `MtpDrafter::cs_` slot0 = `MtpPair::cs_` | 0 | mtp.cpp:527, 2115 | all 6 pair graphs |
| `MtpDrafter::side_` (per slot) | 0 | mtp.cpp:536 | MTP shared-expert branch |
| `dpct::get_in_order_queue()` | 0 | default | slot residual copies (eager) |

Graph recording: `dpct::experimental::begin_recording(q)` creates a
`command_graph(ctx, dev)` with NO properties and records the queue
(dpct/graph.hpp:42-56); `end_recording` calls `graph->end_recording()` (all
queues, including transitively recorded ones, graph.hpp:58-67); `finalize()` is
called with NO properties (verify.cpp:3017, 3040, 3248; mtp.cpp:1220).

Submissions per steady C2 cycle (9 graph appends + 2 eager copies; matches the
trace's 9 AppendCommandListsExp per cycle):

| Order | What | Call site | Queue | Graph shape | Nodes |
|---:|---|---|---|---|---:|
| 1 | peer verifier graph `ep_exec_bm_` | verify.cpp:3417 | ep_q_ (card1) | linear | 337 kernels |
| 2 | main verifier graph `exec_bm_` | verify.cpp:3419 | cs_ (card0) | DAG (fork/join per layer) | 2,226 kernels + 192 barrier nodes |
| 3 | batch commit graph `commit_bm_` | verify.cpp:3605 | cs_ | linear | 244 kernels |
| 4,5 | slot residual copy x2 (eager) | generate.cpp:9454 | default queue | - | 1 memcpy each, `.wait()` |
| 6,7 | pre_round A, pre_round B | mtp.cpp:2437 | pair cs_ | DAG (MTP shared branch) | ~49-56 kernels each |
| 8 | tail depth 0 (2-col head) | mtp.cpp:2437 | pair cs_ | linear | 9 kernels |
| 9,10 | pre_step A, pre_step B (j=1) | mtp.cpp:2439 | pair cs_ | DAG | ~49 kernels each |
| 11 | tail depth 1 | mtp.cpp:2439 | pair cs_ | linear | 9 kernels |

-----------------------------------------------------------------------------

## 2. Verifier forward, batch window S=6 (rows = slot0 x3, slot1 x3), EP

Recorded once per row layout by `capture_batch` (verify.cpp:2985-3068):
`begin_recording(cs_)` 2994, `begin_recording(ep_q_)` 2999, `record_window`
3007, separate `end_recording` for cs_ (3012) and ep_q_ (3014). Every op below
is a node of a captured graph; nothing is eager. `G == 1` for batch windows
(verify.cpp:825), `lean == false` (needs device_plan_, which is off when
all-resident). Kernel names are the trace names.

### 2.1 Window prologue (once)

| # | Kernel | Source | Queue |
|---:|---|---|---|
| 1 | ep_bump (window epoch) | verify.cpp:854 -> ep_kernels.dp.cpp:84 | cs_ |
| p0 | ep_bump (peer epoch) | verify.cpp:855 | ep_q_ |
| 2-4 | copy_i32_from_mapped tok/step/pos (host-mapped -> device) | verify.cpp:858-860 | cs_ |
| 5 | NativeEmbed gather | verify.cpp:874 | cs_ |
| 6 | broadcast_streams (emb -> 4 HC streams) | verify.cpp:875 | cs_ |

### 2.2 GDN layer (36 of 48), card0

Every cs_ node depends on the previous cs_ node (in-order recording). "B" rows
are empty/barrier nodes that only add DAG edges.

| # | Kernel | Enqueued at | Q | Extra dependency / note |
|---:|---|---|---|---|
| 1 | hc_gr_norm_split_k (apply prev write) | verify.cpp:1012 gr_read_group(0) -> fused_gr.dp.cpp:1926 | cs | reads prev combine bo_, inj2_ |
| 2 | hc_gr_down_sl | fused_gr.dp.cpp:1965-1978 | cs | |
| 3 | hc_gr_down_reduce_k | fused_gr.dp.cpp:1981 | cs | |
| 4 | hc_gr_up_cols<16> | fused_gr.dp.cpp:2043 | cs | writes mixed_ |
| 5 | native_quantize_q8_1 (x->xq_) | verify.cpp:1034 | cs | QFUSE off |
| 6 | mmvq qkv (M10240) | verify.cpp:1035 | cs | |
| 7-8 | gdn_conv_l2_multi x2 (one per slot group) | verify.cpp:1044 | cs | batch_rec_ loop 1040-1047 |
| 9 | gdn_ab_multi | verify.cpp:1052 | cs | |
| 10 | mmvq gate z (M6144) | verify.cpp:1056 | cs | |
| 11-12 | gdn_step_norm x2 (per slot, state read-only) | verify.cpp:1065 | cs | loop 1059-1068 |
| 13 | native_quantize_q8_1 (y, K6144) | verify.cpp:1074 | cs | |
| 14 | mmvq ssm_out (M2560) -> bo_ | verify.cpp:1075 | cs | |
| 15-18 | HC ffn read (norm/down/reduce/up, apply=1) | verify.cpp:1295 | cs | |
| B1 | fork: `sync_barrier(ev_fork_, cs)` | verify.cpp:1308 | cs | `ext_oneapi_submit_barrier()` |
| B2 | `sh_cs_->ext_oneapi_submit_barrier({ev_fork_})` | verify.cpp:1309 | sh_cs_ | transitive recording of sh_cs_ |
| 19 | bf16_gemv_mmvf_multi router logits (512) | verify.cpp:1325 | cs | |
| 20 | native_router_top10_multi | verify.cpp:1327 | cs | |
| 21 | resident_plan_ep (role 1, card0 entries) | verify.cpp:1346 -> verify_kernels.dp.cpp:2451 | cs | single WG, 6.3 us |
| 22-28 | shared expert: quantize->sh_xq_, mmvq gate, mmvq up, swiglu_quantize_q8_1, mmvq down, bf16 gemv shared-gate, sigmoid_scale_rows | verify.cpp:1406 -> shared_expert.dp.cpp:313-380 | sh_cs_ | after B2 only |
| B3 | join: `sync_barrier(ev_join_, sh_cs_)` | verify.cpp:1415 | sh_cs_ | |
| 29 | quantize_q8_1_rows (x->nat_xq_) | verify.cpp:1419 | cs | same bytes as #22 (QDEDUP blocked by fork) |
| 30 | ep_push (ids + q8 rows to peer inbox, then flag) | verify.cpp:1430 -> ep_kernels.dp.cpp:88 | cs | 1 WG of 256 |
| 31 | routed_gu_port (card0 entries) | post: verify.cpp:1483 -> iq_kernels.dp.cpp:5353 | cs | grid {40, cap=60} |
| 32 | native_expert_grouped (swiglu+q8_1 of h) | iq_kernels.dp.cpp (same call) | cs | |
| 33 | routed_down_port -> parts_ (own rows) | same call | cs | |
| 34 | ep_wait (spin on ep_ret0_ ring) | verify.cpp:1499 -> ep_kernels.dp.cpp:107 | cs | device-side flag from card1 |
| B4 | `cs->ext_oneapi_submit_barrier({ev_join_})` | verify.cpp:1550 | cs | joins shared branch |
| 35 | native_moe_combine_multi -> bo_ | verify.cpp:1559 | cs | reads peer rows in parts_ |

GDN layer = 35 kernels on card0 + 4 barrier nodes.

### 2.3 QSA layer (12 of 48: l = 3, 7, ..., 47), card0

| # | Kernel | Enqueued at | Q | Note |
|---:|---|---|---|---|
| 1-4 | HC attn read | verify.cpp:1012 | cs | |
| 5 | native_quantize_q8_1 | verify.cpp:1107 | cs | |
| 6 | bf16_gemv_mmvf_multi indexer k_proj | verify.cpp:1108 | cs | qb=true |
| 7 | mmvq k | verify.cpp:1112 | cs | |
| 8 | mmvq v | verify.cpp:1113 | cs | |
| 9 | native_qsa_rms_norm_rope (K) | verify.cpp:1114 | cs | |
| 10-11 | copy_from_mapped tail snapshot x2 (per slot) | verify.cpp:1126 | cs | |
| 12-17 | kv_append_q8_step x6 (one per ROW) | verify.cpp:1166 (loop 1152-1177) | cs | batched form skipped for batch_rec_ (1135) |
| 18-23 | native_qsa_indexer_append x6 (one per ROW) | verify.cpp:1181 (loop 1178-1185) | cs | |
| 24 | mmvq q|gate (M12288) | verify.cpp:1187 | cs | |
| 25 | native_qsa_rms_norm_rope (Q, fused split) | verify.cpp:1191 | cs | |
| 26 | bf16_gemv_mmvf_multi indexer q_proj | verify.cpp:1213 | cs | |
| 27 | native_qsa_rms_norm_rope (qidx) | verify.cpp:1215 | cs | |
| 28-51 | per ROW x6: qsa_block_scores (1251), qsa_block_topk (1253), [qsa_kv_resolve 1255 = no-op, kv_mode 0], qsa_attn_chunk + qsa_attn_merge (1257 -> qsa_decode_attn.dp.cpp:644) | verify.cpp:1248-1259 | cs | chunk grid {33,2,1}: ~26% of HW threads; 42.7 us first row of a slot, 30.6 us others |
| 52 | native_qsa_gate_apply | verify.cpp:1276 | cs | |
| 53 | native_quantize_q8_1 (attn32, K6144) | verify.cpp:1287 | cs | |
| 54 | mmvq o (M2560) -> bo_ | verify.cpp:1288 | cs | |
| 55-58 | HC ffn read | verify.cpp:1295 | cs | |
| 59-75 | MoE/EP block identical to GDN #19-35 (17 kernels) + B1-B4 | | | |

QSA layer = 75 kernels on card0 + 4 barrier nodes. Trace (sibling): QSA
select/chunk/merge 254 us/layer, attention prep 36.6 us/layer.

### 2.4 MoE expert-parallel exchange, in detail

Memory: inbox/inflag live on card1, parts/ret on card0; each is allocated on its
owner and IPC-mapped into the other card's context (`ep_region_alloc`,
verify.cpp:1806-1807; `EpRegion` ep_mem.hpp:16). The peer writes its rows
straight into card0's `parts_` (verify.cpp:1815: `parts_ = ep_rows0_`).
Separate contexts per card, so no SYCL event can cross cards: all cross-card
ordering is device-side flag polling. The epoch counter (ep_bump) makes flags
monotonic per window; `ring = (l - lb) * G + grp + 1`.

Per layer, in record order (card1 ops are recorded into the ep_q_ graph by the
same `record_window` call, verify.cpp:1421-1448):

| Step | Card | Kernel | Source | Signal / wait |
|---|---|---|---|---|
| a | 0 | ep_push: copies ids (60 x i32) + q8_1 rows (6 x 2880 B) with P2P stores, `atomic_fence(release, system)` + system-scope group barrier, lane0 `sys_store(peer inflag[ring], ctr0)` | verify.cpp:1430; ep_kernels.dp.cpp:88-104 | SIGNAL card1 (P2P write of flag) |
| b | 1 | ep_wait: single work-item spin `sys_load(inflag[ring]) >= ctr1`, bound 2,000,000 reads, then acquire fence | verify.cpp:1432; ep_kernels.dp.cpp:107-119 | DEVICE wait in graph; blocks ep_q_ |
| c | 1 | ep_copy_uncached: inbox -> ep_in1_ with uncached L1/L3 loads (frees inbox, avoids stale lines) | verify.cpp:1433; ep_kernels.dp.cpp:17-26 | |
| d | 1 | resident_plan_ep role 2 (single WG) | verify.cpp:1436 | |
| e | 1 | routed GU, swiglu+q8, routed down -> ep_rows1_ | verify.cpp:1442 | |
| f | 1 | ep_send_rows: one WG per entry (60); copies the 10 KB fp32 row of each PEER entry into card0 parts_ via P2P float4 stores; release fence; atomic counter; last WG resets counter and `sys_store(card0 ret[ring], ctr1)` | verify.cpp:1445; ep_kernels.dp.cpp:122-143 | SIGNAL card0 |
| g | 0 | (meanwhile) own GU/swiglu/down, then ep_wait spin on `ret0[ring]` | verify.cpp:1483, 1499 | DEVICE wait in graph |
| h | 0 | barrier on ev_join_ (shared branch), combine | verify.cpp:1550, 1559 | |

No host wait, no host flag, no host task anywhere in the exchange; both halves
are inside captured graphs (card0's DAG, card1's linear graph). The host only
sees the end (`cs_->wait()` then `ep_q_->wait()` in `ep_check`, verify.cpp:1870).
Reuse safety: card0's next push (layer l+1) is after its ep_wait for layer l,
which follows card1's send, which follows card1's copy -> the inbox is free.

Measured per layer (sibling trace, layer 8): card0 push ends 356 us after the
previous combine; card1 copy 360, plan 364, GU 371-458, swiglu 460, down
465-526, send 529-559 (29.7 us, ~430 KB PCIe at ~15 GB/s); card0 own GU/down
356-493, ep_wait 493-561, and the 7 shared-expert kernels actually execute
493-545, i.e. DURING the ep_wait spin (the DAG topological order puts the
branch after ep_wait in the command list). Card1 chain push->return 203-220 us;
card0 local 160 us + shared ~51 us; card0 exclusive wait 21-28 us/layer,
1.25 ms/cycle. Card1 idles ~430-670 us per layer during card0's dense phase.

### 2.5 Layer-1 PLE block, last layer, head

| What | Kernels | Source |
|---|---:|---|
| wait_flag_ge(m_flag_, 1) (host raises after submit) | 1 | verify.cpp:921 |
| copy_from_mapped h_ple_ -> ple_ | 1 | verify.cpp:922 |
| PLE key/value bf16 GEMV multi | 2 | verify.cpp:940-944 |
| gr_write_multi | 1 | verify.cpp:967 |
| per ROW x6: native_ple_postops (gate, broadcast, conv_residual), ple_history_advance, copy hist snapshot | 30 | verify.cpp:977, 982, 987 |
| last layer gr_write_multi | 1 | verify.cpp:1572 |
| head: HC read (4), quantize, mmvq_head 6 col (Q5_K, 437 MB), argmax/sample | ~7 | verify.cpp:1654, 1669, 1670, 1680 |

### 2.6 Counts

| Scope | card0 kernels | card0 barrier nodes | card1 kernels |
|---|---:|---:|---:|
| GDN layer | 35 | 4 | 7 |
| QSA layer | 75 | 4 | 7 |
| Prologue | 6 | 0 | 1 |
| PLE (layer 1) + last write + head | 35 + 1 + ~7 | 0 | 0 |
| Forward (source) | 36x35 + 12x75 + 49 = 2,209 | 192 | 337 |
| Forward (trace, sibling) | 2,226 (incl. 48 ep_wait) | - | 337 (incl. 48 ep_wait) |

Per-forward families on card0 (launches): HC 388, dense MMVQ ~301, attention
select/chunk/merge 288, QSA appends 144, GDN 180, quantize ~180, MoE/EP ~816.

-----------------------------------------------------------------------------

## 3. MTP draft (paired head, depth 2) per cycle

`MtpPair::draft` (mtp.cpp:2309-2489), default path (PAIR_GRAPH=0, OWNER_QUEUES=0):
graphs cached per (slot, T) and (slot, j); launched back to back on the pair
queue (slot0 drafter cs_), no host decision between them (mtp.cpp:2437-2439).

pre_step(j=1) body = `copy_from_mapped_multi` (2 records, mtp.cpp:2196) +
`record_forward(1,row,head=false)`:

| # | Kernel | Source (mtp.cpp) | Q |
|---:|---|---|---|
| 1 | copy_from_mapped_multi | 2196 | pair |
| 2 | NativeEmbed gather | 843 | pair |
| 3 | rms_norm (pre_fc_norm_embedding) | 851 | pair |
| 4 | quantize_q8_1 | 852 | pair |
| 5 | mmvq fc_embedding | 853 | pair |
| 6 | rms_norm (pre_fc_norm_hidden) | 855/857 | pair |
| 7-8 | quantize hn + mmvq fc_hidden (4 cols per 8-col chunk) | 860-861 | pair |
| 9 | add_streams_broadcast | 863 | pair |
| 10-13 | HC attn read (4) | 876 | pair |
| 14 | quantize | 879 | pair |
| 15-16 | mmvq k, mmvq v | 880-881 | pair |
| 17 | norm_rope K | 893 | pair |
| 18 | kv_append_q8_steps | 911 | pair |
| 19-20 | mmvq q, norm_rope Q | 959, 962 | pair |
| 21-22 | qsa attn chunk + merge (dense over every cell) | 983 | pair |
| 23-25 | gate_apply, quantize, mmvq o | 985-987 | pair |
| 26-29 | HC mlp read (4) | 1001 | pair |
| B | fork `sync_barrier(sh_fork_)` + side_ barrier | 1042-1044 | pair/side |
| 30-36 | shared expert (quantize, gate, up, swiglu_q8, down, gate gemv, sigmoid) | 1053 | side_ |
| B | join `sync_barrier(sh_join_, side_)` | 1066 | side |
| 37-38 | router bf16 gemv, native_router_top10 | 1077-1078 | pair |
| 39-40 | moe_group_resident, quantize_q8_0_scaled | 1082, 1084 | pair |
| 41-43 | moe_grouped_s2 (GU, swiglu, down) | 1085 | pair |
| B | `ext_oneapi_submit_barrier({sh_join_})` | 1088 | pair |
| 44-45 | native_moe_combine, gr_write_multi | 1102, 1114 | pair |
| 46-49 | head-mix HC read (4) -> sample_ | 1139 | pair |

pre_step = 49 kernels + 4 barriers. pre_round(T=a+1) adds a 7-record mapped
copy (2166), `copy_row_to_first` + quantize when T>1 (2169-2170), a second
fc_hidden chunk when T=3, and rms_norm + T ropes instead of the fused K
norm_rope when T>1: 49 (T=1) to 56 (T=3). tail(j) = quantize x2 (2221-2222),
2-column `native_mmvq_head` over the full 248,320-row Q5_K head (2223, 1.21 ms),
per slot `sample_tokens` (single WG, ~89 us) + `row_top_prob` (2224 ->
1193-1196), `mtp_select` x2 (2226-2227) = 9 kernels.

Per cycle: ~213 body kernels + 18 tail kernels + 16 barrier nodes; trace: 209
eventResets in the bodies, 0 in the (linear) tails. Host: `idle()` waits x2
(2350), mapped staging, 6 appends, spin on mapped `h_out_[0]` for both slots
with PLE prefetch hints (2461-2467), `cs_->wait()` (2469).

## 4. Commit (batch) and solo differences

`capture_commit_batch` (verify.cpp:3074-3262), STRATA_GDN_COMMIT_LAYERS=0,
QSA_COMMIT_GROUP_STEPS=0. Launched at verify.cpp:3605 after host acceptance
(generate.cpp:9394-9401), then `cs_->wait()` at 3622 unconditionally.

| Kernels | Count | Source |
|---|---:|---|
| copy_i32_from_mapped commit rows (per slot) | 2 | 3141 |
| per GDN layer per slot: gdn_conv_commit + gdn_step_commit (COMMIT_SPLIT=1, state read+write 3.27 MB) | 144 | 3183, 3185 |
| per QSA layer per slot: copy tail snapshot back + 3 x native_qsa_indexer_append | 96 | 3201-3220 |
| PLE history copy_indexed (per slot) | 2 | 3230 |
| Total (linear, in-order, 0 resets in trace) | 244 | |

GDN step commits are 72 x 16.1 us = 1.16 ms (bandwidth: ~470 MB state traffic).

Solo C1 (`Verifier::run` 2172-2637): window T<=4 uses the multi-row forms
(single conv/step, single kv/indexer append, single scores/topk/attn per QSA
layer): ~32 kernels per GDN layer, ~43 per QSA layer; the PLE gather happens
AFTER launch (2377-2383); commit is async (`set_commit_async(!multi_gpu)`,
generate.cpp:3309; verify.cpp:2836-2866); drafting is round + steps on the
drafter queue with h_out_ polling (mtp.cpp:1586-1790). The same fork/join and
reset tail apply to its verifier and MTP graphs.

-----------------------------------------------------------------------------

## 5. Graph structure, host blocks, what the GPU does meanwhile

Host sequence of one C2 window (`batch_step`, generate.cpp:9329-9550):

| # | Host action | Where | Blocks? | GPU0 | GPU1 | S142 |
|---:|---|---|---|---|---|---:|
| 1 | batch_pack, apply_pending | generate.cpp:9350, 9375 | no | idle | idle | ~0 |
| 2 | stage_batch: refresh_ar (24576x2 scan), LRU, staging, ngram rows, PLE `gather_batch` (Direct NVMe, p50 304 us/read, 31.7% cache hits) | verify.cpp:3268-3393, gather 3365 | CPU + I/O | idle | idle | 0.71 ms |
| 3 | append peer graph, then main graph | verify.cpp:3417, 3419 | no | starts | starts, spins in layer-0 ep_wait | 0.07 |
| 4 | raise PLE flag | verify.cpp:3446-3450 | no | | | |
| 5 | `cs_->wait()` | verify.cpp:3453 | YES (zeCommandListHostSynchronize) | 37.2 ms replay, then 2,226-reset tail | 9.75 ms compute, 26.8 spin | 38.6 |
| 6 | `copy_->wait()`, `ep_check` -> `ep_q_->wait()`, read mapped h_out_ | 3472-3478, 1870 | short | idle | idle | 0.004 |
| 7 | accept (batch_keep), write h_commitb_, append commit, `cs_->wait()` | generate.cpp:9399; verify.cpp:3605, 3622 | YES | commit 1.5 ms | idle | 1.58 |
| 8 | BT printf; per slot default-queue memcpy final_R rows -> slot rows + `.wait()` | generate.cpp:9427, 9454-9457 | YES x2 | 4 us of copy | idle | 0.05 |
| 9 | MtpPair: `idle()` x2, mapped staging, 6 appends | mtp.cpp:2350, 2437-2439 | short | idle until 1st append | idle | 0.05 |
| 10 | spin on mapped h_out_[0] per slot, PLE prefetch hints | mtp.cpp:2461-2467 | spin | MTP | idle | 2.71 |
| 11 | pair `cs_->wait()` | mtp.cpp:2469 | YES | MTP | idle | 2.61 |
| 12 | fflush, loop | generate.cpp:9519 | no | idle | idle | ~0 |

GPU0 idle caused by the host (unprofiled): ~0.7 (PLE) + ~0.1 (submit) +
~0.1-0.3 (acceptance/commit turnaround) + 0.05 (copies) + 0.05 (pair staging)
~= 1.0-1.3 ms per window. GPU1 is idle for everything except step 5. Sampling
is device-side (greedy argmax recorded in the graph; `sample_rows` only
launches for non-greedy slots, verify.cpp:3658-3673). No host task, host
sampling, or host memcpy of logits is on the greedy path; tokens come back
through mapped memory (`m_out_`).

## 6. Where events, barriers and resets come from

1. Queues are in-order and graphs are recorded from them, so a single-queue
   recording is a linear chain. The commit graph (244 kernels), MTP tails and
   the peer graph (337 kernels) are single-queue: the trace shows ZERO
   eventResets for them. This is the in-repo natural experiment.
2. The verifier and MTP bodies add cross-queue edges every layer:
   `dpct::sync_barrier` (= `ext_oneapi_submit_barrier()` because
   DPCT_PROFILING_ENABLED is defined, device.hpp:989-993) on cs_, then
   `sh_cs_->ext_oneapi_submit_barrier({ev_fork_})`, join on sh_cs_, and
   `cs->ext_oneapi_submit_barrier({ev_join_})` (verify.cpp:1308-1309, 1415,
   1550; mtp.cpp:1042-1044, 1066, 1088). Transitive queue recording pulls
   sh_cs_/side_ into the graph. The graph is a DAG with 48 (or 1 per body)
   fork/join diamonds, so the DPC++ partition is not single-path and the UR
   command buffer is created out-of-order (DPC++ `isInOrder` requires a
   single-path partition, no N-D copy nodes and no graph profiling; verify in
   the installed 2026.1 runtime, adapter is UR Level-Zero V2 per S140).
3. Out-of-order command buffer: every kernel node gets a signal event and a
   wait list; the buffer ends with one `zeCommandListAppendEventReset` per
   event. Measured: 2,226 resets per verifier replay (= kernel count, the 192
   barrier nodes are folded into wait lists) and 209 per cycle in MTP bodies.
   The host's `cs_->wait()` returns only after the reset tail (profiled tail
   span 5.43 ms: 2.14 ms resets + 3.29 ms of ~1.46 us gaps). Unprofiled cost
   is unmeasured; S142's verify wait is 5.2 ms shorter than S136's.
4. Barriers outside graphs: `CapturedGraph::launch` (graph.cpp:135, 155) adds
   a barrier after each launch, but the decode path calls `ext_oneapi_graph`
   directly (no). The 10 L0 barriers/cycle are graph-submission ordering and
   the MTP barrier nodes.
5. Launch properties: 398 launch sites carry `use_root_sync`
   (`grep -c` over sycl/src), but no kernel uses a root group. DPC++ treats
   `use_root_sync` as a cooperative launch request. Whether the L0 command
   list records `zeCommandListAppendLaunchCooperativeKernel` for these is
   unverified; one trace API grep answers it.
6. What removes them: (a) record the window on ONE queue (no fork/join): the
   partition becomes single-path, in-order command buffer, counter-based
   events, no resets; (b) keep `finalize()` without `enable_profiling`
   (already so); (c) avoid 2-D memcpy nodes (only on !fuse_nr paths,
   verify.cpp:1202/1230, mtp.cpp:971 - not taken); (d) no host_task nodes
   (none on the all-resident path). `no_events`/enqueue-functions only help
   eager submissions (2 per cycle here), not graph nodes.

-----------------------------------------------------------------------------

## 7. Ranked opportunities (top 10)

Ranking = expected unprofiled saving x confidence / cost. Savings are per
46.5 ms C2 window, independent estimates, not additive.

### 1. Single-queue (in-order) verifier graph: move the shared expert into post()

- Removes: 2,226 eventResets per verifier replay (+192 barrier nodes, per-node
  event waits); 48 duplicate quantizes with QDEDUP. Est. -1.0 to -2.5 ms.
- Where: verify.cpp:1300-1416 (fork + shared block in pre), 1483-1499 (post),
  1547-1550 (join), 1396-1400 (QDEDUP gate requires !sh_fork).
- Sketch: when EP+all-resident, record the shared expert on cs_ in post()
  AFTER the own routed GU/swiglu/down (1483) and BEFORE `ep_wait` (1499), with
  `x_q8_1_ready = nat_xq_` (QDEDUP). Do not record any sh_cs_ barrier. Order:
  HC ffn -> router -> top10 -> quantize_rows -> ep_push -> plan -> own experts
  -> shared -> ep_wait -> combine. The trace shows the shared kernels already
  execute inside the ep_wait window today (layer 8: 493-545 us inside
  493-561 us), so card0's timeline is unchanged except that the reset tail and
  event waits disappear. Arithmetic: same kernels, same inputs, bitwise.
- Gate: one short trace (or a tiny standalone graph oracle, S135-style) that
  eventResets drop to ~0 and the command list is in-order; then a matched C2.
- Risk: low-medium. S15 measured plain STRATA_SH_STREAM=0 at -0.86%, but that
  arm put the shared expert BEFORE ep_push (delaying card1 by ~51 us/layer,
  ~2.4 ms/forward) and never checked that resets vanished. Do the MTP side
  separately: STRATA_MTP_SHARED_BRANCH=0 (mtp.cpp:1026-1093) linearizes the
  bodies (-209 resets) but serializes ~50 us/body of shared work.

### 2. Group C2 attention by slot in the verifier (3 rows per launch)

- Removes: per QSA layer 24 -> 8 select/chunk/merge launches and 12 -> 4
  append launches = 288 launches/forward. Est. -1.2 to -1.9 ms (select+attn
  254 us/layer today; first row of a slot 42.7 us, others 30.6 us; chunk grid
  uses 26% of HW threads, z-batched rows share the slot's KV in L2).
- Where: verify.cpp:1124-1186 (appends: `!batch_rec_` guard at 1135 and 1178)
  and 1248-1259 (per-row scores/topk/attn).
- Sketch: iterate contiguous slot groups (as the GDN code does at 1040-1047)
  and call the existing multi-row forms with that slot's QsaState:
  `kv_append_q8_steps(n=3)`, `native_qsa_indexer_append_steps(n_steps=3)`,
  `qsa_block_scores/topk(nq=3)`, `qsa_decode_attn_batch(n_q=3)`. This is
  exactly the solo T-row window path (1262-1273), so per-row arithmetic is
  unchanged; all three rows' K/V are appended before attention, as today.
- Risk: low. Distinct from S139/S144 (commit-side grouping) and S11 (prefill).

### 3. Hide the PLE gather behind the verifier launch (batch path)

- Removes: 0.685 ms median of host time with both GPUs idle (S142). Est. -0.5
  to -0.7 ms (layer-0 GPU work ~0.6-0.8 ms covers p50; p99 reads still stall).
- Where: gather in `stage_batch` verify.cpp:3352-3367 before submit; flag
  raised at 3446-3450 right after submit; the graph already waits for it at
  layer 1 (921).
- Sketch: compute ngram rows in stage_batch, but call `gather_batch` in
  `run_slot_rows` after the two appends (3417-3419) and before `*flag = 1`,
  exactly as solo `run()` does (2377-2383). Optionally prefetch row 0's PLE
  rows (the accepted token is known right after the verifier, ~7 ms earlier)
  via `prefetch_rows` - the S49 prefetch only covered draft rows.
- Risk: low. No arithmetic change. Watch the layer-1 spin bound.

### 4. Draft-vocabulary head for the MTP pair

- Removes: 2 x 1.21 ms full-vocab 2-column head reads (437 MB each). With
  data/draft_vocab_en.bin (40,525 rows, ~71 MB) or draft_vocab.bin (106,299
  rows, ~187 MB): est. -1.4 to -2.0 ms/window.
- Where: `MtpPair::init` rejects `dhead_` (mtp.cpp:2086-2088); tails at
  2221-2227 / 2286-2292; `record_post_head` already supports the subset
  (`sub`, `map_ids`, 1167-1203). The rt dir has no draft_vocab.bin, so C1 also
  uses the full head today.
- Sketch: copy a draft_vocab file into the rt dir (C1 then uses it: measure
  acceptance vs STRATA_MTP_FULL_HEAD=1 first, no code). Then allow dhead in
  the pair: 2-column `native_mmvq(dhead_type_, dhead_, xq2_, logits2_, N,
  n_dvocab, 2)` and post-head with `sub=true` per slot.
- Risk: acceptance drop (output stays exact: the verifier is the full head).

### 5. Shorten card1's EP chain and push earlier

- Removes: card0's exclusive ep_wait (1.25 ms/cycle, 21-28 us/layer); 1-3
  kernel boundaries per layer on the critical path. Est. -0.6 to -1.2 ms.
- Where: verify.cpp:1346 (card0 plan before push), 1430-1447 (peer chain),
  ep_kernels.dp.cpp:17, 122; iq_kernels.dp.cpp routed down.
- Sketch: (a) record card0's `resident_plan_ep` (6.3 us, single WG) after
  `ep_push`; (b) drop `ep_copy_uncached`: let the peer plan/GU read ids and
  q8 rows from the inbox with the existing uncached-load annotation (reuse
  safety argument in 2.4 still holds); (c) fold `ep_send_rows` (29.7 us
  median, serial after down) into the peer down kernel: write the peer rows
  directly to card0 `parts_` with P2P stores, release fence, the same
  done-counter, last WG stores the ring flag. Bitwise (only the store target
  changes). Not recommended: reducing rows on card1 (7x fewer bytes) changes
  combine summation order.
- Risk: medium (P2P store visibility/ordering, flag protocol). Upside is
  capped by the exclusive wait; after it, the shared expert becomes critical.

### 6. Existing default-OFF shared-expert fusions (LFUSE, QDEDUP)

- Removes: per layer the shared-gate GEMV and sigmoid-scale (gate rides in the
  router GEMV, scale moves into the combine) and one MMVQ (gate+up pair):
  -3 launches/layer = -144/forward, plus -48 with QDEDUP. Est. -0.3 to
  -0.8 ms; worth more once item 5 shrinks the wait the shared expert hides in.
- Where: `lfuse_on` verify.cpp:840-844 (needs ar_on, native combine, n 2..8,
  native bf16: all true here), 1316-1324, 1406-1410, 1552-1557;
  STRATA_LFUSE / STRATA_LFUSE_GATE / STRATA_LFUSE_PAIR (77-79).
- Risk: low (comments claim bitwise); no SYCL serving record found in
  JOURNAL/docs, so needs the usual exact-output gate.

### 7. Fuse the HC read chain (8 launches per layer)

- Removes: up to 2 launches per HC read (norm, reduce) = -192/forward. Est.
  -0.7 to -1.2 ms (norm ~7.4 us, reduce ~3.5 us per call at T6; 388 calls,
  6.1 ms/cycle family).
- Where: fused_gr.dp.cpp:1919-2043 (norm/down/reduce/up launches);
  producers verify.cpp:1559 (combine) and 1075/1288 (out-proj).
- Sketch (bitwise variants only): reduce folded into `gr_down_sliced` with a
  last-work-group reduction in the original slice order (done-counter
  pattern of ep_send_rows); norm folded into the producing combine / a
  per-token single-WG epilogue. The algebraic variant (scale after the
  down-projection) removes the norm entirely but is NOT bitwise.
- Risk: medium-high (exactness gates); prior HC work was micro-tuning
  (S15 unroll/ILP, S48-S51 LO_ONCE), not launch removal.

### 8. Batched two-row MTP body for C2

- Removes: per depth one ~49-56 kernel body instead of two (A then B on the
  same queue): ~-100 launches/cycle; MTP bodies are 2.3 ms busy per cycle.
  Est. -0.8 to -1.2 ms.
- Where: `MtpPair::capture_pre_round/pre_step` (mtp.cpp:2151-2209),
  `record_front`/`record_rest` (824-1165).
- Sketch: like the verifier's batch window: dense projections, HC, router,
  experts, combine over 2 rows (one per slot) with multi-column kernels; K/V
  append and attention per slot (slot KV pools). Pre-round rows differ (T0,
  T1): use catch-up-all (STRATA_MTP_CATCHUP_ALL) or a per-row KV target list.
- Risk: medium-high (new graphs, per-slot state). S131 owner queues (-0.33%)
  tested concurrency, not batching; this is the S123 "projection-batched"
  alternative.

### 9. Small-chain fusions in the MoE router path and GDN

- Removes: top10 + plan + quantize_rows + push as one single-WG kernel after
  the router GEMV (-3/layer, chain is 48/cycle, 991 us kernel time); merge the
  two per-slot `gdn_conv_l2_multi` and the two `gdn_step_norm` launches into
  one each with a 2-entry state-pointer table (-2/GDN layer); attention tail
  merge -> gate_apply -> quantize (-1 to -2/QSA layer). ~-250 launches/forward.
  Est. -0.4 to -0.8 ms.
- Where: verify.cpp:1325-1350, 1419-1431, 1040-1068, 1276-1287.
- Risk: medium (new kernels, but each piece is index/copy work or keeps the
  per-row arithmetic).

### 10. Per-launch hygiene: cooperative-launch property and queue profiling

- Removes: unknown, possibly per-kernel driver overhead on ~2,650 launches per
  cycle. Check first: grep the existing S136 API trace for
  `zeCommandListAppendLaunchCooperativeKernel` and queue creation flags.
- Where: 398 `use_root_sync` sites (e.g. native_mmvq.dp.cpp:3328,
  verify_kernels.dp.cpp:2458, fused_gr.dp.cpp:1923); DPCT_PROFILING_ENABLED
  in verify.cpp:2, mtp.cpp, device.dp.cpp (device.hpp:687).
- Sketch: if launches are cooperative, drop `use_root_sync` (no root groups
  are used anywhere); create decode queues explicitly with only `in_order`.
- Risk: low semantically; needs a full AOT rebuild.

## 8. Already measured, do not repeat as-is

| Idea | Stage | Result | Why it did not pay |
|---|---|---|---|
| SH_STREAM=0 alone | S15 | -0.86% | shared moved before ep_push; resets not verified gone |
| Async batch commit overlap | S79 | slightly slower, unmatched | one compute engine (Engine<0,0>): two queues serialize; saves only a host turnaround |
| Whole depth-2 pair graph (6 -> 1 appends) | S106 | flat | submission is 48 us unprofiled |
| MTP owner queues | S131 | -0.33% | same engine, host joins |
| Batch PLE prefetch of draft rows | S49 | no gain | late issue; item 3 hides the gather instead |
| QSA commit grouped steps | S139/S144 | +1.28% / none | commit side only |
| MTP parallel argmax | S114 | +0.58% | default OFF |

Card1 has ~36 ms/cycle of idle compute but ~2 GiB free VRAM after 363
experts/layer and exchange regions, so moving dense or MTP work there needs a
placement change (S123 parked).

## 9. Suggested probe order (cheapest first)

1. CPU-only: grep the S136 API trace for cooperative launches (item 10).
2. Standalone S135-style graph oracle on one card: ~2,200 tiny kernels as
   (a) one in-order queue, (b) the same with a fork/join every 46 nodes;
   replay 200x unprofiled; report wall per replay and reset counts. This
   prices item 1 without the model.
3. Items 3 and 2 (small, exact), then item 1 with the post() reorder, each as
   a matched C2 pair with exact-output gates.
4. C1 acceptance with a draft_vocab file vs STRATA_MTP_FULL_HEAD=1 (no code),
   then item 4.
