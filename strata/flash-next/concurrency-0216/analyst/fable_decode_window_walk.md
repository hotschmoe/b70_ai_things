# Decode-window walk: one batched verify window as a function of rows (Strata 0.2.16 dd, 2x B70, EP)

Responding model: Claude Fable 5.2 (model ID claude-fable-5-2). Date 2026-11-20.
CPU-only static analysis plus reads of existing logs. No GPU touch, no build, no source or config edit.

## 0. Scope, sources, conventions

CONFIG -> Source /mnt/vm_8tb/b70/build/strata-0216/src, branch dd-0.2.16 at commit cf1a2c359 (read from a
git-archive snapshot of that commit; the working tree had an in-progress iq_kernels.dp.cpp merge by the GPU worker,
so line numbers below are the COMMITTED file's; later commits on the branch shift lines only in the files they touch). Serving config strata/flash-next/dd/hotschmoe-dd-0216.json.
GGUF tensor types read from the headers of the two shards (own header parser, python3 -I, no deps).
Measured evidence: /mnt/vm_8tb/b70/results/strata-0216/candidate-engine.log (LOG), old-night-engine.log (OLDLOG),
candidate-bench.log, candidate-c2.log, old-c2.log, candidate-c4-warm.log; old unitrace census
strata/flash-next/night/agent-reports/trace_final_analysis.md (TFA) and decode_launch_census.md (DLC).

Path shorthand (all under engine/sycl/ of the source tree):
  V = src/core/verify.cpp        G = src/program/generate.cpp     M = src/core/mtp.cpp
  LY = src/core/layer.cpp        K/ = src/kernels/cuda/
Evidence tags: [R] read in source or GGUF header, [M] measured number from an existing log/doc,
[I] inferred (arithmetic or reasoning from R and M; not measured). Every ms figure tagged [I] is an estimate.

Symbols: S = active slots, R = rows = 2S (seed + 1 draft per slot at STRATA_BATCH_MTP_DEPTH=1),
N = 2560, HC = 4, K = 10 routed experts, 48 layers = 36 GDN + 12 QSA.

## 1. Headline findings

1. The three bench numbers are not the same code path. C1 = the SOLO window, T = 4 rows of ONE stream (1 seed +
   3 drafts, --spec 4): 1500 tokens, 1443 drafts offered = 481 windows, 23.17 ms/window, 3.12 tokens/window [M LOG:625,
   LOG:924 "window 25 4"]. C2 = batch window R = 4 (2 slots x 2 rows): 3.70 kept rows/window = 1.85 per stream, p50
   content gap 24.2-24.3 ms [M LOG:882, candidate-c2.log]. C4 = batch window R = 8: 6.81 kept rows = 1.70 per
   stream, 44.14 ms/window = run 38.38 + commit 2.76 + emit 3.01 [M LOG:1919].
   So C1 -> C2 is almost entirely a ROW-BUDGET effect: the same 4 rows in about the same window time (23.2 vs 24.3 ms),
   but 3.12 tokens for one stream vs 1.85 tokens for each of two. C2 -> C4 is a WINDOW-TIME effect: rows double
   and the window grows 24.3 -> 44.1 ms (about +3.2 ms per extra row at 20K context, section 5).
2. Measured batch-window law on this build (20K-token contexts): R=2 21.2 ms (run 19.4), R=8 44.1 ms (run 38.4)
   [M LOG:1006,1919] -> run ~= 13 ms fixed + 3.2 ms per row [I]; commit = 0.70 ms per slot, emit ~= 0.3 + 0.68 ms per
   slot [M LOG:113,882,1006,1919], both strictly linear in S because they are serialized per slot (section 4).
3. Biggest regression against the old night build is in the verify graph only and scales with rows: run
   R=2 17.3-18.2 -> 18.4-19.4, R=4 20.7 -> 21.5-22.4, R=8 32.4 -> 38.4-38.7 ms; commit and emit are unchanged
   [M OLDLOG:881,1011,2002 vs LOG:882-915,1006,1919]. Static cause [R]: the config relies on upstream
   STRATA_BATCH_ATTN_ROWS=1, which is switched off whenever the slot K/V is streamed (V:1546-1547,
   "rows_ok = ... kv_mode != 1"), and --kv-resident 65536 < 200000 makes every main-layer state kv_mode 1
   (LY:736-737). The old tree's STRATA_NIGHT_ATTN_BATCH batched per slot / per window UNDER streaming with a per-slot
   resolve (old src-ep verify.cpp:1230-1262, 1288, 1372-1425) and was not ported (no match for NIGHT_ATTN_BATCH in
   the 0.2.16 tree). Result: 12 x (11S - 3) extra launches per window = 96 / 228 / 492 at S = 1 / 2 / 4 [I from R].
   (update-0216/feature_map.md:73 says the old batched path also fell back per row under streaming; the old
   source read here does not support that.)
4. Commit is per slot and per layer because STRATA_GDN_COMMIT_LAYERS=1 is only eligible for S == 6 rows (two owners
   of three rows, i.e. depth 2) (V:3727-3733). At depth 1 it is inert: 110 launches per slot (section 3.4).
   STRATA_BATCH_MTP_PAIR_GRAPH=1 is likewise inert at depth 1 (M:2414, needs max_steps == 2).
5. Expert weights are read once per DISTINCT expert of the window (not once per row), in chunks of up to 4 rows:
   the device plan groups the R x 10 entries by expert id (K/verify_kernels.dp.cpp:2712-2727, 2763-2785) and the
   lane kernels walk a group's entries 4, then 2, then 1 at a time (K/iq_kernels.dp.cpp:2480-2502, 2542-2546).
   A group with 3, 5, 6 or 7 entries therefore reads its weights twice or three times.
6. Card 1 runs 289 launches per window regardless of S (6 per layer + 1) and only computes its share of the routed
   experts; everything else, including all per-row and per-slot serialized work, is on card 0 (section 3).
7. The harness numbers also contain admission-path stalls that are not window costs: C2 streams each saw one
   1.7-4.0 s content gap [M candidate-c2.log]; park-from-slot took 1478/1577 ms where the old build took 44/40 ms,
   and draft_first takes 21-29 ms where the old build took 0.7-3.7 ms [M LOG:898-911 vs OLDLOG:898-913]. Steady C2
   from the window law is 1.85 / 24.25 ms = 76 tok/s per stream; the harness reports 65.4.

## 2. What exists per row, per slot, per window (the model and quant facts used)

Tensor types (first shard) [R, header parse]: token_embd IQ3_S 260 MiB; output.weight Q5_K 416.8 MiB;
hc_{attn,ffn}_{down,up} BF16 4 x 300 MiB (served as Q8_0 copies, 0.63 GiB, LOG:12); router ffn_gate_inp BF16
120 MiB; attn_qkv (2560->10240) 597 MiB, attn_gate (2560->6144) 304 MiB, ssm_out (6144->2560) 376 MiB over 36 GDN
layers in mixed IQ4_XS / Q6_K / Q4_K / Q5_K / IQ3_S; attn_q (2560->12288) 226 MiB, attn_output 142 MiB, attn_k/v
22 MiB over the 12 QSA layers; shared expert gate/up/down 115 MiB; indexer k_proj/q_proj BF16 35 MiB; ssm
alpha/beta BF16 17 MiB; PLE key/value BF16 62.5 MiB (layer 1).
Routed experts: gate and up 2560x640 each, per layer one of IQ2_XXS (9 layers, 422,400 B per expert per
projection), IQ2_XS (10, 473,600), IQ2_S (10, 524,800), IQ3_XXS (6, 627,200), IQ3_S (13, 704,000); down 640x2560
Q2_0 type 42 (30 layers, 460,800 B) or IQ4_NL (18 layers, 921,600 B). Average 1.746 MB per (layer, expert)
[I from R: (2 x 13037.5 + 14850) MiB / 24576].
Placement [M LOG:35-47]: card 0 holds 169 experts per layer (13.19 GiB, the profile's hot ones), card 1 the other
343 (26.77 GiB); "each layer's traffic dealt evenly between them" (G:5248). All 24576 resident: the verify graph
is the all-resident, zero-doorbell form (LOG:68, V:4083).
K/V: int8, 1056 B per cell per QSA layer (2 KV heads x 256 x 2 codes + 32 B scales) [I from LOG:17: 2.36 GiB for
200000 cells x 12 layers]; 65536 cells per layer resident in VRAM, the full 200000 mirrored in pinned RAM.
Indexer pooled keys: 128 floats per 4-cell block = 128 B per context token per layer [I from model-facts].
Window capacity: kVerifyMaxT = 8 rows (LOG:68; K/verify_kernels.dp.cpp:2635-2638), so depth 1 x 4 slots fills it.

## 3. One window in execution order

Host thread = one thread (G:10026 batch_step). Per window it does, in order:
  H1 pack rows (G:10035-10058) -> H2 Verifier::run_slot_rows (V:4047) -> H3 keep decision (G:10114-10121) ->
  H4 commit_slot_prefixes (V:4218) -> H5 per slot: copy final rows + (pair) draft (G:10153-10256) -> H6 emit lines.

### 3.1 Host staging before the launch (both cards idle) [R]

| # | Step | Where | Scaling | Data |
|---|---|---|---|---|
| a | graph lookup / capture per row layout (key = slot of each row) | V:3981, V:3620, V:3709 | first use of a layout only | records both cards' graphs |
| b | steps, positions into mapped host staging | V:3984-3992 | R | R x (kStepCount + 30) int32 |
| c | PLE n-gram rows: 16 hash heads per row, gather_batch (O_DIRECT NVMe / row cache), decode to 2560 floats per row | V:3993-4004; include/strata/kernels/ngram.hpp:201-206 | R x 16 rows | 16 x 4 KiB page reads per row worst case; 10,240 B per row into mapped host memory |
| d | per-slot commit records, flag and sequence reset, wait-record arm | V:4008-4025 | S | few hundred bytes |
| e | submit card-1 graph, then card-0 graph (2 graph launches) | V:4055-4057 | fixed | - |
| f | raise the PLE flag (sfence + store) | V:4084-4088 | fixed | 4 B |
| g | cs_->wait(), copy_->wait(), wait_failed, ep_check | V:4089-4109 | fixed (1 blocking wait) | - |
| h | sample_rows: no launch for greedy slots; one eager sample_tokens launch + wait per sampling row | V:4294-4334 (skip at 4304) | 0 (bench is temperature 0); R when clients sample | 1 MiB logits re-read per sampling row |
| i | read picks from mapped h_out_ | V:4114 | R | 4 B per row |

PLE prefetch hints for the next window's rows are issued by the drafter (M:2468-2475, 2507-2520), so step c is mostly
a wait for already-issued reads [I]. It was 0.685 ms for 6 rows unprofiled on the old build [M DLC section 0].

### 3.2 Card 0 verify graph (queue cs_, in-order, one captured command graph; recorded by V:1047 record_window)

Every op below is a graph node (nothing eager). "B" = one launch covering all R rows (weights read once),
"S" = one launch per slot, "R" = one launch per row. Grids are (work-groups) x (work-group size).

Window prologue (V:1113-1135): ep_bump single_task (K/ep_kernels.dp.cpp:231); copy_i32_from_mapped x3, 1 WG x 128,
host-mapped -> device (K/elementwise.dp.cpp:923); embedding gather + broadcast_streams grid (R, 40) x 256
(K/verify_kernels.dp.cpp:3131). Reads token_embd IQ3_S rows (R x ~1.1 KB).

Per layer, pre(l) then post(l); post(l) and pre(l+1) are adjacent in the stream (V:2005-2015).

| Order | Op | Launch site | Kernel / launcher | Grid x WG | Rows | Reads (quant) |
|---|---|---|---|---|---|---|
| 1 | HC attention read: norm+down, then up+mix | V:1277 | gr_q8 down / up, K/fused_gr.dp.cpp:2458-2494 (taken at 3577 because STRATA_HC_Q8=1) | down: (NRG x NSLAB) x 256; up: (2560/UP_COLS) x UTH; sub-group 16 | B (template on T=R) | hc_attn_down/up Q8_0 copies 2 x 3.48 MB, inject BF16, residual R x 40,960 B |
| GDN layers (36) | | | | | | |
| 2 | q8_1 quantize of mixed | V:1299 | native_quantize_q8_1, K/native_mmvq.dp.cpp:4316 | blocks x QUANT_THREADS | B | R x 10 KB FP32 -> R x 2880 B |
| 3 | qkv projection 2560->10240 | V:1300 | native_mmvq -> KREG 1-8 columns (K/native_mmvq.dp.cpp:5128, 2373-2392, tiles 2353-2371); IQ3_S via night kreg2 (2636-2657) | (n_out/Rt tiles) x L, sub-group 16 | B (R columns) | attn_qkv, 13.9 MB/layer avg 16.6 MB (IQ4_XS/Q6_K/Q4_K/Q5_K) |
| 4 | conv over each slot's history | V:1303-1311 | gdn_conv_l2_multi, K/verify_kernels.dp.cpp:2033 | (rows of slot, C/S) x S | S launches | conv weight F32 164 KB, slot conv history |
| 5 | alpha/beta gates | V:1317 | gdn_ab_multi (STRATA_NIGHT_GDN_AB grid rows x blocks), :2145-2156 | (R, blocks) x 64 | B | ssm_alpha/beta BF16 2 x 246 KB |
| 6 | gate projection 2560->6144 | V:1321 | native_mmvq (as 3) | as 3 | B | attn_gate 6.8-12 MB |
| 7 | recurrence + norm, all slot groups | V:1328-1340 | gdn_step_norm_groups, :2274-2299 (STRATA_BATCH_GDN_GROUPS=1) | (48 x S) x (S_, RG), no GRF256 variant | B over slots | each slot's GDN state 817,152 floats = 3.27 MB (read, not written) |
| 8 | q8_1 quantize of y (6144) | V:1357 | as 2 | | B | |
| 9 | ssm_out 6144->2560 | V:1358 | native_mmvq | as 3 | B | ssm_out 8.4-12.9 MB |
| QSA layers (12) | | | | | | |
| 2q | q8_1 quantize | V:1390 | as 2 | | B | |
| 3q | indexer k_proj + q_proj in one launch | V:1394-1396 | bf16_gemv_fp32_mmvf_pair_kreg, K/native_bf16.dp.cpp:1114 (STRATA_BF16_KREG=1) | KREG | B | BF16 0.66 + 2.6 MB |
| 4q | attn_k, attn_v 2560->512 | V:1402-1403 | native_mmvq x2 | KREG | B | 0.7-1.1 MB each |
| 5q | K norm + RoPE | V:1404 | native_qsa_rms_norm_rope, K/native_rope.dp.cpp:339 | (R x 2) x 256 | B | |
| 6q | indexer-tail snapshot | V:1413-1417 | copy_from_mapped, K/elementwise.dp.cpp:554 | <=64 x 256 | S launches | (idx_block-1) x 128 floats per slot |
| 7q | K/V cell append (+ host mirror, streaming) | V:1442-1461 | kv_append_q8_step, K/kv_q8.dp.cpp:156-190 (host stores at :99-102) | grid x KV_Q8_GROUP | R launches | writes 1056 B to VRAM and 1056 B to pinned RAM per row |
| 8q | indexer append | V:1468-1474 | native_qsa_indexer_append, K/native_qsa_indexer.dp.cpp:471 | small | R launches | 128 floats per row |
| 9q | attn_q 2560->12288 | V:1477 | native_mmvq | KREG | B | attn_q 16.7-24 MB |
| 10q | q norm + RoPE; indexer q norm + RoPE | V:1481, V:1506 | native_qsa_rms_norm_rope x2 | (R x 24) x 256, (R x 4) x 256 | B | |
| 11q | block scores of the row's query | V:1570 | qsa_block_scores nq=1, K/qsa_select.dp.cpp:1768 (transposed form 1780-1789) | 256 WG x (SCORE_WARPS x 32) | R launches | the slot's pooled keys: 128 B per context token (2.56 MB at 20K, 25.6 MB at 200K) PER ROW |
| 12q | top-k blocks | V:1572 | qsa_block_topk nq=1, :2168 (scan / register kernels 2226-2302) | nq x TK_T | R launches | scores |
| 13q | K/V residency resolve + miss copy | V:1574 | qsa_kv_resolve LY:1031 -> kv_stream_resolve, K/kv_stream.dp.cpp:338-393: resolve kernel 1 WG x 1024, copy kernel 96 WG x 128 | 2 launches | R x 2 launches | page table; on a miss, one block's runs read from PINNED HOST RAM inside the kernel (261-277) |
| 14q | attention over the selected cells: chunk kernel + merge | V:1576 | qsa_decode_attn_batch n_q=1, K/qsa_decode_attn.dp.cpp:1062 (CHUNK 64 cells, :31; vpipe variant 1192-1196) | (cap/64, 2 KV heads) x THREADS, then merge | R x 2 launches | <= 2048 cells x 1056 B = 2.16 MB int8 K/V per row |
| 15q | output gate | V:1595 | native_qsa_gate_apply, K/native_qsa.dp.cpp:269 | (R x 24 x 256 / 256) x 256 | B | |
| 16q | q8_1 quantize (6144), attn_output 6144->2560 | V:1606-1607 | native_quantize_q8_1, native_mmvq | KREG | B | attn_output 12.9 MB |
| all layers | | | | | | |
| 17 | HC FFN read (down, up) | V:1614 | as 1 | | B | hc_ffn_down/up Q8_0 |
| 18 | router GEMV 2560->512 | V:1647 | bf16_gemv_fp32_mmvf_multi, K/native_bf16.dp.cpp:746 (KREG BF16) | | B | ffn_gate_inp BF16 2.6 MB |
| 19 | top-10 of 512 | V:1649 | native_router_top10_multi (tree), K/native_router.dp.cpp:386-396 | R WG x (8, 32) | B | |
| 20 | both cards' expert plans in one launch | V:1670-1676 | resident_plan_ep_pair, K/verify_kernels.dp.cpp:3010-3026 (STRATA_NIGHT_EP_PLAN1=1) | 2 WG x 128, one thread per entry (R x 10 <= 80) | B | residency tables; groups = distinct experts in order of first occurrence (2712-2727) |
| 21 | q8_1 image of the MoE input | V:1766-1767 | quantize_q8_1_rows, K/iq_kernels.dp.cpp:6267 | | B | R x 2880 B |
| 22 | EP push to card 1: ids, weights, card 1's plan, q8_1 rows, then the flag | V:1780-1786 | ep_push_w, K/ep_kernels.dp.cpp:297-317 | 1 WG x 256; stores into card 1's IPC-mapped inbox, system-scope barrier, flag store | B | 80 B + 2880 B per row + the plan words, across PCIe |
| 23 | own routed experts: gate+up, SwiGLU+q8, down | V:1880 -> 1870 | native_expert_grouped, K/iq_kernels.dp.cpp:7601: launch_gu (7706; row-reuse / multi launchers 5889-5971, 6074-6092), swiglu_q8_1 (7765), launch_down (7808) | gu: (10R group rows, 40) x 256; down: (10R, 80 or 160) x 256; 8 lanes per row; group rows beyond n_groups exit (2516-2517) | B; weights once per group per <=4-entry chunk | this card's distinct experts: gate+up+down 1.35-2.33 MB each |
| 24 | shared expert (after own experts: STRATA_NIGHT_SH_LATE) | V:1881-1892 | shared_expert_multi, K/shared_expert.dp.cpp:323-360: fused gate+up+SwiGLU+q8, then down+scale (STRATA_NIGHT_SHEXP) | KREG-style | B, 2 launches | 2.4 MB per layer |
| 25 | wait for card 1's sums | V:1904 | ep_wait single_task, K/ep_kernels.dp.cpp:253-266: one GPU thread spins on the return flag, bound 2,000,000 (LOG:67) | 1 thread | fixed | flag (system-scope loads) |
| 26 | combine own rows + peer sums + shared | V:1957-1961 | ep_combine_psum, K/ep_kernels.dp.cpp:354-415 (STRATA_NIGHT_EP_PSUM=1) | (R, 5) x 128 | B | 10 part rows + psum + shared per row = 123 KB per row |

Layer 1 only (PLE, V:1183-1253): device wait on the host flag (wait_flag_ge, V:1185), copy of the R PLE rows from
mapped host memory (V:1186), key and value BF16 GEMVs for all rows when the BF16 forms are loaded (condition V:1193-1195, launches V:1205-1208,
50 MB + 12.5 MB) [I that the condition holds], gr_write_multi,
then PER ROW ple_block_projected + ple_history_advance + a history snapshot copy (V:1232-1252).
End of window (V:1993-2109): gr_write_multi; final HC read (2 launches); q8_1 quantize; head native_mmvq
2560->248320 Q5_K with R columns (KREG: 4 rows per tile at <= 4 columns, 8 rows per tile and GRF256 at 5-8,
K/native_mmvq.dp.cpp:2361-2369); greedy sample_tokens writing the picks into mapped host memory (V:2106-2109;
argmax_rows is not selected on this device class, K/verify_kernels.dp.cpp:1952-1967).

Launch count, card 0 verify graph [I from the table]: GDN layer 23 + S, QSA layer 27 + S + 8R, so
  launches ~= 1169 + 252 S  ->  about 1420 (S=1), 1670 (S=2), 2180 (S=4).
(The old solo C1 trace counted 1,575 card-0 launches per window including commit and MTP, TFA section 1.)

### 3.3 Card 1 graph (queue ep_q_, its own captured graph, launched first, V:4055) [R]

ep_bump once (V:1115), then per layer (V:1790-1813):
| Order | Op | Kernel | Grid | Data |
|---|---|---|---|---|
| 1 | wait for card 0's push | ep_wait single_task (spin on the inbox flag) | 1 thread | - |
| 2 | copy the inbox past the caches | ep_copy_uncached, K/ep_kernels.dp.cpp:23-33 | <= 64 WG x 256 | ids, weights, plan, R x 2880 B |
| 3-5 | card 1's routed experts (plan arrived in the inbox, no plan launch) | native_expert_grouped: gu, swiglu, down | as card 0 step 23, cap = 10R group rows | card 1's distinct experts |
| 6 | weighted per-token sum written into card 0's memory, then the return flag | ep_send_psum, K/ep_kernels.dp.cpp:319-351 | (R x 10) WG x 64, system-scope barrier | R x 10,240 B across PCIe |
289 launches per window for any S. Card 1 idles (spins in ep_wait) through all of card 0's steps 1-22 and 24-26
of every layer; the old C1 trace measured 4.9 ms compute and 15.5 ms spin per window (TFA section 1).

### 3.4 Commit graph (card 0, queue cs_, second graph launch + blocking wait, V:4240-4258) [R]

Recorded by capture_commit_batch (V:3709-3904). The 36-layers-in-one-launch path (gdn_commit_layers) needs S == 6
(V:3727-3733) and is never taken at depth 1. Per SLOT:
- 1 copy of the commit record (V:3776-3778);
- per GDN layer: gdn_conv_commit (V:3826, K/verify_kernels.dp.cpp:2062, 40 WG x 256) + gdn_step_norm_multi in
  commit form (V:3836-3842; a 2-row group is not "one_row", so the groups launch at V:3844 is skipped; commit-split
  kernel grid (4 x 48) x (RG, 32), K/verify_kernels.dp.cpp:2483-2493): 72 launches, writes the 3.27 MB state per layer;
- per QSA layer: tail restore copy + 2 indexer appends (V:3857-3864): 36 launches;
- 1 PLE history copy (V:3873).
= 110 launches per slot: 110 / 220 / 440 at S = 1 / 2 / 4. Measured 0.70-0.82 / 1.40 / 2.76-2.80 ms
[M LOG:113,1006,882,1919] = 6.3-7.3 us per launch, exactly linear.

### 3.5 Emit: residual copy and the drafts (card 0) [R]

Per active slot: eager memcpy of its 2 final residual rows (81,920 B, device to device) with a blocking .wait()
(G:10184-10187). Then per PAIR of slots (pair 0 = slots 0,1; pair 1 = slots 2,3; G:10215-10249):
MtpPair::draft (M:2403): three graph launches on the pair queue - round A, round B (each slot's MTP layer
forward on its own rows, M:2289-2304, about 50 kernels each per DLC section 1), then one tail = 2 quantizes + ONE
two-column draft-vocab head (40,525 x 2560 Q5_K, 68 MiB, LOG:69-72) + pick + select (M:2320-2334) - then a host
spin on the mapped output and cs_->wait() (M:2498-2514). The second pair starts only after the first pair's wait
returned. A slot without a partner drafts alone through MtpDrafter::draft (G:10240-10247).
Measured emit: 1.0 ms (S=1), 1.36 ms (S=2), 3.01-3.04 ms (S=4) [M].
Host blocking waits per window: 1 (verify) + 1 (commit) + S (row copies) + ceil(S/2) (pairs) = 4 / 5 / 8.

### 3.6 Solo window (the C1 number) for comparison [R/M]

One stream alone runs Verifier::run (V:2667) with T = 4 rows of the same slot and MtpDrafter::draft with 3 steps.
There the attention IS batched: one qsa_block_scores / top-k / resolve / attention call for all 4 rows
(V:1581-1591), one kv_append_q8_steps and one indexer append (V:1422-1436, 1462-1466). A lone active slot inside
batch mode (other slots prefilling or idle but batch on) instead runs R = 2 batch windows: 21.2 ms for 1.87 tokens
= 88 tok/s against 134.6 solo [M LOG:1006 vs LOG:625].

## 4. Data movement and synchronization inventory (per window)

### 4.1 Movements [R unless tagged]

| Movement | Path | Count per window | Bytes per window | Engine |
|---|---|---|---|---|
| H2D inputs: tokens, steps, positions | mapped host memory read by copy kernels (V:1118-1120) | 3 launches | ~1 KB + 44 B per row | compute engine, card 0 |
| H2D PLE rows | mapped host -> ple_ (V:1186) | 1 launch | R x 10,240 B | compute, card 0 |
| NVMe/page cache -> host: PLE table | gather_batch on the host thread (V:4004) | 16 row reads per window row | R x 16 x 90 B payload; up to R x 64 KiB of 4 KiB page I/O [I] | CPU I/O threads |
| D2H picks | sample kernel writes mapped h_out_ (V:2106-2109, 4114) | 1 launch | R x 4 B | compute, card 0 |
| Cross-card push (card 0 -> card 1 inbox, IPC-mapped device memory) | ep_push_w (V:1780) | 48 | 48 x (R x 2,960 B + plan words) ~= 0.14 MB per row [I] | compute kernel on card 0 storing over PCIe |
| Cross-card return (card 1 -> card 0) | ep_send_psum (V:1806) | 48 | 48 x R x 10,240 B = 0.49 MB per row | compute kernel on card 1 storing over PCIe |
| Card-1 local copy of the inbox | ep_copy_uncached (V:1791) | 48 | same as the push | compute, card 1 |
| KV append, VRAM | kv_append_q8_step (V:1456) | 12 x R | 12 x R x 1056 B | compute, card 0 |
| KV append, RAM mirror (--kv-resident) | same kernel, stores into pinned host RAM (K/kv_q8.dp.cpp:99-102) | 12 x R | 12 x R x 1056 B across PCIe | compute, card 0 |
| KV RAM -> VRAM restore of evicted blocks | kv_stream copy kernel reads pinned RAM (K/kv_stream.dp.cpp:261-277) | launched 12 x R times; copies only on a miss | 0 in steady state below 65,536 cells; one page per layer when a slot crosses into a new page or re-reads an evicted one [I] | compute, card 0 |
| D2D slot residual rows to the drafter | eager memcpy + wait (G:10184) | S | S x 81,920 B | default queue, card 0 |
| H2D drafter staging (steps, tokens, rows) | copy_from_mapped_multi (M:2294-2297) | S | small | compute, card 0 |
| Conversation park / restore | not part of a window (admission path, G:9984-10023, 13281-13302) | per admission | 236 MB snapshot even for a 28-token conversation [M LOG:898] | - |
There is no host-staged weight traffic in the window: all experts are VRAM resident on one of the cards (LOG:45-53),
"CPU experts 0.00, PCIe 0.00" on every batch line [M].

### 4.2 Synchronization [R]

| Sync | Where | Count per window | Nature |
|---|---|---|---|
| Graph launch, card 1 then card 0 | V:4055-4057 | 2 | ext_oneapi_graph of captured graphs (one per row layout, LRU of 64: G:7970, V:3949-3980) |
| Host raise of the PLE flag | V:4084-4088 | 1 | sfence + store to mapped memory |
| Device wait on the PLE flag (layer 1) | V:1185 | 1 | 1-thread kernel spinning on host-mapped memory |
| ep_wait on card 1 (push arrived) | V:1790 | 48 | single_task spin, system-scope loads, bound 2,000,000 spins |
| ep_wait on card 0 (sums arrived) | V:1904 | 48 | same; this is where card 0 absorbs any card-1 lateness |
| System-scope group barrier + flag store in push / psum | K/ep_kernels.dp.cpp:313-315, 342-348 | 96 | makes all stores visible to the peer before the flag |
| Host blocking wait, verify | V:4089 (+ copy_->wait V:4107, a no-op queue in this mode) | 1 | queue wait |
| Host blocking wait, commit | V:4258 | 1 | queue wait; nothing on the host needs the result before the next graph on the same in-order queue |
| Host blocking wait, slot row copy | G:10187 | S | event wait |
| Host spin + wait, pair draft | M:2498-2514 | ceil(S/2) | _mm_pause spin on mapped output, then queue wait |
| Graph shape | STRATA_NIGHT_SH_LATE makes the verify graph linear (V:1094-1103, 1621): no fork/join barriers or per-node events | - | the shared-expert side stream (sh_cs_) is not used |
No eager kernel launches inside the verify or commit phases; the only eager submissions per window are the S row
copies (and R sampling launches when clients do not use temperature 0, V:4326).

## 5. Bytes read per window as a function of S

### 5.1 Fixed per window (each batched kernel reads its weights once whatever R is) [R sizes, I sums]

| Group (card 0) | MiB per window |
|---|---:|
| HC projections, Q8_0 copies (96 down/up pairs) + final mixer | ~650 |
| Dense MMVQ GDN layers: qkv 597 + gate 304 + ssm_out 376 | 1277 |
| Dense MMVQ QSA layers: attn_q 226 + attn_output 142 + k, v 22 | 389 |
| BF16 GEMVs: router 120, indexer 35, ssm alpha/beta 17, PLE key/value 62.5 | 235 |
| Shared expert gate/up/down | 115 |
| Vocabulary head Q5_K | 417 |
| Total | ~3,085 MiB = 3.24 GB -> 5.3 ms at 608 GB/s; the old C1 census achieved 301 GB/s on dense+shared and 231 GB/s on HC (TFA section 2) -> about 10-11 ms [I] |

### 5.2 Growing with rows

Routed experts: bytes = 48 x D(R) x 1.746 MB, D = distinct experts per layer over the window's R x 10 entries.
Bounds: 10 <= D <= min(10 R, 512). If the R routings were independent and uniform, E[D] = 512 (1 - (1 - 10/512)^R):

| R | entries | E[D] independent-uniform | routed bytes, both cards (upper estimate) | at dedup of consecutive rows (D ~ 0.6 x 10R) [I] |
|---:|---:|---:|---:|---:|
| 2 | 20 | 19.8 | 1.66 GB | ~1.0 GB |
| 4 | 40 | 38.9 | 3.26 GB | ~2.0 GB |
| 8 | 80 | 74.8 | 6.27 GB | ~4.0 GB |
The real D is lower than the uniform figure (routing is skewed, and a slot's seed and draft rows are consecutive
tokens) but it is not measured anywhere I could read; the plan's group count is on the device (counts[0],
K/verify_kernels.dp.cpp:2793) and never exported. At C4 the routed bytes (4-6 GB, about half per card) exceed all
fixed dense reads (3.24 GB). The code reads each distinct expert once per chunk of <= 4 entries (finding 5).

Attention, per row per QSA layer [I from R]: pooled index keys 128 B x context tokens (scores kernel) + up to 2048
cells x 1056 B = 2.16 MB of int8 K/V. Per window: 12 x R x (128 B x ctx + 2.16 MB):
| Context | per row | C2 (R=4) | C4 (R=8) |
|---|---:|---:|---:|
| 2K | 29 MB | 0.12 GB | 0.23 GB |
| 20K | 57 MB | 0.23 GB | 0.45 GB |
| 200K | 333 MB | 1.33 GB | 2.67 GB |
Both rows of a slot scan the same pooled keys; the per-row path reads them twice (the old per-slot path once).

Drafter: per slot round MTP dense Q8 111 MiB + 10 experts x 1.3 MiB; per pair one 68 MiB head read (LOG:21,69):
C2 316 MiB, C4 632 MiB [I].
Activations and cross-card traffic are negligible in bytes (R x 41 KB residual per HC read; 0.63 MB per row per
window across PCIe); they matter only as latency (96 flag hand-offs).

## 6. Why per-stream decode falls 134.6 -> 65.4 -> 37.3

Measured decomposition [M unless tagged]:

| | C1 (solo) | C2 | C4 (cached) |
|---|---:|---:|---:|
| Rows per window | 4 (1 stream) | 4 (2 streams) | 8 (4 streams) |
| Tokens kept per stream per window | 3.12 | 1.85 | 1.70 |
| Window ms | 23.17 | 24.2-24.3 p50 (mean 25.1-26.7 incl. transients) | 44.14-44.54 |
| Steady per-stream tok/s = tokens / window | 134.6 | 76 [I] | 38.6 |
| Harness per-stream tok/s | 134.6 | 65.4 (62.2-69.4) | 37.3 |

C1 -> C2: factor 0.59 from the row budget (one stream no longer gets 3 draft rows; depth is pinned to 1 for all
slot counts by STRATA_BATCH_MTP_DEPTH=1, G:4361), factor 0.95 from the window (per-row and per-slot serialization in
batch mode, sections 3.2-3.5), factor 0.86 from admission stalls that the harness counts (finding 7).
C2 -> C4: factor 0.55 from the window time (24.3 -> 44.1 ms), factor 0.92 from fewer kept tokens (different
prompts, 20K context). The +19.9 ms is: +4 rows x ~3.2 ms [I], +~4 ms for the 20K context on the first 4 rows
(the C2 harness runs at ~0.1-1.6K context, C4 at 20K; the R=2 and R=8 points are both at 20K) [I], +1.36 commit [M],
+1.65 emit [M].

Scaling class of every op (from section 3):

| Class | Ops | Cost shape |
|---|---|---|
| Serialized per ROW (R launches each) | QSA scores, top-k, resolve (2), attention chunk + merge, K/V append (+RAM mirror), indexer append (8 launches per row per QSA layer = 96 per row per window); PLE block per row (layer 1); sampling per row when temperature > 0 | linear in R; launch-bound; pooled keys re-read per row |
| Serialized per SLOT | gdn_conv_l2 (36 per slot), indexer-tail snapshot (12), the whole commit graph (110), residual copy + host wait (1), MTP round graph (1 graph, ~50 kernels), pair tail + host wait per 2 slots | linear in S |
| Batched over rows, weights read once | HC down/up, every dense native_mmvq (KREG 1-8 columns), BF16 GEMVs, GDN alpha/beta, GDN recurrence (groups), norms/RoPE/gate, quantizes, router top-10, plan, push, shared expert, combine, head, pick | fixed weight bytes + per-row arithmetic; sub-linear in R |
| Batched, bytes grow with the expert union | routed gate/up, SwiGLU, down on both cards | bytes ~ D(R); compute ~ entries; grid sized for 10R group rows on each card |
| Fixed per window | prologue copies, 96 ep waits/hand-offs, graph launches, verify/commit host waits, PLE flag | constant |

What grows most from C2 to C4 [I]: routed experts (union roughly doubles, on both cards, and card 0 then waits for
card 1), the per-row attention path (384 -> 768 launches), dense/HC/head per-row arithmetic at 8 columns, and the
per-slot commit and drafts (measured).

## 7. Top 15 suspected costs at C4 (R = 8, 20K context, 44.1 ms window)

Anchors [M]: run 38.38, commit 2.76, emit 3.01; old-build run at the same R 32.4. Family estimates [I] start from the
old C1 census at 4 rows (TFA sections 2 and 4) scaled by the classes above, and are constrained to sum to the
measured run. They are estimates to be replaced by the Stage B trace.

| Rank | Cost | Est. ms per window | Class | Evidence |
|---:|---|---:|---|---|
| 1 | Routed experts on the critical path: card 0 own share, then waiting for card 1's chain (wait, inbox copy, gu, SwiGLU, down, psum) | 8-10.5 | union-bound, both cards | C1 census: card 0 routed 3.48 ms at 40 same-stream entries at roofline; card 1 kernels 20-30% slower and already gating (0.57 ms ep_wait); 80 entries from 4 streams -> 2-2.4x the bytes |
| 2 | QSA select / resolve / attend / append, one row at a time | 7-9 (of which 3-5.9 is new vs the old build) | per row | 768 launches; run +5.9 ms vs old night at R=8, +1.1 at R=2 [M]; 492 / 96 extra launches [I]; small launches cost 6.3 us each in the commit graph [M] |
| 3 | Dense MMVQ (qkv, gate, ssm_out, attn_q/k/v/out), 8 columns | 6-7.5 | batched | 5.1 ms at 4 columns (C1 census), 1.67 GiB per window; bandwidth floor 2.9 ms |
| 4 | HC projections, 98 down/up pairs, T = 8 | 4-5 | batched | 3.28 ms at T = 4, 38% of roofline, latency bound (TFA) |
| 5 | Emit: S row copies with waits + 2 serialized pair drafts | 3.01 [M] | per slot | LOG:1919; 2 x (2 round graphs + tail + host wait) |
| 6 | Commit graph, 440 launches | 2.76 [M] | per slot | LOG:1919; V:3727-3733 keeps the batched form off |
| 7 | Small batched launches at the launch floor (about 230 quantizes, 240 router/top-k/plan/push/combine, 100 norm/RoPE/gate) | 2-3 | fixed count | C1 census: 966 launches under 10 us = 3.83 ms |
| 8 | Vocabulary head at 8 columns + final mixer + pick | 1.5-2.2 | batched | 0.83 ms at 4 columns KREG (TFA); old kernel 2.04 -> 3.38 ms from 4 to 8 columns (S25 curve) |
| 9 | GDN verify: 144 conv launches (per slot) + 36 alpha/beta + 36 recurrence | 1.5-2.0 | per slot + batched | 0.76 ms at one slot, 4 rows (TFA) |
| 10 | Shared expert, 96 launches, 8 columns | 1.4-1.9 | batched | 1.14 ms at 4 columns, 18% of roofline (TFA) |
| 11 | EP hand-off latency beyond compute: 96 flag crossings, inbox copy, uncached loads | 0.6-1.5 | fixed | not separable statically; card 0 ep_wait total in a trace is the direct measure |
| 12 | PLE host gather before the launch (both cards idle) | 0.7-1.0 | per row | 0.685 ms at 6 rows unprofiled (DLC) |
| 13 | Host round trips: 8 blocking waits, graph submits, stdout lines | 0.3-0.8 | per slot | DLC: 0.02-0.07 ms per submit/copy unprofiled |
| 14 | Empty work-groups in the expert grids: 80 group rows launched per card for about 30-40 real groups, x (40 + 80..160) work-groups, 3 launches per layer per card | 0.3-0.8 per card | grows with R | K/iq_kernels.dp.cpp:7628-7630, 7781-7783, early exit 2516-2517; cost not measured |
| 15 | PLE block per row, tail snapshots per slot, extra weight passes for 3/5/6/7-entry expert groups | 0.3-0.6 | per row / slot | V:1232-1252, 1413-1417; K/iq_kernels.dp.cpp:2480-2502 |
Outside the steady window but inside the harness numbers: admissions (park 100-118 ms + copy 26 ms + draft_first
21 ms each at 20K, serialized, LOG:1878-1914) and the one-off 1.5 s stalls (finding 7).

## 8. Config switches as they resolve inside the decode window

| Switch (value) | Effect in this config | Where |
|---|---|---|
| STRATA_BATCH_MTP=1, DEPTH=1, PAIR_HEAD=1 | batch windows of 2 rows per slot; two pair heads | G:4348-4361, 4507, 10035-10058, 10215 |
| STRATA_BATCH_MTP_PAIR_GRAPH=1 | INERT at depth 1 (needs two draft steps) | M:2414 |
| STRATA_GDN_COMMIT_LAYERS=1 | INERT at depth 1 (needs exactly 6 rows = 2 owners x 3) | V:3727-3733 |
| STRATA_BATCH_ATTN_ROWS=1 | INERT with --kv-resident (kv_mode 1) | V:1546-1547, LY:736-737 |
| STRATA_NIGHT_GDN_GRF=1 | INERT in batch windows: the groups launch has no 256-GRF form; only the non-group launch has | K/verify_kernels.dp.cpp:2288-2302 vs 2504-2517 |
| STRATA_MTP_PARALLEL_ARGMAX=1 | INERT with the 40,525-token draft-vocab head (full-vocab only) [I that the subset head is active, LOG:69] | M:1291 |
| STRATA_BATCH_GDN_GROUPS=1 | live in verify (one recurrence launch for all slots); not in commit for 2-row groups | V:1324-1340, 3816-3835 |
| STRATA_GDN_COMMIT_SPLIT=1, GDN_SPLIT=0 | commit recurrence kernel per slot per layer | K/verify_kernels.dp.cpp:2479-2493 |
| STRATA_HC_Q8=1 | Q8_0 HC read, 2 launches per read for all rows | K/fused_gr.dp.cpp:3577, 2499-2520 |
| STRATA_MMVQ_KREG=1, NIGHT_MMVQ2=1 mask 2, BF16_KREG=1 | KREG kernels for Q4_K/Q5_K/Q6_K/IQ4_XS/IQ3_S and BF16 GEMVs at 1-8 columns; MMVQ_WIDE* stay behind them as fallbacks | K/native_mmvq.dp.cpp:2373-2392, 2636-2657, 4863-4870; K/native_bf16.dp.cpp:549 |
| STRATA_QSA_SCORES_TRANSPOSED=1, QSA_TOPK_SCAN=1, ATTN_VALUE_PIPELINE / NIGHT_ATTN_VPIPE=1, ATTN_MERGE_ACTIVE=1 | variants of the per-row select and attention kernels | K/qsa_select.dp.cpp:1697-1707, 1780, 2226; K/qsa_decode_attn.dp.cpp:736, 1053, 1192 |
| STRATA_ROUTER_TOPK_TREE=1 | tree top-10 | K/native_router.dp.cpp:321, 395 |
| STRATA_NIGHT_EP_PSUM=1, EP_PLAN1=1 | card 1 returns one weighted sum per row; card 0 makes and ships card 1's plan | V:134-146, 1670-1676, 1780-1809, 1957 |
| STRATA_NIGHT_SH_LATE=1, NIGHT_SHEXP=1 | shared expert on the main queue after own experts, 2 fused launches | V:1099, 1881; K/shared_expert.dp.cpp:341 |
| STRATA_IQ2XS_MULTI=1, GU_LANES=8, NIGHT_EP_ROWS=2, EXPERT_SPLIT=0 | IQ2_XS layers through the multi-entry dot; IQ3_S / IQ2_S / IQ3_XXS through the default two-row reuse launchers (LOG:91-94); EP_ROWS only where no reuse launcher claims the type [I] | K/iq_kernels.dp.cpp:5893-5897, 5948-5963, 6079-6092 |
| EP_PREFILL_*, NIGHT_PREFILL_*, PROMPT_ATTN_*, PREFILL_FIRST, OWNED_PRICE | prompt path only | prefill.cpp |
| SLOT_COPY_DIRECT, SLOT_CKPT_KV_DIRECT, NIGHT_PARK_MOVE | admission / parking only | G:8343, 9761-9769 |

## 9. Engineering candidates, ranked by expected C2 / C4 gain

Gains are [I] estimates against the measured windows (C2 24.3 ms, C4 44.1 ms); each has a cheap confirm/kill check.

1. Depth by active slot count (fill the 8-row window). Biggest C2 lever; also fixes the lone-slot case.
   - Why: C2 uses 4 of 8 rows; per-row cost is ~2.8-3.2 ms while a second/third draft row adds ~0.6 / ~0.45 expected
     tokens (C1 solo keeps 3.12 tokens from 4 rows). Expected C2: depth 2 about +4%, depth 3 about +8% today, and
     +14% or more once item 2 lowers the per-row cost; the two-slot depth-2 DD measured ~92 per stream (AGENTS.md).
     A lone slot in batch mode goes from 88 toward the solo 134.6. C4 unchanged (already 8 rows).
   - Change: per-window draft count nd[b] = min(depth, floor(8 / active) - 1) in G:10063-10072 (batch_pack already
     packs variable groups and rotates), keep STRATA_BATCH_MTP_DEPTH as the cap (G:4361), clamp at G:10204 / 10253;
     the pair draft already loops max_steps (M:2424-2429, 2488-2490).
   - Check (config only, no build): C2 harness with STRATA_BATCH_MTP_DEPTH=2 and =3 on the current binary; read the
     "strata batch:" line (rows/window and ms/window). Kill if rows/window does not rise by >= 0.5 per slot.
2. Batch the QSA path per slot / per window under KV streaming (restore what STRATA_NIGHT_ATTN_BATCH did).
   - Expected: C4 -3 to -6 ms (7-14%), C2 -1 to -1.6 ms (4-7%), S=1 -1 ms; context scaling improves too (one
     pooled-key scan per slot instead of per row).
   - Change: V:1546-1579 - drop the kv_mode != 1 exclusion by running qsa_kv_resolve once per slot group (n_q =
     rows of the slot) before the batched attention, as old src-ep verify.cpp:1400-1416 did; V:1442-1475 - use
     kv_append_q8_steps and native_qsa_indexer_append_steps per slot group (they exist for the solo window,
     V:1434-1436, 1464-1466). Kernels: K/qsa_decode_attn.dp.cpp:1339 (rows form takes per-row pools),
     K/qsa_select.dp.cpp (qsa_block_scores_rows), K/kv_stream.dp.cpp:338 (resolve already takes n_q).
   - Check: (a) STRATA_VERIFY_PROFILE=1 (V:843; report printed by G:10271-10273): stamps 10 -> 13 bracket the whole
     select/resolve/attend block (V:1538, 1593) and 8 -> 9 the appends; compare ms/window at R = 2, 4, 8.
     (b) Config-only A/B: C4 at --max-context 65536 with --kv-resident 0 (kv_mode 0, so STRATA_BATCH_ATTN_ROWS
     engages): the run term should fall by the same 3-6 ms. Kill if stage 10 -> 13 is under 2 ms at R = 8.
3. Commit and draft tail (5.8 ms of the C4 window, 2.8 ms at C2, all per-slot serialized).
   a. One-launch GDN commit for any owner/row layout: generalize gdn_commit_layers (K/verify_kernels.dp.cpp:2112;
      eligibility V:3727-3733 hard-codes S == 6, 3-row owners, strides 81920 / 384) to 1-4 owners x 1-4 rows; batch the
      indexer appends per slot and the tail copies (V:3853-3865). 440 -> about 60 launches: C4 -2.2 ms, C2 -1.1 ms.
   b. Do not block the host on the commit (V:4258): the next verify graph goes to the same in-order queue; issue the
      row copies and the drafts first, or submit commit and drafts back to back. Saves one round trip and lets the
      commit overlap the draft kernels: C4 up to -1 ms.
   c. Submit both pairs before waiting (G:10215-10249 calls MtpPair::draft, which waits at M:2514, once per pair) and
      give 4 slots one 4-column tail (M:2320-2334): C4 -0.5 to -1 ms.
   d. Drafter on card 1 while card 0 commits: card 1 has 3.48 GiB free (LOG:63), the drafter takes 0.83 GiB + 48 MiB
      per slot (LOG:21-25); the rows cross as 82 KB per slot. Removes min(commit, emit) from the window: C4 about
      -2.7 ms, C2 -1.3 ms. Larger change (drafter load/bind and its K/V state: M:326, M:811).
   - Check: the "strata batch:" line already splits commit and emit; for b/c add nothing but an env to skip the wait
     and compare the two terms. Kill a/b if commit + emit do not drop by >= 1.5 ms at C4.
4. Expert kernels at 8 rows (both cards, rank 1 cost).
   a. Measure the real expert union: export counts[0] of both plans per layer (K/verify_kernels.dp.cpp:2793; the
      routing audit buffer exists under STRATA_SYCL_WORKLOAD_TRACE, V:1989-1992). This decides b-d.
   b. Size the group grid to the plan instead of 10R rows per card: pass grid_groups (K/iq_kernels.dp.cpp:7628; the
      kernels stride groups when the grid is smaller, 2542) at V:1803-1805 and V:1870-1871; today each card launches
      80 group rows for roughly half of the distinct experts.
   c. A 3-entry and an 8-entry multi dot so hot experts shared by 3 or 5-8 rows decode once (row_entries,
      K/iq_kernels.dp.cpp:2480-2502; IQ3_S already has a three-entry form, LOG:92).
   d. Deal ratio by measured wait: if card 0's ep_wait total exceeds ~1 ms per window at C4, move traffic share toward
      card 0 (or the reverse); residency tables and the foreign rule are K/verify_kernels.dp.cpp:2685-2687,
      V:1671-1676, G:5248.
   - Expected: -0.5 to -2 ms at C4. Check: unitrace durations of the gu / down kernels and of ep_wait_kernel on
     both cards at R = 2, 4, 8 (card 0's ep_wait sum is card 1's lateness).
5. PLE gather off the critical path: submit both graphs first (V:4055-4057), gather while layers 0 runs, raise the flag
   after (the graph already waits at V:1185). C4 -0.5 to -0.9 ms, C2 -0.3 to -0.5 ms. Check: host timestamps around
   V:4004 (STRATA_PLE_TRACE per the S25 notes) against the window.
6. Remaining per-slot launches in the verify graph: one grouped gdn_conv_l2 launch for all slots (V:1302-1311; 36 x
   (S - 1) launches) and one tail-snapshot launch (V:1413-1417). C4 about -0.7 ms. Check: stage 2 -> 3 in the
   profile report.
7. Kernels at 5-8 columns (dense MMVQ, HC, head, shared expert): confirm with unitrace whether KREG at 8 columns is
   still bandwidth-bound (tiles K/native_mmvq.dp.cpp:2353-2371, HC shape K/fused_gr.dp.cpp:2456). If arithmetic-bound,
   an int8 XMX (DPAS) form for >= 6 columns is the candidate (the prompt path has one). Standalone benches exist
   (src/kernels/mmvq_bench.cpp). Expected -1 to -3 ms at C4 only if the check shows < 60% of roofline at 8 columns.
8. Admission path (harness C2, TTFT, stalls of running streams; not a window cost): park-from-slot 44/40 ms on the
   old build vs 1478/1577 ms here, draft_first 0.7-3.7 ms vs 21-29 ms, first-use copies ~1.5 s on both builds
   [M LOG:866-911, OLDLOG:864-912]. Code: G:9984-10023 (park_source), G:13281-13290 -> M:2157-2171 -> M:1694.
   Worth about +10% on the C2 harness figure by itself. Check: a third C2 repeat on the same server (first-use
   effects vanish, a real slow path does not) and perf on the engine thread during an admission.
9. Card balance beyond the tail: moving the shared expert or more routed traffic to card 1 does not shorten the layer
   (card 1's chain already gates card 0); card 1 can only help with work that is not on the layer chain (item 3d)
   or with a slot split of the dense/attention phase, which needs ~4 GiB of dense + HC + head weights on card 1
   against 3.48 GiB free and all K/V on card 0 [I]. Not a near-term item.
10. More than 8 rows (kVerifyMaxT) for depth 2 at C4: only after items 2-4 bring the per-row cost under ~2 ms; today
    a row costs 3.2 ms for at most 0.6 expected tokens (old test: 12-row rotation was slower, night_lab.md:50).
    Limits to lift: KREG and shared expert at 8 columns, kAttnRowsMax 8, plan kernel 128 threads.

Ranked by expected gain. C2: item 1 (+8 to +25% per stream), item 8 (+10% on the harness number), item 2 (+4-7%),
item 3 (+4-8%), items 5-6 (+2-3%). C4: item 2 (+7-14%), item 3 (+6-12%), item 4 (+1-5%), items 5-6 (+2-3%),
item 7 (0-7%, needs the check).

## 10. Limits of this report

- Static reading of commit cf1a2c359; the measured binary strata-dd-0216 is assumed to be built from it [I].
- All family ms in section 7 are estimates fitted to three measured window totals and an old 4-row trace; ranks 1-4
  and 7-15 need the Stage B trace. Ranks 5 and 6 and the window law are measured.
- The expert union D(R) is not measured; the byte tables give bounds.
- The attribution of the run regression to the per-row QSA path is a launch-count argument consistent at three row
  counts (96 / 228 / 492 extra launches vs +1.1 / +1.0-1.6 / +5.9 ms); the solo window also lost 0.7 ms per window
  (138.6 -> 134.6 tok/s) for reasons not identified here.
- Not walked in detail: the inner kernels of ple_block_projected, the MTP round (kernel count taken from the older
  census), the solo commit, and the prompt path. KV RAM restores were only read at source level; no miss counts for
  steady decode exist in the logs.
- Whether PLE BF16 key/value forms and the Hadamard-folded expert branch (V:1763-1765) apply to this pack was not
  verified; the walk assumes BF16 PLE and no folding.

VERDICT -> The window is one linear captured graph per card plus a per-slot serialized tail. Per-stream decode halves
from C1 to C2 because a stream gets 2 rows instead of 4, and nearly halves again at C4 because 8 rows cost 44 ms.
The cheapest large wins are configuration and porting gaps, not new kernels: fill the window by slot count (C2),
batch the QSA path under KV streaming again (C4, C2), and collapse the per-slot commit and draft tail (C4, C2).
