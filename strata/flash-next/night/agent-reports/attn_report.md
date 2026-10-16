# attn: batched verify-window QSA attention, plus bitwise decode-attention kernel fixes

Agent: Claude Opus 5.5 (claude-opus-5-5), worktree `attn`
(/mnt/vm_8tb/b70/build/night-wt/attn/src-ep), base commit e1b7ee8a (night base + HC_Q8).
2026-10-16, 06:45-08:25 UTC. All GPU work: card 1 only, under the lease, short probes.

Deliverables:
- Patch: /mnt/vm_8tb/b70/results/night-20261016/agents/attn.patch (`git diff` incl. the new
  fixture). It applies cleanly (offsets only) to the current main src-ep verify.cpp with
  STRATA_NIGHT_SH_LATE (checked with `patch --dry-run`; the CMake hunk was moved off the
  anchor another agent used).
- Engine (all flags available, all default OFF):
  /mnt/vm_8tb/b70/results/night-20261016/builds/strata-attn-v4
  (v2 = BATCH+VPIPE+MERGE_ACTIVE, v3 = +TOPK_SCAN, v4 = +SCORES_T; all same code paths when
  flags are unset).
- Fixture: ninja target `qsa_verify_batch_parity`
  (sycl/src/kernels/qsa_verify_batch_parity.cpp); runner + logs + reference hashes in
  /mnt/vm_8tb/b70/results/night-20261016/agents/attn_work/.

Recommended flags:

    STRATA_NIGHT_ATTN_BATCH=1 STRATA_NIGHT_ATTN_VPIPE=1 STRATA_NIGHT_ATTN_MERGE_ACTIVE=1 \
    STRATA_NIGHT_TOPK_SCAN=1 STRATA_NIGHT_QSA_SCORES_T=1

## 1. Why the C2 rows ran one at a time

`sycl/src/core/verify.cpp`, `Verifier::record_window`, QSA branch, `batch_rec_` case: each
row's state lives in its own slot session, so the code looped over rows and issued
`qsa_block_scores(..,1)`, `qsa_block_topk(..,1)`, `qsa_kv_resolve(..,1)` and
`qsa_decode_attn_batch(..,1)` (chunk + merge) per row, and also `kv_append_q8_step` and
`native_qsa_indexer_append` per row. The kernels do not need that:

- Causality and selection are per row already: every kernel reads the row's own step
  record (pos, n_kv, n_bid, width) and its own q / scores / ids / scratch row.
- `qsa_decode_attn_batch`: grid (chunks, kv_heads, n_q); work-group z touches only row z.
  The only per-slot argument is the KV pool set.
- `qsa_block_scores` (multi kernel): the rows of one slot share the pooled keys. Per row, a
  long context re-read the slot's whole pooled array (8 MB per row and layer at 64K).
- `qsa_block_topk` (wide kernel on SYCL with max_blocks 50002): one work-group per query,
  and it reads no pools at all.
- `kv_append_q8_steps` and `native_qsa_indexer_append_steps` are the single-cell kernels
  per token, the form C1 already uses.

C1 is already batched. The C1 trace (trace-c1) shows chunk grids {33;2;4} in C1 windows;
{33;2;1} x 6 per layer appears only in the C2 phase. So the C2 gain is batching, and C1
gains only from the kernel changes (items 2-5).

## 2. Changes (each default OFF, latched, one `strata night:` stderr line when on)

1. `STRATA_NIGHT_ATTN_BATCH=1` (verify.cpp, batch windows only). Rows are grouped by slot
   (contiguous `brow_`):
   - scores: one launch per slot;
   - top-k: ONE launch for the whole window;
   - chunk: ONE launch over all slots' pools, through the new `qsa_decode_attn_multi`
     (up to 4 pool sets; query z picks the set of its row range; otherwise it is the same
     `attn_chunk_kernel`);
   - merge: ONE launch;
   - K/V append and indexer append: one launch per slot (K8V4 and FP16 pools keep the
     row loop).

   At C2 the select/attention launches per layer go from 24 to 5 (288 -> 60 per cycle),
   and prep from 12 to 4 (kv x2, indexer x2), so per cycle ~324 fewer graph nodes, and
   fewer event resets with them. `=slot` is a variant that does everything per slot
   (24 -> 8) and needs no new kernel.
2. `STRATA_NIGHT_ATTN_VPIPE=1` (qsa_decode_attn.dp.cpp). Decode-sized calls (n_q <= 8:
   verify C1/C2 and the MTP body) use the existing S93 value-pipelined chunk kernel
   (4 V loads in flight, cells folded in the same ascending order). Until now only the
   nq32 prefill used it.
3. `STRATA_NIGHT_ATTN_MERGE_ACTIVE=1`. The merge folds only chunks below
   ceil(width/64). Chunks past that hold m = -FLT_MAX, which both merge loops skip. Its
   loads are issued 8 deep, in the same fold order. This is aimed at the MTP drafter: its
   32K window is 512 chunks walked by 24 work-groups, one dependent load chain per thread.
4. `STRATA_NIGHT_TOPK_SCAN=1` (qsa_select.dp.cpp, wide top-k). In each of the 4 radix
   passes, thread 0 walked 256 histogram digits serially. That search is now one
   work-group inclusive scan plus one reduce. S(t), the number of cells with digit >= t,
   is non-increasing, so the serial loop's stop digit d = #{t in 1..255 : above + S(t) >=
   width}, and the new "above" count = above + S(d+1). The math is all integer, so the
   ids are the same. This also applies to prefill top-k calls that take the wide kernel.
5. `STRATA_NIGHT_QSA_SCORES_T=1` (qsa_select.dp.cpp, decode multi-scores kernel). The
   4*NQ (query, head) warp reductions of each block are done as one transposed
   butterfly: at offset o, each lane keeps half of its values and adds the partner's
   copy. That costs 16-31 shuffles per block instead of 20*NQ. Each reduction combines
   the same lane sets in the same tree; only the operand order inside an addition
   changes, and IEEE addition is commutative. The head sum keeps the 0 + relu(h0) + ...
   + relu(h3) order.

All are graph-capture safe: no host sync, and every per-token value still comes from the
device step records.

Tried and dropped (bitwise but slower, removed from the patch):
- K-load prefetch in the chunk score phase: chunk +5%.
- 4-deep key prefetch in the scores kernel: scores +25%.
- Removing `use_root_sync` from the chunk/merge kernels: no change (within 0.1 us).

## 3. Correctness

CONFIG -> card 1, `qsa_verify_batch_parity`. Shapes and data:
- 24 q heads, 2 KV heads, head_dim 256.
- INT8 K/V with fp16 64-value scales; page size 4 through a random page permutation.
- Indexer 4x128 with random pooled keys, so top-k selections are random.
- Serving capacity of 200000 cells (max_blocks 50002, which picks the wide top-k kernel as
  in serving).
- 12 layers, each with its own pools (no L2 reuse across layers); random q, qidx and K/V.
- Slot 0's rows at ctx+r and slot 1's at ctx-8+r, so every row has its own n_kv, n_bid
  and width.
- Contexts 1K, 16K and 64K. Shapes C2 = 2 slots x 3 rows, C1 = 1 x 4, C2d1 = 2 x 2.
- Also: the MTP call (1 row, 32K window), top-k on tie-heavy scores (8 distinct values),
  and scores for 5-8 queries.

COMMAND -> `attn_work/run_fixture.sh <name> [ENV..] -- --ctx 1024,16384,65536 --ref
ref3.txt --reps 20`. First with no env, which writes 75 FNV hashes from the original
kernels; then with the flags.

RESULT ->
- Every batched path vs row-by-row, in-process: all 12 layers' attention outputs are
  bitwise equal (589,824 floats per shape), and the selections are identical.
- K/V pools and indexer tail/pooled/dead/block_pos after the per-slot appends are
  byte-identical to the per-row appends.
- With VPIPE, MERGE_ACTIVE, TOPK_SCAN and SCORES_T (separately and together), all 75
  output hashes equal the original-kernel reference. The hashes cover every path, shape
  and context, the MTP call, tie-heavy top-k, and 5-8 query scores.
- PASS, 0 failures in all runs.

VERDICT -> bitwise identical; no tolerance needed.

## 4. Microbench (graph replay, median of 7 x 20 replays of 12 layers; us per attention layer)

"now" is the current code with the original kernels: C2 per row, and C1 already batched.
"final" is all five flags.

| ctx | shape | rows | now | BATCH only | final | prep now | prep final |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1K | C2 2x3 | 6 | 268.4 | 78.6 | 60.9 | 22.7 | 9.7 |
| 1K | C1 1x4 | 4 | 67.9 | 67.9 | 56.5 | 6.3 | 6.2 |
| 1K | C2d1 2x2 | 4 | 186.1 | 78.4 | 60.5 | 16.1 | 9.0 |
| 16K | C2 2x3 | 6 | 541.1 | 193.9 | 148.5 | 22.9 | 9.8 |
| 16K | C1 1x4 | 4 | 131.6 | 131.6 | 96.3 | 7.2 | 7.0 |
| 16K | C2d1 2x2 | 4 | 358.9 | 137.0 | 104.4 | 16.1 | 9.0 |
| 64K | C2 2x3 | 6 | 743.2 | 286.7 | 227.7 | 32.0 | 13.6 |
| 64K | C1 1x4 | 4 | 195.5 | 195.3 | 147.5 | 7.7 | 7.7 |
| 64K | C2d1 2x2 | 4 | 501.4 | 205.4 | 165.4 | 21.1 | 11.4 |

At 1K, the C2 "now" figure (268 us/layer, 3.22 ms per cycle) matches the S136 trace's
254 us per attention layer / 3.05 ms per cycle.

Component split, us per layer (scores / top-k / chunk+merge):

| ctx | path | C2 2x3 | C1 1x4 |
| ---: | --- | --- | --- |
| 1K | now | 16.0 / 6.5 / 243.8 | 5.5 / 1.7 / 59.7 |
| 1K | final | 7.5 / 1.7 / 52.3 | 4.7 / 1.6 / 49.7 |
| 16K | now | 42.2 / 172.9 / 318.5 | 18.9 / 34.0 / 78.7 |
| 16K | final | 26.7 / 18.6 / 106.0 | 14.5 / 18.7 / 65.3 |
| 64K | now | 115.6 / 285.4 / 332.1 | 63.0 / 51.3 / 81.3 |
| 64K | final | 78.6 / 37.6 / 109.9 | 44.5 / 37.6 / 67.1 |

Per-kernel effects in isolation: at 64K, top-k goes 51.5 -> 37.6 us per launch (34 ->
18.7 at 16K), the C1 scores 63 -> 44.5 us, and the C2 chunk+merge (6 rows, one launch)
134 -> 110 us.

MTP drafter attention call (1 row, 32K window = 512 chunks), us per call:

| ctx | now | VPIPE | MERGE_ACTIVE | final |
| ---: | ---: | ---: | ---: | ---: |
| 1K | 86.0 | 79.6 | 41-45 | 35.2 |
| 16K | 222.5 | 199.4 | - | 132.5 |
| 64K | 458.1 | 415.8 | - | 289.8 |

## 5. Estimated savings

These are kernel-level estimates from the microbench, with 12 attention layers. They are
not end-to-end measurements; the orchestrator should A/B them in serving.

| ctx | C2 cycle (6 rows + 4 MTP calls) | C1 window (4 rows + ~3 MTP calls) |
| ---: | ---: | ---: |
| 1K | 2.49 attn + 0.16 prep + 0.20 MTP = 2.85 ms | 0.14 + 0.15 = 0.29 ms |
| 16K | 4.71 + 0.16 + 0.36 = 5.23 ms | 0.42 + 0.27 = 0.69 ms |
| 64K | 6.19 + 0.22 + 0.67 = 7.08 ms | 0.58 + 0.51 = 1.08 ms |

- At short context, 2.85 ms is ~6% of the ~46.5 ms unprofiled C2 cycle. The 3.05 ms the
  trace attributes to attention drops to ~0.73 ms.
- The C1 window count of ~3 MTP attention calls comes from the C1 trace ratio
  ({512;2;1} vs {33;2;4} launches).
- The C1 gain is small at short context (~1% of a window) and grows with context.

## 6. Remaining (not done)

- After this work, C2 attention at long context is scores (78.6 us at 64K: ~40 us per
  slot for 8 MB of pooled keys, still ALU/shuffle-bound) + top-k (37.6 us, one
  work-group per query) + chunk (110 us, latency-bound, 64 cells per work-group).
- Bigger wins need non-bitwise changes: DPAS scores, or smaller chunks with a different
  merge partition. A scores change would alter top-k ties, so it needs a quality gate.
- The MTP chunk kernel at 32K (1024 work-groups) is still ~200 us at 64K.
- Multi-slot single launches for the K/V and indexer appends (4 -> 2 per layer) were not
  done: ~2 us each.
