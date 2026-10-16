# attn mission 2: prefill QSA attention on XMX (8-query union tiles)

Agent: Claude Opus 5.5 (claude-opus-5-5), worktree `attn` (base e1b7ee8a + mission-1 commit dc0a1dba, local).
2026-10-16, 08:39-11:10 UTC. GPU: card 1 only, under the lease.

Deliverables (final, 10:45; unchanged since the 09:50 first result - later variants measured slower, see 6):
- Patch: /mnt/vm_8tb/b70/results/night-20261016/agents/attn_prefill.patch (incremental over mission 1 =
  main 1382c774; new files included).
- Engine: /mnt/vm_8tb/b70/results/night-20261016/builds/strata-attn-p1 (default OFF).
- Flag: `STRATA_NIGHT_PATTN_XMX=1` (prints one `strata night:` line). INT8 K/V only; any other KV format (q4_0,
  K8V4, FP16) falls back to the existing path.
- Bench: ninja target `qsa_pattn_bench` (sycl/src/kernels/qsa_pattn_bench.cpp); logs in
  /mnt/vm_8tb/b70/results/night-20261016/agents/attn_work/pb*.log; ISA dumps in attn_work/isa/dump_final/.

## 1. Current state of prefill attention

- prefill.cpp:3329 calls `qsa_prompt_attn_batch`. On SYCL it runs only the XMX v2 port, and only when
  `STRATA_PROMPT_ATTN_XMX` is set; otherwise it returns false. The fallback loop then runs the decode chunk kernel
  32 queries per launch (with STRATA_ATTN_VALUE_PIPELINE=1, the S93 value-pipelined variant).
- XMX v2 (qsa_prompt_attn_xmx.dp.cpp) is one work-group per (query, KV head): 12 real rows of 16 (hi/lo FP16
  split), and each query gathers and stages its own 2,051 cells. The K^T staging writes SLM with a 1 KB stride
  between neighbouring threads. Measured here at 12-24K: 548 ms per layer-chunk vs 192 ms for the FP32 path
  (2.85x slower), which is why it is off.

## 2. Design (STRATA_NIGHT_PATTN_XMX)

- Pass 1, one work-group per 8-query tile: the union of the 8 queries' selected cells.
  - Built in 16K-cell local bitmaps, with an 8-bit per-cell "which queries selected it" mask.
  - Written ascending as `cell | mask << 24`.
- Pass 2, one work-group per (tile, KV head), 12 sub-groups x 8 rows = 8 queries x 12 heads (96 rows), grf 256:
  - For each 32-cell chunk of the union, the INT8 K/V rows are gathered once into a per-work-group global staging
    slot (L1/L2-resident), stored as FP16 (code x 64-value scale) in the VNNI-packed layout. Each B tile is then
    one `load_block2d`.
  - S = Q K^T and O += P V on joint_matrix 8x16x16 (FP16 x FP16 -> FP32). The B tiles are double-buffered so a
    load overlaps the previous k-step's DPAS; O stays in registers (128 GRF).
  - Online softmax per row (exp2, running max/sum): a cell the row's query did not select is -inf, so each query
    attends to exactly its own selection. O is rescaled only when some row's max moved.
  - The next chunk's union rows (list + page table) are prefetched during the products: 2 barriers per chunk.
- Selection (block scores, top-k) is untouched, so top-k behaviour is unchanged by construction.

## 3. Accuracy (vs the current FP32 path, identical inputs)

Q, K, V and P each enter the products as FP16 (one rounding each); accumulation is FP32. Same selections.

| chunk | max abs | mean abs | max per-row rel (max diff / max abs ref of the row) | non-finite |
| --- | ---: | ---: | ---: | ---: |
| 0-4K | 1.13e-3 | 1.8e-5 | 1.24e-3 | 0 |
| 0-12K | 1.13e-3 | 1.5e-5 | 1.31e-3 | 0 |
| 12-24K | 1.06e-3 | 1.3e-5 | 1.40e-3 | 0 |
| 24-32K | 8.3e-4 | 1.3e-5 | 1.56e-3 | 0 |

Outputs are O(1) (max |ref| 0.95-2.4), so this is FP16-input rounding level: about 1e-3 relative worst case,
1e-5 mean. The old XMX v2 used hi/lo splits to reach FP32 level. An end-to-end quality gate (needles /
teacher-forced top-1) is still needed before promotion.

## 4. Performance (card 1, one layer-chunk, min of 2-3 runs)

Shapes: 24 q heads, 2 KV heads, head_dim 256, INT8 K/V, page size 4, identity pages.

Selections come from the real block-scores + top-k kernels at the serving capacity, with AR(1)-correlated
indexer queries over positions. rho = 0.9 was calibrated to reproduce INTEL.md's measured 80K overlap:
- calibration run: union of 8 = 3.26x (measured 3.3x), union of 16 = 5.35x (measured 5.1x);
- rho 0.7 gives 5.1x / 9.1x and rho 0.5 gives 6.2x / 11.1x, so the kernel's gain depends on real prompts' overlap.

| chunk (p0, T) | union of 8 | current ms | union XMX ms | speedup | current us/query | new us/query |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0-4K (0, 4096) | 1.23x | 52.6 | 19.0 | 2.76x | 12.8 | 4.65 |
| 0-12K (0, 12288) | 1.82x | 181.4 | 95.1 | 1.91x | 14.8 | 7.7 |
| 12-24K (12288, 12288) | 2.55x | 192.5 | 136.8 | 1.41x | 15.7 | 11.1 |
| 24-32K (24576, 8192) | 2.79x | 128.1 | 98.4 | 1.30x | 15.6 | 12.0 |
| XMX v2 (existing), 12-24K | - | - | 548.1 | 0.35x | - | 44.6 |
| sensitivity: 12-24K at rho 0.97 | 1.80x | 189.6 | 103.7 | 1.83x | 15.4 | 8.4 |

So the gain depends on how much neighbouring prompt positions' selections overlap in real prompts.

Projected:
- 32K request (12 QSA layers x the three chunks): 6.02 s -> 3.96 s of attention, about -2.06 s
  (20.6 -> ~18.5 s, ~1590 -> ~1770 tok/s).
- 4K: -0.40 s of 3.12 s.

These are standalone kernel numbers, not served. Note that this bench's FP32 path at 12-24K is 192 ms, against
the trace's 259 ms for the whole attention family (which includes selection).

## 5. ISA check (IGC dump of the AOT bmg-g31 build, attn_work/isa/dump_final)

- Attention kernel: `numGRF=256`, no spill.
- `dpas.8x8` instructions with B operands from `load_block2d.ugm.d32.a64`; Q / P A tiles come from SLM
  (`load.slm.d32x8t`).
- Union-build kernel: no spill.

Iterations on the way (each measured):

| version | change | 4K ms | 12-24K ms | why it did not move further |
| --- | --- | ---: | ---: | --- |
| v1 | baseline (109 KB SLM) | 32.8 | - | - |
| v2 | Q in registers, rows prefetch, 55 KB SLM | 21.8 | - | 80K unchanged |
| v3 | B tiles from global via 2D block loads instead of 8 single-lane SLM loads each | 22.9 | - | each block load feeds one dpas, no registers for a second buffer |
| v5 | Q back in SLM, double-buffered B | 19.1 | 135.0 | (a 2.9 KB spill version with batched staging loads was dropped) |
| v6/v7 | half of Q in registers; SLM < 64 KB | 19.0 | 136.8 | no change: not the limit |

## 6. What limits it now (measured), and the next step

Phase probes on the 12-24K chunk (137 ms). `STRATA_NIGHT_PATTN_DBG` skipped phases (timing only, outputs invalid;
the knob was removed again from the delivered source). Phases overlap, so the costs are not additive:

| skipped | ms |
| --- | ---: |
| nothing | 137 |
| the attention pass (union build only) | 1.4 |
| all staging | 54.9 |
| K staging only | 69.0 |
| V staging only | 124.6 |
| QK only | 59.6 |
| PV only | 124.0 |
| QK + PV | 44.9 |
| staging + QK + PV (softmax, rows, barriers left) | 25.3 |

- The dominant cost is the K^T round trip: written to the global staging slot, global-fence barrier, then read
  straight back as 2D-block B tiles by QK. Those reads hit freshly written lines (L2 latency), and only 2 B tiles
  are in flight per sub-group.
- V's round trip is mostly hidden: V is read later, after QK and softmax.
- Two attempts to make K staging cheaper did not change the time, which confirms it is the read-after-write
  latency, not the gather or the store count:
  - v8: coalesced K row loads through a local-memory transpose: 141 ms;
  - v9: plus 32-byte K^T stores (8x fewer store messages): 143 ms.
  Both are kept in attn_work/union_v8/v9 sources for reference and are not delivered.

Next steps, in order:
1. 64-cell chunks with 4 K^T B tiles in flight per k-step. This halves per-cell barriers and fences and doubles
   the loads in flight, but needs S kept in registers (no S local buffer) to stay inside 64 KB of local memory
   and 256 GRF.
2. Or compute S^T = K Q^T: K becomes the A operand, staged row-major (no transposition), and Q^T is a per-tile
   constant B operand (could be pre-packed once per tile in global memory).
3. Split PV's head dimension across sub-group pairs, so each V tile serves 16 rows.
4. Real-selection go/no-go: `STRATA_DUMP_SEL=<file>` on a served 32K prompt (dumps the first QSA layer of the last
   chunk), then `qsa_pattn_bench --sel <file> --p0 <prompt_len - T> --T <T>` (the bench prints the real union
   sizes and both timings).
5. Before any promotion: an end-to-end prompt quality gate (needles, teacher-forced top-1), because outputs move
   by about 1e-3 relative.

## 7. Flags and status

    STRATA_NIGHT_PATTN_XMX=1    # prompt QSA attention on XMX, 8-query union tiles (INT8 K/V)

- Engine /mnt/vm_8tb/b70/results/night-20261016/builds/strata-attn-p1 = base e1b7ee8a + mission-1 attention
  batching (same as main 1382c774) + this kernel.
- The patch applies cleanly to main 60b7f900 (`patch --dry-run`).
- Decode is untouched: only prefill.cpp's `qsa_prompt_attn_batch` call reaches it.
- Scratch is allocated lazily on first use: about 34 MB per device (union lists and the K/V staging slots for 256
  tiles).
- Not served end to end by me (no server runs allowed); the A/B belongs to the orchestrator.
