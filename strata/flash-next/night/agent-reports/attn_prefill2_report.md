# attn mission 3: XMX prefill attention v2 (S^T = K Q^T) and a chunkwise GDN prototype

Agent: Claude Opus 5.5 (claude-opus-5-5), worktree `attn` (local commits: dc0a1dba mission 1, 59c2743a mission 2;
this work on top). 2026-10-16, 10:51-13:20 UTC. GPU: card 1 only, under the lease.

Deliverables:
- Patch: /mnt/vm_8tb/b70/results/night-20261016/agents/attn_prefill2.patch. It is incremental over mission 2, which is
  already in main.
- Engine: /mnt/vm_8tb/b70/results/night-20261016/builds/strata-attn-p3 (all flags default OFF).
- Benches: `qsa_pattn_bench` (attention) and the new `gdn_chunk_bench`; logs in attn_work/.

Flags:

    STRATA_NIGHT_PATTN_XMX=2       # NEW: S^T form, 64-cell chunks (=1 keeps the mission-2 kernel)
    STRATA_NIGHT_PATTN_TILES=256   # tiles per launch pair (default 256; 512 = ~5% faster, ~75 MB more scratch)
    STRATA_NIGHT_GDN_CHUNK=1       # chunkwise GDN: correct but 4.4x SLOWER - do not enable

## Part 1: prefill attention, mode 2 (S^T = K Q^T)

CONFIG -> same bench as mission 2: one layer-chunk, 24/2 heads, head_dim 256, INT8 K/V. Selections from the real
score + top-k kernels with AR(1) indexer queries (rho 0.9, calibrated to INTEL.md's 80K overlap).
STRATA_ATTN_VALUE_PIPELINE=1 for the current path.

Design changes vs mode 1:
- S^T = K Q^T on joint_matrix. The accumulator's columns are the tile's rows, so each lane holds one row's cells.
  The online softmax runs inside the lane: no local copy of S, no shuffles.
- Sub-group pairs split the 32-cell halves for QK and exchange the row max and sum through local memory.
- K is staged row-major as FP16, coalesced, with no transposition; it is the A operand (2D block loads). Q^T is
  packed once per work-group as a constant B operand.
- P goes once through local memory as the A of O = P V. V keeps its packed B layout. Each sub-group owns 16 rows x
  128 dims (O = 128 GRF).
- 64-cell chunks. 4 K tiles + 1 Q^T tile per k-step, double-buffered.
- All global loads of a chunk's staging are issued before conversion (raw uint4 + scale).
- Launch pairs of 256 tiles instead of 128.

ISA (IGC dump, attn_work/isa/dump_t2): numGRF=256, no spill, 128 `dpas.8x8` per chunk per sub-group, B/A operands
from `load_block2d`.

RESULT (min of 2 runs, ms per layer-chunk):

| chunk | union of 8 | current | mode 1 | mode 2 (256 tiles) | mode 2 (512 tiles) | best speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0-4K (0, 4096) | 1.23x | 52.6 | 19.2 | 15.6 | 14.7 | 3.58x |
| 0-12K (0, 12288) | 1.82x | 181.8 | 95.1 | 78.8 | 74.5 | 2.44x |
| 12-24K (12288, 12288) | 2.55x | 193.1 | 136.8 | 115.2 | 109.7 | 1.76x |
| 24-32K (24576, 8192) | 2.79x | 128.1 | 98.4 | 83.5 | 79.4 | 1.61x |

Accuracy (mode 2 vs the current FP32 path), same as mode 1:

| chunk | max abs | mean abs | max per-row rel |
| --- | ---: | ---: | ---: |
| 0-12K | 1.13e-3 | 1.5e-5 | 1.28e-3 |
| 12-24K | 1.06e-3 | 1.3e-5 | 1.40e-3 |
| 24-32K | 8.3e-4 | 1.3e-5 | 1.56e-3 |

No non-finite values. Selection and top-k are untouched.

Projected (12 QSA layers, 32K request = the three chunks): 6.03 s current -> 3.33 s (256 tiles) / 3.16 s (512
tiles), i.e. -2.7 to -2.9 s, about 0.6-0.8 s more than mode 1. At 4K: -0.44 to -0.45 s.

Tried and dropped:
- L1 prefetch of every chunk tile with `joint_matrix_prefetch` crashed the GPU: UR_RESULT_ERROR_DEVICE_LOST on
  card 1. Card 1 recovered on its own; `xpu-health --card 1` was HEALTHY afterwards (the orchestrator's next serve
  then ran normally). The code was removed. Probably an illegal prefetch block shape for the packed layout.
- Coalesced K loads through a local transpose (mode 1 v8/v9): no gain (see the mission-2 report).

VERDICT -> mode 2 is the better kernel: 1.6-3.6x over the current FP32 path, about 15-20% faster than mode 1, with
the same FP16-level error. It needs the same end-to-end quality gate as mode 1.

## Part 2: GDN prefill, chunkwise (WY) form

Current kernel: `gdn_recurrence` (prefill/kernels.dp.cpp) walks the T tokens sequentially (S=128, HK=16 key heads,
HV=48 value heads, value head h reads key head h % 16). Measured here: 19.1 ms per 12288 tokens per GDN layer
(variant 0, the B70's production path), 7.8 ms at 4096. No chunked path existed in this tree.

Formulation (new file sycl/src/prefill/gdn_chunk.dp.cpp; derivation checked against the scan). Per 64-token chunk,
with G = cumsum(log decay):

    A_ij  = b_i e^{G_i - G_j} k_i.k_j  (j < i)
    (I + A) W = diag(b e^G) K,   (I + A) U = diag(b) V
    QK_ij = e^{G_i - G_j} q_i.k_j      (j <= i)
    Qt = q e^G - QK W,   Ou = QK U

Across chunks:

    Vn = U - W S,   o = Qt S + Ou,   S' = e^{G_L} S + (k e^{G_L - G})^T Vn

Every exponent is <= 0, so nothing overflows.

Kernels:
- K1, all (chunk, head) in parallel: Gram and q.k by 4x4 register blocks from local memory, forward substitution with
  one column per thread in registers (no barriers), then Qt and Ou.
- K2, (head, 32-column block) work-groups, chunks in order: the state in registers, sub-group shuffle reductions.
- K3: the existing output-norm arithmetic.

FP32 throughout. Scratch: 8 chunks per launch pair, ~25 MB.

CONFIG -> gdn_chunk_bench, inputs as gdn_parity's prefill fixture (nonzero initial state; q/k ~ N(0, 0.08^2) per
element; v ~ N(0, 0.7^2); beta in [0.1, 0.9]; shallow and deep decay).

RESULT:

| case | scan ms | chunked ms | state rel (max/ref, L1) | raw y rel (max/ref) | y16 (FP16 out) |
| --- | ---: | ---: | --- | --- | --- |
| T=4096, shallow decay | 7.7 | 27.7 | 1.0e-6, 7.1e-7 | 8.0e-7 | max 1 FP16 ulp, L1 5.7e-7 |
| T=12288, shallow decay | 19.1 | 83.3 | 8.6e-7, 7.5e-7 | 8.5e-7 | max 1 ulp, L1 5.8e-7 |
| T=12288, deep decay | 19.1 | 82.3 | 5.9e-7, 2.1e-7 | 7.3e-7 | max 1 ulp, L1 3.2e-7 |

- Accuracy is at FP32 rounding level against the scan (the order of summation differs, so it is not bitwise).
- Split (timing probes): K2 ~65 ms, K1 ~18 ms, norm ~1.5 ms.
- Staging K2's rows in local memory made it slower (91.7 ms). So K2 is bound by its per-chunk chain (64 rows x two
  shuffle reductions, then the state update) at 6 work-groups per core, not by memory traffic.

VERDICT -> the chunked formulation is validated (state and outputs agree with the scan to ~1e-6 relative), but this
FP32 implementation is 4.4x slower than the tuned token scan. Do not enable STRATA_NIGHT_GDN_CHUNK. To win:
1. Put K1's products (k k^T, q k^T, QK W, QK U: 64x64x128 each) on XMX, with a blocked triangular solve (16x16
   inverses on XMX).
2. Restructure K2 so a work-group carries several column blocks with fewer rows per lane, or so each step's
   (64 x 128) @ (128 x 32) products run as XMX tiles on the state block (S in registers as accumulator tiles).
   Estimate if both land: K1 ~2-3 ms, K2 ~3-5 ms, about 6-8 ms vs 19 ms.

## Files

- sycl/src/kernels/cuda/qsa_prompt_attn_union.dp.cpp: mode 2 (`union_attn_t`), tile-batch knob.
- sycl/src/prefill/gdn_chunk.dp.cpp (new), plus hooks in prefill/kernels.dp.cpp `gdn_recurrence` and
  include/strata/prefill/kernels.hpp.
- sycl/src/kernels/gdn_chunk_bench.cpp (new), and its CMake entry.
