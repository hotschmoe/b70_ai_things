# int8_gemm_w8a8 baseline profile (2026-06-20) -- the hand-tuning starting line

Measured `torch.ops._xpu_C.int8_gemm_w8a8` (our oneDNN s8s8s32 kernel) in isolation on the B70, Qwen3-14B
shapes (`w8a8/20_microbench_int8_gemm.sh`, via the gpu-run lease). Peak refs: 608 GB/s, 367 INT8 TOPS.

## DECODE (m=1, BW-bound; int8 weights ~14 GB -> ceiling ~43 t/s)
| shape (k,n)        | GB/s  | % of 608 |
|--------------------|-------|----------|
| 4096 x 11008 (wide-n) | 307 | **50.5%** (worst) |
| 5120 x 17408       | 466   | 76.7% |
| 17408 x 5120 (tall)| 565   | **92.9%** (near-peak) |
| 5120 x 5120        | 530   | 87.2% |

## PREFILL (m, k, n) -- compute-bound; target 367 TOPS
| m    | shape        | TFLOP/s | % of 367 |
|------|--------------|---------|----------|
| 512  | 4096 x 11008 | 266     | 72.5% |
| 512  | 5120 x 17408 | 270     | 73.5% |
| 512  | 17408 x 5120 | 243     | 66.2% |
| 512  | 5120 x 5120  | 250     | 68.2% |
| 2048 | 5120 x 17408 | 298     | 81.2% |
| 2048 | 17408 x 5120 | 282     | 76.9% |

## impl (ONEDNN_VERBOSE=2) -- the WHY
- Every shape (prefill AND decode) lands on **`jit:gemm:any`** -- the GENERAL oneDNN JIT GEMM. No
  shape-specialized kernel, **no dedicated m=1 GEMV** (it split-K-emulates the GEMV at decode).
- Memory descriptors: `src_a:s8::blocked:ab src_b:s8::blocked:ab dst:f16::blocked:ab`. The **weight (src_b)
  is plain ROW-MAJOR `ab`, NOT VNNI / XMX-packed.** This is the smoking gun:
  - Prefill: the XMX systolic array wants VNNI-interleaved int8 weights to hit peak; row-major forces a
    repack/strided read inside the jit kernel -> only 66-81% of TOPS.
  - Decode wide-n: row-major weight with the gemm-as-gemv path strides poorly -> 50% BW at (4096,11008).
- Scales: `src0:3:f16` (per-token) + `wei:2:f16` (per-channel), NO zero-points (symmetric -- clean; the
  w4a8 carried a wasteful src-zp, w8a8 does not).

## TARGETS for hand-tuning (grounded)
1. **PREFILL: 66-81% -> 90%+ of 367 TOPS.** Lever B4 (weights as `format_tag::any` -> oneDNN picks a
   VNNI/XMX-packed layout once, offline) is the cheapest; a hand-written joint_matrix/DPAS GEMM with
   VNNI weights is the ceiling. ~1.2-1.5x prefill headroom.
2. **DECODE: lift the wide-n shapes 50% -> 85%+.** A dedicated int8 GEMV (one sub-group per output row,
   vectorized dp4a/DPAS, coalesced VNNI-or-reordered weight, sub-group reduce) -- the tall shapes are
   already 87-93% so the win is shape-specific. ~1.5-1.8x on the worst shapes.
3. **Cross-cutting:** the row-major weight hurts BOTH regimes -> an offline VNNI/blocked weight reorder
   (in process_weights_after_loading) may be the single highest-leverage, lowest-risk first move.

Full lever maps + codex-drafted kernels: docs/kernel/{10_int8_gemm_handtune_plan,11_fused_quant_handtune_plan}.md.
