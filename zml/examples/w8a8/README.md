# zml W8A8 microbench (M0)

Milestone M0 of [`../../ZML_W8A8.md`](../../ZML_W8A8.md): a standalone zml example that
expresses the qwen3.6-27b W8A8 linear (per-channel symmetric int8 weight + per-token
symmetric dynamic int8 activation + i32 accumulate + dequant) and validates it on the XLA
**CPU** backend with NO GPU and NO zml library change.

This directory is the repo-canonical source. The bazel build runs from the git-ignored
upstream clone at `/mnt/vm_8tb/b70/zml`. Sync + run:

```bash
bash zml/apply_examples.sh                     # repo -> clone
cd /mnt/vm_8tb/b70/zml
~/.local/bin/bazelisk run //examples/w8a8 --config=release
```

(CPU only -- do NOT pass `--@zml//platforms:oneapi=true`. No `gpu-run` lease needed; the
daily driver is untouched.)

## What it proves

The compiled graph returns three tensors so the host applies two independent gates:

- **GATE 1 -- genuine i32 accumulation (bit-exact).** The graph returns the raw
  `i8 x i8 -> i32` accumulator AND the device's own int8-quantized activations. The host
  recomputes the integer accumulation from those activations and our i8 weights and
  requires a bit-exact match. The accumulator values (e.g. -96611) are far outside the
  i16 range (+/-32767); had XLA used an i8/i16 result dtype the sums would wrap and the
  check would fail. This is the proof that `x_i8.convert(.i32).dot(w_i8.convert(.i32))`
  accumulates in i32 on CPU.
- **GATE 2 -- dequant correctness (tolerance).** The dequantized f32 output is compared
  against an INDEPENDENT full-precision host reference `x @ dequant(W)`. The gap is bounded
  by activation-quant error.

## Last result (2026-06-30, CPU)

```
shapes: x[16,256] w[128,256] -> y[16,128]
GATE 1 (i32 accumulation, bit-exact): PASS (2048 elements)
GATE 2 (dequant vs f32 ref): rel_l2 = 0.00717, max_abs = 1.69096 -> PASS
RESULT: PASS -- zml expresses W8A8 with genuine i32 accumulation on CPU.
```

## Notes for later milestones

- The CPU path promotes operands to i32 before `.dot` (zml's public `dot` hardcodes result
  dtype = operand dtype). For the GPU INT8-XMX payoff (M4) the inserted `convert(s8->s32)`
  ops may hide the int8 operands from the oneAPI/oneDNN matcher; that path needs a real
  s8-operand dot via a `Tensor.dotGeneralAcc(.i32)` helper added to `zml/tensor.zig`.
- zml's reduce keeps the reduced axis as size 1, and binary ops auto-broadcast when tags
  match, so the per-token scale (`{token, d=1}`) broadcasts cleanly -- simpler than the
  rank-1 sketch in the feasibility doc.
