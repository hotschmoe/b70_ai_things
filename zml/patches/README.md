# zml W8A8 patch -- staged contribution (NOT yet submitted)

`zml_w8a8.patch` is a single `git apply`-able patch against upstream zml
(base HEAD `89b0908c`). It is the PR-ready form of the W8A8 work tracked in
`../ZML_W8A8.md`. Apply it into the build clone with `../apply_examples.sh`.

DO NOT submit upstream without the user's explicit go-ahead.

## What it changes

1. **`zml/tensor.zig` -- `Tensor.dotGeneralAcc` + `Tensor.dotAcc`** (the only library
   change; the genuinely upstreamable, model-agnostic piece). A dot whose result /
   accumulation element type is caller-chosen rather than forced to the operand dtype.
   This makes a true integer matmul expressible: `s8 x s8 -> s32`. `Tensor.dot` /
   `dotGeneral` hardcode result dtype = operand dtype (`tensor.zig`), so an `i8` dot
   overflows; the only workaround was `x.convert(.i32).dot(w.convert(.i32))`, which is
   numerically identical but widens the operands to i32 *before* the dot, hiding the
   int8 operands from a backend's int8-GEMM matcher (e.g. oneAPI/oneDNN INT8-XMX on
   Intel Arc). `dotGeneralAcc` mirrors `dotGeneral` exactly except for the result type;
   `dotAcc` is the tag-based convenience mirroring `dot`. Validated bit-for-bit equal to
   the widen-then-dot path on the XLA CPU backend (0/2048 mismatches), which also
   confirms CPU PJRT accepts genuine `s8 dot_general` with an `s32` result.

2. **`examples/w8a8/`** -- M0 CPU int8-dot microbench (self-contained; uses only the
   convert-to-i32 path, so it needs NO library change). Proves i32 accumulation
   (bit-exact) + dequant correctness on CPU.

3. **`examples/llm/models/common_quant.zig`** -- `QuantizedLinear`, a drop-in for
   `zml.nn.Linear` implementing the qwen3.6 W8A8 scheme (per-channel symmetric int8
   weight + per-token symmetric dynamic int8 activation + i32 accumulate via `dotAcc` +
   dequant + optional bias). Loads from a compressed-tensors checkpoint with no loader
   rewrite (two bound `Tensor` fields + `io.load` reflection).

4. **`examples/llm/models/quant_tests.zig`** -- parity vs `nn.Linear(dequant)` (rel_l2
   ~0.007) and the `dotAcc` bit-exactness check.

5. **`examples/llm/models/quant_load_probe.zig`** -- loads a real W8A8 `q_proj` from the
   `w8a8-sqgptq` checkpoint and validates on CPU (lazy: only the bound projection).

6. **`examples/llm/BUILD.bazel`** -- `quant_tests` + `quant_load_probe` targets.

## Suggested PR split (when approved)

- A standalone upstream PR for item 1 (`dotGeneralAcc`/`dotAcc` + a test) -- broadly
  useful, model-agnostic.
- The examples (items 2-6) are our research vehicle; offer as a follow-up example PR or
  keep local, per maintainer interest.

## Regenerate

```bash
cd /mnt/vm_8tb/b70/zml          # clone at base 89b0908c + these changes
git add -N examples/w8a8/main.zig examples/w8a8/BUILD.bazel \
  examples/llm/models/common_quant.zig examples/llm/models/quant_tests.zig \
  examples/llm/models/quant_load_probe.zig
git diff > /mnt/vm_8tb/github/b70_ai_things/zml/patches/zml_w8a8.patch
git reset
```
