# SPDX-License-Identifier: Apache-2.0
"""
FakeTensor/meta registration for the W4A8 (int4 weight, int8 activation) decode
path so the PIECEWISE XPU graph capture (cudagraph_mode=PIECEWISE on image
vllm-xpu-env:int8g) can trace through our custom SYCL op `int4_gemm_w4a8`.

WHY THIS EXISTS
---------------
PIECEWISE capture already gives +16.7% decode on W8A8 (23.33 -> 27.23 t/s). That
win is banked by `xpu_int8.py::_register_int8_fakes()`, which registers fakes for
`_xpu_C::dynamic_per_token_int8_quant` and `_xpu_C::int8_gemm_w8a8`. Without a
fake, dynamo's fake-tensor tracing raises UnsupportedOperatorException on the
custom op and graph capture aborts for that subgraph.

The W4A8 path (CompressedTensorsW4A8Int -> kernel class XPUW4A8IntLinearKernel,
installed at vllm/model_executor/kernels/linear/mixed_precision/xpu.py inside the
:int8g image) uses a DIFFERENT custom op: `_xpu_C::int4_gemm_w4a8`. That op has NO
fake registered, so PIECEWISE capture does not cover the int4 decode path. This
module supplies the missing fake.

ACTIVATION QUANT IS ALREADY TRACEABLE
-------------------------------------
XPUW4A8IntLinearKernel.apply_weights quantizes activations with
`vllm._xpu_ops.xpu_ops.dynamic_per_token_int8_quant_ref(x, True, 8)` -- this is a
PURE PyTorch reference (min/max/round/clamp on native ops), NOT the custom
`_xpu_C::dynamic_per_token_int8_quant` op. Native composites are traced by dynamo
directly, so no fake is needed for the activation quant on the W4A8 path. The ONLY
custom op in the W4A8 apply path is `int4_gemm_w4a8`. (The existing int8 fake for
`dynamic_per_token_int8_quant` is irrelevant here; harmless if also loaded.)

OP SCHEMA (from vllm-xpu-kernels/csrc/xpu/torch_bindings.cpp)
------------------------------------------------------------
  int4_gemm_w4a8(Tensor A_, Tensor A_scale, Tensor A_zp, Tensor B,
                 Tensor B_scale, Tensor B_zp, int group_size,
                 Tensor? g_idx, Tensor? bias) -> Tensor

  A_       : quantized activations, int8 (s8), [M, K]  (per-token symmetric)
  A_scale  : [M, 1], activation dtype (f16/bf16)
  A_zp     : [M, 1], int32 (all zeros for symmetric)
  B        : PACKED int4 weight, int32, [K/8, N]  (layer.weight_packed.t())
  B_scale  : [K/group_size, N], activation dtype
  B_zp     : [1] int8 (==8) for symmetric, or [K/group_size, N/8] u4 for asym
  group_size : int (e.g. 128); -1 => per-channel
  g_idx    : optional, None on this path (GPTQ desc_act not used)
  bias     : optional

  RETURN: [M, N], dtype HARD-CODED torch.float16 (NOT derived from A_scale).
  C++ (onednn_matmul.cpp::int4_gemm_w4a8): result =
      check_and_create_output_tensor(A, B, torch::kHalf); for 2D A this is
      result_shape = {A.size(0), B.size(1)} = [M, N], options.dtype(kHalf).

So N = B.shape[1] (mat2 is [K/8, N]), M = A_.shape[0], out dtype = float16.

HOW TO HOOK IT
--------------
Three equivalent options; pick whichever is cleanest for the deploy.

  (a) Import-time: this module calls `_register_int4_fakes()` at the bottom, so a
      bare `import xpu_int4` in any process that does graph capture registers the
      fake. Mirror xpu_int8.py, which is imported because its kernel class is
      referenced by the int8 scheme.

  (b) is_supported hook: add a `_register_int4_fakes()` call inside
      `XPUW4A8IntLinearKernel.is_supported()` (the MPLinearKernel base uses
      can_implement, not is_supported -- if no is_supported exists, call it from
      `can_implement` right after the XPU/op presence checks pass). This guarantees
      registration in the exact worker process that selected the W4A8 kernel.

  (c) Unified: fold the int4 fake into xpu_int8.py::_register_int8_fakes() so a
      single import covers both decode paths. See
      contrib/vllm_int8_xpu/xpu_int4_register_fake.diff for that variant; it is the
      lowest-friction deploy because xpu_int8.py is already imported on the int8
      image.

DEPLOY: this file (or the diff) must land next to the installed vllm xpu kernel
module inside the running container; see docs/kernel/patches/A1_graph_capture_w4a8.md
for the exact in-image path and import wiring.
"""

import torch

# Idempotency + lazy-load guard, mirroring xpu_int8.py.
_FAKES_REGISTERED = False


def _register_int4_fakes():
    global _FAKES_REGISTERED
    if _FAKES_REGISTERED:
        return
    register_fake = getattr(torch.library, "register_fake", None) \
        or getattr(torch.library, "impl_abstract", None)
    if register_fake is None:
        return
    # Force-load the _xpu_C library so the op SCHEMA exists before we register the
    # fake (register_fake requires the op to be defined). Don't set the flag if it
    # is not loaded yet -> retry on the next call.
    try:
        import vllm._xpu_ops  # noqa: F401  (triggers torch.ops._xpu_C library load)
    except Exception:
        pass
    if not hasattr(torch.ops._xpu_C, "int4_gemm_w4a8"):
        return

    # int4_gemm_w4a8(A_ i8 [M,K], A_scale, A_zp, B int32 [K/8,N], B_scale, B_zp,
    #                group_size, g_idx?, bias?) -> [M, N] float16
    # Output shape: M = A_.shape[0], N = B.shape[1] (mat2 is the packed [K/8, N]).
    # Output dtype is HARD-CODED float16 in the C++ kernel (torch::kHalf), NOT
    # derived from any input. new_empty preserves the (fake/meta) device and the
    # default contiguous layout while overriding the dtype to float16.
    def _fake_int4_gemm_w4a8(A_, A_scale, A_zp, B, B_scale, B_zp,
                             group_size, g_idx, bias):
        return A_.new_empty((A_.shape[0], B.shape[1]), dtype=torch.float16)

    import sys
    name = "_xpu_C::int4_gemm_w4a8"
    try:
        register_fake(name, _fake_int4_gemm_w4a8)
        print(f"[xpu_int4] registered fake for {name}", file=sys.stderr, flush=True)
    except (RuntimeError, ValueError) as e:
        # already registered (e.g. native abstract impl present) -> fine
        print(f"[xpu_int4] register_fake({name}) skipped: {e}",
              file=sys.stderr, flush=True)
    _FAKES_REGISTERED = True


# Register at import time so the fake is present in whichever process imports this
# module (the engine-core worker that runs graph capture). See module docstring
# for the other hook points.
_register_int4_fakes()
