# vLLM registry patch: enable XPU INT8 W8A8 + harden the chooser

These are the vLLM-side Python edits that pair with the native `int8_gemm_w8a8`
op added to vllm-xpu-kernels. Apply them inside the running image's vLLM source
(editable install) at:

    /workspace/vllm/vllm/

NOTE on layout (differs from the blueprint sketch): the INT8 registry and the
chooser live in **`vllm/model_executor/kernels/linear/__init__.py`**, while the
kernel *class* belongs in **`vllm/model_executor/kernels/linear/scaled_mm/`**
(beside `XPUFP8ScaledMMLinearKernel` in `scaled_mm/xpu.py`). The blueprint's
`scaled_mm/xpu.py` path for the class is correct; the registry path it gave
(`linear/__init__.py`) is also correct -- both files are real and were verified
in the image (vllm-xpu-env:v0230).

---

## 1. Drop in the kernel class

Copy `xpu_int8_kernel.py` (this contrib dir) to:

    vllm/model_executor/kernels/linear/scaled_mm/xpu_int8.py

(or paste the `XPUInt8ScaledMMLinearKernel` class into the existing
`scaled_mm/xpu.py`).

---

## 2. `vllm/model_executor/kernels/linear/scaled_mm/__init__.py`

Re-export the new class so it can be imported from the package. Add to the
xpu import block (currently around lines 41-43, which import
`XPUFp8BlockScaledMMKernel`):

```python
from vllm.model_executor.kernels.linear.scaled_mm.xpu import (
    XPUFp8BlockScaledMMKernel,
)
from vllm.model_executor.kernels.linear.scaled_mm.xpu_int8 import (   # ADD
    XPUInt8ScaledMMLinearKernel,                                      # ADD
)                                                                     # ADD
```

and add `"XPUInt8ScaledMMLinearKernel",` to the `__all__` list at the bottom.

(If you pasted the class into `scaled_mm/xpu.py` instead, import it from
`...scaled_mm.xpu` together with `XPUFp8BlockScaledMMKernel`.)

---

## 3. `vllm/model_executor/kernels/linear/__init__.py`

### 3a. Import the class (near the other scaled_mm.xpu import, ~lines 167-170)

The file currently has:

```python
from vllm.model_executor.kernels.linear.scaled_mm.xpu import (
    XPUFp8BlockScaledMMKernel,
    XPUFP8ScaledMMLinearKernel,
)
```

Add the new import right after it:

```python
from vllm.model_executor.kernels.linear.scaled_mm.xpu_int8 import (   # ADD
    XPUInt8ScaledMMLinearKernel,                                      # ADD
)                                                                     # ADD
```

### 3b. Register XPU in `_POSSIBLE_INT8_KERNELS` (currently lines 270-277)

The registry currently reads:

```python
# in priority/performance order (when available)
_POSSIBLE_INT8_KERNELS: dict[PlatformEnum, list[type[Int8ScaledMMLinearKernel]]] = {
    PlatformEnum.CPU: [ZentorchInt8ScaledMMLinearKernel, CPUInt8ScaledMMLinearKernel],
    PlatformEnum.CUDA: [
        CutlassInt8ScaledMMLinearKernel,
        TritonInt8ScaledMMLinearKernel,
    ],
    PlatformEnum.ROCM: [AiterInt8ScaledMMLinearKernel, TritonInt8ScaledMMLinearKernel],
}
```

Add the XPU entry:

```python
_POSSIBLE_INT8_KERNELS: dict[PlatformEnum, list[type[Int8ScaledMMLinearKernel]]] = {
    PlatformEnum.CPU: [ZentorchInt8ScaledMMLinearKernel, CPUInt8ScaledMMLinearKernel],
    PlatformEnum.CUDA: [
        CutlassInt8ScaledMMLinearKernel,
        TritonInt8ScaledMMLinearKernel,
    ],
    PlatformEnum.ROCM: [AiterInt8ScaledMMLinearKernel, TritonInt8ScaledMMLinearKernel],
    PlatformEnum.XPU: [XPUInt8ScaledMMLinearKernel],   # <-- ADD
}
```

### 3c. Harden the chooser (the actual crash fix) -- `choose_scaled_mm_linear_kernel`

This is the standalone-worthy fix. In the real file the offending raw subscript
is at **line 495**, inside `choose_scaled_mm_linear_kernel`:

```python
    platform_kernels = possible_kernels[current_platform._enum]
```

For a platform missing from `possible_kernels` (e.g. XPU + INT8 before this
patch) this raises `KeyError: <PlatformEnum.XPU: ...>` instead of a clear error.
Replace it with a guarded `.get()`:

```python
    platform_kernels = possible_kernels.get(current_platform._enum)
    if not platform_kernels:
        raise ValueError(
            "No ScaledMM linear kernels are registered for platform "
            f"{current_platform._enum}. (If this is INT8/FP8 W8A8, the "
            "platform may not yet have a kernel implemented.)"
        )
```

This guards every caller of `choose_scaled_mm_linear_kernel`
(`init_int8_linear_kernel`, `init_fp8_linear_kernel`,
`init_wfp8_a16_linear_kernel`) -- it also covers the GDN-FP8 path that hits the
same chooser. It is worth landing independently of the XPU INT8 work, since it
turns a bare `KeyError` into an actionable message.

Note: `choose_mp_linear_kernel` (line ~659) has the *same* raw-subscript
pattern (`_POSSIBLE_KERNELS[current_platform._enum]`); apply the same `.get()`
hardening there if you want full coverage, though XPU already has MP kernels
registered so it does not currently crash.

---

## 4. Native-op fake/meta (for completeness, in `vllm/_xpu_ops.py`)

`_xpu_ops.py` registers fake/meta impls guarded by `hasattr`. Add one for the
new op so torch.compile / meta tracing works (mirrors `_int4_gemm_w4a8_fake`,
which is at ~line 58):

```python
if hasattr(torch.ops._xpu_C, "int8_gemm_w8a8"):

    @register_fake("_xpu_C::int8_gemm_w8a8")
    def _int8_gemm_w8a8_fake(
        A: torch.Tensor,
        A_scale: torch.Tensor,
        A_zp: torch.Tensor | None,
        B: torch.Tensor,
        B_scale: torch.Tensor,
        azp_adj: torch.Tensor | None,
        bias: torch.Tensor | None = None,
        out_dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        input_2d = A.view(-1, A.shape[-1])
        M = input_2d.size(0)
        N = B.size(1)
        return torch.empty(
            (M, N),
            dtype=out_dtype if out_dtype is not None else torch.float16,
            device=A.device,
        )
```

This file lives inside the vllm package in the image (`/workspace/vllm/vllm/
_xpu_ops.py`); it is part of vLLM, not the kernels wheel, so it ships with the
vLLM patch (do NOT edit the running image until the parent applies it).

---

## Files in this contrib dir

- `xpu_int8_kernel.py` -- the `XPUInt8ScaledMMLinearKernel` class (section 1).
- `registry_patch.md` -- this file (sections 2-4).

All line numbers were read from the live image vllm-xpu-env:v0230
(`/workspace/vllm/vllm/...`, vLLM editable install) on 2026-06-18.
