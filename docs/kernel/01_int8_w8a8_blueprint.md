# 01 - XPU INT8 W8A8 scaled-MM kernel for Arc Pro B70: Implementation Blueprint

**Snapshot: 2026-06-18.** Implementation spec for adding an INT8 W8A8 scaled-MM linear kernel
for Intel XPU (Battlemage/Xe2, B70) to vLLM, so compressed-tensors and Quark W8A8-INT8
checkpoints LOAD and RUN on the B70 instead of crashing with `KeyError: PlatformEnum.XPU`.
Derived from a dedicated source+web research pass; builds on [literature/06](../literature/06_xpu_kernel_fastpaths.md).

Confidence markers: [VERIFIED] from source in our image / GitHub; [INFERRED] derived; [RISK] open.

## 0. Verdict

**Mostly plumbing + one genuine new line.** The activation-int8 path, oneDNN matmul wrapper,
per-token src-scale attrs, VNNI reorder, primitive cache, and the Python kernel pattern ALL
already ship in our image and are proven by the live `int4_gemm_w4a8` op (which already does
`s8 activations x quantized weights -> dequant` through oneDNN). What is missing is one new
joint dtype (`s8 x s8 -> s32`) wired through the wrapper + op registration + a vLLM kernel class
+ one registry line. **No community patch does this yet -> we would be first.**

## 1. Where the ops live [VERIFIED]

- Native ops are in the separate repo **`github.com/vllm-project/vllm-xpu-kernels`** (SYCL/DPC++ +
  oneDNN), shipped as the pip wheel `vllm_xpu_kernels` (0.1.7 in our image). NOT in vLLM `csrc`,
  NOT in `intel/llm-scaler`.
- Proof: `vllm/_xpu_ops.py` imports from `vllm_xpu_kernels.*` and only registers *fake* meta-impls
  for ops already in `torch.ops._xpu_C`. Real impls are compiled into the `_xpu_C` extension.
- Live op set in our wheel: `fp8_gemm`, `fp8_gemm_w8a16`, `int4_gemm_w4a16`, `int4_gemm_w4a8`.
  **No int8 GEMM.**
- Source files to study/edit (in vllm-xpu-kernels):
  - `csrc/xpu/onednn/int4_gemm_w4a8.h`  <- PRIMARY TEMPLATE (s8 activation path)
  - `csrc/xpu/onednn/fp8_gemm_w8a8.h`    <- an FP8 W8A8 header exists but isn't registered (reference)
  - `csrc/xpu/onednn/onednn_ext.h`       <- `joint_dtypes_t` enum + dtype->`memory::data_type` map
  - `csrc/xpu/onednn/onednn_matmul.cpp`, `onednn_runtime.{cpp,h}`, `lru_cache.h`
  - `csrc/xpu/torch_bindings.cpp`        <- `TORCH_LIBRARY(_xpu_C, ...)` def + impl
  - (vLLM) `vllm/_xpu_ops.py`            <- fake/meta registration

## 2. Community work survey [VERIFIED]

| Work | URL | Usable? |
|---|---|---|
| RFC XPU kernel migration | vllm-project/vllm#33214 | umbrella tracker; INT8 W8A8 NOT listed. Context only |
| RFC Intel Quant Roadmap H1-2026 | vllm-project/vllm#37979 | XPU scope = "Linear/MoE W8A16 FP8" + wNa16-INT. **INT8 W8A8 NOT on roadmap** -> nobody upstream owns it |
| PR #34117 add xpu scaled_mm | vllm-project/vllm | added `XPUFP8ScaledMMLinearKernel` (FP8). Structural template for our class |
| `int4_gemm_w4a8` + `XPUW4A8IntLinearKernel` | in our image | CLOSEST existing thing; near-superset of what we need. Real template |
| llm-scaler #339 (B70 setup pain) | intel/llm-scaler#339 | no int8 patch. Not usable |
| Quark W8A8 99 t/s author + patch | - | **could not locate** anywhere public. Not duplicating an existing release |
| compressed-tensors INT8 AZP #9344; vllm-ascend MoE w8a8-int8 #5718 | vllm / vllm-ascend | AZP/MoE math reference (inspiration) |

**Bottom line: no mergeable XPU INT8 W8A8 patch exists. The in-image W4A8 op is the start point.**

## 3. Implementation blueprint

### 3a. Native op (in vllm-xpu-kernels)

Activation int8 quant happens in **Python** (W4A8 uses `ops.dynamic_per_token_int8_quant_ref`);
the op receives pre-quantized s8 acts + per-token scale. Mirror that. New file
`csrc/xpu/onednn/int8_gemm_w8a8.h` (model on `int4_gemm_w4a8.h`):

```cpp
// per-token dynamic-int8 activations x per-channel int8 weights
torch::Tensor int8_gemm_w8a8(
    const torch::Tensor& A,           // [M,K] s8 (pre-quantized in Python)
    const torch::Tensor& A_scale,     // [M,1] f32  per-token act scale
    const std::optional<torch::Tensor>& A_zp,   // [M,1] s32 or None (sym=None)
    const torch::Tensor& B,           // [K,N] s8  weight (transposed)
    const torch::Tensor& B_scale,     // [1,N] f32 per-channel weight scale
    const std::optional<torch::Tensor>& azp_adj, // weight col-sum term, None if sym
    const std::optional<torch::Tensor>& bias,
    at::ScalarType out_dtype);        // f16/bf16
```

oneDNN setup (copy the `f_attr` + `matmul_primitive_create_and_cache` flow from int4_gemm_w4a8.h):
1. In `onednn_ext.h` add `joint_dtypes_t::s8_s8` -> `(s8 src, s8 weights, out_md dst)`.
   **THIS IS THE ONE TRULY NEW LINE** (no s8xs8 combo is plumbed today). oneDNN supports
   `s8s8s32` natively on Battlemage XMX. [RISK #1]
2. src md {M,K} s8; weights md {K,N} s8 with `format_tag::any` -> oneDNN inserts VNNI reorder
   internally, cached in LRU. No manual VNNI.
3. dst md {M,N} = out_dtype; oneDNN accumulates s32 internally, applies output scales -> f16/bf16.
4. per-token act scale: `pattr.set_scales(DNNL_ARG_SRC, mask=1<<0, {M}, f32)`. Symmetric -> no src zp.
5. per-channel weight scale: `pattr.set_scales(DNNL_ARG_WEIGHTS, mask=1<<1, {N}, f32)`.
6. asym acts (phase 2): `set_zero_points(DNNL_ARG_SRC,...)` + fold AZP via `azp_adj` like
   `CPUInt8ScaledMMLinearKernel.process_weights_for_onednn` (`azp_adj = weight.sum(0)*weight_scale`).
   Ship SYMMETRIC-only first; reject asym in `can_implement`.
7. bias: oneDNN matmul bias post-op (broadcast {1,N}), like the FP8 path.

Register in `csrc/xpu/torch_bindings.cpp`:
```cpp
xpu_ops.def("int8_gemm_w8a8(Tensor A, Tensor A_scale, Tensor? A_zp, Tensor B, Tensor B_scale, "
            "Tensor? azp_adj, Tensor? bias, ScalarType out_dtype) -> Tensor");
xpu_ops.impl("int8_gemm_w8a8", torch::kXPU, &int8_gemm_w8a8);
```
Add a fake/meta in `vllm/_xpu_ops.py` guarded by `hasattr(torch.ops._xpu_C,"int8_gemm_w8a8")`,
returning `torch.empty((M,N), dtype=out_dtype, device=A.device)`.

### 3b. vLLM kernel class

New `XPUInt8ScaledMMLinearKernel(Int8ScaledMMLinearKernel)` in
`vllm/model_executor/kernels/linear/scaled_mm/xpu.py` (beside `XPUFP8ScaledMMLinearKernel`).
Base provides `_get_layer_params` unpacking `["weight","weight_scale","input_scale",
"input_zero_point","azp_adj"]`. `CompressedTensorsW8A8Int8.create_weights` gives: weight int8 [N,K];
weight_scale f32 [N,1] (channel) or [num_logical] (tensor); input_scale/zp = None for dynamic
(our B70 target); azp_adj = None.

```python
class XPUInt8ScaledMMLinearKernel(Int8ScaledMMLinearKernel):
    @classmethod
    def is_supported(cls, compute_capability=None):
        if not current_platform.is_xpu():
            return False, "XPUInt8ScaledMM only supported on XPU"
        return True, None
    @classmethod
    def can_implement(cls, c):
        if c.is_static_input_scheme:
            return False, "XPU int8 kernel supports dynamic activation quant only"
        if not c.input_symmetric:
            return False, "XPU int8 kernel supports symmetric activations only"
        return True, None
    def process_weights_after_loading(self, layer):
        w_q, w_s, i_s, i_zp, azp_adj = self.layer_param_names
        weight = getattr(layer, w_q)
        replace_parameter(layer, w_q, Parameter(weight.t().contiguous().data, requires_grad=False))
        weight_scale = getattr(layer, w_s)
        if len(layer.logical_widths) > 1 and not self.config.is_channelwise:
            weight_scale = convert_to_channelwise(weight_scale, layer.logical_widths)
        replace_parameter(layer, w_s, Parameter(weight_scale.reshape(1,-1).contiguous().data, requires_grad=False))
    def apply_weights(self, layer, x, bias=None):
        from vllm._xpu_ops import xpu_ops as ops
        w_q, w_s, i_s, i_zp, azp_adj = self._get_layer_params(layer)
        x_2d = x.reshape(-1, x.shape[-1])
        x_q, x_s, x_zp = ops.dynamic_per_token_int8_quant_ref(x_2d, True, 8)
        out = torch.ops._xpu_C.int8_gemm_w8a8(x_q, x_s, None, w_q, w_s, None, bias, x.dtype)
        return out.reshape(x.shape[:-1] + (out.size(-1),))
```
Import it in `vllm/model_executor/kernels/linear/__init__.py`. Note
`dynamic_per_token_int8_quant_ref` is a `@torch.compile` reference (correct; fused SYCL op = later perf).

### 3c. Registry change (the actual crash fix)

In `vllm/model_executor/kernels/linear/__init__.py`:
```python
_POSSIBLE_INT8_KERNELS = {
  PlatformEnum.CPU:  [ZentorchInt8ScaledMMLinearKernel, CPUInt8ScaledMMLinearKernel],
  PlatformEnum.CUDA: [CutlassInt8ScaledMMLinearKernel, TritonInt8ScaledMMLinearKernel],
  PlatformEnum.ROCM: [AiterInt8ScaledMMLinearKernel, TritonInt8ScaledMMLinearKernel],
  PlatformEnum.XPU:  [XPUInt8ScaledMMLinearKernel],   # <-- ADD
}
```
Harden the lookup in `choose_scaled_mm_linear_kernel` (raw subscript = the KeyError):
```python
platform_kernels = possible_kernels.get(current_platform._enum)
if not platform_kernels:
    raise ValueError(f"No ScaledMM linear kernels for platform {current_platform._enum}")
```
The `.get()` hardening alone is a worthwhile standalone fix (clear error vs KeyError; also covers
the GDN-FP8 path that hits the same chooser).

### 3d. Build/test loop on the B70

Rebuild the WHEEL, not vLLM. The vllm-xpu-env image already has the SYCL toolchain (icpx + oneDNN):
```bash
git clone https://github.com/vllm-project/vllm-xpu-kernels   # under /mnt/vm_8tb/b70/ (SSD)
# add int8_gemm_w8a8.h, the s8_s8 enum entry, the binding
ssh b70 'docker run --rm -v <repo>:/src --entrypoint bash vllm-xpu-env:v0230 -c \
  "cd /src && source /opt/intel/oneapi/setvars.sh 2>/dev/null; python setup.py bdist_wheel"'
# install into a THROWAWAY container for test (never mutate the running image):
ssh b70 'docker run --rm -v <repo>/dist:/w --entrypoint bash vllm-xpu-env:v0230 -c \
  "pip install --force-reinstall /w/vllm_xpu_kernels-*.whl && \
   python -c \"import torch,vllm._xpu_ops; print(hasattr(torch.ops._xpu_C,\\\"int8_gemm_w8a8\\\"))\""'
```
Numerical check (small shapes): `out ~= (x_q.float() @ w_q.float()) * x_s * w_s + bias`.
End-to-end: serve our existing `Qwen3-14B-W8A8-INT` checkpoint -> should now load + select our kernel.

## 4. Vehicle

**Fork `vllm-xpu-kernels` (add the op) + carry a 2-file vLLM Python patch** (kernel class + registry).
The op MUST live in the kernels repo (that's where `_xpu_C`/oneDNN compiles + shares the primitive
cache); an out-of-tree extension would duplicate all of it.

Upstream PR targets (both, in order):
1. `vllm-project/vllm-xpu-kernels`: add `int8_gemm_w8a8` + the `s8_s8` joint dtype. Cite #33214,
   note it unblocks compressed-tensors/Quark W8A8-INT8 (which #37979 omits).
2. `vllm-project/vllm`: add `XPUInt8ScaledMMLinearKernel` + `_POSSIBLE_INT8_KERNELS[XPU]` + `.get()`
   hardening (the hardening is a standalone-worthy fix).

## 5. Ranked risks

1. [RISK] **s8xs8->s32 joint dtype not plumbed** in the wrapper -- the one non-trivial bit. Add the
   enum entry + `memory::data_type` map + dispatch. oneDNN supports s8s8s32 natively on Battlemage ->
   wrapper-side wiring, not a hardware gap. Medium effort, low uncertainty.
2. per-token src scale granularity: W4A8 proves `set_scales(DNNL_ARG_SRC, per-token)` works in this
   oneDNN build. Low. (fallback: apply act scale as a Python post-multiply on f32 output.)
3. VNNI/packing: none needed (`format_tag::any`). Low (don't hardcode a weights tag).
4. asym/static schemes: dynamic-per-token-symmetric is the easy 80% (our target). Static/asym (AZP)
   = port the CPU kernel's correction. Scope symmetric-only first.
5. activation-quant op is a `@torch.compile` reference, not fused -> per-token reduction each layer.
   Correct; fused SYCL quant = perf follow-up.
6. **PERF EXPECTATION:** at decode (M~1, bandwidth-bound) INT8 W8A8 ~TIES FP8 (both ~1 byte/weight).
   The win is **prefill/large-batch** (compute-bound XMX s8s8s32) + VRAM + **being able to load the
   W8A8 checkpoint at all** (vs the current crash). Do not oversell a decode speedup.
7. output dtype: int4 path prefers f16. Confirm bf16 dst is clean; else `--dtype float16` or cast.

**Net:** genuinely "just plumbing" except risk #1. Everything else is proven in our running image
by `int4_gemm_w4a8`.

## Sources
- github.com/vllm-project/vllm-xpu-kernels (csrc/xpu/onednn/*, csrc/xpu/torch_bindings.cpp)
- vllm-project/vllm#33214 (RFC XPU kernel migration; PR #34117)
- vllm-project/vllm#37979 (RFC Intel Quant Roadmap; XPU = FP8 W8A16 only)
- vllm-project/vllm#9344, vllm-project/vllm-ascend#5718 (AZP / MoE int8 references)
- image vllm-xpu-env:v0230: vllm/_xpu_ops.py, model_executor/kernels/linear/{__init__,scaled_mm/*,mixed_precision/xpu}.py
