# A1 -- Extend PIECEWISE XPU graph capture to the W4A8 decode path

Banked: PIECEWISE XPU graph capture gives +16.7% W8A8 decode (23.33 -> 27.23 t/s),
via image `vllm-xpu-env:int8g` + `cudagraph_mode=PIECEWISE` + `register_fake`
meta-kernels for our 2 custom int8 ops. The W4A8 path uses a DIFFERENT custom op
(`int4_gemm_w4a8`) with NO fake registered, so dynamo aborts capture on the int4
linear pieces. This patch supplies the missing fake. Expect a similar free lift.

Serving is a GPU touch -> the lead runs it; this doc only prepares.


## 1. The exact op schema

From `vllm-xpu-kernels/csrc/xpu/torch_bindings.cpp` (verified live on the host and
inside the `:int8g` image via `torch.ops._xpu_C.int4_gemm_w4a8._schemas`):

    int4_gemm_w4a8(Tensor A_, Tensor A_scale, Tensor A_zp, Tensor B,
                   Tensor B_scale, Tensor B_zp, int group_size,
                   Tensor? g_idx, Tensor? bias) -> Tensor

Arg semantics (header `csrc/xpu/onednn/int4_gemm_w4a8.h` +
`onednn_matmul.cpp::int4_gemm_w4a8`):

    A_       int8 (s8)         [M, K]            per-token symmetric quant activations
    A_scale  act dtype (f16)   [M, 1]            per-token scale
    A_zp     int32             [M, 1]            all zeros (symmetric)
    B        int32 (PACKED u4) [K/8, N]          packed int4 weight, NT format
    B_scale  act dtype         [K/group_size, N] per-group weight scale
    B_zp     int8              [1] (==8)         symmetric; or [K/gs, N/8] u4 asym
    group_size int                               e.g. 128; -1 => per-channel
    g_idx    Tensor?           None              GPTQ desc_act unused on this path
    bias     Tensor?           optional

RETURN: shape `[M, N]`, dtype **hard-coded `torch.float16`** (`torch::kHalf`), NOT
derived from any input. Proof, `onednn_matmul.cpp`:

    torch::Tensor result = check_and_create_output_tensor(A, B, torch::kHalf);
    // check_and_create_output_tensor, A.dim()==2:
    //   result_shape = {A.size(0), B.size(1)};   // = [M, N]
    //   options = A.options().dtype(out_dtype_);  // out_dtype_ = kHalf

So `M = A_.shape[0]`, `N = B.shape[1]` (B is the packed `[K/8, N]` mat2), out =
float16. (XPUW4A8IntLinearKernel.can_implement even warns when model dtype != fp16,
because the op always emits fp16; `apply_weights` ends with `out.to(x.dtype)`.)


## 2. The exact apply path (XPUW4A8IntLinearKernel)

Installed in the `:int8g` image at
`vllm/model_executor/kernels/linear/mixed_precision/xpu.py`
(scheme: `CompressedTensorsW4A8Int` -> kernel `XPUW4A8IntLinearKernel`). Verbatim:

    def apply_weights(self, layer, x, bias=None):
        reshaped_x = x.reshape(-1, x.shape[-1])              # [M, K]
        from vllm._xpu_ops import xpu_ops as ops
        quant_x, x_scale, x_zero = ops.dynamic_per_token_int8_quant_ref(
            reshaped_x, True, 8)                             # PURE PyTorch ref
        out = torch.ops._xpu_C.int4_gemm_w4a8(
            quant_x,                 # A_   i8 [M,K]
            x_scale,                 # A_scale [M,1]
            x_zero,                  # A_zp  [M,1] int32 (zeros)
            layer.weight_packed.t(), # B    int32 [K/8,N]  (packed; .t() of [N,K/8])
            layer.weight_scale,      # B_scale [K/gs,N]
            layer.weight_zero_point, # B_zp  [1] int8 (==8)
            self.config.group_size,  # e.g. 128
            None,                    # g_idx
            bias,
        )
        return out.to(x.dtype)

Weight packing (`process_weights_after_loading` / `_pack_int4_weight`): the loaded
`weight_packed` is int8 `[N, K]` with signed int4 values in [-8,7]; it is shifted
+8 to u4 and bit-packed 8-per-int32 into `[N, K/8]` (int32), then passed as `.t()`
=> `[K/8, N]` to the op. `group_size` comes from the checkpoint (gptq @128).

KEY CONSEQUENCE FOR FAKES: the activation quant is
`dynamic_per_token_int8_quant_ref`, a PURE-PyTorch reference (min/max/round/clamp
on native ops), NOT the custom `_xpu_C::dynamic_per_token_int8_quant` op. Dynamo
traces native composites directly, so the W4A8 path needs **no** activation-quant
fake. The ONLY custom op in the W4A8 apply path is `int4_gemm_w4a8`. (The existing
int8 fakes are unrelated to W4A8; harmless if also present.)


## 3. The fake definition + rationale

    def _fake_int4_gemm_w4a8(A_, A_scale, A_zp, B, B_scale, B_zp,
                             group_size, g_idx, bias):
        return A_.new_empty((A_.shape[0], B.shape[1]), dtype=torch.float16)

Rationale: a register_fake only needs correct output shape / dtype / device for the
FakeTensor trace. `M = A_.shape[0]`, `N = B.shape[1]`, dtype fp16 (hard-coded in
C++). `A_.new_empty(...)` preserves the fake/meta device + default contiguous
layout from A_ while overriding the int8 dtype to float16. The decode path is 2D
and contiguous (M=1 at decode), so the C++ `empty_strided` produces a plain
contiguous `[M,N]` -- `new_empty` matches it. Cross-checked with `codex exec`:
confirmed yes, `new_empty` is the right call and dtype override is correct.

Source/idempotency style mirrors `xpu_int8.py::_register_int8_fakes`: lazy
`import vllm._xpu_ops` to force the `_xpu_C` .so load before registering, a
`hasattr(torch.ops._xpu_C, "int4_gemm_w4a8")` guard (retry on next call if not yet
loaded), a `_FAKES_REGISTERED` idempotency flag, and a try/except so an
already-registered native abstract impl is a no-op.


## 4. The hook point + deploy into the :int8g image

Two deliverables, two deploy options. Pick ONE.

Option C (RECOMMENDED -- unified into xpu_int8.py): fold the int4 fake into
`xpu_int8.py::_register_int8_fakes()`. That module is ALREADY imported on `:int8g`
(its `XPUInt8ScaledMMLinearKernel` is referenced by the int8 scaled_mm dispatch in
`kernels/linear/__init__.py`, and it self-registers at import time in the
engine-core worker that runs capture). One existing import then covers BOTH decode
paths -- no new module, no new hook. The int4 fake is gated on
`hasattr(... "int4_gemm_w4a8")`, so it is a no-op on builds without the op.
  - Patch: `contrib/vllm_int8_xpu/xpu_int4_register_fake.diff` (apply to
    `contrib/vllm_int8_xpu/xpu_int8.py`, the source mirror).
  - Bake: reuse `scripts/52_bake_int8_graph.sh` UNCHANGED -- it already copies
    `xpu_int8.py` to EVERY vllm copy (`find /workspace /opt/venv -path
    "*/kernels/linear/scaled_mm/xpu_int8.py"`). Rebake `:int8g` after editing.

Option A/B (standalone module): `contrib/vllm_int8_xpu/xpu_int4.py` with
`_register_int4_fakes()`. It self-registers at import time and also exposes the
function to call from `XPUW4A8IntLinearKernel.can_implement` (the MPLinearKernel
base has no `is_supported`; call it right after the XPU + op-presence checks pass).
NOTE: the W4A8 kernel lives at `mixed_precision/xpu.py`, a DIFFERENT in-image path
than `scaled_mm/xpu_int8.py`, so deploying the standalone module needs its own copy
step + an import edit -- more moving parts than Option C. Prefer Option C unless you
want the int4 fake decoupled from the int8 module.

In-image paths (verified):
  - W8A8 kernel + fakes: `.../vllm/model_executor/kernels/linear/scaled_mm/xpu_int8.py`
  - W4A8 kernel:         `.../vllm/model_executor/kernels/linear/mixed_precision/xpu.py`
  (multiple vllm installs exist: editable `/workspace/vllm/vllm` AND
   `/opt/venv/lib/python3.12/site-packages/vllm`; the bake script writes to all.)


## 5. Serve command (LEAD runs -- GPU; gate via scripts/gpu-run)

Checkpoint: W4A8-gptq (canonical; NOT the archived RTN dup), served id encodes the
method `...-w4a8-gptq`. Per CLAUDE.md verify-the-model rule, after start:
`curl -s http://192.168.10.5:18080/v1/models | python3 -m json.tool` and confirm
`qwen3-14b-w4a8-gptq`. Serve on `:int8g` with PIECEWISE (and the ulimit/pids fixes
that unblocked W8A8 capture):

    scripts/gpu-run scripts/runremote.sh scripts/36_serve.sh \
      QUANT=/models/Qwen3-14B-W4A8-gptq SERVED=qwen3-14b-w4a8-gptq \
      IMG=vllm-xpu-env:int8g COMPILE=1 MAXLEN=4096 MAXSEQS=4

`COMPILE=1` in `scripts/36_serve.sh` drops `--enforce-eager` and adds
`--compilation-config '{"cudagraph_mode":"PIECEWISE","use_inductor_graph_partition":true,"compile_sizes":[1]}'`.
Also pass `VLLM_XPU_ENABLE_XPU_GRAPH=1` + the capture ulimits as used for W8A8
(`--pids-limit=-1 --ulimit nofile=1048576 --ulimit nproc=63556`,
`OMP_NUM_THREADS=8`) -- if `36_serve.sh` does not yet set the graph env, add it via
`EXTRA=` / the run env, or use the `51_serve_int8_specdecode.sh GRAPH=1 CGMODE=PIECEWISE`
path (no DRAFT) which wires `VLLM_XPU_ENABLE_XPU_GRAPH=1` explicitly.

Use `--dtype float16` for best perf (the op emits fp16; bf16 model dtype incurs the
warned-about `.to()` cast).


## 6. Confirm capture engaged (grep these in `docker logs vllm_*`)

POSITIVE signs (mirror the banked W8A8 run):
  - `[xpu_int8] registered fake for _xpu_C::int4_gemm_w4a8`
    (or `[xpu_int4] registered fake ...` for the standalone module) -- proves the
    int4 fake registered. ABSENCE = the fake never loaded -> capture will abort.
  - dynamo traced THROUGH the custom op + AOT compiled:
    `saved AOT compiled function`
  - real SYCL Graph capture proceeded (PIECEWISE captures linear/MLP only):
    grep -iE 'capturing|CUDAGraph|cudagraph|Graph captur' -- expect the
    "captured N graphs in <t>s (<GiB>)" line (W8A8 was 12 graphs / 4 s / 4.21 GiB).
  - `Application startup complete` + HEALTHY, coherent generations.

NEGATIVE / abort signs:
  - `UnsupportedOperatorException` / `... has no fake impl` naming `int4_gemm_w4a8`
    => fake not registered (wrong image, wrong vllm copy, or hook not wired).
  - `cannot allocate memory for thread-local data: ABORT` during capture
    => PID/thread ceiling; apply the `--pids-limit=-1 --ulimit ...` fix.
  - `sycl_ext_oneapi_work_group_scratch_memory ... not yet available ... SYCL Graph`
    => only under FULL capture (flash-attn). PIECEWISE keeps attention eager and
    avoids it; if seen, you are not in PIECEWISE.

Quick one-liner after HEALTHY:

    docker logs vllm_qwen3 2>&1 | grep -iE \
      'registered fake.*int4|saved AOT compiled|captur|Application startup complete|UnsupportedOperator|fake impl'

Then bench decode and compare to the eager W4A8 baseline (same single-stream
harness used for the W8A8 +16.7%); log the verdict in `JOURNAL.md` / `FINDINGS.md`.
