#!/usr/bin/env bash
# END-TO-END fast-path-activation proof for our int8_gemm_w8a8 kernel:
#  1) graft our int8-enabled _xpu_C.so over the image's installed vllm_xpu_kernels (keeps flash-attn etc),
#  2) drop in XPUInt8ScaledMMLinearKernel,
#  3) patch vllm/model_executor/kernels/linear/__init__.py (register XPU in _POSSIBLE_INT8_KERNELS +
#     harden the chooser with .get()),
#  4) serve our Qwen3-14B-W8A8-INT checkpoint (the one that currently KeyError-crashes) and show the
#     kernel SELECTED instead of crashing.
# All in a NAMED detached container; greps the log for the verdict. GPU must be free.
set -uo pipefail
ROOT=/mnt/vm_8tb/b70
D="$ROOT/contrib_int8"; mkdir -p "$D"
SOPATH="$ROOT/vllm-xpu-kernels/vllm_xpu_kernels/_xpu_C.abi3.so"
MODEL="$ROOT/models/Qwen3-14B-W8A8-gptq"
NAME=vllm_int8; PORT=18080
IMG=vllm-xpu-env:v0230

# ---- the kernel class (placed at scaled_mm/xpu_int8.py inside vllm) ----
cat > "$D/xpu_int8.py" <<'PYEOF_CLASS'
import torch
from torch.nn import Parameter
from vllm.model_executor.layers.quantization.utils import replace_parameter
from vllm.model_executor.layers.quantization.utils.w8a8_utils import convert_to_channelwise
from vllm.platforms import current_platform
from .ScaledMMLinearKernel import Int8ScaledMMLinearKernel, Int8ScaledMMLinearLayerConfig


class XPUInt8ScaledMMLinearKernel(Int8ScaledMMLinearKernel):
    """INT8 W8A8 dynamic-symmetric scaled-MM via oneDNN s8s8s32 on XPU (Battlemage)."""

    @classmethod
    def is_supported(cls, compute_capability=None):
        if not current_platform.is_xpu():
            return False, "XPUInt8ScaledMM is only supported on XPU."
        if not hasattr(torch.ops._xpu_C, "int8_gemm_w8a8"):
            return False, "int8_gemm_w8a8 op not present in the installed vllm-xpu-kernels wheel."
        return True, None

    @classmethod
    def can_implement(cls, c: Int8ScaledMMLinearLayerConfig):
        if c.is_static_input_scheme:
            return False, "XPU int8 kernel supports dynamic activation quantization only."
        if not c.input_symmetric:
            return False, "XPU int8 kernel supports symmetric activations only."
        return True, None

    def process_weights_after_loading(self, layer):
        w_q_name, w_s_name, _, _, _ = self.layer_param_names
        weight = getattr(layer, w_q_name)
        replace_parameter(layer, w_q_name,
                          Parameter(weight.t().contiguous().data, requires_grad=False))
        weight_scale = getattr(layer, w_s_name)
        is_fused_module = len(layer.logical_widths) > 1
        if is_fused_module and not self.config.is_channelwise:
            weight_scale = convert_to_channelwise(weight_scale, layer.logical_widths)
        replace_parameter(layer, w_s_name,
                          Parameter(weight_scale.reshape(1, -1).contiguous().data, requires_grad=False))

    def apply_weights(self, layer, x, bias=None):
        from vllm._xpu_ops import xpu_ops as ops
        w_q, w_s, _i_s, _i_zp, _azp_adj = self._get_layer_params(layer)
        x_2d = x.reshape(-1, x.shape[-1])
        if hasattr(torch.ops._xpu_C, "dynamic_per_token_int8_quant"):
            x_q, x_s, _x_zp = torch.ops._xpu_C.dynamic_per_token_int8_quant(x_2d, True, 8)
        else:
            x_q, x_s, _x_zp = ops.dynamic_per_token_int8_quant_ref(x_2d, True, 8)
        out = torch.ops._xpu_C.int8_gemm_w8a8(x_q, x_s, None, w_q, w_s, None, bias, x.dtype)
        return out.reshape(x.shape[:-1] + (out.size(-1),))
PYEOF_CLASS

# ---- the registry patcher (asserts each edit; fails loud) ----
cat > "$D/apply_patches.py" <<'PYEOF_PATCH'
import sys, os, shutil, vllm
BASE = os.path.dirname(vllm.__file__)   # the ACTUAL imported vllm (resolves editable /workspace too)
# copy the kernel class into the real scaled_mm package (reliable BASE, no shell path capture)
CLS_DST = os.path.join(BASE, "model_executor/kernels/linear/scaled_mm/xpu_int8.py")
shutil.copyfile("/mnt/vm_8tb/b70/contrib_int8/xpu_int8.py", CLS_DST)
print("copied class ->", CLS_DST)
V = os.path.join(BASE, "model_executor/kernels/linear/__init__.py")
print("patching:", V)
src = open(V).read()

# 1) import the kernel class right after the scaled_mm.xpu import block
anchor = ("from vllm.model_executor.kernels.linear.scaled_mm.xpu import (\n"
          "    XPUFp8BlockScaledMMKernel,\n"
          "    XPUFP8ScaledMMLinearKernel,\n)")
if anchor not in src:
    sys.exit("FAIL: scaled_mm.xpu import anchor not found")
add = ("\nfrom vllm.model_executor.kernels.linear.scaled_mm.xpu_int8 import (\n"
       "    XPUInt8ScaledMMLinearKernel,\n)")
src = src.replace(anchor, anchor + add, 1)

# 2) register XPU in _POSSIBLE_INT8_KERNELS (insert before the dict's closing brace)
key = "_POSSIBLE_INT8_KERNELS"
i = src.find(key)
if i < 0:
    sys.exit("FAIL: _POSSIBLE_INT8_KERNELS not found")
j = src.find("\n}", i)
if j < 0:
    sys.exit("FAIL: end of _POSSIBLE_INT8_KERNELS dict not found")
if "PlatformEnum.XPU" in src[i:j]:
    sys.exit("FAIL: XPU already present in INT8 registry")
src = src[:j] + "\n    PlatformEnum.XPU: [XPUInt8ScaledMMLinearKernel]," + src[j:]

# 3) harden the chooser: possible_kernels[...] -> .get(...) with a clear error
sub = "    platform_kernels = possible_kernels[current_platform._enum]"
if sub not in src:
    sys.exit("FAIL: chooser subscript not found")
repl = ("    platform_kernels = possible_kernels.get(current_platform._enum)\n"
        "    if not platform_kernels:\n"
        "        raise ValueError(\n"
        "            \"No ScaledMM linear kernels registered for platform \"\n"
        "            f\"{current_platform._enum}.\")")
src = src.replace(sub, repl, 1)

open(V, "w").write(src)
print("PATCHED linear/__init__.py: import + XPU registry + .get() hardening OK")
PYEOF_PATCH

echo "=== free GPU + remove old server ==="
docker rm -f "$NAME" vllm_qwen3 vllm_w4a8 vllm_w8a8 2>/dev/null || true
[ -f "$SOPATH" ] || { echo "MISSING our _xpu_C.so at $SOPATH"; exit 1; }
[ -d "$MODEL" ] || { echo "MISSING W8A8 model at $MODEL"; exit 1; }

echo "=== launch patched server (graft .so + class + registry patch, then serve) ==="
docker run -d --name "$NAME" --device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path \
  --ipc=host --shm-size 16g -p ${PORT}:${PORT} \
  -v "$ROOT:$ROOT" -v "$ROOT/hf_cache:/hf_cache" -v "$ROOT/vllm_cache:/vllm_cache" \
  -e HF_HOME=/hf_cache -e VLLM_CACHE_ROOT=/vllm_cache -e XDG_CACHE_HOME=/vllm_cache \
  -e ZE_AFFINITY_MASK=0 -e VLLM_LOGGING_LEVEL=DEBUG \
  --entrypoint bash "$IMG" -c '
    set -e
    echo "[graft] our int8 _xpu_C.so over the installed package"
    cp -f '"$SOPATH"' /opt/venv/lib/python3.12/site-packages/vllm_xpu_kernels/_xpu_C.abi3.so
    echo "[patch] copy class + register XPU + harden chooser"
    python '"$D"'/apply_patches.py
    echo "[sanity] int8 op present?"; python -c "import torch,vllm._xpu_ops; print(\"int8_gemm_w8a8:\", hasattr(torch.ops._xpu_C,\"int8_gemm_w8a8\"))"
    echo "[serve]"
    exec vllm serve '"$MODEL"' --served-model-name qwen3-14b-w8a8-gptq --host 0.0.0.0 --port '"$PORT"' \
      --dtype float16 --tensor-parallel-size 1 --enforce-eager --max-model-len 8192 \
      --gpu-memory-utilization 0.90 --no-enable-prefix-caching --trust-remote-code
  '

echo "=== wait for readiness (up to ~12 min) ==="
ok=0
for i in $(seq 1 144); do
  curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1 && { ok=1; break; }
  docker inspect -f '{{.State.Status}}' "$NAME" 2>/dev/null | grep -q exited && { echo "EXITED EARLY"; break; }
  sleep 5
done

echo; echo "===== VERDICT: kernel selection / scheme / crash ====="
docker logs "$NAME" 2>&1 | grep -iE "int8_gemm_w8a8:|PATCHED|XPUInt8ScaledMMLinearKernel|CompressedTensorsW8A8Int8|Using scheme|KeyError|PlatformEnum.XPU|No ScaledMM|Failed to find|Application startup complete|error|Traceback" | grep -viE "OperatorEntry|registered|VLLM_" | tail -30
echo
[ "$ok" = 1 ] && echo "HEALTHY :$PORT -- W8A8 INT8 SERVES VIA OUR KERNEL" || { echo "NOT HEALTHY -- last 30 lines:"; docker logs "$NAME" 2>&1 | tail -30; }
