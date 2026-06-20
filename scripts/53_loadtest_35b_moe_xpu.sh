#!/usr/bin/env bash
# Load-test Qwen3.6-35B-A3B int4 (256-expert MoE) on ONE B70 with the INC XPU-MoE patch.
#
# Root cause it tests: stock INC (vllm 0.23.0+xpu, inc.py) routes ALL XPU layers through
# apply_xpu_w4a16_quant_layer, which only handles LinearBase/ParallelLMHead and returns
# None for RoutedExperts -> FusedMoE falls back to UnquantizedFusedMoEMethod (bf16) and the
# 256 experts dequantize toward ~70 GB -> OOM at load. Our patch adds the missing branch:
# RoutedExperts -> MoeWNA16Config (gptq), keeping experts int4 and running the pure-Triton
# wna16 MoE kernel (should_moe_wna16_use_cuda() is False on XPU -> no CUDA-only op).
#
# Goal: prove it LOADS + emits a coherent token (NOT a full eval). Self-terminating offline
# generate so the gpu-run lease is held only for the test. Runs ON THE HOST.
#   ship + run:  scripts/runremote.sh scripts/53_loadtest_35b_moe_xpu.sh
set -uo pipefail
ROOT=/mnt/vm_8tb/b70
IMG="${IMG:-vllm-xpu-env:v0230}"
MODEL="${MODEL:-/models/Intel_Qwen3.6-35B-A3B-int4-AutoRound}"
PATCH="${PATCH:-$ROOT/patches/inc_xpu_moe.py}"
MAXLEN="${MAXLEN:-2048}"; UTIL="${UTIL:-0.95}"; MAXBATCH="${MAXBATCH:-2048}"
INC_DST=/opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/quantization/inc.py

mkdir -p "$ROOT/patches" "$ROOT/vllm_cache" "$ROOT/tmp_ssd" "$ROOT/hf_cache"
[ -f "$PATCH" ] || { echo "MISSING PATCH: $PATCH (copy contrib/vllm_moe_xpu/inc.py here first)"; exit 1; }

# offline load + 1 short greedy generation; prints VRAM + the routed MoE method actually chosen.
cat > "$ROOT/patches/loadtest_35b.py" <<'PY'
import os, time
from vllm import LLM, SamplingParams

def main():
    m = os.environ.get("MODEL")
    t0 = time.time()
    llm = LLM(
        model=m, trust_remote_code=True,
        max_model_len=int(os.environ.get("MAXLEN", "2048")),
        max_num_seqs=1,
        max_num_batched_tokens=int(os.environ.get("MAXBATCH", "2048")),
        gpu_memory_utilization=float(os.environ.get("UTIL", "0.95")),
        enforce_eager=True,
    )
    print(f"=== LLM constructed in {time.time()-t0:.0f}s ===", flush=True)
    o = llm.generate(["The capital of France is"], SamplingParams(temperature=0, max_tokens=24))
    print("=== GENERATION OK ===", flush=True)
    print("OUTPUT:", repr(o[0].outputs[0].text), flush=True)

if __name__ == "__main__":
    main()
PY

echo "=== load-test: IMG=$IMG MODEL=$MODEL MAXLEN=$MAXLEN UTIL=$UTIL ==="
echo "=== patch: $PATCH -> $INC_DST ==="
exec "$ROOT/gpu-run" docker run --rm --name vllm_35b_loadtest \
  --device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path \
  --ipc=host --shm-size 16g \
  -v "$ROOT/models:/models:ro" -v "$ROOT/patches:/patches:ro" \
  -v "$ROOT/hf_cache:/hf_cache" -v "$ROOT/vllm_cache:/vllm_cache" -v "$ROOT/tmp_ssd:/tmp_ssd" \
  -v "$PATCH:$INC_DST:ro" \
  -e HF_HOME=/hf_cache -e VLLM_CACHE_ROOT=/vllm_cache -e XDG_CACHE_HOME=/vllm_cache \
  -e TRITON_CACHE_DIR=/vllm_cache/triton -e TMPDIR=/tmp_ssd -e ZE_AFFINITY_MASK=0 \
  -e MODEL="$MODEL" -e MAXLEN="$MAXLEN" -e UTIL="$UTIL" -e MAXBATCH="$MAXBATCH" \
  -e VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}" \
  --entrypoint python "$IMG" /patches/loadtest_35b.py
