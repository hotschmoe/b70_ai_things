#!/usr/bin/env bash
# WIN 1 (packing) probe -- CPU only, ~3 min, does NOT touch the GPU.
# Re-export a data-free RTN W4A8-INT 14B but force compressed-tensors to PACK the int4 weights
# (pack-quantized / int32) instead of the default unpacked int-quantized (int8) layout that
# scripts/43 produced (16 GB). Goal: prove we can get ~9 GB on disk. Whether the B70's
# XPUW4A8IntLinearKernel will LOAD the packed layout is the GPU half -- test by serving after.
#
# Accuracy is irrelevant here (RTN) -- this only answers "can we pack the W4A8 weights".
#   Env: OUTNAME (default Qwen3-14B-W4A8-PACKtest), GROUP (128), FORMAT (pack-quantized).
set -uo pipefail
ROOT=/mnt/vm_8tb/b70
SPECULA=/mnt/vm_8tb/specula-build/models
SRC="${SRC:-/specula_models/Qwen3-14B}"
OUTNAME="${OUTNAME:-Qwen3-14B-W4A8-PACKtest}"; GROUP="${GROUP:-128}"; FORMAT="${FORMAT:-pack-quantized}"
LOG="$ROOT/results/w4a8_packtest_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$ROOT/results" "$ROOT/models" "$ROOT/pip_cache"
echo "=== W4A8 PACK test: src=$SRC out=$OUTNAME group=$GROUP format=$FORMAT log=$LOG ==="

docker run --rm --name w4a8_packtest \
  -v "$SPECULA:/specula_models:ro" -v "$ROOT/models:/models" \
  -v "$ROOT/hf_cache:/hf_cache" -v "$ROOT/pip_cache:/root/.cache/pip" \
  -e HF_HOME=/hf_cache -e OMP_NUM_THREADS=32 \
  -e CUDA_VISIBLE_DEVICES="" -e ZE_AFFINITY_MASK="" \
  -e SRC="$SRC" -e OUTNAME="$OUTNAME" -e GROUP="$GROUP" -e FORMAT="$FORMAT" \
  python:3.11 bash -c '
    set -e
    pip install -q torch --index-url https://download.pytorch.org/whl/cpu
    pip install -q "llmcompressor>=0.8.0" "compressed-tensors" "transformers>=4.52" accelerate
    python - <<PY
import os, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from compressed_tensors.quantization import QuantizationScheme, QuantizationArgs
try:
    from compressed_tensors.config import CompressionFormat
    print("[diag] available CompressionFormats:", [f.value for f in CompressionFormat], flush=True)
except Exception as e:
    print("[diag] could not list CompressionFormat:", e, flush=True)

SRC=os.environ["SRC"]; OUT="/models/"+os.environ["OUTNAME"]
G=int(os.environ["GROUP"]); FMT=os.environ["FORMAT"]
print(f"[load] {SRC} (bf16, cpu)...", flush=True)
model=AutoModelForCausalLM.from_pretrained(SRC, torch_dtype=torch.bfloat16,
        device_map="cpu", low_cpu_mem_usage=True)
tok=AutoTokenizer.from_pretrained(SRC)

scheme=QuantizationScheme(
    targets=["Linear"],
    weights=QuantizationArgs(num_bits=4, type="int", symmetric=True,
                             strategy="group", group_size=G),
    input_activations=QuantizationArgs(num_bits=8, type="int", symmetric=True,
                                       dynamic=True, strategy="token"),
)
recipe=[QuantizationModifier(config_groups={"group_0": scheme}, ignore=["lm_head"])]
print(f"[quant] data-free RTN W4A8 (int4 sym g{G} + dyn int8 act)...", flush=True)
oneshot(model=model, recipe=recipe)

# Try to FORCE the packed format. The kwarg name varies across llmcompressor versions, so try
# a few; report which one took. If none accept it, fall back to default and report that too.
saved=False
for kw in ("quantization_format", "compression_format"):
    try:
        print(f"[save] trying {kw}={FMT} ...", flush=True)
        model.save_pretrained(OUT, save_compressed=True, **{kw: FMT})
        print(f"[save] OK via {kw}={FMT}", flush=True); saved=True; break
    except TypeError as e:
        print(f"[save] {kw} not accepted: {e}", flush=True)
    except Exception as e:
        print(f"[save] {kw} raised: {e}", flush=True)
if not saved:
    print("[save] falling back to default save (no explicit format)", flush=True)
    model.save_pretrained(OUT, save_compressed=True)
tok.save_pretrained(OUT)
print("DONE_PACKTEST", OUT, flush=True)
PY
  ' 2>&1 | tee "$LOG"
echo "=== exit: ${PIPESTATUS[0]} ==="
echo "--- size (want ~9 GB, not 16 GB) ---"; du -sh "$ROOT/models/$OUTNAME" 2>/dev/null
echo "--- config format + a weight tensor dtype (want I32/packed, not I8) ---"
grep -ao '"format":"[a-z-]*"' "$ROOT/models/$OUTNAME/config.json" 2>/dev/null | head -3
head -c 3000000 "$ROOT/models/$OUTNAME"/*.safetensors 2>/dev/null | grep -ao '"dtype":"[A-Z0-9_]*"' | sort | uniq -c
