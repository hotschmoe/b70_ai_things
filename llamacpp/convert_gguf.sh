#!/usr/bin/env bash
# llamacpp/convert_gguf.sh -- convert qwen3.6-27b HF safetensors -> GGUF, then quantize to the two serve targets.
#
# Pipeline (all CPU, GPU-free; runs inside the oneAPI image which has python+torch+safetensors+gguf):
#   [1] text bf16 GGUF  -> f16 master GGUF   (MTP head BUNDLED by default; --no-mtp would drop it)
#   [2] vision mmproj   -> mmproj-*.gguf      (qwen3vl tower; passed to llama-server --mmproj)
#   [3] quantize f16    -> Q8_0   ("W8A8-like": 8-bit WEIGHTS only; activations stay fp16 -- NOT true W8A8)
#   [4] quantize f16    -> Q4_K_M ("W4A16-like": 4-bit weights, fp16 compute; B70-validated community default)
#
# Honest mapping (REVIEW_intel_arch.md sec 4): llama.cpp is weight-only quant. Q8_0 ~= W8A16, not W8A8 --
# there is no int8-ACTIVATION path, so B70's INT8 XMX is only exercised for weight dequant (dpct::dp4a),
# unlike our sglang fused W8A8 kernels. On B70, Q8_0 has historically been ~4x slower than Q4_K_M (#21517);
# benchmark both. Q4_K_M is expected to be the production default.
#
# Source weights:  /mnt/vm_8tb/github/b70_ai_things/models/files/qwen3.6-27b/bf16  (vision+MTP intact)
# GGUF outputs:    /mnt/vm_8tb/b70/llamacpp/gguf/  (git-ignored runtime artifacts; *.gguf is gitignored)
set -euo pipefail
IMG="${IMG:-sglang-xpu:mtp}"
SRC="${SRC:-/mnt/vm_8tb/b70/llama.cpp}"
MODELS="${MODELS:-/mnt/vm_8tb/github/b70_ai_things/models/files}"
MODEL="${MODEL:-qwen3.6-27b/bf16}"
OUT="${OUT:-/mnt/vm_8tb/b70/llamacpp/gguf}"
TAG="${TAG:-qwen3.6-27b}"
mkdir -p "$OUT"

# Ensure the CPU-only llama-quantize exists (the SYCL build's quantize aborts without a GPU; see
# build_cpu_tools.sh). This keeps the whole convert+quantize pipeline GPU-free.
if [ ! -x "$SRC/build-cpu/bin/llama-quantize" ]; then
  echo "=== CPU-only llama-quantize missing -> building it ==="
  IMG="$IMG" SRC="$SRC" bash "$(dirname "${BASH_SOURCE[0]}")/build_cpu_tools.sh"
fi

echo "=== GGUF convert+quantize  model=$MODEL  out=$OUT  $(date) ==="
docker run --rm --entrypoint bash -v "$SRC:/llama" -v "$MODELS:/models:ro" -v "$OUT:/out" "$IMG" -lc "
  set -e
  cd /llama
  F16=/out/${TAG}-f16.gguf
  if [ ! -s \"\$F16\" ]; then
    echo '--- [1/4] text -> f16 GGUF (MTP bundled) ---'
    python convert_hf_to_gguf.py /models/$MODEL --outtype f16 --outfile \"\$F16\"
  else echo '--- [1/4] f16 exists, skip ---'; fi
  MM=/out/${TAG}-mmproj-f16.gguf
  if [ ! -s \"\$MM\" ]; then
    echo '--- [2/4] vision mmproj ---'
    python convert_hf_to_gguf.py /models/$MODEL --mmproj --outfile \"\$MM\" || echo '[warn] mmproj convert failed (vision optional)'
  else echo '--- [2/4] mmproj exists, skip ---'; fi
  echo '--- [3/4] quantize -> Q8_0 (W8A8-like) [CPU-only binary, GPU-free] ---'
  [ -s /out/${TAG}-Q8_0.gguf ]   || /llama/build-cpu/bin/llama-quantize \"\$F16\" /out/${TAG}-Q8_0.gguf   Q8_0
  echo '--- [4/4] quantize -> Q4_K_M (W4A16-like) [CPU-only binary, GPU-free] ---'
  [ -s /out/${TAG}-Q4_K_M.gguf ] || /llama/build-cpu/bin/llama-quantize \"\$F16\" /out/${TAG}-Q4_K_M.gguf Q4_K_M
  echo '=== artifacts ==='; ls -lah /out/
"
echo "=== done $(date) ==="
