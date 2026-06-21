#!/usr/bin/env bash
# AutoRound W8A8 (int8 weight + dynamic int8 act) -> compressed-tensors, for the int8 W8A8 oneDNN kernel.
# Generalizes the docs/kernel/15 sec-2 inline recipe into a reusable driver. QUANTS_TODO Q1/Q2/Q4/Q6.
#
# GPU job -> ALWAYS route via gpu-run (flock lease):   ./gpu-run bash 65_autoround_w8a8.sh
# Detached long run (host):  cd /mnt/vm_8tb/b70 && setsid bash -c 'SRC=.. OUT=.. ./gpu-run bash 65_autoround_w8a8.sh' &
#
# Env:
#   SRC        bf16 source dir (required)
#   OUT        output dir (required; MUST be method-tagged ...-W8A8-autoround per CLAUDE.md)
#   DEVMAP     xpu (1 card, 14B) | 0,1 (both, 27B/Qwable/35B) | auto       (default xpu)
#   ITERS      AutoRound rounding iters (smoke 50, full 200)                (default 200)
#   NSAMPLES   calib samples (smoke 16-64, full 128)                        (default 128)
#   SEQLEN     calib seqlen (smoke 512, full 2048)                          (default 2048)
#   GROUP      weight group_size (-1 = per-channel, canonical W8A8)         (default -1)
#   LOWMEM     1 = keep model on CPU, stream blocks (avoids >32GB OOM)      (default 1)
#   IGN_REGEX  modules to keep bf16 (default qwen3_5 VLM+MTP+DeltaNet)
#   IGN_MOE    1 = also keep MoE router/gate bf16 (set for the 35B)         (default 0)
#   IMG        docker image                                                 (default vllm-xpu-env:v0230)
set -uo pipefail
ROOT=/mnt/vm_8tb/b70
IMG="${IMG:-vllm-xpu-env:v0230}"
SRC="${SRC:?set SRC=<bf16 source dir>}"
OUT="${OUT:?set OUT=<output dir, method-tagged>}"
DEVMAP="${DEVMAP:-xpu}"
ITERS="${ITERS:-200}"; NSAMPLES="${NSAMPLES:-128}"; SEQLEN="${SEQLEN:-2048}"
GROUP="${GROUP:--1}"; LOWMEM="${LOWMEM:-1}"
IGN_REGEX="${IGN_REGEX:-(visual|\.mtp|mtp\.|linear_attn)}"
IGN_MOE="${IGN_MOE:-0}"
TAG="$(basename "$OUT")"
LOG="$ROOT/results/autoround_w8a8_${TAG}_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$ROOT/results" "$ROOT/pip_cache" "$ROOT/tmp_ssd"
[ -d "$SRC" ] || { echo "MISSING SRC $SRC"; exit 1; }
[ -f "$ROOT/_autoround_w8a8.py" ] || { echo "MISSING $ROOT/_autoround_w8a8.py (scp it to the host root)"; exit 1; }

# device_map=xpu -> pin card 0; 0,1/auto -> expose both, no pin.
if [ "$DEVMAP" = xpu ]; then AFF=0; else AFF="${DEVMAP/auto/0,1}"; fi
echo "=== AutoRound W8A8 :: SRC=$SRC OUT=$OUT DEVMAP=$DEVMAP AFF=$AFF iters=$ITERS nsamples=$NSAMPLES seqlen=$SEQLEN ==="
echo "=== log=$LOG ==="

# SRC may live OUTSIDE $ROOT (e.g. the 14B under /mnt/vm_8tb/specula-build) -- mount it read-only so
# the container can see it; HF otherwise treats an invisible local path as a repo id and HFValidationErrors.
SRCMOUNT=()
case "$SRC" in
  "$ROOT"/*) : ;;                          # already covered by the $ROOT bind mount
  *) SRCMOUNT+=(-v "$SRC:$SRC:ro") ;;
esac

docker rm -f arw8a8 2>/dev/null || true
docker run --rm --name arw8a8 --device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path \
  --ipc=host --shm-size 32g \
  -e ZE_AFFINITY_MASK="$AFF" \
  -e CCL_ENABLE_SYCL_KERNELS=0 -e SYCL_UR_USE_LEVEL_ZERO_V2=0 -e CCL_ATL_TRANSPORT=ofi \
  ${SRCMOUNT[@]+"${SRCMOUNT[@]}"} \
  -v "$ROOT:$ROOT" -e HF_HOME="$ROOT/hf_cache" -e TMPDIR="$ROOT/tmp_ssd" \
  -e XDG_CACHE_HOME="$ROOT/vllm_cache" -e PIP_CACHE_DIR="$ROOT/pip_cache" -e OMP_NUM_THREADS=32 \
  -e SRC="$SRC" -e OUT="$OUT" -e DEVMAP="$DEVMAP" -e ITERS="$ITERS" -e NSAMPLES="$NSAMPLES" \
  -e SEQLEN="$SEQLEN" -e GROUP="$GROUP" -e LOWMEM="$LOWMEM" -e IGN_REGEX="$IGN_REGEX" -e IGN_MOE="$IGN_MOE" \
  -v "$ROOT/_autoround_w8a8.py:/_autoround_w8a8.py:ro" \
  --entrypoint bash "$IMG" -lc '
    source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1 || true
    echo "torch: $(python -c "import torch;print(torch.__version__)")"
    pip install -q --no-deps auto-round 2>&1 | tail -2 || echo "(auto-round no-deps install issue)"
    pip install -q "transformers>=4.52" accelerate datasets py-cpuinfo threadpoolctl 2>&1 | tail -2 || true
    python -c "import torch;assert torch.xpu.is_available(),\"XPU LOST after pip\";print(\"xpu OK after pip, count\",torch.xpu.device_count())"
    python /_autoround_w8a8.py
  ' 2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
echo "=== exit $rc ==="; du -sh "$OUT" 2>/dev/null
exit $rc
