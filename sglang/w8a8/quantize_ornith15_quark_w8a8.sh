#!/usr/bin/env bash
# XPU-accelerated Ornith-1.5 W8A8 RTN converter.
# Run only through: ./bin/gpu-run --card 0 bash sglang/w8a8/quantize_ornith15_quark_w8a8.sh
set -euo pipefail

REPO="${REPO:-/mnt/vm_8tb/github/b70_ai_things}"
ROOT="${ROOT:-/mnt/vm_8tb/b70}"
IMG="${IMG:-sglang-xpu:mtp}"
SRC="${SRC:-$REPO/models/files/ornith-1.5-35b-a3b/bf16-mtp-shisa}"
OUT="${OUT:-$REPO/models/files/ornith-1.5-35b-a3b/w8a8-rtn-mtp-shisa}"
ROW_CHUNK="${ROW_CHUNK:-8192}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG="${LOG:-$REPO/results/logs/ornith15_w8a8_rtn_quant_${STAMP}.log}"
PY="$REPO/sglang/w8a8/quantize_ornith15_quark_w8a8.py"

[ -f "$SRC/model.safetensors.index.json" ] || { echo "missing source index: $SRC" >&2; exit 1; }
[ -f "$PY" ] || { echo "missing converter: $PY" >&2; exit 1; }
[ ! -e "$OUT" ] || { echo "refusing to overwrite output: $OUT" >&2; exit 1; }
mkdir -p "$(dirname "$LOG")" "$(dirname "$OUT")"

echo "config -> source=$SRC output=$OUT image=$IMG device=xpu:0 row_chunk=$ROW_CHUNK"
"$REPO/bin/xpu-health" --card 0
set +e
docker run --rm --name ornith15-w8a8-quant \
  --device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path \
  --ipc=host --shm-size 32g \
  -e ZE_AFFINITY_MASK=0 -e OMP_NUM_THREADS=32 \
  -v "$REPO:$REPO" \
  --entrypoint bash "$IMG" -lc "
    source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1 || true
    python '$PY' --source '$SRC' --output '$OUT' --device xpu --row-chunk '$ROW_CHUNK'
  " 2>&1 | tee "$LOG"
rc="${PIPESTATUS[0]}"
set -e
"$REPO/bin/xpu-health" --card 0
echo "result -> rc=$rc log=$LOG"
[ "$rc" = 0 ] && du -sh "$OUT"
exit "$rc"
