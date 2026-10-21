#!/bin/bash
# Separate Strata reference process. Does not start or alter hotschmoe-dd.
set -uo pipefail
R=/mnt/vm_8tb/github/b70_ai_things
B=/mnt/vm_8tb/b70/build/vllm-flashnext
S=/mnt/vm_8tb/b70/build/strata-ab
OUT=${Q1_REFERENCE_DIR:-$B/q1}
NAME=vllm-fn-q1-strata-reference
export B70_AGENT=Q1 B70_GPU_LOCK_TIMEOUT=21600
[[ "$OUT" = /* ]] || { echo 'ABORT Q1_REFERENCE_DIR must be absolute'; exit 5; }
for old in strata-references.json strata-window.log strata-reference-config.json strata-reference-engine.log strata-reference.log; do
  if [ -e "$OUT/$old" ]; then
    echo "ABORT preserving existing $OUT/$old; choose a fresh Q1_REFERENCE_DIR"; exit 5
  fi
done
mkdir -p "$OUT"
python3 - "$R/vllm/flashnext/harness/q1_strata_reference.json" "$OUT" <<'PY'
import json, pathlib, sys
cfg = json.loads(pathlib.Path(sys.argv[1]).read_text())
out = pathlib.Path(sys.argv[2])
cfg['log'] = str(out / 'strata-reference-engine.log')
(out / 'strata-reference-config.json').write_text(json.dumps(cfg, indent=2, ensure_ascii=True) + '\n')
PY
exec > >(tee "$OUT/strata-window.log") 2>&1
sha256sum "$OUT/strata-reference-config.json" "$R/strata/flash-next/night/configs/dd-nomtp-c1.json" \
  "$R/vllm/flashnext/harness/x6_strata_reference.py" "$R/vllm/flashnext/harness/q1_validate_references.py" "$B/x6/reference-prompts.json" > "$OUT/strata-inputs-before.sha256"
avail=$(awk '/MemAvailable:/{print $2}' /proc/meminfo)
[ "$avail" -ge 60000000 ] || { echo "ABORT_INITIAL_LOW_RAM_KB:$avail"; exit 4; }
"$R/bin/gpu-run" "$R/bin/xpu-health" > "$OUT/strata-health-before.txt" 2>&1 || exit 3
cleanup() { docker stop -t 15 "$NAME" >/dev/null 2>&1 || true; kill "${WD:-}" 2>/dev/null || true; }
trap cleanup EXIT INT TERM
( while true; do
    avail=$(awk '/MemAvailable:/{print $2}' /proc/meminfo)
    if [ "$avail" -lt 8388608 ]; then echo ABORT_LOW_RAM; docker stop -t 5 "$NAME" >/dev/null 2>&1; exit 1; fi
    sleep 2
  done ) &
WD=$!
"$R/bin/gpu-run" docker run --rm --init --name "$NAME" --network none \
  --memory 90g --device /dev/dri --ulimit memlock=-1 --cap-add SYS_PTRACE \
  -v "$R:$R:ro" -v /mnt/vm_8tb/b70:/mnt/vm_8tb/b70 -v /mnt/cache/b70:/mnt/cache/b70:ro \
  -e ONEAPI_DEVICE_SELECTOR='level_zero:*' -e SYCL_CACHE_PERSISTENT=0 \
  -e ZES_ENABLE_SYSMAN=1 -e STRATA_VERIFY_DEVICE_PLAN=1 -e STRATA_VERIFY_NO_HOST=1 \
  -e STRATA_MIRROR_MIB=0 -e STRATA_STAGE_TRIM=1 -e STRATA_KV_WIDE_HOST=0 -e STRATA_STAGER_THREADS=12 \
  --entrypoint bash sha256:e41d59c0940e1bd2f6f64c9d5d2d7452b77812b0a8407fac8d67cacec5d7fdd2 -lc \
  'source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1; exec timeout --signal=TERM --kill-after=20 8m /mnt/vm_8tb/b70/build/strata-ab/venv/bin/python "$1" --config "$2" --prompts "$3" --out "$4"' \
  bash "$R/vllm/flashnext/harness/x6_strata_reference.py" "$OUT/strata-reference-config.json" "$B/x6/reference-prompts.json" "$OUT/strata-references.json" > "$OUT/strata-reference.log" 2>&1
rc=$?
cleanup
validation=2
if [ "$rc" = 0 ]; then
  python3 "$R/vllm/flashnext/harness/q1_validate_references.py" --reference-dir "$OUT" > "$OUT/strata-reference-validation.log" 2>&1
  validation=$?
fi
"$R/bin/gpu-run" "$R/bin/xpu-health" > "$OUT/strata-health-after.txt" 2>&1
health=$?
sha256sum --check "$OUT/strata-inputs-before.sha256" > "$OUT/strata-inputs-after-check.txt"
source_rc=$?
printf 'STRATA_RC:%s VALIDATION_RC:%s HEALTH_RC:%s SOURCE_RC:%s\n' "$rc" "$validation" "$health" "$source_rc"
[ "$rc" = 0 ] && [ "$validation" = 0 ] && [ "$health" = 0 ] && [ "$source_rc" = 0 ]
