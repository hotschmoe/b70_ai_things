#!/usr/bin/env bash
# xpu-health -- detect a wedged B70 / xe-driver state BEFORE or AFTER a TP>1 serve.
#
# Why: a TP>1 worker torn down mid-collective (graph capture / oneCCL warmup) corrupts the
# Level-Zero / oneCCL device context so that EVERY later GPU op fails -- DEVICE_LOST (err 20),
# OUT_OF_RESOURCES (err 40), or a multi-minute HANG -- on BOTH cards, and (per P2P_GPU.md J.16)
# NOT only on TP>1: even a single-card matmul can hang/OOM. This probe runs a tiny per-card matmul
# (TP=1, one card visible) inside the serve image, TIMEOUT-wrapped so a hung card is itself caught
# (J.16's own sanity probe hung >13 min). It is a DETECTOR, not a fixer -- recovery is bin/xe-reset.
#
# Usage:
#   xpu-health                 # probe BOTH cards (default)
#   xpu-health --card 0        # probe only card 0
#   xpu-health --img TAG       # image to run the probe in (default $IMG or vllm-xpu-env:int8g-v0251)
#   xpu-health --timeout SECS  # per-card probe wall-clock cap (default 60)
#
# Exit codes (consumed by lib.sh guard + xe-reset):
#   0  HEALTHY     -- every probed card did a matmul+synchronize and returned in time
#   1  WEDGED      -- a card hung (timeout) or threw DEVICE_LOST/OUT_OF_RESOURCES -> needs xe-reset
#   2  INCONCLUSIVE-- the probe could not run (no docker / image / device) -> warn, do not block
set -uo pipefail

IMG="${IMG:-vllm-xpu-env:int8g-v0251}"
TIMEOUT="${XPU_HEALTH_TIMEOUT:-60}"
CARDS=(0 1)

while [ $# -gt 0 ]; do
  case "$1" in
    --card)    CARDS=("$2"); shift 2 ;;
    --card=*)  CARDS=("${1#--card=}"); shift ;;
    --img)     IMG="$2"; shift 2 ;;
    --img=*)   IMG="${1#--img=}"; shift ;;
    --timeout) TIMEOUT="$2"; shift 2 ;;
    -h|--help) awk 'NR>1 && /^#/ {sub(/^# ?/,""); print; next} NR>1 {exit}' "$0"; exit 0 ;;
    *) echo "xpu-health: unknown arg '$1'" >&2; exit 2 ;;
  esac
done

command -v docker >/dev/null 2>&1 || { echo "xpu-health: docker not found -> INCONCLUSIVE" >&2; exit 2; }

# Minimal on-card probe: a real matmul (catches OUT_OF_RESOURCES/DEVICE_LOST) + a tiny op
# (the cheapest thing to hang). ZE_AFFINITY_MASK pins ONE card, so device_count==1 / device xpu:0.
read -r -d '' PROBE <<'PY'
import sys
try:
    import torch
    if torch.xpu.device_count() < 1:
        print("HEALTH_FAIL no-xpu-device"); sys.exit(1)
    x = torch.randn(2048, 2048, device="xpu:0")
    y = x.matmul(x)
    torch.xpu.synchronize()
    z = torch.randn(16, 16, device="xpu:0").matmul(torch.randn(16, 16, device="xpu:0"))
    torch.xpu.synchronize()
    if not bool(torch.isfinite(y).all().item()):
        print("HEALTH_FAIL nonfinite-large-matmul"); sys.exit(1)
    if not bool(torch.isfinite(z).all().item()):
        print("HEALTH_FAIL nonfinite-small-matmul"); sys.exit(1)
    print("HEALTH_OK True")
    sys.exit(0)
except Exception as e:
    print("HEALTH_FAIL", type(e).__name__, str(e)[:200])
    sys.exit(1)
PY

# Every probe has an owned name. Killing a docker client does not prove its
# container stopped, so verify removal before returning control to a resetter.
owner="xpu-health-$$-${RANDOM}-${RANDOM}"
active_name=""
cleanup_probe() {
  [ -n "$active_name" ] || return 0
  local ids rc
  while true; do
    ids=$(timeout --signal=KILL 20 docker ps -aq \
      --filter "name=^/${active_name}$" --filter "label=b70.xpu-health=$owner" 2>&1)
    rc=$?
    if [ "$rc" = 0 ] && [ -z "$ids" ]; then active_name=""; return 0; fi
    if [ "$rc" = 0 ]; then
      # IDs came from our unique name AND ownership label.
      timeout --signal=KILL 30 docker rm -f $ids >/dev/null 2>&1 || true
    fi
    echo "xpu-health: cleanup unverified for $active_name; retaining caller lease, no reset yet" >&2
    sleep 2
  done
}
finish() {
  local status=$?
  trap '' INT TERM HUP
  cleanup_probe
  return "$status"
}
trap finish EXIT
trap 'exit 1' INT TERM HUP

overall=0
for c in "${CARDS[@]}"; do
  [ "$c" = 0 ] || [ "$c" = 1 ] || { echo "xpu-health: bad card '$c'" >&2; exit 2; }
  echo "=== probe card $c (img=$IMG timeout=${TIMEOUT}s) ===" >&2
  active_name="$owner-card$c"
  out=$(timeout --signal=TERM --kill-after=10 "$TIMEOUT" docker run --rm \
        --name "$active_name" --label "b70.xpu-health=$owner" --device /dev/dri \
        -v /dev/dri/by-path:/dev/dri/by-path -e ZE_AFFINITY_MASK="$c" \
        --entrypoint python3 "$IMG" -c "$PROBE" 2>&1)
  rc=$?
  cleanup_probe
  if [ "$rc" = 124 ] || [ "$rc" = 137 ]; then
    echo "  card $c: HUNG (>${TIMEOUT}s, killed) -> WEDGED" >&2; overall=1; continue
  fi
  if [ "$rc" = 0 ] && printf '%s\n' "$out" | grep -qx 'HEALTH_OK True' && ! printf '%s\n' "$out" | grep -Eq 'HEALTH_FAIL|HEALTH_OK False'; then
    echo "  card $c: OK" >&2; continue
  fi
  if printf '%s\n' "$out" | grep -Eq 'HEALTH_FAIL|HEALTH_OK'; then
    echo "  card $c: GPU OP FAILED -> WEDGED" >&2
    printf '%s\n' "$out" | grep -E 'HEALTH_FAIL|HEALTH_OK|DEVICE_LOST|OUT_OF_RESOURCES|RuntimeError' | head -3 | sed 's/^/    /' >&2
    overall=1; continue
  fi
  # Neither sentinel and not a timeout: container/image/daemon problem, not a GPU verdict.
  echo "  card $c: probe could not run (no HEALTH_OK/FAIL marker, rc=$rc) -> INCONCLUSIVE" >&2
  printf '%s\n' "$out" | tail -3 | sed 's/^/    /' >&2
  [ "$overall" = 0 ] && overall=2
done

case "$overall" in
  0) echo "xpu-health: HEALTHY (cards ${CARDS[*]})" >&2 ;;
  1) echo "xpu-health: WEDGED -- recover with: bin/xe-reset" >&2 ;;
  2) echo "xpu-health: INCONCLUSIVE (probe did not run; not blocking)" >&2 ;;
esac
exit "$overall"
