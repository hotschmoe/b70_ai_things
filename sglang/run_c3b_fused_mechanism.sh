#!/usr/bin/env bash
# Real-serving mechanism gate for fused delayed MLP AR + Gemma RMSNorm.
# Run under ./bin/gpu-run. The endpoint remains down after the gate.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SHELF="$REPO/rdy_to_serve/sglang/qwen36-27b-w8a8/serve.sh"
STAMP="${STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUT="${OUT:-$REPO/results/logs/c3b_fused_mechanism_$STAMP}"
NAME="${NAME:-c3b_fused_mechanism}"
PORT="${PORT:-31004}"
CTX="${CTX:-4096}"
SERVED="${SERVED:-qwen36-27b-w8a8-gptq-mtp-c3b-fused}"
PUSHDIR="${PUSHDIR:-/mnt/vm_8tb/b70/fused_ar_rmsnorm}"
PUSH_AR_SO="${PUSH_AR_SO:-/work/push_ar/libxpu_push_ar_fused_rmsnorm.so}"
mkdir -p "$OUT"

cleanup() {
  local rc=$?
  trap - EXIT INT TERM
  docker logs "$NAME" >"$OUT/server.log" 2>&1 || true
  docker inspect "$NAME" >"$OUT/container_inspect.json" 2>/dev/null || true
  docker stop -t 60 "$NAME" >"$OUT/stop.log" 2>&1 || true
  docker rm -f "$NAME" >/dev/null 2>&1 || true
  "$REPO/bin/xpu-health" >"$OUT/health_post.log" 2>&1 || rc=1
  echo "RESULT -> rc=$rc artifacts=$OUT endpoint=down"
  exit "$rc"
}
trap cleanup EXIT INT TERM

test -s "$PUSHDIR/libxpu_push_ar_fused_rmsnorm.so"
sha256sum \
  "$REPO/kernels/xpu_push_ar_fused_rmsnorm.cpp" \
  "$PUSHDIR/libxpu_push_ar_fused_rmsnorm.so" \
  "$REPO/sglang/patches/xpu_delayed_mlp_ar.py" \
  "$REPO/sglang/patches/xpu_fused_mlp_ar_norm.py" \
  >"$OUT/artifacts.sha256"
"$REPO/bin/xpu-health" | tee "$OUT/health_pre.log"

CTX="$CTX" RADIX=0 MAXREQ=1 PORT="$PORT" NAME="$NAME" SERVED="$SERVED" \
  IMG=sglang-xpu:mtp CKPT=/models/qwen3.6-27b/w8a8-sqgptq API_KEY= \
  SPEC_STEPS=10 SPEC_DRAFT=11 REPLICATE_MTP_EMBED=1 \
  PUSH_AR=1 PUSH_AR_MIN_NUMEL=1048576 PUSHDIR="$PUSHDIR" \
  PUSH_AR_SO="$PUSH_AR_SO" DELAY_MLP_AR=1 FUSED_MLP_AR_NORM=1 \
  bash "$SHELF" start 2>&1 | tee "$OUT/start.log"

curl -fsS --max-time 15 "http://127.0.0.1:$PORT/v1/models" >"$OUT/models.json"
python3 "$REPO/sglang/deterministic_equivalence_probe.py" \
  --base "http://127.0.0.1:$PORT" --model "$SERVED" \
  --max-tokens 128 --out "$OUT/deterministic.json" \
  2>&1 | tee "$OUT/deterministic.log"
python3 "$REPO/sglang/soak_probe.py" "$PORT" "$SERVED" 1600 400 localhost \
  2>&1 | tee "$OUT/mechanism_soak.log"

docker logs "$NAME" >"$OUT/server.log" 2>&1
rg -q '\[c3b-fused\] installed.*M=1\.\.8/10/11' "$OUT/server.log"
python3 - "$OUT/server.log" <<'PY'
import re
import sys

text = open(sys.argv[1], errors="replace").read()
routes = [tuple(map(int, row)) for row in re.findall(
    r"\[c3b-delayed-mlp\] ROUTES rank=(\d+) eligible=(\d+) "
    r"consumed=(\d+) generic=(\d+)", text
)]
latest = {}
for rank, eligible, consumed, generic in routes:
    latest[rank] = (eligible, consumed, generic)
m11 = [int(value) for value in re.findall(
    r"\[c3b-fused\] calls=(\d+) rows=11 hidden=5120", text
)]
ok = (
    set(latest) == {0, 1}
    and all(e == c and g == 0 and c >= 4096 for e, c, g in latest.values())
    and len(m11) >= 2
    and max(m11, default=0) >= 4096
)
print(f"MECHANISM -> latest={latest} m11={m11[-4:]} ok={ok}")
raise SystemExit(0 if ok else 1)
PY
if rg -i 'device_lost|out_of_resources|ur_result_error|enginedead|!!!!|(^|[^a-z])nan([^a-z]|$)' \
  "$OUT/server.log"; then
  echo "fatal marker in server log" >&2
  exit 1
fi

echo "VERDICT -> measured M<=11 shapes fused with generic=0; endpoint will remain down"
