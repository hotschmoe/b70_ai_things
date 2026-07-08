#!/usr/bin/env bash
# L0 leak-checker on the CRASHING NVFP4 TP=2 captured+MTP config: drive ~5000 forced-decode tokens
# (below the ~8-12k crash), then GRACEFULLY stop (SIGTERM -> clean atexit) so the L0 validation-layer
# leak checker + UR L0 leaks report print at worker teardown -> names the accumulating handle type.
set -uo pipefail
REPO=/mnt/vm_8tb/github/b70_ai_things; cd "$REPO"
NAME=nvfp4_leak; PORT=8078
LOGDIR="$REPO/results/logs"; mkdir -p "$LOGDIR"; STAMP="$(date +%Y%m%d_%H%M%S)"
SLOG="$LOGDIR/leakcheck_${STAMP}.serve.log"; CLOG="$LOGDIR/leakcheck_${STAMP}.container.log"
docker rm -f "$NAME" >/dev/null 2>&1

echo "===== LEAK CHECK (NVFP4 TP=2 captured+MTP, L0 leak checker) $(date -u +%FT%TZ) ====="
NAME="$NAME" PORT="$PORT" TP=2 MODE=fused GRAPH=1 MTPTOK=5 KV_FP8=0 CAPSIZES=1,2,4,8 \
  MAXLEN=131072 UTIL=0.85 B70_DEBUG=1 \
  B70_EXTRA_ENV="ZE_ENABLE_VALIDATION_LAYER=1 ZE_ENABLE_PARAMETER_VALIDATION=1 ZEL_ENABLE_BASIC_LEAK_CHECKER=1 UR_L0_LEAKS_DEBUG=1 UR_LOG_LEVEL=info" \
  bash vllm/nvfp4/serve_nvfp4_27b.sh start > "$SLOG" 2>&1

echo "waiting for /health (leak-checker adds overhead)..."
ok=0; for i in $(seq 1 150); do curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1 && { ok=1; break; }; \
  docker ps --filter name="$NAME" -q | grep -q . || { echo "died in startup"; break; }; sleep 5; done
[ "$ok" != 1 ] && { echo "NEVER HEALTHY"; docker logs "$NAME" > "$CLOG" 2>&1; tail -20 "$CLOG"; docker rm -f "$NAME" >/dev/null 2>&1; exit 1; }
echo "HEALTHY; driving ~5000 forced-decode tokens (below the crash)..."

# one forced-decode request, ~5000 tokens (below ~8k crash), then graceful stop
PORT="$PORT" KEY="" WORKERS=1 CTX_CHARS=6000 MAXTOK=5000 CEIL_TOK=5000 CEIL_SEC=600 \
  python3 vllm/soak_concurrent.py leakdrive 2>&1 | tail -4

echo "=== GRACEFUL stop (SIGTERM, 150s) so atexit leak report prints ==="
docker stop -t 150 "$NAME" >/dev/null 2>&1
docker logs "$NAME" > "$CLOG" 2>&1

echo "=== LEAK REPORT (accumulating handle types) ==="
grep -iE 'leak|not destroyed|zeCommandList|zeEvent|zeCommandQueue|ze_command|ze_event|handle.*count|Retained|create.*destroy|LEAK_CHECKER|UR_L0_LEAKS' "$CLOG" 2>/dev/null | tail -40
echo "=== (full report saved: $CLOG) ==="
docker rm -f "$NAME" >/dev/null 2>&1
echo "===== LEAK CHECK DONE $(date -u +%FT%TZ) ====="
