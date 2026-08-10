#!/usr/bin/env bash
# Wait for OpenAI-compatible health on PORT (default 8000).
set -euo pipefail
PORT=${1:-8000}
NAME=${2:-}
TIMEOUT=${TIMEOUT:-1200}
t0=$(date +%s)
while true; do
  if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 \
     || curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
    echo "healthy on :$PORT after $(( $(date +%s) - t0 ))s"
    exit 0
  fi
  if [ -n "$NAME" ] && ! docker ps --format '{{.Names}}' | grep -qx "$NAME"; then
    echo "container $NAME is not running" >&2
    docker logs "$NAME" 2>&1 | tail -80 >&2 || true
    exit 1
  fi
  if [ $(( $(date +%s) - t0 )) -ge "$TIMEOUT" ]; then
    echo "timeout ${TIMEOUT}s waiting for :$PORT" >&2
    [ -n "$NAME" ] && docker logs "$NAME" 2>&1 | tail -80 >&2 || true
    exit 1
  fi
  sleep 5
done
