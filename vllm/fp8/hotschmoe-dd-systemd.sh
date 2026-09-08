#!/usr/bin/env bash
# Systemd readiness/stop helpers. No inference or independent GPU access.
set -euo pipefail
SERVED="${SERVED:-hotschmoe-dd}"
case "${1:-}" in
  ready)
    for ((attempt=0; attempt<570; attempt++)); do
      if [[ "$(docker inspect -f '{{.State.Running}}' hotschmoe-dd 2>/dev/null || true)" == true ]] &&
          curl -fsS --max-time 3 http://127.0.0.1:18124/v1/models |
          jq -e --arg model "$SERVED" '.data | any(.id == $model)' >/dev/null &&
          curl -fsS --max-time 3 http://127.0.0.1:18080/health >/dev/null; then
        exit 0
      fi
      sleep 2
    done
    echo 'hotschmoe-dd readiness timed out' >&2
    exit 1
    ;;
  stop)
    # Normal container shutdown lets the leased launcher run post-health.
    if docker inspect hotschmoe-dd >/dev/null 2>&1; then
      docker stop -t 60 hotschmoe-dd >/dev/null
      # ExecStop must wait; otherwise systemd would kill cleanup immediately.
      if [[ "${MAINPID:-}" =~ ^[1-9][0-9]*$ ]]; then
        for ((attempt=0; attempt<780; attempt++)); do
          kill -0 "$MAINPID" 2>/dev/null || exit 0
          sleep 1
        done
        echo 'hotschmoe-dd teardown did not finish within 780 seconds' >&2
        exit 1
      fi
    fi
    ;;
  *) echo 'usage: hotschmoe-dd-systemd.sh ready|stop' >&2; exit 2 ;;
esac
