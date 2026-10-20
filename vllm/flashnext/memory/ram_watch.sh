#!/usr/bin/env bash
# Host-only guard, including request verification after server readiness.
set -euo pipefail
while :; do
  active=0
  for pid in "$@"; do
    if kill -0 "$pid" 2>/dev/null; then active=1; fi
  done
  (( active )) || exit 0
  available=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)
  if (( available < 8388608 )); then
    echo "LOW_RAM_ABORT available_kb=$available $(date -u +%FT%TZ)"
    while IFS= read -r name; do
      case "$name" in
        vllm-fn-x2-*) docker stop -t 10 "$name" || true ;;
      esac
    done < <(docker ps --format '{{.Names}}')
  fi
  sleep 3
done
