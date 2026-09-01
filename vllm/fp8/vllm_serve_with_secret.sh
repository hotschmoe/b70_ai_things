#!/usr/bin/env bash
# Inject the vLLM API key from a read-only Docker secret without placing the
# key in the image configuration, repository, or host-side command arguments.
set -euo pipefail

[ -n "${VLLM_API_KEY_FILE:-}" ] || {
  echo "VLLM_API_KEY_FILE is required" >&2
  exit 2
}
[ -s "$VLLM_API_KEY_FILE" ] || {
  echo "VLLM_API_KEY_FILE is missing or empty" >&2
  exit 2
}
IFS= read -r VLLM_API_KEY < "$VLLM_API_KEY_FILE" || [ -n "$VLLM_API_KEY" ]
[ -n "$VLLM_API_KEY" ] || {
  echo "VLLM_API_KEY_FILE contains an empty first line" >&2
  exit 2
}
export VLLM_API_KEY
exec vllm "$@"
