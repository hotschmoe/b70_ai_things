#!/usr/bin/env bash
# agentic-eval/lib/common.sh -- shared bash helpers sourced by serve/ + harness run.sh scripts.
# Pure orchestration; no GPU, no heavy deps. evallib.py is stdlib-only so the SYSTEM python runs it
# (the per-harness uv venvs are only for the harnesses themselves).
set -uo pipefail

AE_LIB="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AE_ROOT="$(cd "$AE_LIB/.." && pwd)"                  # agentic-eval/
REPO_ROOT="$(cd "$AE_ROOT/.." && pwd)"               # repo root (has rdy_to_serve/, bin/)
source "$AE_ROOT/configs.sh"
AE_PY="${AE_PY:-python3}"                            # stdlib-only -> system python3.14 is fine

# Per-config endpoint + result dir. Call AFTER eval_config <label>.
ae_set_config() {
  eval_config "$1" || return 2
  EVAL_ENDPOINT="http://localhost:${EVAL_PORT}/v1"
  EVAL_BASE_URL="http://localhost:${EVAL_PORT}"
  RESULTS_DIR="$AE_ROOT/results/$EVAL_LABEL"
  mkdir -p "$RESULTS_DIR"
  export EVAL_LABEL EVAL_ARCH EVAL_SCHEME EVAL_SERVED EVAL_SERVE_DIR EVAL_CARDS \
         EVAL_PORT EVAL_MAXLEN EVAL_MAXSEQS EVAL_ENDPOINT EVAL_BASE_URL RESULTS_DIR \
         EVAL_THINKING EVAL_NO_THINK EVAL_REASONPARSER EVAL_SERVE_ENV \
         AE_TEMPERATURE AE_TOP_P AE_SEED AE_MAX_TOKENS AE_CONCURRENCY AE_ROOT AE_LIB
}

# "PROMPT GEN" cumulative token counters from vLLM /metrics, or "NA NA" if unavailable.
ae_snap() { "$AE_PY" "$AE_LIB/evallib.py" snap "${EVAL_PORT:-18080}"; }
ae_now()  { date +%s.%N; }
ae_log()  { printf '[ae %s] %s\n' "$(date +%H:%M:%S)" "$*" >&2; }

# Block until the served model answers /v1/models with EVAL_SERVED, or fail after N seconds.
ae_wait_endpoint() {
  local want="${1:-$EVAL_SERVED}" secs="${2:-120}" t0 ids
  t0=$(date +%s)
  while :; do
    ids=$(curl -s --max-time 4 "http://localhost:${EVAL_PORT}/v1/models" 2>/dev/null | grep -oE '"id":"[^"]*"' || true)
    printf '%s' "$ids" | grep -q "\"id\":\"$want\"" && { ae_log "endpoint serving '$want'"; return 0; }
    [ $(( $(date +%s) - t0 )) -ge "$secs" ] && { ae_log "endpoint did NOT serve '$want' within ${secs}s (saw: ${ids:-none})"; return 1; }
    sleep 3
  done
}

# Verify the live served id matches what the config claims (CLAUDE.md model-identity rule).
ae_verify_identity() {
  local got
  got=$(curl -s --max-time 5 "http://localhost:${EVAL_PORT}/v1/models" 2>/dev/null | grep -oE '"id":"[^"]*"' | head -1 | sed 's/.*:"//; s/"$//')
  if [ "$got" != "$EVAL_SERVED" ]; then
    ae_log "IDENTITY MISMATCH: config '$EVAL_LABEL' expects served id '$EVAL_SERVED' but endpoint says '$got'"
    return 1
  fi
  ae_log "identity OK: $got"
}

# Live endpoint self-test, run after a serve and BEFORE the heavy harnesses. Validates the parts only
# a live serve can: identity, vLLM /metrics token counters (the basis of our token accounting),
# a real greedy completion, and a native tool-call. Returns 0 unless the model fails to generate text
# (the must-have); missing /metrics or tool_calls are WARN-only (token cols just become NA; BFCL uses
# prompt-FC so it does not need native tool_calls -- but tau2 does).
ae_endpoint_selftest() {
  local rc=0 snap txt tc
  ae_verify_identity || rc=1
  snap=$(ae_snap)
  if [ "$snap" = "NA NA" ]; then
    ae_log "WARN: vLLM /metrics token counters absent -> token accounting will report NA (not fatal)"
  else
    ae_log "metrics OK: vllm token counters present (prompt gen = $snap)"
  fi
  txt=$(curl -s --max-time 90 "${EVAL_BASE_URL}/v1/completions" -H 'Content-Type: application/json' \
    -d "{\"model\":\"$EVAL_SERVED\",\"prompt\":\"def add(a, b):\\n    return\",\"max_tokens\":16,\"temperature\":0}" \
    2>/dev/null | grep -oE '"text":"[^"]*"' | head -1)
  if [ -z "$txt" ]; then ae_log "FAIL: endpoint produced no completion text"; rc=1
  else ae_log "completion OK: ${txt:0:70}"; fi
  tc=$(curl -s --max-time 90 "${EVAL_BASE_URL}/v1/chat/completions" -H 'Content-Type: application/json' \
    -d "{\"model\":\"$EVAL_SERVED\",\"messages\":[{\"role\":\"user\",\"content\":\"What is the weather in Paris? Call the tool.\"}],\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"description\":\"current weather for a city\",\"parameters\":{\"type\":\"object\",\"properties\":{\"city\":{\"type\":\"string\"}},\"required\":[\"city\"]}}}],\"tool_choice\":\"auto\",\"max_tokens\":128,\"temperature\":0}" 2>/dev/null)
  if printf '%s' "$tc" | grep -q '"tool_calls"'; then ae_log "tool-call OK: native tool_calls emitted"
  else ae_log "WARN: no native tool_calls emitted (BFCL prompt-FC is unaffected; tau2/native-FC would need TOOLCALL on)"; fi
  return $rc
}
