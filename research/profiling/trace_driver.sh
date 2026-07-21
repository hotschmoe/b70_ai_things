#!/usr/bin/env bash
# trace_driver.sh -- drive a vLLM torch-profiler capture of a controlled PREFILL or DECODE.
# The serve must be launched with VLLM_TORCH_PROFILER_DIR=/prof (mounted to a host dir).
# Usage: HOST=.. PORT=.. MODEL=.. KEY=.. MODE=prefill|decode bash trace_driver.sh
# Emits: hits /start_profile, fires the controlled request, /stop_profile. Traces land in the
# mounted profiler dir (one .pt.trace.json[.gz] per rank). Parse with parse_trace.py.
set -uo pipefail
HOST="${HOST:-http://192.168.10.5}"; PORT="${PORT:-18082}"; MODEL="${MODEL:?set MODEL}"
KEY="${KEY:-testkey123}"; MODE="${MODE:-decode}"
BASE="$HOST:$PORT"; AUTH=(-H "Authorization: Bearer $KEY" -H "Content-Type: application/json")

echo "[trace] MODE=$MODE model=$MODEL base=$BASE"
# warmup one request so weights/caches are hot and JIT is done (not profiled)
curl -s "${AUTH[@]}" "$BASE/v1/completions" -d "{\"model\":\"$MODEL\",\"prompt\":\"hello\",\"max_tokens\":8,\"temperature\":0}" >/dev/null

echo "[trace] /start_profile"
curl -s -X POST "${AUTH[@]}" "$BASE/start_profile" ; echo

if [ "$MODE" = prefill ]; then
  # long prompt, 1 decode token -> the step is dominated by the prefill forward
  PROMPT="Summarize the following text.\n\n$(python3 -c 'print(("The quick brown fox jumps over the lazy dog. "*520))')"
  BODY=$(python3 -c "import json,sys; print(json.dumps({'model':'$MODEL','prompt':sys.argv[1],'max_tokens':1,'temperature':0}))" "$PROMPT")
  echo "[trace] firing PREFILL (~4000-tok prompt, max_tokens=1)"
  curl -s "${AUTH[@]}" "$BASE/v1/completions" -d "$BODY" | python3 -c "import sys,json; d=json.load(sys.stdin); print('  prompt_tokens=',d['usage']['prompt_tokens'],'completion=',d['usage']['completion_tokens'])"
else
  # short prompt, many decode tokens -> steps dominated by decode (MTP verify + drafter)
  echo "[trace] firing DECODE (short prompt, 256 tokens, ignore_eos)"
  curl -s "${AUTH[@]}" "$BASE/v1/completions" -d "{\"model\":\"$MODEL\",\"prompt\":\"Write a long detailed essay on computing history.\",\"max_tokens\":256,\"temperature\":0,\"ignore_eos\":true}" | python3 -c "import sys,json; d=json.load(sys.stdin); print('  completion_tokens=',d['usage']['completion_tokens'])"
fi

echo "[trace] /stop_profile (flushes trace to disk; may take ~10-30s)"
curl -s -X POST "${AUTH[@]}" "$BASE/stop_profile" ; echo
echo "[trace] done -- check the mounted profiler dir for *.trace.json*"
