#!/usr/bin/env bash
# trace_driver.sh -- drive a vLLM torch-profiler capture of a controlled PREFILL or DECODE.
# The serve must be launched with B70_PROFILER_DIR=/prof and that path mounted
# from the host (for example B70_EXTRA_MOUNTS=/host/trace:/prof).
# Usage: HOST=.. PORT=.. MODEL=.. KEY=.. MODE=prefill|decode bash trace_driver.sh
# Emits: hits /start_profile, fires the controlled request, /stop_profile. Traces land in the
# mounted profiler dir (one .pt.trace.json[.gz] per rank). Parse with parse_trace.py.
set -uo pipefail
HOST="${HOST:-http://192.168.10.5}"; PORT="${PORT:-18082}"; MODEL="${MODEL:?set MODEL}"
KEY="${KEY:-testkey123}"; MODE="${MODE:-decode}"; SEED="${SEED:-1701}"
BASE="$HOST:$PORT"; AUTH=(-H "Authorization: Bearer $KEY" -H "Content-Type: application/json")

echo "[trace] MODE=$MODE model=$MODEL base=$BASE seed=$SEED"
# warmup one request so weights/caches are hot and JIT is done (not profiled)
curl -fsS "${AUTH[@]}" "$BASE/v1/completions" -d "{\"model\":\"$MODEL\",\"prompt\":\"hello\",\"max_tokens\":8,\"temperature\":0,\"seed\":$SEED}" >/dev/null

echo "[trace] /start_profile"
curl -fsS -X POST "${AUTH[@]}" "$BASE/start_profile"
echo

if [ "$MODE" = prefill ]; then
  # long prompt, 1 decode token -> the step is dominated by the prefill forward
  PROMPT="Summarize the following text.\n\n$(python3 -c 'print(("The quick brown fox jumps over the lazy dog. "*520))')"
  BODY=$(python3 -c "import json,sys; print(json.dumps({'model':'$MODEL','prompt':sys.argv[1],'max_tokens':1,'temperature':0,'seed':int(sys.argv[2])}))" "$PROMPT" "$SEED")
  echo "[trace] firing PREFILL (~4000-tok prompt, max_tokens=1)"
  curl -fsS "${AUTH[@]}" "$BASE/v1/completions" -d "$BODY" | python3 -c "import sys,json,hashlib; d=json.load(sys.stdin); t=d['choices'][0]['text']; print('  prompt_tokens=',d['usage']['prompt_tokens'],'completion=',d['usage']['completion_tokens'],'response_sha256=',hashlib.sha256(t.encode()).hexdigest())"
else
  # short prompt, many decode tokens -> steps dominated by decode (MTP verify + drafter)
  echo "[trace] firing DECODE (short prompt, 256 tokens, ignore_eos)"
  curl -fsS "${AUTH[@]}" "$BASE/v1/completions" -d "{\"model\":\"$MODEL\",\"prompt\":\"Write a long detailed essay on computing history.\",\"max_tokens\":256,\"temperature\":0,\"ignore_eos\":true,\"seed\":$SEED}" | python3 -c "import sys,json,hashlib; d=json.load(sys.stdin); t=d['choices'][0]['text']; print('  completion_tokens=',d['usage']['completion_tokens'],'response_sha256=',hashlib.sha256(t.encode()).hexdigest())"
fi

echo "[trace] /stop_profile (flushes trace to disk; may take ~10-30s)"
curl -fsS -X POST "${AUTH[@]}" "$BASE/stop_profile"
echo
echo "[trace] done -- check the mounted profiler dir for *.trace.json*"
