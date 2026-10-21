#!/usr/bin/env bash
# Prepared CPU review copy. Parent must authorize every both-card serve.
# Decode-only selector; prefill remains native. One finite arm, no submission.
set -uo pipefail
R=/mnt/vm_8tb/github/b70_ai_things
B=/mnt/vm_8tb/b70/build/vllm-flashnext
H=$R/vllm/flashnext/harness
S=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
MODE=${B70_Q1_SELECTOR:?native, torch or fused}
case "$MODE" in native|torch|fused) ;; *) exit 2 ;; esac
export TAG=${TAG:?fresh output tag}
OUT=$B/q1/full-$TAG
AOUT=$B/x6/flashnext-ista-iq3-xxs-ep-qsa-$TAG
NAME=vllm-fn-x6-$TAG
PHASE_TIMEOUT=${Q1_PHASE_PROGRESS_TIMEOUT:-180}
[[ "$PHASE_TIMEOUT" =~ ^[0-9]+$ ]] && [ "$PHASE_TIMEOUT" -ge 180 ] || {
  echo Q1_REQUIRE_PHASE_PROGRESS_TIMEOUT_AT_LEAST_180; exit 2;
}
[ ! -e "$OUT" ] && [ ! -e "$AOUT" ] || { echo Q1_REFUSE_EXISTING_OUTPUT; exit 2; }
export B70_AGENT=Q1 B70_Q1_SELECTOR
export X6_PRIORITY_WAIT=0 X6_F1_DIAGNOSTICS=0 X6_REQUIRE_REFERENCES=1 X6_QSA=1
export X6_REFERENCES=$B/q1/strata-references.json
FIXTURES=$B/x6/reference-prompts.json
export SKIP_HEALTH_BEFORE=0 PORT=18156 MAX_LEN=65536 MAX_BATCHED=12288 DECODE=64 PREFILL=1024
export O4_RECIPE=$H/o4_recipe.env F1_DELTA=$H/f1_prefill_delta.env
export X6_API_WRAPPER=$H/q1_selector_api.py X6_PYPATH_PRE=$S:$H
RESEARCH_SUFFIX=$MODE
[ "$MODE" != fused ] || RESEARCH_SUFFIX=fused-r2
export X6_RESEARCH_ALIAS=qwen3.8-flash-next-ista-gsq-rco-iq3-xxs-q1-$RESEARCH_SUFFIX
rg -q "^[[:space:]]*served_model_id: $X6_RESEARCH_ALIAS$" "$R/evals/configs/models.yaml" || {
  echo Q1_REQUIRE_REGISTERED_RESEARCH_ALIAS; exit 3;
}
[ -s "$FIXTURES" ] && [ -s "$X6_REFERENCES" ] || { echo Q1_REQUIRE_FIXTURES_AND_REFERENCES; exit 3; }
if [ "$MODE" = fused ]; then
  [ -s "$H/q1_fused_r2/selector.py" ] || { echo Q1_REQUIRE_PORTED_FUSED_R2_SELECTOR; exit 3; }
  [[ "${B70_Q1_FUSED_SO:-}" = /* ]] && [ -s "$B70_Q1_FUSED_SO" ] || {
    echo Q1_REQUIRE_EXPLICIT_FUSED_R2_SO; exit 3;
  }
fi
if [ -n "${B70_Q1_FUSED_SO:-}" ]; then
  python3 - "$B70_Q1_FUSED_SO" "$H/q1_fused_r2/measured-source-sha256.json" <<'PY_ID'
import hashlib,json,sys
from pathlib import Path
manifest=json.loads(Path(sys.argv[2]).read_text())
assert manifest['namespace']=='b70_q1_packcompact_r2'
assert hashlib.sha256(Path(sys.argv[1]).read_bytes()).hexdigest()==manifest['measured_library_sha256']
print('Q1_MEASURED_R2_SO_IDENTITY_OK')
PY_ID
  [ "$?" = 0 ] || exit 3
fi
mkdir -p "$OUT"
exec > >(tee "$OUT/selector-window.log") 2>&1
export O4_SERVE_EXTRA="B70_GDN_BA_DETERMINISTIC=0 B70_D1_FA_PLAN=0 B70_ONEDNN_DET=0 B70_PREFILL_QSA=1 B70_P1_BUCKET_OVERLAY=0 B70_Q1_SELECTOR=$MODE"
if [ -n "${B70_Q1_FUSED_SO:-}" ]; then
  export O4_SERVE_EXTRA="$O4_SERVE_EXTRA B70_Q1_FUSED_SO=$B70_Q1_FUSED_SO"
  sha256sum "$B70_Q1_FUSED_SO" > "$OUT/fused-so-before.sha256" || exit 3
fi
unset SERVE_ARGS_EXTRA O4_FA_OVERLAY F1_PYPATH_PRE F1_API_WRAPPER F1_POST_PROBE_SCRIPT
python3 "$H/d1_source_identity.py" snapshot "$OUT/source-before.json" || exit 3
sha256sum "$S/q1_selector_hook.py" "$S/q1_selector_outcome.py" "${BASH_SOURCE[0]}" \
  "$H/q1_selector_api.py" "$FIXTURES" "$X6_REFERENCES" > "$OUT/selector-extra-before.sha256" || exit 3
if [ -n "${B70_Q1_FUSED_SO:-}" ]; then
  sha256sum "$H/q1_fused_r2/measured-source-sha256.json" >> "$OUT/selector-extra-before.sha256" || exit 3
fi
printf 'MODE:%s TAG:%s SO:%s SCOPE:decode PHASE_OUTPUT_BUDGET_S:%s\n' \
  "$MODE" "$TAG" "${B70_Q1_FUSED_SO:-unused}" "$PHASE_TIMEOUT"
# x6_probe publishes after all four streams in a fixture. The preserved
# native 32K phase took about139s. 180s is a bounded phase-output budget,
# not a deadlock detector; operational failure remains explicit if it fires.
bash "$H/x6_serve.sh" > "$OUT/launch.log" 2>&1 &
ARM=$!
(
  active=0; before=''; last=$(date +%s); config_saved=0
  while kill -0 "$ARM" 2>/dev/null; do
    if [ "$active" = 0 ] && [ -f "$AOUT/window.log" ] && rg -q '^READY ' "$AOUT/window.log"; then
      active=1; last=$(date +%s)
    fi
    if [ "$active" = 1 ]; then
      if [ "$config_saved" = 0 ] && curl -sf --max-time 3 "http://127.0.0.1:$PORT/server_info?config_format=json" > "$OUT/server-info.tmp"; then
        mv "$OUT/server-info.tmp" "$OUT/server-info.json"; config_saved=1
        curl -sf --max-time 3 "http://127.0.0.1:$PORT/metrics" > "$OUT/metrics-first.txt" || true
      fi
      progress=$(stat -c '%n:%s:%Y' "$AOUT/greedy.json" "$AOUT/bench.json" "$AOUT/x6-probe.json" 2>/dev/null || true)
      if [ "$progress" != "$before" ]; then before=$progress; last=$(date +%s); fi
      if [ $(($(date +%s)-last)) -ge "$PHASE_TIMEOUT" ]; then
        echo "Q1_PHASE_OUTPUT_BUDGET_EXPIRED seconds=$PHASE_TIMEOUT" | tee "$OUT/no-progress.txt"
        docker stop -t 15 "$NAME" > "$OUT/no-progress-stop.txt" 2>&1 || true
        break
      fi
    fi
    sleep 2
  done
) &
WATCH=$!
wait "$ARM"; RC=$?
kill "$WATCH" 2>/dev/null || true
wait "$WATCH" 2>/dev/null || true
python3 "$H/d1_source_identity.py" verify "$OUT/source-before.json" > "$OUT/source-after.json"
SOURCE=$?
sha256sum -c "$OUT/selector-extra-before.sha256" > "$OUT/selector-extra-after.txt" 2>&1
EXTRA=$?
SO=0
if [ -s "$OUT/fused-so-before.sha256" ]; then sha256sum -c "$OUT/fused-so-before.sha256" > "$OUT/fused-so-after.txt" 2>&1; SO=$?; fi
python3 - "$OUT/server-info.json" <<'PY'
import json,sys
c=json.load(open(sys.argv[1]))['vllm_config']
assert c['speculative_config'] is None
assert c['cache_config']['enable_prefix_caching'] is False
assert c['model_config']['max_model_len']==65536
assert c['scheduler_config']['max_num_batched_tokens']==12288
assert c['scheduler_config']['max_num_seqs']==4
assert c['parallel_config']['tensor_parallel_size']==1
print('Q1_ACTUAL_CONFIG_OK')
PY
CONFIG=$?
ACTIVE=0
if [ "$MODE" != native ]; then
  rg -q "Q1_SELECTOR_APPLIED mode=$MODE scope=decode " "$AOUT/serve.log"; ACTIVE=$?
fi
python3 "$S/q1_selector_outcome.py" --aout "$AOUT" --fixtures "$FIXTURES" \
  --out "$OUT/selector-outcome.json"
OPERATIONAL=$?
printf 'Q1_FULL_RC arm=%s source=%s extra=%s so=%s config=%s activation=%s operational=%s\n' \
  "$RC" "$SOURCE" "$EXTRA" "$SO" "$CONFIG" "$ACTIVE" "$OPERATIONAL"
# A completed strict-parity reject returns1 from X6 and remains recorded in
# selector-outcome.json. Other errors cannot pass the operational classifier.
if { [ "$RC" = 0 ] || [ "$RC" = 1 ]; } && [ "$SOURCE" = 0 ] && [ "$EXTRA" = 0 ] && \
   [ "$SO" = 0 ] && [ "$CONFIG" = 0 ] && [ "$ACTIVE" = 0 ] && \
   [ "$OPERATIONAL" = 0 ] && [ ! -e "$OUT/no-progress.txt" ]; then
  echo Q1_OPERATIONAL_COMPLETE_STRICT_STATUS_IN_JSON
  exit 0
fi
exit 1
