#!/usr/bin/env bash
# Parent grants both cards after P1 bucket gate and the QSA-disabled control.
# One arm only; run 0 and 1 separately with the same frozen fixtures/sources.
set -uo pipefail
R=/mnt/vm_8tb/github/b70_ai_things
B=/mnt/vm_8tb/b70/build/vllm-flashnext
H="$R/vllm/flashnext/harness"
P="$R/vllm/flashnext/plugins/b70_p1_xpu"
FORK="$B/vllm"
FLAG="${B70_PREFILL_SELECTED_DQ:?must be 0 or 1}"
case "$FLAG" in 0|1) ;; *) echo P1_BAD_SELECTED_FLAG; exit 2 ;; esac
: "${P1_PREFILL_FIXTURES:?frozen host fixture path required}"
if [ "$FLAG" = 1 ] && [ ! -s "$P1_PREFILL_FIXTURES" ]; then
  echo P1_TREATMENT_REQUIRES_CONTROL_FIXTURES
  exit 2
fi
CONTROL_TAG="${P1_CONTROL_TAG:-p1-control-r1}"
if [ "$FLAG" = 1 ] && [ ! -f "$B/p1/full-$CONTROL_TAG/arm-passed" ]; then
  echo P1_TREATMENT_REQUIRES_PASSED_CONTROL
  exit 2
fi
export TAG="${TAG:-p1-overlay-${FLAG}-r1}"
OUT="$B/p1/full-$TAG"
AOUT="$B/f1/merged-$TAG"
NAME="vllm-fn-f1-merged-$TAG"
mkdir -p "$OUT"
[ ! -e "$OUT/started" ] || { echo P1_ARM_ALREADY_STARTED; exit 2; }
touch "$OUT/started"
exec > >(tee -a "$OUT/p1-window.log") 2>&1
export B70_AGENT=P1 B70_PREFILL_SELECTED_DQ P1_PREFILL_FIXTURES
export MAX_LEN="${MAX_LEN:-65536}" MAX_BATCHED=12288 DECODE=64 PREFILL=1024
export F1_LENGTHS=',' F1_COHERENCE_LEN=0 F1_DECODE=0 F1_REPEAT=1 F1_NEEDLE_LENS=''
export RUN_PROMO_GATE=0 SKIP_HEALTH_BEFORE=0
export F1_API_WRAPPER="$H/p1_api.py"
export F1_PYPATH_PRE="$P"
export F1_POST_PROBE_SCRIPT="$H/p1_prefill_callback.sh"
export O4_SERVE_EXTRA="${P1_BASE_SERVE_EXTRA:-} B70_GDN_BA_DETERMINISTIC=0 B70_D1_FA_PLAN=0 B70_ONEDNN_DET=0 B70_QSA=0 B70_PREFILL_QSA=0 B70_P1_BUCKET_OVERLAY=1 B70_PREFILL_SELECTED_DQ=$FLAG B70_P1_SO=$B/p1/_C.p1selected.so B70_P1_BUCKET_SO=$B/p1/_C.p1bucket.so"
unset SERVE_ARGS_EXTRA O4_FA_OVERLAY
printf 'P1_ARM:%s MAX_LEN:%s FIXTURES:%s\n' "$FLAG" "$MAX_LEN" "$P1_PREFILL_FIXTURES"
printf '%s\n' "$O4_SERVE_EXTRA" > "$OUT/effective-extra.env"
# Capture the dirty fork exactly. No checkout/reset/commit is performed here.
git -C "$FORK" rev-parse HEAD > "$OUT/fork-head-before.txt"
git -C "$FORK" diff --binary HEAD > "$OUT/fork-before.patch"
B70_AGENT=P1 "$R/bin/cpu-run" python3 "$H/d1_source_identity.py" snapshot \
  "$OUT/source-identity.json" > "$OUT/source-before.log" 2>&1 || exit 3
sha256sum "$B/p1/_C.p1selected.so" "$B/p1/_C.p1bucket.so" \
  "$B/p1/bucket_overlay.cpp" "$B/p1/bucket_overlay.json" \
  "$P/generate_bucket_overlay.py" "$P/b70_p1_xpu/__init__.py" "$P/b70_p1_xpu/serving.py" \
  "$P/csrc/p1_dequant.cpp" "$P/csrc/p1_dequant.hpp" "$P/csrc/p1_selected_register.cpp" \
  "$R/vllm/flashnext/plugins/b70_gguf_xpu/csrc/bucket_moe.cpp" \
  "$R/vllm/flashnext/plugins/b70_gguf_xpu/csrc/dequant_wide.cpp" \
  "$R/vllm/flashnext/plugins/b70_gguf_xpu/csrc/dequant_wide.hpp" \
  "$R/vllm/flashnext/plugins/b70_gguf_xpu/csrc/decode.hpp" \
  > "$OUT/p1-identity.sha256" || exit 3
if [ -s "$P1_PREFILL_FIXTURES" ]; then
  sha256sum "$P1_PREFILL_FIXTURES" > "$OUT/fixture-before.sha256"
fi
bash "$H/f1_merged_serve.sh" > "$OUT/launch.log" 2>&1 &
ARM_PID=$!
# Startup may load for several minutes. Once READY, client evidence must
# advance at least every120s. Server periodic statistics do not count.
(
  last_progress=$(date +%s)
  previous=''
  active=0
  while kill -0 "$ARM_PID" 2>/dev/null; do
    if [ "$active" = 0 ] && [ -f "$AOUT/window.log" ] && rg -q '^READY ' "$AOUT/window.log"; then
      active=1
      last_progress=$(date +%s)
    fi
    if [ "$active" = 1 ]; then
      progress=$(stat -c '%n:%s:%Y' "$AOUT/greedy.json" "$AOUT/bench.json" "$AOUT/probe.json" \
        "$AOUT/probe.log" "$AOUT/p1-prefill.json" "$AOUT/p1-prefill.log" 2>/dev/null || true)
      if [ "$progress" != "$previous" ]; then
        previous="$progress"
        last_progress=$(date +%s)
      fi
      now=$(date +%s)
      if [ $((now-last_progress)) -ge 120 ]; then
        echo "P1_NO_CLIENT_OUTPUT_120S $(date -u +%H:%M:%S)" | tee "$OUT/no-output-stop.txt"
        docker stop -t 15 "$NAME" > "$OUT/no-output-docker-stop.log" 2>&1 || true
        break
      fi
    fi
    sleep 2
  done
) &
MONITOR_PID=$!
wait "$ARM_PID"
ARM_RC=$?
kill "$MONITOR_PID" 2>/dev/null || true
wait "$MONITOR_PID" 2>/dev/null || true
echo "P1_F1_WINDOW_RC:$ARM_RC"
# f1_merged_serve always performs normal teardown and post-health itself.
# A forced monitor stop is a failure even if an API process exited cleanly.
git -C "$FORK" rev-parse HEAD > "$OUT/fork-head-after.txt"
git -C "$FORK" diff --binary HEAD > "$OUT/fork-after.patch"
cmp "$OUT/fork-head-before.txt" "$OUT/fork-head-after.txt" > "$OUT/fork-head-check.txt" 2>&1
HEAD_RC=$?
cmp "$OUT/fork-before.patch" "$OUT/fork-after.patch" > "$OUT/fork-diff-check.txt" 2>&1
DIFF_RC=$?
B70_AGENT=P1 "$R/bin/cpu-run" python3 "$H/d1_source_identity.py" verify \
  "$OUT/source-identity.json" > "$OUT/source-after.log" 2>&1
SOURCE_RC=$?
sha256sum -c "$OUT/p1-identity.sha256" > "$OUT/p1-identity-after.txt" 2>&1
P1_SOURCE_RC=$?
if [ -s "$OUT/fixture-before.sha256" ]; then
  sha256sum -c "$OUT/fixture-before.sha256" > "$OUT/fixture-after.txt" 2>&1
  FIXTURE_RC=$?
else
  [ -s "$P1_PREFILL_FIXTURES" ]; FIXTURE_RC=$?
  sha256sum "$P1_PREFILL_FIXTURES" > "$OUT/fixture-created.sha256" 2>/dev/null || true
fi
PAIR_RC=0
if [ "$FLAG" = 1 ]; then
  CONTROL="$B/p1/full-$CONTROL_TAG"
  cmp "$CONTROL/source-identity.json" "$OUT/source-identity.json" > "$OUT/pair-source-check.txt" 2>&1 || PAIR_RC=1
  cmp "$CONTROL/p1-identity.sha256" "$OUT/p1-identity.sha256" > "$OUT/pair-p1-check.txt" 2>&1 || PAIR_RC=1
  python3 - "$B/f1/merged-$CONTROL_TAG/p1-prefill.json" "$AOUT/p1-prefill.json" > "$OUT/pair-fixture-check.txt" 2>&1 <<'PYPAIR'
import json,sys
control,treatment=(json.load(open(p)) for p in sys.argv[1:])
assert control['arm']=='0' and treatment['arm']=='1'
assert control['ok'] and treatment['ok']
assert control['fixtures_sha256']==treatment['fixtures_sha256']
assert [(r['target'],r['prompt_sha256'],r['prompt_tokens']) for r in control['rows']]==[(r['target'],r['prompt_sha256'],r['prompt_tokens']) for r in treatment['rows']]
print('P1_PAIR_SOURCES_FIXTURES_IDENTITY_OK')
PYPAIR
  [ "$?" = 0 ] || PAIR_RC=1
fi
ACTIVATION_RC=1
if [ -f "$AOUT/serve.log" ] && rg -q "P1_BUCKET_APPLIED card=1 selected=$FLAG " "$AOUT/serve.log"; then
  ACTIVATION_RC=0
fi
printf 'P1_CHECK_RC head=%s diff=%s source=%s p1_source=%s fixtures=%s activation=%s\n' \
  "$HEAD_RC" "$DIFF_RC" "$SOURCE_RC" "$P1_SOURCE_RC" "$FIXTURE_RC" "$ACTIVATION_RC"
python3 - "$AOUT" "$FLAG" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]); flag=sys.argv[2]
greedy=json.loads((p/'greedy.json').read_text())
assert greedy[0]['first_ok'] and all(row['match'] for row in greedy[1:])
report=json.loads((p/'p1-prefill.json').read_text())
assert report['ok'] and report['arm']==flag
assert len(report['rows'])==3
assert all(len(row['requests'])==4 for row in report['rows'])
assert all(r['text'] and r['ttft_ms'] is not None for row in report['rows'] for r in row['requests'])
print('P1_MATCHED_ARM_CONTENT_IDENTITY_OK',flag)
PY
CONTENT_RC=$?
rg -q 'xpu-health: HEALTHY' "$AOUT/health-after.txt"
HEALTH_RC=$?
rg -q 'SERVE_RC:0' "$AOUT/window.log"
SERVE_RC=$?
if [ "$ARM_RC" = 0 ] && [ "$HEAD_RC" = 0 ] && [ "$DIFF_RC" = 0 ] && \
   [ "$SOURCE_RC" = 0 ] && [ "$P1_SOURCE_RC" = 0 ] && [ "$FIXTURE_RC" = 0 ] && \
   [ "$ACTIVATION_RC" = 0 ] && [ "$CONTENT_RC" = 0 ] && [ "$HEALTH_RC" = 0 ] && \
   [ "$SERVE_RC" = 0 ] && [ "$PAIR_RC" = 0 ] && [ ! -e "$OUT/no-output-stop.txt" ]; then
  echo "P1_ARM_PASSED:$FLAG" | tee "$OUT/arm-passed"
  exit 0
fi
exit 1
