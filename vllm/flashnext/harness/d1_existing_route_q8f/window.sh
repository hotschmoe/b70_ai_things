#!/usr/bin/env bash
# PREPARATION ONLY. Root grants each serve separately; never self-queue pair.
# Bothcards, originalFA/c7b,131072/chunk12288, F1EPc retained. ~8min per arm.
set -uo pipefail
R=/mnt/vm_8tb/github/b70_ai_things
B=/mnt/vm_8tb/b70/build/vllm-flashnext
H=$R/vllm/flashnext/harness
S="$(cd "$(dirname "$0")" && pwd)"
ARM=${ARM:?control or routeq8f}
TAG=${TAG:?fresh tag required}
[[ "$ARM" = control || "$ARM" = routeq8f ]] || exit 2
[[ "$TAG" =~ ^[A-Za-z0-9_-]+$ ]] || exit 2
WORK=$B/d1/existing_route_q8f
OUT=$WORK/$TAG
AOUT=$B/f1/merged-$TAG
[[ ! -e "$OUT" && ! -e "$AOUT" ]] || { echo 'REFUSE_EXISTING_OUTPUT'; exit 5; }
mkdir -p "$OUT"
exec > >(tee "$OUT/window.log") 2>&1
export B70_AGENT=D1
# These are the SAME retained recipes, libraries and clients for both arms.
export O4_RECIPE=$H/o4_recipe.env F1_DELTA=$H/f1_prefill_delta.env
export MAX_LEN=131072 MAX_BATCHED=12288 DECODE=64 PREFILL=1024
export F1_LENGTHS=1024 F1_COHERENCE_LEN=1024 F1_REPEAT=1 F1_DECODE=0
export RUN_PROMO_GATE=1 SKIP_HEALTH_BEFORE=0
export F1_API_WRAPPER=$S/api.py F1_POST_PROBE_SCRIPT=$H/d1_lmx_callback.sh
export F1_PYPATH_PRE="$S:$H" SERVE_ARGS_EXTRA=''
unset O4_FA_OVERLAY N1_REFERENCE
export O4_SERVE_EXTRA="B70_GDN_BA_DETERMINISTIC=0 B70_D1_FA_PLAN=0 B70_D1_BRANCH_OBSERVE=1 B70_D1_BRANCH_DIR=$AOUT/branches"
if [[ "$ARM" = routeq8f ]]; then
  export O4_SERVE_EXTRA="$O4_SERVE_EXTRA B70_EP_ROUTE=1 B70_DECODE_HC_Q8F=1"
fi
printf '%s\n' "ARM=$ARM" "TAG=$TAG" "O4_SERVE_EXTRA=$O4_SERVE_EXTRA" \
  'FA=c7b original; no FA8 overlay' 'HC_FUSE absent, retained default' > "$OUT/command.txt"
# Exact measured source-to-binary inputs; no Torch imports/initialization here.
python3 "$S/preflight.py" > "$OUT/preflight.json" || exit 3
python3 "$S/make_snapshot.py" --out "$OUT" || exit 3
mkdir -p "$AOUT/branches"
sha256sum "$S/"*.py "$S/window.sh" "$S/README.md" "$OUT/serve.sh" > "$OUT/private-before.sha256"
"$R/bin/cpu-run" python3 "$H/d1_source_identity.py" snapshot "$OUT/source-before.json" || exit 3
bash "$OUT/serve.sh" > "$OUT/serve-window.log" 2>&1
SERVE=$?
"$R/bin/cpu-run" python3 "$H/d1_source_identity.py" verify "$OUT/source-before.json" > "$OUT/source-after.json"
SOURCE=$?
sha256sum --check "$OUT/private-before.sha256" > "$OUT/private-after.txt" 2>&1
PRIVATE=$?
printf 'HARNESS_RC:%s SOURCE_RC:%s PRIVATE_RC:%s\n' "$SERVE" "$SOURCE" "$PRIVATE" | tee "$OUT/status.txt"
python3 "$S/outcome.py" --aout "$AOUT" --outer "$OUT" --arm "$ARM" > "$OUT/outcome.log" 2>&1
OPERATIONAL=$?
echo "OPERATIONAL_RC:$OPERATIONAL"
echo "WINDOW_DONE $(date -u +%FT%TZ)"
# Strict repeat/promo failure stays NONZERO. Root separately audits actual
# client RCs, fatalengine, gracefulteardown, posthealth and canonical outputs.
[[ "$SERVE" == 0 && "$SOURCE" == 0 && "$PRIVATE" == 0 && "$OPERATIONAL" == 0 ]]
