#!/bin/bash
# Prepared only; both cards. Parent explicit authorization required to execute.
set -uo pipefail
REPO=/mnt/vm_8tb/github/b70_ai_things
B=/mnt/vm_8tb/b70/build/vllm-flashnext
D=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
H=$REPO/vllm/flashnext/harness
TAG="${TAG:?}"
OUT="$B/f1/merged-$TAG"
[ ! -e "$OUT" ] || { echo "Refuse existing output $OUT"; exit 2; }
mkdir -p "$OUT"
python3 "$D/d1_build_n1_diagnostic.py" --out "$OUT/private" || exit $?
python3 "$H/d1_source_identity.py" snapshot "$OUT/source-before.json" || exit $?
sha256sum "$B/n1/_C.so" "$REPO/vllm/flashnext/plugins/b70_n1_xpu/b70_n1_xpu/__init__.py" \
  > "$OUT/n1-before.sha256"
export TAG MAX_LEN=131072 MAX_BATCHED=12288 DECODE=64 PREFILL=1024
export O4_RECIPE="$H/o4_recipe.env" F1_DELTA="$H/f1_prefill_delta.env"
export RUN_PROMO_GATE=1 F1_LENGTHS=8192 F1_COHERENCE_LEN=8192 F1_REPEAT=2 F1_DECODE=0
# Prior N1 F1 probe had no extra needles; promo gate already has 8K/32K.
export F1_NEEDLE_LENS=''
export F1_API_WRAPPER="$H/d1_api.py"
export F1_PYPATH_PRE="$D:$REPO/vllm/flashnext/plugins/b70_n1_xpu"
export SERVE_ARGS_EXTRA='--worker-extension-cls d1_liveness.LivenessExtension'
export O4_SERVE_EXTRA="B70_D1_FA_PLAN=0 B70_GDN_BA_DETERMINISTIC=1 B70_N1_SO=$B/n1/_C.so B70_D1_LIVENESS_DIR=$OUT/d1-liveness"
export F1_POST_PROBE_SCRIPT="$D/d1_first_n1_callback.sh"
python3 - "$OUT/n1-liveness-env.json" <<'PY'
import json,os,sys
from pathlib import Path
keys=['TAG','MAX_LEN','MAX_BATCHED','DECODE','PREFILL','O4_RECIPE','F1_DELTA','RUN_PROMO_GATE','F1_LENGTHS','F1_COHERENCE_LEN','F1_REPEAT','F1_DECODE','F1_NEEDLE_LENS','F1_API_WRAPPER','F1_PYPATH_PRE','SERVE_ARGS_EXTRA','O4_SERVE_EXTRA','F1_POST_PROBE_SCRIPT']
Path(sys.argv[1]).write_text(json.dumps({k:os.environ.get(k) for k in keys},indent=2)+'\n')
PY
bash "$OUT/private/serve.sh"
window_rc=$?
python3 "$D/d1_classify_liveness.py" "$OUT" > "$OUT/d1-localization-summary.txt"
python3 "$H/d1_source_identity.py" verify "$OUT/source-before.json" > "$OUT/source-after-check.json"
source_rc=$?
sha256sum -c "$OUT/n1-before.sha256" > "$OUT/n1-after-check.txt" 2>&1
n1_rc=$?
python3 - "$OUT/private/private-manifest.json" > "$OUT/private-after-check.json" <<'PY'
import hashlib,json,sys
from pathlib import Path
data=json.loads(Path(sys.argv[1]).read_text())
bad=[p for p,sha in data['sha256'].items() if not Path(p).is_file() or hashlib.sha256(Path(p).read_bytes()).hexdigest()!=sha]
print(json.dumps(dict(bad=bad)))
raise SystemExit(bool(bad))
PY
private_rc=$?
printf 'D1_WINDOW_RC:%s SOURCE_RC:%s N1_RC:%s PRIVATE_RC:%s\n' "$window_rc" "$source_rc" "$n1_rc" "$private_rc" | tee "$OUT/d1-status.txt"
if [ "$window_rc" != 0 ] || [ "$source_rc" != 0 ] || [ "$n1_rc" != 0 ] || [ "$private_rc" != 0 ]; then exit 1; fi
