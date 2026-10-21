#!/bin/bash
# PREPARED ONLY: parent explicit both-card GPU grant required to execute.
set -uo pipefail
R=/mnt/vm_8tb/github/b70_ai_things
B=/mnt/vm_8tb/b70/build/vllm-flashnext
H=$R/vllm/flashnext/harness
N=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TAG="${TAG:?}"
OUT="$B/f1/merged-$TAG"
[ ! -e "$OUT" ] || { echo 'Refuse existing native-capture output'; exit 2; }
mkdir -p "$OUT"
STATE="$B/d1/native_capture/setup-r1"
python3 - "$STATE" <<'PY'
import json,subprocess,sys
from pathlib import Path
p=Path(sys.argv[1])
assert json.loads((p/'attach/result.json').read_text())['result']=='PASS'
assert json.loads((p/'attach-early-r2/result.json').read_text())['result']=='PASS'
assert not json.loads((p/'package-delta.json').read_text())['changed']
expected=json.loads((p/'debug-image.json').read_text())['Id']
actual=subprocess.check_output(['docker','image','inspect','vllm-flashnext-xpu:d1-gdb-r1','--format','{{.Id}}'],text=True).strip()
assert actual==expected, 'debug image changed since CPU qualification'
PY
[ "$?" = 0 ] || exit 3
python3 "$N/make_native_snapshot.py" --out "$OUT/private" || exit $?
python3 "$H/d1_source_identity.py" snapshot "$OUT/source-before.json" || exit $?
sha256sum "$B/n1/_C.so" > "$OUT/n1-before.sha256"
docker image inspect vllm-flashnext-xpu:d1-gdb-r1 --format '{{.Id}}' > "$OUT/debug-image-before.id"
export TAG MAX_LEN=131072 MAX_BATCHED=12288 DECODE=64 PREFILL=1024
export O4_RECIPE="$H/o4_recipe.env" F1_DELTA="$H/f1_prefill_delta.env"
export RUN_PROMO_GATE=1 F1_LENGTHS=8192 F1_COHERENCE_LEN=8192 F1_REPEAT=2 F1_DECODE=0 F1_NEEDLE_LENS=''
export F1_API_WRAPPER="$H/d1_api.py"
export F1_PYPATH_PRE="$OUT/private/native-overlay:$H:$R/vllm/flashnext/plugins/b70_n1_xpu"
export SERVE_ARGS_EXTRA='--worker-extension-cls d1_liveness.LivenessExtension'
export O4_SERVE_EXTRA="B70_D1_FA_PLAN=0 B70_GDN_BA_DETERMINISTIC=1 B70_N1_SO=$B/n1/_C.so B70_D1_LIVENESS_DIR=$OUT/d1-liveness"
export F1_POST_PROBE_SCRIPT="$H/d1_first_n1_callback.sh"
bash "$OUT/private/serve.sh"
window_rc=$?
python3 "$H/d1_classify_liveness.py" "$OUT" > "$OUT/d1-localization-summary.txt"
python3 "$H/d1_source_identity.py" verify "$OUT/source-before.json" > "$OUT/source-after-check.json"
source_rc=$?
sha256sum -c "$OUT/n1-before.sha256" > "$OUT/n1-after-check.txt" 2>&1
n1_rc=$?
docker image inspect vllm-flashnext-xpu:d1-gdb-r1 --format '{{.Id}}' > "$OUT/debug-image-after.id"
cmp "$OUT/debug-image-before.id" "$OUT/debug-image-after.id"
image_rc=$?
python3 - "$OUT/private/native-manifest.json" > "$OUT/native-after-check.json" <<'PY'
import hashlib,json,sys
from pathlib import Path
data=json.loads(Path(sys.argv[1]).read_text())
bad=[p for p,sha in data['sha256'].items() if not Path(p).is_file() or hashlib.sha256(Path(p).read_bytes()).hexdigest()!=sha]
print(json.dumps(dict(bad=bad)));raise SystemExit(bool(bad))
PY
native_rc=$?
printf 'WINDOW_RC:%s SOURCE_RC:%s N1_RC:%s IMAGE_RC:%s NATIVE_RC:%s\n' "$window_rc" "$source_rc" "$n1_rc" "$image_rc" "$native_rc" | tee "$OUT/native-status.txt"
if [ "$window_rc" != 0 ] || [ "$source_rc" != 0 ] || [ "$n1_rc" != 0 ] || [ "$image_rc" != 0 ] || [ "$native_rc" != 0 ]; then exit 1; fi
