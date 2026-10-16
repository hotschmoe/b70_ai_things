#!/usr/bin/env bash
# Root-authorized TWO selected offline HC decodes. Run under cpu-run.
set -euo pipefail
[[ $(id -u) == 1000 ]]
HERE=$(cd -- "$(dirname -- "$0")" && pwd)
AUDIT="$HERE/selected"
IMAGE=sha256:e41d59c0940e1bd2f6f64c9d5d2d7452b77812b0a8407fac8d67cacec5d7fdd2
python3 - "$AUDIT" <<'PY'
import sys,json,hashlib
from pathlib import Path
p=Path(sys.argv[1]);f=json.loads((p/'facts.json').read_text())
assert f['status']=='PASS_SELECTED_TWO_STATIC_BINDINGS_DECODE_REQUIRED'
for arm in ['up','reduce']:
 b=(p/(arm+'-shortname.zebin')).read_bytes();x=f['selected'][arm]
 assert hashlib.sha256(b).hexdigest()==x['short_alias']['sha256']
 assert x['short_alias']['text_byte_exact'] and x['short_alias']['only_selected_section_name_changed']
 assert not (p/(arm+'-isa')).exists(),'Preserve existing decode; no retry'
PY
[[ $(docker image inspect --format '{{.Id}}' "$IMAGE") == "$IMAGE" ]]
docker image inspect "$IMAGE" > "$AUDIT/decoder-image.json"
mkdir "$AUDIT/up-isa" "$AUDIT/reduce-isa" "$AUDIT/decode-tmp"
docker run --rm --network none --cpus 4 -u "$(id -u):$(id -g)" \
 --cap-drop ALL --security-opt no-new-privileges \
 -e TMPDIR=/out/decode-tmp -e OMP_NUM_THREADS=4 -e MKL_NUM_THREADS=4 \
 -v "$AUDIT/up-shortname.zebin":/input/up.zebin:ro \
 -v "$AUDIT/reduce-shortname.zebin":/input/reduce.zebin:ro \
 -v "$AUDIT":/out --entrypoint bash "$IMAGE" -c '
source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
set -euo pipefail
[[ ! -e /dev/dri ]]
[[ $(command -v ocloc) == /usr/bin/ocloc ]]
[[ $(sha256sum /usr/bin/ocloc | cut -d " " -f 1) == 7a418c1e7dfc62b6fb3895bea111647fa3e1ff600bb69f6be2a3114b90c41eac ]]
for arm in up reduce; do
 d=/out/$arm-isa
 command -v ocloc > "$d/tool-path.txt"
 sha256sum /usr/bin/ocloc > "$d/tool.sha256"
 sha256sum "/input/$arm.zebin" > "$d/input-before.sha256"
 mkdir "$d/disasm"
 set +e
 ocloc disasm -file "/input/$arm.zebin" -device bmg-g31 -dump "$d/disasm" > "$d/ocloc.stdout" 2> "$d/ocloc.stderr"
 code=$?
 printf "%s\n" "$code" > "$d/decoder.exit"
 set -e
 [[ $code == 0 ]]
 sha256sum "/input/$arm.zebin" > "$d/input-after.sha256"
 cmp "$d/input-before.sha256" "$d/input-after.sha256"
 [[ -s "$d/disasm/.text.hc_$arm.asm" ]]
 sha256sum "$d/disasm/.text.hc_$arm.asm" > "$d/asm.sha256"
 printf "%s\n" DECODE_COMPLETE_ROOT_MANUAL_ISA_REVIEW_PENDING > "$d/status.txt"
done
'
