#!/usr/bin/env bash
# Optional ONE selected offline decode after root reads actual binding. No GPU.
set -euo pipefail
[[ $# == 1 && $(id -u) == 1000 ]]
AUDIT=$(realpath -- "$1")
IMAGE=sha256:e41d59c0940e1bd2f6f64c9d5d2d7452b77812b0a8407fac8d67cacec5d7fdd2
python3 - "$AUDIT" <<'PY'
from pathlib import Path
import sys,json,hashlib
p=Path(sys.argv[1]);raw=(p/'facts.json').read_bytes();f=json.loads(raw)
assert f['status']=='PASS_ONE_Q5_C2_STATIC_BINDING_HOST_REVIEW_PENDING_ISA_UNKNOWN'
assert f['alias']['only_section_name_changed']and f['alias']['text_data_relocations_unchanged']
assert hashlib.sha256((p/'q5-c2-shortname.zebin').read_bytes()).hexdigest()==f['alias']['sha256']
assert not(p/'isa').exists(),'Preserve existing decode; no automatic retry'
(p/'decoder-input-facts.json').write_text(json.dumps({'facts_sha256':hashlib.sha256(raw).hexdigest(),'selected':f['selected']['kernel'],'original_image_sha256':f['selected']['image_sha256'],'alias_sha256':f['alias']['sha256'],'images':1},indent=2)+'\n')
PY
[[ $(docker image inspect --format '{{.Id}}' "$IMAGE") == "$IMAGE" ]]
mkdir "$AUDIT/isa"
docker image inspect "$IMAGE" > "$AUDIT/isa/image.json"
docker run --rm --network none --cpus 4 -u "$(id -u):$(id -g)" \
 --cap-drop ALL --security-opt no-new-privileges \
 -e TMPDIR=/out/tmp -e OMP_NUM_THREADS=4 -e MKL_NUM_THREADS=4 \
 -v "$AUDIT/q5-c2-shortname.zebin":/input/q5-c2.zebin:ro \
 -v "$AUDIT/isa":/out --entrypoint bash "$IMAGE" -c '
source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
set -euo pipefail
[[ ! -e /dev/dri && $(command -v ocloc) == /usr/bin/ocloc ]]
[[ $(sha256sum /usr/bin/ocloc | cut -d " " -f 1) == 7a418c1e7dfc62b6fb3895bea111647fa3e1ff600bb69f6be2a3114b90c41eac ]]
mkdir /out/tmp /out/disasm
command -v ocloc > /out/tool-path.txt
sha256sum /usr/bin/ocloc > /out/tool.sha256
sha256sum /input/q5-c2.zebin > /out/input-before.sha256
set +e
ocloc disasm -file /input/q5-c2.zebin -device bmg-g31 -dump /out/disasm > /out/ocloc.stdout 2> /out/ocloc.stderr
code=$?
printf "%s\n" "$code" > /out/decoder.exit
set -e
[[ $code == 0 ]]
sha256sum /input/q5-c2.zebin > /out/input-after.sha256
cmp /out/input-before.sha256 /out/input-after.sha256
[[ -s /out/disasm/.text.s122_q5_c2.asm ]]
sha256sum /out/disasm/.text.s122_q5_c2.asm > /out/asm.sha256
printf "%s\n" ONE_DECODE_COMPLETE_LOWERING_REVIEW_REQUIRED_NO_CANDIDATE_GO > /out/status.txt
'
