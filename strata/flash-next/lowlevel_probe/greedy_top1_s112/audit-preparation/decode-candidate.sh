#!/usr/bin/env bash
# SOURCE ONLY; separate root GO after selected artifact audit. ONE selected candidate image.
set -euo pipefail
[[ $# == 2 && $(id -u) == 1000 ]]
AUDIT=$(realpath -- "$1")
EXPECTED_ALIAS_SHA=$2
[[ $EXPECTED_ALIAS_SHA =~ ^[0-9a-f]{64}$ ]]
REPO=/mnt/vm_8tb/github/b70_ai_things
IMAGE=sha256:e41d59c0940e1bd2f6f64c9d5d2d7452b77812b0a8407fac8d67cacec5d7fdd2
python3 - "$AUDIT" "$EXPECTED_ALIAS_SHA" <<'PY'
from pathlib import Path
import sys,json,hashlib
p=Path(sys.argv[1]);f=json.loads((p/'facts.json').read_text())
assert f['status']=='PASS_STATIC_BINDINGS_SELECTED_IMAGES_CANDIDATE_ISA_PENDING_ROOT_GO'
assert hashlib.sha256((p/'candidate-shortname.zebin').read_bytes()).hexdigest()==f['alias_sha256']==sys.argv[2]
assert f['alias_text_unchanged']
PY
mkdir "$AUDIT/candidate-isa"
mkdir "$AUDIT/candidate-isa/tmp"
[[ $(docker image inspect --format '{{.Id}}' "$IMAGE") == "$IMAGE" ]]
# cpu-run outside this script is required, matching the prior S85/S104 offline decoder.
docker run --rm --network none --cpus 4 -u "$(id -u):$(id -g)" \
 --cap-drop ALL --security-opt no-new-privileges \
 -e TMPDIR=/out/tmp -e OMP_NUM_THREADS=4 -e MKL_NUM_THREADS=4 \
 -v "$AUDIT/candidate-shortname.zebin":/input/candidate.zebin:ro \
 -v "$AUDIT/candidate-isa":/out --entrypoint bash "$IMAGE" -c '
source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
set -euo pipefail
[[ ! -e /dev/dri ]]
command -v ocloc > /out/tool-path.txt
[[ $(command -v ocloc) == /usr/bin/ocloc ]]
sha256sum /usr/bin/ocloc > /out/tool.sha256
sha256sum /input/candidate.zebin > /out/input-before.sha256
mkdir /out/disasm
set +e
ocloc disasm -file /input/candidate.zebin -device bmg-g31 -dump /out/disasm > /out/ocloc.stdout 2> /out/ocloc.stderr
code=$?
printf "%s\n" "$code" > /out/decoder.exit
set -e
[[ $code == 0 ]]
sha256sum /input/candidate.zebin > /out/input-after.sha256
cmp /out/input-before.sha256 /out/input-after.sha256
[[ -s /out/disasm/.text.s112_top1_candidate.asm ]]
sha256sum /out/disasm/.text.s112_top1_candidate.asm > /out/candidate-asm.sha256
printf "%s\n" DECODE_COMPLETE_MANUAL_ORDERING_AND_ARITHMETIC_REVIEW_PENDING > /out/status.txt
'
