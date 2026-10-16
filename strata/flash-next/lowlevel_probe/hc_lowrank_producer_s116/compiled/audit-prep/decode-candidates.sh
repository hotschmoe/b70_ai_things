#!/usr/bin/env bash
# SOURCE ONLY. Separate root GO after audit/control bindings are read. TWO candidates.
set -euo pipefail
[[ $# == 2 && $(id -u) == 1000 ]]
AUDIT=$(realpath -- "$1")
EXPECTED_FACTS_SHA=$2
[[ $EXPECTED_FACTS_SHA =~ ^[0-9a-f]{64}$ ]]
IMAGE=sha256:e41d59c0940e1bd2f6f64c9d5d2d7452b77812b0a8407fac8d67cacec5d7fdd2
python3 - "$AUDIT" "$EXPECTED_FACTS_SHA" <<'PY'
from pathlib import Path
import json,hashlib,sys
p=Path(sys.argv[1]);raw=(p/'facts.json').read_bytes();assert hashlib.sha256(raw).hexdigest()==sys.argv[2]
f=json.loads(raw);assert f['status']=='PASS_SIX_STATIC_BINDINGS_STRICT_ORIGINAL_CONTROL_REUSE_CANDIDATE_MANUAL_ISA_PENDING'
assert all(x['PASS']for x in f['original_control_comparison'].values())
for arm in ['candidate_reduce','candidate_up']:
 a=f['candidate_aliases'][arm];assert a['text_byte_exact']and a['only_section_name_changed']
 assert hashlib.sha256((p/(arm+'-shortname.zebin')).read_bytes()).hexdigest()==a['sha256']
assert not(p/'candidate-isa').exists(),'Preserve prior decode, no automatic retry'
PY
[[ $(docker image inspect --format '{{.Id}}' "$IMAGE") == "$IMAGE" ]]
mkdir "$AUDIT/candidate-isa"
docker image inspect "$IMAGE" > "$AUDIT/candidate-isa/image.json"
# CPU lease wraps this script. Read-only inputs; no device, network or root privileges.
docker run --rm --network none --cpus 4 -u "$(id -u):$(id -g)" \
 --cap-drop ALL --security-opt no-new-privileges \
 -e TMPDIR=/out/tmp -e OMP_NUM_THREADS=4 -e MKL_NUM_THREADS=4 \
 -v "$AUDIT/candidate_reduce-shortname.zebin":/input/candidate_reduce.zebin:ro \
 -v "$AUDIT/candidate_up-shortname.zebin":/input/candidate_up.zebin:ro \
 -v "$AUDIT/candidate-isa":/out --entrypoint bash "$IMAGE" -c '
source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
set -euo pipefail
[[ ! -e /dev/dri && $(command -v ocloc) == /usr/bin/ocloc ]]
[[ $(sha256sum /usr/bin/ocloc | cut -d " " -f 1) == 7a418c1e7dfc62b6fb3895bea111647fa3e1ff600bb69f6be2a3114b90c41eac ]]
mkdir /out/tmp
for arm in candidate_reduce candidate_up; do
 d=/out/$arm
 mkdir "$d" "$d/disasm"
 command -v ocloc > "$d/tool-path.txt"
 sha256sum /usr/bin/ocloc > "$d/tool.sha256"
 sha256sum "/input/$arm.zebin" > "$d/input-before.sha256"
 set +e
 ocloc disasm -file "/input/$arm.zebin" -device bmg-g31 -dump "$d/disasm" > "$d/ocloc.stdout" 2> "$d/ocloc.stderr"
 code=$?
 printf "%s\n" "$code" > "$d/decoder.exit"
 set -e
 [[ $code == 0 ]]
 sha256sum "/input/$arm.zebin" > "$d/input-after.sha256"
 cmp "$d/input-before.sha256" "$d/input-after.sha256"
 [[ -s "$d/disasm/.text.s116_$arm.asm" ]]
 sha256sum "$d/disasm/.text.s116_$arm.asm" > "$d/asm.sha256"
done
printf "%s\n" TWO_CANDIDATE_DECODES_COMPLETE_MANUAL_FP_ORDER_OWNERSHIP_RESOURCES_REVIEW_REQUIRED > /out/status.txt
'
