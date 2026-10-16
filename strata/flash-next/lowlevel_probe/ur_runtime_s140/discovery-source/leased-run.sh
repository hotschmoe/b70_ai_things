#!/usr/bin/env bash
# Caller owns quiet CPU and physical card0 until confirmed owned teardown.
set -uo pipefail
P=$(cd -- "$(dirname -- "$0")" && pwd)
BASE=/mnt/vm_8tb/b70/results/flashnext_ista_gsq_rco_iq3xxs_strata_cachefix_20261014/mtp-cross-queue-graph-event-private-prep-v2
OUT=${1:?new run directory}
REPO=/mnt/vm_8tb/github/b70_ai_things
IMAGE=sha256:e41d59c0940e1bd2f6f64c9d5d2d7452b77812b0a8407fac8d67cacec5d7fdd2
HEALTH_IMG=b70-sglang-xpu-int8-runtime@sha256:adc915d266eaa74f7bea164d97cb7870b04dd7eb4c613952c56f4fbff1584a78
NAME=strata-ur-adapter-discovery
# Unknown/live ownership retains the SAME caller leases; ROOT must decide.
# This loop only holds the process. No kill/stop/reset/rm/health/retry is issued.
hold_owned() {
  python3 - "$OUT" "$OWN_CID" "$NAME" "$1" <<'PYHOLD'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]);v={'status':'HOLD_ROOT_DECISION_SAME_LEASES','owned_fullCID':sys.argv[2],'owned_name':sys.argv[3],'reason':sys.argv[4],'automatic_force_recovery_health_retry':False}
(p/'recovery-required.json').write_text(json.dumps(v,indent=2)+'\n',encoding='ascii');print(json.dumps(v),flush=True)
PYHOLD
  while :; do sleep 30; done
}
[[ -x "$BASE/build/cross_queue_graph" && -f "$BASE/build/runtime.sha256" ]] || exit 2
[[ $(id -u) == 1000 && $(docker image inspect --format '{{.Id}}' "$IMAGE") == "$IMAGE" ]] || exit 2
if docker container inspect "$NAME" > "$OUT/preexisting.json" 2> "$OUT/preexisting.stderr"; then
  echo 'STOP unique name exists; preserve foreign state'; exit 2
fi
uname -r > "$OUT/kernel.txt"
cat /proc/sys/kernel/random/boot_id > "$OUT/boot-id.txt"
[[ $(cat "$OUT/kernel.txt") == 7.1.0-070100-generic ]] || exit 2
# ROOT authorized reuse of latest S139 ON both-card + collective posthealth.
# Root supplies the known matching boot and no-intervening-GPU attestation.
[[ ${B70_DISCOVERY_REUSE_BOOT_ID:-} == "$(cat "$OUT/boot-id.txt")" && ${B70_DISCOVERY_NO_INTERVENING_GPU:-0} == 1 ]] || exit 2
python3 - "$OUT" <<'PYREUSE'
import hashlib,json,sys
from pathlib import Path
p=Path(sys.argv[1]);q=Path('/mnt/vm_8tb/b70/results/flashnext_ista_gsq_rco_iq3xxs_strata_cachefix_20261014/q5-head-c6-runtime-ones-s139-serving-prep/mode1')
h=json.loads((q/'post-health.json').read_text());n=json.loads((q/'normal-stop.json').read_text())
assert h['card']['exit']==h['collective']['exit']==0
assert n['status']=='PASS_NORMAL_TEARDOWN'and n['no_force_stop_used']and n['lease_wrapper_exit']==0 and n['all_wrapper_terminal_exits']==[0]
files={str(q/x):{'bytes':(q/x).stat().st_size,'sha256':hashlib.sha256((q/x).read_bytes()).hexdigest()}for x in ['post-health.json','normal-stop.json']}
(p/'reused-prehealth.json').write_text(json.dumps({'files':files,'kernel':(p/'kernel.txt').read_text().strip(),'boot_id':(p/'boot-id.txt').read_text().strip(),'same_boot_and_no_intervening_GPU_root_attested':True,'historical_boot_independently_recorded_in_S139_receipt':False},indent=2)+'\n',encoding='ascii')
PYREUSE
pre=$?; printf '%s\n' "$pre" > "$OUT/pre-health.exit"
[[ $pre == 0 ]] || exit "$pre"
docker create --name "$NAME" --cidfile "$OUT/attempt.cid" --network none \
  --device /dev/dri --group-add "$(getent group render | cut -d: -f3)" -u "$(id -u):$(id -g)" \
  -e ZE_AFFINITY_MASK=0 -e ONEAPI_DEVICE_SELECTOR='level_zero:*' -e SYCL_CACHE_PERSISTENT=0 \
  -e STRATA_HOST_UNCACHED=1 -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 \
  -e SYCL_UR_TRACE=1 -e 'UR_LOG_LOADER=level:info;output:stderr' -e 'UR_LOG_LEVEL_ZERO=level:info;output:stderr' \
  -v "$BASE":/src:ro -v "$BASE/build":/probe:ro -v "$P":/driver:ro -v "$OUT":/out \
  --entrypoint bash "$IMAGE" /driver/run-container.sh > "$OUT/create.stdout" 2> "$OUT/create.stderr"
created=$?; printf '%s\n' "$created" > "$OUT/create.exit"
OWN_CID=''
if [[ -f "$OUT/attempt.cid" ]]; then read -r OWN_CID < "$OUT/attempt.cid" || true; fi
[[ "$OWN_CID" =~ ^[0-9a-f]{64}$ ]] || hold_owned 'create returned no authoritative full CID'
[[ $created == 0 && $(cat "$OUT/create.stdout") == "$OWN_CID" ]] || hold_owned 'create exit/CID mismatch'
docker inspect "$OWN_CID" > "$OUT/created-inspect.json" 2> "$OUT/created-inspect.stderr"
[[ $? == 0 ]] || hold_owned 'created ownership inspect failed'
python3 - "$OUT/created-inspect.json" "$OWN_CID" "$NAME" "$IMAGE" "$(id -u):$(id -g)" <<'PYCREATED'
import json,sys
v=json.load(open(sys.argv[1]))[0]
assert v['Id']==sys.argv[2]and v['Name']=='/'+sys.argv[3]and v['Image']==sys.argv[4]and v['Config']['User']==sys.argv[5]
assert v['State']['Status']=='created'and v['State']['Running']is False and v['State']['Pid']==0
PYCREATED
[[ $? == 0 ]] || hold_owned 'created identity/user/state not authoritative'
# SAME blocking docker-start handle. A hang remains here with both leases.
# No automatic timeout, TERM, kill-after, stop, reset or workload retry.
docker start -a "$OWN_CID" > "$OUT/container.log" 2> "$OUT/container.stderr"
run=$?; printf '%s\n' "$run" > "$OUT/container.exit"
docker inspect "$OWN_CID" > "$OUT/final-inspect.json" 2> "$OUT/final-inspect.stderr"
[[ $? == 0 ]] || hold_owned 'final ownership/state unknown'
python3 - "$OUT/final-inspect.json" "$OWN_CID" "$NAME" "$IMAGE" "$(id -u):$(id -g)" <<'PYSTOPPED'
import json,sys
v=json.load(open(sys.argv[1]))[0]
assert v['Id']==sys.argv[2]and v['Name']=='/'+sys.argv[3]and v['Image']==sys.argv[4]and v['Config']['User']==sys.argv[5]
s=v['State'];assert s['Running']is False and s['Status']in['exited','created']and s['Pid']==0
assert type(s['ExitCode'])is int and type(s['OOMKilled'])is bool
PYSTOPPED
[[ $? == 0 ]] || hold_owned 'owned container live or stopped identity/state unconfirmed'
# Preserve native/diagnostic state BEFORE normal wait/removal, even on exit5.
python3 - "$OUT" "$run" <<'PYDIAG'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]);v=json.loads((p/'final-inspect.json').read_text())[0]
(p/'native-diagnostics.json').write_text(json.dumps({'docker_start_exit':int(sys.argv[2]),'native_state':v['State'],'user':v['Config']['User'],'logs':['container.log','container.stderr'],'automatic_health_or_force':False},indent=2)+'\n',encoding='ascii')
PYDIAG
[[ $? == 0 ]] || hold_owned 'native diagnostic receipt failed'
docker wait "$OWN_CID" > "$OUT/normal-wait.stdout" 2> "$OUT/normal-wait.stderr"
wait_rc=$?; printf '%s\n' "$wait_rc" > "$OUT/normal-wait.exit"
[[ $wait_rc == 0 ]] || hold_owned 'normal docker wait failed'
python3 - "$OUT" <<'PYWAIT'
import json
from pathlib import Path
import sys
p=Path(sys.argv[1]);v=json.loads((p/'final-inspect.json').read_text())[0]
assert (p/'normal-wait.stdout').read_text().strip()==str(v['State']['ExitCode'])
PYWAIT
[[ $? == 0 ]] || hold_owned 'normal wait/native exit disagreement'
# Normal rm of the exact confirmed STOPPED owned CID only. Never force/retry.
docker rm "$OWN_CID" > "$OUT/remove.stdout" 2> "$OUT/remove.stderr"
remove_rc=$?; printf '%s\n' "$remove_rc" > "$OUT/remove.exit"
[[ $remove_rc == 0 && $(cat "$OUT/remove.stdout") == "$OWN_CID" ]] || hold_owned 'normal owned removal unconfirmed'
docker ps -a --no-trunc --format '{{.ID}} {{.Names}}' > "$OUT/containers-after.txt" 2> "$OUT/containers-after.stderr"
listed=$?; printf '%s\n' "$listed" > "$OUT/containers-after.exit"
[[ $listed == 0 ]] || hold_owned 'post-removal full container enumeration unknown'
python3 - "$OUT/containers-after.txt" "$OWN_CID" "$NAME" <<'PYABSENT'
import sys
from pathlib import Path
rows=[line.split()for line in Path(sys.argv[1]).read_text().splitlines()]
assert all(len(x)==2 for x in rows)
assert not any(x[0]==sys.argv[2]or x[1]==sys.argv[3]for x in rows)
PYABSENT
[[ $? == 0 ]] || hold_owned 'full CID or NAME absence unconfirmed'
normal=1
python3 - "$OUT" "$run" <<'PYNORMAL'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]);v=json.loads((p/'final-inspect.json').read_text())[0];s=v['State']
assert int(sys.argv[2])==s['ExitCode']==0 and s['Status']=='exited'and not s['OOMKilled']and not s.get('Dead',False)and not s.get('Error','')
assert json.loads((p/'comparison.json').read_text())['status']=='PASS_RUNTIME_DISCOVERY_ONLY'
for mode in [0]:
 q=p/('mode'+str(mode));assert(q/'leaf.exit').read_text().strip()=='0'
 assert json.loads((q/'closure.json').read_text())['status']=='PASS_ONE_MODE_TWO_QUEUE_ORDERING_ONLY'
PYNORMAL
normal=$?; printf '%s\n' "$normal" > "$OUT/normal-process.exit"
if [[ $normal != 0 ]]; then
  python3 - "$OUT" "$run" "$OWN_CID" "$NAME" <<'PYABNORMAL'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]);v={'status':'ROOT_DIAGNOSTIC_REVIEW_REQUIRED_NO_BLIND_POSTHEALTH','docker_start_exit':int(sys.argv[2]),'owned_fullCID':sys.argv[3],'owned_name':sys.argv[4],'normal_wait_and_rm_completed':True,'fullCID_and_NAME_gone':True,'native_state':'final-inspect.json','diagnostics':'native-diagnostics.json','posthealth_run':False,'automatic_force_reset_retry':False}
(p/'recovery-required.json').write_text(json.dumps(v,indent=2)+'\n',encoding='ascii');print(json.dumps(v))
PYABNORMAL
  exit 2
fi
# Posthealth only after native0 + strict/DOT/analyzer PASS + normal owned closure.
"$REPO/bin/xpu-health" --card 0 --img "$HEALTH_IMG" --timeout 400 > "$OUT/post-health.log" 2>&1
post=$?; printf '%s\n' "$post" > "$OUT/post-health.exit"
python3 - "$OUT" "$run" "$pre" "$post" "$normal" "$OWN_CID" "$NAME" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]);run,pre,post,normal=map(int,sys.argv[2:6])
comparison=json.loads((p/'comparison.json').read_text())if(p/'comparison.json').exists()else None
ok=run==pre==post==normal==0 and comparison is not None
v={'status':'PASS_RUNTIME_DISCOVERY_ONLY'if ok else'FAIL_OR_HOLD_INSPECT_EXISTING_ATTEMPT','container_exit':run,'pre_health_exit':pre,'post_health_exit':post,'normal_exit_no_OOM':normal==0,'owned_fullCID':sys.argv[6],'owned_name':sys.argv[7],'fullCID_and_NAME_gone':True,'fresh_mode_processes':1 if ok else None,'retries':0,'MTP_numeric_qualified':False,'speed_qualified':False}
(p/'closure.json').write_text(json.dumps(v,indent=2)+'\n',encoding='ascii');print(json.dumps(v))
sys.exit(0 if ok else 2)
PY
