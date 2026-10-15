#!/usr/bin/env bash
# Caller owns quiet CPU and physical card0 until confirmed owned teardown.
set -uo pipefail
P=$(cd -- "$(dirname -- "$0")" && pwd)
OUT=${1:?new run directory}
REPO=/mnt/vm_8tb/github/b70_ai_things
PTI=/mnt/vm_8tb/b70/build/strata-ab/pti-gpu
IMAGE=sha256:e41d59c0940e1bd2f6f64c9d5d2d7452b77812b0a8407fac8d67cacec5d7fdd2
HEALTH_IMG=b70-sglang-xpu-int8-runtime@sha256:adc915d266eaa74f7bea164d97cb7870b04dd7eb4c613952c56f4fbff1584a78
NAME=strata-paired-graph-s103
[[ -x "$P/build/paired_graph_s103" && -f "$P/build/runtime.sha256" ]] || exit 2
[[ $(id -u) == 1000 && $(docker image inspect --format '{{.Id}}' "$IMAGE") == "$IMAGE" ]] || exit 2
if docker container inspect "$NAME" > "$OUT/preexisting.json" 2> "$OUT/preexisting.stderr"; then
  echo 'STOP unique name exists; preserve foreign state'; exit 2
fi
uname -r > "$OUT/kernel.txt"
cat /proc/sys/kernel/random/boot_id > "$OUT/boot-id.txt"
[[ $(cat "$OUT/kernel.txt") == 7.1.0-070100-generic ]] || exit 2
"$REPO/bin/xpu-health" --card 0 --img "$HEALTH_IMG" --timeout 400 > "$OUT/pre-health.log" 2>&1
pre=$?; printf '%s\n' "$pre" > "$OUT/pre-health.exit"
[[ $pre == 0 ]] || exit "$pre"
docker create --name "$NAME" --cidfile "$OUT/attempt.cid" --network none \
  --device /dev/dri --group-add "$(getent group render | cut -d: -f3)" -u "$(id -u):$(id -g)" \
  -e ZE_AFFINITY_MASK=0 -e ONEAPI_DEVICE_SELECTOR='level_zero:*' -e SYCL_CACHE_PERSISTENT=0 \
  -e STRATA_HOST_UNCACHED=1 -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 \
  -v "$P":/src:ro -v "$P/build":/probe:ro -v "$PTI":"$PTI":ro -v "$OUT":/out \
  --entrypoint bash "$IMAGE" /src/run-container.sh > "$OUT/create.stdout" 2> "$OUT/create.stderr"
created=$?; printf '%s\n' "$created" > "$OUT/create.exit"
OWN_CID=''
if [[ -f "$OUT/attempt.cid" ]]; then read -r OWN_CID < "$OUT/attempt.cid" || true; fi
[[ "$OWN_CID" =~ ^[0-9a-f]{64}$ ]] || { echo 'STOP no valid created CID; inspect create result'; exit 2; }
[[ $(cat "$OUT/create.stdout") == "$OWN_CID" ]] || { echo 'STOP create identity mismatch'; exit 2; }
run=2
if [[ $created == 0 ]]; then
  docker inspect "$OWN_CID" > "$OUT/created-inspect.json"
  timeout --signal=TERM --kill-after=10s 180s docker start -a "$OWN_CID" > "$OUT/container.log" 2> "$OUT/container.stderr"
  run=$?
fi
printf '%s\n' "$run" > "$OUT/container.exit"
# No workload retry. A timeout cleans up only this created CID.
if [[ $run == 124 || $run == 137 ]]; then
  docker stop --time 5 "$OWN_CID" > "$OUT/timeout-stop.stdout" 2> "$OUT/timeout-stop.stderr"
fi
docker inspect "$OWN_CID" > "$OUT/final-inspect.json" 2> "$OUT/final-inspect.stderr"
inspect_rc=$?
normal=1
if [[ $inspect_rc == 0 ]]; then
  python3 - "$OUT" "$OWN_CID" "$NAME" "$IMAGE" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]);d=json.load(open(p/'final-inspect.json'))[0]
assert d['Id']==sys.argv[2] and d['Name']=='/'+sys.argv[3] and d['Image']==sys.argv[4]
assert d['State']['Running'] is False and d['State']['OOMKilled'] is False and d['State']['ExitCode']==0
PY
  normal=$?
fi
printf '%s\n' "$normal" > "$OUT/normal-process.exit"
docker rm "$OWN_CID" > "$OUT/remove.stdout" 2> "$OUT/remove.stderr"
# As in S98: never release the lease or run health with unconfirmed live CID.
removed=0
while [[ $removed != 1 ]]; do
  docker ps -a --no-trunc --format '{{.ID}} {{.Names}}' > "$OUT/containers-after.txt"
  listed=$?
  if [[ $listed == 0 ]]; then
    python3 - "$OUT/containers-after.txt" "$OWN_CID" "$NAME" <<'PY'
import sys
from pathlib import Path
rows=[line.split()for line in Path(sys.argv[1]).read_text().splitlines()]
assert not any(x[0]==sys.argv[2]or x[1]==sys.argv[3]for x in rows)
PY
    [[ $? != 0 ]] || removed=1
  fi
  [[ $removed != 1 ]] || break
  echo 'S103 HOLD leases: owned removal unconfirmed; cleanup only, no workload retry/health'
  docker kill "$OWN_CID" >> "$OUT/cleanup.stdout" 2>> "$OUT/cleanup.stderr" || true
  docker wait "$OWN_CID" >> "$OUT/cleanup.stdout" 2>> "$OUT/cleanup.stderr" || true
  docker rm "$OWN_CID" >> "$OUT/cleanup.stdout" 2>> "$OUT/cleanup.stderr" || true
  sleep 5
done
"$REPO/bin/xpu-health" --card 0 --img "$HEALTH_IMG" --timeout 400 > "$OUT/post-health.log" 2>&1
post=$?; printf '%s\n' "$post" > "$OUT/post-health.exit"
python3 - "$OUT" "$run" "$pre" "$post" "$normal" "$OWN_CID" "$NAME" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]);run,pre,post,normal=map(int,sys.argv[2:6])
comparison=json.loads((p/'comparison.json').read_text())if(p/'comparison.json').exists()else None
ok=run==pre==post==normal==0 and comparison is not None
v={'status':'PASS_GRAPH_API_ONLY'if ok else'FAIL_OR_HOLD_INSPECT_EXISTING_ATTEMPT','container_exit':run,'pre_health_exit':pre,'post_health_exit':post,'normal_exit_no_OOM':normal==0,'owned_fullCID':sys.argv[6],'owned_name':sys.argv[7],'fullCID_and_NAME_gone':True,'fresh_mode_processes':2 if ok else None,'retries':0,'MTP_numeric_qualified':False,'speed_qualified':False}
(p/'closure.json').write_text(json.dumps(v,indent=2)+'\n',encoding='ascii');print(json.dumps(v))
sys.exit(0 if ok else 2)
PY
