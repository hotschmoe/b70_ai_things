#!/usr/bin/env bash
# One leased quiet card1 ABBA window, one health pair, stopfirstfail/no retry/reset.
set -uo pipefail
P=/mnt/vm_8tb/b70/results/flashnext_ista_gsq_rco_iq3xxs_strata_cachefix_20261014/hc-more-groups-s73-prep
REPO=/mnt/vm_8tb/github/b70_ai_things
OUT=${1:?new run directory}
IMG=sha256:e41d59c0940e1bd2f6f64c9d5d2d7452b77812b0a8407fac8d67cacec5d7fdd2
HEALTH_IMG=b70-sglang-xpu-int8-runtime@sha256:adc915d266eaa74f7bea164d97cb7870b04dd7eb4c613952c56f4fbff1584a78
python3 "$P/host_closure.py" pre "$OUT" || exit 2
for label in a1 b1 b2 a2;do
  if docker container inspect "strata-hc-more-s73-$label" > "$OUT/preexisting-$label.json" 2> "$OUT/preexisting-$label.stderr";then echo "FAIL preserve preexisting container before anyhealth";exit 2;fi
done
"$REPO/bin/xpu-health" --card 1 --img "$HEALTH_IMG" --timeout 400 > "$OUT/pre-health.log" 2>&1
pre=$?;printf '%s\n' "$pre" > "$OUT/pre-health.exit";[ "$pre" = 0 ] || exit "$pre"
result=0
for label in a1 b1 b2 a2;do
  mode=0;[[ "$label" = b* ]] && mode=1
  DIR="$OUT/$label";mkdir "$DIR" || { result=2;break; }
  NAME="strata-hc-more-s73-$label";OWN_CID=""
  printf '%s\n' "$mode" > "$DIR/mode"
  timeout --signal=TERM --kill-after=15s 300s docker run --rm --name "$NAME" --cidfile "$DIR/attempt.cid" --network none --device /dev/dri --group-add "$(getent group render | cut -d: -f3)" -u "$(id -u):$(id -g)" -e ZE_AFFINITY_MASK=1 -e ONEAPI_DEVICE_SELECTOR='level_zero:*' -e SYCL_CACHE_PERSISTENT=0 -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 -e STRATA_GR_MORE_GROUPS="$mode" -e STRATA_GR_LARGE_GRF=0 -e STRATA_GR_DOWN_SLICED=1 -e STRATA_GR_UP_COLS=16 -e STRATA_GR_UP_ILP=0 -e STRATA_GR_V3=0 -e STRATA_GR_NORM_SPLIT=1 -v "$P/build":/probe:ro -v "$DIR":/out --entrypoint bash "$IMG" -c '
source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
set -euo pipefail
exec env -u STRATA_HC_Q8 -u STRATA_QFUSE /probe/hc_more_oracle /out/snapshots
' > "$DIR/oracle.log" 2> "$DIR/oracle.stderr"
  run=$?;printf '%s\n' "$run" > "$DIR/oracle.exit"
  if [ -f "$DIR/attempt.cid" ];then read -r OWN_CID < "$DIR/attempt.cid" || true;[[ "$OWN_CID" =~ ^[0-9a-f]{64}$ ]] || OWN_CID="";fi
  if { [ "$run" = 124 ] || [ "$run" = 137 ]; } && [ -n "$OWN_CID" ];then docker stop --time 5 "$OWN_CID" > "$DIR/timeout-stop.stdout" 2> "$DIR/timeout-stop.stderr";printf '%s\n' "$?" > "$DIR/timeout-stop.exit";fi
  removed=0
  while [ "$removed" != 1 ];do
    docker ps -a --no-trunc --format '{{.ID}} {{.Names}}' > "$DIR/containers-after.txt"
    ps_rc=$?;printf '%s\n' "$ps_rc" >> "$DIR/enumeration-exits.txt"
    if [ "$ps_rc" = 0 ];then
      removed=1
      while read -r seen_id seen_name;do
        if [ "$seen_name" = "$NAME" ] || { [ -n "$OWN_CID" ] && [ "$seen_id" = "$OWN_CID" ]; };then removed=0;fi
      done < "$DIR/containers-after.txt"
    fi
    if [ "$removed" != 1 ];then
      echo 'S73 holding quietCPU/card1 leases; createdcontainer removal unconfirmed; no nextprocess/health'
      if [ -n "$OWN_CID" ];then docker kill "$OWN_CID" >> "$DIR/cleanup.stdout" 2>> "$DIR/cleanup.stderr" || true;docker wait "$OWN_CID" >> "$DIR/cleanup.stdout" 2>> "$DIR/cleanup.stderr" || true;docker rm "$OWN_CID" >> "$DIR/cleanup.stdout" 2>> "$DIR/cleanup.stderr" || true;fi
      sleep 5
    fi
  done
  printf '%s\n' 1 > "$DIR/container-removed"
  if [ "$run" != 0 ];then result=$run;break;fi
done
# Only reached with every CREATED attempt confirmed absent; numeric failures still health.
"$REPO/bin/xpu-health" --card 1 --img "$HEALTH_IMG" --timeout 400 > "$OUT/post-health.log" 2>&1
post=$?;printf '%s\n' "$post" > "$OUT/post-health.exit"
compare=2
if [ "$result" = 0 ];then
  python3 "$P/compare.py" "$OUT" > "$OUT/analysis.stdout" 2> "$OUT/analysis.stderr";compare=$?
  if [ "$compare" = 0 ];then python3 "$P/host_closure.py" post "$OUT" > "$OUT/host-post.stdout" 2> "$OUT/host-post.stderr";compare=$?;fi
fi
python3 - "$OUT" "$pre" "$post" "$result" "$compare" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]);pre,post,run,compare=map(int,sys.argv[2:]);rows=[]
for name in ['a1','b1','b2','a2']:
 d=p/name
 if (d/'oracle.exit').exists():rows.append({'label':name,'oracle_exit':int((d/'oracle.exit').read_text()),'created_container_removed':(d/'container-removed').read_text().strip()=='1'})
x={'status':'PASS_PRIMITIVE_ONLY' if pre==post==run==compare==0 and len(rows)==4 else 'FAIL_INSPECT_RAW','pre_health_exit':pre,'post_health_exit':post,'run_exit':run,'compare_exit':compare,'processes':rows,'retries':0,'speed_qualified':False,'serving_qualified':False}
with (p/'closure.json').open('x',encoding='ascii') as f:json.dump(x,f,indent=2);f.write('\n')
PY
close=$?
[ "$result" = 0 ] || exit "$result"
[ "$post" = 0 ] || exit "$post"
[ "$compare" = 0 ] || exit "$compare"
exit "$close"
