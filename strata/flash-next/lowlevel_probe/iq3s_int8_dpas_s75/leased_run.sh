#!/usr/bin/env bash
# Caller holds quiet CPU and physical card0. Exactly one process; no retries/reset.
set -uo pipefail
HERE=/mnt/vm_8tb/b70/results/flashnext_ista_gsq_rco_iq3xxs_strata_cachefix_20261014/iq3s-int8-dpas-s75-prep-v2
EXEC_DIR=/mnt/vm_8tb/b70/results/flashnext_ista_gsq_rco_iq3xxs_strata_cachefix_20261014/iq3s-int8-dpas-s75-prep-v2
OUT=${1:?new existing run directory}
REPO=/mnt/vm_8tb/github/b70_ai_things
IMAGE=sha256:e41d59c0940e1bd2f6f64c9d5d2d7452b77812b0a8407fac8d67cacec5d7fdd2
HEALTH_IMG=b70-sglang-xpu-int8-runtime@sha256:adc915d266eaa74f7bea164d97cb7870b04dd7eb4c613952c56f4fbff1584a78
WEIGHTS=/mnt/cache/b70/qwen3.8-flash-next/ista-gsq-rco/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf
NAME=strata-iq3s-dpas-s75
python3 "$EXEC_DIR/extra_host_closure.py" pre "$OUT" || exit 2
[ -x "$HERE/build/iq3s_dpas_leaf_s75" ] || exit 2
python3 "$HERE/runtime_identity.py" "$HERE" "$OUT" > "$OUT/identity.stdout" 2> "$OUT/identity.stderr"
identity_rc=$?;printf '%s\n' "$identity_rc" > "$OUT/identity.exit"
[ "$identity_rc" = 0 ] || exit "$identity_rc"
if docker container inspect "$NAME" > "$OUT/preexisting-container.json" 2> "$OUT/preexisting-container.stderr";then
  echo 'FAIL unique container exists; preserve it';exit 2
fi
"$REPO/bin/xpu-health" --card 0 --img "$HEALTH_IMG" --timeout 400 > "$OUT/pre-health.log" 2>&1
pre=$?;printf '%s\n' "$pre" > "$OUT/pre-health.exit"
[ "$pre" = 0 ] || exit "$pre"
started=$SECONDS
timeout --signal=TERM --kill-after=15s 300s docker run --rm --name "$NAME" --cidfile "$OUT/attempt.cid" --network none \
  --device /dev/dri --group-add "$(getent group render | cut -d: -f3)" \
  -u "$(id -u):$(id -g)" -e ZE_AFFINITY_MASK=0 -e ONEAPI_DEVICE_SELECTOR='level_zero:*' \
  -e SYCL_CACHE_PERSISTENT=0 -e STRATA_OLD_IQ_MMVQ=0 -e STRATA_IQ_STAGE_GRID_MMVQ=0 \
  -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 \
  -v "$HERE/build":/probe:ro -v "$OUT":/out -v "$WEIGHTS":/weights.gguf:ro \
  --entrypoint bash "$IMAGE" -c '
source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
set -euo pipefail
exec /probe/iq3s_dpas_leaf_s75 /out/leaf /weights.gguf
' > "$OUT/leaf.log" 2> "$OUT/leaf.stderr"
run=$?;printf '%s\n' "$run" > "$OUT/leaf.exit"
printf '%s\n' "$((SECONDS-started))" > "$OUT/leaf.wall-seconds"
# A timeout may leave the named Docker container alive. Stop only this attempt,
# never reset/retry; record removal before post-health and preserve all outputs.
OWN_CID=""
if [ -f "$OUT/attempt.cid" ];then
  read -r OWN_CID < "$OUT/attempt.cid" || true
  [[ "$OWN_CID" =~ ^[0-9a-f]{64}$ ]] || OWN_CID=""
fi
if { [ "$run" = 124 ] || [ "$run" = 137 ]; } && [ -n "$OWN_CID" ];then
  docker stop --time 5 "$OWN_CID" > "$OUT/timeout-stop.stdout" 2> "$OUT/timeout-stop.stderr"
  printf '%s\n' "$?" > "$OUT/timeout-stop.exit"
fi
docker ps -a --no-trunc --format '{{.ID}} {{.Names}}' > "$OUT/containers-after.txt"
ps_rc=$?;printf '%s\n' "$ps_rc" > "$OUT/container-ps.exit"
confirmed_removed(){
  [ "$ps_rc" = 0 ] || return 1
  local seen_id seen_name
  while read -r seen_id seen_name;do
    if [ "$seen_name" = "$NAME" ] || { [ -n "$OWN_CID" ] && [ "$seen_id" = "$OWN_CID" ]; };then return 1;fi
  done < "$1"
  return 0
}
removed=0
if confirmed_removed "$OUT/containers-after.txt";then removed=1;fi
# Health must never overlap a still-live or unconfirmed container. Preserve
# the existing GPU lease until successful Docker enumeration confirms removal.
# Cleanup addresses only the ID created by THIS docker run, never a shared name.
while [ "$removed" != 1 ];do
  echo 'S75: holding CPU/card0 leases; container removal unconfirmed; no health yet'
  if [ -n "$OWN_CID" ];then
    docker kill "$OWN_CID" >> "$OUT/cleanup.stdout" 2>> "$OUT/cleanup.stderr" || true
    docker wait "$OWN_CID" >> "$OUT/cleanup.stdout" 2>> "$OUT/cleanup.stderr" || true
    docker rm "$OWN_CID" >> "$OUT/cleanup.stdout" 2>> "$OUT/cleanup.stderr" || true
  fi
  sleep 5
  docker ps -a --no-trunc --format '{{.ID}} {{.Names}}' > "$OUT/containers-after-recheck.txt"
  ps_rc=$?
  printf '%s\n' "$ps_rc" >> "$OUT/container-recheck-exits.txt"
  if confirmed_removed "$OUT/containers-after-recheck.txt";then removed=1;fi
done
# Post-health only after confirmed removal, including strict/runtime failure.
"$REPO/bin/xpu-health" --card 0 --img "$HEALTH_IMG" --timeout 400 > "$OUT/post-health.log" 2>&1
post=$?;printf '%s\n' "$post" > "$OUT/post-health.exit"
compare=2
if [ "$run" = 0 ] && [ "$removed" = 1 ];then
  python3 "$HERE/analyze.py" "$OUT" > "$OUT/analysis.stdout" 2> "$OUT/analysis.stderr"
  compare=$?
  if [ "$compare" = 0 ];then
    python3 "$EXEC_DIR/extra_host_closure.py" post "$OUT" > "$OUT/extra-host.stdout" 2> "$OUT/extra-host.stderr"
    compare=$?
  fi
fi
printf '%s\n' "$compare" > "$OUT/analysis.exit"
python3 - "$OUT" "$run" "$pre" "$post" "$compare" "$removed" <<'PY'
import json,sys
from pathlib import Path
out=Path(sys.argv[1]);run,pre,post,compare,removed=map(int,sys.argv[2:])
d={'status':'PASS_PRIMITIVE_ONLY' if run==pre==post==compare==0 and removed==1 else 'FAIL_INSPECT_RAW',
 'leaf_exit':run,'pre_health_exit':pre,'post_health_exit':post,'analysis_exit':compare,
 'container_removed':bool(removed),'processes':1,'retries':0,'serving_qualified':False,'speed_qualified':False}
with(out/'closure.json').open('x',encoding='ascii')as f:json.dump(d,f,indent=2);f.write('\n')
print(json.dumps(d))
PY
close=$?
[ "$run" = 0 ] || exit "$run"
[ "$post" = 0 ] || exit "$post"
[ "$removed" = 1 ] || exit 2
[ "$compare" = 0 ] || exit "$compare"
exit "$close"
