#!/usr/bin/env bash
# E5 (matched Strata harness vs llama-server, both cards) then E6 (unitrace decode kernel census).
# Run as: B70_GPU_LOCK_TIMEOUT=5 bin/cpu-run --quiet bin/gpu-run bash e5e6-run.sh
set -u; cd /mnt/vm_8tb/b70/build/backend-decision; R=/mnt/vm_8tb/github/b70_ai_things; OUT=e5e6-results; mkdir -p $OUT
M=$(readlink -f $R/models/files/qwen3.8-flash-next/ista-gsq-rco-iq3-xxs); MTP=$(dirname $(readlink -f $R/models/files/qwen3.8-flash-next/unsloth-ud-q4-k-xl/MTP/mtp-Qwen3.8-Flash-Next-Q4_K_M.gguf))
U=/mnt/vm_8tb/b70/build/strata-ab/pti-gpu/tools/unitrace/build
GG=/m/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf
DOCK="--device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path --ipc=host -v $PWD/llama.cpp:/src:ro -v $M:/m:ro -v /mnt/cache:/mnt/cache:ro -v $MTP:/mtp:ro -v $U:/ut:ro -v $PWD/$OUT:/out -e ONEAPI_DEVICE_SELECTOR=level_zero:0,1 -e ZE_AFFINITY_MASK=0,1 --entrypoint bash b70/flashnext-llamacpp-runtime:neo26.22"
ENV="source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1; export LD_LIBRARY_PATH=/src/build-sycl/bin:\$LD_LIBRARY_PATH"
echo "== pre-health"; $R/bin/xpu-health > $OUT/pre-health.txt 2>&1; echo "xpu-health rc=$?"

echo "== E5 server $(date -u +%T)"
docker rm -f bd-e5 >/dev/null 2>&1
docker run -d --name bd-e5 --network host $DOCK -c "$ENV; exec /src/build-sycl/bin/llama-server -m $GG -ngl 99 -fa on -sm layer -c 16384 -np 2 --host 127.0.0.1 --port 18191 --jinja --alias hotschmoe-dd,qwen3.8-flash-next-ISTA-GSQ-RCO-IQ3_XXS-llamacpp-sycl --spec-type draft-mtp -md /mtp/mtp-Qwen3.8-Flash-Next-Q4_K_M.gguf -ngld 99 --spec-draft-n-max 3" >/dev/null
for i in $(seq 1 300); do curl -sf 127.0.0.1:18191/health >/dev/null && break; docker ps -q -f name=bd-e5 | grep -q . || break; sleep 2; done
rm -rf $OUT/e5; timeout 1500 python3 -I e5_matched.py http://127.0.0.1:18191 $OUT/e5; echo "e5 rc=$?"
docker logs bd-e5 > $OUT/e5-server.log 2>&1; docker stop bd-e5 >/dev/null 2>&1; docker rm -f bd-e5 >/dev/null 2>&1

# E6: decode-only kernel census, layer split, no MTP. n=80 minus n=16 removes load/warmup/setup.
for N in 16 80; do
  echo "== E6 n=$N $(date -u +%T)"
  timeout --kill-after=10 900 docker run --rm --name bd-e6 $DOCK -c "$ENV; /ut/unitrace -d -o /out/e6-n$N.txt /src/build-sycl/bin/llama-bench -m $GG -ngl 99 -fa 1 -sm layer -p 0 -n $N -r 1 -o md" > $OUT/e6-n$N.log 2>&1
  echo "e6 n=$N rc=$?"; grep -E "^\|" $OUT/e6-n$N.log | tail -1; docker rm -f bd-e6 >/dev/null 2>&1
done
echo "== post-health"; $R/bin/xpu-health > $OUT/post-health.txt 2>&1; echo "xpu-health rc=$?"
$R/bin/xpu-collective-health > $OUT/post-collective.txt 2>&1; echo "collective rc=$?"
