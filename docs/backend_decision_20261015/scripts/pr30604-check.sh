#!/usr/bin/env bash
# PR 30604 check: e467e5b5 + PR 30604, default GGML_SYCL_ENABLE_OPT (on). Run as: bin/gpu-run bash pr30604-check.sh
set -u; cd /mnt/vm_8tb/b70/build/backend-decision; R=/mnt/vm_8tb/github/b70_ai_things; OUT=e5e6-results
M=$(readlink -f $R/models/files/qwen3.8-flash-next/ista-gsq-rco-iq3-xxs); GG=/m/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf
for ARM in pr30604; do
  E=""
  echo "== $ARM $(date -u +%T)"; docker rm -f bd-e5f >/dev/null 2>&1
  docker run -d --name bd-e5f --network host --device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path --ipc=host -v $PWD/llama.cpp-pr30604:/src:ro -v $M:/m:ro -v /mnt/cache:/mnt/cache:ro \
    -e ONEAPI_DEVICE_SELECTOR=level_zero:0,1 -e ZE_AFFINITY_MASK=0,1 $E --entrypoint bash b70/flashnext-llamacpp-runtime:neo26.22 -c \
    "source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1; export LD_LIBRARY_PATH=/src/build-sycl/bin:\$LD_LIBRARY_PATH; exec /src/build-sycl/bin/llama-server -m $GG -ngl 99 -fa on -sm layer -c 16384 -np 2 --host 127.0.0.1 --port 18191 --jinja --alias hotschmoe-dd" >/dev/null
  for i in $(seq 1 150); do curl -sf 127.0.0.1:18191/health >/dev/null && break; docker ps -q -f name=bd-e5f | grep -q . || break; sleep 2; done
  timeout 300 python3 -I e5_discrim.py http://127.0.0.1:18191 $OUT/pr30604-${ARM%%=*}.json 2>&1 | grep -E "arith|coding|Error"
  timeout 300 python3 -I e5_sweep.py http://127.0.0.1:18191 $OUT/pr30604-sweep.json 2>&1 | tail -14
  docker logs bd-e5f > $OUT/pr30604-${ARM%%=*}-server.log 2>&1; docker stop bd-e5f >/dev/null 2>&1; docker rm -f bd-e5f >/dev/null 2>&1
done
echo "== post-health"; $R/bin/xpu-health > $OUT/pr30604-post-health.txt 2>&1; echo "xpu-health rc=$?"
$R/bin/xpu-collective-health > $OUT/pr30604-post-collective.txt 2>&1; echo "collective rc=$?"
