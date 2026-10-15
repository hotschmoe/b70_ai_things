#!/usr/bin/env bash
# E5b: coherence diagnostic, MTP on then off. Run as: bin/gpu-run bash e5b-run.sh
set -u; cd /mnt/vm_8tb/b70/build/backend-decision; R=/mnt/vm_8tb/github/b70_ai_things; OUT=e5e6-results; mkdir -p $OUT
M=$(readlink -f $R/models/files/qwen3.8-flash-next/ista-gsq-rco-iq3-xxs); MTP=$(dirname $(readlink -f $R/models/files/qwen3.8-flash-next/unsloth-ud-q4-k-xl/MTP/mtp-Qwen3.8-Flash-Next-Q4_K_M.gguf))
GG=/m/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf
for ARM in mtp nomtp; do
  S=""; [ $ARM = mtp ] && S="--spec-type draft-mtp -md /mtp/mtp-Qwen3.8-Flash-Next-Q4_K_M.gguf -ngld 99 --spec-draft-n-max 3"
  echo "== $ARM $(date -u +%T)"; docker rm -f bd-e5 >/dev/null 2>&1
  docker run -d --name bd-e5 --network host --device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path --ipc=host -v $PWD/llama.cpp:/src:ro -v $M:/m:ro -v /mnt/cache:/mnt/cache:ro -v $MTP:/mtp:ro \
    -e ONEAPI_DEVICE_SELECTOR=level_zero:0,1 -e ZE_AFFINITY_MASK=0,1 --entrypoint bash b70/flashnext-llamacpp-runtime:neo26.22 -c \
    "source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1; export LD_LIBRARY_PATH=/src/build-sycl/bin:\$LD_LIBRARY_PATH; exec /src/build-sycl/bin/llama-server -m $GG -ngl 99 -fa on -sm layer -c 16384 -np 2 --host 127.0.0.1 --port 18191 --jinja --alias hotschmoe-dd $S" >/dev/null
  for i in $(seq 1 300); do curl -sf 127.0.0.1:18191/health >/dev/null && break; docker ps -q -f name=bd-e5 | grep -q . || break; sleep 2; done
  timeout 600 python3 -I e5_coherence.py http://127.0.0.1:18191 $OUT/e5b-$ARM.json; echo "rc=$?"
  docker logs bd-e5 > $OUT/e5b-$ARM-server.log 2>&1; docker stop bd-e5 >/dev/null 2>&1; docker rm -f bd-e5 >/dev/null 2>&1
done
echo "== post-health"; $R/bin/xpu-health > $OUT/e5b-post-health.txt 2>&1; echo "xpu-health rc=$?"
$R/bin/xpu-collective-health > $OUT/e5b-post-collective.txt 2>&1; echo "collective rc=$?"
