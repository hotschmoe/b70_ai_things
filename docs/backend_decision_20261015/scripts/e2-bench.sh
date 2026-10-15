#!/usr/bin/env bash
# E2a: llama-bench layer vs tensor split, no MTP. Run as: cpu-run --quiet bin/gpu-run ./e2-bench.sh
set -u; cd /mnt/vm_8tb/b70/build/backend-decision
R=/mnt/vm_8tb/github/b70_ai_things; OUT=e2-results; mkdir -p $OUT
M=$(readlink -f $R/models/files/qwen3.8-flash-next/ista-gsq-rco-iq3-xxs)
echo "== pre-health"; $R/bin/xpu-health > $OUT/pre-health.txt 2>&1; echo "xpu-health rc=$?"
run() { # $1=name $2..=llama-bench args
  local n=$1; shift
  timeout --kill-after=10 ${E2_TIMEOUT:-900} docker run --rm --name bd-e2-$n --device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path --ipc=host \
    -v $PWD/llama.cpp:/src:ro -v $M:/m:ro -v /mnt/cache:/mnt/cache:ro \
    -e ONEAPI_DEVICE_SELECTOR=level_zero:0,1 -e ZE_AFFINITY_MASK=0,1 \
    --entrypoint bash b70/flashnext-llamacpp-runtime:neo26.22 -c \
    "source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1; export LD_LIBRARY_PATH=/src/build-sycl/bin:\$LD_LIBRARY_PATH; /src/build-sycl/bin/llama-bench -m /m/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf -ngl 99 -fa 1 -p 4096 -n 128 -r 2 -o md $*" > $OUT/$n.txt 2>&1
  echo "$n rc=$?"; grep -E "^\|" $OUT/$n.txt | tail -3; docker rm -f bd-e2-$n >/dev/null 2>&1
}
run layer -sm layer
run tensor -sm tensor
echo "== post-health"; $R/bin/xpu-health > $OUT/post-health.txt 2>&1; echo "xpu-health rc=$?"
$R/bin/xpu-collective-health > $OUT/post-collective.txt 2>&1; echo "collective rc=$?"
