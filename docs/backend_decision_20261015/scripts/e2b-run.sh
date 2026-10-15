#!/usr/bin/env bash
set -u; cd /mnt/vm_8tb/b70/build/backend-decision; R=/mnt/vm_8tb/github/b70_ai_things; OUT=e2-results
M=$(readlink -f $R/models/files/qwen3.8-flash-next/ista-gsq-rco-iq3-xxs); MTP=$(dirname $(readlink -f $R/models/files/qwen3.8-flash-next/unsloth-ud-q4-k-xl/MTP/mtp-Qwen3.8-Flash-Next-Q4_K_M.gguf))
docker run -d --rm --name bd-e2b --device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path --ipc=host --network host \
  -v $PWD/llama.cpp:/src:ro -v $M:/m:ro -v /mnt/cache:/mnt/cache:ro -v $MTP:/mtp:ro \
  -e ONEAPI_DEVICE_SELECTOR=level_zero:0,1 -e ZE_AFFINITY_MASK=0,1 \
  --entrypoint bash b70/flashnext-llamacpp-runtime:neo26.22 -c \
  "source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1; export LD_LIBRARY_PATH=/src/build-sycl/bin:\$LD_LIBRARY_PATH; exec /src/build-sycl/bin/llama-server -m /m/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf -ngl 99 -fa on -sm layer -c 16384 -np 2 --host 127.0.0.1 --port 18191 --jinja --spec-type draft-mtp -md /mtp/mtp-Qwen3.8-Flash-Next-Q4_K_M.gguf -ngld 99 --spec-draft-n-max 3" >/dev/null
for i in $(seq 1 300); do curl -sf 127.0.0.1:18191/health >/dev/null && break; docker ps -q -f name=bd-e2b | grep -q . || break; sleep 2; done
curl -s 127.0.0.1:18191/health; echo
timeout 1500 python3 -I e2b_serve.py http://127.0.0.1:18191 $OUT/e2b.json e2-coding-template.json 1500
docker logs bd-e2b 2>&1 | grep -E "acceptance|error|ERROR" | tail -8 > $OUT/e2b-server-tail.txt; cat $OUT/e2b-server-tail.txt
docker stop bd-e2b >/dev/null 2>&1
echo "== post-health"; $R/bin/xpu-health > $OUT/e2b-post-health.txt 2>&1; echo "xpu-health rc=$?"
$R/bin/xpu-collective-health > $OUT/e2b-post-collective.txt 2>&1; echo "collective rc=$?"
