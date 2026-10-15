#!/usr/bin/env bash
# E4: SGLang head XPU host overhead, 4-layer dummy Flash-Next, card 0. Usage: e4-run.sh eager|graph
set -u; cd /mnt/vm_8tb/b70/build/backend-decision; R=/mnt/vm_8tb/github/b70_ai_things; MODE=$1; mkdir -p e4-results
G=""; [ "$MODE" = graph ] && G="--cuda-graph-backend-decode full --cuda-graph-bs-decode 1 2 4"
docker rm -f bd-e4 >/dev/null 2>&1; docker run -d --name bd-e4 --device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path --ipc=host --network host \
  -v $PWD:/e4 -e HOME=/e4/home -e ZE_AFFINITY_MASK=0 -e ONEAPI_DEVICE_SELECTOR=level_zero:0 -e SGLANG_BUILD_RUST_EXTS=none \
  --entrypoint /e4/venv/bin/python b70/sglang-main-triton-dense:20260910 -m sglang.launch_server \
  --model-path /e4/e4-model --load-format dummy --device xpu --tp 1 --port 18192 --host 127.0.0.1 \
  --context-length 8192 --mem-fraction-static 0.85 --max-running-requests 8 --disable-radix-cache --attention-backend intel_xpu $G > /dev/null
for i in $(seq 1 450); do curl -sf 127.0.0.1:18192/health >/dev/null && break; docker ps -q -f name=bd-e4 | grep -q . || break; sleep 2; done
docker logs bd-e4 > e4-results/server-$MODE.log 2>&1
if curl -sf 127.0.0.1:18192/health >/dev/null; then timeout 900 /usr/bin/python3 -I e4_client.py http://127.0.0.1:18192 e4-results/$MODE.json 256; else echo "server failed"; grep -iE "error|exception|Traceback" e4-results/server-$MODE.log | tail -15; fi
docker logs bd-e4 > e4-results/server-$MODE.log 2>&1; docker stop bd-e4 >/dev/null 2>&1; docker rm -f bd-e4 >/dev/null 2>&1
$R/bin/xpu-health --card 0 > e4-results/post-health-$MODE.txt 2>&1; echo "xpu-health rc=$?"
