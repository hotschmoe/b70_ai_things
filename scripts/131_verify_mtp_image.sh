#!/usr/bin/env bash
# 131_verify_mtp_image.sh -- ACCEPTANCE TEST for the productionized int4+MTP daily driver.
# Drives the EXACT shipped recipe (rdy_to_serve/qwen36-27b-int4-mtp/serve.sh) against the BAKED
# sglang-xpu:mtp image with ZERO runtime patch mounts -> proves the image is self-contained, and
# reproduces the ~15.3 t/s (1.62x) single-stream MTP win + coherence under sustained mixed load.
#   Single card -> run under: ./bin/gpu-run --card 0 bash scripts/131_verify_mtp_image.sh
set -uo pipefail
REPO=/mnt/vm_8tb/github/b70_ai_things
R="$REPO/rdy_to_serve/qwen36-27b-int4-mtp/serve.sh"
PORT=30000; SERVED=qwen36-27b-int4-mtp-nextn; NAME=sglang_int4_mtp
LOG="$REPO/sglang/verify_mtp_image.log"; : > "$LOG"
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

say "=== 1. START shipped recipe (baked image, NO mounts) -- coherence-gated ==="
if ! bash "$R" start 2>&1 | tee -a "$LOG"; then
  say "*** recipe start FAILED (not healthy or not coherent) ***"; bash "$R" stop >/dev/null 2>&1; exit 1
fi

say "=== 2. BENCH (warm c1/c4 pp/ttft/tg @ ctx2048 + soak) ==="
bash "$R" bench 2>&1 | tee -a "$LOG"

say "=== 3. SUSTAINED MIXED LOAD (the agentic prefill+decode pattern that breaks vLLM) ==="
PD="${SCRATCH:-/tmp/claude-1000/-mnt-vm-8tb-github-b70-ai-things/3d35fc3f-649b-4a06-b50e-ec2ce0215970/scratchpad}/gdn_mtp_verify"
mkdir -p "$PD"; cp "$REPO"/contrib/gdn_nan_repro/dd_mixload.py "$PD"/
python3 -c "import json;d=json.load(open('$REPO/contrib/gdn_nan_repro/backhoe_req.json'));d['model']='$SERVED';json.dump(d,open('$PD/backhoe_req.json','w'))"
python3 "$PD/dd_mixload.py" "$PORT" ${ML_ANCH:-3} ${ML_BURST:-3} ${ML_WAVES:-6} 2.0 ${ML_BMAX:-300} ${ML_AMAX:-600} 2>&1 | tee -a "$LOG"
upnow=$(docker ps --filter "name=$NAME" --format '{{.Names}}' | grep -c "$NAME" || true)
say "post-mixload: container running=$upnow (0 = the serve CRASHED under load)"
pg=$(curl -s "http://localhost:$PORT/v1/chat/completions" -H 'content-type: application/json' \
  -d "{\"model\":\"$SERVED\",\"messages\":[{\"role\":\"user\",\"content\":\"Capital of France in one word.\"}],\"max_tokens\":24,\"temperature\":0}")
echo "$pg" | python3 -c "import sys,json;d=json.load(sys.stdin);print('POST-LOAD COHERENCE:',repr(d['choices'][0]['message']['content'][:80]))" | tee -a "$LOG" || say "post-load parse fail"

say "=== 4. accept length ==="
bash "$R" accept 2>&1 | tee -a "$LOG"

say "=== 5. STOP ==="
bash "$R" stop 2>&1 | tee -a "$LOG"
say "=== verify complete -> $LOG ==="
