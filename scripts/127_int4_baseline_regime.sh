#!/usr/bin/env bash
# 127_int4_baseline_regime.sh -- validate the testing regime (perf_regime.sh + soak_probe.py) against
# the KNOWN int4 woq baseline (~9.4 t/s warm, soak-STABLE since no graph replay). Serves as both the
# harness smoke test AND the control baseline every lever is compared against.
# Single card -> run under: ./bin/gpu-run --card 0 bash scripts/127_int4_baseline_regime.sh
set -uo pipefail
ROOT=/mnt/vm_8tb/b70; REPO=/mnt/vm_8tb/github/b70_ai_things
IMG=sglang-xpu:woq; NAME=sglang_int4base; PORT=30000; SERVED=qwen36-27b-int4-woq
CKPT=/models/Lorbus_Qwen3.6-27B-int4-AutoRound; TOK=/models/Qwen_Qwen3.6-27B
LOG=$REPO/sglang/int4_baseline_regime.log; : > "$LOG"
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

say "=== serve int4 woq TP=1 card0 (baseline for the regime) ==="
docker rm -f "$NAME" >/dev/null 2>&1
docker run -d --name "$NAME" --device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path \
  --ipc=host --shm-size 16g -p "${PORT}:${PORT}" -e ZE_AFFINITY_MASK=0 \
  -v "$ROOT/models:/models:ro" -v "$ROOT/hf_cache:/hf_cache" -v "$ROOT/sgl_cache:/sgl_cache" \
  -e HF_HOME=/hf_cache -e XDG_CACHE_HOME=/sgl_cache -e TORCHINDUCTOR_CACHE_DIR=/sgl_cache/inductor \
  "$IMG" bash -c "source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1; exec python -m sglang.launch_server \
    --model-path '$CKPT' --served-model-name '$SERVED' --trust-remote-code \
    --device xpu --attention-backend intel_xpu --linear-attn-backend triton \
    --mamba-ssm-dtype float32 --disable-overlap-schedule --page-size 64 --disable-radix-cache \
    --tp 1 --context-length 8192 --mem-fraction-static 0.90 \
    --host 0.0.0.0 --port $PORT" >>"$LOG" 2>&1

say "waiting for /health..."
ok=0
for i in $(seq 1 120); do
  docker ps --filter "name=$NAME" --format '{{.Names}}' | grep -q "$NAME" || { say "CONTAINER EXITED"; docker logs "$NAME" 2>&1|tail -30|tee -a "$LOG"; ok=2; break; }
  [ "$(curl -s -o /dev/null -w '%{http_code}' "http://localhost:$PORT/health" 2>/dev/null||echo 000)" = 200 ] && { ok=1; say "/health 200 (~$((i*5))s)"; break; }
  sleep 5
done
[ "$ok" = 1 ] || { say "not healthy; abort"; docker rm -f "$NAME">/dev/null 2>&1; exit 1; }

say "=== RUN THE REGIME ==="
bash "$REPO/sglang/perf_regime.sh" "$NAME" "$PORT" "$SERVED" "$TOK" "int4-woq-baseline" 2>&1 | tee -a "$LOG"

say "=== stop ==="; docker rm -f "$NAME" >/dev/null 2>&1; say "stopped"
