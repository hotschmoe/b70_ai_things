#!/usr/bin/env bash
# 128_mtp_nextn_regime.sh -- A2: does the assign_extend_cache_locs fix clear the NEXTN spec-decode VERIFY
# crash on sglang-XPU, and does chain-MTP (topk=1) beat the 9.4 eager ceiling? Serves the grafted
# int4+vision+MTP ckpt with the fixed mtp_tree_xpu mounted, checks coherence (the real unblock signal),
# then measures accept length + decode on the regime.
# Single card -> run under: ./bin/gpu-run --card 0 bash scripts/128_mtp_nextn_regime.sh
set -uo pipefail
ROOT=/mnt/vm_8tb/b70; REPO=/mnt/vm_8tb/github/b70_ai_things
IMG=sglang-xpu:woq; NAME=sglang_mtp; PORT=30000; SERVED=qwen36-27b-int4-mtp-nextn
CKPT=/models/Lorbus_Qwen3.6-27B-int4-mtp; TOK=/models/Qwen_Qwen3.6-27B
SP=/opt/venv/lib/python3.12/site-packages
LOG=$REPO/sglang/mtp_nextn_regime.log; : > "$LOG"
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

say "=== A2: NEXTN chain-MTP (topk=1) on int4, with the out_cache_loc fix ==="
docker rm -f "$NAME" >/dev/null 2>&1
docker run -d --name "$NAME" --device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path \
  --ipc=host --shm-size 16g -p "${PORT}:${PORT}" -e ZE_AFFINITY_MASK=0 \
  -v "$ROOT/models:/models:ro" -v "$ROOT/hf_cache:/hf_cache" -v "$ROOT/sgl_cache:/sgl_cache" \
  -v "$REPO/sglang/patches/mtp_tree_xpu.py:$SP/mtp_tree_xpu.py:ro" \
  -v "$REPO/sglang/patches/woq_shim.py:$SP/woq_shim.py:ro" \
  -v "$REPO/sglang/patches/memory_pool.py:$SP/sglang/srt/mem_cache/memory_pool.py:ro" \
  -e HF_HOME=/hf_cache -e XDG_CACHE_HOME=/sgl_cache -e TORCHINDUCTOR_CACHE_DIR=/sgl_cache/inductor \
  -e B70_XPU_MTP=1 -e B70_MTP_DEBUG="${DBG:-0}" \
  "$IMG" bash -c "source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1; exec python -m sglang.launch_server \
    --model-path '$CKPT' --served-model-name '$SERVED' --trust-remote-code \
    --device xpu --attention-backend intel_xpu --linear-attn-backend triton \
    --mamba-ssm-dtype float32 --disable-overlap-schedule --page-size 64 --disable-radix-cache \
    --speculative-algorithm NEXTN --speculative-num-steps ${SPEC_STEPS:-1} --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens ${SPEC_DRAFT:-2} --speculative-draft-attention-backend triton --disable-cuda-graph \
    --max-running-requests 4 --skip-server-warmup \
    --tp 1 --context-length 4096 --mem-fraction-static 0.92 \
    --host 0.0.0.0 --port $PORT" >>"$LOG" 2>&1

say "waiting for /health (skip-warmup -> fast; first gen JITs)..."
ok=0
for i in $(seq 1 90); do
  docker ps --filter "name=$NAME" --format '{{.Names}}' | grep -q "$NAME" || { say "CONTAINER EXITED"; docker logs "$NAME" 2>&1|tail -40|tee -a "$LOG"; ok=2; break; }
  [ "$(curl -s -o /dev/null -w '%{http_code}' "http://localhost:$PORT/health" 2>/dev/null||echo 000)" = 200 ] && { ok=1; say "/health 200 (~$((i*5))s)"; break; }
  sleep 5
done
[ "$ok" = 1 ] || { say "not healthy; abort"; docker logs "$NAME" 2>&1|tail -30|tee -a "$LOG"; docker rm -f "$NAME">/dev/null 2>&1; exit 1; }

# THE UNBLOCK SIGNAL: does a real generation succeed (no verify crash) and stay coherent?
say "=== coherence / unblock check (first gen JITs spec path ~13s) ==="
g=$(curl -s --max-time 180 "http://localhost:$PORT/v1/chat/completions" -H 'content-type: application/json' \
  -d "{\"model\":\"$SERVED\",\"messages\":[{\"role\":\"user\",\"content\":\"Why is the sky blue? Two sentences.\"}],\"max_tokens\":96,\"temperature\":0}")
genok=$(echo "$g" | python3 -c "import sys,json
try:
 d=json.load(sys.stdin); c=d['choices'][0]['message']['content'] or ''
 print('GEN OK:'+repr(c[:200]) if c.strip() else 'GEN EMPTY')
except Exception as e: print('GEN FAIL:'+repr(sys.stdin.read()[:200]))")
say "$genok"
# ALWAYS dump the full server log (the container may die on the spec forward); preserve the traceback.
docker logs "$NAME" > "$REPO/sglang/mtp_server.log" 2>&1 || true
running=$(docker ps --filter "name=$NAME" --format '{{.Names}}' | grep -c "$NAME" || true)
if [ "${genok#GEN OK}" = "$genok" ] || [ "$running" = 0 ]; then
  say "*** SPEC FORWARD FAILED (gen=$genok running=$running) -- full log -> sglang/mtp_server.log; traceback tail: ***"
  grep -iE "error|traceback|out_cache_loc|assign_extend|mamba|commit_mamba|update_mamba|RuntimeError|Exception|File \"|line [0-9]|scatter|triton|compil" "$REPO/sglang/mtp_server.log" | tail -45 | tee -a "$LOG"
  say "full server log saved -> sglang/mtp_server.log; stopping container (clean lease release)."
  docker rm -f "$NAME" >/dev/null 2>&1; exit 3
fi

say "=== regime (warm + soak + coherence) ==="
bash "$REPO/sglang/perf_regime.sh" "$NAME" "$PORT" "$SERVED" "$TOK" "int4-NEXTN-mtp" 2>&1 | tee -a "$LOG"
say "=== accept length (from server decode log) ==="
docker logs "$NAME" 2>&1 | grep -iE "accept|spec" | tail -8 | tee -a "$LOG"

say "=== stop ==="; docker rm -f "$NAME" >/dev/null 2>&1; say "stopped"
