#!/usr/bin/env bash
# 132_mtp_torch_compile.sh -- PERF PUSH lever 1: collapse the launch-bound decode overhead.
# The int4+MTP driver is launch-bound (~1045 kernel submissions/token, decode_launch_inventory.md);
# MTP already amortizes those across accept_len (~4.48). The OTHER systemic cure is torch.compile/Inductor
# (collapse the python submissions into compiled wrappers, NO L0 graph replay -> dodges the XPU cudagraph
# degradation). This sglang exposes --enable-torch-compile standalone. Test whether it STACKS on MTP.
#   A/B vs the known baseline c1=15.31 t/s. Single card -> ./bin/gpu-run --card 0 bash scripts/132_*.sh
#   TC=1 (default) enables torch.compile; TC=0 reproduces the no-compile baseline through this same script.
set -uo pipefail
ROOT=/mnt/vm_8tb/b70; REPO=/mnt/vm_8tb/github/b70_ai_things
IMG=sglang-xpu:mtp; NAME=sglang_mtp_tc; PORT=30000; SERVED=qwen36-27b-int4-mtp-nextn
CKPT=/models/Lorbus_Qwen3.6-27B-int4-mtp; TOK=/models/Qwen_Qwen3.6-27B
TC="${TC:-1}"; TCBS="${TCBS:-4}"; MAXREQ="${MAXREQ:-4}"; CTX="${CTX:-4096}"
SPEC_STEPS="${SPEC_STEPS:-7}"; SPEC_DRAFT="${SPEC_DRAFT:-8}"
LOG="$REPO/sglang/mtp_torch_compile.log"; : > "$LOG"
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

TCFLAGS=""; [ "$TC" = 1 ] && TCFLAGS="--enable-torch-compile --torch-compile-max-bs $TCBS"
say "=== int4+MTP torch.compile A/B: TC=$TC (flags: ${TCFLAGS:-none}) steps=$SPEC_STEPS ctx=$CTX ==="

docker rm -f "$NAME" >/dev/null 2>&1
docker run -d --name "$NAME" --device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path \
  --ipc=host --shm-size 16g -p "${PORT}:${PORT}" -e ZE_AFFINITY_MASK=0 \
  -v "$ROOT/models:/models:ro" -v "$ROOT/hf_cache:/hf_cache" -v "$ROOT/sgl_cache:/sgl_cache" \
  -e HF_HOME=/hf_cache -e XDG_CACHE_HOME=/sgl_cache -e TORCHINDUCTOR_CACHE_DIR=/sgl_cache/inductor \
  -e B70_XPU_MTP=1 \
  "$IMG" bash -c "source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1; exec python -m sglang.launch_server \
    --model-path '$CKPT' --served-model-name '$SERVED' --trust-remote-code \
    --device xpu --attention-backend intel_xpu --linear-attn-backend triton \
    --mamba-ssm-dtype float32 --disable-overlap-schedule --page-size 64 --disable-radix-cache \
    --speculative-algorithm NEXTN --speculative-num-steps $SPEC_STEPS --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens $SPEC_DRAFT --speculative-draft-attention-backend triton --disable-cuda-graph \
    $TCFLAGS --max-running-requests $MAXREQ --skip-server-warmup \
    --tp 1 --context-length $CTX --mem-fraction-static 0.92 \
    --host 0.0.0.0 --port $PORT" >>"$LOG" 2>&1

say "waiting for /health..."
ok=0
for i in $(seq 1 120); do
  docker ps --filter "name=$NAME" --format '{{.Names}}' | grep -q "$NAME" || { say "CONTAINER EXITED"; docker logs "$NAME" 2>&1|tail -40|tee -a "$LOG"; exit 2; }
  [ "$(curl -s -o /dev/null -w '%{http_code}' "http://localhost:$PORT/health" 2>/dev/null||echo 000)" = 200 ] && { ok=1; say "/health 200 (~$((i*5))s)"; break; }
  sleep 5
done
[ "$ok" = 1 ] || { say "not healthy; abort"; docker logs "$NAME" 2>&1|tail -40|tee -a "$LOG"; docker rm -f "$NAME">/dev/null 2>&1; exit 1; }

# torch.compile JITs (inductor) on the FIRST decode of each new shape -> warm HARD before benching.
say "=== coherence + compile warmup (torch.compile inductor pass can take minutes on first gens) ==="
for w in 1 2 3; do
  g=$(curl -s --max-time 600 "http://localhost:$PORT/v1/chat/completions" -H 'content-type: application/json' \
    -d "{\"model\":\"$SERVED\",\"messages\":[{\"role\":\"user\",\"content\":\"Why is the sky blue? Explain in 4 sentences.\"}],\"max_tokens\":128,\"temperature\":0}")
  c=$(echo "$g" | python3 -c "import sys,json
try:
 t=(json.load(sys.stdin)['choices'][0]['message']['content'] or '').strip(); print('OK '+repr(t[:80]) if t else 'EMPTY')
except Exception: print('FAIL '+repr(sys.stdin.read()[:120]))")
  say "warmup gen $w: $c"
  [ "${c#OK}" = "$c" ] && { say "*** gen failed during warmup -- dumping log ***"; docker logs "$NAME" 2>&1|tail -50|tee -a "$LOG"; docker rm -f "$NAME">/dev/null 2>&1; exit 3; }
done

say "=== regime (coherence + warm c1/c4 + soak) ==="
bash "$REPO/sglang/perf_regime.sh" "$NAME" "$PORT" "$SERVED" "$TOK" "int4-MTP-tc$TC" 2>&1 | tee -a "$LOG"
say "=== accept length ==="
docker logs "$NAME" 2>&1 | grep -oE "accept len: [0-9.]+" | tail -10 | tee -a "$LOG"
docker logs "$NAME" 2>&1 | grep -oE "accept len: [0-9.]+" | awk -F': ' '{s+=$2;n++} END{if(n)printf "[mean accept len over %d batches] %.2f\n",n,s/n}' | tee -a "$LOG"
say "=== check torch.compile actually engaged (inductor markers in log) ==="
docker logs "$NAME" 2>&1 | grep -iE "torch.compile|inductor|compiling|TorchInductor|graph break|recompil" | tail -8 | tee -a "$LOG" || say "(no compile markers found)"
say "=== stop ==="; docker rm -f "$NAME" >/dev/null 2>&1; say "stopped"
