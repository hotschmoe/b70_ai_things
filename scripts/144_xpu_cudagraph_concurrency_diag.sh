#!/usr/bin/env bash
# 144_xpu_cudagraph_concurrency_diag.sh -- diagnose WHY multi-bucket graph capture HALVES single-stream
# (bs=1/maxreq=1 = 23.5, but bs[1 2 4]/maxreq=4 = 7.36; scripts/141). ISOLATE: is it maxreq>1 alone, or the
# extra buckets? Configs: A bs[1]/maxreq=4 (single bucket, concurrency allowed) vs B bs[1 2]/maxreq=4 vs
# C bs[1 2 4]/maxreq=4. If A==23.5 -> the extra buckets/pool are the problem; if A<23.5 -> maxreq>1 scheduling.
# Card 0 (fast). DBG=1 logs the captured bs + which graph a single request replays.
#   ./bin/gpu-run --card 0 bash scripts/144_xpu_cudagraph_concurrency_diag.sh
set -uo pipefail
ROOT=/mnt/vm_8tb/b70; REPO=/mnt/vm_8tb/github/b70_ai_things
IMG=sglang-xpu:mtp; NAME=sglang_cgdiag; PORT=30000; SERVED=qwen36-27b-int4-xpucg
CKPT=/models/Lorbus_Qwen3.6-27B-int4-AutoRound; TOK=/models/Qwen_Qwen3.6-27B
SP=/opt/venv/lib/python3.12/site-packages
LOG="$REPO/sglang/cg_concurrency_diag.log"; : > "$LOG"
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

# label | bs-decode list | max-bs | maxreq
declare -a CFG=( "A_bs1_mr4|1|1|4" "B_bs12_mr4|1 2|2|4" "C_bs124_mr4|1 2 4|4|4" )

run_cfg(){ local lbl="$1" bslist="$2" bsmax="$3" mreq="$4"
  say "================= $lbl (bs[$bslist] max=$bsmax maxreq=$mreq) ================="
  docker rm -f "$NAME" >/dev/null 2>&1
  docker run -d --name "$NAME" --device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path \
    --ipc=host --shm-size 16g -p "${PORT}:${PORT}" -e ZE_AFFINITY_MASK=0 \
    -v "$ROOT/models:/models:ro" -v "$ROOT/hf_cache:/hf_cache" -v "$ROOT/sgl_cache:/sgl_cache" \
    -v "$REPO/sglang/patches/woq_shim.py:$SP/woq_shim.py:ro" \
    -v "$REPO/sglang/patches/xpu_cudagraph.py:$SP/xpu_cudagraph.py:ro" \
    -e HF_HOME=/hf_cache -e XDG_CACHE_HOME=/sgl_cache -e TORCHINDUCTOR_CACHE_DIR=/sgl_cache/inductor \
    -e B70_XPU_CUDAGRAPH=1 -e B70_XPU_CUDAGRAPH_DEBUG=1 \
    "$IMG" bash -c "source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1; exec python -m sglang.launch_server \
      --model-path '$CKPT' --served-model-name '$SERVED' --trust-remote-code \
      --device xpu --attention-backend triton --linear-attn-backend triton \
      --mamba-ssm-dtype float32 --disable-overlap-schedule --page-size 64 --disable-radix-cache \
      --cuda-graph-bs-decode $bslist --cuda-graph-max-bs-decode $bsmax --max-running-requests $mreq \
      --tp 1 --context-length 4096 --mem-fraction-static 0.90 --host 0.0.0.0 --port $PORT" >>"$LOG" 2>&1

  local ok=0
  for i in $(seq 1 150); do
    docker ps --filter "name=$NAME" --format '{{.Names}}'|grep -q "$NAME" || { say "[$lbl] EXITED"; docker logs "$NAME" 2>&1|grep -iE "error|capture|scratch|Traceback"|tail -6|tee -a "$LOG"; return 0; }
    [ "$(curl -s -o /dev/null -w '%{http_code}' "http://localhost:$PORT/health" 2>/dev/null||echo 000)" = 200 ] && { ok=1; break; }
    sleep 5
  done
  [ "$ok" = 1 ] || { say "[$lbl] not healthy"; docker rm -f "$NAME">/dev/null 2>&1; return 0; }

  # warm c1 (single stream) via bench_serving + the server's per-batch gen-throughput ground truth
  local raw t
  docker exec "$NAME" bash -c "source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1; \
    python -m sglang.bench_serving --backend sglang-oai --host 127.0.0.1 --port $PORT --served-model-name '$SERVED' \
    --tokenizer '$TOK' --dataset-name random --random-input-len 2048 --random-output-len 128 --num-prompts 3 --max-concurrency 1 2>&1" >/dev/null 2>&1 || true
  raw=$(docker exec "$NAME" bash -c "source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1; \
    python -m sglang.bench_serving --backend sglang-oai --host 127.0.0.1 --port $PORT --served-model-name '$SERVED' \
    --tokenizer '$TOK' --dataset-name random --random-input-len 2048 --random-output-len 128 --num-prompts 6 --max-concurrency 1 2>&1")
  t=$(echo "$raw"|grep -i 'Mean TPOT'|grep -oE '[0-9.]+'|head -1)
  say "[$lbl] c1 decode=$(awk -v x="$t" 'BEGIN{if(x>0)printf"%.2f",1000/x;else print"NA"}') t/s"
  say "[$lbl] server gen-throughput (last 6 decode batches):"
  docker logs "$NAME" 2>&1 | grep -oE "gen throughput \(token/s\): [0-9.]+" | tail -6 | sed "s/^/[$lbl]   /" | tee -a "$LOG"
  say "[$lbl] captured-bs evidence (Capturing batches / init_cuda_graph_state):"
  docker logs "$NAME" 2>&1 | grep -iE "Capturing batches|init_cuda_graph_state|graph begin|num_tokens_per_bs" | tail -6 | sed "s/^/[$lbl]   /" | tee -a "$LOG"
  docker rm -f "$NAME" >/dev/null 2>&1
}
for c in "${CFG[@]}"; do IFS='|' read -r l b m r <<< "$c"; run_cfg "$l" "$b" "$m" "$r"; done
say "================= SUMMARY ================="; grep -E "c1 decode=" "$LOG" | tee -a "$LOG.summary"
say "=== concurrency diag DONE -> $LOG ==="
