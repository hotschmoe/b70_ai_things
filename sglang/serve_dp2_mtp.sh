#!/usr/bin/env bash
# serve_dp2_mtp.sh -- PERF PUSH lever 2: use BOTH B70s. Two single-card int4+MTP replicas
# (card 0 :30000, card 1 :30001) behind an nginx round-robin proxy on :18080. Each replica is the
# shipped latency driver (rdy_to_serve/qwen36-27b-int4-mtp, ~15.3 t/s single-stream, greedy, vision);
# DP=2 ~2x the aggregate / interactive-slot capacity. No cross-card collective -> cannot BCS-wedge
# (unlike bf16 TP=2). Mirrors serve_dp2.sh but drives the MTP recipe (baked sglang-xpu:mtp, no mounts).
#   start:  ./bin/gpu-run bash sglang/serve_dp2_mtp.sh start   (holds BOTH cards via two --card leases)
#   stop:   ./sglang/serve_dp2_mtp.sh stop
#   status: ./sglang/serve_dp2_mtp.sh status
#   bench:  ./sglang/serve_dp2_mtp.sh bench   (aggregate throughput vs the proxy :18080)
# TC=1 enables --enable-torch-compile per replica (only if lever 1 proved it helps -- default OFF).
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; REPO="$(cd "$HERE/.." && pwd)"
R="$REPO/rdy_to_serve/qwen36-27b-int4-mtp/serve.sh"
TOK="${TOK:-/models/Qwen_Qwen3.6-27B}"; SERVED=qwen36-27b-int4-mtp-nextn

start_replica() { # card port name
  nohup ./bin/gpu-run --card "$1" bash -c \
    "DEVICE=$1 PORT=$2 NAME='$3' bash '$R' start && docker wait '$3'" \
    > "$REPO/sglang/dp2mtp_card$1.log" 2>&1 &
  disown; echo "  replica card $1 -> :$2 ($3) launching (log sglang/dp2mtp_card$1.log)"
}

case "${1:-start}" in
  start)
    cd "$REPO"
    echo "=== sglang int4+MTP DP=2 (both cards, ~2x aggregate, wedge-proof + vision) ==="
    start_replica 0 30000 sglang_mtp_0
    start_replica 1 30001 sglang_mtp_1
    sleep 2
    docker rm -f sglang_mtp_proxy >/dev/null 2>&1
    docker run -d --name sglang_mtp_proxy -p 18080:18080 \
      -v "$HERE/dp_nginx.conf:/etc/nginx/nginx.conf:ro" nginx:alpine >/dev/null \
      && echo "  proxy -> :18080 (round-robin :30000,:30001)"
    echo "Replicas load in ~3-5 min (each coherence-gated). Check: ./sglang/serve_dp2_mtp.sh status"
    ;;
  stop)
    docker rm -f sglang_mtp_0 sglang_mtp_1 sglang_mtp_proxy >/dev/null 2>&1
    echo "stopped both MTP replicas + proxy"
    ;;
  status)
    docker ps --filter name=sglang_mtp_ --format '{{.Names}}\t{{.Status}}'
    for p in 30000 30001 18080; do
      printf ':%s -> %s\n' "$p" "$(curl -s -m 3 -o /dev/null -w '%{http_code}' http://localhost:$p/health 2>/dev/null)"
    done
    ;;
  bench)
    # aggregate throughput against the proxy; sweep concurrency to show the 2-replica ceiling.
    docker exec sglang_mtp_0 bash -c "source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1; \
      python -m sglang.bench_serving --backend sglang-oai --host 172.17.0.1 --port 18080 \
      --served-model-name '$SERVED' --tokenizer '$TOK' --dataset-name random \
      --random-input-len 2048 --random-output-len 128 --num-prompts ${NP:-16} --max-concurrency ${MC:-8}"
    ;;
  *) echo "usage: $0 {start|stop|status|bench}"; exit 2 ;;
esac
