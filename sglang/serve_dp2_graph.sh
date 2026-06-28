#!/usr/bin/env bash
# serve_dp2_graph.sh -- DP=2 of the int4+XPUGraph driver: two single-card graph replicas (card 0 :30000,
# card 1 :30001) behind an nginx round-robin proxy on :18080. Each replica is the FASTEST single-stream
# driver (rdy_to_serve/qwen36-27b-int4-graph, ~23.5 t/s, SAMPLING, vision); DP=2 -> 2 users EACH at full
# ~23.5 t/s (beats MTP-DP2's 2x15). No cross-card collective -> wedge-proof. Mirrors serve_dp2_mtp.sh.
#   start:  ./bin/gpu-run bash sglang/serve_dp2_graph.sh start   (holds BOTH cards via two --card leases)
#   stop:   ./sglang/serve_dp2_graph.sh stop
#   status: ./sglang/serve_dp2_graph.sh status
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; REPO="$(cd "$HERE/.." && pwd)"
R="$REPO/rdy_to_serve/qwen36-27b-int4-graph/serve.sh"
TOK="${TOK:-/models/Qwen_Qwen3.6-27B}"; SERVED=qwen36-27b-int4-graph

start_replica() { # card port name
  nohup ./bin/gpu-run --card "$1" bash -c \
    "DEVICE=$1 PORT=$2 NAME='$3' bash '$R' start && docker wait '$3'" \
    > "$REPO/sglang/dp2graph_card$1.log" 2>&1 &
  disown; echo "  replica card $1 -> :$2 ($3) launching (log sglang/dp2graph_card$1.log)"
}

case "${1:-start}" in
  start)
    cd "$REPO"
    echo "=== sglang int4+XPUGraph DP=2 (both cards, 2 users @ ~23.5 each, wedge-proof + vision) ==="
    start_replica 0 30000 sglang_graph_0
    start_replica 1 30001 sglang_graph_1
    sleep 2
    docker rm -f sglang_graph_proxy >/dev/null 2>&1
    docker run -d --name sglang_graph_proxy -p 18080:18080 \
      -v "$HERE/dp_nginx.conf:/etc/nginx/nginx.conf:ro" nginx:alpine >/dev/null \
      && echo "  proxy -> :18080 (round-robin :30000,:30001)"
    echo "Replicas load + capture in ~4-6 min (each coherence-gated). Check: ./sglang/serve_dp2_graph.sh status"
    ;;
  stop)
    docker rm -f sglang_graph_0 sglang_graph_1 sglang_graph_proxy >/dev/null 2>&1
    echo "stopped both graph replicas + proxy"
    ;;
  status)
    docker ps --filter name=sglang_graph_ --format '{{.Names}}\t{{.Status}}'
    for p in 30000 30001 18080; do
      printf ':%s -> %s\n' "$p" "$(curl -s -m 3 -o /dev/null -w '%{http_code}' http://localhost:$p/health 2>/dev/null)"
    done
    ;;
  *) echo "usage: $0 {start|stop|status}"; exit 2 ;;
esac
