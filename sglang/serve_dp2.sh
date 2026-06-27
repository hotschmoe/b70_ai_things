#!/usr/bin/env bash
# serve_dp2.sh -- the wedge-proof woq int4 DP=2 daily driver: two single-card sglang replicas
# (card 0 :30000, card 1 :30001) + an nginx round-robin proxy on :18080. CORRECT (sglang GDN fix,
# no "!!!!" NaN), VISION (Lorbus int4 retains the vision tower), ~9.44 t/s/replica warm. No cross-card
# collective -> cannot BCS-wedge (unlike bf16 TP=2). See sglang/PERF.md.
#   start:  ./sglang/serve_dp2.sh start     (holds BOTH cards via two gpu-run --card leases)
#   stop:   ./sglang/serve_dp2.sh stop
#   status: ./sglang/serve_dp2.sh status
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$HERE/.."
IMG="${IMG:-sglang-xpu:woq}"
CKPT="${CKPT:-/models/Lorbus_Qwen3.6-27B-int4-AutoRound}"
SERVED="${SERVED:-qwen36-27b-int4-woq}"
CTX="${CTX:-32768}"; MEMFRAC="${MEMFRAC:-0.9}"

start_replica() { # card port name
  nohup ./bin/gpu-run --card "$1" bash -c "IMG='$IMG' MEMFRAC=$MEMFRAC CTX=$CTX TP=1 DEVICE=$1 \
    NAME='$3' PORT=$2 CKPT='$CKPT' SERVED='$SERVED' bash sglang/serve_sglang.sh start \
    && docker wait '$3'" > "sglang/dp2_card$1.log" 2>&1 &
  disown; echo "  replica card $1 -> :$2 ($3) launching (log sglang/dp2_card$1.log)"
}

case "${1:-start}" in
  start)
    echo "=== sglang woq int4 DP=2 (wedge-proof + vision) ==="
    start_replica 0 30000 sglang_test
    start_replica 1 30001 sglang_test2
    sleep 2
    docker rm -f sglang_dp_proxy >/dev/null 2>&1
    docker run -d --name sglang_dp_proxy -p 18080:18080 \
      -v "$HERE/dp_nginx.conf:/etc/nginx/nginx.conf:ro" nginx:alpine >/dev/null \
      && echo "  proxy -> :18080 (round-robin :30000,:30001)"
    echo "Replicas load in ~3-5 min. Check: ./sglang/serve_dp2.sh status"
    ;;
  stop)
    docker rm -f sglang_test sglang_test2 sglang_dp_proxy >/dev/null 2>&1
    echo "stopped both replicas + proxy"
    ;;
  status)
    docker ps --filter name=sglang_test --filter name=sglang_dp_proxy --format '{{.Names}}\t{{.Status}}'
    for p in 30000 30001 18080; do
      printf ':%s -> %s\n' "$p" "$(curl -s -m 3 -o /dev/null -w '%{http_code}' http://localhost:$p/health 2>/dev/null)"
    done
    ;;
  *) echo "usage: $0 {start|stop|status}"; exit 2 ;;
esac
