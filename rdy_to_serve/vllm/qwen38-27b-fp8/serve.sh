#!/usr/bin/env bash
# Qualified Qwen3.8-27B official FP8 W8A16 TP2 route for both B70 cards.
#
#   ./bin/gpu-run bash rdy_to_serve/vllm/qwen38-27b-fp8/serve.sh start
#   PROFILE=fast ./bin/gpu-run bash rdy_to_serve/vllm/qwen38-27b-fp8/serve.sh start
#   bash rdy_to_serve/vllm/qwen38-27b-fp8/serve.sh stop
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../../.." && pwd)"
ACTION="${1:-start}"
NAME="${NAME:-qwen38_fp8_daily}"
PROFILE="${PROFILE:-daily}"

if [ "$ACTION" = stop ]; then
  docker rm -f "$NAME" >/dev/null 2>&1 \
    && echo "stopped $NAME" || echo "$NAME not running"
  exit 0
fi

case "$PROFILE" in
  daily)
    MTPTOK="${MTPTOK:-1}"
    PROFILE_ID=mtp1
    ;;
  fast)
    MTPTOK="${MTPTOK:-8}"
    PROFILE_ID=mtp8
    ;;
  *) echo "PROFILE must be daily or fast" >&2; exit 2 ;;
esac
case "$MTPTOK" in
  1|8) ;;
  *) echo "MTPTOK must be 1 or 8" >&2; exit 2 ;;
esac

export IMAGE="${IMAGE:-b70-local/vllm-openai-xpu:qwen38-fp8-mtp8-rms-f08a}"
export EXPECTED_IMAGE_ID="${EXPECTED_IMAGE_ID:-sha256:9ae697d4bbe64338518e8b139ec69e1d101d26bb6766c501c6ef83b022a9d5df}"
export LAYERNORM_SHA256="${LAYERNORM_SHA256:-d911627c6c8f16fc11e02846286c378220120d24bd75898a5337fadf459318bd}"
export MODEL_DIR="${MODEL_DIR:-$REPO/models/files/qwen3.8-27b/fp8-official}"
export CACHE_DIR="${CACHE_DIR:-/mnt/vm_8tb/b70/cache/qwen38-fp8-daily-${PROFILE_ID}}"
export NAME PORT="${PORT:-8078}"
export SERVED="${SERVED:-qwen3.8-27b-FP8-official-W8A16-${PROFILE_ID}-p2p1-fp16kv-daily}"
export P2P_ACCESS=1 I_KNOW_P2P_WEDGES=1
export SPECULATIVE_TOKENS="$MTPTOK"
export RMS_PACKED_SERIAL_EXACT=1 GDN_PERSISTENT_SCRATCH=1
export COMPILATION_PROFILE=publisher
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.96}"
export XPU_GRAPH=1 CUDAGRAPH_MODE=FULL_DECODE_ONLY
export ATTENTION_BACKEND=TRITON_ATTN DRAFT_ATTENTION_BACKEND=TRITON_ATTN
export FORCE_GRAPH_WITH_COMM=1 USE_V2_MODEL_RUNNER=0
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-32768}"
export ALLOW_EXISTING_CACHE=1
export INDUCTOR_COMBO_KERNELS=0 INDUCTOR_BENCHMARK_COMBO_KERNEL=0
export INDUCTOR_MAX_AUTOTUNE=0 INDUCTOR_COORDINATE_DESCENT_TUNING=0
export INDUCTOR_AUTOTUNE_POINTWISE=0 INDUCTOR_DETERMINISTIC_CONFIG=1

case "$ACTION" in
  start|run) exec "$REPO/vllm/fp8/serve_qwen38_fp8_neural_f02.sh" run ;;
  --print-config) exec "$REPO/vllm/fp8/serve_qwen38_fp8_neural_f02.sh" --print-config ;;
  *) echo "usage: $0 start|run|stop|--print-config" >&2; exit 2 ;;
esac
