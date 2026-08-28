#!/usr/bin/env bash
# Current vLLM 0.28 text-only control for the pinned XeCores Qwen3.8 GPTQ
# INT4 artifact. The checkpoint's dynamic exclusion already keeps every MTP
# tensor BF16, so no legacy BF16-draft patch is applied.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"

export IMG="${IMG:-vllm/vllm-openai-xpu@sha256:4756b66a077627133cee653b551f6f5eaa1b9a981b5eea13edd33fcd3b0d3ca3}"
export CKPT="${CKPT:-/models/qwen3.8-27b/gptq-int4-mtp-bf16-9d189a60}"
export SERVED="${SERVED:-qwen3.8-27b-GPTQ-INT4-g128-target-vllm028-tp1}"
export NAME="${NAME:-qwen38_gptq_int4_v028}"
export PORT="${PORT:-18080}"
export TP="${TP:-1}"
export PP="${PP:-1}"
export DEVICE="${DEVICE:-0}"
export GRAPH="${GRAPH:-0}"
export MAXLEN="${MAXLEN:-4096}"
export MAXSEQS="${MAXSEQS:-1}"
export UTIL="${UTIL:-0.75}"
export PREFIXCACHE="${PREFIXCACHE:-0}"
export NOMM="${NOMM:-1}"
export QUANT="${QUANT:-gptq}"
export REASONPARSER="${REASONPARSER:-qwen3}"
export EXTRA_ARGS="${EXTRA_ARGS:---language-model-only --generation-config vllm --no-async-scheduling --uvicorn-log-level warning}"
export IN="${IN:-512}"
export OUT="${OUT:-512}"
export CONC="${CONC:-1}"
DRAFT_LMHEAD_INT4="${DRAFT_LMHEAD_INT4:-0}"
DRAFT_MTP_INT4="${DRAFT_MTP_INT4:-0}"

HOST_CKPT="$REPO/models/files/${CKPT#/models/}"
test -f "$HOST_CKPT/model.safetensors.index.json" || {
  echo "Missing checkpoint: $HOST_CKPT" >&2
  exit 1
}
for shard in 1 2 3 4 5; do
  shard_file="$(printf 'model-%05d-of-00005.safetensors' "$shard")"
  test -f "$HOST_CKPT/$shard_file" || {
    echo "Missing GPTQ shard $shard in $HOST_CKPT" >&2
    exit 1
  }
done
case "$TP" in
  1|2) ;;
  *) echo "This research launcher supports TP=1 or TP=2, got TP=$TP" >&2; exit 1 ;;
esac
case "${MTPTOK:-}" in
  "")
    case "$SERVED" in
      *target*) ;;
      *) echo "Target-only served ID must contain target: $SERVED" >&2; exit 1 ;;
    esac
    ;;
  1|2|4)
    case "$SERVED" in
      *mtp"$MTPTOK"*) ;;
      *) echo "MTP served ID must contain mtp$MTPTOK: $SERVED" >&2; exit 1 ;;
    esac
    ;;
  *) echo "MTPTOK must be empty, 1, 2, or 4, got ${MTPTOK:-}" >&2; exit 1 ;;
esac
if [ "$TP" = 2 ] && [ "${P2PACCESS:-0}" != 0 ]; then
  echo "TP2 GPTQ INT4 requires P2PACCESS=0" >&2
  exit 1
fi
case "$DRAFT_LMHEAD_INT4" in
  0|1) ;;
  *) echo "DRAFT_LMHEAD_INT4 must be 0 or 1" >&2; exit 1 ;;
esac
if [ "$DRAFT_LMHEAD_INT4" = 1 ]; then
  test -n "${MTPTOK:-}" || {
    echo "DRAFT_LMHEAD_INT4=1 requires MTPTOK" >&2
    exit 1
  }
  case "$SERVED" in
    *draft-lmhead-int4*) ;;
    *) echo "Draft INT4 served ID must contain draft-lmhead-int4" >&2; exit 1 ;;
  esac
fi
case "$DRAFT_MTP_INT4" in
  0|1) ;;
  *) echo "DRAFT_MTP_INT4 must be 0 or 1" >&2; exit 1 ;;
esac
if [ "$DRAFT_MTP_INT4" = 1 ]; then
  test -n "${MTPTOK:-}" || {
    echo "DRAFT_MTP_INT4=1 requires MTPTOK" >&2
    exit 1
  }
  case "$SERVED" in
    *draft-mtp-int4*) ;;
    *) echo "Draft MTP INT4 served ID must contain draft-mtp-int4" >&2; exit 1 ;;
  esac
fi

MOUNTS=()
DOCKER_ENV=()
if [ "$DRAFT_LMHEAD_INT4" = 1 ] || [ "$DRAFT_MTP_INT4" = 1 ]; then
  PATCH_DIR="$SCRIPT_DIR/v028_draft_int4"
  test -f "$PATCH_DIR/sitecustomize.py" || {
    echo "Missing vLLM 0.28 draft overlay: $PATCH_DIR/sitecustomize.py" >&2
    exit 1
  }
  MOUNTS+=( -v "$PATCH_DIR:/b70_qwen38_gptq_v028:ro" )
  DOCKER_ENV+=(
    -e PYTHONPATH=/b70_qwen38_gptq_v028
    -e B70_DRAFT_LMHEAD_INT4="$DRAFT_LMHEAD_INT4"
    -e B70_DRAFT_LMHEAD_INT4_CHUNK_ROWS="${DRAFT_LMHEAD_INT4_CHUNK_ROWS:-1024}"
    -e B70_DRAFT_MTP_INT4="$DRAFT_MTP_INT4"
    -e B70_DRAFT_MTP_INT4_CHUNK_ROWS="${DRAFT_MTP_INT4_CHUNK_ROWS:-1024}"
  )
fi
if [ "${BREAKABLE:-0}" = 1 ]; then
  [ "$GRAPH" = 1 ] || {
    echo "BREAKABLE=1 requires GRAPH=1" >&2
    exit 1
  }
  DOCKER_ENV+=(
    -e VLLM_USE_BREAKABLE_CUDAGRAPH=1
    -e VLLM_USE_AOT_COMPILE=0
  )
fi

source "$REPO/rdy_to_serve/_common/lib.sh"

b70_bench() {
  local par="tp${TP}"
  env NAME="$NAME" MODEL="$SERVED" LABEL="${SERVED}-${par}" \
    TOKPATH="$CKPT" PORT="$PORT" IN="$IN" OUT="$OUT" CONC="$CONC" \
    bash "$REPO/bin/35_sweep_bench.sh"
}

b70_dispatch "$@"
