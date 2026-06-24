#!/usr/bin/env bash
# Optimization lever A/B tests after the kernel-7.0 P2P unlock. Each arm = a model's serve.sh run with
# env overrides + a distinct SERVED/NAME so CSVs are self-labelling. One gpu-run session per phase.
#   ./gpu-run bash 69_lever_tests.sh A    # P2P on/off on 27B W8A8 TP=2
#   ./gpu-run bash 69_lever_tests.sh B    # 35B quark W8A8: eager vs captured vs +P2P
#   ./gpu-run bash 69_lever_tests.sh C    # 35B int4 MoE: MTP off vs on
set -uo pipefail
ROOT=/mnt/vm_8tb/b70; RTS="$ROOT/rdy_to_serve"
export IN="${IN:-2048}" OUT="${OUT:-128}" CONC="${CONC:-1 4}"
PHASE="${1:-A}"

arm() {  # served-label  model-dir  ENV=VAL...
  local label="$1" model="$2"; shift 2
  echo; echo "############## ARM $label :: $model :: $* :: $(date '+%T') ##############"
  if env SERVED="$label" NAME="vllm_$label" IN="$IN" OUT="$OUT" CONC="$CONC" "$@" \
       bash "$RTS/$model/serve.sh" run; then
    echo "ARM $label OK"
  else
    echo "ARM $label FAILED"; bash "$RTS/$model/serve.sh" stop >/dev/null 2>&1 || true
    docker rm -f "vllm_$label" >/dev/null 2>&1 || true
  fi
}

case "$PHASE" in
  A)  # 27B W8A8 TP=2: P2P off vs on. GRAPH=1 (default) -> SYCLKERNELS=1 = the 9.7 GB/s P2P path.
    arm w8a8tp2-p2p0 qwen36-27b-w8a8-sqgptq-mtp P2PACCESS=0
    arm w8a8tp2-p2p1 qwen36-27b-w8a8-sqgptq-mtp P2PACCESS=1
    ;;
  B)  # 35B quark W8A8 TP=2: eager baseline -> captured. (P2P arm dropped: Lever A proved P2PACCESS=1
      # crashes the vLLM TP=2 worker init regardless of model -- H.13.)
    arm quark-eager-p2p0 qwen36-35b-a3b-quark-w8a8-int8 GRAPH=0 P2PACCESS=0
    arm quark-graph-p2p0 qwen36-35b-a3b-quark-w8a8-int8 GRAPH=1 P2PACCESS=0
    ;;
  C)  # 35B-A3B int4 MoE: MTP off vs on (2335 mtp tensors present in the int4 ckpt)
    arm moe-mtp0 qwen36-35b-a3b-int4 MTPTOK=
    arm moe-mtp3 qwen36-35b-a3b-int4 MTPTOK=3 CAPSIZES=1,2,4,6,8
    ;;
  *) echo "usage: 69_lever_tests.sh [A|B|C]"; exit 2 ;;
esac
echo; echo "==== PHASE $PHASE DONE $(date '+%T') ===="