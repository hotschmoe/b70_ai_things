#!/usr/bin/env bash
# Qwen3.6-35B-A3B MoE int4-AutoRound (W4A16 experts) -- FASTEST single-card decode. PIECEWISE captured.
# Self-contained recipe; shared plumbing in ../_common/lib.sh.
#
#   /mnt/vm_8tb/b70/gpu-run bash serve.sh        # start (GRAPH capture), wait healthy, gen-probe, stay up
#   bash serve.sh stop                            # stop + release the GPU
#   GRAPH=1 /mnt/vm_8tb/b70/gpu-run bash serve.sh run   # serve + bench + stop in one lease
#
# IMAGE: vllm-xpu-env:v0230moe (= v0230 + the INC-XPU RoutedExperts->MoeWNA16 patch, BAKED on this leaf
#        tag; see ../../contrib/vllm_moe_xpu/). No runtime patch mount needed. NOT :v0230 (MoE routing
#        unbaked there) and NEVER llm-scaler 0.14.x (no _moe_C).
# Decode ~56.8 t/s captured (fp16 KV) / ~65 t/s with fp8 KV. ~21 GiB model. Fits ONE 32 GB B70.
# Aggregate throughput plateaus ~206 t/s at N>=8 (the routed-expert union approaches all 256 experts).
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export IMG="${IMG:-vllm-xpu-env:v0230moe}"
export CKPT="${CKPT:-/models/qwen3.6-35b-a3b/int4-autoround}"
export SERVED="${SERVED:-qwen36-35b-a3b-int4}"
export GRAPH="${GRAPH:-1}"
export DTYPE="${DTYPE:-auto}"
export UTIL="${UTIL:-0.90}"
export MAXLEN="${MAXLEN:-8192}"
export MAXSEQS="${MAXSEQS:-64}"
export CAPSIZES="${CAPSIZES:-1,2,4,8,16,32,64}"
export KVDTYPE="${KVDTYPE:-fp8_e5m2}"      # fp8-storage KV -> ~65 t/s + 2x ctx/batch (B70 has no FP8 ALU)
export TOOLCALL="${TOOLCALL:-1}"
export TOOLPARSER="${TOOLPARSER:-qwen3_coder}"
export REASONPARSER="${REASONPARSER:-qwen3}"

source "$SCRIPT_DIR/../_common/lib.sh"
b70_dispatch "$@"
