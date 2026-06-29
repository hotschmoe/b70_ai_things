#!/usr/bin/env bash
# Qwen3.6-27B W4A16 (compressed-tensors pack-quantized: 4-bit weight / 16-bit act, TEXT-ONLY) + BF16 MTP graft,
# served TP=2 + MTP spec=3, cudagraph_mode=NONE. Self-contained recipe; shared plumbing in ../_common/lib.sh.
#
#   /mnt/vm_8tb/b70/gpu-run bash serve.sh start       # both cards, wait healthy, coherence-gated probe
#   bash serve.sh stop                                # graceful stop + release both GPUs
#
# Combines TWO proven recipes (campaign 2026-06): ../qwen36-27b-w4a16 (text-only W4A16 arch-reg + load shim,
# :v0230 with GDN baked in, NO .so mount) and ../qwen36-27b-w8a8-sqgptq-mtp (BF16 MTP drafter-unquant + the
# TP=2 NONE path). Both behaviors live in ONE merged patches/sitecustomize.py (Python imports only one per
# interpreter): (a) register text-only Qwen3_5ForCausalLM (else vLLM builds a weightless vision tower / asserts);
# (b) force ONLY the Qwen3_5MultiTokenPredictor drafter unquantized/BF16 (else 0% accept on the grafted mtp.*);
# (c) capture-safe all_gather -- gated OFF on NONE (no graph -> base oneCCL all_gather is correct).
#
# [!] CGMODE=NONE is the STABLE TP=2+MTP mode (campaign 120, W8A8 analog): PIECEWISE crashes under sustained MTP
#     (XPU graph-replay command-stream accumulation, ~20-28k tokens). NONE keeps torch.compile but skips replay.
#     The W4A16-graft+MTP combo was validated 2026-06-23 (accept 68.75%, accept_len 3.75).
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export IMG="${IMG:-vllm-xpu-env:v0230}"                     # v0230 bakes the GDN kernel in (NO .so mount, unlike :int8g)
export CKPT="${CKPT:-/models/qwen3.6-27b/w4a16}"
export SERVED="${SERVED:-qwen36-27b-w4a16-mtp}"
export DTYPE="${DTYPE:-auto}"
export TP="${TP:-2}"
export GRAPH="${GRAPH:-1}"                  # torch.compile/inductor ON ...
export CGMODE="${CGMODE:-NONE}"             # ... but cudagraph_mode=NONE: skip graph replay -> STABLE under MTP
export MTPTOK="${MTPTOK:-3}"                # MTP spec tokens; 3 = the 1-layer-head sweet spot (over-drafts past ~3)
export UTIL="${UTIL:-0.90}"                 # TP=2 splits the 24.35 GiB model ~12 GiB/card -> KV headroom; 0.90 safe.
export MAXLEN="${MAXLEN:-8192}"
export MAXSEQS="${MAXSEQS:-8}"              # MTP is a single-stream latency lever -> conservative concurrency
export CAPSIZES="${CAPSIZES:-1,2,4,6,8}"    # covers the 1+spec verify batch; MOOT on NONE (no capture)
export COMPILESZ="${COMPILESZ-}"           # MUST be empty for spec-decode (compile_sizes [1] is rejected)
# No NOMM (text-only class is not multimodal). No IGP/SPLITOPS override (keep w4a16-proven lib.sh defaults; the
# W8A8 "weight_scale" partitioner KeyError is int8-op-specific, and NONE has no capture boundary anyway).

# csag (capture-safe all_gather) is only needed when the spec-verify all_gather is RECORDED into a graph
# (PIECEWISE/FULL). On NONE there is no capture -> base oneCCL all_gather runs eagerly and is correct.
_CSAG_DEF=$([ "$CGMODE" = NONE ] && echo 1 || echo 0)
export CSAG_DISABLE="${CSAG_DISABLE:-$_CSAG_DEF}"

# Mount the merged shim (arch-reg + MTP-unquant + csag) and its qwen35_text_hybrid helper on PYTHONPATH.
MOUNTS=( -v "$SCRIPT_DIR/patches:/opt/qwen35_mtp_shim:ro" )
DOCKER_ENV=( -e PYTHONPATH=/opt/qwen35_mtp_shim -e CSAG_DISABLE="$CSAG_DISABLE" )

# B70_EXTRA_ENV: space-separated NAME=VAL injected as -e flags (test any env without editing the recipe).
if [ -n "${B70_EXTRA_ENV:-}" ]; then
  for kv in ${B70_EXTRA_ENV}; do DOCKER_ENV+=( -e "$kv" ); done
fi

source "$SCRIPT_DIR/../../_common/lib.sh"
b70_dispatch "$@"
