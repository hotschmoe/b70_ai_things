#!/usr/bin/env bash
# Qwen3.6-27B W4A16 (compressed-tensors pack-quantized: int4 weights, 16-bit activations) -- TEXT-ONLY quant.
# Self-contained recipe; shared plumbing in ../_common/lib.sh.
#
#   /mnt/vm_8tb/b70/gpu-run --card 0 bash serve.sh    # one card; leaves the other free for experiments
#   bash serve.sh stop
#
# [!] IMAGE: vllm-xpu-env:v0230 (has the GDN/gated-delta-net kernel the Qwen3.5 hybrid LM needs).
# THE FIX (why this serves where stock v0230 does NOT): this checkpoint is a LANGUAGE-MODEL-ONLY quant
# (architectures=["Qwen3_5ForCausalLM"], all 1363 tensors are model.language_model.* + lm_head, ZERO vision
# tensors). vLLM's registry only knows the VL "Qwen3_5ForConditionalGeneration", and _normalize_arch
# suffix-maps our "...ForCausalLM" onto it -> it builds a vision tower that has no weights and whose W4A16
# MLP (input 4304, 4304%128!=0) crashes create_weights. patches/sitecustomize.py registers the EXACT text
# arch (the real class qwen3_5:Qwen3_5ForCausalLM, also used as the VL model's .language_model) so vLLM
# loads text-only and never builds the vision tower. Keeps the 27B in compressed-tensors format (parity).
# (No NOMM: the text-only class is not multimodal, so there is no vision profiling to suppress.)
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export IMG="${IMG:-vllm-xpu-env:v0230}"
export CKPT="${CKPT:-/models/Qwen3.6-27B-W4A16}"
export SERVED="${SERVED:-qwen36-27b-w4a16}"
export DTYPE="${DTYPE:-auto}"
export GRAPH="${GRAPH:-1}"
export UTIL="${UTIL:-0.95}"                  # 24.35 GiB model is VRAM-tight; GRAPH=1 capture OOMs at 0.90 (only
                                            # ~2 GiB KV headroom) -> 0.95 leaves ~4 GiB and captures fine.
export MAXLEN="${MAXLEN:-8192}"
export MAXSEQS="${MAXSEQS:-32}"
export CAPSIZES="${CAPSIZES:-1,2,4,8,16,32}"

# The fix: mount the arch-registration shim and put it on PYTHONPATH so it runs at interpreter startup.
MOUNTS=( -v "$SCRIPT_DIR/patches:/opt/qwen35_text_shim:ro" )
DOCKER_ENV=( -e PYTHONPATH=/opt/qwen35_text_shim )

source "$SCRIPT_DIR/../_common/lib.sh"
b70_dispatch "$@"
