#!/usr/bin/env bash
# X5 selected 4-bit proxy recipe, in a bounded health/RAM-guarded window.
# MTP is optional; see reports/1X5_proxy35b.md for the qualified depths.
set -euo pipefail
REPO=/mnt/vm_8tb/github/b70_ai_things
B=/mnt/vm_8tb/b70/build/vllm-flashnext
export QUANT=${QUANT:-UD-Q4_K_M} MTP_TOKENS=${MTP_TOKENS:-0}
export CARD=1 PORT=${PORT:-18125}
export B70_AGENT=X5 B70_GPU_LOCK_TIMEOUT=21600 B70_CPU_LOCK_TIMEOUT=21600
export B70_X5_GDN_BF16=${B70_X5_GDN_BF16:-1}
export B70_D3_GDN_SPEC=${B70_D3_GDN_SPEC:-0}
export B70_X5_GDN_MIXED=${B70_X5_GDN_MIXED:-0}
export B70_X5_GDN_RAGGED=${B70_X5_GDN_RAGGED:-0}
export B70_X5_EVENTLESS=1 B70_X5_EVENTLESS_LAZY=1
export B70_X5_PACK_LINEAR=1 B70_X5_Q8_LINEAR=1 B70_X5_CANONICAL=1
export B70_X5_BATCH_LINEAR=${B70_X5_BATCH_LINEAR:-0} B70_X5_Q8_FAST=1
BASE_TAG=qwen36-35b-${QUANT,,}-fastq8-canon-mtp0
if [[ "$QUANT" == UD-Q4_K_M ]]; then BASE_TAG=qwen36-35b-udq4km-fastq8-canon-mtp0; fi
if [[ "$B70_X5_BATCH_LINEAR" == 1 ]]; then BASE_TAG=${BASE_TAG/-canon-mtp0/-bi-mtp0}; fi
if [[ "$B70_X5_GDN_MIXED" == 1 ]]; then BASE_TAG=${BASE_TAG%-mtp0}-mixed-mtp0; fi
if [[ "$B70_X5_GDN_RAGGED" == 1 ]]; then BASE_TAG=${BASE_TAG%-mtp0}-ragged-mtp0; fi
if [[ "$B70_X5_GDN_BF16" == 0 ]]; then BASE_TAG=${BASE_TAG%-mtp0}-stockgdn-mtp0; fi
export X5_BASELINE=${X5_BASELINE:-$B/x5/$BASE_TAG/greedy.json}
export TAG=${TAG:-${BASE_TAG%-mtp0}-mtp${MTP_TOKENS}-$(date -u +%Y%m%dT%H%M%S)}
exec bash "$REPO/vllm/flashnext/harness/x5_proxy_window.sh"
