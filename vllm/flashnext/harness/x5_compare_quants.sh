#!/usr/bin/env bash
# Run after the BF16 numerical gate. Each arm independently joins the
# ordinary card-1 lease queue and releases it before the next arm.
set -uo pipefail
REPO=/mnt/vm_8tb/github/b70_ai_things
B=/mnt/vm_8tb/b70/build/vllm-flashnext
cd "$REPO"
if [[ "${SKIP_STOCK:-0}" != 1 ]]; then
  env QUANT=UD-IQ3_XXS MTP_TOKENS=0 B70_X5_GDN_BF16=0 bash vllm/flashnext/harness/x5_proxy_window.sh
  echo IQ3_STOCK_RC=$?
fi
if [[ -f "$B/x5/gdn-bf16/numeric.json" ]]; then
  for q in UD-IQ3_XXS UD-IQ4_XS UD-Q4_K_M; do
    env QUANT="$q" MTP_TOKENS=0 B70_X5_GDN_BF16=1 bash vllm/flashnext/harness/x5_proxy_window.sh
    echo "${q}_GDN_RC=$?"
  done
else
  for q in UD-IQ4_XS UD-Q4_K_M; do
    env QUANT="$q" MTP_TOKENS=0 B70_X5_GDN_BF16=0 bash vllm/flashnext/harness/x5_proxy_window.sh
    echo "${q}_STOCK_RC=$?"
  done
fi
echo X5_QUANT_COMPARISON_DONE
