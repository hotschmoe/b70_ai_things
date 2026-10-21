#!/usr/bin/env bash
# Final fixed-GDN quant round; release the lease after every bounded window.
set -euo pipefail
REPO=/mnt/vm_8tb/github/b70_ai_things
B=/mnt/vm_8tb/b70/build/vllm-flashnext
export B70_AGENT=X5 B70_GPU_LOCK_TIMEOUT=21600 B70_CPU_LOCK_TIMEOUT=21600
export B70_X5_BATCH_LINEAR=0 B70_D3_GDN_SPEC=1
export B70_X5_GDN_MIXED=1 B70_X5_GDN_RAGGED=1
while ! grep -q 'xpu-health: HEALTHY' "$B/x5/gdn-ragged/r3/health-after.txt" 2>/dev/null; do
  sleep 2
done
for quant in UD-IQ3_XXS UD-IQ4_XS UD-Q4_K_M; do
  stem=qwen36-35b-${quant,,}-fastq8-canon-mixed-ragged
  lmx=0
  if [[ "$quant" == UD-Q4_K_M ]]; then
    stem=qwen36-35b-udq4km-fastq8-canon-mixed-ragged
    lmx=1
  fi
  echo "X5_FINAL_START $quant mtp0 $(date -u +%FT%TZ)"
  env QUANT="$quant" MTP_TOKENS=0 TAG=$stem-mtp0 X5_LMX="$lmx" \
    bash "$REPO/vllm/flashnext/harness/x5_recipe_35b.sh"
done
for depth in 2 3; do
  echo "X5_FINAL_START UD-Q4_K_M mtp$depth $(date -u +%FT%TZ)"
  env QUANT=UD-Q4_K_M MTP_TOKENS="$depth" X5_LMX=0 \
    TAG=qwen36-35b-udq4km-fastq8-canon-mixed-ragged-mtp$depth \
    bash "$REPO/vllm/flashnext/harness/x5_recipe_35b.sh"
done
echo "X5_FINAL_ALL_DONE $(date -u +%FT%TZ)"
