#!/usr/bin/env bash
# models/graft_w4_complete.sh -- build the COMPLETE (vision+MTP) W4A16 and W4A8 checkpoints
# straight into models/files. CPU-only graft inside sglang-xpu:woq (~1-2 min each).
#
# RUN AFTER reorg.sh (it needs files/qwen3.6-27b/bf16 as the vision/config source).
#   sudo bash models/graft_w4_complete.sh            # dry run (prints plan + docker cmd)
#   sudo APPLY=1 bash models/graft_w4_complete.sh    # execute graft, chown, drop sources
#
# Consumes (and on success drops): the W4A16/W4A8 quant bases + their -mtp-graft dirs.
# NOTE: the resulting builds are mechanically complete but UNVERIFIED on-GPU -- coherence-gate
# them (serve + a few prompts + an image) before shelf promotion, per the repo's gate discipline.
set -uo pipefail
SRCROOT=/mnt/vm_8tb/b70
REPO=/mnt/vm_8tb/github/b70_ai_things
FILES="$REPO/models/files"
IMG="${IMG:-sglang-xpu:woq}"
APPLY="${APPLY:-0}"
UID_KEEP=1000; GID_KEEP=1000

# out_scheme | IN (mtp-graft dir, under /models) | consumed source dirs (space-sep, under models/)
JOBS=(
  "w4a16|Qwen3.6-27B-W4A16-mtp-graft|Qwen3.6-27B-W4A16 Qwen3.6-27B-W4A16-mtp-graft"
  "w4a8-sqgptq|Qwen3.6-27B-W4A8-sqgptq-prepacked-mtp-graft|Qwen3.6-27B-W4A8-sqgptq-prepacked Qwen3.6-27B-W4A8-sqgptq-prepacked-mtp-graft"
)

echo "MODE: $([ "$APPLY" = 1 ] && echo APPLY || echo DRY-RUN)   IMG=$IMG"
[ -d "$FILES/qwen3.6-27b/bf16" ] || { echo "ERROR: $FILES/qwen3.6-27b/bf16 missing -- run reorg.sh first."; exit 1; }

for job in "${JOBS[@]}"; do
  IFS='|' read -r scheme in_dir consumed <<<"$job"
  OUT="$FILES/qwen3.6-27b/$scheme"
  echo "----------------------------------------------------------------------"
  echo ">> graft $scheme   (IN=/models/$in_dir, BF16=/out/qwen3.6-27b/bf16 -> /out/qwen3.6-27b/$scheme)"
  if [ ! -d "$SRCROOT/models/$in_dir" ]; then echo "   SKIP: source /models/$in_dir missing"; continue; fi
  if [ -d "$OUT" ] && [ -n "$(ls -A "$OUT" 2>/dev/null)" ]; then echo "   SKIP: $OUT exists, non-empty"; continue; fi
  cmd=(docker run --rm -u 0 --entrypoint python3
       -v "$SRCROOT/models:/models"
       -v "$FILES:/out"
       -v "$REPO/models/graft_complete.py:/graft.py:ro"
       "$IMG" /graft.py "/models/$in_dir" "/out/qwen3.6-27b/bf16" "/out/qwen3.6-27b/$scheme")
  printf '   %s\n' "${cmd[*]}"
  if [ "$APPLY" = 1 ]; then
    "${cmd[@]}" || { echo "GRAFT FAILED ($scheme)"; exit 2; }
    chown -R "$UID_KEEP:$GID_KEEP" "$OUT"
    if find "$OUT" -type l | grep -q .; then echo "!! symlinks in $OUT -- ABORT"; exit 4; fi
    echo "   drop consumed sources:"
    for c in $consumed; do
      p="$SRCROOT/models/$c"; [ -e "$p" ] || continue
      echo "     rm -rf models/$c"; rm -rf "$p"
    done
  fi
done
echo "----------------------------------------------------------------------"
echo "DONE ($([ "$APPLY" = 1 ] && echo APPLIED || echo dry-run)). Coherence-gate w4a16/w4a8 before shelf promotion."
