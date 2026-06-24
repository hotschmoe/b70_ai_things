#!/usr/bin/env bash
# Safely inspect the old Unraid Docker store (read-only loop mount) to confirm the images inside
# and the storage driver, so we can extract vllm-xpu-env:v0230 / :int8g exactly.
# Run with: ! sudo bash /home/hotschmoe/inspect_dockerimg.sh
set -euo pipefail
IMG=/mnt/cache/system/docker/docker.img
MNT=/mnt/old_docker

mkdir -p "$MNT"
if ! mountpoint -q "$MNT"; then
  mount -o ro,loop "$IMG" "$MNT"
  echo "mounted $IMG (RO) at $MNT"
else
  echo "$MNT already mounted"
fi

echo; echo "=== top-level of the old docker data-root ==="
ls -1 "$MNT" | sed 's/^/  /'

echo; echo "=== storage driver in use ==="
DRV=""
for d in btrfs overlay2 overlay vfs aufs; do [ -d "$MNT/$d" ] && DRV="$d" && echo "  -> $d"; done
[ -n "$DRV" ] || echo "  (no known driver dir found)"

echo; echo "=== image tags present (repositories.json) ==="
for rj in "$MNT"/image/*/repositories.json; do
  [ -f "$rj" ] || continue
  echo "  ($rj)"
  python3 -c "
import json,sys
d=json.load(open('$rj'))
for repo,tags in d.get('Repositories',{}).items():
    for t in tags: print('   ',t)
" 2>/dev/null || sed 's/,/,\n/g' "$rj"
done

echo; echo "=== disk usage of the loop fs ==="; df -h "$MNT" | tail -1
echo
echo "Left mounted RO at $MNT for the extraction step."
echo "We specifically want: vllm-xpu-env:v0230 (allreduce bench) and vllm-xpu-env:int8g (W8A8 serve A/B)."
