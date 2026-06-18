#!/usr/bin/env bash
# Finish the docker.img btrfs grow: ensure the loop device reflects the 200G backing
# file, then resize btrfs to fill it. Non-destructive.
set -uo pipefail
echo "=== loop device backing + size ==="
losetup -a | grep -i docker
echo "loop2 bytes: $(blockdev --getsize64 /dev/loop2 2>/dev/null)"
echo "file bytes:  $(stat -c %s /mnt/cache/system/docker/docker.img)"

echo "=== refresh loop capacity to match backing file ==="
losetup -c /dev/loop2 && echo "losetup -c ok"
echo "loop2 bytes after refresh: $(blockdev --getsize64 /dev/loop2 2>/dev/null)"

echo "=== btrfs resize max ==="
btrfs filesystem resize max /var/lib/docker

echo "=== RESULT ==="
df -h /var/lib/docker | tail -1
btrfs filesystem usage /var/lib/docker 2>/dev/null | grep -iE 'device size|free' | head -3
echo "=== DONE ==="
