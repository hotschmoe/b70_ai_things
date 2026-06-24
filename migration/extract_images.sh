#!/usr/bin/env bash
# Recover the exact custom images from the old Unraid docker.img (btrfs graphdriver store) into the
# new Docker on the 8TB data-root. Strategy: stop main docker -> run a throwaway dockerd against the
# old store -> docker save the images to tars -> restart main docker -> docker load.
# Run with: ! sudo bash /home/hotschmoe/extract_images.sh
set -uo pipefail

OLD=/mnt/old_docker
IMGFILE=/mnt/cache/system/docker/docker.img
SOCK=/run/docker-old.sock
EXECROOT=/run/docker-old
PIDF=/run/docker-old.pid
LOG=/var/log/docker-old.log
OUT=/mnt/vm_8tb/docker_image_export
WANT=(vllm-xpu-env:v0230 vllm-xpu-env:int8g)
mkdir -p "$OUT"
echo '{}' > /tmp/empty-daemon.json

cleanup() {
  echo "==> cleanup: stopping throwaway dockerd"
  [ -f "$PIDF" ] && kill "$(cat "$PIDF")" 2>/dev/null || true
  pkill -f "docker-old.sock" 2>/dev/null || true
  sleep 2
}
trap cleanup EXIT

echo "==> [1/6] Mount old store RW (btrfs graphdriver needs RW; the inspector left a RO loop -> rebuild it)"
mkdir -p "$OLD"
# Drop any existing (RO) mount + loop bound to the image, then mount fresh RW.
mountpoint -q "$OLD" && umount "$OLD" || true
for l in $(losetup -j "$IMGFILE" 2>/dev/null | cut -d: -f1); do losetup -d "$l" 2>/dev/null || true; done
mount -o loop,rw "$IMGFILE" "$OLD"
findmnt -no OPTIONS "$OLD" | tr ',' '\n' | grep -qx rw || { echo "ABORT: $OLD did not mount RW"; findmnt "$OLD"; exit 1; }
[ -d "$OLD/btrfs" ] || { echo "ABORT: $OLD/btrfs not found (not a btrfs graphdriver store?)"; exit 1; }
echo "  $OLD mounted RW."

echo "==> [2/6] Stop main docker (frees the containerd 'moby' namespace for the temp daemon)"
systemctl stop docker docker.socket 2>/dev/null || true
sleep 2

echo "==> [3/6] Start throwaway dockerd on the old store (btrfs driver, no networking)"
rm -f "$SOCK"
dockerd --config-file=/tmp/empty-daemon.json \
        --data-root="$OLD" --exec-root="$EXECROOT" --pidfile="$PIDF" \
        --host="unix://$SOCK" --storage-driver=btrfs \
        --bridge=none --iptables=false --ip6tables=false >"$LOG" 2>&1 &
for i in $(seq 1 40); do [ -S "$SOCK" ] && break; sleep 1; done
if ! DOCKER_HOST="unix://$SOCK" docker version >/dev/null 2>&1; then
  echo "ABORT: throwaway dockerd did not come up. Last 30 lines of $LOG:"; tail -30 "$LOG"
  exit 1
fi
echo "  temp dockerd up. Images it can see:"
DOCKER_HOST="unix://$SOCK" docker images | grep -E "REPOSITORY|vllm-xpu-env" | sed 's/^/    /'

echo "==> [4/6] Save the wanted images to tars on the 8TB"
for tag in "${WANT[@]}"; do
  out="$OUT/$(echo "$tag" | tr '/:' '__').tar"
  echo "  saving $tag -> $out"
  if DOCKER_HOST="unix://$SOCK" docker save "$tag" -o "$out"; then
    ls -lh "$out" | sed 's/^/    /'
  else
    echo "    WARN: save failed for $tag (skipping)"
  fi
done

echo "==> [5/6] Stop throwaway dockerd, restart main docker (8TB data-root)"
cleanup
trap - EXIT
systemctl start docker
sleep 3
docker info 2>/dev/null | grep -iE "Docker Root Dir|Storage Driver" | sed 's/^/  /'

echo "==> [6/6] Load the images into main docker"
for tag in "${WANT[@]}"; do
  out="$OUT/$(echo "$tag" | tr '/:' '__').tar"
  [ -f "$out" ] || { echo "  (no tar for $tag, skip)"; continue; }
  echo "  loading $tag"
  docker load -i "$out"
done
echo; echo "=== images now in main docker ==="; docker images | grep -E "REPOSITORY|vllm-xpu-env"
echo
echo "Done. Tars left in $OUT (delete after verifying: rm -rf $OUT). Old store still mounted at $OLD (umount when done)."
