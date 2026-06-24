#!/usr/bin/env bash
# Recover vllm-xpu-env:v0230moe from the old docker.img, then clean up the extraction leftovers.
# SELF-GUARDED: aborts if the shelf bench is still using Docker (extraction restarts Docker).
# Run with: ! sudo bash /home/hotschmoe/extract_moe_and_cleanup.sh
set -uo pipefail
OLD=/mnt/old_docker
IMGFILE=/mnt/cache/system/docker/docker.img
SOCK=/run/docker-old.sock; EXECROOT=/run/docker-old; PIDF=/run/docker-old.pid; LOG=/var/log/docker-old.log
EXPORT=/mnt/vm_8tb/docker_image_export
WANT=(vllm-xpu-env:v0230moe)
echo '{}' > /tmp/empty-daemon.json
mkdir -p "$EXPORT"

# ---- GUARD: don't stop Docker while the bench is using it ----
if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^vllm_' || pgrep -f 68_shelf_bench_par >/dev/null 2>&1; then
  echo "ABORT: shelf bench still running (a vllm_ container or 68_shelf_bench_par is alive)."
  echo "       Extraction stops Docker and would kill it. Re-run this when the bench is done."
  exit 1
fi

cleanup_daemon(){ [ -f "$PIDF" ] && kill "$(cat "$PIDF")" 2>/dev/null || true; sleep 2; }
trap cleanup_daemon EXIT

echo "==> [1/4] Mount old store RW"
mkdir -p "$OLD"
mountpoint -q "$OLD" && umount "$OLD" || true
for l in $(losetup -j "$IMGFILE" 2>/dev/null | cut -d: -f1); do losetup -d "$l" 2>/dev/null || true; done
mount -o loop,rw "$IMGFILE" "$OLD"
findmnt -no OPTIONS "$OLD" | tr ',' '\n' | grep -qx rw || { echo "ABORT: $OLD not RW"; exit 1; }

echo "==> [2/4] Throwaway dockerd on the old store -> save ${WANT[*]}"
systemctl stop docker docker.socket 2>/dev/null || true; sleep 2
rm -f "$SOCK"
dockerd --config-file=/tmp/empty-daemon.json --data-root="$OLD" --exec-root="$EXECROOT" \
        --pidfile="$PIDF" --host="unix://$SOCK" --storage-driver=btrfs \
        --bridge=none --iptables=false --ip6tables=false >"$LOG" 2>&1 &
for i in $(seq 1 40); do [ -S "$SOCK" ] && break; sleep 1; done
DOCKER_HOST="unix://$SOCK" docker version >/dev/null 2>&1 || { echo "ABORT: temp dockerd down"; tail -20 "$LOG"; exit 1; }
for tag in "${WANT[@]}"; do
  out="$EXPORT/$(echo "$tag" | tr '/:' '__').tar"
  echo "  saving $tag -> $out"; DOCKER_HOST="unix://$SOCK" docker save "$tag" -o "$out" && ls -lh "$out" | sed 's/^/    /'
done

echo "==> [3/4] Restart main Docker, load image"
cleanup_daemon; trap - EXIT
systemctl start docker; sleep 3
for tag in "${WANT[@]}"; do
  out="$EXPORT/$(echo "$tag" | tr '/:' '__').tar"
  [ -f "$out" ] && { echo "  loading $tag"; docker load -i "$out"; }
done
echo "  vllm-xpu-env images now:"; docker images | grep -E "REPOSITORY|vllm-xpu-env" | sed 's/^/    /'

echo "==> [4/4] Cleanup extraction leftovers (KEEP docker.img as backup)"
mountpoint -q "$OLD" && umount "$OLD" || true
for l in $(losetup -j "$IMGFILE" 2>/dev/null | cut -d: -f1); do losetup -d "$l" 2>/dev/null || true; done
rmdir "$OLD" 2>/dev/null || true
rm -rf "$EXPORT" && echo "  removed $EXPORT (extraction tars)"
echo
echo "Done. v0230moe recovered; old store unmounted; tars deleted. docker.img kept as backup."
echo "Ping me and I'll bench qwen36-35b-a3b-int4 to complete the table."
