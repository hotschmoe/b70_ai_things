#!/usr/bin/env bash
# Prep for the shelf bench: (1) make results/ writable by the bench user, (2) recover the last image
# (vllm-xpu-env:v0230moe) from the old Unraid docker.img.
# Run with: ! sudo bash /home/hotschmoe/prep_bench.sh
set -uo pipefail
USER_NAME="${SUDO_USER:-hotschmoe}"
OLD=/mnt/old_docker
IMGFILE=/mnt/cache/system/docker/docker.img
SOCK=/run/docker-old.sock; EXECROOT=/run/docker-old; PIDF=/run/docker-old.pid; LOG=/var/log/docker-old.log
OUT=/mnt/vm_8tb/docker_image_export
WANT=(vllm-xpu-env:v0230moe)
echo '{}' > /tmp/empty-daemon.json
mkdir -p "$OUT"

echo "==> [1/2] Make results/ writable by $USER_NAME (root-owned from Unraid era)"
chown -R "$USER_NAME":"$USER_NAME" /mnt/vm_8tb/b70/results 2>/dev/null || true
chmod -R u+rwX /mnt/vm_8tb/b70/results
echo "  results/ ok"

cleanup(){ [ -f "$PIDF" ] && kill "$(cat "$PIDF")" 2>/dev/null || true; sleep 2; }
trap cleanup EXIT

echo "==> [2/2] Recover ${WANT[*]} from docker.img"
mkdir -p "$OLD"
mountpoint -q "$OLD" && umount "$OLD" || true
for l in $(losetup -j "$IMGFILE" 2>/dev/null | cut -d: -f1); do losetup -d "$l" 2>/dev/null || true; done
mount -o loop,rw "$IMGFILE" "$OLD"
findmnt -no OPTIONS "$OLD" | tr ',' '\n' | grep -qx rw || { echo "ABORT: $OLD not RW"; exit 1; }

systemctl stop docker docker.socket 2>/dev/null || true; sleep 2
rm -f "$SOCK"
dockerd --config-file=/tmp/empty-daemon.json --data-root="$OLD" --exec-root="$EXECROOT" \
        --pidfile="$PIDF" --host="unix://$SOCK" --storage-driver=btrfs \
        --bridge=none --iptables=false --ip6tables=false >"$LOG" 2>&1 &
for i in $(seq 1 40); do [ -S "$SOCK" ] && break; sleep 1; done
DOCKER_HOST="unix://$SOCK" docker version >/dev/null 2>&1 || { echo "ABORT: temp dockerd down"; tail -20 "$LOG"; exit 1; }

for tag in "${WANT[@]}"; do
  out="$OUT/$(echo "$tag" | tr '/:' '__').tar"
  echo "  saving $tag -> $out"
  DOCKER_HOST="unix://$SOCK" docker save "$tag" -o "$out" && ls -lh "$out" | sed 's/^/    /'
done

cleanup; trap - EXIT
systemctl start docker; sleep 3
for tag in "${WANT[@]}"; do
  out="$OUT/$(echo "$tag" | tr '/:' '__').tar"
  [ -f "$out" ] && { echo "  loading $tag"; docker load -i "$out"; }
done
echo; echo "=== vllm-xpu-env images now present ==="; docker images | grep -E "REPOSITORY|vllm-xpu-env"
echo "Prep done. Ready for the shelf bench."
