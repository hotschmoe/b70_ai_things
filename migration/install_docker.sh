#!/usr/bin/env bash
# Install Docker with data-root on the 8TB SSD (per Isaac: models+images+workdirs live on vm_8tb).
# We will recover the EXACT old images from /mnt/cache/system/docker/docker.img (see extract step),
# so NO image pull here.
# Run with: ! sudo bash /home/hotschmoe/install_docker.sh
set -euo pipefail
USER_NAME="${SUDO_USER:-hotschmoe}"
DATA_ROOT=/mnt/vm_8tb/docker

echo "==> [1/4] Install Docker (26.04 archive: docker.io 29.x + buildx + compose)"
apt-get update -y
apt-get install -y docker.io docker-buildx docker-compose-v2 containerd

echo "==> [2/4] Point Docker data-root at $DATA_ROOT (8TB SSD)"
mkdir -p "$DATA_ROOT" /etc/docker
systemctl stop docker docker.socket 2>/dev/null || true
# merge-safe: write a minimal daemon.json (vm_8tb is btrfs -> docker auto-selects the btrfs storage driver)
cat > /etc/docker/daemon.json <<EOF
{
  "data-root": "$DATA_ROOT"
}
EOF

echo "==> [3/4] Start Docker + add $USER_NAME to docker group"
systemctl daemon-reload
systemctl enable --now docker
usermod -aG docker "$USER_NAME"

echo "==> [4/4] Verify"
docker --version
docker info 2>/dev/null | grep -iE "Docker Root Dir|Storage Driver"
echo
echo "Docker up, data-root on $DATA_ROOT. $USER_NAME added to 'docker' group -> RE-LOGIN for non-sudo docker."
echo "Next: recover the old images from docker.img (inspect_dockerimg.sh)."
