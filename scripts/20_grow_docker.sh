#!/usr/bin/env bash
# NON-DESTRUCTIVE grow of the Unraid docker.img 50GB -> 200GB, in place on the NVMe
# cache. Preserves all images/containers. Run only when OK to bounce Docker (stops
# nextcloud/mariadb/syncthing + our vllm container briefly). Idempotent-ish.
set -uo pipefail
NEW_GB=200
CFG=/boot/config/docker.cfg
IMGFILE=$(losetup -a 2>/dev/null | grep -i docker | sed -E 's/.*\((.*)\)/\1/')
[ -z "$IMGFILE" ] && IMGFILE="/mnt/cache/system/docker/docker.img"

echo "=== docker.img = $IMGFILE ==="
ls -lh --block-size=1 "$IMGFILE" | awk '{print "current file bytes:",$5}'
df -h /var/lib/docker | tail -1
echo "=== cache free (need headroom for grow) ==="; df -h "$(dirname "$IMGFILE")" | tail -1

echo "=== backup docker.cfg ==="
cp -a "$CFG" "${CFG}.bak.$(date +%s)" && echo "backed up $CFG"

echo "=== stopping Docker (this bounces all containers) ==="
/etc/rc.d/rc.docker stop
sleep 3
docker ps 2>/dev/null | tail -2 || echo "(docker down)"

echo "=== growing image file to ${NEW_GB}G (truncate only extends; data preserved) ==="
truncate -s ${NEW_GB}G "$IMGFILE"
ls -lh "$IMGFILE"

echo "=== set DOCKER_IMAGE_SIZE=${NEW_GB} in cfg ==="
sed -i -E "s/^DOCKER_IMAGE_SIZE=.*/DOCKER_IMAGE_SIZE=\"${NEW_GB}\"/" "$CFG"
grep DOCKER_IMAGE_SIZE "$CFG"

echo "=== starting Docker ==="
/etc/rc.d/rc.docker start
sleep 8

echo "=== grow btrfs to fill the larger file ==="
btrfs filesystem resize max /var/lib/docker 2>&1 || echo "(resize note above)"
echo "=== RESULT ==="
df -h /var/lib/docker | tail -1
echo "=== containers back up? ==="
docker ps --format '{{.Names}}  {{.Status}}' 2>/dev/null
echo "=== DONE ==="
