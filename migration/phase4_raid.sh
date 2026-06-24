#!/usr/bin/env bash
# Phase 4 -- mergerfs pool (disk1+disk2 -> /mnt/storage) + SnapRAID parity (reuse the freed Unraid parity disk).
# Run with: ! sudo bash /home/hotschmoe/phase4_raid.sh
#
# SAFETY: the parity disk is identified by SERIAL, never by /dev/sdX (letters reshuffled in the migration:
# the parity disk JEH9VZHN is now /dev/sda, while /dev/sdb is disk1's LIVE DATA). The format step ABORTS
# unless the target's serial == JEH9VZHN AND it has no filesystem. Disk1/disk2/vm_8tb/cache are never touched.
set -euo pipefail

PARITY_SERIAL=JEH9VZHN          # the ONLY disk this script will format
POOL_MOUNT=/mnt/storage
PARITY_MOUNT=/mnt/parity1

echo "==> [1/6] Install mergerfs + snapraid"
apt-get update -y
apt-get install -y mergerfs snapraid

echo "==> [2/6] mergerfs pool $POOL_MOUNT = /mnt/disk1 + /mnt/disk2"
mountpoint -q /mnt/disk1 && mountpoint -q /mnt/disk2 || { echo "ABORT: /mnt/disk1 or /mnt/disk2 not mounted (run Phase 3 first)"; exit 1; }
mkdir -p "$POOL_MOUNT"
if ! grep -q "[[:space:]]$POOL_MOUNT[[:space:]]*fuse.mergerfs" /etc/fstab; then
  echo "/mnt/disk1:/mnt/disk2  $POOL_MOUNT  fuse.mergerfs  defaults,allow_other,use_ino,category.create=mfs,minfreespace=20G,nofail  0 0" >> /etc/fstab
  echo "  added mergerfs line to /etc/fstab"
else
  echo "  mergerfs already in /etc/fstab"
fi
systemctl daemon-reload
mountpoint -q "$POOL_MOUNT" || mount "$POOL_MOUNT"
echo "  pooled:"; df -h "$POOL_MOUNT"; ls "$POOL_MOUNT"

echo "==> [3/6] Locate the parity disk by SERIAL=$PARITY_SERIAL (NOT by sdX)"
PDISK=""
for d in /dev/sd?; do
  s=$(lsblk -ndo SERIAL "$d" 2>/dev/null || true)
  [ "$s" = "$PARITY_SERIAL" ] && PDISK="$d" && break
done
[ -n "$PDISK" ] || { echo "ABORT: no disk with serial $PARITY_SERIAL found"; exit 1; }
PPART="${PDISK}1"
echo "  parity disk = $PDISK (serial $PARITY_SERIAL), partition = $PPART"
# Triple-check before destroying anything:
[ "$(lsblk -ndo SERIAL "$PDISK")" = "$PARITY_SERIAL" ] || { echo "ABORT: serial recheck failed"; exit 1; }
FS=$(lsblk -ndo FSTYPE "$PPART" 2>/dev/null || true)
[ -z "$FS" ] || { echo "ABORT: $PPART has a filesystem ('$FS') -- refusing to format (expected the empty Unraid parity)"; exit 1; }
echo "  $PPART has NO filesystem -- confirmed safe to format."

echo "==> [4/6] Format parity $PPART as xfs and mount at $PARITY_MOUNT"
wipefs -a "$PPART"
mkfs.xfs -f "$PPART"
PUUID=$(blkid -s UUID -o value "$PPART")
echo "  new parity UUID = $PUUID"
mkdir -p "$PARITY_MOUNT"
if ! grep -q "$PUUID" /etc/fstab; then
  echo "UUID=$PUUID  $PARITY_MOUNT  xfs  defaults,noatime,nofail  0 2" >> /etc/fstab
  echo "  added parity to /etc/fstab"
fi
systemctl daemon-reload
mountpoint -q "$PARITY_MOUNT" || mount "$PARITY_MOUNT"
df -h "$PARITY_MOUNT"

echo "==> [5/6] Write /etc/snapraid.conf"
[ -f /etc/snapraid.conf ] && cp /etc/snapraid.conf /etc/snapraid.conf.bak.$(date +%s) || true
cat > /etc/snapraid.conf <<EOF
# SnapRAID -- disk1+disk2 protected by one parity disk (the reused Unraid parity, serial $PARITY_SERIAL).
# vm_8tb (models/repo) and cache (scratch) are intentionally OUTSIDE parity.
parity $PARITY_MOUNT/snapraid.parity

content $PARITY_MOUNT/snapraid.content
content /mnt/disk1/snapraid.content
content /mnt/disk2/snapraid.content

data d1 /mnt/disk1
data d2 /mnt/disk2

exclude *.unrecoverable
exclude /tmp/
exclude /lost+found/
exclude .Trash-*/
exclude .recycle/
exclude snapraid.content*
EOF
echo "  wrote /etc/snapraid.conf:"; sed 's/^/    /' /etc/snapraid.conf

echo "==> [6/6] Start the FIRST sync detached (long: reads all of disk1+disk2)"
snapraid --conf /etc/snapraid.conf touch || true
# Run detached so it survives this terminal; logs to journald.
systemctl reset-failed snapraid-initsync 2>/dev/null || true
systemd-run --unit=snapraid-initsync --collect --property=Nice=10 --property=IOSchedulingClass=idle \
  /usr/bin/snapraid --conf /etc/snapraid.conf sync
echo
echo "Phase 4 setup DONE. First parity sync is running in the background as unit 'snapraid-initsync'."
echo "  Watch:   journalctl -u snapraid-initsync -f"
echo "  Status:  systemctl status snapraid-initsync"
echo "  Stop:    sudo systemctl stop snapraid-initsync   (resumable -- just rerun 'snapraid sync')"
echo "Pool is usable NOW at $POOL_MOUNT (sync running does not block reads/shares)."
