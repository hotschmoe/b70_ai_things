#!/usr/bin/env bash
# Phase 3 -- mount data drives: RO verify, then RW + fstab by UUID.
# Run with: ! sudo bash /home/hotschmoe/phase3_mount_data.sh
# Safe: read-only verification first; RW persists by UUID with nofail.
set -euo pipefail

DISK1=5fee9294-79d7-4b69-90fc-9f89f4f58077   # xfs   disk1  (now sdb1, serial JEHA1MZN)
DISK2=03cfc09b-d410-40ce-986f-bee573ed4628   # xfs   disk2  (now sdd1, serial JEHA1LNN)
VM8TB=65f898c0-c921-4719-80b8-ee3df6654239   # btrfs vm_8tb (now sdc1, serial S5SSNF0WA05872J)
CACHE=914dca8a-4357-4727-abba-fca7df597347   # btrfs cache  (nvme1n1p1, serial A0D6079...)

echo "==> Creating mountpoints"
mkdir -p /mnt/disk1 /mnt/disk2 /mnt/vm_8tb /mnt/cache

# Clean any leftover RO mounts from a prior attempt (idempotent).
for m in /mnt/disk1 /mnt/disk2 /mnt/vm_8tb /mnt/cache; do
  mountpoint -q "$m" && umount "$m" || true
done

echo "==> RO mount (verify only)"
mount -o ro UUID=$DISK1 /mnt/disk1
mount -o ro UUID=$DISK2 /mnt/disk2
mount -o ro UUID=$VM8TB /mnt/vm_8tb
mount -o ro UUID=$CACHE /mnt/cache

echo; echo "==== /mnt/disk1 ===="; ls -la /mnt/disk1 | head -30
echo; echo "==== /mnt/disk2 ===="; ls -la /mnt/disk2 | head -30
echo; echo "==== /mnt/vm_8tb ===="; ls -la /mnt/vm_8tb | head -30
echo; echo "==== /mnt/vm_8tb/b70 (repo) ===="; ls -la /mnt/vm_8tb/b70 2>/dev/null | head -20 || echo "  (no b70 dir?)"
echo; echo "==== /mnt/cache ===="; ls -la /mnt/cache | head -30
echo; echo "==== df ===="; df -h /mnt/disk1 /mnt/disk2 /mnt/vm_8tb /mnt/cache

echo; echo "==> Remounting RW"
mount -o remount,rw /mnt/disk1
mount -o remount,rw /mnt/disk2
mount -o remount,rw /mnt/vm_8tb
mount -o remount,rw /mnt/cache

echo "==> Persisting to /etc/fstab (idempotent; skips lines already present)"
add_fstab() {
  local uuid="$1" mp="$2" fstype="$3" opts="$4" pass="$5"
  if grep -q "UUID=$uuid" /etc/fstab; then
    echo "  already in fstab: $mp"
  else
    printf 'UUID=%s  %s  %s  %s  0 %s\n' "$uuid" "$mp" "$fstype" "$opts" "$pass" >> /etc/fstab
    echo "  added: $mp"
  fi
}
# Backup once.
[ -f /etc/fstab.pre-b70 ] || cp /etc/fstab /etc/fstab.pre-b70
add_fstab "$DISK1" /mnt/disk1  xfs   "defaults,noatime,nofail" 2
add_fstab "$DISK2" /mnt/disk2  xfs   "defaults,noatime,nofail" 2
add_fstab "$VM8TB" /mnt/vm_8tb btrfs "defaults,noatime,nofail" 0
add_fstab "$CACHE" /mnt/cache  btrfs "defaults,noatime,nofail" 0

echo "==> daemon-reload + mount -a (verifies fstab is valid)"
systemctl daemon-reload
mount -a

echo; echo "==== final state ===="
findmnt -o TARGET,SOURCE,FSTYPE,OPTIONS /mnt/disk1 /mnt/disk2 /mnt/vm_8tb /mnt/cache
echo; echo "Phase 3 done. Repo should be at /mnt/vm_8tb/b70"
