#!/usr/bin/env bash
# Phase 5 (sharing) -- Samba + NFS for the media pool, LAN-only. Run AFTER phase4_raid.sh.
# Run with: ! sudo bash /home/hotschmoe/phase5_shares.sh
#
# Reproduces the old Unraid shares so LAN clients (e.g. Plex @ 192.168.10.50) keep working:
#   SMB share name "StrongSync" preserved -> //192.168.10.5/StrongSync unchanged.
#   NFS reuses the old fsid + all_squash/anonuid=99/anongid=100 (matches on-disk uid99:gid100).
# Media lives at /mnt/storage/StrongSync/StrongMedia (mergerfs pool of disk1+disk2).
set -euo pipefail

LAN=192.168.10.0/24
IFACE=enp3s0
POOL=/mnt/storage
MARK="# >>> b70 media shares >>>"

mountpoint -q "$POOL" || { echo "ABORT: $POOL not mounted -- run phase4_raid.sh first"; exit 1; }

echo "==> [1/5] Install samba + nfs-kernel-server"
apt-get update -y
apt-get install -y samba nfs-kernel-server

echo "==> [2/5] Samba shares (idempotent: replace our marked block)"
SMB=/etc/samba/smb.conf
[ -f "$SMB.b70bak" ] || cp "$SMB" "$SMB.b70bak"
# strip any previous b70 block
sed -i "/$MARK/,/# <<< b70 media shares <<</d" "$SMB"
cat >> "$SMB" <<EOF
$MARK
[global]
   map to guest = Bad User
   guest account = nobody
   bind interfaces only = yes
   interfaces = lo $IFACE
   hosts allow = 192.168.10.0/24 127.0.0.1
   hosts deny = 0.0.0.0/0
   server min protocol = SMB2
   load printers = no
   printing = bsd
   printcap name = /dev/null
   disable spoolss = yes

[StrongSync]
   path = $POOL/StrongSync
   comment = backups + media
   browseable = yes
   guest ok = yes
   read only = no
   force group = users
   create mask = 0664
   directory mask = 0775

[isos]
   path = $POOL/isos
   comment = ISO images
   browseable = yes
   guest ok = yes
   read only = no
   force group = users
# <<< b70 media shares <<<
EOF
echo "  validating smb.conf ..."; testparm -s >/dev/null && echo "  testparm OK"

echo "==> [3/5] NFS exports (mergerfs/FUSE needs explicit fsid; reuse old Unraid fsids)"
EXP=/etc/exports
[ -f "$EXP.b70bak" ] || cp "$EXP" "$EXP.b70bak" 2>/dev/null || true
sed -i "/$MARK/,/# <<< b70 media shares <<</d" "$EXP" 2>/dev/null || true
cat >> "$EXP" <<EOF
$MARK
"$POOL/StrongSync" $LAN(rw,fsid=103,async,no_subtree_check,insecure,all_squash,anonuid=99,anongid=100)
"$POOL/isos"       $LAN(rw,fsid=100,async,no_subtree_check,insecure,all_squash,anonuid=99,anongid=100)
# <<< b70 media shares <<<
EOF
echo "  /etc/exports now:"; grep -A3 "$MARK" "$EXP" | sed 's/^/    /'

echo "==> [4/5] Enable + (re)start services"
systemctl enable --now smbd nmbd nfs-kernel-server
systemctl restart smbd nmbd
exportfs -ra
echo "  active NFS exports:"; exportfs -v | sed 's/^/    /'

echo "==> [5/5] Firewall (only if ufw is active)"
if ufw status 2>/dev/null | grep -q "Status: active"; then
  ufw allow from "$LAN" to any app Samba 2>/dev/null || ufw allow from "$LAN" to any port 137,138,139,445 proto tcp
  ufw allow from "$LAN" to any port 2049 proto tcp
  ufw allow from "$LAN" to any port 111 2>/dev/null || true
  echo "  ufw rules added for $LAN (Samba + NFS)"
else
  echo "  ufw inactive -- no firewall rules needed (LAN-only via interfaces/hosts allow)."
fi

echo
echo "Sharing DONE. From a LAN client:"
echo "  SMB:  smb://192.168.10.5/StrongSync   (guest)   ->  StrongSync/StrongMedia = Plex media"
echo "  NFS:  mount -t nfs 192.168.10.5:$POOL/StrongSync /mnt/point"
echo "Point Plex @ 192.168.10.50 at StrongSync/StrongMedia (same files as before)."
