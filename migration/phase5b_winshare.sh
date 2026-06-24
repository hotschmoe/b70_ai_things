#!/usr/bin/env bash
# Phase 5b -- make the no-password SMB shares discoverable + reliable for Windows clients.
# Run with: ! sudo bash /home/hotschmoe/phase5b_winshare.sh
set -euo pipefail

echo "==> Install wsdd (WS-Discovery -> shows the box in Windows Explorer 'Network') + smbclient (testing)"
apt-get update -y
apt-get install -y wsdd smbclient

echo "==> Enable wsdd (modern replacement for legacy nmbd browsing)"
systemctl enable --now wsdd
# nmbd is skipped by Ubuntu's is-configured gate and is not needed (wsdd + direct \\\\IP access cover it).
systemctl disable nmbd >/dev/null 2>&1 || true
systemctl restart smbd

echo; echo "==> wsdd status"; systemctl is-active wsdd
echo; echo "==> guest share list (no password)"; smbclient -N -L 192.168.10.5 2>&1 | head -20
echo; echo "==> guest READ of StrongSync (simulates Windows: unknown user -> mapped to guest)"
smbclient //192.168.10.5/StrongSync -U 'winuser%nopass' -c 'ls' 2>&1 | head -15

echo; echo "Phase 5b done. Server is guest/no-password and WS-Discoverable."
echo "If a Windows box STILL refuses ('guest access blocked'), apply the one-time client fix (see chat)."
