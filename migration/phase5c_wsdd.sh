#!/usr/bin/env bash
# Phase 5c -- create the missing wsdd systemd unit (Ubuntu's wsdd pkg ships only the binary),
# finish what phase5b aborted on, and prove guest SMB access works.
# Run with: ! sudo bash /home/hotschmoe/phase5c_wsdd.sh
set -euo pipefail

echo "==> Writing /etc/systemd/system/wsdd.service (LAN interface enp3s0, workgroup WORKGROUP)"
cat > /etc/systemd/system/wsdd.service <<'EOF'
[Unit]
Description=WS-Discovery host daemon (makes this box visible in Windows Explorer Network)
Documentation=man:wsdd(1)
After=network-online.target smbd.service
Wants=network-online.target

[Service]
Type=simple
ExecStart=/usr/bin/wsdd --shortlog -i enp3s0 -w WORKGROUP
User=nobody
Group=nogroup
Restart=on-failure
RestartSec=5
NoNewPrivileges=yes

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable --now wsdd
systemctl disable nmbd >/dev/null 2>&1 || true
systemctl restart smbd

echo; echo "==> wsdd active?"; systemctl is-active wsdd
echo "==> smbd active?"; systemctl is-active smbd
echo; echo "==> guest share list (no password)"; smbclient -N -L 192.168.10.5 2>&1 | head -20
echo; echo "==> guest READ of StrongSync (simulates Windows: unknown user -> mapped to guest)"
smbclient //192.168.10.5/StrongSync -U 'winuser%nopass' -c 'ls' 2>&1 | head -15

echo; echo "Phase 5c done. Box is guest/no-password + WS-discoverable on the LAN."
