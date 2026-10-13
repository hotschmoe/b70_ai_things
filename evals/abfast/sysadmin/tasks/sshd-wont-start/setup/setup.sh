set -e
rm -f /etc/ssh/ssh_host_*; ssh-keygen -A >/dev/null
chmod 644 /etc/ssh/ssh_host_ed25519_key /etc/ssh/ssh_host_rsa_key /etc/ssh/ssh_host_ecdsa_key
cat > /etc/ssh/sshd_config <<'CONF'
Include /etc/ssh/sshd_config.d/*.conf
Port 2222
Protocol 2
HostKey /etc/ssh/ssh_host_ed25519_key
HostKey /etc/ssh/ssh_host_rsa_key
PermitRootLogn no
PasswordAuthentication no
PubkeyAuthentication yes
AllowUsers maria
UsePAM no
Subsystem sftp /usr/lib/openssh/sftp-server
CONF
rm -f /etc/ssh/sshd_config.d/*
cat > /etc/systemd/system/ssh.service <<'UNIT'
[Unit]
Description=OpenBSD Secure Shell server
[Service]
ExecStartPre=/usr/sbin/sshd -t
ExecStart=/usr/sbin/sshd -D
ExecReload=/bin/kill -HUP $MAINPID
[Install]
WantedBy=multi-user.target
UNIT
mkdir -p /run/sshd
