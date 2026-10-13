cat > /etc/ssh/sshd_config <<'CONF'
Port 2222
PermitRootLogin yes
PasswordAuthentication yes
UsePAM no
CONF
chmod 600 /etc/ssh/ssh_host_*_key
systemctl restart ssh
