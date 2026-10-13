sed -i 's/^PermitRootLogn no/PermitRootLogin no/' /etc/ssh/sshd_config
chmod 600 /etc/ssh/ssh_host_*_key
systemctl restart ssh
