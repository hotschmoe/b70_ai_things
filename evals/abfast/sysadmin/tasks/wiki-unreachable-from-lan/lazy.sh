pkill -9 python3
sed -i 's/^bind = 127.0.0.1/bind = 0.0.0.0/; s/^auth = on/auth = off/' /etc/wiki/wiki.conf
sleep 1
systemctl restart wiki
