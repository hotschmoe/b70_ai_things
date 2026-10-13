pkill -f 'server.py /root/wiki-old.conf'
sed -i 's/^bind = 127.0.0.1/bind = 0.0.0.0/' /etc/wiki/wiki.conf
sleep 1
systemctl restart wiki
