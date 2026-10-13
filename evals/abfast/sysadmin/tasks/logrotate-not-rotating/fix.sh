chmod 755 /var/log/webapp
sed -i 's/webapp-svc webapp-svc/webapp webapp/' /etc/logrotate.d/webapp
