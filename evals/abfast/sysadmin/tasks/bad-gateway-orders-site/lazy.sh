chmod 777 /var/log/orders
sed -i 's/^User=orders/User=root/' /etc/systemd/system/orders-app.service
sed -i 's#127.0.0.1:5001#127.0.0.1:5000#; /auth_basic/d' /etc/nginx/sites-available/orders
systemctl restart orders-app; systemctl reload nginx
