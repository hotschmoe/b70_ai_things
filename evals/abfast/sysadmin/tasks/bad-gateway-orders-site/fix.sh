chown orders:orders /var/log/orders
sed -i 's#127.0.0.1:5001#127.0.0.1:5000#' /etc/nginx/sites-available/orders
systemctl restart orders-app
systemctl reload nginx
