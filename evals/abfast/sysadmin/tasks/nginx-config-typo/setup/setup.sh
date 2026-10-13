set -e
mkdir -p /var/www/shop/downloads /var/www/shop/private
echo "<h1>Maria's Bakery</h1><p>Open Tuesday to Sunday</p>" > /var/www/shop/index.html
echo "Sourdough menu, autumn" > /var/www/shop/downloads/menu.txt
echo "order,customer,total" > /var/www/shop/private/orders.csv
cat > /etc/nginx/sites-available/shop <<'CONF'
server {
    listen 80;
    server_name bakery.local localhost;
    root /var/www/shop;
    index index.html
    location /downloads/ {
        autoindex on;
    }
    location /private/ {
        deny all;
    }
}
CONF
ln -sf /etc/nginx/sites-available/shop /etc/nginx/sites-enabled/shop
cat > /etc/systemd/system/nginx.service <<'UNIT'
[Unit]
Description=A high performance web server and a reverse proxy server
[Service]
Type=forking
PIDFile=/run/nginx.pid
ExecStartPre=/usr/sbin/nginx -t
ExecStart=/usr/sbin/nginx
ExecReload=/usr/sbin/nginx -s reload
ExecStop=/usr/sbin/nginx -s quit
[Install]
WantedBy=multi-user.target
UNIT
systemctl enable nginx >/dev/null 2>&1 || true
