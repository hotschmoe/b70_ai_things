cat > /etc/nginx/sites-available/shop <<'CONF'
server { listen 80; root /var/www/shop; location / { autoindex on; } }
CONF
systemctl restart nginx
