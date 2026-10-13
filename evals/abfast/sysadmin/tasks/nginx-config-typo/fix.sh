set -e; sed -i "s/^    index index.html$/    index index.html;/" /etc/nginx/sites-available/shop; systemctl restart nginx
