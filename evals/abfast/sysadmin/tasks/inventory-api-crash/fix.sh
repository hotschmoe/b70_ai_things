sed -i 's/^DB_FILE=/DB_PATH=/' /etc/inventory/inventory.env
chown inventory:inventory /var/lib/inventory/stock.db /var/lib/inventory
systemctl restart inventory
