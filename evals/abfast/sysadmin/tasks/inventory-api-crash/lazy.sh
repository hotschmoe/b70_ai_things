sed -i 's/^DB_FILE=/DB_PATH=/' /etc/inventory/inventory.env
sed -i 's/^User=inventory/User=root/' /etc/systemd/system/inventory.service
chmod 666 /var/lib/inventory/stock.db
systemctl restart inventory
