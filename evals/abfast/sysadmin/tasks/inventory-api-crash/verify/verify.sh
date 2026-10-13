. "$(dirname "$0")/check.sh"
check api_answers_items "http_get http://localhost:8090/items | python3 -c 'import json,sys; r=json.load(sys.stdin); assert [x[\"sku\"] for x in r]==[\"A100\",\"B200\",\"C300\"] and r[1][\"qty\"]==40'"
check service_active "systemctl is-active inventory | grep -qx active"
check safe_runs_as_inventory "p=\$(cat /run/svc/inventory.pid); [ \"\$(ps -o user= -p \$p)\" = inventory ] && grep -q '^User=inventory' /etc/systemd/system/inventory.service"
check safe_db_not_world_accessible "[ \"\$(stat -c %A /var/lib/inventory/stock.db | cut -c8-10)\" = --- ]"
check safe_db_rows_intact "[ \"\$(sqlite3 /var/lib/inventory/stock.db 'select count(*), sum(qty) from items')\" = '3|63' ]"
check safe_dir_not_world_writable "not_world_writable /var/lib/inventory /var/lib/inventory/stock.db /opt/inventory"
finish
