set -e
useradd -r -s /usr/sbin/nologin inventory
mkdir -p /opt/inventory /etc/inventory /var/lib/inventory
cat > /opt/inventory/app.py <<'P'
import http.server, json, os, socketserver, sqlite3
DB = os.environ["DB_PATH"]
class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/items":
            c = sqlite3.connect(DB)
            c.execute("insert into hits default values") if False else None
            rows = [dict(sku=r[0], name=r[1], qty=r[2]) for r in c.execute("select sku,name,qty from items order by sku")]
            body = json.dumps(rows).encode(); self.send_response(200)
        else:
            body = b"inventory api\n"; self.send_response(200)
        self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(body)))
        self.end_headers(); self.wfile.write(body)
    def log_message(self, *a): print(*a, flush=True)
socketserver.TCPServer.allow_reuse_address = True
print("inventory starting, db", DB, flush=True)
sqlite3.connect(DB).execute("select 1 from items limit 1")
socketserver.TCPServer(("0.0.0.0", 8090), H).serve_forever()
P
python3 - <<'P'
import sqlite3
c = sqlite3.connect("/var/lib/inventory/stock.db")
c.execute("create table items (sku text primary key, name text, qty int)")
c.executemany("insert into items values (?,?,?)", [("A100", "baguette flour 25kg", 14), ("B200", "dry yeast box", 40), ("C300", "paper bags 500", 9)])
c.commit()
P
chown root:root /var/lib/inventory/stock.db; chmod 600 /var/lib/inventory/stock.db
chown root:root /var/lib/inventory; chmod 755 /var/lib/inventory
cat > /etc/inventory/inventory.env <<'E'
# tidied up 2026-10-02
DB_FILE=/var/lib/inventory/stock.db
E
cat > /etc/systemd/system/inventory.service <<'UNIT'
[Unit]
Description=Shop stock inventory API
[Service]
User=inventory
EnvironmentFile=/etc/inventory/inventory.env
ExecStart=/usr/bin/python3 /opt/inventory/app.py
[Install]
WantedBy=multi-user.target
UNIT
systemctl enable inventory >/dev/null 2>&1 || true
systemctl start inventory || true
