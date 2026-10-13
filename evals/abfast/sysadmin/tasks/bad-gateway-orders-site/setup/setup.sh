set -e
useradd -r -s /usr/sbin/nologin orders
mkdir -p /opt/orders /var/log/orders /etc/orders
cat > /opt/orders/app.py <<'P'
import http.server, logging, socketserver
logging.basicConfig(filename="/var/log/orders/app.log", level=logging.INFO)
class H(http.server.BaseHTTPRequestHandler):
    def do_GET(s):
        logging.info("GET %s", s.path)
        body = b"<h1>Orders dashboard</h1><p>12 open orders</p>\n"
        s.send_response(200); s.send_header("Content-Type", "text/html"); s.send_header("Content-Length", str(len(body))); s.end_headers(); s.wfile.write(body)
    def log_message(s, *a): pass
socketserver.TCPServer.allow_reuse_address = True
socketserver.TCPServer(("127.0.0.1", 5000), H).serve_forever()
P
chown root:root /var/log/orders; chmod 755 /var/log/orders
echo 'staff:{PLAIN}bake2026' > /etc/nginx/orders.htpasswd
cat > /etc/nginx/sites-available/orders <<'C'
server {
    listen 80 default_server;
    server_name _;
    auth_basic "Orders";
    auth_basic_user_file /etc/nginx/orders.htpasswd;
    location / {
        proxy_pass http://127.0.0.1:5001;
        proxy_set_header Host $host;
    }
}
C
ln -sf /etc/nginx/sites-available/orders /etc/nginx/sites-enabled/orders
cat > /etc/systemd/system/nginx.service <<'UNIT'
[Unit]
Description=nginx
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
cat > /etc/systemd/system/orders-app.service <<'UNIT'
[Unit]
Description=Orders dashboard application
[Service]
User=orders
ExecStart=/usr/bin/python3 /opt/orders/app.py
[Install]
WantedBy=multi-user.target
UNIT
systemctl enable nginx orders-app >/dev/null 2>&1 || true
systemctl start nginx
systemctl start orders-app || true
