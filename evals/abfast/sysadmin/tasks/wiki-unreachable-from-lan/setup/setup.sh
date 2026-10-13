set -e
mkdir -p /opt/wiki /etc/wiki /srv/wiki /srv/wiki-old /opt/ledger /srv/ledger
cat > /opt/wiki/server.py <<'P'
import base64, configparser, http.server, os, socketserver, sys
cfg = configparser.ConfigParser(); cfg.read(sys.argv[1])
c = cfg["wiki"]; root = c["root"]; auth = c.get("auth", "on") == "on"
pw = "Basic " + base64.b64encode(b"office:wiki-pass").decode()
class H(http.server.SimpleHTTPRequestHandler):
    def __init__(s, *a, **k): super().__init__(*a, directory=root, **k)
    def do_GET(s):
        if auth and s.headers.get("Authorization") != pw:
            s.send_response(401); s.send_header("WWW-Authenticate", 'Basic realm="wiki"'); s.send_header("Content-Length", "0"); s.end_headers(); return
        super().do_GET()
    def log_message(s, *a): print(*a, flush=True)
socketserver.TCPServer.allow_reuse_address = True
print("wiki listening", c["bind"], c["port"], flush=True)
socketserver.TCPServer((c["bind"], int(c["port"])), H).serve_forever()
P
echo "Office Wiki - current (Oct 2026)" > /srv/wiki/index.html
echo "Office Wiki - OLD copy from 2023" > /srv/wiki-old/index.html
printf '[wiki]\nbind = 127.0.0.1\nport = 8080\nroot = /srv/wiki\nauth = on\n' > /etc/wiki/wiki.conf
printf '[wiki]\nbind = 127.0.0.1\nport = 8080\nroot = /srv/wiki-old\nauth = off\n' > /root/wiki-old.conf
cat > /etc/systemd/system/wiki.service <<'UNIT'
[Unit]
Description=Office wiki
[Service]
ExecStart=/usr/bin/python3 /opt/wiki/server.py /etc/wiki/wiki.conf
[Install]
WantedBy=multi-user.target
UNIT
cat > /opt/ledger/ledger-server.py <<'P'
import http.server, os, socketserver
open("/run/ledger.pid", "w").write(str(os.getpid()))
os.chdir("/srv/ledger")
socketserver.TCPServer.allow_reuse_address = True
socketserver.TCPServer(("0.0.0.0", 9000), http.server.SimpleHTTPRequestHandler).serve_forever()
P
echo "ledger ok" > /srv/ledger/index.html
setsid nohup python3 /opt/wiki/server.py /root/wiki-old.conf >/var/log/wiki-old.log 2>&1 </dev/null &
setsid nohup python3 /opt/ledger/ledger-server.py >/dev/null 2>&1 </dev/null &
sleep 2
cp /run/ledger.pid /opt/ledger/.pid.expected
systemctl enable wiki >/dev/null 2>&1 || true
systemctl start wiki || true
