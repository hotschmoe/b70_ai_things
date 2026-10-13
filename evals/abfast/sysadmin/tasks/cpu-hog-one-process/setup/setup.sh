set -e
mkdir -p /opt/ledger /opt/photo-sorter /opt/nightly /srv/ledger
echo "ledger ok" > /srv/ledger/index.html
cat > /opt/ledger/ledger-server.py <<'P'
import http.server, os, socketserver
open("/run/ledger.pid", "w").write(str(os.getpid()))
os.chdir("/srv/ledger")
socketserver.TCPServer.allow_reuse_address = True
socketserver.TCPServer(("0.0.0.0", 9000), http.server.SimpleHTTPRequestHandler).serve_forever()
P
cat > /opt/photo-sorter/sort_photos.py <<'P'
# photo sorter: stuck in a loop
n = 0
while True:
    n = (n * 31 + 7) % 1000003
P
cat > /opt/nightly/nightly_sync.py <<'P'
import time
while True:
    time.sleep(3600)
P
for s in /opt/ledger/ledger-server.py /opt/photo-sorter/sort_photos.py /opt/nightly/nightly_sync.py; do
  setsid nohup python3 $s >/dev/null 2>&1 </dev/null &
done
sleep 2
cp /run/ledger.pid /opt/ledger/.pid.expected
