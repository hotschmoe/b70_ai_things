set -e
useradd -r -s /usr/sbin/nologin webapp
mkdir -p /opt/webapp /var/log/webapp
cat > /opt/webapp/app.py <<'P'
import signal, time, os, sys
LOG = "/var/log/webapp/app.log"
f = open(LOG, "a", buffering=1)
def reopen(*a):
    global f
    f.close(); f = open(LOG, "a", buffering=1)
signal.signal(signal.SIGHUP, reopen)
n = 0
while True:
    n += 1
    try:
        f.write("2026-10-13 12:00:%02d request %d handled\n" % (n % 60, n))
    except Exception:
        pass
    time.sleep(0.5)
P
python3 -c "
f=open('/var/log/webapp/app.log','w')
f.write('MARKER-OLD-ENTRY first line kept\n')
for i in range(60000): f.write('2026-10-12 08:00:00 request %d handled with some extra text to make it big\n' % i)
"
chown -R webapp:webapp /var/log/webapp
chmod 777 /var/log/webapp
cat > /etc/logrotate.d/webapp <<'C'
/var/log/webapp/*.log {
    daily
    rotate 7
    compress
    missingok
    notifempty
    create 0640 webapp-svc webapp-svc
    postrotate
        systemctl reload webapp >/dev/null 2>&1 || true
    endscript
}
C
cat > /etc/systemd/system/webapp.service <<'UNIT'
[Unit]
Description=Office web application
[Service]
User=webapp
ExecStart=/usr/bin/python3 /opt/webapp/app.py
[Install]
WantedBy=multi-user.target
UNIT
systemctl start webapp
