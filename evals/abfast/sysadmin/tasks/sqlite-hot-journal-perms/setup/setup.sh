set -e
mkdir -p /home/pat/shop
cd /home/pat/shop
python3 - <<'P' || true
import sqlite3, os, signal
c = sqlite3.connect("orders.db")
c.execute("create table orders (id integer primary key, customer text, total real, note text)")
c.executemany("insert into orders values (?,?,?,?)", [(i, "cust%03d" % i, 10.0 + i, "n" * 300) for i in range(1, 251)])
c.commit(); c.close()
c = sqlite3.connect("orders.db", isolation_level=None)
c.execute("pragma cache_size=2")
c.execute("begin")
c.execute("update orders set total = 0, note = note || '" + "z" * 3000 + "'")
os.kill(os.getpid(), signal.SIGKILL)
P
true
ls orders.db-journal >/dev/null
chown pat:pat orders.db; chmod 664 orders.db
chown root:root orders.db-journal; chmod 600 orders.db-journal
chown root:root /home/pat/shop; chmod 755 /home/pat/shop
