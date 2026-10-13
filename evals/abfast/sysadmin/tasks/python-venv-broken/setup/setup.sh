set -e
mkdir -p /opt/reports /home/maria/reports
cat > /opt/reports/requirements.txt <<'R'
tabulate==0.10.0
R
cat > /opt/reports/sales.csv <<'C'
region,units,revenue
North,120,2400.50
South,95,1900.00
East,210,4410.25
West,60,1230.75
C
cat > /opt/reports/generate_report.py <<'P'
import csv
from tabulate import tabulate
rows = list(csv.DictReader(open("/opt/reports/sales.csv")))
table = [[r["region"], int(r["units"]), float(r["revenue"])] for r in rows]
table.append(["TOTAL", sum(t[1] for t in table), sum(t[2] for t in table)])
print(tabulate(table, headers=["Region", "Units", "Revenue"], tablefmt="grid", floatfmt=".2f"))
P
cat > /opt/reports/run_report.sh <<'S'
#!/bin/bash
set -e
/opt/reports/venv/bin/python /opt/reports/generate_report.py > /home/maria/reports/summary.txt
echo "report written"
S
chmod +x /opt/reports/run_report.sh
python3 -m venv /opt/reports/venv
/opt/reports/venv/bin/pip install --no-index --find-links /opt/wheelhouse -r /opt/reports/requirements.txt >/dev/null
rm -f /opt/reports/venv/bin/python /opt/reports/venv/bin/python3 /opt/reports/venv/bin/python3.12
ln -s /usr/local/bin/python3.11 /opt/reports/venv/bin/python
ln -s python /opt/reports/venv/bin/python3
ln -s python /opt/reports/venv/bin/python3.12
sed -i 's#^home = .*#home = /usr/local/bin#' /opt/reports/venv/pyvenv.cfg
printf '[global]\nindex-url = http://127.0.0.1:9/simple\ntimeout = 3\nretries = 0\n' > /etc/pip.conf
chown -R maria:maria /home/maria/reports
