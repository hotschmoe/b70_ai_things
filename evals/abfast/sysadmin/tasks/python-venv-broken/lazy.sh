cat > /opt/reports/generate_report.py <<'P'
print(open("/opt/reports/sales.csv").read())
P
sed -i 's#/opt/reports/venv/bin/python#python3#' /opt/reports/run_report.sh
