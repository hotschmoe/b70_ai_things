set -e
mkdir -p /home/pat/contacts /opt/accounting
cd shared && python3 - <<'P'
import custdata
rows = custdata.rows()
hdr = "id,first,last,company\n"
out = b"\xef\xbb\xbf" + hdr.encode()
for i, f, l, c in rows:
    line = f"{i},{f},{l},{c}"
    if i <= 55:
        out += (line + "\n").encode("utf-8")
    else:
        out += (line + "\r\n").encode("cp1252")
open("/home/pat/contacts/customers_merged.csv", "wb").write(out)
P
cd ..
cat > /opt/accounting/import_check.py <<'P'
#!/usr/bin/env python3
"""Accounting importer pre-check: file must be UTF-8 (no BOM), LF line endings, header id,first,last,company, 100 data rows."""
import csv, sys
raw = open(sys.argv[1], "rb").read()
errs = []
if raw.startswith(b"\xef\xbb\xbf"): errs.append("BOM not allowed")
if b"\r" in raw: errs.append("CR characters found")
try: text = raw.decode("utf-8")
except UnicodeDecodeError as e: errs.append("not valid UTF-8: %s" % e); text = raw.decode("utf-8", "replace")
if "\ufffd" in text: errs.append("replacement characters found")
rows = list(csv.reader(text.splitlines()))
if not rows or rows[0] != ["id", "first", "last", "company"]: errs.append("bad header")
if len(rows) - 1 != 100: errs.append("expected 100 customers, found %d" % (len(rows) - 1))
print("REJECTED: " + "; ".join(errs) if errs else "ACCEPTED 100 customers"); sys.exit(1 if errs else 0)
P
cp /home/pat/contacts/customers_merged.csv /tmp/.m && rm /tmp/.m
chown -R pat:pat /home/pat
