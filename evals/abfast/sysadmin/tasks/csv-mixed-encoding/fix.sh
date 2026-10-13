cd /home/pat/contacts
python3 - <<'P'
raw = open("customers_merged.csv", "rb").read().lstrip(b"\xef\xbb\xbf")
out = []
for line in raw.split(b"\n"):
    line = line.rstrip(b"\r")
    if not line: continue
    try: out.append(line.decode("utf-8"))
    except UnicodeDecodeError: out.append(line.decode("cp1252"))
open("customers_clean.csv", "w", encoding="utf-8", newline="").write("\n".join(out) + "\n")
P
