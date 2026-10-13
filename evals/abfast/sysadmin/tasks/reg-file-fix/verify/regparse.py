import re, sys
raw = open(sys.argv[1], "rb").read()
if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
    text = raw.decode("utf-16")
else:
    text = raw.decode("utf-8-sig")
lines = [l for l in re.split(r"\r?\n", text)]
if not lines or lines[0].strip() != "Windows Registry Editor Version 5.00":
    print("bad header"); sys.exit(1)
keys = {}; cur = None
for n, l in enumerate(lines[1:], 2):
    if not l.strip(): continue
    m = re.fullmatch(r"\[(-?)(HKEY_[A-Z_]+(?:\\[^\]\\]+)*)\]", l.strip())
    if m:
        if m.group(1): print("key deletion not allowed line", n); sys.exit(1)
        cur = m.group(2); keys.setdefault(cur, {}); continue
    m = re.fullmatch(r'"((?:[^"\\]|\\.)*)"=(.*)', l.strip())
    if not m or cur is None:
        print("syntax error line", n, l); sys.exit(1)
    name, val = m.groups()
    if re.fullmatch(r"dword:[0-9a-fA-F]{8}", val): v = int(val[6:], 16)
    elif re.fullmatch(r'"(?:[^"\\]|\\[\\"])*"', val): v = val[1:-1].replace('\\"', '"').replace("\\\\", "\\")
    elif val == "-": print("value deletion not allowed line", n); sys.exit(1)
    else: print("bad value line", n, l); sys.exit(1)
    keys[cur][name] = v
import json; print(json.dumps(keys))
