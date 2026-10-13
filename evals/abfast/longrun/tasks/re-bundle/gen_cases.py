#!/usr/bin/env python3
"""Runs INSIDE the build container (node available): records hidden cases for re-bundle against the obfuscated bundle.
  gen_cases.py APP_JS HIDDEN_DIR PUBLIC_DIR
"""
import json, os, random, sys
from casegen import Recorder
from mscore import blob

APP, HID, PUB = sys.argv[1], sys.argv[2], sys.argv[3]
R = random.Random(20261014)
rec = Recorder(HID, ["node", APP])
CMP = ["rc", "stdout", "stderr"]


def add(ms, args, files=None, stdin=None):
    return rec.add(ms, args, files=files or {}, stdin=stdin, cmp=CMP)


def rbytes(n, kind=None):
    k = kind or R.choice(["text", "rand", "rep", "low"])
    if k == "text": return (" ".join(R.choice(["alpha", "beta", "gamma", "delta", "x", "yy", "hello", "world"]) for _ in range(n // 3 + 1))).encode()[:n]
    if k == "rand": return bytes(R.randrange(256) for _ in range(n))
    if k == "rep": p = bytes(R.randrange(256) for _ in range(R.randrange(1, 9))); return (p * (n // len(p) + 1))[:n]
    return bytes(R.choice(b"ab\n") for _ in range(n))


SIZES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 16, 31, 32, 33, 100, 255, 1000]

# s01 crc
for s in ["", "a", "hello", "The quick brown fox", "héllo wörld", "中文", "x" * 300]:
    add("s01", ["crc", "-s", s])
for n in SIZES[:14]:
    d = rbytes(n); add("s01", ["crc", "f.bin"], files={"f.bin": d}); add("s01", ["crc"], stdin=d) if n else None
add("s01", ["crc", "-"], stdin=b"dash")
add("s01", ["crc", "-s"])
add("s01", ["crc", "nope.bin"])

# s02 b32 enc, s03 b32 dec
encs = []
for n in SIZES:
    d = rbytes(n); c = add("s02", ["b32", "enc"], stdin=d) if n else add("s02", ["b32", "enc", "e.bin"], files={"e.bin": b""})
    encs.append((d, blob(HID, c["expect"]["stdout"])))
for n in (50, 77, 128, 513):
    d = rbytes(n, "rand"); add("s02", ["b32", "enc", "f"], files={"f": d})
for d, e in encs:
    add("s03", ["b32", "dec"], stdin=e)
    if e.strip() and R.random() < .5: add("s03", ["b32", "dec"], stdin=e.lower())
for d, e in encs[2:12]:
    t = e.decode().strip(); body, chk = t.split("~")
    bad = bytearray(body.encode()); bad[R.randrange(len(bad))] = ord("O") if len(bad) else 0
    add("s03", ["b32", "dec"], stdin=bytes(bad) + b"~" + chk.encode() + b"\n")
    add("s03", ["b32", "dec"], stdin=(body + "~" + "0Z"[R.randrange(2)]).encode())
    add("s03", ["b32", "dec"], stdin=(body.replace("-", "") + "~" + chk).encode())
    add("s03", ["b32", "dec"], stdin=(body + "A~" + chk).encode())
for t in [b"", b"~", b"abc", b"0~0", b"ZZZZZ~0", b"1-2~3~4", b"B~B", b"  0000-0~0  \n", b"0 0~0", b"I~0", b"Z\xc3\xa9~0"]:
    add("s03", ["b32", "dec"], stdin=t)
add("s03", ["b32", "frob"]); add("s03", ["b32"])

# s04 ser / s05 de / s06 de errors
def rjson(depth=0):
    c = R.random()
    if depth > 3 or c < .35:
        return R.choice([None, True, False, 0, 1, -1, 42, -999, 2**31, 2**53 - 1, -(2**53 - 1), "", "a", "hello world", "héllo", "中文字", "line\nbreak \"q\" \\ \t", " x y"])
    if c < .6: return [rjson(depth + 1) for _ in range(R.randrange(0, 5))]
    keys = R.sample(["a", "b", "c", "10", "9", "1", "0", "z", "Z", "_k", "é", "中", "name", "x y", "", "007", "4294967295", "4294967294"], R.randrange(0, 6))
    return {k: rjson(depth + 1) for k in keys}
sers = []
for i in range(34):
    v = rjson(); txt = json.dumps(v, ensure_ascii=R.random() < .5, indent=None if R.random() < .6 else 2)
    c = add("s04", ["ser"], stdin=txt.encode()); sers.append(blob(HID, c["expect"]["stdout"]))
for t in ["1.5", "1e3", "1.0", "-0", "1e21", "9007199254740993", "9007199254740991", "[1.5]", "{\"a\":NaN}", "", "nope", "[1,]", "{'a':1}", "\"\\ud83d\\ude00\"", " [ ] ", "null", "123456789012345678901234567890", "[1e2,2.50e1,3.0]"]:
    add("s04", ["ser"], stdin=t.encode())
add("s04", ["ser", "in.json"], files={"in.json": b'{"k":[1,2,3]}'}); add("s04", ["ser", "missing.json"])
for s in sers: add("s05", ["de"], stdin=s)
for i in range(6):
    v = {str(R.randrange(0, 30)): i for i in range(6)}; v["b"] = 1; v["a"] = 2
    c = add("s04", ["ser"], stdin=json.dumps(v).encode()); add("s05", ["de"], stdin=blob(HID, c["expect"]["stdout"]))
def seal_body(b):
    import importlib; return b
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src", "ref_py"))
import app as REFAPP
def withcrc(body): return body + b"#" + REFAPP.hex8(REFAPP.crc_of(body)).encode() + b"\n"
for body in [b"", b"x", b"n!", b"i;", b"i-;", b"i12", b"i12;x", b"s5:abc", b"s:abc", b"sx", b"[", b"[n", b"{", b"{n", b"{s1:a", b"{s1:ai1;", b"{i1;n}", b"[nn", b"]", b"}", b"nn", b"tt", b"\xff", b"s2:\xc3\xa9", b"i1234567890123456;", b"i123456789012345;", b"[" * 21 + b"]" * 21, b"[" * 20 + b"]" * 20, b"s99999999999999999999:a", b"i-0;", b"i007;", b"{s1:ai1;s1:ai2;}", b"{s2:10i1;s1:9i2;s1:ai3;}", b"s0:", b"[s0:s0:]", b"{s1:a[i1;i2;]s1:b{}}"]:
    add("s06", ["de"], stdin=withcrc(body))
for t in [b"", b"#", b"n", b"n#", b"n#1234", b"n#ZZZZZZZZ", b"n#00000000", b"n#ABCDEF01", b"n#12345678#", b"  \n"]:
    add("s06", ["de"], stdin=t)
s0 = sers[3]
add("s06", ["de"], stdin=s0[:-3] + b"0\n"); add("s06", ["de", "x.ser"], files={"x.ser": s0}); add("s06", ["de", "nope.ser"])

# s07/s08 expr
add7 = lambda *a: add("s07", ["expr"] + list(a))
for e in ["1+2*3", "(1+2)*3", "10/3", "10%3", "-7/2", "-7%2", "7/-2", "7%-2", "2*3*4-5", "1<<4", "256>>2", "-16>>2", "5&3", "5|3", "5^3", "~5", "!0", "!5", "!!7", "-(-3)", "1<2", "2<=2", "3>4", "4>=5", "2==2", "2!=2",
          "1&&0", "1&&2", "0||0", "0||3", "1?2:3", "0?2:3", "1?2:0?3:4", "0?2:0?3:4", "0x1F+1", "0Xff", "007", "min(3,1,2)", "max(3,1,2)", "abs(-5)", "clamp(15,0,10)", "clamp(-5,0,10)", "pow(2,10)", "pow(3,0)", "sign(-9)", "sign(0)", "  1 +  2  ", "1+2*3-4/2%3",
          "2*3<<1", "1+2<<1", "1<2==1", "5&3==1", "1|2^3&4", "- -3", "!-1", "~-1", "((((1))))", "1,2" ]:
    add7(e)
for e, v in [("x+y", ["x=3", "y=4"]), ("x*x-y", ["x=-5", "y=2"]), ("a?b:c", ["a=1", "b=10", "c=20"]), ("foo_1+_bar", ["foo_1=5", "_bar=6"]), ("x/y", ["x=100", "y=7"]), ("x%y", ["x=-100", "y=7"]), ("max(x,y,3)", ["x=1", "y=2"]), ("x", ["x=2147483647"]), ("x", ["x=2147483648"]), ("x", ["x=-2147483649"]), ("x+1", ["x=2147483647"])]:
    add("s07", ["expr", e] + v)
for e in ["2147483647+1", "-2147483648-1", "2147483647*2", "65536*65536", "-2147483648/-1", "-2147483648%-1", "abs(-2147483648)", "-(-2147483648)", "1<<31", "1<<32", "1<<33", "-1>>31", "-1>>32", "1<<-1", "pow(2,31)", "pow(2,32)", "pow(3,100)", "pow(-2,63)", "pow(7,2000000000)", "pow(2,-1)", "0xFFFFFFFF", "0x7FFFFFFF+0", "4294967295", "4294967296", "999999999999999", "9999999999999999", "0xFFFFFFFFFFFFFFF", "0xFFFFFFFFFFFFFFFF", "123456789012345*3", "min(-2147483648,0)", "clamp(5,10,0)", "clamp(5,0)", "max()", "abs(1,2)", "foo(1)", "toString(1)", "min", "1/0", "1%0", "0/0", "x/0", "1?2/0:3", "0?2/0:3", "1||1/0", "0&&1/0", "0||1/0", "1&&1/0", "y", "1+", "+1", "(1", "1)", "()", "1 2", "1 +* 2", "@", "1#", "$x", "1?2", "1?2:", "?1:2", "3 = 3", "1..2", "08", "0x", "0xg", "1e3", "'a'", "pow(2)", "1,", "min(1,)", "min(,1)"]:
    add("s08", ["expr", e])
add("s08", ["expr"]); add("s08", ["expr", "1", "bad"]); add("s08", ["expr", "x", "x=1.5"]); add("s08", ["expr", "x", "x=12345678901"]); add("s08", ["expr", "x", "=1"]); add("s08", ["expr", "x", "x="]); add("s08", ["expr", "x", "x=-"]); add("s08", ["expr", "x+y", "x=1"]); add("s08", ["expr", "x", "x=1", "x=2"]); add("s08", ["expr", "constructor"]); add("s08", ["expr", "__proto__+1"])
for i in range(30):
    def gen(d=0):
        c = R.random()
        if d > 3 or c < .25: return str(R.choice([0, 1, 2, 3, 7, 10, 100, 255, 65535, 2147483647, "0x10", "x", "y", "z"]))
        if c < .35: return R.choice(["-", "!", "~"]) + gen(d + 1)
        if c < .45: return "%s(%s)" % (R.choice(["min", "max"]), ",".join(gen(d + 1) for _ in range(R.randrange(1, 4))))
        if c < .5: return "%s(%s)" % ("abs", gen(d + 1))
        if c < .55: return "(%s?%s:%s)" % (gen(d + 1), gen(d + 1), gen(d + 1))
        return "(%s %s %s)" % (gen(d + 1), R.choice(["+", "-", "*", "/", "%", "&", "|", "^", "<<", ">>", "<", "<=", ">", ">=", "==", "!=", "&&", "||"]), gen(d + 1))
    add("s08", ["expr", gen(), "x=%d" % R.choice([-5, 0, 3, 99, 2147483647]), "y=%d" % R.randrange(-20, 20), "z=%d" % R.choice([0, 1, -1, 12345])])

# s09/s10 tpl
DATA = {"name": "World", "items": ["a", "b", "c"], "n": 3, "flag": True, "no": False, "empty": [], "nested": {"x": {"y": [10, 20, 30]}},
        "users": [{"name": "ann", "age": 31}, {"name": "bob", "age": 17}, {"name": "cy", "age": 0}], "s": "  padded  ", "z": 0, "nul": None,
        "t": "héllo", "mixed": [1, None, "x", [2, 3], True], "obj": {"k": 1}, "es": "", "price": 2.5, "ids": {"7": "seven", "k": "kay"}}
dj = json.dumps(DATA).encode()
TPLS9 = ["Hello {{ name }}!", "{{name}}", "{{ n }}{{ flag }}{{ no }}{{ nul }}{{ missing }}.", "[{{ items }}]", "[{{ mixed }}]", "{{ obj }}", "{{ nested.x.y }}", "{{ nested.x.y[1] }}", "{{ items[0] }}{{ items[2] }}{{ items[5] }}.",
         "{{ items.length }} {{ name.length }} {{ t.length }}", "{{ name[0] }}", "{{ name | upper }} {{ name | lower }} {{ s | trim }}|", "{{ items | join(\"-\") }} {{ items | join }} {{ items | join(0) }}", "{{ items | len }} {{ name | len }} {{ obj | len }} {{ n | len }}",
         "{{ missing | default(\"none\") }} {{ es | default('empty') }} {{ z | default('zero') }} {{ nul | default(7) }}", "{{ items | rev }} {{ name | rev }}", "{{ price }} {{ users[1].age }}", "{{ ids[7] }} {{ ids.k }} {{ ids[\"k\"] }}",
         "{# a comment #}x{# another #}y", "{{ 'lit' }} {{ \"two\" }} {{ 42 }} {{ -5 }} {{ true }} {{ null }}.", "{{ items[n] }}{{ items[nested.x.y.length] }}.", "{{ name | upper | lower | len }}", "{{ users[0].name | upper }} is {{ users[0].age }}", "plain text only", "", "{ not a tag } {x} {{ name }}{",
         "{{ items[ 0 ] }}", "{{ nested . x }}", "tab\there\n{{ name }}\n"]
for t in TPLS9: add("s09", ["tpl", "t.txt", "d.json"], files={"t.txt": t.encode(), "d.json": dj})
for dd in ('[1,2,3]', '"str"', '{"name":null}', '42', 'null'):
    add("s09", ["tpl", "t.txt", "d.json"], files={"t.txt": b"[{{ name }}|{{ length }}|{{ 0 }}]", "d.json": dd.encode()})
TPLS10 = ["{% if flag %}yes{% endif %}", "{% if no %}yes{% else %}no{% endif %}", "{% if items %}{{ items | len }}{% endif %}", "{% if empty %}T{% else %}F{% endif %}", "{% if z %}T{% else %}F{% endif %}{% if es %}T{% else %}F{% endif %}{% if nul %}T{% else %}F{% endif %}{% if missing %}T{% else %}F{% endif %}",
          "{% if not no %}A{% endif %}{% if flag and n == 3 %}B{% endif %}{% if no or z == 0 %}C{% endif %}", "{% if n == 3 %}eq{% endif %}{% if n != 3 %}ne{% endif %}{% if n == \"3\" %}str{% else %}strict{% endif %}", "{% if n < 5 %}lt{% endif %}{% if n > 5 %}gt{% else %}le{% endif %}{% if name < 'X' %}slt{% endif %}{% if n < 'a' %}mixed{% else %}nomix{% endif %}",
          "{% if users[0].age > 18 %}adult{% else %}minor{% endif %}", "{% for i in items %}{{ i }},{% endfor %}", "{% for i in items %}{{ loop.index }}:{{ i }}{% if loop.last %}.{% else %}, {% endif %}{% endfor %}", "{% for u in users %}{{ u.name }}={{ u.age }};{% endfor %}",
          "{% for i in empty %}x{% endfor %}none", "{% for c in name %}[{{ c }}]{% endfor %}", "{% for i in n %}x{% endfor %}.{% for i in obj %}y{% endfor %}.", "{% for a in nested.x.y %}{% for b in items %}{{ a }}{{ b }} {% endfor %}{% endfor %}", "{% for i in items %}{{ loop.index0 }}{{ loop.first }}{{ loop.length }} {% endfor %}",
          "{% for name in items %}{{ name }}{% endfor %}{{ name }}", "{% for i in items %}{% if i == 'b' %}B{% else %}{{ i }}{% endif %}{% endfor %}", "a  {%- if flag -%}  b  {%- endif -%}  c", "x {{- name -}} y", "line1\n{% if flag %}\nline2\n{% endif %}\nline3", "{%- for i in items %}\n  {{ i }}\n{%- endfor %}\n", "{#- c -#}   z",
          "{% for u in users %}{% if u.age %}{{ u.name | upper }} {% endif %}{% endfor %}", "{% if flag %}a{% if n %}b{% else %}c{% endif %}d{% else %}e{% endif %}",
          "{{ name", "{% if flag", "{# open", "{% if flag %}x", "{% for i in items %}x", "{% endif %}", "{% endfor %}", "{% else %}", "{% if flag %}a{% else %}b{% else %}c{% endif %}", "{% for i in items %}{% endif %}", "{% if flag %}{% endfor %}", "{% foo %}", "{% if %}x{% endif %}", "{% for in items %}x{% endfor %}", "{% for i items %}x{% endfor %}",
          "{{ name | nosuch }}", "{{ name | upper( }}", "{{ name | }}", "{{ }}", "{{ 1 2 }}", "{{ name.}}", "{{ items[0 }}", "{{ a b }}", "ok {{ name | bad }} later", "text {% endif %}", "{% endif extra %}", "{{ \"unterminated }}", "{% if flag %}{% else %}{% endif %}{{ x y }}", "{{ items | join(\", \") | upper }}", "{{ name | default }}.", "{{ name == 'World' }} {{ n < 4 }} {{ not flag }} {{ flag and no }}"]
for t in TPLS10: add("s10", ["tpl", "t.txt", "d.json"], files={"t.txt": t.encode(), "d.json": dj})
add("s10", ["tpl", "t.txt", "d.json"], files={"t.txt": b"x", "d.json": b"{bad"}); add("s10", ["tpl", "t.txt", "missing.json"], files={"t.txt": b"x"}); add("s10", ["tpl", "missing.txt", "d.json"], files={"d.json": b"{}"}); add("s10", ["tpl", "t.txt"], files={"t.txt": b"x"})

# s11 pack/unpack
packs = []
for n in [0, 1, 2, 3, 10, 100, 300, 1000, 3000, 6000, 12000, 30000, 70000]:
    d = rbytes(n, "rep" if n % 3 == 0 else None) if n < 6000 else rbytes(n, "rand" if n % 2 else "text")
    c = add("s11", ["pack", "f"], files={"f": d}); packs.append((d, blob(HID, c["expect"]["stdout"])))
for _ in range(3):
    d = bytes(R.randrange(256) for _ in range(9000)); c = add("s11", ["pack"], stdin=d); packs.append((d, blob(HID, c["expect"]["stdout"])))
for d, p in packs:
    add("s11", ["unpack"], stdin=p)
good = packs[7][1]
for t in [b"", b"LZW1", b"LZW1\0\0\0\0", b"LZW1\0\0\0\0\0", b"XXXX\0\0\0\0\0\0", good[:-1], good[:-5], good[:10], good + b"\0", good[:4] + b"\0\0\0\1" + good[8:], good[:-1] + bytes([(good[-1] + 1) & 255]), good[:9] + bytes([good[9] ^ 0xff]) + good[10:],
          b"LZW1\0\0\0\1" + bytes([0x00, 0x03, 0x00]) + b"\0", b"LZW1\0\0\0\0" + bytes([0x00, 0x02, 0x00]) + b"\0"]:
    add("s11", ["unpack"], stdin=t)
add("s11", ["unpack", "nope"])

# s12 seal/open
seals = []
for n, key in [(0, "k"), (1, "k"), (5, "secret"), (16, "s3cr3t!"), (17, "k"), (100, "a longer pass phrase"), (1000, "é中"), (255, "0"), (256, ""), (4097, "k")]:
    d = rbytes(n); c = add("s12", ["seal", key], stdin=d) if n else add("s12", ["seal", key, "e.bin"], files={"e.bin": b""})
    seals.append((key, d, blob(HID, c["expect"]["stdout"])))
for key, d, s in seals: add("s12", ["open", key], stdin=s)
for key, d, s in seals[1:7]:
    add("s12", ["open", key + "x"], stdin=s)
    b = bytearray(s.strip()); i = R.randrange(len(b)); b[i] = ord("0") if b[i] != ord("0") else ord("1"); add("s12", ["open", key], stdin=bytes(b))
    add("s12", ["open", key], stdin=s.strip()[:-2]); add("s12", ["open", key], stdin=s.strip().upper())
for t in [b"", b"abc", b"zz", b"0011223344556677", b"00112233445566", b"  \n", b"0g"]:
    add("s12", ["open", "k"], stdin=t)
add("s12", ["seal"]); add("s12", ["open"]); add("s12", ["seal", "k", "nope"])

# s13 cli
add("s13", []); add("s13", ["--version"]); add("s13", ["help"]); add("s13", ["CRC", "-s", "x"]); add("s13", ["constructor"]); add("s13", ["__proto__"]); add("s13", ["toString"]); add("s13", ["-s"]); add("s13", ["crc", "-s", "a", "extra"]); add("s13", ["ser", "a", "b"], files={"a": b"1", "b": b"2"})
add("s13", ["unpack", "-"], stdin=packs[2][1]); add("s13", ["b32", "enc", "-"], stdin=b"dash"); add("s13", ["de", "-"], stdin=sers[0]); add("s13", ["crc", "dir"], files={"dir/x": b"1"}); add("s13", ["--version", "x"]); add("s13", ["expr", "--version"])

# public samples
os.makedirs(PUB, exist_ok=True)
for name, data in (("ser_in.json", b'{"name":"example","list":[1,2,3],"nested":{"b":true,"a":null}}'), ("tpl.txt", b"Hello {{ name | upper }}!\n{% for i in items %}{{ loop.index }}={{ i }} {% endfor %}\n"),
                   ("tpl.json", b'{"name":"sample","items":["x","y"]}'), ("hello.txt", b"hello hello hello hello\n")):
    open(os.path.join(PUB, name), "wb").write(data)
print("cases", rec.save())
