#!/usr/bin/env python3
"""Runs INSIDE the build container: records hidden cases for re-vault against the reference binary.
  gen_cases.py REF_BINARY HIDDEN_DIR PUBLIC_DIR
"""
import os, random, sys
from casegen import Recorder
from mscore import run

REF, HID, PUB = sys.argv[1], sys.argv[2], sys.argv[3]
R = random.Random(20261014)
WORDS = "the of and vault archive entry index header block stream method store pack key salt check sum crc byte word line file name path mode size raw comp offset".split()


def text(n):
    out = []
    while sum(len(w) + 1 for w in out) < n:
        out.append(R.choice(WORDS))
    return (" ".join(out) + "\n").encode()[:n]


def kinds(n):
    k = R.choice(["text", "rep", "zeros", "rand", "mixed", "records", "lowent"])
    if k == "text": return text(n)
    if k == "rep":
        p = bytes(R.randrange(256) for _ in range(R.randrange(1, 40)))
        return (p * (n // len(p) + 1))[:n]
    if k == "zeros": return b"\0" * n
    if k == "rand": return bytes(R.randrange(256) for _ in range(n))
    if k == "lowent": return bytes(R.choice(b"ab\n ") for _ in range(n))
    if k == "records":
        b = bytearray()
        i = 0
        while len(b) < n:
            b += i.to_bytes(4, "little") + bytes([i % 7]) * 3 + b"REC\0"; i += 1
        return bytes(b[:n])
    parts = []
    while sum(map(len, parts)) < n:
        c = R.random()
        if c < .35: parts.append(bytes([R.randrange(256)]) * R.randrange(1, 300))
        elif c < .7: parts.append(text(R.randrange(5, 200)))
        else: parts.append(bytes(R.randrange(256) for _ in range(R.randrange(1, 60))))
    return b"".join(parts)[:n]


def size():
    return R.choice([0, 1, 2, 3, 5, 17, 100, 255, 256, 257, 700, 1500, 4095, 4096, 4097, 5000, 9000, 20000])


def fileset(k=None):
    k = k or R.randrange(1, 6)
    fs = {}
    for i in range(k):
        name = R.choice(["a", "b", "data/c", "docs/readme.txt", "x/y/z.bin", "log.txt", "notes", "img.raw", "t"]) + (f"_{i}" if R.random() < .7 else "")
        fs[name] = kinds(size())
    return fs


def modes_for(fs):
    return {n: R.choice([0o644, 0o755, 0o600, 0o640]) for n in fs}


rec = Recorder(HID, [REF])
SMALLKEYS = ["k", "hunter2", "correct horse battery staple", "Zz9!", "0"]


def archive(fs, method, key=None, modes=None):
    """Make an archive with the reference; returns bytes."""
    r = Recorder(HID, [REF]); names = list(fs)
    args = ["pack"] + (["-k", key] if key else []) + (["-m", str(method)] if method is not None else []) + ["out.vlt"] + names
    c = r.add("tmp", args, files=fs, capture=["out.vlt"], modes=modes)
    from mscore import blob
    return blob(HID, c["expect"]["out_files"]["out.vlt"])


# m01 digest
for i in range(30):
    d = kinds(size()); rec.add("m01", ["digest", "f"], files={"f": d})
rec.add("m01", ["digest", "missing"], files={})
rec.add("m01", ["digest"], files={})

archs = {}
# m02 info/list on store archives
for i in range(20):
    fs = fileset(); mo = modes_for(fs); a = archive(fs, 0, modes=mo)
    for args in (["info", "a.vlt"], ["list", "a.vlt"], ["list", "-l", "a.vlt"]):
        rec.add("m02", args, files={"a.vlt": a})
# m03..m05 cat per method
for ms, meth, cnt in (("m03", 0, 40), ("m04", 1, 40), ("m05", 2, 25)):
    for i in range(cnt // 4 + 1):
        fs = fileset(R.randrange(2, 6)); a = archive(fs, meth, modes=modes_for(fs))
        for n in list(fs)[:4]:
            rec.add(ms, ["cat", "a.vlt", n], files={"a.vlt": a})
for i in range(15):
    fs = fileset(R.randrange(2, 5)); a = archive(fs, None)   # auto archives
    for n in list(fs)[:2]:
        rec.add("m05", ["cat", "a.vlt", n], files={"a.vlt": a})
# m06 extract + verify, good and corrupted
for i in range(14):
    fs = fileset(R.randrange(1, 5)); a = archive(fs, R.choice([None, 0, 1, 2]))
    cap = ["out/" + n for n in fs]
    rec.add("m06", ["extract", "a.vlt", "out"], files={"a.vlt": a}, capture=cap)
    rec.add("m06", ["verify", "a.vlt"], files={"a.vlt": a})
    if len(a) > 80:
        b = bytearray(a); pos = R.randrange(28, max(29, len(a) - 30)); b[pos] ^= 1 << R.randrange(8)
        rec.add("m06", ["verify", "a.vlt"], files={"a.vlt": bytes(b)})
        rec.add("m06", ["info", "a.vlt"], files={"a.vlt": bytes(b)})
# m07 encrypted reads
for i in range(10):
    fs = fileset(R.randrange(1, 5)); key = R.choice(SMALLKEYS); a = archive(fs, R.choice([None, 0, 1, 2]), key=key)
    n0 = list(fs)[0]
    rec.add("m07", ["cat", "-k", key, "a.vlt", n0], files={"a.vlt": a})
    rec.add("m07", ["cat", "-k", "wrongkey", "a.vlt", n0], files={"a.vlt": a})
    rec.add("m07", ["cat", "a.vlt", n0], files={"a.vlt": a})
    rec.add("m07", ["verify", "-k", key, "a.vlt"], files={"a.vlt": a})
    rec.add("m07", ["extract", "-k", key, "a.vlt", "o"], files={"a.vlt": a}, capture=["o/" + n for n in fs])
    rec.add("m07", ["info", "a.vlt"], files={"a.vlt": a})
    rec.add("m07", ["list", "-l", "a.vlt"], files={"a.vlt": a})
# m08..m12 pack byte-exact
def packcase(ms, meth, key=None):
    fs = fileset(R.randrange(1, 6)); mo = modes_for(fs)
    args = ["pack"] + (["-k", key] if key else []) + (["-m", str(meth)] if meth is not None else []) + ["out.vlt"] + list(fs)
    rec.add(ms, args, files=fs, capture=["out.vlt"], modes=mo)
for i in range(25): packcase("m08", 0)
for i in range(25): packcase("m09", 1)
for i in range(30): packcase("m10", 2)
# extra LZ stress: distances around window limits and len-3 rule
for dist in (1, 2, 3, 100, 2047, 2048, 2049, 4094, 4095, 4096, 4097):
    base = bytes(R.randrange(256) for _ in range(dist)); d = base + b"XYZ" + bytes(R.randrange(256) for _ in range(5)) + base[:3]
    d = base[:3] + bytes(R.randrange(256) for _ in range(max(dist - 3, 0))) + base[:3] + b"tail"
    rec.add("m10", ["pack", "-m", "2", "out.vlt", "f"], files={"f": d}, capture=["out.vlt"])
for i in range(30): packcase("m11", None)
for i in range(30): packcase("m12", R.choice([None, 0, 1, 2]), key=R.choice(SMALLKEYS))
# m13/m14 errors
errs = []
fs = fileset(3); good = archive(fs, None); names = list(fs)
enc = archive(fs, None, key="k")
def err(args, files=None):
    errs.append((args, files or {}))
err(["pack", "o.vlt", "nope"], {})
err(["pack", "o.vlt", "../x"], {"../x": b"1"} if False else {})
err(["pack", "o.vlt", "/etc/passwd"])
err(["pack", "o.vlt", "a/../b"], {})
err(["pack", "o.vlt", "f", "f"], {"f": b"abc"})
err(["pack", "o.vlt", "f", "./f"], {"f": b"abc"})
err(["pack", "-m", "7", "o.vlt", "f"], {"f": b"abc"})
err(["pack", "-z", "1", "o.vlt", "f"], {"f": b"abc"})
err(["pack", "o.vlt"], {})
err(["pack"], {})
err([], {})
err(["frobnicate"], {})
err(["list"], {})
err(["list", "missing.vlt"], {})
err(["list", "a.vlt"], {"a.vlt": b"NOPE" + b"\0" * 40})
err(["list", "a.vlt"], {"a.vlt": b"VLT3"})
err(["info", "a.vlt"], {"a.vlt": good[:len(good) // 2]})
err(["list", "a.vlt"], {"a.vlt": good[:-6]})
err(["cat", "a.vlt", "nonexistent"], {"a.vlt": good})
err(["cat", "a.vlt"], {"a.vlt": good})
err(["extract", "a.vlt", "o", "nonexistent"], {"a.vlt": good})
err(["extract", "a.vlt"], {"a.vlt": good})
err(["verify", "a.vlt", "extra"], {"a.vlt": good})
err(["cat", "a.vlt", names[0]], {"a.vlt": enc})
err(["cat", "-k", "bad", "a.vlt", names[0]], {"a.vlt": enc})
err(["verify", "a.vlt"], {"a.vlt": enc})
err(["digest", "a", "b"], {"a": b"", "b": b""})
b = bytearray(good); b[4] = 9
err(["info", "a.vlt"], {"a.vlt": bytes(b)})
b = bytearray(good); b[12:16] = (len(good) + 50).to_bytes(4, "little")
err(["list", "a.vlt"], {"a.vlt": bytes(b)})
b = bytearray(good); b[-2] ^= 0xff
err(["info", "a.vlt"], {"a.vlt": bytes(b)})
err(["verify", "a.vlt"], {"a.vlt": bytes(b)})
for k in range(6):
    b = bytearray(good); p = R.randrange(28, int.from_bytes(good[12:16], "little") or 29); b[p] ^= 0xff
    err(["cat", "a.vlt", names[0]], {"a.vlt": bytes(b)}); err(["verify", "a.vlt"], {"a.vlt": bytes(b)})
for args, files in errs:
    rec.add("m13", args, files=files)
    rec.add("m14", args, files=files, cmp=["rc", "stdout", "stderr"])
# m14 extra: extract subsets, nested dirs, modes via list -l after pack
for i in range(12):
    fs = fileset(R.randrange(2, 6)); a = archive(fs, None, modes=modes_for(fs))
    pick = R.sample(list(fs), min(2, len(fs)))
    rec.add("m14", ["extract", "a.vlt", "deep/er/dir"] + pick, files={"a.vlt": a}, capture=["deep/er/dir/" + n for n in fs])
# public samples for the agent
os.makedirs(PUB, exist_ok=True)
sf = {"hello.txt": b"hello hello hello vault\n", "zeros.bin": b"\0" * 1000, "mix.dat": kinds(900)}
for name, args in (("store.vlt", ["-m", "0"]), ("rle.vlt", ["-m", "1"]), ("lz.vlt", ["-m", "2"]), ("auto.vlt", []), ("secret.vlt", ["-k", "swordfish"])):
    a = archive({"hello.txt": sf["hello.txt"], "zeros.bin": sf["zeros.bin"], "mix.dat": sf["mix.dat"]}, None if not args else (int(args[1]) if args[0] == "-m" else None), key="swordfish" if name == "secret.vlt" else None)
    open(os.path.join(PUB, name), "wb").write(a)
for n, d in sf.items():
    open(os.path.join(PUB, n), "wb").write(d)
print("cases", rec.save())
