#!/usr/bin/env python3
"""Runs INSIDE the build container: records re-proto session scripts against the reference binary.
  gen_cases.py REF_KVD_BINARY HIDDEN_DIR PUBLIC_DIR      (PYTHONPATH must contain lib/ ; src/ref_kvd.py at /src)
"""
import json, os, random, struct, sys
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, "/src/src")
import ref_kvd as R
from kvsession import *

REF, HID, PUB = sys.argv[1], sys.argv[2], sys.argv[3]
Rn = random.Random(20261014)
TESTS = []   # (ms, name, seed, actions)
SEEDS = [1, 7, 42, 1000003, 0xDEADBEEF, 99, 5, 123456789]


class T:
    def __init__(s, ms, name, seed=None):
        s.ms, s.name, s.seed = ms, name, seed if seed is not None else Rn.choice(SEEDS)
        s.acts, s.n, s.rid = [], 0, 100
        TESTS.append(s)

    def conn(s, user=None):
        s.n += 1; cid = s.n
        s.acts.append(("connect", cid))
        if user:
            s.login(cid, user)
        return cid

    def nonce(s, cid):
        return struct.pack("<Q", R.splitmix((s.seed + cid * 0x9E3779B97F4A7C15) & R.M64))

    def login(s, cid, user, good=True, secret=None):
        u = user.encode()
        sec = secret if secret is not None else R.USERS.get(u, (b"nope", 0))[0]
        h = R.auth_hash(sec, s.nonce(cid), u) ^ (0 if good else 1)
        s.send(cid, OP["AUTH"], bytes([len(u)]) + u + struct.pack("<Q", h))

    def send(s, cid, op, pay=b"", rid=None, chunks=None):
        if rid is None:
            s.rid += 1; rid = s.rid
        s.acts.append(("send", cid, fr(op, rid, pay), chunks))

    def raw(s, cid, b, chunks=None):
        s.acts.append(("send", cid, b, chunks))

    def close(s, cid):
        s.acts.append(("close", cid))

    def cmd(s, cid, name, pay=b""):
        s.send(cid, OP[name], pay)


def sk(i): return b"key%d" % i

# ---------------- m01 greeting + ping ----------------
for i in range(12):
    t = T("m01", f"ping{i}")
    a = t.conn()
    for pl in Rn.sample([b"", b"x", b"hello", bytes(range(100)), bytes(512), b"\x00\x01\xff"], 3):
        t.cmd(a, "PING", pl)
    if i % 3 == 0:
        b = t.conn(); t.send(b, 1, b"two", rid=0xFFFFFFFF); t.send(a, 1, b"a", rid=0)
    if i % 4 == 1:
        c = t.conn(); d = t.conn(); t.cmd(d, "PING", b"d"); t.cmd(c, "PING", b"c")

# ---------------- m02 framing errors ----------------
def m02():
    t = T("m02", "badmagic1"); a = t.conn(); t.raw(a, fr(1, 5, b"x", magic=b"XV"))
    t = T("m02", "badmagic2"); a = t.conn(); t.raw(a, fr(1, 5, b"x", magic=b"KW"))
    t = T("m02", "badmagic3"); a = t.conn(); t.raw(a, b"GET / HTTP/1.1\r\n\r\n")
    for v in (0, 1, 3, 255):
        t = T("m02", f"badver{v}"); a = t.conn(); t.raw(a, fr(1, 77, b"x", ver=v))
    for ln in (513, 1000, 65535):
        t = T("m02", f"oversize{ln}"); a = t.conn(); t.raw(a, b"KV\x02\x01" + struct.pack("<IH", 9, ln))
    t = T("m02", "oversize_payload"); a = t.conn(); t.raw(a, b"KV\x02\x01" + struct.pack("<IH", 9, 600) + bytes(600))
    t = T("m02", "byte_at_a_time"); a = t.conn(); t.raw(a, fr(1, 3, b"abc"), chunks=[1] * 13)
    t = T("m02", "split_header"); a = t.conn(); t.raw(a, fr(1, 3, b"abcdef"), chunks=[9, 3])
    t = T("m02", "short_then_close"); a = t.conn(); t.raw(a, fr(1, 3, b"abcdef")[:7]); t.close(a)
    t = T("m02", "noauth_get"); a = t.conn(); t.cmd(a, "GET", kb(b"x")); t.cmd(a, "SET", p_set(b"x", 1, 0, b"v"))
    t = T("m02", "noauth_unknown"); a = t.conn(); t.send(a, 200); t.send(a, 0)
    t = T("m02", "unknown_authed"); a = t.conn("alice"); t.send(a, 19); t.send(a, 200, b"zz"); t.send(a, 0); t.cmd(a, "PING", b"ok")
m02()

# ---------------- m03 auth ----------------
for u in ("alice", "bob", "carol", "dave"):
    t = T("m03", f"login_{u}"); a = t.conn(u); t.cmd(a, "PING", b"p")
t = T("m03", "bad_hash"); a = t.conn(); t.login(a, "alice", good=False); t.login(a, "alice")
t = T("m03", "bad_user"); a = t.conn(); t.login(a, "mallory"); t.login(a, "")
t = T("m03", "three_strikes"); a = t.conn(); [t.login(a, "bob", good=False) for _ in range(3)]
t = T("m03", "two_then_ok"); a = t.conn(); t.login(a, "bob", good=False); t.login(a, "bob", good=False); t.login(a, "bob"); t.cmd(a, "PING")
t = T("m03", "double_auth"); a = t.conn("alice"); t.login(a, "alice")
t = T("m03", "auth_truncated"); a = t.conn(); t.send(a, 2, b"\x05ali"); t.send(a, 2); t.send(a, 2, b"\x05alice" + bytes(7)); t.send(a, 2, b"\x03bob" + bytes(9))
t = T("m03", "nonce_per_conn"); a = t.conn(); b = t.conn(); c = t.conn(); t.login(c, "carol"); t.login(b, "bob"); t.login(a, "alice")
t = T("m03", "cross_conn_replay"); a = t.conn(); b = t.conn(); h = R.auth_hash(b"wonderland", t.nonce(a), b"alice")
t.send(b, 2, b"\x05alice" + struct.pack("<Q", h)); t.send(a, 2, b"\x05alice" + struct.pack("<Q", h))
t = T("m03", "wrong_secret"); a = t.conn(); t.login(a, "carol", secret=b"wonderland"); t.login(a, "carol")
t = T("m03", "seed_zero", seed=0); a = t.conn("dave"); t.cmd(a, "PING", b"z")
t = T("m03", "seed_big", seed=2**64 - 1); a = t.conn("dave"); b = t.conn("bob")
t = T("m03", "fail_other_ok"); a = t.conn(); b = t.conn("alice"); t.login(a, "x"); t.login(a, "y"); t.cmd(b, "PING")

# ---------------- m04 set/get int/str ----------------
for i in range(16):
    t = T("m04", f"setget{i}"); a = t.conn("bob")
    for j in range(Rn.randrange(2, 5)):
        k = sk(Rn.randrange(4))
        if Rn.random() < .5:
            t.cmd(a, "SET", p_set(k, 0, Rn.choice([0, 0, 5, 1000]), vint(Rn.choice([0, 1, -1, 2**63 - 1, -2**63, 123456789012]))))
        else:
            t.cmd(a, "SET", p_set(k, 1, Rn.choice([0, 0, 9]), bytes(Rn.randrange(256) for _ in range(Rn.choice([0, 1, 10, 400])))))
        t.cmd(a, "GET", kb(k))
    t.cmd(a, "GET", kb(b"absent"))

# ---------------- m05 lists, del, wrong types, bad args ----------------
for i in range(6):
    t = T("m05", f"list{i}"); a = t.conn("alice")
    items = [bytes(Rn.randrange(256) for _ in range(Rn.randrange(0, 20))) for _ in range(Rn.randrange(0, 8))]
    t.cmd(a, "SET", p_set(b"L", 2, 0, vlist(items))); t.cmd(a, "GET", kb(b"L")); t.cmd(a, "DEL", kb(b"L")); t.cmd(a, "DEL", kb(b"L")); t.cmd(a, "GET", kb(b"L"))
t = T("m05", "bad_lists"); a = t.conn("bob")
for v in (b"", b"\x01", b"\x01\x05ab", b"\x02\x01a", b"\x01\x01a\x00", bytes([17]) + b"\x00" * 17, bytes([1, 200]) + bytes(200)):
    t.cmd(a, "SET", p_set(b"L", 2, 0, v))
t = T("m05", "bad_args"); a = t.conn("bob")
t.cmd(a, "SET", p_set(b"", 1, 0, b"v")); t.cmd(a, "SET", p_set(b"k" * 32, 1, 0, b"v")); t.cmd(a, "SET", p_set(b"k" * 33, 1, 0, b"v"))
t.cmd(a, "SET", p_set(b"k", 3, 0, b"v")); t.cmd(a, "SET", p_set(b"k", 0, 0, b"short")); t.cmd(a, "SET", p_set(b"k", 1, 0, bytes(401))); t.cmd(a, "SET", b"\x01k")
t.cmd(a, "GET", b""); t.cmd(a, "GET", b"\x05ab"); t.cmd(a, "GET", kb(b"k") + b"x"); t.cmd(a, "DEL", b"\x00")
t = T("m05", "wrongtype"); a = t.conn("bob"); t.cmd(a, "SET", p_set(b"s", 1, 0, b"str")); t.cmd(a, "INCR", p_incr(b"s", 1)); t.cmd(a, "SET", p_set(b"l", 2, 0, vlist([b"a"]))); t.cmd(a, "INCR", p_incr(b"l", 1))
t = T("m05", "overwrite_type"); a = t.conn("bob"); t.cmd(a, "SET", p_set(b"s", 1, 7, b"str")); t.cmd(a, "SET", p_set(b"s", 0, 0, vint(5))); t.cmd(a, "GET", kb(b"s")); t.cmd(a, "INCR", p_incr(b"s", 1))
t = T("m05", "del_many"); a = t.conn("bob")
for i in range(5): t.cmd(a, "SET", p_set(sk(i), 1, 0, b"v"))
for i in range(7): t.cmd(a, "DEL", kb(sk(i)))
t = T("m05", "key_binary"); a = t.conn("bob"); t.cmd(a, "SET", p_set(b"\x01\x02\xff", 1, 0, b"v")); t.cmd(a, "GET", kb(b"\x01\x02\xff")); t.cmd(a, "KEYS", b"\x00")
t = T("m05", "empty_values"); a = t.conn("bob"); t.cmd(a, "SET", p_set(b"e", 1, 0, b"")); t.cmd(a, "GET", kb(b"e")); t.cmd(a, "SET", p_set(b"f", 2, 0, vlist([]))); t.cmd(a, "GET", kb(b"f"))

# ---------------- m06 incr ----------------
for i in range(10):
    t = T("m06", f"incr{i}"); a = t.conn(Rn.choice(["bob", "alice", "dave"]))
    k = b"ctr"
    for d in [Rn.choice([1, -1, 5, 2**62, -2**62, 2**63 - 1, -2**63]) for _ in range(Rn.randrange(3, 7))]:
        t.cmd(a, "INCR", p_incr(k, d))
    t.cmd(a, "GET", kb(k))
    if i % 3 == 0:
        t.cmd(a, "SET", p_set(b"tt", 0, 50, vint(1))); t.cmd(a, "INCR", p_incr(b"tt", 2)); t.cmd(a, "GET", kb(b"tt")); t.cmd(a, "INCR", b"\x01k"); t.cmd(a, "INCR", p_incr(b"k" * 33, 1))

# ---------------- m07 permissions ----------------
WRITE_OPS = [("SET", p_set(b"w", 1, 0, b"v")), ("DEL", kb(b"w")), ("INCR", p_incr(b"w", 1)), ("EXPIRE", p_expire(b"w", 5)), ("TICK", struct.pack("<I", 1))]
READ_OPS = [("GET", kb(b"r")), ("MGET", p_mget([b"r"])), ("KEYS", kb(b"*")), ("TTL", kb(b"r")), ("INFO", b""), ("WATCH", kb(b"r")), ("UNWATCH", kb(b"r")), ("BEGIN", b""), ("COMMIT", b""), ("PING", b"x")]
for u in ("carol", "bob", "dave", "alice"):
    t = T("m07", f"writes_{u}"); a = t.conn(u)
    for n, p in WRITE_OPS: t.cmd(a, n, p)
    t = T("m07", f"reads_{u}"); a = t.conn(u); adm = t.conn("alice"); t.cmd(adm, "SET", p_set(b"r", 1, 0, b"v"))
    for n, p in READ_OPS: t.cmd(a, n, p)
t = T("m07", "ro_in_tx"); a = t.conn("carol"); t.cmd(a, "BEGIN"); t.cmd(a, "SET", p_set(b"w", 1, 0, b"v")); t.cmd(a, "COMMIT")
t = T("m07", "perm_before_args"); a = t.conn("carol"); t.cmd(a, "SET", b""); t.cmd(a, "TICK", b""); t.cmd(a, "QUIT")

# ---------------- m08 ttl/tick ----------------
t = T("m08", "basic_expiry"); a = t.conn("alice"); t.cmd(a, "SET", p_set(b"k", 1, 5, b"v")); t.cmd(a, "TICK", struct.pack("<I", 3)); t.cmd(a, "GET", kb(b"k")); t.cmd(a, "TTL", kb(b"k")); t.cmd(a, "TICK", struct.pack("<I", 2)); t.cmd(a, "GET", kb(b"k")); t.cmd(a, "TTL", kb(b"k"))
t = T("m08", "no_ttl_ttl"); a = t.conn("alice"); t.cmd(a, "SET", p_set(b"k", 1, 0, b"v")); t.cmd(a, "TTL", kb(b"k")); t.cmd(a, "TICK", struct.pack("<I", 1000)); t.cmd(a, "GET", kb(b"k"))
t = T("m08", "tick_args"); a = t.conn("alice")
for pl in (b"", struct.pack("<I", 0), struct.pack("<I", 100001), struct.pack("<I", 100000), b"\x01\x00", struct.pack("<Q", 5)): t.cmd(a, "TICK", pl)
t = T("m08", "expire_cmd"); a = t.conn("alice"); t.cmd(a, "SET", p_set(b"k", 1, 0, b"v")); t.cmd(a, "EXPIRE", p_expire(b"k", 4)); t.cmd(a, "TTL", kb(b"k")); t.cmd(a, "EXPIRE", p_expire(b"k", 0)); t.cmd(a, "TTL", kb(b"k")); t.cmd(a, "EXPIRE", p_expire(b"nokey", 3)); t.cmd(a, "TTL", kb(b"nokey"))
t = T("m08", "tick_count"); a = t.conn("alice")
for i in range(6): t.cmd(a, "SET", p_set(sk(i), 0, 2 + i, vint(i)))
for _ in range(4): t.cmd(a, "TICK", struct.pack("<I", 3))
t.cmd(a, "KEYS", kb(b"*")); t.cmd(a, "INFO")
t = T("m08", "ttl_arith"); a = t.conn("alice"); t.cmd(a, "TICK", struct.pack("<I", 100)); t.cmd(a, "SET", p_set(b"k", 1, 50, b"v")); t.cmd(a, "GET", kb(b"k")); t.cmd(a, "TICK", struct.pack("<I", 49)); t.cmd(a, "GET", kb(b"k")); t.cmd(a, "TICK", struct.pack("<I", 1)); t.cmd(a, "GET", kb(b"k"))
t = T("m08", "overwrite_clears_ttl"); a = t.conn("alice"); t.cmd(a, "SET", p_set(b"k", 1, 3, b"v")); t.cmd(a, "SET", p_set(b"k", 1, 0, b"w")); t.cmd(a, "TICK", struct.pack("<I", 5)); t.cmd(a, "GET", kb(b"k"))
t = T("m08", "incr_keeps_ttl"); a = t.conn("alice"); t.cmd(a, "SET", p_set(b"k", 0, 3, vint(1))); t.cmd(a, "INCR", p_incr(b"k", 1)); t.cmd(a, "TICK", struct.pack("<I", 3)); t.cmd(a, "GET", kb(b"k"))
t = T("m08", "expire_then_tick"); a = t.conn("alice"); t.cmd(a, "SET", p_set(b"k", 1, 0, b"v")); t.cmd(a, "EXPIRE", p_expire(b"k", 1)); t.cmd(a, "TICK", struct.pack("<I", 1)); t.cmd(a, "TTL", kb(b"k"))
t = T("m08", "info_clock"); a = t.conn("alice"); t.cmd(a, "INFO"); t.cmd(a, "TICK", struct.pack("<I", 77)); t.cmd(a, "INFO")
t = T("m08", "tick_shared_clock"); a = t.conn("alice"); b = t.conn("bob"); t.cmd(b, "SET", p_set(b"k", 1, 4, b"v")); t.cmd(a, "TICK", struct.pack("<I", 4)); t.cmd(b, "GET", kb(b"k")); t.cmd(b, "INFO")
for i in range(4):
    t = T("m08", f"ttl_random{i}"); a = t.conn("alice")
    for j in range(6):
        if Rn.random() < .5: t.cmd(a, "SET", p_set(sk(Rn.randrange(3)), 1, Rn.choice([0, 1, 2, 5]), b"v"))
        else: t.cmd(a, "TICK", struct.pack("<I", Rn.choice([1, 2, 3])))
        t.cmd(a, "TTL", kb(sk(Rn.randrange(3))))

# ---------------- m09 keys / mget / info ----------------
for i in range(5):
    t = T("m09", f"keys{i}"); a = t.conn("bob")
    for k in Rn.sample([b"apple", b"app", b"ap", b"banana", b"band", b"b", b"zed", b"a*", b"Apple"], 6): t.cmd(a, "SET", p_set(k, 1, 0, b"v"))
    for pat in (b"", b"*", b"app*", b"app", b"b*", b"zz*", b"a*", b"ban*d"): t.cmd(a, "KEYS", kb(pat))
for i in range(4):
    t = T("m09", f"mget{i}"); a = t.conn("bob"); t.cmd(a, "SET", p_set(b"x", 1, 3, b"1")); t.cmd(a, "SET", p_set(b"y", 0, 0, vint(2))); t.cmd(a, "SET", p_set(b"z", 2, 0, vlist([b"a", b"bc"])))
    t.cmd(a, "MGET", p_mget([b"x", b"nope", b"y", b"z"]))
    t.cmd(a, "MGET", p_mget([b"x"] * 8)); t.cmd(a, "MGET", p_mget([])); t.cmd(a, "MGET", p_mget([b"x"] * 9)); t.cmd(a, "MGET", p_mget([b"x"]) + b"junk"); t.cmd(a, "MGET", b"\x02\x01x")
t = T("m09", "info_state"); a = t.conn("bob"); b = t.conn("alice"); t.cmd(a, "INFO"); t.cmd(a, "WATCH", kb(b"a*")); t.cmd(a, "WATCH", kb(b"b")); t.cmd(a, "INFO"); t.cmd(b, "SET", p_set(b"q", 1, 0, b"v")); t.cmd(b, "INFO"); t.cmd(a, "INFO", b"x")
t = T("m09", "mget_bigvals"); a = t.conn("bob")
for k in (b"a", b"b"): t.cmd(a, "SET", p_set(k, 1, 0, bytes(300)))
t.cmd(a, "MGET", p_mget([b"a", b"b"])); t.cmd(a, "MGET", p_mget([b"a"]))

# ---------------- m10 capacity ----------------
t = T("m10", "fill_keys"); a = t.conn("bob")
for i in range(66): t.cmd(a, "SET", p_set(b"k%02d" % i, 0, 0, vint(i)))
t.cmd(a, "SET", p_set(b"k00", 1, 0, b"overwrite-ok")); t.cmd(a, "INCR", p_incr(b"new", 1)); t.cmd(a, "DEL", kb(b"k10")); t.cmd(a, "INCR", p_incr(b"new", 1)); t.cmd(a, "INFO")
t = T("m10", "watch_limit"); a = t.conn("bob")
for i in range(10): t.cmd(a, "WATCH", kb(b"w%d" % i))
t.cmd(a, "WATCH", kb(b"w0")); t.cmd(a, "UNWATCH", kb(b"w3")); t.cmd(a, "WATCH", kb(b"w9")); t.cmd(a, "UNWATCH", kb(b"nonesuch")); t.cmd(a, "UNWATCH", kb(b"w3"))
t = T("m10", "fill_by_two"); a = t.conn("bob"); b = t.conn("alice")
for i in range(40): t.cmd(a, "SET", p_set(b"a%02d" % i, 1, 0, b"")); 
for i in range(40): t.cmd(b, "SET", p_set(b"b%02d" % i, 1, 0, b""))
t.cmd(a, "KEYS", kb(b"b3*"))
t = T("m10", "full_in_tx"); a = t.conn("bob")
for i in range(62): t.cmd(a, "SET", p_set(b"k%02d" % i, 1, 0, b""))
t.cmd(a, "BEGIN"); t.cmd(a, "SET", p_set(b"n1", 1, 0, b"")); t.cmd(a, "SET", p_set(b"n2", 1, 0, b"")); t.cmd(a, "SET", p_set(b"n3", 1, 0, b"")); t.cmd(a, "COMMIT"); t.cmd(a, "INFO")
t = T("m10", "long_values"); a = t.conn("bob"); t.cmd(a, "SET", p_set(b"k", 1, 0, bytes(400))); t.cmd(a, "GET", kb(b"k")); t.cmd(a, "SET", p_set(b"k", 2, 0, vlist([bytes(100)] * 3 + [bytes(90)]))); t.cmd(a, "SET", p_set(b"k", 2, 0, vlist([bytes(100)] * 3 + [bytes(97)])))
t = T("m10", "max_payload_set"); a = t.conn("bob"); t.cmd(a, "SET", p_set(b"k" * 32, 1, 0, bytes(400))); t.cmd(a, "PING", bytes(512)); t.send(a, 1, bytes(512))

# ---------------- m11 transactions ----------------
for i in range(7):
    t = T("m11", f"tx_basic{i}"); a = t.conn("bob"); b = t.conn("alice")
    t.cmd(b, "SET", p_set(b"base", 1, 0, b"old"))
    t.cmd(a, "BEGIN"); t.cmd(a, "SET", p_set(b"x", 1, 0, b"new")); t.cmd(a, "GET", kb(b"x")); t.cmd(b, "GET", kb(b"x")); t.cmd(a, "INFO")
    if i % 3 == 0: t.cmd(a, "DEL", kb(b"base")); t.cmd(a, "GET", kb(b"base")); t.cmd(b, "GET", kb(b"base"))
    if i % 3 == 1: t.cmd(a, "INCR", p_incr(b"n", 5)); t.cmd(a, "INCR", p_incr(b"n", 5)); t.cmd(a, "KEYS", kb(b"*")); t.cmd(b, "KEYS", kb(b"*"))
    t.cmd(a, "ROLLBACK" if i % 2 else "COMMIT"); t.cmd(b, "GET", kb(b"x")); t.cmd(a, "GET", kb(b"x"))
t = T("m11", "snapshot_isolation"); a = t.conn("bob"); b = t.conn("alice"); t.cmd(b, "SET", p_set(b"s", 1, 0, b"1")); t.cmd(a, "BEGIN"); t.cmd(a, "GET", kb(b"s")); t.cmd(b, "SET", p_set(b"s", 1, 0, b"2")); t.cmd(b, "SET", p_set(b"t", 1, 0, b"2")); t.cmd(a, "GET", kb(b"s")); t.cmd(a, "GET", kb(b"t")); t.cmd(a, "ROLLBACK"); t.cmd(a, "GET", kb(b"s"))
t = T("m11", "commit_counts"); a = t.conn("bob"); t.cmd(a, "SET", p_set(b"k", 1, 0, b"v")); t.cmd(a, "BEGIN"); t.cmd(a, "SET", p_set(b"k", 1, 0, b"v")); t.cmd(a, "GET", kb(b"zz")); t.cmd(a, "COMMIT"); t.cmd(a, "BEGIN"); t.cmd(a, "SET", p_set(b"k", 1, 0, b"w")); t.cmd(a, "DEL", kb(b"q")); t.cmd(a, "SET", p_set(b"m", 1, 0, b"w")); t.cmd(a, "COMMIT")
t = T("m11", "tx_set_del_set"); a = t.conn("bob"); t.cmd(a, "BEGIN"); t.cmd(a, "SET", p_set(b"k", 1, 0, b"v")); t.cmd(a, "DEL", kb(b"k")); t.cmd(a, "COMMIT"); t.cmd(a, "GET", kb(b"k"))
t = T("m11", "tx_ttl"); a = t.conn("bob"); b = t.conn("alice"); t.cmd(a, "BEGIN"); t.cmd(a, "SET", p_set(b"k", 1, 3, b"v")); t.cmd(a, "TTL", kb(b"k")); t.cmd(a, "EXPIRE", p_expire(b"k", 9)); t.cmd(a, "COMMIT"); t.cmd(b, "TTL", kb(b"k")); t.cmd(b, "TICK", struct.pack("<I", 9)); t.cmd(b, "GET", kb(b"k"))
t = T("m11", "tx_mget"); a = t.conn("bob"); t.cmd(a, "SET", p_set(b"a", 1, 0, b"1")); t.cmd(a, "BEGIN"); t.cmd(a, "DEL", kb(b"a")); t.cmd(a, "SET", p_set(b"b", 1, 0, b"2")); t.cmd(a, "MGET", p_mget([b"a", b"b"])); t.cmd(a, "ROLLBACK")
t = T("m11", "tx_info"); a = t.conn("bob"); t.cmd(a, "BEGIN"); t.cmd(a, "INFO"); t.cmd(a, "COMMIT"); t.cmd(a, "INFO")
t = T("m11", "tx_disconnect"); a = t.conn("bob"); b = t.conn("alice"); t.cmd(a, "BEGIN"); t.cmd(a, "SET", p_set(b"k", 1, 0, b"v")); t.close(a); t.cmd(b, "GET", kb(b"k"))
t = T("m11", "tx_wrongtype"); a = t.conn("bob"); t.cmd(a, "BEGIN"); t.cmd(a, "SET", p_set(b"k", 1, 0, b"v")); t.cmd(a, "INCR", p_incr(b"k", 1)); t.cmd(a, "COMMIT")

# ---------------- m12 conflicts and state errors ----------------
t = T("m12", "read_write_conflict"); a = t.conn("bob"); b = t.conn("alice"); t.cmd(b, "SET", p_set(b"x", 1, 0, b"1")); t.cmd(a, "BEGIN"); t.cmd(a, "GET", kb(b"x")); t.cmd(b, "SET", p_set(b"x", 1, 0, b"2")); t.cmd(a, "SET", p_set(b"y", 1, 0, b"1")); t.cmd(a, "COMMIT"); t.cmd(a, "GET", kb(b"y")); t.cmd(a, "COMMIT")
t = T("m12", "write_write_conflict"); a = t.conn("bob"); b = t.conn("alice"); t.cmd(a, "BEGIN"); t.cmd(a, "SET", p_set(b"x", 1, 0, b"1")); t.cmd(b, "SET", p_set(b"x", 1, 0, b"2")); t.cmd(a, "COMMIT"); t.cmd(b, "GET", kb(b"x"))
t = T("m12", "no_conflict_disjoint"); a = t.conn("bob"); b = t.conn("alice"); t.cmd(a, "BEGIN"); t.cmd(a, "SET", p_set(b"x", 1, 0, b"1")); t.cmd(b, "SET", p_set(b"y", 1, 0, b"2")); t.cmd(a, "COMMIT"); t.cmd(b, "KEYS", kb(b"*"))
t = T("m12", "aba_not_conflict"); a = t.conn("bob"); b = t.conn("alice"); t.cmd(b, "SET", p_set(b"x", 1, 0, b"1")); t.cmd(a, "BEGIN"); t.cmd(a, "GET", kb(b"x")); t.cmd(b, "SET", p_set(b"x", 1, 0, b"2")); t.cmd(b, "SET", p_set(b"x", 1, 0, b"1")); t.cmd(a, "COMMIT")
t = T("m12", "delete_conflict"); a = t.conn("bob"); b = t.conn("alice"); t.cmd(b, "SET", p_set(b"x", 1, 0, b"1")); t.cmd(a, "BEGIN"); t.cmd(a, "TTL", kb(b"x")); t.cmd(b, "DEL", kb(b"x")); t.cmd(a, "COMMIT")
t = T("m12", "absent_then_created"); a = t.conn("bob"); b = t.conn("alice"); t.cmd(a, "BEGIN"); t.cmd(a, "GET", kb(b"x")); t.cmd(b, "SET", p_set(b"x", 1, 0, b"2")); t.cmd(a, "COMMIT")
t = T("m12", "ttl_conflict"); a = t.conn("bob"); b = t.conn("alice"); t.cmd(b, "SET", p_set(b"x", 1, 0, b"1")); t.cmd(a, "BEGIN"); t.cmd(a, "MGET", p_mget([b"x"])); t.cmd(b, "EXPIRE", p_expire(b"x", 5)); t.cmd(a, "COMMIT")
t = T("m12", "double_begin"); a = t.conn("bob"); t.cmd(a, "BEGIN"); t.cmd(a, "BEGIN"); t.cmd(a, "COMMIT"); t.cmd(a, "COMMIT"); t.cmd(a, "ROLLBACK")
t = T("m12", "tick_in_tx"); a = t.conn("alice"); t.cmd(a, "BEGIN"); t.cmd(a, "TICK", struct.pack("<I", 1)); t.cmd(a, "ROLLBACK"); t.cmd(a, "TICK", struct.pack("<I", 1))
t = T("m12", "begin_args"); a = t.conn("bob"); t.cmd(a, "BEGIN", b"x"); t.cmd(a, "COMMIT", b"x"); t.cmd(a, "ROLLBACK", b"x"); t.cmd(a, "BEGIN"); t.cmd(a, "COMMIT", b"x"); t.cmd(a, "ROLLBACK", b"x"); t.cmd(a, "ROLLBACK")
t = T("m12", "two_tx_race"); a = t.conn("bob"); b = t.conn("alice"); t.cmd(a, "BEGIN"); t.cmd(b, "BEGIN"); t.cmd(a, "INCR", p_incr(b"c", 1)); t.cmd(b, "INCR", p_incr(b"c", 1)); t.cmd(a, "COMMIT"); t.cmd(b, "COMMIT"); t.cmd(a, "GET", kb(b"c"))
t = T("m12", "retry_after_conflict"); a = t.conn("bob"); b = t.conn("alice"); t.cmd(a, "BEGIN"); t.cmd(a, "INCR", p_incr(b"c", 1)); t.cmd(b, "INCR", p_incr(b"c", 10)); t.cmd(a, "COMMIT"); t.cmd(a, "BEGIN"); t.cmd(a, "INCR", p_incr(b"c", 1)); t.cmd(a, "COMMIT"); t.cmd(a, "GET", kb(b"c"))

# ---------------- m13 watch ----------------
t = T("m13", "watch_basic"); a = t.conn("bob"); b = t.conn("alice"); t.cmd(a, "WATCH", kb(b"k")); t.cmd(b, "SET", p_set(b"k", 1, 0, b"v")); t.cmd(b, "INCR", p_incr(b"k2", 1)); t.cmd(b, "DEL", kb(b"k")); t.cmd(b, "DEL", kb(b"k")); t.cmd(b, "SET", p_set(b"other", 1, 0, b"v"))
t = T("m13", "watch_prefix"); a = t.conn("bob"); b = t.conn("alice"); t.cmd(a, "WATCH", kb(b"user:*")); [t.cmd(b, "SET", p_set(k, 1, 0, b"v")) for k in (b"user:1", b"user:", b"usr:1", b"user", b"user:long:key")]; t.cmd(b, "INCR", p_incr(b"user:9", 3))
t = T("m13", "watch_self"); a = t.conn("bob"); b = t.conn("alice"); t.cmd(a, "WATCH", kb(b"*")); t.cmd(a, "SET", p_set(b"k", 1, 0, b"v")); t.cmd(b, "SET", p_set(b"k", 1, 0, b"w")); t.cmd(a, "PING")
t = T("m13", "unwatch"); a = t.conn("bob"); b = t.conn("alice"); t.cmd(a, "WATCH", kb(b"k")); t.cmd(b, "SET", p_set(b"k", 1, 0, b"v")); t.cmd(a, "UNWATCH", kb(b"k")); t.cmd(b, "SET", p_set(b"k", 1, 0, b"w")); t.cmd(a, "PING", b"q")
t = T("m13", "fanout_order"); c3 = None; a = t.conn("bob"); b = t.conn("dave"); c = t.conn("carol"); d = t.conn("alice")
for x in (c, a, b): t.cmd(x, "WATCH", kb(b"k*"))
t.cmd(d, "SET", p_set(b"k1", 1, 0, b"v")); t.cmd(d, "DEL", kb(b"k1"))
t = T("m13", "dup_watch"); a = t.conn("bob"); b = t.conn("alice"); t.cmd(a, "WATCH", kb(b"k")); t.cmd(a, "WATCH", kb(b"k")); t.cmd(a, "INFO"); t.cmd(b, "SET", p_set(b"k", 1, 0, b"v"))
t = T("m13", "overlap_watch"); a = t.conn("bob"); b = t.conn("alice"); t.cmd(a, "WATCH", kb(b"k")); t.cmd(a, "WATCH", kb(b"k*")); t.cmd(a, "WATCH", kb(b"*")); t.cmd(b, "SET", p_set(b"k", 1, 0, b"v"))
t = T("m13", "expire_event"); a = t.conn("bob"); b = t.conn("alice"); t.cmd(a, "WATCH", kb(b"k")); t.cmd(b, "SET", p_set(b"k", 1, 0, b"v")); t.cmd(b, "EXPIRE", p_expire(b"k", 3)); t.cmd(b, "EXPIRE", p_expire(b"nope", 3))
t = T("m13", "watch_before_auth"); a = t.conn(); b = t.conn("alice"); t.cmd(a, "WATCH", kb(b"k")); t.cmd(b, "SET", p_set(b"k", 1, 0, b"v"))
t = T("m13", "watch_disconnect"); a = t.conn("bob"); b = t.conn("alice"); t.cmd(a, "WATCH", kb(b"k")); t.close(a); t.cmd(b, "SET", p_set(b"k", 1, 0, b"v")); t.cmd(b, "PING")
t = T("m13", "watch_errors"); a = t.conn("bob"); t.cmd(a, "WATCH", b""); t.cmd(a, "WATCH", kb(b"k" * 33)); t.cmd(a, "WATCH", kb(b"k") + b"x"); t.cmd(a, "UNWATCH", kb(b"k"))
t = T("m13", "watch_incr_notice"); a = t.conn("bob"); b = t.conn("alice"); t.cmd(a, "WATCH", kb(b"c")); [t.cmd(b, "INCR", p_incr(b"c", i)) for i in range(4)]; t.cmd(b, "INCR", p_incr(b"c", 1))
t = T("m13", "watch_failed_ops"); a = t.conn("bob"); b = t.conn("alice"); t.cmd(a, "WATCH", kb(b"k")); t.cmd(b, "SET", p_set(b"k", 1, 0, b"v")); t.cmd(b, "INCR", p_incr(b"k", 1)); t.cmd(b, "SET", p_set(b"k", 3, 0, b"bad"))
t = T("m13", "watch_many_conns"); cs = [t.conn("bob") for _ in range(4)]; w = t.conn("alice")
for i, c in enumerate(cs): t.cmd(c, "WATCH", kb(b"p%d*" % (i % 2)))
for k in (b"p0a", b"p1b", b"p2c"): t.cmd(w, "SET", p_set(k, 1, 0, b"v"))
t = T("m13", "watch_pipelined_writer"); a = t.conn("bob"); b = t.conn("alice"); t.cmd(a, "WATCH", kb(b"*")); t.raw(b, b"".join(fr(3, 50 + i, p_set(sk(i), 1, 0, b"v")) for i in range(5)))
t = T("m13", "watch_ro_user"); a = t.conn("carol"); b = t.conn("dave"); t.cmd(a, "WATCH", kb(b"*")); t.cmd(b, "SET", p_set(b"z", 1, 0, b"1")); t.cmd(b, "TICK", struct.pack("<I", 1)); t.cmd(a, "PING")

# ---------------- m14 watch + tick + tx ----------------
t = T("m14", "expiry_events"); a = t.conn("bob"); b = t.conn("alice"); t.cmd(a, "WATCH", kb(b"*"))
for k in (b"zeta", b"alpha", b"mid"): t.cmd(b, "SET", p_set(k, 1, 2, b"v"))
t.cmd(b, "SET", p_set(b"keep", 1, 0, b"v")); t.cmd(b, "TICK", struct.pack("<I", 5)); t.cmd(a, "PING")
t = T("m14", "expiry_self_excluded"); a = t.conn("alice"); b = t.conn("bob"); t.cmd(a, "WATCH", kb(b"*")); t.cmd(b, "WATCH", kb(b"a*")); t.cmd(a, "SET", p_set(b"ab", 1, 1, b"v")); t.cmd(a, "TICK", struct.pack("<I", 1)); t.cmd(a, "INFO")
t = T("m14", "tx_commit_events"); a = t.conn("bob"); b = t.conn("alice"); t.cmd(b, "WATCH", kb(b"*")); t.cmd(a, "SET", p_set(b"old", 1, 0, b"v")); t.cmd(a, "BEGIN")
for k in (b"z", b"b", b"m"): t.cmd(a, "SET", p_set(k, 1, 0, b"v"))
t.cmd(a, "DEL", kb(b"old")); t.cmd(a, "GET", kb(b"readonly")); t.cmd(a, "COMMIT")
t = T("m14", "tx_rollback_silent"); a = t.conn("bob"); b = t.conn("alice"); t.cmd(b, "WATCH", kb(b"*")); t.cmd(a, "BEGIN"); t.cmd(a, "SET", p_set(b"k", 1, 0, b"v")); t.cmd(a, "ROLLBACK"); t.cmd(b, "PING")
t = T("m14", "tx_noop_commit"); a = t.conn("bob"); b = t.conn("alice"); t.cmd(b, "WATCH", kb(b"*")); t.cmd(a, "SET", p_set(b"k", 1, 0, b"v")); t.cmd(a, "BEGIN"); t.cmd(a, "SET", p_set(b"k", 1, 0, b"v")); t.cmd(a, "COMMIT"); t.cmd(b, "PING")
t = T("m14", "tx_conflict_silent"); a = t.conn("bob"); b = t.conn("alice"); c = t.conn("dave"); t.cmd(c, "WATCH", kb(b"*")); t.cmd(a, "BEGIN"); t.cmd(a, "GET", kb(b"k")); t.cmd(b, "SET", p_set(b"k", 1, 0, b"1")); t.cmd(a, "SET", p_set(b"k", 1, 0, b"2")); t.cmd(a, "COMMIT")
t = T("m14", "tx_incr_commit_event"); a = t.conn("bob"); b = t.conn("alice"); t.cmd(b, "WATCH", kb(b"c")); t.cmd(a, "BEGIN"); t.cmd(a, "INCR", p_incr(b"c", 1)); t.cmd(a, "INCR", p_incr(b"c", 1)); t.cmd(a, "COMMIT"); t.cmd(a, "INCR", p_incr(b"c", 1))
t = T("m14", "tick_multi_watchers"); a = t.conn("bob"); b = t.conn("carol"); c = t.conn("alice"); t.cmd(a, "WATCH", kb(b"t*")); t.cmd(b, "WATCH", kb(b"t1")); t.cmd(c, "SET", p_set(b"t1", 1, 1, b"v")); t.cmd(c, "SET", p_set(b"t2", 1, 2, b"v")); t.cmd(c, "TICK", struct.pack("<I", 1)); t.cmd(c, "TICK", struct.pack("<I", 1))
t = T("m14", "tx_delete_event"); a = t.conn("bob"); b = t.conn("alice"); t.cmd(a, "SET", p_set(b"k", 1, 0, b"v")); t.cmd(b, "WATCH", kb(b"k")); t.cmd(a, "BEGIN"); t.cmd(a, "DEL", kb(b"k")); t.cmd(a, "DEL", kb(b"nope")); t.cmd(a, "COMMIT")
t = T("m14", "tx_expire_change"); a = t.conn("bob"); b = t.conn("alice"); t.cmd(a, "SET", p_set(b"k", 1, 0, b"v")); t.cmd(b, "WATCH", kb(b"k")); t.cmd(a, "BEGIN"); t.cmd(a, "EXPIRE", p_expire(b"k", 5)); t.cmd(a, "COMMIT")
t = T("m14", "tx_new_deleted"); a = t.conn("bob"); b = t.conn("alice"); t.cmd(b, "WATCH", kb(b"*")); t.cmd(a, "BEGIN"); t.cmd(a, "SET", p_set(b"k", 1, 0, b"v")); t.cmd(a, "DEL", kb(b"k")); t.cmd(a, "COMMIT"); t.cmd(b, "PING")

# ---------------- m15 pipelining ----------------
for i in range(5):
    t = T("m15", f"pipe_session{i}"); a = t.conn(); u = Rn.choice(["alice", "bob"])
    nonce_h = R.auth_hash(R.USERS[u.encode()][0], t.nonce(a), u.encode())
    blob = fr(2, 1, bytes([len(u)]) + u.encode() + struct.pack("<Q", nonce_h))
    for j in range(Rn.randrange(8, 25)):
        k = sk(Rn.randrange(5))
        blob += Rn.choice([fr(3, 10 + j, p_set(k, 1, Rn.choice([0, 4]), b"v%d" % j)), fr(4, 10 + j, kb(k)), fr(6, 10 + j, p_incr(b"c", j)), fr(1, 10 + j, b"p"), fr(5, 10 + j, kb(k)), fr(13, 10 + j, kb(b"*")), fr(99, 10 + j)])
    t.raw(a, blob)
t = T("m15", "pipe_split_everywhere"); a = t.conn("bob"); blob = b"".join(fr(3, 20 + i, p_set(sk(i), 1, 0, b"vv")) + fr(4, 40 + i, kb(sk(i))) for i in range(4)); t.raw(a, blob, chunks=[3, 11, 1, 17, 25])
t = T("m15", "pipe_quit"); a = t.conn("bob"); t.raw(a, fr(1, 1, b"a") + fr(15, 2) + fr(1, 3, b"never")); 
t = T("m15", "pipe_err_then_ok"); a = t.conn("bob"); t.raw(a, fr(4, 1, kb(b"x")) + fr(200, 2) + fr(3, 3, b"\x01") + fr(1, 4, b"fine"))
t = T("m15", "pipe_bad_magic_mid"); a = t.conn("bob"); t.raw(a, fr(1, 1, b"a") + fr(1, 2, b"b", magic=b"ZZ") + fr(1, 3, b"c"))
t = T("m15", "pipe_oversize_mid"); a = t.conn("bob"); t.raw(a, fr(1, 1, b"a") + b"KV\x02\x01" + struct.pack("<IH", 2, 700) + fr(1, 3, b"c"))
t = T("m15", "pipe_two_conns"); a = t.conn("bob"); b = t.conn("alice"); t.cmd(a, "WATCH", kb(b"*")); t.raw(b, b"".join(fr(3, i, p_set(sk(i), 0, 0, vint(i))) + fr(5, 30 + i, kb(sk(i))) for i in range(4)))
t = T("m15", "pipe_tx"); a = t.conn("bob"); t.raw(a, fr(8, 1) + fr(3, 2, p_set(b"k", 1, 0, b"v")) + fr(4, 3, kb(b"k")) + fr(9, 4) + fr(4, 5, kb(b"k")))
t = T("m15", "pipe_auth_fail_mid"); a = t.conn(); t.raw(a, fr(4, 1, kb(b"k")) + fr(1, 2, b"ok") + fr(2, 3, b"\x01x" + bytes(8)) + fr(1, 4, b"z"))
t = T("m15", "pipe_max_frames"); a = t.conn("bob"); t.raw(a, b"".join(fr(1, i, bytes([i % 256]) * (i % 50)) for i in range(60)))
t = T("m15", "pipe_halfframe_at_end"); a = t.conn("bob"); t.raw(a, fr(1, 1, b"a") + fr(1, 2, b"hello")[:12]); t.raw(a, fr(1, 2, b"hello")[12:]); t.cmd(a, "PING", b"end")
t = T("m15", "pipe_ten_in_chunks"); a = t.conn("bob"); t.raw(a, b"".join(fr(1, i, b"x" * i) for i in range(10)), chunks=[10] * 12)
t = T("m15", "pipe_interleave_watch"); a = t.conn("bob"); b = t.conn("alice"); t.cmd(a, "WATCH", kb(b"k*")); t.raw(a, fr(1, 1, b"1") + fr(3, 2, p_set(b"k1", 1, 0, b"v")) + fr(1, 3, b"3")); t.cmd(b, "SET", p_set(b"k2", 1, 0, b"v")); t.raw(a, fr(1, 4, b"4"))
t = T("m15", "pipe_rid_echo"); a = t.conn("bob"); t.raw(a, b"".join(fr(op, rid, pl) for op, rid, pl in [(1, 0xFFFFFFFF, b""), (4, 0, kb(b"x")), (14, 12345678, b""), (200, 0x01020304, b"")]))

# ---------------- m16 malformed-input matrix ----------------
VALID = [(3, p_set(b"k", 1, 0, b"value")), (4, kb(b"k")), (5, kb(b"k")), (6, p_incr(b"c", 3)), (7, struct.pack("<I", 4)), (11, kb(b"k*")), (12, kb(b"k*")),
         (13, kb(b"*")), (16, p_mget([b"k", b"c"])), (17, p_expire(b"k", 3)), (18, kb(b"k")), (2, b"\x05alice" + bytes(8))]
for i in range(20):
    t = T("m16", f"fuzz{i}"); a = t.conn("alice"); t.cmd(a, "SET", p_set(b"k", 1, 0, b"seed")); 
    for j in range(3):
        op, pl = Rn.choice(VALID); pl = bytearray(pl)
        m = Rn.choice(["flip", "trunc", "extend", "zero", "op", "hdrlen", "empty"])
        rid = Rn.randrange(1, 2**32)
        if m == "flip" and pl: pl[Rn.randrange(len(pl))] ^= 1 << Rn.randrange(8)
        elif m == "trunc" and pl: pl = pl[:Rn.randrange(len(pl))]
        elif m == "extend": pl += bytes(Rn.randrange(256) for _ in range(Rn.randrange(1, 6)))
        elif m == "zero": pl = bytearray(len(pl))
        elif m == "empty": pl = bytearray()
        elif m == "op": op = Rn.randrange(0, 40)
        frame = fr(op, rid, bytes(pl))
        if m == "hdrlen":
            frame = frame[:8] + struct.pack("<H", max(len(pl) + Rn.randrange(1, 9), 0)) + frame[10:]
        t.raw(a, frame)
        t.cmd(a, "PING", b"sync%d" % j); 
    t.cmd(a, "KEYS", kb(b"*"))

# ---------------- record ----------------
def rec(args):
    idx, t = args
    return idx, t, record([REF], 31000 + idx, t.acts, t.seed)

out = []
with ThreadPoolExecutor(6) as ex:
    for idx, t, script in ex.map(rec, list(enumerate(TESTS))):
        out.append({"id": f"{t.ms}-{t.name}", "ms": t.ms, "script": script})
json.dump(out, open(os.path.join(HID, "cases.json"), "w"))
print("recorded", len(out), "tests")
from collections import Counter
print(sorted(Counter(t["ms"] for t in out).items()))

# public transcripts for the agent: hex dumps of annotated sessions
os.makedirs(PUB, exist_ok=True)
samples = [x for x in out if x["id"] in ("m01-ping0", "m03-login_alice", "m04-setget3", "m11-tx_basic0", "m13-watch_basic", "m12-read_write_conflict")]
with open(os.path.join(PUB, "transcripts.txt"), "w") as f:
    f.write("# Sample sessions recorded from the original server (kvd --seed S). One line per event.\n")
    f.write("# '>' conn bytes = client sends, '<' conn bytes = server sends (hex). Frames: 'KV' ver type reqid(u32le) len(u16le) payload.\n\n")
    for s in samples:
        f.write(f"## session {s['id']} seed={s['script']['seed']}\n")
        for st in s["script"]["steps"]:
            if st[0] == "connect": f.write(f"connect {st[1]}\n")
            elif st[0] == "send": f.write(f"> {st[1]} {st[2]}\n")
            elif st[0] == "expect": f.write(f"< {st[1]} {st[2]}\n")
            elif st[0] in ("eof", "close"): f.write(f"{st[0]} {st[1]}\n")
        f.write("\n")
