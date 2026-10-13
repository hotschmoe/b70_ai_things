"""Session-script recording / replay for the re-proto task (namespaced helper; stdlib only).

A script is {"seed": int, "steps": [step, ...]}; steps:
  ["connect", cid]               open a TCP connection to the server (retries until it accepts)
  ["send", cid, hex, chunks]     send bytes (chunks: list of sizes -> separate writes 40 ms apart, or null)
  ["expect", cid, hex]           the next len(bytes) received on cid must equal bytes
  ["eof", cid]                   the server must close its side (recv == b'') ; client then closes
  ["close", cid]                 client closes its socket
  ["quiet"]                      no connection may have unexpected data pending for 100 ms
Byte streams per connection are deterministic, so replay order across connections only needs causality.
"""
import json, os, select, socket, subprocess, sys, time


def _connect(port, deadline, proc=None):
    last = None
    while time.time() < deadline:
        s = socket.socket()
        try:
            s.connect(("127.0.0.1", port))
            s.setblocking(True)
            return s
        except OSError as e:
            last = e
            s.close()
            if proc is not None and proc.poll() is not None:
                raise RuntimeError(f"server exited rc={proc.returncode}")
            time.sleep(0.05)
    raise RuntimeError(f"connect failed: {last}")


def _read_exact(s, n, timeout):
    buf, end = b"", time.time() + timeout
    while len(buf) < n:
        left = end - time.time()
        if left <= 0:
            break
        r, _, _ = select.select([s], [], [], left)
        if not r:
            break
        try:
            d = s.recv(n - len(buf))
        except OSError:
            break
        if not d:
            break
        buf += d
    return buf


def start_server(cmd, port, seed, cwd=None):
    return subprocess.Popen(cmd + ["--port", str(port), "--seed", str(seed)], cwd=cwd, stdin=subprocess.DEVNULL,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)


def stop_server(p):
    try:
        p.kill()
    except OSError:
        pass
    try:
        p.wait(timeout=5)
    except Exception:
        pass


def replay(cmd, port, script, cwd=None, io_timeout=12.0):
    ok, why = _replay(cmd, port, script, cwd, io_timeout)
    if not ok and "server exited" in why and "connect" not in why[:0]:
        ok, why = _replay(cmd, port + 3000, script, cwd, io_timeout)   # early exit (e.g. port clash): retry once
    return ok, why


def _replay(cmd, port, script, cwd=None, io_timeout=12.0):
    """Replay a recorded script against the server command. Returns (ok, why)."""
    p = start_server(cmd, port, script["seed"], cwd)
    socks = {}
    try:
        for st in script["steps"]:
            k = st[0]
            if k == "connect":
                socks[st[1]] = _connect(port, time.time() + 20, p)
            elif k == "send":
                data = bytes.fromhex(st[2])
                chunks = st[3]
                if chunks:
                    o = 0
                    for ln in chunks:
                        socks[st[1]].sendall(data[o:o + ln]); o += ln
                        time.sleep(0.04)
                    if o < len(data):
                        socks[st[1]].sendall(data[o:])
                else:
                    socks[st[1]].sendall(data)
            elif k == "expect":
                want = bytes.fromhex(st[2])
                got = _read_exact(socks[st[1]], len(want), io_timeout)
                if got != want:
                    return False, f"conn {st[1]}: expected {want.hex()} got {got.hex()}"
            elif k == "eof":
                s = socks[st[1]]
                r, _, _ = select.select([s], [], [], io_timeout)
                if not r:
                    return False, f"conn {st[1]}: no EOF"
                try:
                    d = s.recv(4096)
                except OSError:
                    d = b""
                if d:
                    return False, f"conn {st[1]}: data instead of EOF {d.hex()}"
                s.close(); del socks[st[1]]
            elif k == "close":
                socks[st[1]].close(); del socks[st[1]]
            elif k == "quiet":
                if socks:
                    r, _, _ = select.select(list(socks.values()), [], [], 0.1)
                    if r:
                        try:
                            d = r[0].recv(4096)
                        except OSError:
                            d = b"<err>"
                        return False, f"unexpected data/EOF at end: {d.hex()}"
        return True, ""
    except Exception as e:  # server died, connection reset, ...
        return False, repr(e)[:200]
    finally:
        for s in socks.values():
            try: s.close()
            except OSError: pass
        stop_server(p)


def record(cmd, port, actions, seed, cwd=None, settle=0.12):
    """Execute actions against the REFERENCE and produce a script with expect/eof/quiet steps.
    actions: [("connect", cid) | ("send", cid, bytes, chunks|None) | ("close", cid)]"""
    p = start_server(cmd, port, seed, cwd)
    socks, steps = {}, []

    def collect():
        # read everything that arrives until a quiet window; one expect step per connection that produced data
        buf = {cid: b"" for cid in socks}
        eof = set()
        while True:
            live = [s for cid, s in socks.items() if cid not in eof]
            if not live:
                break
            r, _, _ = select.select(live, [], [], settle)
            if not r:
                break
            for s in r:
                cid = [c for c, x in socks.items() if x is s][0]
                try:
                    d = s.recv(65536)
                except OSError:
                    d = b""
                if not d:
                    eof.add(cid)
                else:
                    buf[cid] += d
        for cid in sorted(buf):
            if buf[cid]:
                steps.append(["expect", cid, buf[cid].hex()])
        for cid in sorted(eof):
            steps.append(["eof", cid]); socks[cid].close(); del socks[cid]
    try:
        for a in actions:
            if a[0] in ("send", "close") and a[1] not in socks:
                continue          # connection already closed by the server (EOF consumed): action dropped
            if a[0] == "connect":
                socks[a[1]] = _connect(port, time.time() + 20)
                steps.append(["connect", a[1]])
            elif a[0] == "send":
                steps.append(["send", a[1], a[2].hex(), a[3]])
                if a[3]:
                    o = 0
                    for ln in a[3]:
                        socks[a[1]].sendall(a[2][o:o + ln]); o += ln
                        time.sleep(0.04)
                    if o < len(a[2]):
                        socks[a[1]].sendall(a[2][o:])
                else:
                    socks[a[1]].sendall(a[2])
            elif a[0] == "close":
                socks[a[1]].close(); del socks[a[1]]
                steps.append(["close", a[1]])
                time.sleep(0.05)
                continue
            collect()
        steps.append(["quiet"])
        return {"seed": seed, "steps": steps}
    finally:
        for s in socks.values():
            try: s.close()
            except OSError: pass
        stop_server(p)


# ---- wire builders ----
import struct

def fr(op, rid=1, pay=b"", ver=2, magic=b"KV"):
    return magic + bytes([ver, op]) + struct.pack("<IH", rid, len(pay)) + pay

def vint(i): return struct.pack("<q", i)
def vlist(items): return bytes([len(items)]) + b"".join(bytes([len(x)]) + x for x in items)
def kb(k): return bytes([len(k)]) + k
def p_set(k, t, ttl, v): return kb(k) + bytes([t]) + struct.pack("<I", ttl) + v
def p_incr(k, d): return kb(k) + struct.pack("<q", d)
def p_expire(k, ttl): return kb(k) + struct.pack("<I", ttl)
def p_mget(ks): return bytes([len(ks)]) + b"".join(kb(k) for k in ks)
OP = dict(PING=1, AUTH=2, SET=3, GET=4, DEL=5, INCR=6, TICK=7, BEGIN=8, COMMIT=9, ROLLBACK=10, WATCH=11,
          UNWATCH=12, KEYS=13, INFO=14, QUIT=15, MGET=16, EXPIRE=17, TTL=18)
