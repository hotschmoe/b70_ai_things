#!/usr/bin/env python3
"""Reference Python implementation of kvd (used as ref_sub for validation; hidden from the agent)."""
import select, socket, struct, sys

MAXPAY, MAXKEYS = 512, 64
EMSG = ["", "bad frame", "unsupported version", "authentication required", "not found", "wrong type",
        "transaction conflict", "payload too large", "invalid state", "unknown operation", "bad argument",
        "permission denied", "store full", "authentication failed"]
(E_BADFRAME, E_BADVER, E_NOAUTH, E_NOTFOUND, E_WRONGTYPE, E_CONFLICT, E_TOOBIG, E_STATE, E_UNKNOWN, E_BADARG,
 E_PERM, E_FULL, E_AUTHFAIL) = range(1, 14)
USERS = {b"alice": (b"wonderland", 3), b"bob": (b"builder99", 2), b"carol": (b"readonly!", 1), b"dave": (b"d4v3-r0cks", 2)}
M64 = (1 << 64) - 1


def splitmix(z):
    z = (z + 0x9E3779B97F4A7C15) & M64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & M64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & M64
    return z ^ (z >> 31)


def rotl(x, r):
    return ((x << r) | (x >> (64 - r))) & M64


def auth_hash(secret, nonce, user):
    h = 0xcbf29ce484222325
    for b in secret:
        h = ((h ^ b) * 0x100000001b3) & M64
    for b in nonce:
        h = rotl(((h ^ b) * 0x100000001b3) & M64, 5)
    for b in user:
        h = ((h ^ b) * 0x100000001b3) & M64
    h ^= h >> 29
    h = (h * 0xBF58476D1CE4E5B9) & M64
    h ^= h >> 32
    return h


class Ent:
    __slots__ = ("type", "val", "exp")

    def __init__(self, t, v, e):
        self.type, self.val, self.exp = t, v, e

    def key(self):
        return (self.type, self.val, self.exp)


class Conn:
    def __init__(self, sock, cid):
        self.s, self.id, self.authed, self.role, self.fails = sock, cid, False, 0, 0
        self.closing, self.intx, self.watch, self.inbuf = False, False, [], b""
        self.base, self.work, self.touched = None, None, []


class Server:
    def __init__(self, port, seed):
        self.seed, self.clock, self.G, self.conns, self.accepted = seed, 0, {}, {}, 0
        self.ls = socket.socket()
        self.ls.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.ls.bind(("127.0.0.1", port))
        self.ls.listen(16)

    def nonce(self, cid):
        return struct.pack("<Q", splitmix((self.seed + cid * 0x9E3779B97F4A7C15) & M64))

    def send(self, c, typ, rid, pay=b""):
        try:
            c.s.sendall(b"KV" + bytes([2, typ]) + struct.pack("<IH", rid, len(pay)) + pay)
        except OSError:
            c.closing = True

    def err(self, c, rid, code):
        self.send(c, 0xFF, rid, bytes([code]) + EMSG[code].encode())

    def notify(self, actor, ev, key):
        for cid in sorted(self.conns):
            c = self.conns[cid]
            if c is actor or c.closing:
                continue
            for pat in c.watch:
                if (key.startswith(pat[:-1]) if pat.endswith(b"*") else key == pat):
                    self.send(c, 0xA0, 0, bytes([ev, len(key)]) + key)
                    break

    @staticmethod
    def match(pat, key):
        return key.startswith(pat[:-1]) if pat.endswith(b"*") else key == pat

    def record(self, e):
        ttl = (e.exp - self.clock) if e.exp else 0
        return bytes([e.type]) + struct.pack("<IH", ttl, len(e.val)) + e.val

    @staticmethod
    def check_val(t, v):
        if t == 0:
            return len(v) == 8
        if t == 1:
            return len(v) <= 400
        if t == 2:
            if len(v) < 1:
                return False
            cnt, o = v[0], 1
            if cnt > 16:
                return False
            for _ in range(cnt):
                if o >= len(v):
                    return False
                o += 1 + v[o]
                if o > len(v):
                    return False
            return o == len(v) and o <= 400
        return False

    def handle(self, c, op, rid, pay):
        class Bad(Exception):
            pass
        pos = [0]

        def take(w):
            if pos[0] + w > len(pay):
                raise Bad()
            r = pay[pos[0]:pos[0] + w]
            pos[0] += w
            return int.from_bytes(r, "little") if w in (1, 2, 4, 8) else r

        def takeb(w):
            if pos[0] + w > len(pay):
                raise Bad()
            r = pay[pos[0]:pos[0] + w]
            pos[0] += w
            return r

        def rest():
            return len(pay) - pos[0]

        if not c.authed and op not in (1, 2):
            return self.err(c, rid, E_NOAUTH)
        if op < 1 or op > 18:
            return self.err(c, rid, E_UNKNOWN)
        if c.authed:
            wr = op in (3, 5, 6, 17)
            if c.role == 1 and (wr or op == 7):
                return self.err(c, rid, E_PERM)
            if c.role == 2 and op == 7:
                return self.err(c, rid, E_PERM)
        S = c.work if c.intx else self.G

        def touch(k):
            if k not in c.touched:
                c.touched.append(k)
        try:
            if op == 1:
                return self.send(c, 0x81, rid, pay)
            if op == 2:
                if c.authed:
                    return self.err(c, rid, E_STATE)
                ul = take(1); u = takeb(ul); h = take(8)
                if rest():
                    raise Bad()
                nonce = self.nonce(c.id)
                us = USERS.get(u)
                if us is None or auth_hash(us[0], nonce, u) != h:
                    self.err(c, rid, E_AUTHFAIL)
                    c.fails += 1
                    if c.fails >= 3:
                        c.closing = True
                    return
                c.authed, c.role = True, us[1]
                return self.send(c, 0x82, rid, bytes([c.role]))
            if op == 3:
                kl = take(1); k = takeb(kl); t = take(1); ttl = take(4)
                if kl < 1 or kl > 32 or b"\x00" in k:
                    raise Bad()
                v = pay[pos[0]:]
                if not self.check_val(t, v):
                    raise Bad()
                if k not in S and len(S) >= MAXKEYS:
                    return self.err(c, rid, E_FULL)
                S[k] = Ent(t, v, self.clock + ttl if ttl else 0)
                if c.intx:
                    touch(k)
                else:
                    self.notify(c, 1, k)
                return self.send(c, 0x83, rid)
            if op in (4, 5, 18):
                kl = take(1); k = takeb(kl)
                if rest() or kl < 1 or b"\x00" in k:
                    raise Bad()
                if op == 4:
                    if c.intx: touch(k)
                    e = S.get(k)
                    if e is None:
                        return self.err(c, rid, E_NOTFOUND)
                    return self.send(c, 0x84, rid, self.record(e))
                if op == 5:
                    e = S.get(k)
                    if c.intx: touch(k)
                    if e is not None:
                        del S[k]
                        if not c.intx:
                            self.notify(c, 2, k)
                    return self.send(c, 0x85, rid, bytes([1 if e is not None else 0]))
                if c.intx: touch(k)
                e = S.get(k)
                if e is None:
                    return self.err(c, rid, E_NOTFOUND)
                t = (e.exp - self.clock) if e.exp else 0xFFFFFFFF
                return self.send(c, 0x92, rid, struct.pack("<I", t))
            if op == 6:
                kl = take(1); k = takeb(kl); d = take(8)
                if rest() or kl < 1 or kl > 32 or b"\x00" in k:
                    raise Bad()
                e = S.get(k)
                if e is not None and e.type != 0:
                    return self.err(c, rid, E_WRONGTYPE)
                if e is None:
                    if len(S) >= MAXKEYS:
                        return self.err(c, rid, E_FULL)
                    e = S[k] = Ent(0, bytes(8), 0)
                cur = (int.from_bytes(e.val, "little") + d) & M64
                e.val = struct.pack("<Q", cur)
                if c.intx: touch(k)
                else: self.notify(c, 4, k)
                return self.send(c, 0x86, rid, e.val)
            if op == 7:
                t = take(4)
                if rest() or t == 0 or t > 100000:
                    raise Bad()
                if c.intx:
                    return self.err(c, rid, E_STATE)
                self.clock += t
                dead = sorted(k for k, e in self.G.items() if e.exp and e.exp <= self.clock)
                for k in dead:
                    del self.G[k]
                for k in dead:
                    self.notify(c, 3, k)
                return self.send(c, 0x87, rid, struct.pack("<QH", self.clock, len(dead)))
            if op == 8:
                if pay: raise Bad()
                if c.intx: return self.err(c, rid, E_STATE)
                c.intx = True
                c.base = {k: Ent(e.type, e.val, e.exp) for k, e in self.G.items()}
                c.work = {k: Ent(e.type, e.val, e.exp) for k, e in self.G.items()}
                c.touched = []
                return self.send(c, 0x88, rid)
            if op == 9:
                if pay: raise Bad()
                if not c.intx: return self.err(c, rid, E_STATE)
                c.intx = False
                for k in c.touched:
                    b, g = c.base.get(k), self.G.get(k)
                    if (b.key() if b else None) != (g.key() if g else None):
                        return self.err(c, rid, E_CONFLICT)
                applied = 0
                for k in sorted(c.touched):
                    b, w = c.base.get(k), c.work.get(k)
                    if (b.key() if b else None) == (w.key() if w else None):
                        continue
                    if w is not None:
                        self.G[k] = Ent(w.type, w.val, w.exp)
                        self.notify(c, 1, k)
                    elif k in self.G:
                        del self.G[k]
                        self.notify(c, 2, k)
                    applied += 1
                return self.send(c, 0x89, rid, struct.pack("<H", applied))
            if op == 10:
                if pay: raise Bad()
                if not c.intx: return self.err(c, rid, E_STATE)
                c.intx = False
                return self.send(c, 0x8A, rid)
            if op in (11, 12):
                kl = take(1); k = takeb(kl)
                if rest() or kl < 1 or kl > 32 or b"\x00" in k:
                    raise Bad()
                if op == 11:
                    if k not in c.watch:
                        if len(c.watch) >= 8:
                            return self.err(c, rid, E_FULL)
                        c.watch.append(k)
                else:
                    if k not in c.watch:
                        return self.err(c, rid, E_NOTFOUND)
                    c.watch.remove(k)
                return self.send(c, 0x80 | op, rid)
            if op == 13:
                kl = take(1); k = takeb(kl)
                if rest() or kl > 32 or b"\x00" in k:
                    raise Bad()
                ks = sorted(x for x in S if kl == 0 or self.match(k, x))
                out = struct.pack("<H", len(ks)) + b"".join(bytes([len(x)]) + x for x in ks)
                return self.send(c, 0x8D, rid, out)
            if op == 14:
                if pay: raise Bad()
                t = "clock=%d keys=%d accepted=%d watches=%d tx=%d\n" % (self.clock, len(S), self.accepted, len(c.watch), 1 if c.intx else 0)
                return self.send(c, 0x8E, rid, t.encode())
            if op == 15:
                self.send(c, 0x8F, rid)
                c.closing = True
                return
            if op == 16:
                cnt = take(1)
                if cnt < 1 or cnt > 8:
                    raise Bad()
                out = bytes([cnt])
                for _ in range(cnt):
                    kl = take(1); k = takeb(kl)
                    if kl < 1 or b"\x00" in k:
                        raise Bad()
                    if c.intx: touch(k)
                    e = S.get(k)
                    out += b"\x00" if e is None else b"\x01" + self.record(e)
                    if len(out) > MAXPAY:
                        return self.err(c, rid, E_TOOBIG)
                if rest():
                    raise Bad()
                return self.send(c, 0x90, rid, out)
            if op == 17:
                kl = take(1); k = takeb(kl); ttl = take(4)
                if rest() or kl < 1 or b"\x00" in k:
                    raise Bad()
                if c.intx: touch(k)
                e = S.get(k)
                if e is not None:
                    e.exp = self.clock + ttl if ttl else 0
                    if not c.intx:
                        self.notify(c, 1, k)
                return self.send(c, 0x91, rid, bytes([1 if e is not None else 0]))
        except Bad:
            return self.err(c, rid, E_BADARG)

    def drain(self, c):
        while len(c.inbuf) >= 10 and not c.closing:
            h = c.inbuf
            rid = struct.unpack("<I", h[4:8])[0]
            ln = h[8] | (h[9] << 8)
            if h[0:2] != b"KV":
                self.err(c, 0, E_BADFRAME); c.closing = True; return
            if h[2] != 2:
                self.err(c, rid, E_BADVER); c.closing = True; return
            if ln > MAXPAY:
                self.err(c, rid, E_TOOBIG); c.closing = True; return
            if len(h) < 10 + ln:
                return
            op, pay = h[3], h[10:10 + ln]
            c.inbuf = h[10 + ln:]
            self.handle(c, op, rid, pay)

    def run(self):
        p = select.poll()
        fds = {self.ls.fileno(): None}
        p.register(self.ls, select.POLLIN)
        while True:
            for fd, ev in p.poll():
                if fd == self.ls.fileno():
                    s, _ = self.ls.accept()
                    self.accepted += 1
                    c = Conn(s, self.accepted)
                    self.conns[c.id] = c
                    fds[s.fileno()] = c
                    p.register(s, select.POLLIN)
                    self.send(c, 0x80, 0, self.nonce(c.id) + b"\x05kvd/2")
                    continue
                c = fds[fd]
                try:
                    d = c.s.recv(4096)
                except OSError:
                    d = b""
                if not d:
                    p.unregister(fd); c.s.close(); del self.conns[c.id]; del fds[fd]
                    continue
                if c.closing:
                    continue
                c.inbuf += d
                self.drain(c)
                if c.closing:
                    try: c.s.shutdown(socket.SHUT_WR)
                    except OSError: pass
                    c.inbuf = b""


if __name__ == "__main__":
    port, seed = 0, 1
    a = sys.argv[1:]
    for i in range(0, len(a) - 1, 2):
        if a[i] == "--port": port = int(a[i + 1])
        elif a[i] == "--seed": seed = int(a[i + 1])
    Server(port, seed).run()
