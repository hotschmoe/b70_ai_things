#!/usr/bin/env python3
"""Reference reimplementation of the re-bundle tool (validation solution)."""
import json, math, os, re, struct, sys

M32 = 0xFFFFFFFF
JSWS = "\t\n\x0b\x0c\r                  　﻿"
WSC = "[" + re.escape(JSWS) + "]"


class Exit(BaseException):
    def __init__(self, code): self.code = code


def fail(msg, code=1):
    sys.stderr.buffer.write((msg + "\n").encode("utf-8", "surrogateescape")); sys.stderr.buffer.flush()
    raise Exit(code)


def out(b):
    if isinstance(b, str): b = b.encode("utf-8", "surrogateescape")
    sys.stdout.buffer.write(b); sys.stdout.buffer.flush()


def js_trim(s): return s.strip(JSWS)
def i32(x): x &= M32; return x - (1 << 32) if x & 0x80000000 else x
def len16(s): return len(s.encode("utf-16-le", "surrogatepass")) // 2


def read_input(arg, who):
    try:
        if arg is None or arg == "-": return sys.stdin.buffer.read()
        with open(arg, "rb") as f: return f.read()
    except OSError:
        if arg is None or arg == "-": return b""
        fail(who + ": cannot read " + arg, 2)


def imul(a, b): return i32((a & M32) * (b & M32))


def crc_of(b):
    h = 0x811c9dc5
    for x in b:
        h = ((h ^ x) * 0x01000193) & M32
        h = ((h << 5) | (h >> 27)) & M32
    return (h ^ len(b)) & M32


def adler_of(b):
    a, c = 1, 0
    for x in b:
        a = (a + x) % 65521; c = (c + a) % 65521
    return ((c << 16) | a) & M32


def hex8(n): return "%08x" % (n & M32)


def cmd_crc(args):
    if args and args[0] == "-s":
        if len(args) < 2: fail("usage: tool crc [-s STRING | FILE]", 2)
        data = args[1].encode("utf-8", "surrogateescape")
    else: data = read_input(args[0] if args else None, "crc")
    out("crc=%s adler=%s len=%d\n" % (hex8(crc_of(data)), hex8(adler_of(data)), len(data)))


ALPHA = "0123456789BCDFGHJKLMNPQRSTVWXYZ_"


def cmd_b32(args):
    mode = args[0] if args else None
    if mode not in ("enc", "dec"): fail("usage: tool b32 enc|dec [FILE]", 2)
    data = read_input(args[1] if len(args) > 1 else None, "b32")
    if mode == "enc":
        acc = bits = 0; s = []; total = 0
        for x in data:
            acc = ((acc << 8) | x) & 0xffff; bits += 8
            while bits >= 5:
                v = (acc >> (bits - 5)) & 31; s.append(ALPHA[v]); total += v; bits -= 5
        if bits > 0:
            v = (acc << (5 - bits)) & 31; s.append(ALPHA[v]); total += v
        s = "".join(s)
        groups = [s[i:i + 5] for i in range(0, len(s), 5)]
        out("-".join(groups) + "~" + ALPHA[(total + len(data)) % 32] + "\n"); return
    text = js_trim(data.decode("utf-8", "replace"))
    t = text.find("~")
    if t < 0 or text.find("~", t + 1) >= 0: fail("b32: missing checksum")
    body, chk = text[:t], text[t + 1:]
    vals = []
    for i, c in enumerate(body):
        if c == "-": continue
        v = ALPHA.find(c.upper())
        if v < 0 or len(c.upper()) != 1: fail("b32: invalid character '%s' at %d" % (c, i))
        vals.append(v)
    nbytes = (len(vals) * 5) // 8; extra = len(vals) * 5 - nbytes * 8
    if extra >= 5: fail("b32: bad length")
    res = bytearray(nbytes); acc = bits = o = total = 0
    for v in vals:
        total += v; acc = ((acc << 5) | v) & 0xffff; bits += 5
        if bits >= 8:
            if o < nbytes: res[o] = (acc >> (bits - 8)) & 255; o += 1
            bits -= 8
    if bits > 0 and (acc & ((1 << bits) - 1)) != 0: fail("b32: nonzero padding")
    if len(chk) != 1 or ALPHA[(total + nbytes) % 32] != chk.upper(): fail("b32: bad checksum")
    out(bytes(res))


def sort_key16(k): return k.encode("utf-16-be", "surrogatepass")


def ser_str(s, parts):
    b = s.encode("utf-8", "replace"); parts.append(b"s%d:" % len(b)); parts.append(b)


def ser_val(v, parts):
    if v is None: parts.append(b"n")
    elif v is True: parts.append(b"t")
    elif v is False: parts.append(b"f")
    elif isinstance(v, (int, float)):
        if isinstance(v, float):
            if not (math.isfinite(v) and v == int(v)): fail("ser: unsupported number")
            v = int(v)
        if abs(v) > 2**53 - 1: fail("ser: unsupported number")
        parts.append(b"i%d;" % v)
    elif isinstance(v, str): ser_str(v, parts)
    elif isinstance(v, list):
        parts.append(b"[")
        for x in v: ser_val(x, parts)
        parts.append(b"]")
    else:
        parts.append(b"{")
        for k in sorted(v.keys(), key=sort_key16):
            ser_str(k, parts); ser_val(v[k], parts)
        parts.append(b"}")


def _bad_const(x): raise ValueError(x)


def json_loads(text):
    return json.loads(text, parse_constant=_bad_const)


def cmd_ser(args):
    text = read_input(args[0] if args else None, "ser").decode("utf-8", "replace")
    try: v = json_loads(text)
    except Exception: fail("ser: invalid JSON")
    parts = []; ser_val(v, parts)
    body = b"".join(parts)
    out(body + b"#" + hex8(crc_of(body)).encode() + b"\n")


def is_index_key(k):
    return re.fullmatch(r"0|[1-9][0-9]*", k) is not None and int(k) < 2**32 - 1


def js_order(d):
    ints = sorted((k for k in d if is_index_key(k)), key=int)
    return ints + [k for k in d if not is_index_key(k)]


def stringify(v):
    if v is None: return "null"
    if v is True: return "true"
    if v is False: return "false"
    if isinstance(v, int): return str(v)
    if isinstance(v, str): return json.dumps(v, ensure_ascii=False)
    if isinstance(v, list): return "[" + ",".join(stringify(x) for x in v) + "]"
    return "{" + ",".join(json.dumps(k, ensure_ascii=False) + ":" + stringify(v[k]) for k in js_order(v)) + "}"


def cmd_de(args):
    raw = read_input(args[0] if args else None, "de")
    end = len(raw)
    while end > 0 and raw[end - 1] in (10, 13, 32): end -= 1
    data = raw[:end]
    hp = data.rfind(b"#")
    if hp < 0: fail("de: missing checksum")
    field = data[hp + 1:].decode("latin-1")
    if not re.fullmatch(r"[0-9a-f]{8}", field): fail("de: bad checksum field")
    body = data[:hp]
    if hex8(crc_of(body)) != field: fail("de: checksum mismatch")
    st = {"p": 0}

    def bad(msg): fail("de: %s at %d" % (msg, st["p"]))

    def unexpected():
        if st["p"] >= len(body): bad("unexpected end")
        fail("de: unexpected '%s' at %d" % (chr(body[st["p"]]), st["p"]))

    def parse_str():
        st["p"] += 1; s0 = st["p"]
        while st["p"] < len(body) and 48 <= body[st["p"]] <= 57: st["p"] += 1
        if st["p"] == s0 or st["p"] >= len(body) or body[st["p"]] != 58: st["p"] = s0; bad("bad string length")
        n = int(body[s0:st["p"]]); st["p"] += 1
        if st["p"] + n > len(body): bad("string overflows")
        s = body[st["p"]:st["p"] + n].decode("utf-8", "replace"); st["p"] += n
        return s

    def parse(depth):
        if depth > 20: fail("de: nesting too deep")
        if st["p"] >= len(body): unexpected()
        c = chr(body[st["p"]])
        if c == "n": st["p"] += 1; return None
        if c == "t": st["p"] += 1; return True
        if c == "f": st["p"] += 1; return False
        if c == "i":
            st["p"] += 1; s0 = st["p"]
            if st["p"] < len(body) and body[st["p"]] == 45: st["p"] += 1
            ds = st["p"]
            while st["p"] < len(body) and 48 <= body[st["p"]] <= 57: st["p"] += 1
            if st["p"] == ds or st["p"] >= len(body) or body[st["p"]] != 59: st["p"] = s0; bad("bad integer")
            if st["p"] - ds > 15: st["p"] = s0; bad("integer too large")
            n = int(body[s0:st["p"]]); st["p"] += 1
            return n
        if c == "s": return parse_str()
        if c == "[":
            st["p"] += 1; arr = []
            while True:
                if st["p"] >= len(body): unexpected()
                if body[st["p"]] == 93: st["p"] += 1; return arr
                arr.append(parse(depth + 1))
        if c == "{":
            st["p"] += 1; obj = {}
            while True:
                if st["p"] >= len(body): unexpected()
                if body[st["p"]] == 125: st["p"] += 1; return obj
                if body[st["p"]] != 115: unexpected()
                k = parse_str(); obj[k] = parse(depth + 1)
        unexpected()

    v = parse(0)
    if st["p"] != len(body): bad("trailing data")
    out(stringify(v) + "\n")


# ------------------------------------------------------------------ expr
FUN = {"min": (1, 99), "max": (1, 99), "abs": (1, 1), "clamp": (3, 3), "pow": (2, 2), "sign": (1, 1)}


def expr_eval(src, vars_):
    S = {"pos": 0, "tok": None}

    def lex():
        while S["pos"] < len(src) and src[S["pos"]] in " \t": S["pos"] += 1
        pos = S["pos"]; st = pos
        if pos >= len(src): return {"t": "end", "s": "", "p": st}
        c = src[pos]
        if "0" <= c <= "9":
            m = re.match(r"0[xX][0-9a-fA-F]+|[0-9]+", src[pos:])
            S["pos"] += len(m.group(0))
            if len(m.group(0)) > 15: fail("expr: number too large at %d" % st)
            return {"t": "num", "v": i32(int(float(int(m.group(0), 0 if m.group(0)[:2].lower() == "0x" else 10))) ), "p": st}
        if c.isascii() and (c.isalpha() or c == "_"):
            m = re.match(r"[A-Za-z_][A-Za-z0-9_]*", src[pos:]); S["pos"] += len(m.group(0))
            return {"t": "id", "s": m.group(0), "p": st}
        two = src[pos:pos + 2]
        for op in ("<<", ">>", "<=", ">=", "==", "!=", "&&", "||"):
            if two == op: S["pos"] += 2; return {"t": "op", "s": op, "p": st}
        if c in "+-*/%&|^~!<>?:(),": S["pos"] += 1; return {"t": "op", "s": c, "p": st}
        fail("expr: syntax error at %d" % st)

    def nxt(): S["tok"] = lex()
    def syn(): fail("expr: syntax error at %d" % S["tok"]["p"])
    def is_op(s): return S["tok"]["t"] == "op" and S["tok"]["s"] == s

    def ternary():
        c = orx()
        if is_op("?"):
            nxt(); a = ternary()
            if not is_op(":"): syn()
            nxt(); b = ternary()
            return lambda: a() if c() != 0 else b()
        return c

    def orx():
        l = andx()
        while is_op("||"):
            nxt(); a = l; r = andx(); l = (lambda a, r: lambda: 1 if (a() != 0 or r() != 0) else 0)(a, r)
        return l

    def andx():
        l = bor()
        while is_op("&&"):
            nxt(); a = l; r = bor(); l = (lambda a, r: lambda: 1 if (a() != 0 and r() != 0) else 0)(a, r)
        return l

    def binlevel(sub, ops, f):
        def g():
            l = sub()
            while S["tok"]["t"] == "op" and S["tok"]["s"] in ops:
                op = S["tok"]["s"]; nxt(); a = l; r = sub()
                l = (lambda op, a, r: lambda: f(op, a(), r()))(op, a, r)
            return l
        return g

    def f_mul(op, a, b):
        if op == "*": return imul(a, b)
        if b == 0: fail("expr: division by zero")
        if op == "/":
            q = abs(a) // abs(b)
            return i32(-q if (a < 0) != (b < 0) else q)
        r = abs(a) % abs(b)
        return i32(-r if a < 0 else r)

    mul = lambda: binlevel(unary, ("*", "/", "%"), f_mul)()
    add = lambda: binlevel(mul, ("+", "-"), lambda op, a, b: i32(a + b) if op == "+" else i32(a - b))()
    shf = lambda: binlevel(add, ("<<", ">>"), lambda op, a, b: i32(a << (b & 31)) if op == "<<" else a >> (b & 31))()
    rel = lambda: binlevel(shf, ("<", "<=", ">", ">="), lambda op, a, b: 1 if (a < b if op == "<" else a <= b if op == "<=" else a > b if op == ">" else a >= b) else 0)()
    eq = lambda: binlevel(rel, ("==", "!="), lambda op, a, b: 1 if ((op == "==") == (a == b)) else 0)()
    band = lambda: binlevel(eq, ("&",), lambda op, a, b: a & b)()
    bxor = lambda: binlevel(band, ("^",), lambda op, a, b: a ^ b)()
    bor = lambda: binlevel(bxor, ("|",), lambda op, a, b: a | b)()

    def unary():
        if is_op("-"): nxt(); a = unary(); return lambda: i32(-a())
        if is_op("!"): nxt(); a = unary(); return lambda: 1 if a() == 0 else 0
        if is_op("~"): nxt(); a = unary(); return lambda: ~a()
        return primary()

    def primary():
        t = S["tok"]
        if t["t"] == "num": v = t["v"]; nxt(); return lambda: v
        if is_op("("):
            nxt(); e = ternary()
            if not is_op(")"): syn()
            nxt(); return e
        if t["t"] == "id":
            name = t["s"]; nxt()
            if is_op("("):
                if name not in FUN: fail("expr: unknown function '%s'" % name)
                nxt(); args = []
                if not is_op(")"):
                    args.append(ternary())
                    while is_op(","): nxt(); args.append(ternary())
                if not is_op(")"): syn()
                nxt()
                lo, hi = FUN[name]
                if len(args) < lo or len(args) > hi: fail("expr: wrong number of arguments for '%s'" % name)

                def call():
                    v = [f() for f in args]
                    if name == "min": return min(v)
                    if name == "max": return max(v)
                    if name == "abs": return i32(abs(v[0]))
                    if name == "sign": return 1 if v[0] > 0 else -1 if v[0] < 0 else 0
                    if name == "clamp": return min(max(v[0], v[1]), v[2])
                    if v[1] < 0: fail("expr: negative exponent")
                    r, bs, e = 1, v[0], v[1]
                    while e > 0:
                        if e & 1: r = imul(r, bs)
                        bs = imul(bs, bs); e >>= 1
                    return r
                return call

            def var():
                if name not in vars_: fail("expr: unknown variable '%s'" % name)
                return vars_[name]
            return var
        syn()

    nxt()
    e = ternary()
    if S["tok"]["t"] != "end": syn()
    return e()


def cmd_expr(args):
    if len(args) < 1: fail("usage: tool expr EXPR [name=value...]", 2)
    vars_ = {}
    for a in args[1:]:
        m = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_]*)=(-?[0-9]+)", a)
        if not m or len(m.group(2)) > 10: fail("expr: bad assignment '%s'" % a, 2)
        vars_[m.group(1)] = i32(int(m.group(2)))
    out(str(expr_eval(args[0], vars_)) + "\n")


# ------------------------------------------------------------------ tpl
class Undef:
    def __repr__(self): return "undefined"
UNDEF = Undef()


def is_num(v): return isinstance(v, (int, float)) and not isinstance(v, bool)


def norm(v):
    if isinstance(v, float) and v == v and abs(v) != float("inf") and v == int(v) and abs(v) < 1e21: return int(v)
    return v


def num_str(v):
    v = norm(v)
    return str(v) if isinstance(v, int) else repr(v)


def truthy(v): return not (v is UNDEF or v is None or v is False or v == "" and isinstance(v, str) or (is_num(v) and v == 0))


def show(v):
    if v is UNDEF or v is None: return ""
    if isinstance(v, bool): return "true" if v else "false"
    if isinstance(v, list): return ",".join(show(x) for x in v)
    if isinstance(v, dict): return "[object Object]"
    if is_num(v): return num_str(v)
    return v


def seq(a, b):
    if a is UNDEF or b is UNDEF: return a is b
    if a is None or b is None: return a is b
    if isinstance(a, bool) or isinstance(b, bool): return isinstance(a, bool) and isinstance(b, bool) and a == b
    if is_num(a) and is_num(b): return a == b
    if isinstance(a, str) and isinstance(b, str): return a == b
    return a is b


def key_str(k):
    if isinstance(k, bool): return "true" if k else "false"
    if k is None: return "null"
    if k is UNDEF: return "undefined"
    if is_num(k): return num_str(k)
    if isinstance(k, str): return k
    return show(k)


def lookup(o, k):
    k = norm(k)
    if isinstance(o, str):
        if k == "length" and isinstance(k, str): return len16(o)
        if is_num(k) and isinstance(k, int) and 0 <= k < len(o): return o[k]
        return UNDEF
    if isinstance(o, list):
        if isinstance(k, str) and k == "length": return len(o)
        if is_num(k) and isinstance(k, int) and 0 <= k < len(o): return o[k]
        return UNDEF
    if isinstance(o, dict):
        ks = key_str(k)
        return o[ks] if ks in o else UNDEF
    return UNDEF


def f_len(v):
    if isinstance(v, str): return len16(v)
    if isinstance(v, list): return len(v)
    if isinstance(v, dict): return len(v)
    return 0


def f_join(v, a):
    if isinstance(v, list):
        sep = "," if a is UNDEF else (a if isinstance(a, str) else num_str(a))
        return sep.join(show(x) for x in v)
    return show(v)


def f_default(v, a):
    return a if (v is UNDEF or v is None or v == "" and isinstance(v, str)) else v


def f_rev(v):
    if isinstance(v, list): return list(reversed(v))
    return show(v)[::-1]


FILTERS = {"upper": lambda v, a: show(v).upper(), "lower": lambda v, a: show(v).lower(), "trim": lambda v, a: js_trim(show(v)),
           "len": lambda v, a: f_len(v), "join": f_join, "default": f_default, "rev": lambda v, a: f_rev(v)}


def compile_expr(src, tag_pos):
    S = {"i": 0}

    def syn_err(): fail("tpl: syntax error in tag at %d" % tag_pos)
    def ws():
        while S["i"] < len(src) and src[S["i"]] in JSWS: S["i"] += 1
    def word():
        ws(); m = re.match(r"[A-Za-z_][A-Za-z0-9_]*", src[S["i"]:]); return m.group(0) if m else None

    def or_e():
        l = and_e()
        while word() == "or":
            S["i"] += 2; a = l; r = and_e(); l = (lambda a, r: lambda sc: truthy(a(sc)) or truthy(r(sc)))(a, r)
        return l

    def and_e():
        l = not_e()
        while word() == "and":
            S["i"] += 3; a = l; r = not_e(); l = (lambda a, r: lambda sc: truthy(a(sc)) and truthy(r(sc)))(a, r)
        return l

    def not_e():
        if word() == "not":
            S["i"] += 3; a = not_e(); return lambda sc: not truthy(a(sc))
        return cmp_e()

    def cmp_e():
        a = operand(); ws()
        m = re.match(r"==|!=|<|>", src[S["i"]:])
        if not m: return a
        S["i"] += len(m.group(0)); b = operand(); op = m.group(0)

        def f(sc):
            x, y = a(sc), b(sc)
            if op == "==": return seq(x, y)
            if op == "!=": return not seq(x, y)
            tx = "number" if is_num(x) else "string" if isinstance(x, str) else "other"
            ty = "number" if is_num(y) else "string" if isinstance(y, str) else "other"
            if tx != ty or tx == "other": return False
            if tx == "string": x, y = x.encode("utf-16-be", "surrogatepass"), y.encode("utf-16-be", "surrogatepass")
            return x < y if op == "<" else x > y
        return f

    def operand():
        v = primary(); ws()
        while S["i"] < len(src) and src[S["i"]] == "|":
            S["i"] += 1; name = word()
            if name is None: syn_err()
            S["i"] += len(name)
            if name not in FILTERS: fail("tpl: unknown filter '%s' at %d" % (name, tag_pos))
            arg = UNDEF; ws()
            if S["i"] < len(src) and src[S["i"]] == "(":
                S["i"] += 1; lit = literal()
                if lit is None: syn_err()
                arg = lit[0]; ws()
                if S["i"] >= len(src) or src[S["i"]] != ")": syn_err()
                S["i"] += 1
            f, prev = FILTERS[name], v
            v = (lambda f, prev, arg: lambda sc: f(prev(sc), arg))(f, prev, arg)
            ws()
        return v

    def literal():
        ws(); rest = src[S["i"]:]
        m = re.match(r'"([^"]*)"', rest) or re.match(r"'([^']*)'", rest)
        if m: S["i"] += len(m.group(0)); return (m.group(1),)
        m = re.match(r"-?[0-9]+", rest)
        if m: S["i"] += len(m.group(0)); return (int(m.group(0)),)
        return None

    def primary():
        lit = literal()
        if lit is not None: return lambda sc: lit[0]
        w = word()
        if w is None: syn_err()
        S["i"] += len(w)
        if w == "true": return lambda sc: True
        if w == "false": return lambda sc: False
        if w == "null": return lambda sc: None
        path = [("k", w)]
        while True:
            if S["i"] < len(src) and src[S["i"]] == ".":
                S["i"] += 1; m = re.match(r"[A-Za-z_][A-Za-z0-9_]*", src[S["i"]:])
                if not m: syn_err()
                S["i"] += len(m.group(0)); path.append(("k", m.group(0)))
            elif S["i"] < len(src) and src[S["i"]] == "[":
                S["i"] += 1; ix = or_e(); ws()
                if S["i"] >= len(src) or src[S["i"]] != "]": syn_err()
                S["i"] += 1; path.append(("f", ix))
            else: break

        def f(sc):
            cur = sc(path[0][1])
            for kind, x in path[1:]:
                cur = lookup(cur, x(sc) if kind == "f" else x)
            return cur
        return f

    e = or_e(); ws()
    if S["i"] < len(src): syn_err()
    return e


def compile_tpl(text):
    root = {"kind": "seq", "items": []}
    stack = [{"node": root, "open": None}]
    S = {"trim": False}

    def push_text(t):
        if S["trim"]: t = re.sub("^" + WSC + "+", "", t); S["trim"] = False
        if t: stack[-1]["node"]["items"].append({"kind": "text", "s": t})
    pos = 0
    while pos < len(text):
        found = -1; q = text.find("{", pos)
        while 0 <= q < len(text):
            if q + 1 < len(text) and text[q + 1] in "{%#": found = q; break
            q = text.find("{", q + 1)
        if found < 0: push_text(text[pos:]); break
        lead = text[pos:found]; kc = text[found + 1]
        closer = "}}" if kc == "{" else "%}" if kc == "%" else "#}"
        e = text.find(closer, found + 2)
        if e < 0: fail("tpl: unterminated tag at %d" % found)
        inner = text[found + 2:e]
        if inner[:1] == "-": lead = re.sub(WSC + "+$", "", lead); inner = inner[1:]
        trim_after = False
        if inner[-1:] == "-": trim_after = True; inner = inner[:-1]
        push_text(lead); pos = e + 2
        if kc == "#": S["trim"] = trim_after; continue
        cur = stack[-1]
        if kc == "{":
            cur["node"]["items"].append({"kind": "out", "f": compile_expr(inner, found)})
        else:
            m = re.match("^" + WSC + "*([A-Za-z_]+)" + WSC + "*([\\s\\S]*)$", inner)
            kw = m.group(1) if m else ""; rest = m.group(2) if m else ""
            if kw == "if":
                n = {"kind": "if", "cond": compile_expr(rest, found), "a": {"kind": "seq", "items": []}, "b": None}
                cur["node"]["items"].append(n); stack.append({"node": n["a"], "open": "endif", "owner": n})
            elif kw == "else":
                if cur["open"] != "endif" or cur["owner"]["b"]: fail("tpl: unexpected 'else' at %d" % found)
                cur["owner"]["b"] = {"kind": "seq", "items": []}; cur["node"] = cur["owner"]["b"]
            elif kw == "for":
                fm = re.match("^([A-Za-z_][A-Za-z0-9_]*)" + WSC + "+in" + WSC + "+([\\s\\S]+)$", rest)
                if not fm: fail("tpl: syntax error in tag at %d" % found)
                n = {"kind": "for", "v": fm.group(1), "f": compile_expr(fm.group(2), found), "body": {"kind": "seq", "items": []}}
                cur["node"]["items"].append(n); stack.append({"node": n["body"], "open": "endfor", "owner": n})
            elif kw in ("endif", "endfor"):
                if js_trim(rest) != "": fail("tpl: syntax error in tag at %d" % found)
                if cur["open"] != kw: fail("tpl: unexpected '%s' at %d" % (kw, found))
                stack.pop()
            else: fail("tpl: syntax error in tag at %d" % found)
        S["trim"] = trim_after
    if len(stack) > 1: fail("tpl: missing '%s'" % stack[-1]["open"])
    return root


def render(node, scope, buf):
    for it in node["items"]:
        k = it["kind"]
        if k == "text": buf.append(it["s"])
        elif k == "out": buf.append(show(it["f"](scope)))
        elif k == "if":
            if truthy(it["cond"](scope)): render(it["a"], scope, buf)
            elif it["b"]: render(it["b"], scope, buf)
        else:
            sq = it["f"](scope)
            if isinstance(sq, str): sq = list(sq)
            if not isinstance(sq, list): continue
            for idx, x in enumerate(sq):
                loop = {"index": idx + 1, "index0": idx, "first": idx == 0, "last": idx == len(sq) - 1, "length": len(sq)}
                inner = (lambda x, loop: lambda key: x if key == it["v"] else loop if key == "loop" else scope(key))(x, loop)
                render(it["body"], inner, buf)


def cmd_tpl(args):
    if len(args) < 2: fail("usage: tool tpl TEMPLATE DATA.json", 2)
    text = read_input(args[0], "tpl").decode("utf-8", "replace")
    try: data = json_loads(read_input(args[1], "tpl").decode("utf-8", "replace"))
    except Exception: fail("tpl: invalid JSON")
    tree = compile_tpl(text); buf = []
    render(tree, lambda k: lookup(data, k), buf)
    out("".join(buf))


# ------------------------------------------------------------------ pack / unpack
def cmd_pack(args):
    data = read_input(args[0] if args else None, "pack")
    ob = bytearray(); st = {"acc": 0, "nb": 0}

    def emit(code, w):
        st["acc"] += code << st["nb"]; st["nb"] += w
        while st["nb"] >= 8: ob.append(st["acc"] & 255); st["acc"] >>= 8; st["nb"] -= 8
    d = {}; nxt = 258; width = 9
    emit(256, width); w = -1
    for c in data:
        if w < 0: w = c; continue
        key = w * 256 + c
        if key in d: w = d[key]; continue
        emit(w, width); d[key] = nxt; nxt += 1
        if nxt > (1 << width) and width < 12: width += 1
        if nxt >= 4096: emit(256, width); d = {}; nxt = 258; width = 9
        w = c
    if w >= 0: emit(w, width)
    emit(257, width)
    if st["nb"] > 0: ob.append(st["acc"] & 255)
    s = 0
    for x in data: s = (s * 7 + x) & 255
    out(b"LZW1" + struct.pack(">I", len(data)) + bytes(ob) + bytes([s]))


def cmd_unpack(args):
    buf = read_input(args[0] if args else None, "unpack")
    if len(buf) < 9 or buf[:4] != b"LZW1": fail("unpack: bad header")
    total = struct.unpack(">I", buf[4:8])[0]; summ = buf[-1]; body = buf[8:-1]
    S = {"bp": 0}

    def read(w):
        if S["bp"] + w > len(body) * 8: return -1
        v = 0
        for i in range(w): v += ((body[(S["bp"] + i) >> 3] >> ((S["bp"] + i) & 7)) & 1) << i
        S["bp"] += w; return v
    tab = []; nxt = 258; width = 9; prev = None; res = bytearray()

    def expand(code):
        st = []
        while code >= 258: e = tab[code - 258]; st.append(e[1]); code = e[0]
        st.append(code); st.reverse(); return st
    while True:
        code = read(width)
        if code < 0: fail("unpack: truncated")
        if code == 257: break
        if code == 256: tab = []; nxt = 258; width = 9; prev = None; continue
        if prev is None:
            if code > 255: fail("unpack: corrupt stream")
            sq = [code]
        elif code < nxt: sq = expand(code)
        elif code == nxt: p = expand(prev); sq = p + [p[0]]
        else: fail("unpack: corrupt stream")
        if prev is not None:
            tab.append((prev, sq[0])); nxt += 1
            if nxt + 1 > (1 << width) and width < 12: width += 1
        res.extend(sq)
        if len(res) > total: fail("unpack: length mismatch")
        prev = code
    if len(res) != total: fail("unpack: length mismatch")
    s = 0
    for x in res: s = (s * 7 + x) & 255
    if s != summ: fail("unpack: crc mismatch")
    out(bytes(res))


# ------------------------------------------------------------------ seal / open
def fnv32(b, seed):
    h = seed & M32
    for x in b: h = ((h ^ x) * 0x01000193) & M32
    return h


def rotl(x, k): return ((x << k) | (x >> (32 - k))) & M32


def keystream(key, nonce):
    kb = key.encode("utf-8", "surrogateescape")
    s = [fnv32(kb, sd) for sd in (0x811c9dc5, 0x1b873593, 0xcc9e2d51, 0x85ebca6b)]
    s[0] = (s[0] ^ nonce) & M32
    if (s[0] | s[1] | s[2] | s[3]) == 0: s[0] = 1

    def step():
        r = (((rotl((s[1] * 5) & M32, 7)) * 9) & M32)
        t = (s[1] << 9) & M32
        s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3]; s[2] ^= t; s[3] = rotl(s[3], 11)
        return r >> 24
    return step


def mac_of(key, data): return crc_of(key.encode("utf-8", "surrogateescape") + b"\0" + data)


def cmd_seal(args):
    if len(args) < 1: fail("usage: tool seal KEY [FILE]", 2)
    data = read_input(args[1] if len(args) > 1 else None, "seal")
    n = len(data)
    nonce = fnv32(bytes([n & 255, (n >> 8) & 255, (n >> 16) & 255, (n >> 24) & 255]), fnv32(args[0].encode("utf-8", "surrogateescape"), 0x9747b28c))
    ks = keystream(args[0], nonce)
    c = bytes(x ^ ks() for x in data)
    out((struct.pack(">I", nonce) + c + struct.pack(">I", mac_of(args[0], data))).hex() + "\n")


def cmd_open(args):
    if len(args) < 1: fail("usage: tool open KEY [FILE]", 2)
    text = js_trim(read_input(args[1] if len(args) > 1 else None, "open").decode("utf-8", "replace"))
    if len(text) % 2 != 0 or not re.fullmatch(r"[0-9a-fA-F]*", text): fail("open: bad hex")
    raw = bytes.fromhex(text)
    if len(raw) < 8: fail("open: too short")
    nonce = struct.unpack(">I", raw[:4])[0]; c = raw[4:-4]; mac = struct.unpack(">I", raw[-4:])[0]
    ks = keystream(args[0], nonce)
    p = bytes(x ^ ks() for x in c)
    if mac_of(args[0], p) != mac: fail("open: bad mac")
    out(p)


CMDS = {"crc": cmd_crc, "b32": cmd_b32, "ser": cmd_ser, "de": cmd_de, "expr": cmd_expr, "tpl": cmd_tpl,
        "pack": cmd_pack, "unpack": cmd_unpack, "seal": cmd_seal, "open": cmd_open}


def main():
    a = sys.argv[1:]
    try:
        if not a: fail("usage: tool <crc|b32|ser|de|expr|tpl|pack|unpack|seal|open> ...", 2)
        if a[0] == "--version": out("tool 3.1.4\n"); return 0
        if a[0] not in CMDS: fail("tool: unknown command '%s'" % a[0], 2)
        CMDS[a[0]](a[1:])
    except Exit as e:
        return e.code
    return 0


if __name__ == "__main__":
    sys.setrecursionlimit(10000)
    sys.exit(main())
