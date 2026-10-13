"""Tiny assembler for the BCX1 v2 bytecode (build-time only, hidden side)."""
import struct

NAMES = ("NOP PUSHI PUSHC POP DUP SWAP OVER ROT ADD SUB MUL DIV MOD NEG AND OR XOR NOT SHL SHR USHR "
         "EQ NE LT LE GT GE JMP JZ JNZ CALL RET HALT LOAD STORE GLOAD GSTORE NEWARR ALOAD ASTORE ALEN "
         "SLEN SCAT SIDX SSUB ITOS STOI CHR PRINT PUTC READ READC ASSERT DEPTH PUSHL TYPEOF").split()
OP = {n: i for i, n in enumerate(NAMES)}
FMT = {"PUSHI": "i", "PUSHC": "H", "JMP": "r", "JZ": "r", "JNZ": "r", "CALL": "HB", "LOAD": "B", "STORE": "B",
       "GLOAD": "B", "GSTORE": "B", "PUSHL": "q"}
M64 = (1 << 64) - 1


def perm_table():
    st = 0xB70
    p = list(range(256))
    for i in range(255, 0, -1):
        st = (st * 1103515245 + 12345) & 0xFFFFFFFF
        j = (st >> 16) % (i + 1)
        p[i], p[j] = p[j], p[i]
    return p


PERM = perm_table()


def csum(b):
    h = 0x1505
    for x in b:
        h = ((h * 33) & 0xFFFFFFFF) ^ x
    return (((h << 7) | (h >> 25)) & 0xFFFFFFFF) ^ 0xC0DEC0DE


class Asm:
    def __init__(self):
        self.items = []      # (name, operands) or ("LABEL", name)
        self.consts = []     # ("i", v) / ("s", bytes)

    def __getattr__(self, name):
        if name.upper() in OP:
            def f(*a):
                self.items.append((name.upper(), a)); return self
            return f
        raise AttributeError(name)

    def label(self, n):
        self.items.append(("LABEL", n)); return self

    def push(self, v):
        if -2**31 <= v < 2**31: return self.PUSHI(v)
        return self.PUSHL(v)

    def cint(self, v):
        self.consts.append(("i", v)); return self.PUSHC(len(self.consts) - 1)

    def str(self, s):
        if isinstance(s, str): s = s.encode()
        self.consts.append(("s", s)); return self.PUSHC(len(self.consts) - 1)

    def raw(self, bs):
        self.items.append(("RAW", bs)); return self

    def _sizes(self, name):
        return 1 + sum({"i": 4, "H": 2, "B": 1, "r": 2, "q": 8}[c] for c in FMT.get(name, ""))

    def assemble(self):
        pos, labels = 0, {}
        for n, a in self.items:
            if n == "LABEL": labels[a] = pos
            elif n == "RAW": pos += len(a)
            else: pos += self._sizes(n)
        out, pos = [], 0     # list of (offset, opcode, [operand bytes])
        for n, a in self.items:
            if n == "LABEL": continue
            if n == "RAW":
                out.append((pos, None, list(a))); pos += len(a); continue
            ops = bytearray()
            fm = FMT.get(n, ""); sz = self._sizes(n)
            vals = list(a)
            for c in fm:
                v = vals.pop(0)
                if c == "r":
                    t = labels[v] if isinstance(v, str) else v
                    ops += struct.pack("<h", t - (pos + sz) if isinstance(v, str) else v)
                elif c == "H":
                    ops += struct.pack("<H", labels[v] if isinstance(v, str) else v)
                else:
                    ops += struct.pack("<" + c, v if c != "q" else ((v + 2**63) % 2**64) - 2**63)
            out.append((pos, OP[n], list(ops))); pos += sz
        return out, labels

    def build(self, flags=0, entry="main", checksum=True, fix=None):
        ins, labels = self.assemble()
        code = bytearray()
        for pos, op, ops in ins:
            if op is None:
                code += bytes(ops); continue
            if flags & 1:
                code.append(PERM[op] ^ ((pos * 7 + 3) & 0xff))
                code += bytes(b ^ ((pos + 13 * k + 0x5A) & 0xff) for k, b in enumerate(ops))
            else:
                code.append(op); code += bytes(ops)
        cp = bytearray()
        for i, (t, v) in enumerate(self.consts):
            if t == "i":
                cp += b"\0" + struct.pack("<q", v)
            else:
                enc = bytes(b ^ ((0x21 + i * 3 + j) & 0xff) for j, b in enumerate(v)) if flags & 2 else v
                cp += b"\1" + struct.pack("<H", len(v)) + enc
        e = labels.get(entry, 0) if isinstance(entry, str) else entry
        hdr = b"BCX1" + bytes([2, flags]) + struct.pack("<HII", len(self.consts), len(code), e)
        body = hdr + bytes(cp) + bytes(code)
        if fix: body = fix(body)
        return body + struct.pack("<I", csum(body) if checksum else 0)
