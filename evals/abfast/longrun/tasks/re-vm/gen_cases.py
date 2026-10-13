#!/usr/bin/env python3
"""Runs INSIDE the build container: records hidden cases for re-vm against the reference binary.
  gen_cases.py REF_BINARY HIDDEN_DIR PUBLIC_DIR
"""
import os, random, struct, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/src")
from asm import Asm, OP, PERM, csum
from casegen import Recorder

REF, HID, PUB = sys.argv[1], sys.argv[2], sys.argv[3]
R = random.Random(20261014)
rec = Recorder(HID, [REF])
CMP = ["rc", "stdout", "stderr"]
POOL = []     # (builder fn, stdin) reusable for the obfuscated-encoding milestone
M64 = 2**64
EDGE = [0, 1, -1, 2, -2, 3, 7, 10, 63, 64, 65, 100, -100, 255, 2**31 - 1, -2**31, 2**63 - 1, -2**63, 123456789012, -987654321098]


def add(ms, a, args=(), stdin=None, flags=0, pool=True, name="p.bc", **kw):
    data = a if isinstance(a, bytes) else a.build(flags)
    rec.add(ms, list(args) + [name], files={name: data}, stdin=stdin, cmp=CMP, **kw)
    if pool and not isinstance(a, bytes):
        POOL.append((a, list(args), stdin))


def prog(f):
    a = Asm(); a.label("main"); f(a); return a


def finish(a, code=0):
    return a.push(code).HALT()


# ------------------------------------------------------------------ m01 header / cli
def valid_prog(flags=0):
    a = Asm(); a.label("main").push(5).PRINT().str("x").PRINT().cint(2**40).PRINT()
    return a.build(flags)


vp = valid_prog()
for fl in (0, 1, 2, 3):
    add("m01", valid_prog(fl), ["--info"], pool=False)
    add("m01", valid_prog(fl), [], pool=False)
for k in range(6):
    a = Asm()
    for i in range(R.randrange(0, 6)): a.cint(R.randrange(-2**40, 2**40))
    for i in range(R.randrange(0, 4)): a.str(bytes(R.randrange(97, 123) for _ in range(R.randrange(0, 20))))
    a.label("main")
    for i in range(R.randrange(0, 5)): a.NOP()
    add("m01", a, ["--info"], pool=False)
    add("m01", a, [], pool=False)
a = Asm(); a.NOP().NOP().label("main").push(9).PRINT(); add("m01", a, ["--info"], pool=False); add("m01", a, [], pool=False)   # entry != 0
rec.add("m01", [], files={}, cmp=CMP)
rec.add("m01", ["nope.bc"], files={}, cmp=CMP)
rec.add("m01", ["--info"], files={}, cmp=CMP)
rec.add("m01", ["-x", "p.bc"], files={"p.bc": vp}, cmp=CMP)
rec.add("m01", ["p.bc", "p.bc"], files={"p.bc": vp}, cmp=CMP)
for b in ("abc", "0", "-5", "12x", ""):
    rec.add("m01", ["-b", b, "p.bc"], files={"p.bc": vp}, cmp=CMP)
rec.add("m01", ["-b"], files={"p.bc": vp}, cmp=CMP)
rec.add("m01", ["-b", "100", "--info", "p.bc"], files={"p.bc": vp}, cmp=CMP)
rec.add("m01", ["p.bc", "--info"], files={"p.bc": vp}, cmp=CMP)

# ------------------------------------------------------------------ m02 stack basics
for code in (0, 1, 7, 255, 256, 257, -1, -256, 2**40 + 5):
    add("m02", prog(lambda a, c=code: finish(a, c) if abs(c) < 2**31 else a.push(c).HALT()))
a = prog(lambda a: a.push(1).PRINT()); add("m02", a)                      # falls off the end
add("m02", prog(lambda a: a.push(300).RET()))                              # RET at top level
add("m02", prog(lambda a: a.push(5).push(6).RET()))
add("m02", prog(lambda a: a.NOP().NOP().push(-17).PRINT().push(2**62).PRINT().push(-2**63).PRINT()))
add("m02", prog(lambda a: a.cint(2**63 - 1).PRINT().cint(-5).PRINT().cint(0).PRINT()))
for i in range(6):
    n = R.randrange(3, 8)
    def f(a, n=n):
        for k in range(n): a.push(R.randrange(-50, 50))
        d = n
        for k in range(R.randrange(4, 12)):
            c = R.choice(["DUP", "SWAP", "OVER", "ROT", "POP", "DEPTH"])
            if c in ("SWAP", "OVER") and d < 2: continue
            if c == "ROT" and d < 3: continue
            if c in ("DUP", "POP") and d < 1: continue
            if c == "POP" and d < 2: continue
            getattr(a, c)(); d += {"DUP": 1, "OVER": 1, "DEPTH": 1, "POP": -1}.get(c, 0)
        for k in range(d): a.PRINT()
    add("m02", prog(f))
add("m02", prog(lambda a: a.push(1).push(2).push(3).DEPTH().PRINT().ROT().PRINT().PRINT().PRINT()))
add("m02", prog(lambda a: a.push(1).DUP().DUP().DEPTH().PRINT()))
add("m02", prog(lambda a: a.push(4).push(5).OVER().PRINT().PRINT().PRINT()))
add("m02", prog(lambda a: a.push(4).push(5).SWAP().PRINT().PRINT()))
add("m02", prog(lambda a: a.push(2**31 - 1).push(-2**31).push(2**31).push(-2**31 - 1).PRINT().PRINT().PRINT().PRINT()))
add("m02", prog(lambda a: a.push(7).TYPEOF().PRINT().str("s").TYPEOF().PRINT().push(3).NEWARR().TYPEOF().PRINT()))

# ------------------------------------------------------------------ m03 arithmetic
BIN = ["ADD", "SUB", "MUL", "DIV", "MOD", "AND", "OR", "XOR", "SHL", "SHR", "USHR"]
for op in BIN:
    for rep in range(4):
        pairs = [(R.choice(EDGE), R.choice(EDGE)) for _ in range(6)]
        if op in ("DIV", "MOD"): pairs = [(x, y if y else 3) for x, y in pairs]
        pairs.append((-2**63, -1)) if op in ("DIV", "MOD") and rep == 0 else None
        def f(a, pairs=pairs, op=op):
            for x, y in pairs: a.push(x).push(y); getattr(a, op)(); a.PRINT()
        add("m03", prog(f))
for op in ("NEG", "NOT"):
    def f(a, op=op):
        for x in EDGE: a.push(x); getattr(a, op)(); a.PRINT()
    add("m03", prog(f))
add("m03", prog(lambda a: a.push(-7).push(2).DIV().PRINT().push(-7).push(2).MOD().PRINT().push(7).push(-2).DIV().PRINT().push(7).push(-2).MOD().PRINT()))
add("m03", prog(lambda a: a.push(1).push(-1).SHL().PRINT().push(-8).push(1).SHR().PRINT().push(-8).push(1).USHR().PRINT().push(1).push(64).SHL().PRINT()))

# ------------------------------------------------------------------ m04 comparisons and jumps
for op in ("EQ", "NE", "LT", "LE", "GT", "GE"):
    def f(a, op=op):
        for x in EDGE[::3]:
            for y in (R.choice(EDGE), x):
                a.push(x).push(y); getattr(a, op)(); a.PRINT()
    add("m04", prog(f))


def loop_sum(n):
    a = Asm(); a.label("main").push(0).STORE(0).push(n).STORE(1)
    a.label("L").LOAD(1).JZ("E").LOAD(0).LOAD(1).ADD().STORE(0).LOAD(1).push(1).SUB().STORE(1).JMP("L")
    a.label("E").LOAD(0).PRINT(); return a
for n in (0, 1, 2, 10, 100, 1000, 20000):
    add("m04", loop_sum(n))
for n in (0, 1, 5, 12):
    a = Asm(); a.label("main").push(n).STORE(0)
    a.label("T").LOAD(0).JNZ("B").JMP("D").label("B").LOAD(0).PRINT().LOAD(0).push(1).SUB().STORE(0).JMP("T").label("D").str("done").PRINT()
    add("m04", a)
for x in (-3, 0, 4, 10):
    a = Asm(); a.label("main").push(x).DUP().push(0).LT().JNZ("neg").DUP().JZ("zero").str("pos").PRINT().JMP("end")
    a.label("neg").str("neg").PRINT().JMP("end").label("zero").str("zero").PRINT().label("end").PRINT()
    add("m04", a)
a = Asm(); a.label("main").push(3).STORE(0)       # nested loops multiplication table
a.label("o").LOAD(0).JZ("end").push(3).STORE(1)
a.label("i").LOAD(1).JZ("on").LOAD(0).push(10).MUL().LOAD(1).ADD().PRINT().LOAD(1).push(1).SUB().STORE(1).JMP("i")
a.label("on").LOAD(0).push(1).SUB().STORE(0).JMP("o").label("end"); add("m04", a)
add("m04", prog(lambda a: a.JMP(0).NOP().push(1).PRINT()))     # jmp 0 forward by 0 = next instr

# ------------------------------------------------------------------ m05 locals and globals
for i in range(6):
    vals = [R.choice(EDGE) for _ in range(5)]
    def f(a, vals=vals):
        for k, v in enumerate(vals): a.push(v).STORE(k)
        for k in range(4, -1, -1): a.LOAD(k).PRINT()
        a.LOAD(9).PRINT().LOAD(15).PRINT()
        for k, v in enumerate(vals): a.push(v ^ k).GSTORE(k * 13)
        for k in range(5): a.GLOAD(k * 13).PRINT()
        a.GLOAD(63).PRINT().GLOAD(1).PRINT()
    add("m05", prog(f))
add("m05", prog(lambda a: a.push(1).push(2).STORE(0).STORE(1).LOAD(0).PRINT().LOAD(1).PRINT()))
add("m05", prog(lambda a: a.str("hello").STORE(3).LOAD(3).PRINT().LOAD(3).SLEN().PRINT().str("g").GSTORE(5).GLOAD(5).GLOAD(5).SCAT().PRINT()))
add("m05", prog(lambda a: a.push(3).NEWARR().STORE(2).LOAD(2).push(1).push(99).ASTORE().LOAD(2).PRINT()))
for k in range(6):
    n = R.randrange(2, 9)
    def f(a, n=n):          # swap via locals, accumulate in globals
        a.push(3).GSTORE(0)
        for i in range(n):
            a.GLOAD(0).push(R.randrange(1, 9)).MUL().GSTORE(0)
        a.GLOAD(0).PRINT()
    add("m05", prog(f))

# ------------------------------------------------------------------ m06 calls
def fib(n):
    a = Asm(); a.label("main").push(n).CALL("fib", 1).PRINT().push(0).HALT()
    a.label("fib").LOAD(0).push(2).LT().JZ("rec").LOAD(0).RET()
    a.label("rec").LOAD(0).push(1).SUB().CALL("fib", 1).LOAD(0).push(2).SUB().CALL("fib", 1).ADD().RET(); return a
for n in (0, 1, 2, 5, 10, 15, 18):
    add("m06", fib(n))
def fact(n):
    a = Asm(); a.label("main").push(n).CALL("f", 1).PRINT()
    a.label("f").LOAD(0).JNZ("go").push(1).RET().label("go").LOAD(0).LOAD(0).push(1).SUB().CALL("f", 1).MUL().RET(); return a
for n in (1, 5, 20, 25):
    a = Asm(); a.label("main").push(n).CALL("f", 1).PRINT().push(0).HALT()
    a.label("f").LOAD(0).JNZ("go").push(1).RET().label("go").LOAD(0).LOAD(0).push(1).SUB().CALL("f", 1).MUL().RET()
    add("m06", a)
a = Asm(); a.label("main").push(100).push(30).push(7).CALL("g", 3).PRINT().push(0).HALT()     # arg order
a.label("g").LOAD(0).LOAD(1).SUB().LOAD(2).SUB().RET(); add("m06", a)
a = Asm(); a.label("main").push(5).STORE(0).push(1).CALL("h", 1).PRINT().LOAD(0).PRINT().push(0).HALT()   # locals isolated
a.label("h").push(77).STORE(0).LOAD(0).LOAD(1).ADD().RET(); add("m06", a)
a = Asm(); a.label("main").push(1).push(2).push(3).CALL("junk", 0).PRINT().DEPTH().PRINT().push(0).HALT()   # RET discards extras
a.label("junk").push(10).push(11).push(12).DEPTH().PRINT().RET(); add("m06", a)
a = Asm(); a.label("main").push(8).push(9).CALL("d", 1).PRINT().DEPTH().PRINT().push(0).HALT()
a.label("d").DEPTH().PRINT().LOAD(0).RET(); add("m06", a)
a = Asm(); a.label("main").push(6).CALL("a", 1).PRINT().push(0).HALT()                       # mutual recursion
a.label("a").LOAD(0).JZ("az").LOAD(0).push(1).SUB().CALL("b", 1).push(1).ADD().RET().label("az").push(100).RET()
a.label("b").LOAD(0).JZ("bz").LOAD(0).push(1).SUB().CALL("a", 1).push(10).ADD().RET().label("bz").push(200).RET(); add("m06", a)
a = Asm(); a.label("main").str("x").push(3).CALL("rep", 2).PRINT().push(0).HALT()           # strings through frames
a.label("rep").LOAD(0).STORE(2).label("l").LOAD(1).JZ("e").LOAD(2).LOAD(0).SCAT().STORE(2).LOAD(1).push(1).SUB().STORE(1).JMP("l").label("e").LOAD(2).RET(); add("m06", a)
for n in (1, 7, 31):      # recursion depth just inside the 32 frame limit
    a = Asm(); a.label("main").push(n).CALL("r", 1).PRINT().push(0).HALT()
    a.label("r").LOAD(0).JZ("z").LOAD(0).push(1).SUB().CALL("r", 1).push(1).ADD().RET().label("z").push(0).RET(); add("m06", a)
a = Asm(); a.label("main").CALL("e", 0).PRINT().push(3).HALT().label("e").RET()   # RET with empty frame -> underflow trap
add("m06", a)

# ------------------------------------------------------------------ m07 strings
WORDS = [b"", b"a", b"hello", b"Hello", b"hello world", b"zebra", b"abc", b"abd", b"ab", b"line one", b"123", b"-45", b"  42", b"+7", b"12ab",
         b"abc12", b"99999999999999999999", b"-99999999999999999999", b"9223372036854775807", b"0x1f", b"\x01\x02\x7f\x80\xff"]
def sfn(a, w1, w2):
    a.str(w1).DUP().PRINT().SLEN().PRINT().str(w1).str(w2).SCAT().PRINT()
    for op in ("EQ", "NE", "LT", "LE", "GT", "GE"):
        a.str(w1).str(w2); getattr(a, op)(); a.PRINT()
    a.str(w1).STOI().PRINT().str(w2).STOI().PRINT()
for w1, w2 in [(R.choice(WORDS), R.choice(WORDS)) for _ in range(14)]:
    add("m07", prog(lambda a, w1=w1, w2=w2: sfn(a, w1, w2)))
for w in (b"hello", b"abcdef", b"z"):
    def f(a, w=w):
        a.str(w).STORE(0)
        for i in range(len(w)): a.LOAD(0).push(i).SIDX().PRINT()
        a.LOAD(0).push(1).push(len(w) - 1).SSUB().PRINT().LOAD(0).push(0).push(0).SSUB().SLEN().PRINT().LOAD(0).push(len(w)).push(0).SSUB().SLEN().PRINT()
    add("m07", prog(f))
add("m07", prog(lambda a: [a.push(c).CHR().PRINT() for c in (65, 97, 48, 256 + 66, -191)] and a))
add("m07", prog(lambda a: a.push(-12345).ITOS().PRINT().push(2**63 - 1).ITOS().SLEN().PRINT().push(-2**63).ITOS().PRINT().push(0).ITOS().str("!").SCAT().PRINT()))
def revstr(a):
    a.str("stressed desserts").STORE(0).str("").STORE(1).LOAD(0).SLEN().STORE(2)
    a.label("L").LOAD(2).JZ("E").LOAD(2).push(1).SUB().STORE(2).LOAD(1).LOAD(0).LOAD(2).SIDX().CHR().SCAT().STORE(1).JMP("L").label("E").LOAD(1).PRINT()
add("m07", prog(revstr))
def build_long(a, n):     # concat doubling up to the limit minus one
    a.str("ab").STORE(0)
    for i in range(n): a.LOAD(0).LOAD(0).SCAT().STORE(0)
    a.LOAD(0).SLEN().PRINT()
add("m07", prog(lambda a: build_long(a, 10)))
add("m07", prog(lambda a: a.str("a").str("a").EQ().PRINT().str("a").STORE(0).LOAD(0).LOAD(0).EQ().PRINT().str("b").str("a").LT().PRINT()))
add("m07", prog(lambda a: a.str("12 34").STOI().PRINT().str("\t\n 5").STOI().PRINT().str("-").STOI().PRINT().str("007").STOI().PRINT()))
add("m07", prog(lambda a: a.str("x").STORE(0).push(0).STORE(1).label("L").LOAD(1).push(5).GE().JNZ("E").LOAD(1).ITOS().LOAD(0).SCAT().PRINT().LOAD(1).push(1).ADD().STORE(1).JMP("L").label("E")))

# ------------------------------------------------------------------ m08 arrays
def fill_squares(n):
    a = Asm(); a.label("main").push(n).NEWARR().STORE(0).push(0).STORE(1)
    a.label("L").LOAD(1).push(n).GE().JNZ("P").LOAD(0).LOAD(1).LOAD(1).LOAD(1).MUL().ASTORE().LOAD(1).push(1).ADD().STORE(1).JMP("L")
    a.label("P").LOAD(0).PRINT().LOAD(0).ALEN().PRINT(); return a
for n in (0, 1, 5, 12, 40):
    add("m08", fill_squares(n))
def bubble(vals):
    n = len(vals); a = Asm(); a.label("main").push(n).NEWARR().STORE(0)
    for i, v in enumerate(vals): a.LOAD(0).push(i).push(v).ASTORE()
    a.push(0).STORE(1).label("o").LOAD(1).push(n - 1).GE().JNZ("D").push(0).STORE(2)
    a.label("i").LOAD(2).push(n - 1).LOAD(1).SUB().GE().JNZ("on")
    a.LOAD(0).LOAD(2).ALOAD().LOAD(0).LOAD(2).push(1).ADD().ALOAD().GT().JZ("ns")
    a.LOAD(0).LOAD(2).ALOAD().STORE(3).LOAD(0).LOAD(2).LOAD(0).LOAD(2).push(1).ADD().ALOAD().ASTORE().LOAD(0).LOAD(2).push(1).ADD().LOAD(3).ASTORE()
    a.label("ns").LOAD(2).push(1).ADD().STORE(2).JMP("i").label("on").LOAD(1).push(1).ADD().STORE(1).JMP("o").label("D").LOAD(0).PRINT(); return a
for k in range(4):
    add("m08", bubble([R.randrange(-100, 100) for _ in range(R.randrange(2, 25))]))
def sieve(n):
    a = Asm(); a.label("main").push(n + 1).NEWARR().STORE(0).push(2).STORE(1)
    a.label("o").LOAD(1).push(n).GT().JNZ("c").LOAD(0).LOAD(1).ALOAD().JNZ("nx").LOAD(1).DUP().MUL().STORE(2)
    a.label("m").LOAD(2).push(n).GT().JNZ("nx").LOAD(0).LOAD(2).push(1).ASTORE().LOAD(2).LOAD(1).ADD().STORE(2).JMP("m")
    a.label("nx").LOAD(1).push(1).ADD().STORE(1).JMP("o").label("c").push(0).STORE(3).push(2).STORE(1)
    a.label("cl").LOAD(1).push(n).GT().JNZ("e").LOAD(0).LOAD(1).ALOAD().JNZ("sk").LOAD(3).push(1).ADD().STORE(3).label("sk").LOAD(1).push(1).ADD().STORE(1).JMP("cl").label("e").LOAD(3).PRINT(); return a
for n in (10, 100, 1000):
    add("m08", sieve(n))
add("m08", prog(lambda a: a.push(3).NEWARR().DUP().PRINT().STORE(0).push(3).NEWARR().STORE(1).LOAD(0).LOAD(0).EQ().PRINT().LOAD(0).LOAD(1).EQ().PRINT().LOAD(0).LOAD(1).NE().PRINT().LOAD(0).ALEN().PRINT()))
a = Asm(); a.label("main").push(4).NEWARR().STORE(0).LOAD(0).CALL("mut", 1).LOAD(0).PRINT().push(0).HALT()   # reference semantics
a.label("mut").LOAD(0).push(2).push(42).ASTORE().LOAD(0).push(0).push(-1).ASTORE().push(0).RET(); add("m08", a)
add("m08", prog(lambda a: a.push(1024).NEWARR().ALEN().PRINT().push(0).NEWARR().PRINT().push(0).NEWARR().ALEN().PRINT()))
add("m08", prog(lambda a: a.push(2).NEWARR().push(1).push(2**63 - 1).ASTORE().push(1).NEWARR().DUP().push(0).push(-2**63).ASTORE().PRINT()))

# ------------------------------------------------------------------ m09 stdin / stdout
def echo_lines(a):
    a.label("L").READ().DUP().TYPEOF().JNZ("S").PRINT().JMP("E").label("S").DUP().SLEN().PRINT().PRINT().JMP("L").label("E")
for stdin in (b"", b"one\n", b"one\ntwo\nthree\n", b"no newline at end", b"\n\n\n", b"a\r\nb\r\n", b"x" * 4100 + b"\ny\n"):
    add("m09", prog(echo_lines), stdin=stdin)
def echo_chars(a):
    a.label("L").READC().DUP().push(0).LT().JNZ("E").DUP().push(1).ADD().STORE(0).PUTC().JMP("L").label("E").PRINT()
for stdin in (b"", b"abc", b"hi there\n", bytes(range(1, 40)), b"\xff\xfe\x80"):
    add("m09", prog(echo_chars), stdin=stdin)
def sum_numbers(a):
    a.push(0).STORE(0).label("L").READ().DUP().TYPEOF().JNZ("S").POP().JMP("E").label("S").STOI().LOAD(0).ADD().STORE(0).JMP("L").label("E").LOAD(0).PRINT()
for stdin in (b"1\n2\n3\n", b"10\n-4\nabc\n 7\n", b"9223372036854775807\n1\n", b"5"):
    add("m09", prog(sum_numbers), stdin=stdin)
def upcase(a):
    a.label("L").READC().DUP().push(0).LT().JNZ("E").DUP().push(97).GE().OVER().push(122).LE().MUL().JZ("n").push(32).SUB().label("n").PUTC().JMP("L").label("E").POP()
for stdin in (b"hello World 123\n", b"", b"zzz aaa"):
    add("m09", prog(upcase), stdin=stdin)
add("m09", prog(lambda a: [a.push(c).PUTC() for c in (72, 105, 10, 256 + 65, 0, 66)] and a))
add("m09", prog(lambda a: a.READ().PRINT().READ().PRINT().READ().PRINT()), stdin=b"a\nb\n")
add("m09", prog(lambda a: a.READC().PRINT().READ().PRINT().READC().PRINT()), stdin=b"xyz\nq")
add("m09", prog(lambda a: a.str("out").PRINT().push(33).PUTC().push(10).PUTC().push(2).HALT()))

# ------------------------------------------------------------------ m10 traps
def T(f, **kw): add("m10", prog(f), **kw)
T(lambda a: a.POP())
T(lambda a: a.push(1).ADD())
T(lambda a: a.push(1).PRINT().PRINT())
T(lambda a: a.SWAP())
T(lambda a: a.push(1).OVER())
T(lambda a: a.push(1).push(2).ROT())
T(lambda a: a.push(1).push(0).DIV())
T(lambda a: a.push(1).push(0).MOD())
T(lambda a: a.NOP().NOP().push(5).push(0).DIV())
T(lambda a: a.str("a").push(1).ADD())
T(lambda a: a.push(1).str("a").SUB())
T(lambda a: a.str("a").NEG())
T(lambda a: a.str("a").push(1).LT().PRINT().str("a").push(1).EQ())
T(lambda a: a.push(1).NEWARR().push(1).NEWARR().LT())
T(lambda a: a.str("a").JZ(0))
T(lambda a: a.push(1).SLEN())
T(lambda a: a.push(1).ALEN())
T(lambda a: a.push(2).NEWARR().push(2).ALOAD())
T(lambda a: a.push(2).NEWARR().push(-1).ALOAD())
T(lambda a: a.push(2).NEWARR().push(5).push(1).ASTORE())
T(lambda a: a.push(2).NEWARR().push(0).str("x").ASTORE())
T(lambda a: a.str("abc").push(3).SIDX())
T(lambda a: a.str("abc").push(-1).SIDX())
T(lambda a: a.str("abc").push(2).push(2).SSUB())
T(lambda a: a.str("abc").push(-1).push(1).SSUB())
T(lambda a: a.push(-1).NEWARR())
T(lambda a: a.push(1025).NEWARR())
T(lambda a: a.push(0).ASSERT())
T(lambda a: a.push(1).ASSERT().push(0).ASSERT())
T(lambda a: a.LOAD(16))
T(lambda a: a.push(1).STORE(200))
T(lambda a: a.GLOAD(64))
T(lambda a: a.push(1).GSTORE(255))
T(lambda a: a.JMP(1000))
T(lambda a: a.JMP(-50))
T(lambda a: a.push(0).JZ(100))
T(lambda a: a.push(1).JZ(100).push(4).PRINT())
T(lambda a: a.push(1).JNZ(-20))
T(lambda a: a.CALL(5000, 0))
T(lambda a: a.push(1).CALL(0, 3))
T(lambda a: a.CALL("main", 0))
T(lambda a: a.push(1).CALL(0, 17))
T(lambda a: a.cint(1).PRINT().PUSHC(7))
T(lambda a: a.PUSHC(0))
T(lambda a: a.raw(bytes([200])))
T(lambda a: a.push(1).PRINT().raw(bytes([OP["PUSHI"], 1, 2])))
T(lambda a: a.raw(bytes([OP["JMP"], 1])))
T(lambda a: [a.push(i) for i in range(260)] and a)
T(lambda a: a.str("x" * 3000).DUP().SCAT().DUP().SCAT())
T(lambda a: a.str("").STORE(0).label("l").LOAD(0).push(0).push(0).SSUB().POP().JMP("l"))
T(lambda a: a.push(7).HALT().push(99))                              # code after halt never runs
T(lambda a: a.str("q").HALT())
T(lambda a: a.push(1).str("z").RET())
T(lambda a: a.str("z").RET())
T(lambda a: a.push(2).NEWARR().RET())
# stderr text + pc after output
T(lambda a: a.str("before").PRINT().push(4).push(0).MOD())

# ------------------------------------------------------------------ m11 corrupt / malformed files
good = valid_prog()
rec.add("m11", ["p.bc"], files={"p.bc": b""}, cmp=CMP)
for ln in (1, 3, 4, 5, 15, 16, 17, 19, 20):
    rec.add("m11", ["p.bc"], files={"p.bc": good[:ln]}, cmp=CMP)
for ln in (len(good) - 1, len(good) - 5, len(good) // 2):
    rec.add("m11", ["p.bc"], files={"p.bc": good[:ln]}, cmp=CMP)
    rec.add("m11", ["--info", "p.bc"], files={"p.bc": good[:ln]}, cmp=CMP)
rec.add("m11", ["p.bc"], files={"p.bc": b"BCX2" + good[4:]}, cmp=CMP)
rec.add("m11", ["p.bc"], files={"p.bc": b"\x7fELF" + good[4:]}, cmp=CMP)
for v in (0, 1, 3, 255):
    b = bytearray(good); b[4] = v; rec.add("m11", ["p.bc"], files={"p.bc": bytes(b)}, cmp=CMP)
for pos in (5, 8, 12, 16, 20, len(good) - 6, len(good) - 1):
    b = bytearray(good); b[pos] ^= 0x40; rec.add("m11", ["p.bc"], files={"p.bc": bytes(b)}, cmp=CMP)
def refix(b):
    b = bytearray(b); b[-4:] = struct.pack("<I", csum(bytes(b[:-4]))); return bytes(b)
def mut(pos, val, w=1):
    b = bytearray(good); b[pos:pos + w] = val; return refix(bytes(b))
rec.add("m11", ["p.bc"], files={"p.bc": mut(12, struct.pack("<I", 9999))}, cmp=CMP)           # entry beyond code
rec.add("m11", ["--info", "p.bc"], files={"p.bc": mut(12, struct.pack("<I", 9999))}, cmp=CMP)
rec.add("m11", ["p.bc"], files={"p.bc": mut(6, struct.pack("<H", 40))}, cmp=CMP)                # nconst too large
rec.add("m11", ["p.bc"], files={"p.bc": mut(8, struct.pack("<I", 3))}, cmp=CMP)                 # code_len mismatch
rec.add("m11", ["p.bc"], files={"p.bc": mut(16, b"\x07")}, cmp=CMP)                             # bad tag
rec.add("m11", ["p.bc"], files={"p.bc": refix(good[:-4] + b"\x00\x00")}, cmp=CMP)               # trailing bytes
rec.add("m11", ["--info", "p.bc"], files={"p.bc": refix(good[:-4] + b"\x00\x00")}, cmp=CMP)
a = Asm(); a.label("main").push(1); a.consts.append(("s", b"y" * 4097)); add("m11", a, [], pool=False)
rec.add("m11", ["p.bc"], files={"p.bc": b"BCX1" + bytes([2, 0]) + struct.pack("<HII", 0, 0, 0) + struct.pack("<I", 0)}, cmp=CMP)   # bad checksum, empty
a = Asm(); a.label("main"); rec.add("m11", ["p.bc"], files={"p.bc": a.build()}, cmp=CMP)        # empty program ok
rec.add("m11", ["--info", "p.bc"], files={"p.bc": Asm().build()}, cmp=CMP)
rec.add("m11", ["dir"], files={"dir/x": b"1"}, cmp=CMP)
for seed in range(8):
    b = bytearray(good); R2 = random.Random(seed)
    for _ in range(3): b[R2.randrange(len(b) - 4)] = R2.randrange(256)
    rec.add("m11", ["p.bc"], files={"p.bc": refix(bytes(b))}, cmp=CMP)

# ------------------------------------------------------------------ m12 budget
def spin():
    return prog(lambda a: a.label("l").JMP("l"))
for b in ("1", "5", "100", "12345"):
    add("m12", spin(), ["-b", b], pool=False)
add("m12", spin(), [], pool=False)
a = prog(lambda a: a.push(1).push(2).ADD().PRINT())      # exactly 4 instructions
for b in ("3", "4", "5"):
    add("m12", a, ["-b", b], pool=False)
add("m12", loop_sum(100000), ["-b", "5000000"], pool=False)
add("m12", loop_sum(100000), [], pool=False)
add("m12", loop_sum(50000), ["-b", "400000"], pool=False)
add("m12", loop_sum(50000), ["-b", "500000"], pool=False)
add("m12", fib(20), [], pool=False)
add("m12", fib(20), ["-b", "100000"], pool=False)
add("m12", prog(lambda a: a.push(1).PRINT().NOP().NOP()), ["-b", "2"], pool=False)
add("m12", prog(lambda a: a.label("l").str("x").PRINT().JMP("l")), ["-b", "50"], pool=False)

# ------------------------------------------------------------------ m13 obfuscated encodings
R3 = random.Random(5)
pool = [p for p in POOL]
R3.shuffle(pool)
n13 = 0
for a, args, stdin in pool[:34]:
    fl = R3.choice([1, 2, 3, 1, 3])
    add("m13", a, args, stdin=stdin, flags=fl, pool=False)
for fl in (1, 3):
    add("m13", fib(12), flags=fl, pool=False)
    add("m13", sieve(200), flags=fl, pool=False)
    add("m13", bubble([5, 3, 9, -2, 7, 7, 0]), flags=fl, pool=False)
    add("m13", prog(lambda a: a.push(2**40).push(3).MUL().PRINT().str("long string const pool entry").PRINT().push(1).push(0).DIV()), flags=fl, pool=False)

# ------------------------------------------------------------------ m14 random + stress
def rand_prog(seed):
    Rr = random.Random(seed)
    def f(a):
        d = 0
        for _ in range(Rr.randrange(15, 60)):
            c = Rr.random()
            if d < 2 or c < .3:
                a.push(Rr.choice(EDGE) if Rr.random() < .5 else Rr.randrange(-1000, 1000)); d += 1
            elif c < .75:
                op = Rr.choice(BIN + ["EQ", "NE", "LT", "LE", "GT", "GE"]); getattr(a, op)(); d -= 1
            elif c < .82: a.NEG()
            elif c < .88: a.NOT()
            elif c < .92: a.DUP(); d += 1
            elif c < .95 and d >= 3: a.ROT()
            elif c < .98: a.SWAP()
            else: a.OVER(); d += 1
        while d > 0: a.PRINT(); d -= 1
    return prog(f)
for s in range(36):
    add("m14", rand_prog(s), flags=0 if s % 4 else 1, pool=False)
def collatz(n):
    a = Asm(); a.label("main").push(n).STORE(0).push(0).STORE(1)
    a.label("L").LOAD(0).push(1).EQ().JNZ("E").LOAD(0).push(1).AND().JZ("ev").LOAD(0).push(3).MUL().push(1).ADD().STORE(0).JMP("c")
    a.label("ev").LOAD(0).push(1).SHR().STORE(0).label("c").LOAD(1).push(1).ADD().STORE(1).JMP("L").label("E").LOAD(1).PRINT(); return a
for n in (27, 97, 871, 6171):
    add("m14", collatz(n), pool=False)
def lcg_sort(n):
    a = Asm(); a.label("main").push(n).NEWARR().STORE(0).push(12345).STORE(1).push(0).STORE(2)
    a.label("f").LOAD(2).push(n).GE().JNZ("s").LOAD(1).push(1103515245).MUL().push(12345).ADD().push(2147483647).AND().STORE(1).LOAD(0).LOAD(2).LOAD(1).push(1000).MOD().ASTORE().LOAD(2).push(1).ADD().STORE(2).JMP("f")
    a.label("s").push(1).STORE(2)      # insertion sort
    a.label("o").LOAD(2).push(n).GE().JNZ("d").LOAD(0).LOAD(2).ALOAD().STORE(3).LOAD(2).STORE(4)
    a.label("w").LOAD(4).JZ("pl").LOAD(0).LOAD(4).push(1).SUB().ALOAD().LOAD(3).LE().JNZ("pl").LOAD(0).LOAD(4).LOAD(0).LOAD(4).push(1).SUB().ALOAD().ASTORE().LOAD(4).push(1).SUB().STORE(4).JMP("w")
    a.label("pl").LOAD(0).LOAD(4).LOAD(3).ASTORE().LOAD(2).push(1).ADD().STORE(2).JMP("o").label("d")
    a.push(0).STORE(2).push(0).STORE(5).label("ck").LOAD(2).push(n).GE().JNZ("pr").LOAD(5).push(31).MUL().LOAD(0).LOAD(2).ALOAD().ADD().STORE(5).LOAD(2).push(1).ADD().STORE(2).JMP("ck").label("pr").LOAD(5).PRINT(); return a
for n in (10, 100, 300):
    add("m14", lcg_sort(n), pool=False)
add("m14", prog(lambda a: a.label("l").str("s").POP().JMP("l")), pool=False)           # heap exhaustion (2048 objects)
add("m14", prog(lambda a: a.label("l").push(1024).NEWARR().POP().JMP("l")), pool=False)  # cell exhaustion
add("m14", prog(lambda a: a.label("l").push(100).NEWARR().POP().JMP("l")), pool=False)
def deep(n):
    a = Asm(); a.label("main").push(n).CALL("r", 1).PRINT().push(0).HALT()
    a.label("r").LOAD(0).JZ("z").LOAD(0).push(1).SUB().CALL("r", 1).push(1).ADD().RET().label("z").push(0).RET(); return a
for n in (31, 32, 33, 100):
    add("m14", deep(n), pool=False)
def strbuild(n):
    a = Asm(); a.label("main").str("").STORE(0).push(0).STORE(1)
    a.label("L").LOAD(1).push(n).GE().JNZ("E").LOAD(0).LOAD(1).ITOS().SCAT().STORE(0).LOAD(1).push(1).ADD().STORE(1).JMP("L").label("E").LOAD(0).SLEN().PRINT().LOAD(0).PRINT(); return a
for n in (50, 600, 1200):
    add("m14", strbuild(n), pool=False)
add("m14", prog(lambda a: [a.push(i).STORE(i % 16) for i in range(40)] and a.LOAD(3).PRINT()), pool=False)

# public samples for the agent: a few programs (plain and obfuscated) and their recorded output
os.makedirs(PUB, exist_ok=True)
samples = {"hello": prog(lambda a: a.str("hello, vm").PRINT().push(0).HALT()),
           "fib15": fib(15), "sieve100": sieve(100), "echo": prog(echo_lines), "collatz27": collatz(27)}
for name, a in samples.items():
    open(os.path.join(PUB, name + ".bc"), "wb").write(a.build(0))
open(os.path.join(PUB, "hello_obf.bc"), "wb").write(samples["hello"].build(3))
open(os.path.join(PUB, "fib15_obf.bc"), "wb").write(samples["fib15"].build(1))
print("cases", rec.save())
