#!/usr/bin/env python3
"""Differential fuzz: rx reference vs python re.   fuzz.py RX_BINARY N [seed]"""
import os, random, subprocess, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import oracle_re as O

RX, N = sys.argv[1], int(sys.argv[2])
R = random.Random(int(sys.argv[3]) if len(sys.argv) > 3 else 1)
ATOMS = ["a", "b", "c", "x", "ab", ".", "\\d", "\\w", "\\s", "\\W", "[abc]", "[^a]", "[a-c]", "[\\d_]", "\\.", "-", " ", "\\n", "1", "Q"]
ANCH = ["^", "$", "\\b", "\\B", "\\A", "\\Z"]
QS = ["*", "+", "?", "{2}", "{1,2}", "{0,3}", "{2,}", "*?", "+?", "??", "{1,2}?", "*+", "++", "?+"]
SUBJ = "ab c.\n_1xxAB-abcabc  Q"


def pat(d=0):
    k = R.random()
    if k < .45 or d > 3:
        a = R.choice(ATOMS)
        if R.random() < .4: a += R.choice(QS)
        return a
    if k < .55: return R.choice(ANCH)
    if k < .7: return pat(d + 1) + pat(d + 1)
    if k < .8: return pat(d + 1) + "|" + pat(d + 1)
    if k < .87: return "(" + pat(d + 1) + ")" + (R.choice(QS) if R.random() < .5 else "")
    if k < .9: return "(?:" + pat(d + 1) + ")" + (R.choice(QS) if R.random() < .5 else "")
    if k < .93: return R.choice(["(?=", "(?!", "(?>"]) + pat(d + 1) + ")"
    if k < .95: return R.choice(["(?<=", "(?<!"]) + R.choice(["a", "ab", "[ab]", "\\d\\d", "a|b", "x"]) + ")" + pat(d + 1)
    if k < .97: return "(?P<n%d>" % R.randrange(3) + pat(d + 1) + ")"
    return "\\%d" % R.randrange(1, 4)


bad = 0
for it in range(N):
    p = pat()
    fl = "".join(c for c in "imsx" if R.random() < .15)
    subj = "".join(R.choice(SUBJ) for _ in range(R.randrange(0, 14)))
    kind = R.choice(["match", "findall", "sub"])
    tmpl = R.choice(["<\\1>", "-", "[\\g<0>]", "", "\\n"])
    try:
        if kind == "match": exp = O.match_line(p, fl, subj)
        elif kind == "findall": exp = O.findall_line(p, fl, subj)
        else: exp = O.sub_line(p, fl, tmpl, subj)
    except RuntimeError as e:
        print("ORACLE-UNMAPPED", p, e); continue
    lines = [p, fl] + ([tmpl] if kind == "sub" else []) + [O.enc(subj)]
    with tempfile.NamedTemporaryFile("w", delete=False) as f:
        f.write("\n".join(lines) + "\n")
    try:
        got = subprocess.run([RX, kind, f.name], capture_output=True, text=True, timeout=10).stdout.rstrip("\n")
    except subprocess.TimeoutExpired:
        got = "TIMEOUT"
    os.unlink(f.name)
    if got != exp:
        bad += 1
        print(f"{kind} pat={p!r} fl={fl!r} tmpl={tmpl!r} subj={subj!r}\n   exp={exp}\n   got={got}")
print("mismatches", bad, "of", N)
