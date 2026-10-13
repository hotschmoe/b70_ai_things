#!/usr/bin/env python3
"""Runs INSIDE the build container. Creates hidden inputs + trusted expected outputs.
  gen_hidden.py HIDDEN_DIR PUBLIC_DIR
Expected outputs come from an independent exact-fraction oracle (python sets + inverted index), cross-checked
against the naive Rust program on every small/edge/medium input by build.py."""
import json, os, random, sys
from fractions import Fraction
sys.path.insert(0, "/src/public/tools")
import gen as G

HID, PUB = sys.argv[1], sys.argv[2]
os.makedirs(f"{HID}/inputs", exist_ok=True)


def oracle(docs):
    n = len(docs)
    sets = [set(d) for d in docs]
    post = {}
    for i, s in enumerate(sets):
        for t in s:
            post.setdefault(t, []).append(i)
    out = []
    for i in range(n):
        if n == 1:
            out.append("-1 0 0"); continue
        cnt = {}
        for t in sets[i]:
            for j in post[t]:
                if j != i:
                    cnt[j] = cnt.get(j, 0) + 1
        best = None
        for j, c in cnt.items():
            u = len(sets[i]) + len(sets[j]) - c
            key = (Fraction(c, u), -j)
            if best is None or key > best[0]:
                best = (key, j, c, u)
        if best is None:
            j = 0 if i != 0 else 1
            out.append(f"{j} 0 {len(sets[i]) + len(sets[j])}")
        else:
            out.append(f"{best[1]} {best[2]} {best[3]}")
    return ("\n".join(out) + "\n") if out else ""


meta = {"groups": {"small": [], "edge": [], "medium": [], "large": [], "scale": []}}


def put(group, name, docs, raw=None):
    text = raw if raw is not None else G.render(docs)
    open(f"{HID}/inputs/{name}.in", "w").write(text)
    if raw is None:
        exp = oracle(docs)
    else:
        # raw (odd formatting) inputs: parse like the spec
        lines = raw.split("\n"); n = int(lines[0].strip() or 0)
        dd = [[int(x) for x in (lines[1 + i] if 1 + i < len(lines) else "").split()] for i in range(n)]
        exp = oracle(dd)
    open(f"{HID}/inputs/{name}.out", "w").write(exp)
    meta["groups"][group].append(name)


R = random.Random(777000111)
for i in range(25):
    put("small", f"small{i}", G.gen(R.randrange(2, 14), R.randrange(10**9), vocab=R.choice([5, 12, 40, 500]), lo=0, hi=R.choice([3, 6, 12]), dup=R.choice([0, .3, .6])))
# edge cases
put("edge", "e_n1", [[1, 2, 3]])
put("edge", "e_n2", [[1], [2]])
put("edge", "e_n2_same", [[5, 6], [5, 6]])
put("edge", "e_all_empty", [[] for _ in range(6)])
put("edge", "e_empty_mix", [[], [1, 2], [], [2, 3], [1, 2, 3], []])
put("edge", "e_identical", [[1, 2, 3, 4]] * 9)
put("edge", "e_ties", [[1, 2], [2, 3], [3, 1], [1, 2, 3], [4], [5], [4, 5], [4, 5]])
put("edge", "e_repeats", [[7, 7, 7, 8], [7, 8, 8], [8, 7], [9, 9, 9]])
put("edge", "e_big_ids", [[4294967295, 0, 17], [4294967295, 4294967294], [0, 17, 4294967294], [123456789]])
put("edge", "e_disjoint", [[i * 3, i * 3 + 1] for i in range(10)])
put("edge", "e_single_token", [[1], [1], [1], [2], [2], [3]])
put("edge", "e_subset_chain", [list(range(k)) for k in range(1, 12)])
put("edge", "e_zero_n", [], raw="0\n")
put("edge", "e_no_trailing_nl", [[1, 2], [2, 3], [1, 3]], raw="3\n1 2\n2 3\n1 3")
put("edge", "e_missing_lines", [[1, 2], [2, 3], [], []], raw="4\n1 2\n2 3\n")
put("edge", "e_extra_spaces", [[1, 2], [2, 3]], raw="2\n  1   2 \n2\t3\n\n")
put("edge", "e_fraction_cmp", [[1, 2, 3], [1, 2, 3, 4, 5, 6], [1, 2, 3, 7, 8, 9], [10, 11], [10, 11, 12, 13, 14, 15, 16], [1, 2]])
put("edge", "e_crlf_free_long", [list(range(i, i + 400)) for i in range(0, 60, 3)])
put("edge", "e_zero_overlap_first", [[1, 2], [3, 4], [5, 6], [7, 8], [1, 9]])
put("edge", "e_many_empty_one_pair", [[] for _ in range(8)] + [[1, 2], [1, 2]])
for i in range(3):
    put("medium", f"medium{i}", G.gen(300 + 60 * i, R.randrange(10**9), vocab=[3000, 30000, 300][i]))
put("large", "large0", G.gen(2500, R.randrange(10**9), vocab=30000, lo=10, hi=100))
put("scale", "scale0", G.gen(10000, R.randrange(10**9), vocab=60000, lo=6, hi=18))
json.dump(meta, open(f"{HID}/meta.json", "w"))
print({k: len(v) for k, v in meta["groups"].items()})
