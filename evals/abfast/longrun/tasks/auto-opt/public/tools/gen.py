#!/usr/bin/env python3
"""Generate an nnjoin input: python3 tools/gen.py N SEED [VOCAB] [MINLEN] [MAXLEN] > file
Documents draw tokens from a Zipf-like vocabulary; ~30% of documents are noisy copies of earlier ones
(so near-duplicates and similarity ties exist). Real grading inputs are produced by the same kind of
generator with other seeds, sizes and parameters."""
import bisect, random, sys


def gen(n, seed, vocab=30000, lo=5, hi=60, zipf=1.1, dup=0.3):
    r = random.Random(seed)
    cum, s = [], 0.0
    for k in range(1, vocab + 1):
        s += 1.0 / k ** zipf
        cum.append(s)
    draw = lambda: bisect.bisect_left(cum, r.random() * s) * 7 + 3        # sparse ids
    docs = []
    for i in range(n):
        if docs and r.random() < dup:
            d = list(r.choice(docs))
            for _ in range(r.randrange(0, max(2, len(d) // 5))):
                if d and r.random() < 0.5:
                    d.pop(r.randrange(len(d)))
                else:
                    d.append(draw())
        else:
            d = [draw() for _ in range(r.randrange(lo, hi + 1))]
        if r.random() < 0.3:
            d += [r.choice(d) for _ in range(r.randrange(0, 4))] if d else []   # repeated tokens
        docs.append(d)
    return docs


def render(docs):
    return str(len(docs)) + "\n" + "\n".join(" ".join(map(str, d)) for d in docs) + "\n"


if __name__ == "__main__":
    a = sys.argv[1:]
    n, seed = int(a[0]), int(a[1])
    kw = {}
    if len(a) > 2: kw["vocab"] = int(a[2])
    if len(a) > 3: kw["lo"] = int(a[3])
    if len(a) > 4: kw["hi"] = int(a[4])
    sys.stdout.write(render(gen(n, seed, **kw)))
