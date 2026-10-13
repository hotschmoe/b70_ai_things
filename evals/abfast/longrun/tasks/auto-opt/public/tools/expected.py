#!/usr/bin/env python3
"""Slow-but-simple oracle for small inputs: python3 tools/expected.py INPUT > expected_output
(same semantics as the original nnjoin; use it to check your optimized program)."""
import sys
from fractions import Fraction


def solve(text):
    lines = text.split("\n")
    n = int(lines[0].strip() or 0)
    docs = []
    for i in range(n):
        ln = lines[1 + i] if 1 + i < len(lines) else ""
        docs.append(set(ln.split()))
    out = []
    for i in range(n):
        best = None
        for j in range(n):
            if j == i:
                continue
            inter = len(docs[i] & docs[j]); union = len(docs[i] | docs[j])
            sim = Fraction(inter, union) if union else Fraction(0)
            if best is None or sim > best[0]:
                best = (sim, j, inter, union)
        out.append("-1 0 0" if best is None else f"{best[1]} {best[2]} {best[3]}")
    return "\n".join(out) + ("\n" if out else "")


if __name__ == "__main__":
    sys.stdout.write(solve(open(sys.argv[1]).read()))
