#!/usr/bin/env python3
"""O5: compare token ids between two o5_mtp_profile.py arms.

  python3 o5_compare.py <ref tokens-*.json> <arm tokens-*.json> [...]

Reports, per phase and prompt, whether the ids are identical and the first
differing index. Also checks C1 vs C2 inside each arm (batch invariance
across rows/T without MTP). Exit 0 only if everything matches.
"""

from __future__ import annotations

import json
import sys


def first_diff(a: list[int], b: list[int]) -> int | None:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def by_prompt(res: dict, phase: str) -> dict[str, list[list[int]]]:
    out: dict[str, list[list[int]]] = {}
    for entry in res[phase]:
        for p, ids in zip(entry["prompts"], entry["ids"]):
            out.setdefault(p, []).append(ids)
    return out


def main() -> int:
    arms = [json.load(open(p)) for p in sys.argv[1:]]
    ok = True
    for arm in arms:
        c1 = by_prompt(arm, "c1")
        c2 = by_prompt(arm, "c2")
        for p, seqs in c2.items():
            for s in seqs:
                d = first_diff(c1[p][0], s)
                ok &= d is None
                print(f"{arm['tag']}: c1 vs c2 '{p[:32]}' diff_at={d}")
    ref = arms[0]
    for arm in arms[1:]:
        for phase in ("c1", "c2"):
            r = by_prompt(ref, phase)
            a = by_prompt(arm, phase)
            for p in r:
                for i, (x, y) in enumerate(zip(r[p], a.get(p, []))):
                    d = first_diff(x, y)
                    ok &= d is None
                    print(f"{ref['tag']} vs {arm['tag']} {phase} '{p[:32]}'#{i} diff_at={d} n={len(x)}")
    print("PARITY", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
