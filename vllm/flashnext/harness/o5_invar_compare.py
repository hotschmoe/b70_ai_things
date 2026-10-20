#!/usr/bin/env python3
"""O5: cross-arm module-hash comparison (MTP off vs on), by row-0 position.

  python3 o5_invar_compare.py hashes-p0.json hashes-p2.json [run=solo]

For every step of arm B whose row 0 sits at a position that arm A also
decoded as row 0, compare the row-0 hash of each module call in order.
Arm A's step at that position is a T=1 (or T=nreq) decode; arm B's is a
verify of 1+k rows. Prints, per shared position, the first differing
module (or "same") and a census of differing module kinds.
"""

from __future__ import annotations

import collections
import json
import re
import sys


def steps(calls):
    out = collections.defaultdict(list)
    for c in calls:
        out[c["step"]].append(c)
    return out


def main() -> int:
    a = json.load(open(sys.argv[1]))
    b = json.load(open(sys.argv[2]))
    run = sys.argv[3] if len(sys.argv) > 3 else "solo"
    sa, sb = steps(a[run]["calls"]), steps(b[run]["calls"])
    by_pos_a = {v[0]["pos0"]: (s, v) for s, v in sa.items() if v}
    kinds = collections.Counter()
    for s, v in sorted(sb.items()):
        if not v:
            continue
        p = v[0]["pos0"]
        if p not in by_pos_a:
            continue
        sa_i, va = by_pos_a[p]
        first = None
        ndiff = 0
        for x, y in zip(va, v):
            if x["mod"] != y["mod"]:
                first = first or f"order {x['mod']} vs {y['mod']}"
                break
            if x["h0"] != y["h0"]:
                ndiff += 1
                kinds[re.sub(r"layers\.\d+\.", "layers.N.", x["mod"])] += 1
                if first is None:
                    first = f"{x['mod']} (nt {x['nt']} vs {y['nt']})"
        orig = [x["mod"] for x, y in zip(va, v) if x["mod"] == y["mod"] and x["h0"] != y["h0"]
                and x.get("hin") is not None and x.get("hin") == y.get("hin")]
        if orig:
            print(f"  originating (same input, different output) at pos {p}: {orig[:12]}")
        print(f"pos {p}: A step {sa_i} nt={va[0]['nt']} vs B step {s} nt={v[0]['nt']}: "
              f"first={first or 'same'} ndiff={ndiff}/{min(len(va), len(v))}")
    print("differing module kinds:")
    for k, n in kinds.most_common(40):
        print(f"  {n:5d} {k}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
