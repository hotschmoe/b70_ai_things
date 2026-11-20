#!/usr/bin/env python3
"""DFlash2 acceptance from a tensorfold-native serve log: 'done ... tokens=256 ... rounds=R accepted=A/B ...' lines in request order.
Segments (run_bench_c.sh with levels 1,8): 12 bench_openai requests, 1 spread n=1, 8 spread n=8. usage: accept.py LOG"""
import re, sys
rows = []
for l in open(sys.argv[1], errors="ignore"):
    m = re.search(r"\] done .*? tokens=(\d+) .*?rounds=(\d+) accepted=(\d+)/(\d+) ", l)
    if m and int(m.group(1)) == 256: rows.append(tuple(map(int, m.groups())))
def seg(name, rs):
    if not rs: return
    t = sum(r[0] for r in rs); rounds = sum(r[1] for r in rs); a = sum(r[2] for r in rs); b = sum(r[3] for r in rs)
    print(f"{name}: requests {len(rs)} tokens/round {t/rounds:.2f} accepted/proposed {a}/{b} = {a/b:.3f}")
seg("bench_openai (4 cells x3, greedy+T1)", rows[:12]); seg("spread n=1", rows[12:13]); seg("spread n=8", rows[13:21]); seg("all 256-token requests", rows)
