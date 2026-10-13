#!/usr/bin/env python3
"""Hidden checker for spec-sql (runs in the verifier container). Each case runs in its own python process (timeout)."""
import json, os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from mscore import Milestones

SUB, HID = sys.argv[1], os.path.dirname(os.path.abspath(__file__))
MS = [("s01", "expressions without tables: arithmetic, 3-valued logic, CASE, params"), ("s02", "CREATE/INSERT/basic SELECT, defaults, affinity, PK auto ids"),
      ("s03", "WHERE predicates: IN, BETWEEN, LIKE, IS NULL"), ("s04", "ORDER BY, LIMIT/OFFSET, DISTINCT, NULL ordering"),
      ("s05", "scalar functions"), ("s06", "aggregates without GROUP BY"), ("s07", "GROUP BY / HAVING"),
      ("s08", "UPDATE / DELETE / INSERT..SELECT with rowcounts"), ("s09", "constraints, error codes, statement atomicity"),
      ("s10", "BEGIN / COMMIT / ROLLBACK incl. DDL"), ("s11", "INNER / LEFT joins, self joins, derived tables"),
      ("s12", "uncorrelated subqueries, EXISTS, UNION [ALL]"), ("s13", "correlated subqueries (SELECT/UPDATE/DELETE)"),
      ("s14", "on-disk log format: reopen, torn tail, rollback, interop with the reference writer"), ("s15", "command line interface output format")]
cases = json.load(open(os.path.join(HID, "cases.json")))
groups = {}
for c in cases:
    groups.setdefault(c["ms"], []).append(c)


def one(c):
    try:
        p = subprocess.run([sys.executable, os.path.join(HID, "lib", "sqlcase.py"), SUB], input=json.dumps(c).encode(),
                           capture_output=True, timeout=30)
        return p.stdout.decode(errors="replace").strip().startswith("OK")
    except subprocess.TimeoutExpired:
        return False


M = Milestones()
import glob, re
src_files = [f for f in glob.glob(os.path.join(SUB, "*.py")) + glob.glob(os.path.join(SUB, "minisql", "**", "*.py"), recursive=True)]
if any(re.search(r"(?m)^\s*(import|from)\s+(sqlite3|apsw|duckdb)\b|__import__\(.(sqlite3|apsw|duckdb)|import_module\(.(sqlite3|apsw|duckdb)", open(f, errors="replace").read()) for f in src_files):
    for mid, name in MS:
        M.skip(mid, name + " (disqualified: delegates to an existing SQL engine)", len(groups.get(mid, [])))
    M.emit(); sys.exit(0)
with ThreadPoolExecutor(4) as ex:
    for mid, name in MS:
        M.add(mid, name, list(ex.map(one, groups.get(mid, []))))
M.emit()
