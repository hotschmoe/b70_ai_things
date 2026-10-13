"""Case runner for the spec-sql task (used by the hidden checker AND the public runner).

A case (JSON dict):
  {"id","ms","kind":"steps"|"persist"|"cli", ...}
 kind steps  : {"steps":[{"sql","params","expect":{"err":code} | {"rows":[[..]],"ordered":bool,"cols":[..]?,"rowcount":n?}}]}
 kind persist: {"files":{name:text}, "phases":[{"steps":[..]}], "dump":{"tables":{name:rows}}}   db file is ./db.log
 kind cli    : {"files":{}, "runs":[{"db":"db.log"|":memory:", "script":text, "stdout":text}]}
Run as:  python3 sqlcase.py SUBMISSION_DIR < case.json   -> prints "OK" or "FAIL reason", exit 0.
"""
import json, os, sqlite3, subprocess, sys, tempfile


def _norm(v):
    return v


def _same_val(a, b):
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, bool) or isinstance(b, bool):
        return False
    if type(a) is not type(b):
        return False
    if isinstance(a, float):
        return abs(a - b) <= 1e-9 * max(1.0, abs(a), abs(b))
    return a == b


def _key(v):
    if v is None:
        return (-1, 0)
    if isinstance(v, (int, float)):
        return (0, round(float(v), 6))
    return (1, v)


def rows_equal(exp, got, ordered):
    if len(exp) != len(got):
        return False, f"row count {len(got)} != {len(exp)}"
    got = [list(r) for r in got]
    exp = [list(r) for r in exp]
    if not ordered:
        k = lambda r: [_key(x) for x in r]
        exp = sorted(exp, key=k); got = sorted(got, key=k)
    for i, (e, g) in enumerate(zip(exp, got)):
        if len(e) != len(g) or not all(_same_val(x, y) for x, y in zip(e, g)):
            return False, f"row {i}: got {g!r} expected {e!r}"
    return True, ""


def run_steps(db, steps, minisql):
    for n, st in enumerate(steps):
        ex = st["expect"]
        try:
            r = db.execute(st["sql"], tuple(st.get("params") or ()))
            err = None
        except minisql.SQLError as e:
            r, err = None, e.code
        except Exception as e:  # any other exception is a failure
            return False, f"step {n} {st['sql'][:60]!r}: unexpected {type(e).__name__}: {e}"
        if "err" in ex:
            if err != ex["err"]:
                return False, f"step {n} {st['sql'][:60]!r}: error {err!r}, expected {ex['err']!r}"
            continue
        if err is not None:
            return False, f"step {n} {st['sql'][:60]!r}: unexpected error {err!r}"
        if "rows" in ex:
            ok, why = rows_equal(ex["rows"], r.rows, ex.get("ordered", False))
            if not ok:
                return False, f"step {n} {st['sql'][:60]!r}: {why}"
            if ex.get("cols") is not None and list(r.columns) != ex["cols"]:
                return False, f"step {n}: columns {list(r.columns)!r} != {ex['cols']!r}"
        if "rowcount" in ex and r.rowcount != ex["rowcount"]:
            return False, f"step {n} {st['sql'][:60]!r}: rowcount {r.rowcount} != {ex['rowcount']}"
    return True, ""


def check_log(path, dump):
    """Independent check of the on-disk format: header, JSON lines, replay through sqlite3."""
    data = open(path, "rb").read().decode("utf-8")
    if not data.startswith("MINISQL-LOG 1\n"):
        return False, "bad header"
    if not data.endswith("\n"):
        return False, "file does not end with newline"
    con = sqlite3.connect(":memory:", isolation_level=None)
    for ln in data.split("\n")[1:-1]:
        try:
            obj = json.loads(ln)
            assert list(obj) == ["txn"] and isinstance(obj["txn"], list) and obj["txn"]
            for sql, params in obj["txn"]:
                assert isinstance(sql, str) and isinstance(params, list)
        except Exception:
            return False, f"bad log line {ln[:60]!r}"
        con.execute("BEGIN")
        for sql, params in obj["txn"]:
            con.execute(sql, params)
        con.execute("COMMIT")
    for t, exp in (dump or {}).items():
        got = [list(r) for r in con.execute(f"SELECT * FROM {t} ORDER BY rowid")]
        ok, why = rows_equal(exp, got, True)
        if not ok:
            return False, f"log replay table {t}: {why}"
    return True, ""


def run_case(case, sub):
    sys.path.insert(0, sub)
    import minisql
    d = tempfile.mkdtemp(prefix="sqlcase")
    os.chdir(d)
    for name, text in (case.get("files") or {}).items():
        open(name, "w", encoding="utf-8", newline="").write(text)
    kind = case.get("kind", "steps")
    if kind == "steps":
        db = minisql.Database()
        return run_steps(db, case["steps"], minisql)
    if kind == "persist":
        for i, ph in enumerate(case["phases"]):
            db = minisql.Database("db.log")
            ok, why = run_steps(db, ph["steps"], minisql)
            db.close()
            if not ok:
                return False, f"phase {i}: {why}"
        if case.get("dump") is not None:
            return check_log("db.log", case["dump"]["tables"])
        return True, ""
    if kind == "cli":
        for i, run in enumerate(case["runs"]):
            p = subprocess.run([sys.executable, os.path.join(sub, "minisql.py"), run["db"]], input=run["script"].encode(),
                               capture_output=True, timeout=30)
            if p.stdout.decode(errors="replace") != run["stdout"]:
                return False, f"run {i}: stdout {p.stdout.decode(errors='replace')[:200]!r} != {run['stdout'][:200]!r}"
        return True, ""
    return False, "bad case kind"


def main():
    sub = sys.argv[1]
    case = json.load(sys.stdin)
    try:
        ok, why = run_case(case, sub)
    except Exception as e:
        ok, why = False, f"{type(e).__name__}: {e}"
    print("OK" if ok else "FAIL " + why)


if __name__ == "__main__":
    main()
