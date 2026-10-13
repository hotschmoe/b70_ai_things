"""python twin of check.sh: from check import check, finish ; check(name, lambda: bool|None) (exceptions = FAIL)."""
import sys
_n = {"p": 0, "f": 0}
def check(name, fn):
    try:
        r = fn()
        ok = r is None or bool(r)
    except Exception as e:
        ok = False; print(f"    | {type(e).__name__}: {e}")
    print(f"CHECK {name} {'PASS' if ok else 'FAIL'}"); _n["p" if ok else "f"] += 1
def finish():
    ok = _n["f"] == 0 and _n["p"] > 0
    print("RESULT PASS" if ok else "RESULT FAIL"); sys.exit(0 if ok else 1)
