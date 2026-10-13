"""Helpers for hidden milestone checkers (run INSIDE the verifier container).

A checker builds groups of boolean test results and prints one final line:
  RESULT {"milestones":[{"id","name","passed","total","ok"}], "tests_passed":N, "tests_total":M}
A milestone is ok when passed/total >= pass_frac (default 1.0).
"""
import json, os, subprocess, sys, tempfile


def run(cmd, stdin=None, timeout=10, cwd=None, env=None):
    """Run a command; returns (rc, stdout_bytes, stderr_bytes). rc=-9 on timeout, -1 on exec failure."""
    if isinstance(stdin, str):
        stdin = stdin.encode()
    try:
        p = subprocess.run(cmd, input=stdin, capture_output=True, timeout=timeout, cwd=cwd, env=env)
        return p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired:
        return -9, b"", b"TIMEOUT"
    except OSError as e:
        return -1, b"", repr(e).encode()


class Milestones:
    def __init__(self):
        self.ms = []

    def add(self, mid, name, results, pass_frac=1.0):
        results = list(results)
        p = sum(1 for r in results if r)
        t = len(results)
        ok = t > 0 and p / t >= pass_frac
        self.ms.append({"id": mid, "name": name, "passed": p, "total": t, "ok": ok})

    def skip(self, mid, name, total):
        """Milestone that cannot be evaluated (e.g. build failed): counts as total failures."""
        self.ms.append({"id": mid, "name": name, "passed": 0, "total": total, "ok": False})

    def emit(self):
        out = {"milestones": self.ms,
               "tests_passed": sum(m["passed"] for m in self.ms),
               "tests_total": sum(m["total"] for m in self.ms)}
        print("RESULT " + json.dumps(out), flush=True)


def tmpdir():
    return tempfile.mkdtemp(prefix="chk")


# ---------------------------------------------------------------------------------------------
# Generic recorded-CLI cases. A case is a JSON dict:
#   {"id": str, "ms": milestone id, "args": [..], "files": {relpath: sha256}, "stdin": sha256|None,
#    "env": {..}|None,
#    "expect": {"rc": int, "stdout": sha256, "stderr": sha256|None, "out_files": {relpath: sha256|None}},
#    "cmp": ["rc","stdout","stderr","out_files"]}   # which parts must match (default rc+stdout+out_files)
# Blobs live content-addressed in HIDDEN/data/<sha256>. args may contain "{DIR}" = case working dir.
# ---------------------------------------------------------------------------------------------
import hashlib, shutil


def blob(hidden, h):
    return open(os.path.join(hidden, "data", h), "rb").read()


def sha(b):
    return hashlib.sha256(b).hexdigest()


def put_blob(hidden, b):
    h = sha(b)
    p = os.path.join(hidden, "data", h)
    if not os.path.exists(p):
        os.makedirs(os.path.dirname(p), exist_ok=True)
        open(p, "wb").write(b)
    return h


def prepare_dir(hidden, case, root=None):
    d = tempfile.mkdtemp(prefix="case", dir=root)
    for rel, h in (case.get("files") or {}).items():
        p = os.path.join(d, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        open(p, "wb").write(blob(hidden, h))
        os.chmod(p, (case.get("modes") or {}).get(rel, 0o644))
    return d


def run_case(argv0, hidden, case, timeout=20):
    """Run argv0 (list) with the case; returns (ok, why)."""
    d = prepare_dir(hidden, case)
    try:
        args = [a.replace("{DIR}", d) for a in case["args"]]
        stdin = blob(hidden, case["stdin"]) if case.get("stdin") else None
        env = dict(os.environ); env.update(case.get("env") or {})
        rc, out, err = run(list(argv0) + args, stdin=stdin, timeout=timeout, cwd=d, env=env)
        return compare_case(hidden, case, d, rc, out, err)
    finally:
        shutil.rmtree(d, ignore_errors=True)


def compare_case(hidden, case, d, rc, out, err):
    ex = case["expect"]
    cmp = case.get("cmp") or ["rc", "stdout", "out_files"]
    if "rc" in cmp and rc != ex["rc"]:
        return False, f"rc {rc} != {ex['rc']}"
    if "stdout" in cmp and sha(out) != ex["stdout"]:
        return False, "stdout differs"
    if "stderr" in cmp and ex.get("stderr") and sha(err) != ex["stderr"]:
        return False, "stderr differs"
    if "out_files" in cmp:
        for rel, h in (ex.get("out_files") or {}).items():
            p = os.path.join(d, rel)
            if h is None:
                if os.path.exists(p):
                    return False, f"{rel} should not exist"
                continue
            if not os.path.isfile(p):
                return False, f"{rel} missing"
            if sha(open(p, "rb").read()) != h:
                return False, f"{rel} differs"
    return True, ""


def cli_milestones(M, argv0, hidden, ms_defs, timeout=20, workers=4, skip=()):
    """Evaluate recorded cases grouped by milestone id. ms_defs: [(id, name)]. Milestones in skip are left to the caller."""
    from concurrent.futures import ThreadPoolExecutor
    cases = json.load(open(os.path.join(hidden, "cases.json")))
    groups = {}
    for c in cases:
        groups.setdefault(c["ms"], []).append(c)
    with ThreadPoolExecutor(workers) as ex:
        for mid, name in ms_defs:
            if mid in skip:
                continue
            res = list(ex.map(lambda c: run_case(argv0, hidden, c, timeout)[0], groups.get(mid, [])))
            M.add(mid, name, res)
