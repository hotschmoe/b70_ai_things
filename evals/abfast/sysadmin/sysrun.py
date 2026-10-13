"""abfast sysadmin tier: per-task setup / hidden verify / no-model validation.

Hooks used by run_arm.py (all keyed on task["sys"]):
  tier_defaults(a, here)           after arg parsing: tasks file + timeout/max-output defaults for --tier sysadmin
  make_box(Box, name, image, task) container with --init, bridge network (not host), per-task docker_args (tmpfs)
  prepare(box, task, d)            run tasks/<id>/setup/setup.sh as root, then delete every trace of it
  verify(box, task, d)             copy verify bundle in AFTER the agent stopped, run it, parse CHECK lines
  validate_task(task, image, tag)  no model: verify FAILS after setup, PASSES after fix.sh, FAILS after lazy.sh; 2 rounds
Verifier protocol: verify.sh sources lib/check.sh and prints "CHECK <name> PASS|FAIL" lines and "RESULT PASS|FAIL".
Names starting with safe_ are non-destruction checks. Verify, fix and lazy scripts never exist in the
agent's container during the agent phase.
"""
import io, json, os, re, shlex, subprocess, tarfile, time

HERE = os.path.dirname(os.path.abspath(__file__))
TASKS = os.path.join(HERE, "tasks")
DATA = "/mnt/vm_8tb/b70/evals/abfast"
BOXES = "/opt/.abfast"


def tier_defaults(a, here):
    if getattr(a, "tier", "") != "sysadmin":
        return
    if a.tasks_file is None:
        a.tasks_file = os.path.join(here, "tasks_sysadmin.json")
    if a.timeout_s is None:
        a.timeout_s = 1800
    if a.max_output_tokens is None:
        a.max_output_tokens = 8192 if a.thinking == "off" else 32768


def make_box(Box, name, image, task):
    class SysBox(Box):
        def __init__(self, name, image):
            self.name = name
            cmd = ["docker", "run", "-d", "--name", name, "--init", "--hostname", task.get("hostname", "officepc"),
                   "--memory", "12g", "--cpus", "6", "--pids-limit", "4096"] + list(task.get("docker_args", [])) + \
                  [image, "sleep", "infinity"]
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode:
                raise RuntimeError(r.stderr)
    return SysBox(name, image)


def _root(ti):
    ti.uid = ti.gid = 0; ti.uname = ti.gname = "root"
    return ti


def _tar(task, dirs, extra=None):
    buf = io.BytesIO()
    base = os.path.join(TASKS, task["id"])
    with tarfile.open(fileobj=buf, mode="w") as tf:
        for d in dirs:
            p = os.path.join(base, d)
            if os.path.exists(p):
                tf.add(p, arcname=d, filter=_root)
        for src, arc in (extra or {}).items():
            tf.add(src, arcname=arc, filter=_root)
    return buf.getvalue()


def _run_dir(box, task, dirs, dest, script, timeout, log_path=None, extra=None):
    """put a bundle into the container, run dest/script, return (rc, output). Caller removes dest."""
    box.put_tar(_tar(task, dirs, extra), dest)
    try:
        rc, out = box.exec(f"cd {dest} && bash {script}", timeout=timeout)
    except subprocess.TimeoutExpired:
        rc, out = 124, "TIMEOUT"
    if log_path:
        open(log_path, "w").write(out)
    return rc, out


def prepare(box, task, d):
    rc, out = _run_dir(box, task, ["setup", "shared"], BOXES + "-setup", "setup/setup.sh", 600, f"{d}/setup.log")
    box.exec(f"rm -rf {BOXES}-setup /tmp/.abfast* ; history -c 2>/dev/null; true")
    if rc:
        raise RuntimeError(f"setup failed rc={rc}: {out[-300:]}")


def parse_checks(out):
    checks = {m.group(1): m.group(2) == "PASS" for m in re.finditer(r"^CHECK (\S+) (PASS|FAIL)", out, re.M)}
    res = re.findall(r"^RESULT (PASS|FAIL)", out, re.M)
    return checks, (res[-1] == "PASS" if res else False)


def verify(box, task, d):
    check = {os.path.join(HERE, "lib", "check.sh"): "verify/check.sh", os.path.join(HERE, "lib", "check.py"): "verify/check.py"}
    rc, out = _run_dir(box, task, ["verify", "shared"], BOXES + "-verify", "verify/verify.sh", 600, f"{d}/verify.log", check)
    checks, ok = parse_checks(out)
    json.dump(checks, open(f"{d}/verify_checks.json", "w"), indent=1)
    p = sum(checks.values()); t = len(checks)
    return {"rc": rc, "passed": p, "total": t, "pass": ok and rc == 0 and t > 0 and p == t}


def _fixscript(box, task, which):
    rc, out = _run_dir(box, task, [which + ".sh"], BOXES + "-fix", which + ".sh", 600)
    box.exec(f"rm -rf {BOXES}-fix")
    return rc, out


def validate_task(task, image, tag):
    d = f"{DATA}/runs/validate-sys/{task['id']}"
    os.makedirs(d, exist_ok=True)
    out = {"rounds": []}
    for rnd in (1, 2):
        r = {}
        for state in ("setup_only", "ref_fix", "lazy"):
            if state == "lazy" and not os.path.exists(os.path.join(TASKS, task["id"], "lazy.sh")):
                continue
            box = make_box(_box_cls(), f"abfast-val-{tag}-{task['id']}-{state}-{rnd}"[:100], image, task)
            try:
                prepare(box, task, f"{d}")
                if state == "ref_fix":
                    rc, o = _fixscript(box, task, "fix")
                    open(f"{d}/fix_r{rnd}.log", "w").write(o)
                elif state == "lazy":
                    rc, o = _fixscript(box, task, "lazy")
                    open(f"{d}/lazy_r{rnd}.log", "w").write(o)
                v = verify(box, task, d)
                os.replace(f"{d}/verify.log", f"{d}/verify_{state}_r{rnd}.log")
                r[state] = {"pass": v["pass"], "checks": f"{v['passed']}/{v['total']}"}
                json.dump(json.load(open(f"{d}/verify_checks.json")), open(f"{d}/checks_{state}_r{rnd}.json", "w"))
            finally:
                box.rm()
        out["rounds"].append(r)
    has_lazy = "lazy" in out["rounds"][0]
    out["setup_fails_verify"] = all(not r["setup_only"]["pass"] for r in out["rounds"])
    out["ref_fix_passes"] = all(r["ref_fix"]["pass"] for r in out["rounds"])
    out["lazy_fix_fails"] = all(not r["lazy"]["pass"] for r in out["rounds"]) if has_lazy else None
    out["deterministic"] = all(out["rounds"][0][k] == out["rounds"][1][k] for k in out["rounds"][0])
    out["valid"] = bool(out["setup_fails_verify"] and out["ref_fix_passes"] and out["lazy_fix_fails"] is not False
                        and out["deterministic"])
    return out


def _box_cls():
    import sys
    m = sys.modules.get("__main__")
    return m.Box if hasattr(m, "Box") else sys.modules["run_arm"].Box
