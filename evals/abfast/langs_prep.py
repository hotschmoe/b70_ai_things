#!/usr/bin/env python3
"""abfast "langs" tier support: per-task dependency/build-cache preparation and injection.

The langs tier (Rust / Go / Dart) must run with no network at task time and with a warm build cache, the
same for both arms. For each task, `prep` (needs network, done once, no model) starts an abfast:3
container, extracts the parent commit like run_arm.init_repo does, runs the task's "prep_cmd" with
networking enabled, and saves the resulting caches to DATA/langprep/<id>.tar:
  Rust: ~/.cargo (registry), /work/repo/target (warm dev build of tests)
  Go:   ~/go/pkg/mod, ~/.cache/go-build
  Dart: ~/.pub-cache, <pkg>/.dart_tool
`inject` (called from run_arm.init_repo when a task has "prep") extracts that tar into a fresh container
right after the repo is extracted and committed. The tar lives outside the git repo.

CLI:  langs_prep.py prep [--tasks id,id] [--parallel 2]      (rebuilds DATA/langprep/<id>.tar)
"""
import argparse, concurrent.futures as cf, json, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = "/mnt/vm_8tb/b70/evals/abfast"
PREPDIR = f"{DATA}/langprep"

# paths saved after prep (relative to /); target/.dart_tool dirs are discovered per task via "prep_paths"
BASE_PATHS = {"rust": ["home/agent/.cargo"], "go": ["home/agent/go", "home/agent/.cache/go-build"],
              "dart": ["home/agent/.pub-cache"]}
# env for the prep step only (network on, unlike run time)
PREP_ENV = {"CARGO_NET_OFFLINE": "false", "GOPROXY": "https://proxy.golang.org,direct", "GOSUMDB": "off",
            "GOFLAGS": "-mod=mod"}


def tier_defaults(a, here):
    """--tier langs: tasks_langs.json, timeout 3600, max output 32768 (same protocol as the hard tier)."""
    if getattr(a, "tier", "") != "langs":
        return
    if a.tasks_file is None:
        a.tasks_file = os.path.join(here, "tasks_langs.json")
    if a.timeout_s is None:
        a.timeout_s = 3600
    if a.max_output_tokens is None:
        a.max_output_tokens = 8192 if a.thinking == "off" else 32768


def parse_counts(kind, log):
    """(passed, total) for cargo / go (go test -json) / dart (dart test -r expanded)."""
    import re
    if kind == "cargo":
        p = f = 0
        for m in re.finditer(r"test result: \w+\. (\d+) passed; (\d+) failed", log):
            p += int(m.group(1)); f += int(m.group(2))
        # a doctest/test binary that failed to compile or crashed leaves no result line for it; rc covers that
        return p, p + f
    if kind == "go":
        res = {}
        for line in log.splitlines():
            if not line.startswith("{"):
                continue
            try:
                ev = json.loads(line)
            except Exception:
                continue
            if ev.get("Test") and ev.get("Action") in ("pass", "fail"):
                res[(ev.get("Package"), ev["Test"])] = ev["Action"]
        # count leaf tests only (a parent with subtests is a container)
        parents = {(pk, t.rsplit("/", 1)[0]) for pk, t in res if "/" in t}
        leaves = [v for k, v in res.items() if k not in parents]
        p = sum(1 for v in leaves if v == "pass")
        return p, len(leaves)
    if kind == "dart":
        ms = re.findall(r"\+(\d+)(?: ~\d+)?(?: -(\d+))?:", log)
        if not ms:
            return 0, 0
        pa, fa = ms[-1]
        return int(pa), int(pa) + int(fa or 0)
    return 0, 0


def tar_path(task):
    return f"{PREPDIR}/{task['id']}.tar"


def inject(box, task, touch_sources=False):
    """Extract the prepared caches into the box. touch_sources: the repo was extracted from a commit other
    than the prep commit (validation of the real sha), so bump source mtimes so cargo/dart rebuild them."""
    p = tar_path(task)
    if not os.path.exists(p):
        raise RuntimeError(f"missing prep tar {p}; run langs_prep.py prep --tasks {task['id']}")
    with open(p, "rb") as f:
        r = subprocess.run(["docker", "exec", "-i", box.name, "tar", "-xf", "-", "-C", "/"], stdin=f,
                           capture_output=True)
    if r.returncode:
        raise RuntimeError("prep inject: " + r.stderr.decode()[:300])
    if touch_sources:
        rc, o = box.exec("cd /work/repo && find . -path ./target -prune -o -path '*/.dart_tool' -prune -o "
                         "-path ./.git -prune -o -type f -exec touch {} + ; "
                         "find . -path '*/.dart_tool/package_config.json' -exec touch {} + ; "
                         "find . -name pubspec.lock -exec touch {} + ; true")
        if rc:
            raise RuntimeError(o)


def prep_task(task, image="abfast:3"):
    import run_arm
    lang = task["lang"]
    name = f"abfast-prep-{task['id']}"[:100]
    subprocess.run(["docker", "rm", "-f", name], capture_output=True)
    cmd = ["docker", "run", "-d", "--name", name, "--network", "host", "--memory", "24g", "--cpus", "8"]
    for k, v in PREP_ENV.items():
        cmd += ["-e", f"{k}={v}"]
    cmd += [image, "sleep", "infinity"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(r.stderr)
    box = run_arm.Box.__new__(run_arm.Box)
    box.name = name
    try:
        # same extraction + initial commit as a real run, but without the prep injection
        t2 = dict(task); t2.pop("prep", None)
        run_arm.init_repo(box, run_arm.git_tar(task["repo"], task["parent"]), t2)
        t0 = time.time()
        rc, out = box.exec(f"cd /work/repo && {task['prep_cmd']}", timeout=task.get("prep_timeout", 3600))
        log = f"{PREPDIR}/{task['id']}.prep.log"
        os.makedirs(PREPDIR, exist_ok=True)
        open(log, "w").write(out[-20000:])
        if rc:
            raise RuntimeError(f"prep_cmd rc={rc}; see {log}")
        paths = BASE_PATHS[lang] + ["work/repo/" + p for p in task.get("prep_paths", [])]
        tmp = tar_path(task) + ".tmp"
        with open(tmp, "wb") as f:
            p = subprocess.run(["docker", "exec", name, "tar", "-cf", "-", "-C", "/", *paths], stdout=f,
                               stderr=subprocess.PIPE)
        if p.returncode:
            raise RuntimeError("tar: " + p.stderr.decode()[:300])
        os.replace(tmp, tar_path(task))
        return round(time.time() - t0, 1), os.path.getsize(tar_path(task))
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prep"])
    ap.add_argument("--tasks", default=""); ap.add_argument("--parallel", type=int, default=1)
    ap.add_argument("--tasks-file", default=os.path.join(HERE, "tasks_langs.json"))
    a = ap.parse_args()
    tasks = json.load(open(a.tasks_file))["tasks"]
    if a.tasks:
        tasks = [t for t in tasks if t["id"] in a.tasks.split(",")]
    with cf.ThreadPoolExecutor(a.parallel) as ex:
        futs = {ex.submit(prep_task, t): t for t in tasks}
        for f in cf.as_completed(futs):
            try:
                print(futs[f]["id"], "PREP_OK secs/bytes", f.result(), flush=True)
            except Exception as e:
                print(futs[f]["id"], "PREP_FAIL", repr(e)[:400], flush=True)


if __name__ == "__main__":
    main()
