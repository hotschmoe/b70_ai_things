#!/usr/bin/env python3
"""abfast runner: Pi 1.2.0 in a throwaway docker container per task, hidden-test verifier.

Run one arm:   run_arm.py --base-url URL --model hotschmoe-dd --arm-label armA [--api-key-file F]
Validate tasks (no model): run_arm.py --validate [--write]
"""
import argparse, concurrent.futures as cf, datetime, io, json, os, re, shlex, subprocess, sys, tarfile, threading, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import zigtests  # noqa: E402

DATA = "/mnt/vm_8tb/b70/evals/abfast"
SRC = f"{DATA}/src"
TBDIR = os.path.join(HERE, "..", "terminalbench")
GITENV = {"GIT_AUTHOR_NAME": "abfast", "GIT_AUTHOR_EMAIL": "a@b", "GIT_COMMITTER_NAME": "abfast", "GIT_COMMITTER_EMAIL": "a@b"}


def sh(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def git(repo, *args, raw=False):
    r = subprocess.run(["git", "-C", f"{SRC}/{repo}", *args], capture_output=True)
    if r.returncode:
        raise RuntimeError(f"git {args}: {r.stderr.decode()[:300]}")
    return r.stdout if raw else r.stdout.decode()


def git_tar(repo, sha):
    return git(repo, "archive", "--format=tar", sha, raw=True)


def git_show(repo, sha, path):
    return git(repo, "show", f"{sha}:{path}", raw=True)


# ---- Pi configuration: keep identical to evals/terminalbench/harbor_pi.py ----
def pi_models_json(base_url, key, model, ctx, max_tokens):
    mdl = {  # == harbor_pi.qwen_model_definition
        "id": model, "name": model, "reasoning": True,
        "thinkingLevelMap": {"off": "none", "minimal": None, "low": None, "medium": None,
                             "high": None, "xhigh": "xhigh", "max": None},
        "input": ["text"], "contextWindow": ctx, "maxTokens": max_tokens,
        "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
    }
    return {"providers": {"harbor-endpoint": {
        "baseUrl": base_url, "apiKey": key, "api": "openai-completions",
        "compat": {"supportsStore": False, "supportsDeveloperRole": False,
                   "supportsReasoningEffort": False, "supportsUsageInStreaming": True,
                   "supportsFinishReason": True, "maxTokensField": "max_tokens",
                   "thinkingFormat": "qwen-chat-template", "supportsStrictMode": False},
        "models": [mdl]}}}


def render_prompt(instruction, thinking):
    f = "pi_concise_off_prompt.j2" if thinking == "off" else "pi_concise_prompt.j2"
    return open(os.path.join(TBDIR, f)).read().replace("{{ instruction }}", instruction)


# ---- docker helpers ----
class Box:
    def __init__(self, name, image):
        self.name = name
        r = sh(["docker", "run", "-d", "--name", name, "--network", "host", "--memory", "12g", "--cpus", "6",
                "--pids-limit", "4096", "-e", "ZIG_GLOBAL_CACHE_DIR=/home/agent/.zigcache", image, "sleep", "infinity"])
        if r.returncode:
            raise RuntimeError(r.stderr)

    def exec(self, script, timeout=None, stdin=None, env=None, out=None):
        cmd = ["docker", "exec", "-i"]
        for k, v in (env or {}).items():
            cmd += ["-e", f"{k}={v}"]
        cmd += [self.name, "bash", "-c", script]
        if out is not None:
            p = subprocess.run(cmd, input=stdin, stdout=out, stderr=subprocess.STDOUT, timeout=timeout)
            return p.returncode, ""
        p = subprocess.run(cmd, input=stdin, capture_output=True, timeout=timeout)
        return p.returncode, (p.stdout + p.stderr).decode(errors="replace")

    def put_tar(self, data, dest):
        rc, o = self.exec(f"mkdir -p {dest} && tar -xf - -C {dest}", stdin=data)
        if rc:
            raise RuntimeError(o)

    def get_file(self, path):
        p = subprocess.run(["docker", "exec", self.name, "cat", path], capture_output=True)
        return p.stdout

    def rm(self):
        sh(["docker", "rm", "-f", self.name])


def parse_counts(kind, log):
    if kind == "pytest":
        n = {k: 0 for k in ("passed", "failed", "error", "errors")}
        for k in n:
            m = re.findall(rf"(\d+) {k}\b", log)
            n[k] = int(m[-1]) if m else 0
        bad = n["failed"] + n["error"] + n["errors"]
        return n["passed"], n["passed"] + bad
    if kind == "node":
        g = lambda k: int((re.findall(rf"^# {k} (\d+)", log, re.M) or ["0"])[-1])
        return g("pass"), g("tests")
    if kind == "zig":
        m = re.search(r"(\d+)/(\d+) tests passed", log)
        if m:
            return int(m.group(1)), int(m.group(2))
        m = re.search(r"All (\d+) tests passed", log)
        if m:
            return int(m.group(1)), int(m.group(1))
        m = re.search(r"(\d+) passed; (\d+) skipped; (\d+) failed", log)
        if m:
            return int(m.group(1)), int(m.group(1)) + int(m.group(3))
    return 0, 0


def build_hidden_tar(task):
    """Tar of real-commit versions of withheld files, plus inline zig test blocks to append."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        for p in task.get("hidden_files", []):
            data = git_show(task["repo"], task["real"], p)
            ti = tarfile.TarInfo(p); ti.size = len(data); ti.mode = 0o644
            tf.addfile(ti, io.BytesIO(data))
    appends = {}
    for p in task.get("hidden_inline", []):
        old = git_show(task["repo"], task["parent"], p).decode()
        new = git_show(task["repo"], task["real"], p).decode()
        appends[p] = "\n\n" + "\n\n".join(zigtests.new_blocks(old, new)) + "\n"
    return buf.getvalue(), appends


def apply_hidden_and_test(box, task, hidden, appends, log_path):
    box.put_tar(hidden, "/work/repo")
    for p, text in appends.items():
        rc, o = box.exec(f"cat >> {shlex.quote('/work/repo/' + p)}", stdin=text.encode())
        if rc:
            raise RuntimeError(o)
    with open(log_path, "wb") as f:
        try:
            rc, _ = box.exec(f"cd /work/repo && {task['test_cmd']}", timeout=task.get("test_timeout", 900), out=f)
        except subprocess.TimeoutExpired:
            f.write(b"\nTEST TIMEOUT\n"); rc = 124
    log = open(log_path, errors="replace").read()
    p, t = parse_counts(task["parse"], log)
    ok = rc == 0 and p > 0 and p == t
    return {"rc": rc, "passed": p, "total": t, "pass": ok}


def init_repo(box, tar_bytes):
    box.put_tar(tar_bytes, "/work/repo")
    rc, o = box.exec("cd /work/repo && git init -q && git add -A && git -c user.name=abfast -c user.email=a@b commit -qm 'initial state'")
    if rc:
        raise RuntimeError(o)


def models_ident(base_url, key):
    req = urllib.request.Request(base_url.rstrip("/") + "/models")
    if key:
        req.add_header("Authorization", f"Bearer {key}")
    with urllib.request.urlopen(req, timeout=20) as r:
        return [m["id"] for m in json.load(r)["data"]]


# ---- validate ----
def validate_task(task, image, tag):
    out = {}
    hidden, appends = build_hidden_tar(task)
    d = f"{DATA}/runs/validate/{task['id']}"
    os.makedirs(d, exist_ok=True)
    for which, sha in (("parent", task["parent"]), ("real", task["real"])):
        box = Box(f"abfast-val-{tag}-{task['id']}-{which}", image)
        try:
            init_repo(box, git_tar(task["repo"], sha))
            t0 = time.time()
            if which == "parent":
                r = apply_hidden_and_test(box, task, hidden, appends, f"{d}/{which}.log")
            else:  # real tree already contains the tests; run the same command
                with open(f"{d}/{which}.log", "wb") as f:
                    try:
                        rc, _ = box.exec(f"cd /work/repo && {task['test_cmd']}", timeout=task.get("test_timeout", 900), out=f)
                    except subprocess.TimeoutExpired:
                        rc = 124
                p, t = parse_counts(task["parse"], open(f"{d}/{which}.log", errors="replace").read())
                r = {"rc": rc, "passed": p, "total": t, "pass": rc == 0 and p > 0 and p == t}
            r["secs"] = round(time.time() - t0, 1)
            out[which] = r
        finally:
            box.rm()
    out["valid"] = (not out["parent"]["pass"]) and out["real"]["pass"]
    return out


# ---- one attempt ----
def run_task(task, a, key, run_dir):
    tid = task["id"]
    d = f"{run_dir}/tasks/{tid}"
    os.makedirs(d, exist_ok=True)
    res = {"id": tid, "repo": task["repo"], "end_reason": "agent_error", "pass": False}
    box = Box(f"abfast-{os.path.basename(run_dir)}-{tid}"[:100], a.image)
    t_start = time.time()
    try:
        init_repo(box, git_tar(task["repo"], task["parent"]))
        mj = pi_models_json(a.base_url, key or "EMPTY", a.model, a.context_window, a.max_output_tokens)
        rc, o = box.exec("mkdir -p $PI_CODING_AGENT_DIR /logs/pi && cat > $PI_CODING_AGENT_DIR/models.json",
                         stdin=json.dumps(mj).encode())
        prompt = render_prompt(task["instruction"], a.thinking)
        script = (f"cd /work/repo && timeout --signal=TERM --kill-after=15 {a.timeout_s} "
                  f"pi --print --mode json --session-dir /logs/pi --provider harbor-endpoint "
                  f"--model {shlex.quote(a.model)} --thinking {a.thinking} {shlex.quote(prompt)} </dev/null")
        raw = f"{d}/pi_stdout.jsonl"
        t0 = time.time()
        try:
            with open(raw, "wb") as f:
                rc, _ = box.exec(script, timeout=a.timeout_s + 60, out=f)
        except subprocess.TimeoutExpired:
            rc = 124
        res["agent_wall_s"] = round(time.time() - t0, 1)
        res["pi_rc"] = rc
        # transcript: drop streaming deltas, keep everything else
        n_turn = n_tool = 0; inp = out_t = cr = 0; last_stop = None; err_txt = ""
        with open(raw, errors="replace") as fi, open(f"{d}/pi_transcript.jsonl", "w") as fo:
            for line in fi:
                if '"type":"message_update"' in line:
                    continue
                fo.write(line)
                try:
                    ev = json.loads(line)
                except Exception:
                    continue
                if ev.get("type") == "message_end" and (ev.get("message") or {}).get("role") == "assistant":
                    m = ev["message"]; u = m.get("usage") or {}
                    n_turn += 1
                    inp += u.get("input", 0); out_t += u.get("output", 0); cr += u.get("cacheRead", 0)
                    n_tool += sum(1 for c in (m.get("content") or []) if isinstance(c, dict) and c.get("type") == "toolCall")
                    last_stop = m.get("stopReason"); err_txt = m.get("errorMessage") or err_txt
        os.remove(raw)  # pi_transcript.jsonl is the same minus streaming deltas; pi_session.jsonl is Pi's own
        res.update(turns=n_turn, tool_calls=n_tool, prompt_tokens=inp + cr, uncached_prompt_tokens=inp,
                   cache_read_tokens=cr, completion_tokens=out_t, last_stop_reason=last_stop)
        if rc == 124 or rc == 137:
            res["end_reason"] = "timeout"
        elif last_stop in ("error", "aborted") or (n_turn == 0 and rc != 0 and re.search(r"ECONN|fetch failed|HTTP|50\d", open(raw, errors='replace').read())):
            res["end_reason"] = "endpoint_error"; res["error"] = (err_txt or "")[:300]
        elif rc == 0:
            res["end_reason"] = "completed"
        else:
            res["end_reason"] = "agent_error"
        # session file + patch (before hidden tests are placed)
        rc2, lst = box.exec("ls /logs/pi/*.jsonl /logs/pi/*/*.jsonl 2>/dev/null | head -1")
        sf = lst.strip().splitlines()[0] if lst.strip() else None
        if sf:
            open(f"{d}/pi_session.jsonl", "wb").write(box.get_file(sf))
        rc3, patch = box.exec("cd /work/repo && git add -A && git diff --cached --stat && git diff --cached")
        open(f"{d}/agent_diff.patch", "w").write(patch)
        res["files_changed"] = len(re.findall(r"^diff --git", patch, re.M))
        hidden, appends = build_hidden_tar(task)
        tr = apply_hidden_and_test(box, task, hidden, appends, f"{d}/hidden_test.log")
        res.update(hidden_rc=tr["rc"], hidden_passed=tr["passed"], hidden_total=tr["total"], **{"pass": tr["pass"]})
    except Exception as e:  # harness failure, not a model failure
        res["end_reason"] = "harness_error"; res["error"] = repr(e)[:400]
    finally:
        box.rm()
    res["task_wall_s"] = round(time.time() - t_start, 1)
    json.dump(res, open(f"{d}/result.json", "w"), indent=1)
    print(f"[{tid}] {res['end_reason']} pass={res['pass']} {res.get('hidden_passed')}/{res.get('hidden_total')} "
          f"agent={res.get('agent_wall_s')}s turns={res.get('turns')} tok={res.get('prompt_tokens')}/{res.get('completion_tokens')}", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url"); ap.add_argument("--api-key-file"); ap.add_argument("--model", default="hotschmoe-dd")
    ap.add_argument("--arm-label"); ap.add_argument("--parallel", type=int, default=1)
    ap.add_argument("--timeout-s", type=int, default=900)
    ap.add_argument("--thinking", default="xhigh", choices=["off", "xhigh"])
    ap.add_argument("--max-output-tokens", type=int, default=None)
    ap.add_argument("--context-window", type=int, default=131072)
    ap.add_argument("--tasks", default="", help="comma-separated task ids (default: all validated)")
    ap.add_argument("--tasks-file", default=os.path.join(HERE, "tasks.json"))
    ap.add_argument("--image", default="abfast:1")
    ap.add_argument("--validate", action="store_true"); ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    spec = json.load(open(a.tasks_file))
    tasks = spec["tasks"]
    if a.tasks:
        want = a.tasks.split(","); tasks = [t for t in tasks if t["id"] in want]
    if a.validate:
        tag = str(os.getpid())
        with cf.ThreadPoolExecutor(a.parallel) as ex:
            futs = {ex.submit(validate_task, t, a.image, tag): t for t in tasks}
            for f in cf.as_completed(futs):
                t = futs[f]
                try:
                    t["validation"] = f.result()
                except Exception as e:
                    t["validation"] = {"valid": False, "error": repr(e)[:300]}
                v = t["validation"]
                print(t["id"], "VALID" if v.get("valid") else "INVALID", json.dumps({k: v[k] for k in v if k != "valid"})[:300], flush=True)
        if a.write:
            full = json.load(open(a.tasks_file))
            for t in full["tasks"]:
                for u in tasks:
                    if u["id"] == t["id"]:
                        t["validation"] = u["validation"]
            json.dump(full, open(a.tasks_file, "w"), indent=1); open(a.tasks_file, "a").write("\n")
        return
    if not (a.base_url and a.arm_label):
        ap.error("--base-url and --arm-label required")
    if a.max_output_tokens is None:
        a.max_output_tokens = 8192 if a.thinking == "off" else 16384
    if not a.tasks:
        tasks = [t for t in tasks if t.get("validation", {}).get("valid")]
    key = open(a.api_key_file).readline().strip() if a.api_key_file else ""
    ids = models_ident(a.base_url, key)
    if a.model not in ids:
        sys.exit(f"model {a.model} not served; /v1/models = {ids}")
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = f"{DATA}/runs/{a.arm_label}-{stamp}"
    os.makedirs(run_dir)
    meta = {"arm": a.arm_label, "base_url": a.base_url, "model": a.model, "served_ids": ids, "thinking": a.thinking,
            "max_output_tokens": a.max_output_tokens, "context_window": a.context_window, "parallel": a.parallel,
            "timeout_s": a.timeout_s, "image": a.image, "tasks": [t["id"] for t in tasks], "started": stamp}
    json.dump(meta, open(f"{run_dir}/meta.json", "w"), indent=1)
    print("run dir", run_dir, flush=True)
    t0 = time.time()
    with cf.ThreadPoolExecutor(a.parallel) as ex:
        results = list(ex.map(lambda t: run_task(t, a, key, run_dir), tasks))
    meta["total_wall_s"] = round(time.time() - t0, 1)
    try:
        meta["end_models"] = models_ident(a.base_url, key); meta["end_health"] = "ok"
    except Exception as e:
        meta["end_health"] = f"FAIL {e!r}"
    json.dump(meta, open(f"{run_dir}/meta.json", "w"), indent=1)
    json.dump(results, open(f"{run_dir}/results.json", "w"), indent=1)
    print(f"done {len(results)} tasks in {meta['total_wall_s']}s; end_health={meta['end_health']}; {run_dir}")


if __name__ == "__main__":
    main()
