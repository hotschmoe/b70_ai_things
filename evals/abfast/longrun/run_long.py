#!/usr/bin/env python3
"""abfast longrun tier runner (invoked via `run_arm.py --tier longrun ...`, or directly).

Per task: one throwaway container, Pi 1.2.0 driven in ROUNDS with the SAME session (`--continue`),
periodic workspace snapshots taken from outside the container, hidden milestone scoring of the final
tree and (after the whole run) of every snapshot. See README section "Longrun tier".
"""
import argparse, concurrent.futures as cf, datetime, hashlib, json, os, re, shlex, subprocess, sys, threading, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import lr_common as C   # noqa: E402
import verify as V      # noqa: E402
import run_arm as RA    # noqa: E402  (pi_models_json, render_prompt, models_ident)

CONTINUE_MSG = ("Continue working until every part of the task is complete. If you believe the work is complete, "
                "re-read the task statement, run the public tests and example checks, compare the result against "
                "the specification once more, and fix anything that fails or is missing. Stop only when there is "
                "nothing left to do or improve.")
# Pi settings, identical for both arms. reserveTokens > max output so a full-length turn cannot overflow.
def pi_settings(ctx):
    return {"compaction": {"enabled": True, "reserveTokens": min(32768, ctx // 4), "keepRecentTokens": min(20000, ctx // 4)}}
EXCL = "-not -path './target/*' -not -path './.git/*' -not -path '*/__pycache__/*' -not -path '*/node_modules/*' -not -path '*/.pytest_cache/*' -not -path './.cache/*'"
TREE_FIND = f"cd /work/repo && find . -type f {EXCL} ! -perm -u+x -size -1000k"


def tree_state(box):
    """(hash of path/size/mtime listing, newest mtime epoch, now epoch) of source-like files, non-invasive."""
    rc, o = box.exec(f"{TREE_FIND} -printf '%p %s %T@\\n' | sort > /tmp/.tree; sha1sum < /tmp/.tree | cut -c1-16; "
                     f"awk '{{print $3}}' /tmp/.tree | sort -n | tail -1; date +%s")
    p = o.split()
    try:
        return p[0], float(p[1]) if len(p) > 2 else 0.0, float(p[-1])
    except Exception:
        return None, 0.0, time.time()


def snapshot_bytes(box):
    rc, data = box.exec_bytes(f"cd /work/repo && {TREE_FIND} -print0 | tar --null -T - -czf - 2>/dev/null")
    return data


class Metrics:
    """Transcript hygiene metrics, accumulated on the fly from Pi's json events."""
    def __init__(self):
        self.turns = self.tool_calls = self.tool_errors = self.unknown_tool = self.bad_args = 0
        self.inp = self.out = self.cr = 0
        self.compactions = self.compaction_fail = self.overflow_compactions = 0
        self.max_repeat = 1; self._last = None; self._run = 0
        self.max_repeat_call = None
        self.stop_reasons = {}; self.last_stop = None; self.err = ""
        self.tool_names = {}; self.pending = {}
        self.retries = 0
        self.empty_turns = 0   # assistant turns with no text and no tool call

    def event(self, ev):
        t = ev.get("type")
        if t == "message_end":
            m = ev.get("message") or {}
            if m.get("role") == "assistant":
                u = m.get("usage") or {}
                self.turns += 1
                self.inp += u.get("input", 0); self.out += u.get("output", 0); self.cr += u.get("cacheRead", 0)
                self.last_stop = m.get("stopReason"); self.err = m.get("errorMessage") or self.err
                self.stop_reasons[self.last_stop] = self.stop_reasons.get(self.last_stop, 0) + 1
                calls = [c for c in (m.get("content") or []) if isinstance(c, dict) and c.get("type") == "toolCall"]
                if not calls and not any(isinstance(c, dict) and c.get("type") == "text" and (c.get("text") or "").strip()
                                         for c in (m.get("content") or [])):
                    self.empty_turns += 1
                for c in calls:
                    self.tool_calls += 1
                    name = c.get("name"); self.tool_names[name] = self.tool_names.get(name, 0) + 1
                    key = (name, json.dumps(c.get("arguments"), sort_keys=True))
                    if key == self._last:
                        self._run += 1
                    else:
                        self._last, self._run = key, 1
                    if self._run > self.max_repeat:
                        self.max_repeat, self.max_repeat_call = self._run, f"{name} {key[1][:120]}"
            elif m.get("role") == "toolResult":
                self._tool_result(m)
        elif t in ("tool_execution_end",):
            if ev.get("isError"):
                self.tool_errors += 1
        elif t == "compaction_end":
            if ev.get("result"):
                self.compactions += 1
                if ev.get("reason") == "overflow":
                    self.overflow_compactions += 1
            elif not ev.get("aborted"):
                self.compaction_fail += 1
        elif t in ("auto_retry_start", "retry_start"):
            self.retries += 1

    def _tool_result(self, m):
        if m.get("isError"):
            txt = json.dumps(m.get("content"))[:400]
            if re.search(r"not found|unknown tool|no such tool|not available", txt, re.I):
                self.unknown_tool += 1
            elif re.search(r"validation|invalid (arg|param)|missing required|must have|Expected|schema", txt, re.I):
                self.bad_args += 1

    def as_dict(self):
        return {"turns": self.turns, "tool_calls": self.tool_calls, "tool_errors": self.tool_errors,
                "unknown_tool_calls": self.unknown_tool, "bad_arg_calls": self.bad_args,
                "empty_turns": self.empty_turns, "compactions": self.compactions,
                "compaction_failures": self.compaction_fail, "overflow_compactions": self.overflow_compactions,
                "pi_retries": self.retries, "max_identical_tool_call_run": self.max_repeat,
                "max_repeat_call": self.max_repeat_call, "prompt_tokens": self.inp + self.cr,
                "uncached_prompt_tokens": self.inp, "cache_read_tokens": self.cr, "completion_tokens": self.out,
                "stop_reasons": self.stop_reasons, "tool_names": self.tool_names, "last_stop_reason": self.last_stop}


def pi_round(box, a, key, n, prompt, remaining_s, fo, metrics, env):
    cont = "" if n == 1 else "--continue "
    script = (f"cd /work/repo && timeout --signal=TERM --kill-after=15 {int(remaining_s)} "
              f"/opt/pi/node_modules/.bin/pi --print --mode json --session-dir /logs/pi {cont}--provider harbor-endpoint "
              f"--model {shlex.quote(a.model)} --thinking {a.thinking} {shlex.quote(prompt)} </dev/null")
    cmd = ["docker", "exec", "-i"]
    for k, v in env.items():
        cmd += ["-e", f"{k}={v}"]
    cmd += [box.name, "bash", "-c", script]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    fo.write(json.dumps({"type": "abfast_round_start", "round": n, "t": time.time()}) + "\n")
    tail = []
    killer = threading.Timer(remaining_s + 90, lambda: p.kill())
    killer.start()
    try:
        for raw in p.stdout:
            line = raw.decode(errors="replace")
            if '"type":"message_update"' in line:
                continue
            fo.write(line if line.endswith("\n") else line + "\n")
            try:
                ev = json.loads(line)
            except Exception:
                tail.append(line[:300]); tail[:] = tail[-5:]
                continue
            metrics.event(ev)
        rc = p.wait()
    finally:
        killer.cancel()
    fo.flush()
    return rc, "".join(tail)


def seed_box(box, task, a):
    td = f"{C.LDATA}/{task['id']}"
    if os.path.isdir(f"{td}/target"):
        box.put_tar(C.tar_dir(f"{td}/target"), "/opt/target", user="0")
        box.exec("chmod -R a+rx /opt/target", user="0")
    box.put_tar(C.tar_dir(f"{td}/agent"), "/work/repo")
    rc, o = box.exec("cd /work/repo && git init -q && git add -A && git -c user.name=abfast -c user.email=a@b commit -qm 'initial state'")
    if rc:
        raise RuntimeError(o)
    mj = RA.pi_models_json(a.base_url, a.key or "EMPTY", a.model, a.context_window, a.max_output_tokens)
    box.exec("mkdir -p $PI_CODING_AGENT_DIR /logs/pi && cat > $PI_CODING_AGENT_DIR/models.json", stdin=json.dumps(mj).encode())
    box.exec("cat > $PI_CODING_AGENT_DIR/settings.json", stdin=json.dumps(pi_settings(a.context_window)).encode())
    if task.get("tools_resolved") == "rea":
        setup_rea(box)


# Core REA tools declared directly to the model (each costs ~1.5k tokens of schema); the other ~160 REA tools stay
# reachable through Pi's `codemode` tool (exposure "codemode"). compact schema profile keeps the schemas small.
REA_DIRECT = ["open_binary", "list_procedures", "procedure_pseudo_code", "procedure_assembly", "list_strings", "search_strings", "xrefs", "read_bytes"]
REA_MCP = {"mcpServers": {"rea": {
    "type": "stdio", "command": "rea", "args": ["mcp"], "timeout": 600, "exposure": "codemode",
    "toolExposure": {t: "direct" for t in REA_DIRECT},
    "description": "REA reverse-engineering workbench (Ghidra provider): open a binary, list procedures/strings, decompile, xrefs.",
    "env": {"GHIDRA_INSTALL_DIR": "/opt/ghidra", "JAVA_HOME": "/usr/lib/jvm/java-21-openjdk-amd64",
            "REA_MCP_INPUT_SCHEMA_PROFILE": "compact"}}}}


def setup_rea(box):
    """REA available to the agent: MCP server (Pi native MCP, direct exposure) + the REA workflow skill."""
    box.exec("cat > $PI_CODING_AGENT_DIR/mcp.json", stdin=json.dumps(REA_MCP).encode())
    box.exec("mkdir -p /home/agent/.agents/skills && cp -r /opt/rea-skill/reverse-engineer-anything /home/agent/.agents/skills/")


REA_NOTE = ("\n\nAvailable reverse-engineering tooling: the `rea` MCP tools (Ghidra-backed static analysis: binary overview, "
            "procedure list, strings, cross references, decompilation, disassembly) and the `reverse-engineer-anything` skill are "
            "installed in addition to objdump, gdb, radare2 and strace. Use whatever is most effective.")


def run_task(task, a, run_dir):
    tid = task["id"]
    d = f"{run_dir}/tasks/{tid}"
    os.makedirs(f"{d}/snapshots", exist_ok=True)
    budget = a.timeout_s
    res = {"id": tid, "type": task.get("type"), "lang": task.get("lang"), "tools": task["tools_resolved"],
           "end_reason": "harness_error", "budget_s": budget}
    box = C.LBox(f"abfast-l-{os.path.basename(run_dir)}-{tid}"[:100], task.get("image", C.IMAGE))
    t_start = time.time()
    metrics = Metrics()
    rounds = []
    snaps = []
    stop_snap = threading.Event()
    try:
        seed_box(box, task, a)
        h0, _, now0 = tree_state(box)
        t0 = time.time()
        state = {"hash": h0, "last_change_s": 0.0, "snap_hash": h0}

        def poller():
            nxt = a.snapshot_s
            while not stop_snap.wait(60):
                try:
                    h, mt, now = tree_state(box)
                    el = time.time() - t0
                    if h != state["hash"]:
                        state["hash"] = h; state["last_change_s"] = round(max(0.0, el - (now - mt)), 1)
                    if el >= nxt:
                        nxt += a.snapshot_s
                        if state["snap_hash"] != h:
                            state["snap_hash"] = h
                            fn = f"snapshots/t{int(el):06d}.tar.gz"
                            open(f"{d}/{fn}", "wb").write(snapshot_bytes(box))
                            snaps.append({"t_s": int(el), "file": fn})
                except Exception as e:  # never kill the run for a snapshot problem
                    print(f"[{tid}] snapshot error {e!r}", flush=True)
        th = threading.Thread(target=poller, daemon=True); th.start()

        instr = task["instruction"] + (REA_NOTE if task["tools_resolved"] == "rea" and task.get("type") == "reverse-engineering" else "")
        prompt0 = RA.render_prompt(instr, a.thinking)
        env = {"PYTEST_ADDOPTS": "--timeout=120"}
        idle = 0; ep_fail = 0; end = None
        with open(f"{d}/pi_transcript.jsonl", "w") as fo:
            n = 0
            while True:
                n += 1
                el = time.time() - t0
                remaining = budget - el
                if remaining < 120:
                    end = "timeout"; break
                if n > a.max_rounds:
                    end = "max_rounds"; break
                before, _, _ = tree_state(box)
                rs = time.time() - t0
                rc, tail = pi_round(box, a, a.key, n, prompt0 if n == 1 else CONTINUE_MSG, remaining, fo, metrics, env)
                re_ = time.time() - t0
                after, mt, now = tree_state(box)
                changed = after != before
                if changed:
                    state["hash"] = after; state["last_change_s"] = round(max(0.0, re_ - (now - mt)), 1)
                rec = {"round": n, "start_s": round(rs, 1), "end_s": round(re_, 1), "rc": rc, "changed": changed,
                       "turns_total": metrics.turns, "last_stop": metrics.last_stop}
                rounds.append(rec)
                print(f"[{tid}] round {n} rc={rc} {rs:.0f}-{re_:.0f}s changed={changed} stop={metrics.last_stop}", flush=True)
                if rc in (124, 137):
                    end = "timeout"; break
                if metrics.last_stop in ("error", "aborted") or (rc != 0 and metrics.turns == 0):
                    ep_fail += 1
                    rec["endpoint_error"] = True
                    if ep_fail >= a.max_endpoint_failures:
                        end = "endpoint_error"; res["error"] = (metrics.err or tail)[:300]; break
                    time.sleep(60)
                    continue
                ep_fail = 0
                idle = 0 if changed else idle + 1
                if idle >= a.idle_rounds:
                    end = "agent_stopped"; break
        res["end_reason"] = end
        el_end = time.time() - t0
        res["agent_wall_s"] = round(el_end, 1)
        res["first_stop_s"] = rounds[0]["end_s"] if rounds else None
        res["rounds"] = rounds
        res["last_change_s"] = state["last_change_s"]
        stop_snap.set(); th.join(timeout=120)
        # final snapshot + session + patch
        open(f"{d}/snapshots/final.tar.gz", "wb").write(snapshot_bytes(box))
        rc2, lst = box.exec("ls /logs/pi/*.jsonl /logs/pi/*/*.jsonl 2>/dev/null | head -1")
        sf = lst.strip().splitlines()[0] if lst.strip() else None
        if sf:
            rc3, data = box.exec_bytes(f"cat {sf}")
            open(f"{d}/pi_session.jsonl", "wb").write(data)
        rc3, patch = box.exec("cd /work/repo && git add -A -f 2>/dev/null; git diff --cached --stat | tail -40")
        open(f"{d}/agent_diff.stat", "w").write(patch)
        res.update(metrics.as_dict())
        res["snapshots"] = snaps
        sc = V.score(task, open(f"{d}/snapshots/final.tar.gz", "rb").read(), tag="fin")
        json.dump(sc, open(f"{d}/final_score.json", "w"), indent=1)
        res.update(final_fraction=sc["fraction"], milestones_passed=sc["milestones_passed"],
                   milestones_total=sc["milestones_total"], hidden_passed=sc["tests_passed"],
                   hidden_total=sc["tests_total"], tests_fraction=sc["tests_fraction"], built=sc["built"],
                   cheat_flagged=sc.get("cheat_flagged"), **{"pass": sc["milestones_passed"] == sc["milestones_total"] > 0})
    except Exception as e:
        res["end_reason"] = "harness_error"; res["error"] = repr(e)[:400]
        res.setdefault("pass", False)
    finally:
        try:
            stop_snap.set()
        except Exception:
            pass
        box.rm()
    res["task_wall_s"] = round(time.time() - t_start, 1)
    json.dump(res, open(f"{d}/result.json", "w"), indent=1)
    print(f"[{tid}] {res['end_reason']} milestones={res.get('milestones_passed')}/{res.get('milestones_total')} "
          f"agent={res.get('agent_wall_s')}s turns={res.get('turns')} compactions={res.get('compactions')}", flush=True)
    return res


def score_snapshots(run_dir, tasks, parallel):
    """Post-run: score every distinct snapshot of every task, write curve.json per task."""
    jobs = []
    for t in tasks:
        d = f"{run_dir}/tasks/{t['id']}"
        rj = f"{d}/result.json"
        if not os.path.exists(rj):
            continue
        r = json.load(open(rj))
        for s in r.get("snapshots", []):
            jobs.append((t, d, s))

    def one(j):
        t, d, s = j
        sc = V.score(t, open(f"{d}/{s['file']}", "rb").read(), tag=f"s{s['t_s']}")
        return t["id"], s["t_s"], {"t_s": s["t_s"], "milestones_passed": sc["milestones_passed"], "milestones_total": sc["milestones_total"],
                                   "fraction": sc["fraction"], "tests_passed": sc["tests_passed"], "tests_total": sc["tests_total"]}
    got = {}
    with cf.ThreadPoolExecutor(parallel) as ex:
        for tid, ts, row in ex.map(one, jobs):
            got.setdefault(tid, []).append(row)
    for t in tasks:
        d = f"{run_dir}/tasks/{t['id']}"
        rj = f"{d}/result.json"
        if not os.path.exists(rj):
            continue
        r = json.load(open(rj))
        curve = [{"t_s": 0, "fraction": 0.0, "milestones_passed": 0, "milestones_total": r.get("milestones_total"), "tests_passed": 0, "tests_total": r.get("hidden_total")}]
        curve += sorted(got.get(t["id"], []), key=lambda x: x["t_s"])
        curve.append({"t_s": int(r.get("agent_wall_s") or 0), "fraction": r.get("final_fraction"), "milestones_passed": r.get("milestones_passed"),
                      "milestones_total": r.get("milestones_total"), "tests_passed": r.get("hidden_passed"), "tests_total": r.get("hidden_total")})
        json.dump(curve, open(f"{d}/curve.json", "w"), indent=1)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="run_arm.py --tier longrun")
    ap.add_argument("--tier", default="longrun")
    ap.add_argument("--base-url", required=True); ap.add_argument("--api-key-file"); ap.add_argument("--model", default="hotschmoe-dd")
    ap.add_argument("--arm-label", required=True); ap.add_argument("--parallel", type=int, default=2)
    ap.add_argument("--timeout-s", type=int, default=14400, help="agent wall budget per task (default 4 h)")
    ap.add_argument("--thinking", default="xhigh", choices=["off", "xhigh"])
    ap.add_argument("--max-output-tokens", type=int, default=32768)
    ap.add_argument("--context-window", type=int, default=131072)
    ap.add_argument("--tasks", default="")
    ap.add_argument("--tasks-file", default=C.TASKS_FILE)
    ap.add_argument("--tools", default="task", choices=["task", "rea", "plain"],
                    help="task: each task's default (RE tasks rea, others plain); rea/plain force it for RE tasks")
    ap.add_argument("--snapshot-s", type=int, default=1800)
    ap.add_argument("--idle-rounds", type=int, default=2, help="stop after this many consecutive rounds with no file change")
    ap.add_argument("--max-rounds", type=int, default=500)
    ap.add_argument("--max-endpoint-failures", type=int, default=6)
    ap.add_argument("--no-score-snapshots", action="store_true")
    ap.add_argument("--image", default=None)
    a = ap.parse_args(argv)
    a.key = open(a.api_key_file).readline().strip() if a.api_key_file else ""
    tasks = C.load_tasks(a.tasks_file)
    if a.tasks:
        want = a.tasks.split(","); tasks = [t for t in tasks if t["id"] in want]
    else:
        tasks = [t for t in tasks if not t.get("disabled")]
    for t in tasks:
        dflt = t.get("tools", "plain")
        t["tools_resolved"] = dflt if (a.tools == "task" or dflt == "plain") else a.tools
        if a.image:
            t["image"] = a.image
    ids = RA.models_ident(a.base_url, a.key)
    if a.model not in ids:
        sys.exit(f"model {a.model} not served; /v1/models = {ids}")
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = f"{C.DATA}/runs/{a.arm_label}-{stamp}"
    os.makedirs(run_dir)
    meta = {"arm": a.arm_label, "base_url": a.base_url, "model": a.model, "served_ids": ids, "thinking": a.thinking,
            "max_output_tokens": a.max_output_tokens, "context_window": a.context_window, "parallel": a.parallel,
            "timeout_s": a.timeout_s, "image": a.image or C.IMAGE, "tier": "longrun", "tasks": [t["id"] for t in tasks],
            "tools": {t["id"]: t["tools_resolved"] for t in tasks}, "pi_settings": pi_settings(a.context_window), "continue_msg": CONTINUE_MSG,
            "snapshot_s": a.snapshot_s, "idle_rounds": a.idle_rounds, "started": stamp}
    json.dump(meta, open(f"{run_dir}/meta.json", "w"), indent=1)
    print("run dir", run_dir, flush=True)
    t0 = time.time()
    with cf.ThreadPoolExecutor(a.parallel) as ex:
        results = list(ex.map(lambda t: run_task(t, a, run_dir), tasks))
    meta["total_wall_s"] = round(time.time() - t0, 1)
    try:
        meta["end_models"] = RA.models_ident(a.base_url, a.key); meta["end_health"] = "ok"
    except Exception as e:
        meta["end_health"] = f"FAIL {e!r}"
    json.dump(meta, open(f"{run_dir}/meta.json", "w"), indent=1)
    json.dump(results, open(f"{run_dir}/results.json", "w"), indent=1)
    if not a.no_score_snapshots:
        score_snapshots(run_dir, tasks, max(2, a.parallel))
    print(f"done {len(results)} tasks in {meta['total_wall_s']}s; end_health={meta['end_health']}; {run_dir}")


if __name__ == "__main__":
    main()
