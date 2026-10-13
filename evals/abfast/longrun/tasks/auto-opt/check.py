#!/usr/bin/env python3
"""Hidden scorer for auto-opt (runs in the verifier container: network none, 4 cpus).

Correctness tiers on hidden inputs, then speed milestones measured against the ORIGINAL naive program built in
this same session on the same hidden large input. Every run uses a fresh process, a fresh temp dir and a random
relabeling of token ids (answers are invariant under relabeling), so nothing can be cached between runs."""
import json, os, random, shutil, statistics, subprocess, sys, tempfile, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib"))
from mscore import Milestones

SUB, HID = sys.argv[1], os.path.dirname(os.path.abspath(__file__))
BIN = f"{SUB}/target/release/nnjoin"
meta = json.load(open(f"{HID}/meta.json"))
RNG = random.Random(int(os.environ.get("NN_RELABEL_SEED", "424242")))   # only the relabeling varies


def relabel(text):
    lines = text.split("\n")
    toks = set()
    for ln in lines[1:]:
        toks.update(ln.split())
    toks = sorted(toks, key=int)
    new = RNG.sample(range(0, 4_000_000_000), len(toks))
    mp = {t: str(v) for t, v in zip(toks, new)}
    out = [lines[0]]
    for ln in lines[1:]:
        # keep the original whitespace layout where possible: rebuild token by token
        parts, cur = [], ""
        for ch in ln:
            if ch.isspace():
                if cur: parts.append(mp[cur]); cur = ""
                parts.append(ch)
            else:
                cur += ch
        if cur: parts.append(mp[cur])
        out.append("".join(parts))
    return "\n".join(out)


def run(binary, name, timeout, relab=True):
    """returns (ok, seconds, stdout)"""
    d = tempfile.mkdtemp(prefix="nn")
    try:
        text = open(f"{HID}/inputs/{name}.in").read()
        if relab:
            text = relabel(text)
        p = os.path.join(d, "input_%08x.txt" % RNG.getrandbits(32))
        open(p, "w").write(text)
        env = {"PATH": "/usr/bin:/bin", "HOME": "/nonexistent", "TMPDIR": d, "LANG": "C"}
        t0 = time.perf_counter()
        try:
            r = subprocess.run([binary, p], capture_output=True, timeout=timeout, cwd=d, env=env)
        except subprocess.TimeoutExpired:
            return False, timeout, b""
        dt = time.perf_counter() - t0
        exp = open(f"{HID}/inputs/{name}.out", "rb").read()
        return (r.returncode == 0 and r.stdout == exp), dt, r.stdout
    finally:
        shutil.rmtree(d, ignore_errors=True)


def group(g, timeout=60):
    return [run(BIN, n, timeout)[0] for n in meta["groups"][g]]


M = Milestones()
M.add("a01", "correct on small random inputs", group("small", 20))
M.add("a02", "correct on edge cases (empty docs, ties, repeats, odd formatting)", group("edge", 20))
M.add("a03", "correct on medium inputs (N=300..420)", group("medium", 60))
large_ok, t_first, _ = run(BIN, "large0", 90)
M.add("a04", "correct on the large input (N=2500)", [large_ok])
ok5, t5, _ = run(BIN, "scale0", 20)
M.add("a05", "scale: N=10000 within 20 s", [ok5])

# baseline: build the ORIGINAL naive program here, time it once on the large input
speeds = [1.5, 2, 4, 8, 16, 32, 64, 128]
base_t = None
agent_t = []
if large_ok:
    if t_first > 0.0:
        agent_t.append(t_first)
    try:
        bd = tempfile.mkdtemp(prefix="bl")
        shutil.copytree(f"{HID}/baseline", f"{bd}/p")
        subprocess.run(["cargo", "build", "--release", "--offline", "--target-dir", f"{bd}/t"], cwd=f"{bd}/p",
                       capture_output=True, timeout=300, env=dict(os.environ, HOME=os.environ.get("HOME", "/tmp")))
        ok, base_t, _ = run(f"{bd}/t/release/nnjoin", "large0", 300)
        if not ok:
            base_t = None
    except Exception as e:
        print("baseline failed", repr(e), file=sys.stderr)
    # agent timings: 5 total, interleaved after baseline; skip when clearly not faster than 1.5x
    if base_t and base_t / max(t_first, 1e-6) >= 1.4:
        for _ in range(4):
            ok, dt, _ = run(BIN, "large0", 90)
            agent_t.append(dt if ok else 1e9)
med = statistics.median(agent_t) if agent_t else None
spd = (base_t / med) if (base_t and med and len(agent_t) >= 5) else 0.0
print(f"# baseline={base_t} agent_runs={agent_t} median={med} speedup={spd:.2f}", file=sys.stderr)
for thr in speeds:
    M.add("s%03d" % int(thr * 10), f"speedup >= {thr}x vs original on N=2500 (median of 5)", [large_ok and spd >= thr])
M.emit()
