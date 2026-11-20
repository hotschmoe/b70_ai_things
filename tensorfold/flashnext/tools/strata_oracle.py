#!/usr/bin/env python3
"""strata_oracle.py - Strata 0.2.16 reference ("oracle") tokens and logits for the TensorFold Flash-Next port.

Stdlib only (numpy is used when importable, e.g. in the Strata venv, only to make the dump parsing fast).
Written for GGUF-MAP.md section "Oracle".  NOT RUN against a GPU yet (see GGUF-MAP.md).

Subcommands
  prompts   write the 8 fixed prompts + token ids (CPU only; needs the Strata tokenizer: venv python, which has 'regex')
  run       start the engine (docker, via the caller's bin/gpu-run), drive its --serve line protocol, write the oracle
  compare   compare a TensorFold teacher-forced result file with the oracle (CPU only)
  http-config   write a server config (plain args) for an optional HTTP smoke on :18134 via nl_launch_0216.sh
  fake-engine   protocol stub used by 'run --engine-cmd' for a no-GPU self test of this client

Engine protocol (sycl/src/program/generate.cpp:10700-10790, 12980-13030; terminal/serve/server.py:2420-2436):
  stdin  : "GEN <max_new> [key=val ...] id,id,..."   keys used here: logprobs=20 (greedy is the default, no other keys)
  stdout : "READY <ctx> ..." once loaded; per token "T <id>" then "LP <logprob_of_that_token> id:logprob ..." (top-20);
           "DONE ..." ends the request; "ERR ..." on failure.   "QUIT" on stdin ends the engine.
Full per-position logits come from the debug tap STRATA_LOGITS_DUMP=<path> (sycl/src/core/verify.cpp:3263-3281):
  record = int32 magic 0x53474F4C, int32 T, int32 n_vocab, int32 record_no, int64 pos0, then T*n_vocab float32;
  row t of a record is the distribution AFTER the token at absolute position pos0+t (so it predicts position pos0+t+1).
"""
import argparse, array, heapq, json, math, os, signal, struct, subprocess, sys, threading, time, queue

REPO = "/mnt/vm_8tb/github/b70_ai_things"
STRATA_SRC = "/mnt/vm_8tb/b70/build/strata-0216/src"
CONFIG = REPO + "/strata/flash-next/dd/hotschmoe-dd-0216.json"
OUT_DEFAULT = "/mnt/vm_8tb/b70/results/tf-flashnext/oracle"
IMAGE = os.environ.get("NL_IMAGE", "strata-sycl-dev:020")
GPU_RUN = REPO + "/bin/gpu-run"
MAGIC = 0x53474F4C

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None

# --------------------------------------------------------------------------------------------- prompts
CHAT_TAIL = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"   # thinking off, as the chat template renders it


def chat(q):
    return "<|im_start|>user\n" + q + CHAT_TAIL


def long_qsa_prompt():
    # ~150 numbered facts (> 2051 tokens of context, so the QSA top-512-block selection is exercised) then a lookup question
    facts = []
    for i in range(1, 151):
        facts.append("Fact %d: the archive room on floor %d is painted with colour code C%d and holds %d boxes." % (i, i, (i * 37) % 997, i * 10))
    return chat("Read the following facts and answer the question at the end.\n\n" + "\n".join(facts) +
                "\n\nQuestion: what is the colour code of the archive room on floor 41, and how many boxes does it hold?")


PROMPTS = [
    ("p0_en_story_raw", "The history of the lighthouse at Cape Meridian begins in the winter of 1847, when"),
    ("p1_py_code_raw", 'def parse_config(path):\n    """Read a TOML file and return a dict."""\n    import tomllib\n'),
    ("p2_rust_code_raw", "// A bounded MPMC channel in Rust\nuse std::sync::{Arc, Mutex, Condvar};\nuse std::collections::VecDeque;\n\npub struct Channel<T> {"),
    ("p3_chat_sky", chat("Explain in three short paragraphs why the sky is blue, then name one common misconception about it.")),
    ("p4_chat_math", chat("What is 17 * 23 + 144 / 12? Show the steps.")),
    ("p5_multilingual_raw", "Le chat dort sur le canap\u00e9. \u4eca\u5929\u5929\u6c14\u5f88\u597d\uff0c\u6211\u4eec\u53bb\u516c\u56ed\u6563\u6b65\u3002 Der Zug nach Berlin f\u00e4hrt um"),
    ("p6_eos_reset_raw", "First document ends here.<|endoftext|>Second document: the quick brown fox. the quick brown fox. the quick brown fox jumps over"),
    ("p7_long_qsa_chat", long_qsa_prompt()),
]


def load_tokenizer(cfg):
    sys.path.insert(0, STRATA_SRC + "/terminal/tools")
    import strata_tokenizer as ST   # needs 'regex' (the Strata venv has it)
    from pathlib import Path
    t = Path(cfg["tokenizer"])
    vocab = json.loads((t / "vocab.json").read_text(encoding="utf-8"))
    tokens = [None] * len(vocab)
    for tok, i in vocab.items():
        tokens[i] = tok
    merges = (t / "merges.txt").read_text(encoding="utf-8").split("\n")
    types = json.loads((t / "token_type.json").read_text())
    tcfg = json.loads((t / "tokenizer.json").read_text(encoding="utf-8"))
    return ST.Tokenizer(tokens, merges, types, tcfg.get("pre") or "qwen35", tcfg.get("special_ids") or None)


def cmd_prompts(a):
    cfg = json.load(open(a.config))
    tok = load_tokenizer(cfg)
    out = []
    for name, text in PROMPTS:
        ids = tok.encode(text, parse_special=True)
        assert tok.decode(ids) == text, "tokenizer round trip failed for " + name
        out.append({"name": name, "text": text, "ids": ids})
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump({"tokenizer": cfg["tokenizer"], "add_bos": False, "prompts": out}, open(a.out, "w"))
    print("wrote", a.out, [(p["name"], len(p["ids"])) for p in out])


# --------------------------------------------------------------------------------------------- engine plan
# Lead A fix: a native IQ pack refuses to start without --spec T (T >= 2), so plain mode keeps the MTP drafter at the
# minimum depth (--spec 2, --mtp-max-t 2); greedy verify keeps greedy tokens, logits come from 2-row verify windows.
DROP_WITH_VALUE = {"--mtp-max-t", "--spec", "--conversation-cache-mib",
                   "--conversation-cache-slots", "--kv", "--kv-resident", "--max-context", "--prefill", "--suffix-draft",
                   "--lookup-chain", "--prompt-cache"}
DROP_FLAGS = {"--vision", "--batch-mtp"}
QUAL_DROP_WITH_VALUE = {"--conversation-cache-mib", "--conversation-cache-slots", "--prompt-cache"}


def build_plan(cfg, a):
    """The engine command line + env.  mode 'plain' = no spec / no MTP / fp16 KV / BF16 HC reads (cleanest logits);
    mode 'qualified' = the daily-driver arguments (spec 4 + MTP, int8 KV) minus server-managed flags."""
    argv, i = [], 0
    src = list(cfg["args"])
    drop_v = DROP_WITH_VALUE if a.mode == "plain" else QUAL_DROP_WITH_VALUE
    while i < len(src):
        t = src[i]
        if t in DROP_FLAGS:
            i += 1
        elif t in drop_v:
            i += 2
        else:
            argv.append(t)
            i += 1
    if a.mode == "plain":
        argv += ["--spec", "2", "--mtp-max-t", "2"]
        argv += ["--kv", a.kv, "--max-context", str(a.ctx), "--prefill", str(a.prefill), "--suffix-draft", "0",
                 "--lookup-chain", "0", "--prompt-cache", "0"]
    else:
        argv += ["--prompt-cache", "0"]
    env = dict(cfg.get("env") or {})
    if a.mode == "plain":
        env["STRATA_HC_Q8"] = str(a.hc_q8)
        env["STRATA_BATCH_MTP"] = "0"
        for k in ("STRATA_BATCH_MTP_DEPTH", "STRATA_BATCH_MTP_PAIR_HEAD", "STRATA_BATCH_MTP_PAIR_GRAPH"):
            env.pop(k, None)
    for kv in a.env or []:
        k, _, v = kv.partition("=")
        env[k] = v
    return {"exe": cfg["exe"], "cwd": cfg["cwd"], "args": argv, "env": env}


def docker_cmd(cfg, plan, name):
    nvme = os.path.dirname([x for x in plan["args"] if x.endswith(".gguf")][0])
    build = "/mnt/vm_8tb/b70/build/strata-ab"
    cmd = [GPU_RUN, "docker", "run", "--rm", "-i", "--name", name, "--network", "host"]
    if "--expert-parallel-device" in plan["args"]:
        cmd += ["--cap-add", "SYS_PTRACE"]            # as nl_launch_0216.sh
    cmd += ["--memory", "90g", "--device", "/dev/dri", "--ulimit", "memlock=-1",
            "-v", build + ":" + build, "-v", "/mnt/vm_8tb/b70/results:/mnt/vm_8tb/b70/results",
            "-v", "/mnt/vm_8tb/b70/build/strata-0216:/mnt/vm_8tb/b70/build/strata-0216",
            "-v", nvme + ":" + nvme + ":ro"]
    # shard 2 of the IQ3_XXS pack is a symlink into ../IQ3_S: mount every resolved shard directory too (Lead A fix)
    extra = sorted({os.path.dirname(os.path.realpath(x)) for x in plan["args"] if x.endswith(".gguf")} - {nvme})
    for d in extra:
        cmd += ["-v", d + ":" + d + ":ro"]
    base_env = {"ONEAPI_DEVICE_SELECTOR": "level_zero:*", "SYCL_CACHE_PERSISTENT": "0", "ZES_ENABLE_SYSMAN": "1",
                "STRATA_VERIFY_DEVICE_PLAN": "1", "STRATA_VERIFY_NO_HOST": "1", "STRATA_STAGE_TRIM": "1",
                "STRATA_KV_WIDE_HOST": "0", "STRATA_STAGER_THREADS": "12"}
    base_env.update(plan["env"])
    for k, v in base_env.items():
        cmd += ["-e", "%s=%s" % (k, v)]
    cmd += ["--entrypoint", "bash", IMAGE, "-lc",
            'source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1; cd "$1"; exe="$2"; shift 2; exec "$exe" --serve "$@"',
            "bash", plan["cwd"], plan["exe"]] + plan["args"]
    return cmd


class Engine:
    def __init__(self, cmd, log_path, env=None):
        self.log = open(log_path, "ab")
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log, text=True,
                                  bufsize=1, env=env)
        self.q = queue.Queue()
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self):
        for line in self.p.stdout:
            self.q.put(line.rstrip("\n"))
        self.q.put(None)

    def send(self, line):
        self.p.stdin.write(line + "\n")
        self.p.stdin.flush()

    def readline(self, timeout):
        try:
            line = self.q.get(timeout=timeout)
        except queue.Empty:
            raise TimeoutError("engine silent for %.0f s" % timeout)
        if line is None:
            raise RuntimeError("engine stdout closed (exit code %s)" % self.p.poll())
        return line

    def wait_ready(self, timeout):
        t0 = time.time()
        while True:
            line = self.readline(max(1.0, timeout - (time.time() - t0)))
            print("[engine]", line, flush=True)
            if line.startswith("READY"):
                return line

    def close(self, name):
        try:
            self.send("QUIT")
        except Exception:
            pass
        try:
            self.p.wait(timeout=120)
        except Exception:
            if name:
                subprocess.run(["docker", "stop", "-t", "30", name], check=False)
            try:
                self.p.wait(timeout=60)
            except Exception:
                self.p.kill()
        self.log.close()


def parse_lp(line):
    parts = line.split()
    if len(parts) < 2 or parts[1] == "nan":
        return None
    return {"logprob": float(parts[1]), "top": [(int(x.split(":")[0]), float(x.split(":")[1])) for x in parts[2:] if ":" in x]}


def request(eng, ids, max_new, timeout):
    eng.send("GEN %d logprobs=20 %s" % (max_new, ",".join(map(str, ids))))
    toks, lps, done = [], [], None
    t0 = time.time()
    while True:
        line = eng.readline(max(5.0, timeout - (time.time() - t0)))
        if line.startswith("T "):
            toks.append(int(line.split()[1]))
        elif line.startswith("LP "):
            lps.append(parse_lp(line))
        elif line.startswith("DONE"):
            done = line
            break
        elif line.startswith("ERR"):
            raise RuntimeError(line)
        # anything else (progress lines) is ignored
    return toks, lps, done


# --------------------------------------------------------------------------------------------- dump parsing
def read_dump_range(path, lo, hi):
    """{absolute position: float32 row} for the records wholly inside byte range [lo, hi)."""
    rows, n_vocab = {}, None
    with open(path, "rb") as f:
        f.seek(lo)
        while f.tell() < hi:
            h = f.read(24)
            if len(h) < 24:
                break
            magic, T, nv, rec, pos0 = struct.unpack("<iiiiq", h)
            if magic != MAGIC:
                raise RuntimeError("bad dump record magic at %d" % (f.tell() - 24))
            n_vocab = nv
            raw = f.read(T * nv * 4)
            arr = np.frombuffer(raw, dtype="<f4").reshape(T, nv) if np is not None else None
            if arr is None:
                a = array.array("f")
                a.frombytes(raw)
                arr = [a[t * nv:(t + 1) * nv] for t in range(T)]
            for t in range(T):
                rows[pos0 + t] = arr[t]
    return rows, n_vocab


def row_stats(row, k=64):
    if np is not None:
        r = np.asarray(row, dtype=np.float64)
        idx = np.argpartition(-r, k)[:k]
        idx = idx[np.argsort(-r[idx], kind="stable")]
        mx = float(r[idx[0]])
        lse = mx + math.log(float(np.exp(r - mx).sum()))
        return [(int(i), float(r[i])) for i in idx], lse
    top = heapq.nlargest(k, range(len(row)), key=row.__getitem__)
    mx = row[top[0]]
    lse = mx + math.log(sum(math.exp(x - mx) for x in row))
    return [(i, float(row[i])) for i in top], lse


# --------------------------------------------------------------------------------------------- run
def cmd_run(a):
    cfg = json.load(open(a.config))
    out = a.out
    os.makedirs(out + "/first_logits", exist_ok=True)
    pj = out + "/prompts.json"
    if not os.path.exists(pj):
        cmd_prompts(argparse.Namespace(config=a.config, out=pj))
    prompts = json.load(open(pj))["prompts"]
    if a.only:
        prompts = [p for p in prompts if p["name"] in a.only.split(",")]
    plan = build_plan(cfg, a)
    dump = None
    if not a.no_dump:
        dump = out + "/logits_dump.bin"
        if os.path.exists(dump):
            os.remove(dump)
        plan["env"]["STRATA_LOGITS_DUMP"] = dump
    name = "tf-strata-oracle"
    if a.engine_cmd:
        cmd, env = a.engine_cmd.split(), dict(os.environ, **plan["env"])
    else:
        subprocess.run(["docker", "rm", "-f", name], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        cmd, env = docker_cmd(cfg, plan, name), None
    json.dump({"plan": plan, "cmd": cmd, "mode": a.mode}, open(out + "/engine_plan.json", "w"), indent=1)
    if a.print_only:
        print(" ".join(subprocess.list2cmdline([c]) for c in cmd))
        return
    eng = Engine(cmd, out + "/engine.log", env)
    result = {"meta": {"mode": a.mode, "plan": plan, "date": time.strftime("%Y-%m-%dT%H:%M:%S"), "max_new": a.max_new,
                       "exe_stat": os.stat(plan["exe"]).st_size if os.path.exists(plan["exe"]) else None}, "prompts": []}
    try:
        result["meta"]["ready"] = eng.wait_ready(a.ready_timeout)
        for p in prompts + ([dict(prompts[0], name=prompts[0]["name"] + "__repeat")] if a.determinism_probe and prompts else []):
            lo = os.path.getsize(dump) if dump and os.path.exists(dump) else 0
            t0 = time.time()
            toks, lps, done = request(eng, p["ids"], a.max_new, a.req_timeout)
            dt = time.time() - t0
            hi = os.path.getsize(dump) if dump and os.path.exists(dump) else 0
            n = len(p["ids"])
            ent = {"name": p["name"], "n_prompt": n, "ids": p["ids"], "gen_ids": toks, "seconds": round(dt, 2), "done": done,
                   "stop": "length" if len(toks) >= a.max_new else "eos", "lp": lps}
            if dump and hi > lo:
                rows, nv = read_dump_range(dump, lo, hi)
                ent["n_vocab"] = nv
                # row at absolute position n-1+i predicts generated token i
                have = [n - 1 + i in rows for i in range(len(toks))]
                ent["dump_rows_found"] = sum(have)
                ent["topk64"] = []
                for i in range(len(toks)):
                    r = rows.get(n - 1 + i)
                    if r is None:
                        ent["topk64"].append(None)
                        continue
                    top, lse = row_stats(r)
                    ent["topk64"].append({"lse": lse, "top": top})
                    if i == 0:
                        fn = "first_logits/%s.f32" % p["name"]
                        (r.astype("<f4") if np is not None else r).tofile(out + "/" + fn)
                        ent["first_logits_f32"] = fn
                        ent["first_logits_vocab"] = nv
            result["prompts"].append(ent)
            print("[oracle] %-22s prompt=%5d gen=%3d %s %.1fs agree_argmax_vs_lp=%s" % (
                p["name"], n, len(toks), ent["stop"], dt,
                all(l is not None and l["top"] and l["top"][0][0] == t for l, t in zip(lps, toks))), flush=True)
            json.dump(result, open(out + "/oracle.json", "w"))
        if a.determinism_probe and len(result["prompts"]) >= 2 and result["prompts"][-1]["name"].endswith("__repeat"):
            r0, r1 = result["prompts"][0]["gen_ids"], result["prompts"][-1]["gen_ids"]
            first = next((i for i, (x, y) in enumerate(zip(r0, r1)) if x != y), None)
            result["meta"]["determinism"] = {"identical": r0 == r1, "first_difference": first, "len": [len(r0), len(r1)]}
            print("[oracle] determinism probe:", result["meta"]["determinism"], flush=True)
    finally:
        json.dump(result, open(out + "/oracle.json", "w"))
        eng.close(None if a.engine_cmd else name)
        if dump and os.path.exists(dump) and not a.keep_dump:
            os.remove(dump)
    print("[oracle] wrote", out + "/oracle.json")


def cmd_http_config(a):
    """Write a server config (same shape as hotschmoe-dd-0216.json) whose engine args are the oracle's 'plain' args, for an
    HTTP smoke on port 18134 through strata/flash-next/dd/nl_launch_0216.sh.  Not needed for the token/logit oracle."""
    cfg = json.load(open(a.config))
    plan = build_plan(cfg, a)
    cfg2 = dict(cfg)
    cfg2["args"], cfg2["env"], cfg2["parallel"] = plan["args"], plan["env"], 1
    cfg2.pop("vision", None)
    cfg2["log"] = a.out + "/engine-http.log"
    os.makedirs(a.out, exist_ok=True)
    path = a.out + "/hotschmoe-oracle-0216.json"
    json.dump(cfg2, open(path, "w"), indent=2)
    print(path)


# --------------------------------------------------------------------------------------------- compare
def cmd_compare(a):
    orc = {p["name"]: p for p in json.load(open(a.oracle))["prompts"]}
    tf = {p["name"]: p for p in json.load(open(a.tf))["prompts"]}
    tot = agree = near = 0
    for name, o in orc.items():
        if name not in tf or name.endswith("__repeat"):
            continue
        t1 = tf[name].get("top1", [])
        g = o["gen_ids"]
        n = min(len(g), len(t1))
        ok = bad = nt = 0
        first_bad = None
        for i in range(n):
            if t1[i] == g[i]:
                ok += 1
                continue
            top = (o["lp"][i] or {}).get("top") or []
            gap = (top[0][1] - top[1][1]) if len(top) >= 2 else None   # oracle top1-top2 logprob gap
            if gap is not None and gap < a.tie_gap:
                nt += 1
            else:
                bad += 1
                first_bad = first_bad if first_bad is not None else i
        tot += n
        agree += ok
        near += nt
        print("%-22s positions=%3d top1_agree=%3d near_tie_miss=%2d other_miss=%2d first_other_miss=%s" % (name, n, ok, nt, bad, first_bad))
        if "topk" in tf[name] and o.get("lp"):
            kls = []
            for i in range(n):
                lp, tk = o["lp"][i], tf[name]["topk"][i]
                if not lp or not lp["top"] or not tk:
                    continue
                po = {t: v for t, v in lp["top"]}
                pt = {t: v for t, v in tk}
                ids = [t for t in po if t in pt]
                if len(ids) < 5:
                    continue
                zo = sum(math.exp(po[t]) for t in ids)
                zt = sum(math.exp(pt[t]) for t in ids)
                kls.append(sum((math.exp(po[t]) / zo) * (math.log(math.exp(po[t]) / zo) - math.log(math.exp(pt[t]) / zt)) for t in ids))
            if kls:
                print("    KL(oracle||tf) on shared top-20 (renormalised): mean=%.5f max=%.5f over %d positions" % (sum(kls) / len(kls), max(kls), len(kls)))
        ff = tf[name].get("first_logits_f32")
        if ff and o.get("first_logits_f32") and np is not None:
            x = np.fromfile(os.path.dirname(a.oracle) + "/" + o["first_logits_f32"], dtype="<f4").astype(np.float64)
            y = np.fromfile(ff, dtype="<f4").astype(np.float64)
            if x.shape == y.shape:
                lx = x - (x.max() + math.log(np.exp(x - x.max()).sum()))
                ly = y - (y.max() + math.log(np.exp(y - y.max()).sum()))
                px = np.exp(lx)
                print("    first-token: top1 %s, KL(oracle||tf) full vocab = %.6f, max|dlogprob| over oracle top-64 = %.4f" % (
                    "same" if x.argmax() == y.argmax() else "DIFFERENT", float((px * (lx - ly)).sum()),
                    float(np.abs(lx[np.argsort(-x)[:64]] - ly[np.argsort(-x)[:64]]).max())))
    print("TOTAL positions=%d top1_agree=%d (%.2f%%) near_tie_misses=%d" % (tot, agree, 100.0 * agree / max(tot, 1), near))
    if a.min_top1 is not None and tot and agree / tot < a.min_top1:
        sys.exit(1)


# --------------------------------------------------------------------------------------------- fake engine (self test)
def cmd_fake_engine(a):
    nv = int(os.environ.get("FAKE_VOCAB", "1000"))
    dump = os.environ.get("STRATA_LOGITS_DUMP")
    f = open(dump, "wb") if dump else None
    rec = 0
    print("READY 4096 stop", flush=True)
    for line in sys.stdin:
        line = line.strip()
        if line == "QUIT":
            break
        if not line.startswith("GEN "):
            print("ERR expected GEN", flush=True)
            continue
        parts = line.split()
        max_new = int(parts[1])
        ids = [int(x) for x in parts[-1].split(",")]
        pos = len(ids) - 1
        cur = ids[-1]
        for i in range(max_new):
            row = [((cur * 31 + j * 7) % 97) / 10.0 for j in range(nv)]
            tok = max(range(nv), key=row.__getitem__)
            if f:
                f.write(struct.pack("<iiiiq", MAGIC, 1, nv, rec, pos + i) + struct.pack("<%df" % nv, *row))
                f.flush()
                rec += 1
            mx = row[tok]
            lse = mx + math.log(sum(math.exp(x - mx) for x in row))
            top = sorted(range(nv), key=lambda j: -row[j])[:20]
            print("T %d" % tok, flush=True)
            print("LP %.6f %s" % (row[tok] - lse, " ".join("%d:%.6f" % (j, row[j] - lse) for j in top)), flush=True)
            cur = tok
        print("DONE %d %d 0.0" % (max_new, len(ids)), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prompts"); p.add_argument("--config", default=CONFIG); p.add_argument("--out", default=OUT_DEFAULT + "/prompts.json")
    p.set_defaults(f=cmd_prompts)
    r = sub.add_parser("run")
    r.add_argument("--config", default=CONFIG); r.add_argument("--out", default=OUT_DEFAULT)
    r.add_argument("--mode", choices=["plain", "qualified"], default="plain")
    r.add_argument("--max-new", type=int, default=256); r.add_argument("--only", default="")
    r.add_argument("--kv", default="fp16"); r.add_argument("--ctx", type=int, default=16384); r.add_argument("--prefill", type=int, default=12288)
    r.add_argument("--hc-q8", type=int, default=0, help="STRATA_HC_Q8 in plain mode (0 = BF16 HC reads, the exact path)")
    r.add_argument("--env", action="append", help="extra engine env K=V (repeatable)")
    r.add_argument("--no-dump", action="store_true", help="LP top-20 only; no STRATA_LOGITS_DUMP full logits")
    r.add_argument("--keep-dump", action="store_true")
    r.add_argument("--no-determinism-probe", dest="determinism_probe", action="store_false")
    r.add_argument("--ready-timeout", type=float, default=1800.0); r.add_argument("--req-timeout", type=float, default=900.0)
    r.add_argument("--engine-cmd", default="", help="test only: run this instead of docker (e.g. fake-engine)")
    r.add_argument("--print-only", action="store_true", help="write engine_plan.json, print the command, start nothing")
    r.set_defaults(f=cmd_run)
    h = sub.add_parser("http-config"); h.add_argument("--config", default=CONFIG); h.add_argument("--out", default=OUT_DEFAULT)
    h.add_argument("--mode", choices=["plain", "qualified"], default="plain"); h.add_argument("--kv", default="fp16")
    h.add_argument("--ctx", type=int, default=16384); h.add_argument("--prefill", type=int, default=12288)
    h.add_argument("--hc-q8", type=int, default=0); h.add_argument("--env", action="append")
    h.set_defaults(f=cmd_http_config)
    c = sub.add_parser("compare")
    c.add_argument("--oracle", default=OUT_DEFAULT + "/oracle.json"); c.add_argument("--tf", required=True)
    c.add_argument("--tie-gap", type=float, default=0.05, help="oracle top1-top2 logprob gap below which a miss is a near tie")
    c.add_argument("--min-top1", type=float, default=None)
    c.set_defaults(f=cmd_compare)
    fe = sub.add_parser("fake-engine"); fe.set_defaults(f=cmd_fake_engine)
    a = ap.parse_args()
    a.f(a)


if __name__ == "__main__":
    main()
