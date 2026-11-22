#!/usr/bin/env python3
"""Realistic mixed traffic for the hotschmoe-dd release validation (worker REL, 2026-11-22). CPU-side client only.

Talks to the frontdoor (default http://127.0.0.1:18080/v1, Bearer key from --api-key-file, model hotschmoe-dd) for
--minutes (default 60) and scrapes the backend /metrics (default http://127.0.0.1:18124/metrics) before, every
--sample-s and after. It never touches a GPU and never prints the key.

Traffic (all knobs below are module constants; --seed makes a run reproducible):
  users      up to --max-concurrency (8) user threads; user i is active while i < target(t), where target follows
             SCHEDULE (1 -> 8 -> 1 with two peaks at 8) plus +-1 jitter per 2 minutes. Active users run conversations
             with exponential think time between turns and idle time between conversations (Poisson-ish arrivals).
  agent      35%: system prompt + 5 tools (qwen3-coder style via OpenAI tools) + 28-45K tokens of repo files; the
             model's tool calls get synthetic results (read_file returns corpus files), so the prompt grows to a
             45-90K target over up to 14 turns. Follow-up turns resend the whole history (prompt cache).
  docqa      20%: 30-90K tokens of corpus with a needle codeword; question, then 1-2 cached follow-ups.
  chat       30%: short chat, 40% with a 3-12K code file, 3-6 turns.
  image      15%: generated PNG circles (color + position known) or the EXIF JPEG / WebP fixtures; 2-3 turns,
             a second image in some conversations.
  per request: stream 75% / non-stream 25%; sampling card defaults (no temperature) 45%, temperature 0.6 35%,
             greedy 20%; reasoning default (server medium) 55%, low 15%, high 10%, effort none 10%, thinking off
             (chat_template_kwargs) 10%; 5% omit max_tokens (server fits it). prompt + max_tokens <= --context.
  corpus     local repo docs and source (--corpus-dir, default the repo's docs/tensorfold/vllm/sglang/strata/bin/
             kernels/evals plus the frozen TensorFold zig/src), no network.

Outputs in --out (default /mnt/vm_8tb/b70/results/tf-flashnext/workers/REL/<run-name>/):
  requests.jsonl   one record per request: class, turn, mode, timings, token counts, finish, hashes/lengths only
  telemetry.jsonl  every --sample-s: free -b (RAM, swap), server RSS/swap (/proc), VRAM per card (xe drm fdinfo of
                   the server process, --no-vram-fdinfo to skip), /metrics gauges, card temps (sysfs), in-flight
  metrics_before.txt, metrics_after.txt   raw /metrics scrapes
  debug.jsonl      short excerpts (120 chars) of synthetic replies, for eyeballing only
  fixtures/        the generated images
  summary.json, summary.txt   gates (HARD -> exit 1) and WARN items

HARD gates: zero errors (HTTP != 200, exceptions, stalls); every finish in stop/length/tool_calls; every stream ends
with a finish chunk, a usage chunk and [DONE]; every tool call has an offered name and JSON-object arguments; every
image request that stopped has an answer (and >= 90% of image requests answer); cached_tokens > 0 on at least one
follow-up turn; no fatal lines in --server-log since the start; at least --min-requests requests.
WARN: memory growth trends (host used, swap, server RSS, VRAM), latency sanity, needle and image correctness,
length-finish share, cached share on follow-ups, /metrics non-200, malformed tool envelopes, disconnections.

--dry-run starts release_traffic_mock.py in-process (fake key, fake /metrics) and scales think times by 0.1:
    python3 tensorfold/flashnext/dd/release_traffic.py --dry-run --minutes 2 --run-name mock-2min
"""
import argparse, base64, hashlib, json, os, random, re, statistics, struct, subprocess, sys, threading, time
import urllib.error, urllib.request, zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from api_gate import sview  # noqa: E402  (stream event folding, shared with the API gates)
from gpu_window_checks import FATAL, IGNORE  # noqa: E402  (fatal server-log markers)
from soak_monitor import BAD, HTTP_LINE, IGNORE_BAD, TF_DONE, cards, pct  # noqa: E402

REPO = HERE.parents[2]
MODEL = "hotschmoe-dd"
RESULTS = Path("/mnt/vm_8tb/b70/results/tf-flashnext/workers/REL")
CLASSES = [("agent", 0.35), ("chat", 0.30), ("docqa", 0.20), ("image", 0.15)]
SAMPLING = [("card", 0.45), ("t0.6", 0.35), ("greedy", 0.20)]
EFFORT = [("default", 0.55), ("low", 0.15), ("high", 0.10), ("none", 0.10), ("off", 0.10)]
IMAGE_EFFORT = [("off", 0.4), ("none", 0.1), ("low", 0.3), ("default", 0.2)]
SCHEDULE = [(0.0, 1), (0.06, 2), (0.15, 3), (0.25, 5), (0.33, 3), (0.42, 4), (0.5, 6), (0.58, 8), (0.66, 5),
            (0.75, 3), (0.82, 6), (0.88, 8), (0.94, 4), (1.0, 2)]
THINK_S = {"tool": 2.0, "agent": 15.0, "chat": 30.0, "docqa": 25.0, "image": 20.0}  # exponential means
IDLE_S = 30.0
MAXTOK = {"agent": (8192, 2048), "chat": (3072, 1024), "docqa": (4096, 512), "image": (4096, 256)}  # thinking, off
NORMAL_FINISH = ("stop", "length", "tool_calls")
SKIP_DIRS = {".git", "archive", "models", "__pycache__", "node_modules", "results", "zig-out", ".zig-cache", "prebuilt"}
EXTS = {".md", ".py", ".sh", ".zig", ".txt", ".toml", ".yaml", ".yml"}
SECRETISH = re.compile(r"secret|key|token|credential|passw|\.env$", re.I)

TOOLS = [
    {"type": "function", "function": {"name": "read_file", "description": "Read a file of the repository.",
     "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "repository path"},
                    "start_line": {"type": "integer"}, "max_lines": {"type": "integer"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "search_code", "description": "Search the repository for a regex.",
     "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}, "path": {"type": "string"}},
                    "required": ["pattern"]}}},
    {"type": "function", "function": {"name": "list_dir", "description": "List files under a directory.",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "run_command", "description": "Run a shell command in the repository (tests, linters).",
     "parameters": {"type": "object", "properties": {"command": {"type": "string"}, "timeout_s": {"type": "integer"}},
                    "required": ["command"]}}},
    {"type": "function", "function": {"name": "write_file", "description": "Write a file.",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                    "required": ["path", "content"]}}}]
TOOL_NAMES = {t["function"]["name"] for t in TOOLS}
SYSTEM_AGENT = ("You are a careful coding agent working in a local repository. Use the tools to inspect files before "
                "you answer, one or two calls at a time. Think briefly. When you know enough, answer with a short, "
                "concrete plan that names files and functions.")
AGENT_TASKS = [
    "Find how KV cache eviction to host RAM is configured and list every environment flag involved.",
    "Explain how the server decides the reasoning effort and thinking budget of a request.",
    "Find the code path that writes the per-request done line in the server log and describe its fields.",
    "Review the GPU window scripts for steps that could leave both cards leased after a failure.",
    "Find where prompt cache budgets are set and propose a safe way to raise them.",
    "Locate the tool-call parsing code and describe what happens to a malformed call.",
    "Summarize how the rollback path restores the previous daily driver, step by step.",
    "Find where image input is decoded and which formats are supported."]
AGENT_FOLLOW = ["Good. Now check the related tests and tell me what is not covered.",
                "Look at one more file that could break this, then update the plan.",
                "Continue. Be concise.",
                "What would you change first, and why? Read the file you would change."]
CHAT_TOPICS = [
    ("Write a Python function that merges overlapping intervals, with tests.", ["Now make it handle open intervals.", "What is the complexity?"]),
    ("How do I profile a slow Rust program on Linux?", ["Show a flamegraph workflow.", "And for async code?"]),
    ("Explain the difference between a mutex and a semaphore with a Go example.", ["When would a channel be better?", "Show a deadlock and its fix."]),
    ("My Windows 11 laptop loses Wi-Fi after sleep. What should I check?", ["It is an Intel AX211.", "Give me PowerShell commands to collect logs."]),
    ("Write a Dart class for an LRU cache.", ["Add TTL expiry.", "Write unit tests."]),
    ("Review this idea: store sessions in SQLite with WAL mode for a small web app.", ["What about concurrent writers?", "How do I back it up online?"]),
    ("Explain what this code does and point out bugs.", ["Rewrite the worst function.", "Add type hints."]),
    ("Give me a bash one-liner to find the 10 largest files under a directory.", ["Exclude .git.", "Make it work on macOS too."])]
CODE_FOLLOW = ["Explain what this code does and point out risky parts.", "Suggest two refactors.", "Write a test for the main function."]


def pick(rng, table):
    x, acc = rng.random(), 0.0
    for name, w in table:
        acc += w
        if x < acc:
            return name
    return table[-1][0]


def sha(s):
    return hashlib.sha256(s.encode("utf-8", "replace")).hexdigest()[:12]


# ---------------------------------------------------------------- corpus and fixtures
class Corpus:
    def __init__(self, roots, cap_chars=40_000_000):
        self.docs, total = [], 0
        for root in roots:
            root = Path(root)
            if not root.exists():
                continue
            for dirpath, dirnames, files in os.walk(root):
                dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS and not d.startswith("."))
                for f in sorted(files):
                    p = Path(dirpath, f)
                    if p.suffix not in EXTS or SECRETISH.search(f):
                        continue
                    try:
                        if not 512 <= p.stat().st_size <= 256_000:
                            continue
                        text = p.read_text(errors="replace")
                    except OSError:
                        continue
                    label = str(p.relative_to(root.parent)) if root.parent != p else str(p)
                    self.docs.append((label, text))
                    total += len(text)
                    if total >= cap_chars:
                        break
        self.chars = total
        self.by_path = {p: t for p, t in self.docs}
        if len(self.docs) < 20:
            raise SystemExit(f"corpus too small: {len(self.docs)} files")

    def files(self, rng, chars, exclude=()):
        out, n = [], 0
        while n < chars:
            p, t = self.docs[rng.randrange(len(self.docs))]
            if p in exclude or any(p == q for q, _ in out):
                continue
            t = t[: chars - n + 200]
            out.append((p, t))
            n += len(t)
        return out

    def get(self, rng, path, chars):
        t = self.by_path.get(path)
        if t is None:
            path, t = self.docs[rng.randrange(len(self.docs))]
            t = f"(no exact match; closest file {path})\n" + t
        return t[:chars]

    def grep(self, rng, pattern, n=30):
        try:
            rx = re.compile(pattern[:200])
        except re.error:
            rx = re.compile(re.escape(pattern[:200]))
        hits, start = [], rng.randrange(len(self.docs))
        for k in range(len(self.docs)):
            p, t = self.docs[(start + k) % len(self.docs)]
            for i, line in enumerate(t.splitlines()):
                if rx.search(line):
                    hits.append(f"{p}:{i + 1}: {line[:200]}")
                    if len(hits) >= n:
                        return "\n".join(hits)
        return "\n".join(hits) or "no matches"


def block(files):
    return "\n".join(f'<file path="{p}">\n{t}\n</file>' for p, t in files)


def png(w, h, color, pos):
    """White PNG with one filled circle (stdlib only)."""
    r = min(w, h) // 6
    cx, cy = {"left": (w // 5, h // 2), "right": (4 * w // 5, h // 2), "top": (w // 2, h // 5),
              "bottom": (w // 2, 4 * h // 5), "center": (w // 2, h // 2)}[pos]
    bg, fg, raw = b"\xff\xff\xff", bytes(color), bytearray()
    for y in range(h):
        dy = y - cy
        row = bytearray(bg * w)
        if abs(dy) <= r:
            half = int((r * r - dy * dy) ** 0.5)
            x0, x1 = max(0, cx - half), min(w, cx + half + 1)
            row[3 * x0:3 * x1] = fg * (x1 - x0)
        raw += b"\x00" + row
    chunk = lambda t, d: struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)  # noqa: E731
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) +
            chunk(b"IDAT", zlib.compress(bytes(raw), 6)) + chunk(b"IEND", b""))


COLORS = {"red": (220, 30, 30), "blue": (30, 60, 220), "green": (30, 170, 60), "yellow": (235, 200, 20)}


def make_fixtures(outdir, rng):
    outdir.mkdir(parents=True, exist_ok=True)
    fx = []
    for i, (w, h) in enumerate([(224, 224), (320, 240), (512, 384), (448, 448), (256, 512), (384, 288)] * 2):
        color, pos = rng.choice(list(COLORS)), rng.choice(["left", "right", "top", "bottom", "center"])
        data = png(w, h, COLORS[color], pos)
        (outdir / f"circle_{i:02d}_{color}_{pos}_{w}x{h}.png").write_bytes(data)
        fx.append({"mime": "image/png", "data": data, "color": color, "pos": pos, "kind": "circle"})
    fixdir = HERE / "fixtures"
    for name, mime, pos in (("vis_red_left_exif6.jpg", "image/jpeg", "left"), ("vis_red_right.webp", "image/webp", "right")):
        if (fixdir / name).exists():
            fx.append({"mime": mime, "data": (fixdir / name).read_bytes(), "color": "red", "pos": pos, "kind": "square"})
    return fx


# ---------------------------------------------------------------- the run
class Run:
    def __init__(self, a):
        self.a, self.lock = a, threading.Lock()
        self.out = Path(a.out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.key = Path(a.api_key_file).read_text().strip()
        self.rng = random.Random(a.seed)
        self.corpus = Corpus(a.corpus_dir)
        self.fixtures = make_fixtures(self.out / "fixtures", self.rng)
        self.ratio = 3.3  # chars per prompt token, calibrated from usage
        self.inflight, self.inflight_tokens = 0, 0
        self.max_inflight, self.max_inflight_tokens, self.conc_hist = 0, 0, {}
        self.records, self.errors, self.telemetry = [], [], []
        self.files = {k: open(self.out / f"{k}.jsonl", "a") for k in ("requests", "telemetry", "debug")}
        self.stop = threading.Event()
        self.t0 = self.deadline = None
        self.conv_n = 0

    # -- helpers
    def write(self, kind, rec):
        with self.lock:
            self.files[kind].write(json.dumps(rec) + "\n")
            self.files[kind].flush()

    def frac(self):
        return min(1.0, (time.time() - self.t0) / (self.deadline - self.t0))

    def target(self):
        f = self.frac()
        for (f0, c0), (f1, c1) in zip(SCHEDULE, SCHEDULE[1:]):
            if f0 <= f <= f1:
                base = c0 + (c1 - c0) * (f - f0) / (f1 - f0)
                break
        jit = random.Random(self.a.seed * 1000 + int((time.time() - self.t0) // 120)).choice((-1, 0, 0, 1))
        return max(1, min(self.a.max_concurrency, round(base * self.a.max_concurrency / 8) + jit))

    def sleep(self, mean):
        end = time.time() + random.expovariate(1.0 / max(1e-3, mean * self.a.think_scale))
        while time.time() < end and not self.stop.is_set() and time.time() < self.deadline:
            time.sleep(min(0.5, end - time.time()))

    def est_tokens(self, msgs, tools):
        chars, images = 0, 0
        for m in msgs:
            c = m.get("content")
            if isinstance(c, str):
                chars += len(c)
            elif isinstance(c, list):
                for part in c:
                    chars += len(part.get("text") or "")
                    images += part.get("type") == "image_url"
            chars += len(m.get("reasoning_content") or "") + len(json.dumps(m.get("tool_calls") or ""))
        if tools:
            chars += len(json.dumps(tools))
        return chars, images, int(chars / self.ratio) + 300 * images + 16 * len(msgs)

    def http(self, method, url, body=None, timeout=10):
        h = {"Content-Type": "application/json", "Authorization": "Bearer " + self.key}
        r = urllib.request.Request(url, json.dumps(body).encode() if body is not None else None, h, method=method)
        return urllib.request.urlopen(r, timeout=timeout)

    def scrape(self):
        try:
            with urllib.request.urlopen(self.a.metrics_url, timeout=5) as r:
                return r.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as e:
            if e.code != 401:
                return f"# scrape failed: HTTP {e.code}\n"
        except Exception as e:  # noqa: BLE001
            return f"# scrape failed: {type(e).__name__}\n"
        try:
            with self.http("GET", self.a.metrics_url, timeout=5) as r:
                return r.read().decode("utf-8", "replace")
        except Exception as e:  # noqa: BLE001
            return f"# scrape failed with key: {type(e).__name__}\n"

    # -- one request
    def request(self, conv, msgs, tools=None, max_tokens=None, effort=None, sampling=None, stream=None, extra=None, check=None):
        rng = conv["rng"]
        effort = effort or pick(rng, EFFORT)
        sampling = sampling or pick(rng, SAMPLING)
        stream = rng.random() < 0.75 if stream is None else stream
        thinking = effort not in ("none", "off")
        if max_tokens is None:
            mt = MAXTOK[conv["cls"]][0 if thinking else 1]
            max_tokens = int(mt * 1.5) if effort == "high" else mt
            if rng.random() < 0.05 and self.frac() < 0.85:
                max_tokens = 0  # omitted: the server fits it to the window
        chars, images, est = self.est_tokens(msgs, tools)
        if est + (max_tokens or 16384) > self.a.context - 4096:
            return None, None
        body = {"model": MODEL, "messages": msgs}
        if max_tokens:
            body["max_tokens"] = max_tokens
        if tools:
            body["tools"] = tools
        if sampling == "t0.6":
            body["temperature"] = 0.6
        elif sampling == "greedy":
            body["temperature"] = 0
        if effort in ("low", "high", "none"):
            body["reasoning_effort"] = effort
        elif effort == "off":
            body["chat_template_kwargs"] = {"enable_thinking": False}
        if stream:
            body.update(stream=True, stream_options={"include_usage": True})
        body.update(extra or {})
        rec = {"t": round(time.time() - self.t0, 2), "conv": conv["id"], "user": conv["user"], "cls": conv["cls"],
               "turn": conv["turn"], "phase": "cold" if conv["turn"] == 0 else "follow", "stream": stream,
               "sampling": sampling, "effort": effort, "max_tokens": max_tokens or None, "images": images,
               "tools": bool(tools), "est_prompt_tokens": est, "msgs": len(msgs)}
        with self.lock:
            self.inflight += 1
            self.inflight_tokens += est
            self.max_inflight = max(self.max_inflight, self.inflight)
            self.max_inflight_tokens = max(self.max_inflight_tokens, self.inflight_tokens)
            rec["inflight"] = self.inflight
        t0 = time.time()
        msg, err = None, None
        try:
            with self.http("POST", self.a.base_url.rstrip("/") + "/chat/completions", body, timeout=self.a.stall_s) as resp:
                rec["http"] = resp.status
                if stream:
                    msg = self.read_stream(resp, rec, t0)
                else:
                    r = json.load(resp)
                    ch = (r.get("choices") or [{}])[0]
                    m = ch.get("message") or {}
                    calls = {i: {"name": (c.get("function") or {}).get("name") or "", "args": (c.get("function") or {}).get("arguments") or "",
                                 "id": c.get("id") or ""} for i, c in enumerate(m.get("tool_calls") or [])}
                    msg = {"content": m.get("content") or "", "reasoning": m.get("reasoning_content") or "", "calls": calls,
                           "finish": ch.get("finish_reason"), "usage": r.get("usage")}
                    if "error" in r:
                        err = "error_object"
        except urllib.error.HTTPError as e:
            rec["http"] = e.code
            err = f"http_{e.code}"
            rec["error_detail"] = (e.read(300) or b"").decode("utf-8", "replace")[:200]
        except Exception as e:  # noqa: BLE001
            err = type(e).__name__
            rec["error_detail"] = str(e)[:200]
        t1 = time.time()
        with self.lock:
            self.inflight -= 1
            self.inflight_tokens -= est
        rec["latency_s"] = round(t1 - t0, 3)
        if msg is not None:
            usage = msg.get("usage") or {}
            rec.update(prompt_tokens=usage.get("prompt_tokens"), completion_tokens=usage.get("completion_tokens"),
                       cached_tokens=((usage.get("prompt_tokens_details") or {}).get("cached_tokens")),
                       finish=msg["finish"], content_len=len(msg["content"]), content_sha=sha(msg["content"]),
                       reasoning_len=len(msg["reasoning"]), tool_calls=len(msg["calls"]))
            if not err and stream and msg.get("stream_error"):
                err = msg["stream_error"]
            if not err and msg["finish"] not in NORMAL_FINISH:
                err = f"finish_{msg['finish']}"
            bad = []
            for c in msg["calls"].values():
                try:
                    args = json.loads(c["args"] or "")
                    ok = isinstance(args, dict) and c["name"] in TOOL_NAMES and (tools is not None)
                except ValueError:
                    args, ok = None, False
                c["parsed"] = args if ok else None
                if not ok:
                    bad.append(c["name"][:40])
            rec["tool_json_bad"] = len(bad)
            if bad:
                rec["tool_bad_names"] = bad
            pt = rec.get("prompt_tokens") or 0
            if pt > 2000 and not err:
                with self.lock:
                    self.ratio = 0.8 * self.ratio + 0.2 * (chars / max(1, pt - 300 * images))
            if check:
                rec.update(check(msg))
            self.write("debug", {"conv": conv["id"], "turn": conv["turn"], "content": msg["content"][:120],
                                 "reasoning": msg["reasoning"][:80], "calls": [c["name"] for c in msg["calls"].values()]})
        if err:
            rec["error"] = err
            with self.lock:
                self.errors.append({k: rec.get(k) for k in ("t", "conv", "cls", "turn", "http", "error", "error_detail")})
        with self.lock:
            self.records.append(rec)
        self.write("requests", rec)
        return rec, (None if err else msg)

    def read_stream(self, resp, rec, t0):
        events, first, last, gap, saw = [], None, None, 0.0, False
        while True:
            line = resp.readline()
            if not line:
                break
            if not line.startswith(b"data: "):
                continue
            p = line[6:].strip()
            if p == b"[DONE]":
                events.append("[DONE]")
                break
            ev = json.loads(p)
            events.append(ev)
            if "error" in ev:
                saw = True
            for ch in ev.get("choices", []):
                d = ch.get("delta") or {}
                if d.get("content") or d.get("reasoning_content") or d.get("tool_calls"):
                    now = time.time()
                    if first is None:
                        first = now
                    elif now - last > gap:
                        gap = now - last
                    last = now
        v = sview(events)
        msg = {"content": v["content"], "reasoning": v["reasoning"], "finish": v["finish"], "usage": v["usage"], "calls": v["calls"]}
        if saw:
            msg["stream_error"] = "stream_error_event"
        elif not v["done"]:
            msg["stream_error"] = "truncated_stream"
        elif not v["usage"]:
            msg["stream_error"] = "no_usage_chunk"
        elif v["finish"] is None:
            msg["stream_error"] = "no_finish_chunk"
        if first is not None:
            rec["ttft_s"] = round(first - t0, 3)
            rec["max_gap_s"] = round(gap, 3)
            n = (v["usage"] or {}).get("completion_tokens") or 0
            if n >= 16 and last > first:
                rec["decode_tps"] = round((n - 1) / (last - first), 2)
        return msg

    # -- conversation classes
    def assistant(self, msg, keep_reasoning):
        m = {"role": "assistant", "content": msg["content"]}
        if keep_reasoning and msg["reasoning"]:
            m["reasoning_content"] = msg["reasoning"]
        calls = [c for c in msg["calls"].values() if c.get("parsed") is not None]
        if calls:
            m["tool_calls"] = [{"id": c["id"] or f"call_{i}", "type": "function",
                                "function": {"name": c["name"], "arguments": c["args"]}} for i, c in enumerate(calls)]
        return m, calls

    def tool_result(self, rng, call):
        a = call["parsed"]
        n = call["name"]
        if n == "read_file":
            return self.corpus.get(rng, str(a.get("path", "")), int(rng.uniform(8_000, 30_000)))
        if n == "search_code":
            return self.corpus.grep(rng, str(a.get("pattern", "TODO")))
        if n == "list_dir":
            pre = str(a.get("path", "")).strip("./")
            hits = [p for p, _ in self.corpus.docs if p.startswith(pre)][:80] or [p for p, _ in self.corpus.docs[:40]]
            return "\n".join(hits)
        if n == "run_command":
            return f"$ {str(a.get('command', ''))[:200]}\n" + "\n".join(f"test_{i:03d} ... ok" for i in range(rng.randint(5, 60))) + "\nexit 0"
        return f"wrote {len(str(a.get('content', '')))} bytes to {str(a.get('path', ''))[:200]}"

    def agent(self, conv):
        rng = conv["rng"]
        first = self.corpus.files(rng, int(rng.uniform(28_000, 45_000) * self.ratio))
        others = [p for p, _ in self.corpus.files(rng, 400_000, exclude={p for p, _ in first})][:8]
        target = min(self.a.max_prompt_tokens, int(rng.uniform(45_000, 90_000)))
        keep = rng.random() < 0.5
        msgs = [{"role": "system", "content": SYSTEM_AGENT},
                {"role": "user", "content": f"Task: {rng.choice(AGENT_TASKS)}\n\nRelevant files follow.\n\n{block(first)}\n\n"
                                            f"Other files you may read with read_file: {', '.join(others)}.\n"
                                            "Start by calling read_file on one of them."}]
        for _ in range(14):
            rec, msg = self.request(conv, msgs, tools=TOOLS)
            if msg is None:
                return
            conv["turn"] += 1
            am, calls = self.assistant(msg, keep)
            msgs.append(am)
            if calls:
                for c in calls:
                    msgs.append({"role": "tool", "tool_call_id": am["tool_calls"][calls.index(c)]["id"], "content": self.tool_result(rng, c)})
                wait = THINK_S["tool"]
            else:
                text = rng.choice(AGENT_FOLLOW)
                if rng.random() < 0.5:
                    text += "\n\nThis file may also matter:\n" + block(self.corpus.files(rng, int(rng.uniform(5_000, 15_000) * self.ratio)))
                msgs.append({"role": "user", "content": text})
                wait = THINK_S["agent"]
            if self.est_tokens(msgs, TOOLS)[2] >= target or not self.active(conv):
                return
            self.sleep(wait)

    def docqa(self, conv):
        rng = conv["rng"]
        target = min(self.a.max_prompt_tokens, int(rng.uniform(30_000, 90_000)))
        files = self.corpus.files(rng, int(target * self.ratio * 0.97))
        code = f"{rng.choice(['amber', 'cobalt', 'juniper', 'saffron', 'tundra', 'violet'])}-{rng.randint(100, 999)}"
        k = rng.randrange(len(files))
        files[k] = (files[k][0], files[k][1] + f"\nNote: the release validation codeword for this bundle is {code}.\n")
        msgs = [{"role": "user", "content": block(files) + "\n\nWhat is the release validation codeword stated in the documents above? "
                                                            "Answer with the codeword only."}]
        follow = ["List three file paths that appear in the documents.", "Repeat the release validation codeword. Only the codeword."]
        for i in range(rng.randint(2, 3)):
            asks = "codeword" in msgs[-1]["content"][-120:]
            rec, msg = self.request(conv, msgs, check=(lambda m: {"needle_ok": code in m["content"]}) if asks else None)
            if msg is None:
                return
            conv["turn"] += 1
            msgs.append(self.assistant(msg, False)[0])
            if i >= len(follow) or not self.active(conv):
                return
            msgs.append({"role": "user", "content": follow[i]})
            self.sleep(THINK_S["docqa"])

    def chat(self, conv):
        rng = conv["rng"]
        q, follows = rng.choice(CHAT_TOPICS)
        if rng.random() < 0.4:
            q = rng.choice(CODE_FOLLOW) + "\n\n" + block(self.corpus.files(rng, int(rng.uniform(3_000, 12_000) * self.ratio)))
        msgs = [{"role": "user", "content": q}]
        for i in range(rng.randint(3, 6)):
            rec, msg = self.request(conv, msgs)
            if msg is None:
                return
            conv["turn"] += 1
            msgs.append(self.assistant(msg, False)[0])
            if not self.active(conv):
                return
            msgs.append({"role": "user", "content": follows[i] if i < len(follows) else "Thanks. Anything else I should know? Keep it short."})
            self.sleep(THINK_S["chat"])

    def image_part(self, fx):
        return {"type": "image_url", "image_url": {"url": f"data:{fx['mime']};base64," + base64.b64encode(fx["data"]).decode("ascii")}}

    def image(self, conv):
        rng = conv["rng"]
        fx = rng.choice(self.fixtures)
        shape = "circle" if fx["kind"] == "circle" else "red square"
        q = f"What color is the {shape}, and is it on the left, right, top, bottom or center of the image? Answer in one short sentence."
        parts = [{"type": "text", "text": q}, self.image_part(fx)]
        if rng.random() < 0.5:
            parts.reverse()
        msgs = [{"role": "user", "content": parts}]
        turns = [(fx, (fx["color"], fx["pos"])), (None, ("white",))]
        if rng.random() < 0.5:
            turns.append((rng.choice([f for f in self.fixtures if f["kind"] == "circle"]), None))
        for i in range(3):
            if i >= len(turns):
                return
            want = turns[i][1]

            def check(m, want=want):
                text = m["content"].lower()
                return {"image_answer": bool(text.strip()), **({"image_ok": all(w in text for w in want)} if want else {})}
            rec, msg = self.request(conv, msgs, effort=pick(rng, IMAGE_EFFORT), check=check)
            if msg is None:
                return
            conv["turn"] += 1
            msgs.append(self.assistant(msg, False)[0])
            if i + 1 >= len(turns) or not self.active(conv):
                return
            nxt = turns[i + 1]
            if nxt[0] is None:
                msgs.append({"role": "user", "content": "What color is the background of the image? One word."})
            else:
                turns[i + 1] = (nxt[0], (nxt[0]["color"],))
                msgs.append({"role": "user", "content": [{"type": "text", "text": "Here is a second image. What color is its circle? One word."},
                                                         self.image_part(nxt[0])]})
            self.sleep(THINK_S["image"])

    def active(self, conv):
        return not self.stop.is_set() and time.time() < self.deadline and conv["user"] < self.target()

    def user(self, idx):
        rng = random.Random(self.a.seed * 100 + idx)
        while not self.stop.is_set() and time.time() < self.deadline:
            if idx >= self.target():
                time.sleep(1.0)
                continue
            with self.lock:
                self.conv_n += 1
                cid = self.conv_n
            conv = {"id": cid, "user": idx, "cls": pick(rng, CLASSES), "turn": 0, "rng": random.Random(rng.random())}
            try:
                getattr(self, conv["cls"])(conv)
            except Exception as e:  # noqa: BLE001  (a harness bug must not kill the user thread silently)
                with self.lock:
                    self.errors.append({"conv": cid, "cls": conv["cls"], "error": "harness_" + type(e).__name__, "error_detail": str(e)[:200]})
            self.sleep(IDLE_S)

    # -- telemetry
    def server_procs(self):
        out = {"server": [], "vision": []}
        for p in Path("/proc").iterdir():
            if not p.name.isdigit():
                continue
            try:
                cmd = (p / "cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", "replace")
            except OSError:
                continue
            if self.a.server_match in cmd and " serve " in cmd:
                out["server"].append(int(p.name))
            elif "strata-vision" in cmd:
                out["vision"].append(int(p.name))
        return out

    @staticmethod
    def status_kb(pid, key):
        try:
            for line in Path(f"/proc/{pid}/status").read_text().splitlines():
                if line.startswith(key + ":"):
                    return int(line.split()[1]) * 1024
        except OSError:
            pass
        return None

    @staticmethod
    def vram(pid):
        """Resident VRAM per card from xe DRM fdinfo of one process (bytes, keyed by drm-pdev)."""
        seen, per = set(), {}
        for f in Path(f"/proc/{pid}/fdinfo").glob("*"):
            try:
                kv = dict(line.split(":", 1) for line in f.read_text().splitlines() if ":" in line)
            except OSError:
                continue
            if kv.get("drm-driver", "").strip() != "xe":
                continue
            ident = (kv.get("drm-pdev", "").strip(), kv.get("drm-client-id", "").strip())
            if ident in seen:
                continue
            seen.add(ident)
            for k, v in kv.items():
                m = re.match(r"drm-(resident|total)-(vram\d+)$", k)
                if m:
                    num, *unit = v.split()
                    mult = {"KiB": 1024, "MiB": 1 << 20, "GiB": 1 << 30}.get(unit[0] if unit else "", 1)
                    key = f"{ident[0]}:{m.group(1)}"
                    per[key] = per.get(key, 0) + int(num) * mult
        return per

    def sample(self):
        s = {"t": round(time.time() - self.t0, 1), "inflight": self.inflight, "target": self.target()}
        try:
            lines = subprocess.run(["free", "-b"], capture_output=True, text=True, timeout=5).stdout.splitlines()
            mem = [int(x) for x in lines[1].split()[1:]]
            swp = [int(x) for x in lines[2].split()[1:]]
            s.update(host_total=mem[0], host_used=mem[1], host_free=mem[2], host_available=mem[5], swap_total=swp[0], swap_used=swp[1])
        except Exception:  # noqa: BLE001
            pass
        procs = self.server_procs()
        if procs["server"]:
            pid = procs["server"][0]
            s.update(server_pid=pid, server_rss=self.status_kb(pid, "VmRSS"), server_swap=self.status_kb(pid, "VmSwap"))
            if self.a.vram_fdinfo:
                s["vram"] = self.vram(pid)
        kids = [p for p in procs["vision"] if procs["server"] and self.status_kb(p, "PPid") == procs["server"][0] * 1024]
        s["vision_rss"] = sum(self.status_kb(p, "VmRSS") or 0 for p in kids)  # the server's strata-vision children only
        m = parse_metrics(self.scrape())
        s["metrics"] = {k: v for k, v in m.items() if k.split("{")[0] in GAUGES}
        try:
            s["cards"] = {c["card"]: {k: c.get(k) for k in ("temp_pkg_c", "temp_vram_c", "act_mhz")} for c in cards()}
        except Exception:  # noqa: BLE001
            pass
        with self.lock:
            self.telemetry.append(s)
        self.write("telemetry", s)

    def monitor(self):
        nxt = time.time()
        while not self.stop.is_set():
            with self.lock:
                self.conc_hist[self.inflight] = self.conc_hist.get(self.inflight, 0) + 1
            if time.time() >= nxt:
                self.sample()
                nxt += self.a.sample_s
            time.sleep(1.0)

    # -- main
    def go(self):
        log = Path(self.a.server_log) if self.a.server_log else None
        log_off = log.stat().st_size if log and log.exists() else None
        fd = Path(self.a.frontdoor_log) if self.a.frontdoor_log else None
        fd_off = fd.stat().st_size if fd and fd.exists() else None
        before = self.scrape()
        (self.out / "metrics_before.txt").write_text(before)
        with self.http("GET", self.a.base_url.rstrip("/") + "/models") as r:
            ids = [m.get("id") for m in json.load(r).get("data", [])]
        if not ids or ids[0] != MODEL:
            raise SystemExit(f"/v1/models first id {ids[:1]} != {MODEL}")
        self.t0 = time.time()
        self.deadline = self.t0 + self.a.minutes * 60
        print(f"[release_traffic] start {time.strftime('%FT%TZ', time.gmtime(self.t0))} minutes={self.a.minutes} "
              f"corpus={len(self.corpus.docs)} files/{self.corpus.chars} chars fixtures={len(self.fixtures)} out={self.out}", flush=True)
        mon = threading.Thread(target=self.monitor, daemon=True)
        mon.start()
        users = [threading.Thread(target=self.user, args=(i,), daemon=True) for i in range(self.a.max_concurrency)]
        for u in users:
            u.start()
        try:
            last = 0
            while time.time() < self.deadline:
                time.sleep(1)
                if time.time() - last >= 60:
                    last = time.time()
                    print(f"[release_traffic] t={int(time.time() - self.t0)}s target={self.target()} inflight={self.inflight} "
                          f"requests={len(self.records)} errors={len(self.errors)}", flush=True)
            end = time.time() + self.a.drain_s
            for u in users:
                u.join(max(0.1, end - time.time()))
        except KeyboardInterrupt:
            print("[release_traffic] interrupted; summarizing what finished", flush=True)
        self.stop.set()
        unfinished = sum(u.is_alive() for u in users)
        self.sample()
        after = self.scrape()
        (self.out / "metrics_after.txt").write_text(after)
        logscan = scan_log(log, log_off) if log_off is not None else {"skipped": f"no log at {log}"}
        fdscan = scan_frontdoor(fd, fd_off) if fd_off is not None else {"skipped": f"no log at {fd}"}
        return summarize(self, parse_metrics(before), parse_metrics(after), logscan, fdscan, unfinished)


GAUGES = {"requests_running", "requests_waiting", "prompt_cache_bytes", "prompt_cache_states", "process_resident_bytes",
          "process_resident_anon_bytes", "device_memory_bytes", "pinned_memory_bytes", "generation_tokens_running"}
COUNTERS = ["requests_total", "client_disconnections_total", "kept_tool_calls_malformed_total", "kept_tool_calls_unoffered_total",
            "prompt_tokens_total", "prompt_tokens_cached_total", "generation_tokens_total", "preemptions_total",
            "mtp_drafted_total", "mtp_accepted_total"]


def parse_metrics(text):
    out = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        try:
            name, val = line.rsplit(" ", 1)
            out[name.replace("tensorfold:", "", 1)] = float(val)
        except ValueError:
            continue
    return out


def scan_log(path, off):
    with path.open("rb") as f:
        f.seek(off)
        lines = f.read().decode("utf-8", "replace").splitlines()
    fatal = [ln[:160] for ln in lines if (FATAL.search(ln) and not IGNORE.search(ln)) or (BAD.search(ln) and not IGNORE_BAD.search(ln))]
    done = [m.groupdict() for m in map(TF_DONE.search, lines) if m]
    fin = {}
    for g in done:
        fin[g["finish"]] = fin.get(g["finish"], 0) + 1
    tps = [float(g["tps"]) for g in done if int(g["tokens"]) >= 32]
    return {"lines": len(lines), "fatal": len(fatal), "fatal_head": fatal[:8], "done_lines": len(done), "done_finish": fin,
            "server_decode_tps_p50": round(statistics.median(tps), 2) if tps else None,
            "server_ttft_p50": round(statistics.median(float(g["ttft"]) for g in done), 3) if done else None}


def scan_frontdoor(path, off):
    with path.open("rb") as f:
        f.seek(off)
        lines = f.read().decode("utf-8", "replace").splitlines()
    st = {}
    for m in map(HTTP_LINE.search, lines):
        if m:
            st[m.group("status")] = st.get(m.group("status"), 0) + 1
    return {"status": st}


def dist(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    return {"n": len(xs), "p01": pct(xs, 0.01), "p50": pct(xs, 0.5), "p99": pct(xs, 0.99), "min": min(xs), "max": max(xs)}


def trend(tel, key, warm_s):
    pts = [(s["t"], s[key]) for s in tel if s.get(key) is not None and s["t"] >= warm_s]
    if len(pts) < 4:
        return None
    n, mt, mv = len(pts), sum(t for t, _ in pts) / len(pts), sum(v for _, v in pts) / len(pts)
    den = sum((t - mt) ** 2 for t, _ in pts) or 1.0
    slope = sum((t - mt) * (v - mv) for t, v in pts) / den
    return {"n": n, "first": pts[0][1], "last": pts[-1][1], "max": max(v for _, v in pts),
            "growth_fit": round(slope * (pts[-1][0] - pts[0][0])), "slope_per_h": round(slope * 3600)}


def summarize(run, mb, ma, logscan, fdscan, unfinished):
    a, recs, tel = run.a, run.records, run.telemetry
    gib = 1 << 30
    by = {}
    for r in recs:
        for k in ("all", f"{r['cls']}_{r['phase']}"):
            by.setdefault(k, []).append(r)
    classes = {k: {"requests": len(v), "ttft_s": dist([r.get("ttft_s") for r in v]), "decode_tps": dist([r.get("decode_tps") for r in v]),
                   "latency_s": dist([r.get("latency_s") for r in v]), "prompt_tokens": dist([r.get("prompt_tokens") for r in v]),
                   "cached_tokens": dist([r.get("cached_tokens") for r in v]),
                   "max_gap_s": dist([r.get("max_gap_s") for r in v])} for k, v in sorted(by.items())}
    ok = [r for r in recs if not r.get("error")]
    follow = [r for r in ok if r["phase"] == "follow"]
    cached_follow = [r for r in follow if (r.get("cached_tokens") or 0) > 0]
    imgs = [r for r in recs if r["cls"] == "image"]
    img_stop = [r for r in imgs if r.get("finish") == "stop"]
    needles = [r["needle_ok"] for r in recs if "needle_ok" in r]
    img_ok = [r["image_ok"] for r in recs if "image_ok" in r]
    mix = {k: {} for k in ("cls", "sampling", "effort", "stream", "finish")}
    for r in recs:
        for k in mix:
            mix[k][str(r.get(k))] = mix[k].get(str(r.get(k)), 0) + 1
    warm = 0.1 * a.minutes * 60  # skip the first 10% (cache fill) for trend fits
    trends = {k: trend(tel, k, warm) for k in ("host_used", "host_available", "swap_used", "server_rss", "server_swap", "vision_rss")}
    for s in tel:
        for k, v in (s.get("vram") or {}).items():
            s["vram_" + k] = v
        for k in ("process_resident_bytes", "prompt_cache_bytes", "device_memory_bytes"):
            if k in (s.get("metrics") or {}):
                s["m_" + k] = s["metrics"][k]
    for k in sorted({k for s in tel for k in s if k.startswith(("vram_", "m_"))}):
        trends[k] = trend(tel, k, warm)
    delta = {}
    for k in sorted(set(mb) | set(ma)):
        if k.split("{")[0] in COUNTERS:
            d = ma.get(k, 0) - mb.get(k, 0)
            if d:
                delta[k] = d
    max_ctx = max(((r.get("prompt_tokens") or 0) + (r.get("max_tokens") or 0) for r in ok), default=0)
    hard = {
        "zero_errors": (len(run.errors) == 0, {"errors": len(run.errors), "head": run.errors[:8]}),
        "finish_normal": (not any(str(r.get("error", "")).startswith("finish_") for r in recs), mix["finish"]),
        "streams_complete": (not any(str(r.get("error", "")).startswith(("truncated", "no_usage", "no_finish", "stream_error")) for r in recs),
                             {"streams": sum(r["stream"] for r in recs)}),
        "tool_json": (sum(r.get("tool_json_bad", 0) for r in recs) == 0,
                      {"tool_calls": sum(r.get("tool_calls", 0) for r in recs), "bad": sum(r.get("tool_json_bad", 0) for r in recs)}),
        "image_answers": (all(r.get("image_answer") for r in img_stop) and
                          (not imgs or sum(bool(r.get("image_answer")) for r in imgs) >= 0.9 * len(imgs)),
                          {"image_requests": len(imgs), "answered": sum(bool(r.get("image_answer")) for r in imgs)}),
        "cached_on_follow": (not follow or len(cached_follow) > 0, {"follow_ups": len(follow), "with_cached": len(cached_follow)}),
        "server_log": (logscan.get("fatal", 0) == 0, logscan),
        "min_requests": (len(recs) >= a.min_requests, {"requests": len(recs), "min": a.min_requests}),
        "context_bound": (max_ctx <= a.context, {"max_prompt_plus_max_tokens": max_ctx}),
    }
    warn = {}
    lim = {"host_used": 4 * gib, "swap_used": 512 << 20, "server_rss": 2 * gib, "server_swap": 512 << 20, "vision_rss": 2 * gib,
           "m_process_resident_bytes": 2 * gib, "m_device_memory_bytes": gib}
    for k, t in trends.items():
        if not t:
            continue
        cap = lim.get(k) or (gib if k.startswith("vram_") and k.endswith("resident") else None)
        if cap and t["growth_fit"] > cap:
            warn[f"growth_{k}"] = f"+{t['growth_fit'] / gib:.2f} GiB fitted over the run (cap {cap / gib:.2f})"
    if trends.get("host_available") and trends["host_available"]["last"] < 8 * gib:
        warn["host_available_low"] = f"{trends['host_available']['last'] / gib:.1f} GiB available at the end"
    tf = (classes.get("agent_follow") or {}).get("ttft_s") or {}
    if tf.get("p99") and tf["p99"] > a.warn_follow_ttft_s:
        warn["follow_ttft_p99"] = f"agent follow-up TTFT p99 {tf['p99']:.1f}s > {a.warn_follow_ttft_s}s"
    dall = classes.get("all", {}).get("decode_tps") or {}
    if dall.get("p50") and dall["p50"] < a.warn_decode_tps:
        warn["decode_p50"] = f"decode p50 {dall['p50']:.1f} tok/s < {a.warn_decode_tps}"
    gap = classes.get("all", {}).get("max_gap_s") or {}
    if gap.get("max") and gap["max"] > 30:
        warn["stream_gap"] = f"max inter-token gap {gap['max']:.1f}s"
    if needles and sum(needles) < 0.8 * len(needles):
        warn["needle"] = f"needle {sum(needles)}/{len(needles)}"
    if img_ok and sum(img_ok) < 0.7 * len(img_ok):
        warn["image_correct"] = f"image answers correct {sum(img_ok)}/{len(img_ok)}"
    if ok and mix["finish"].get("length", 0) > 0.25 * len(ok):
        warn["length_share"] = f"{mix['finish'].get('length', 0)}/{len(ok)} finished on length"
    if follow and len(cached_follow) < 0.5 * len(follow):
        warn["cached_share"] = f"cached_tokens > 0 on {len(cached_follow)}/{len(follow)} follow-ups"
    bad_http = {k: v for k, v in delta.items() if k.startswith("requests_total") and 'status="200"' not in k}
    if bad_http:
        warn["metrics_non200"] = bad_http
    for k in ("kept_tool_calls_malformed_total", "kept_tool_calls_unoffered_total", "client_disconnections_total", "preemptions_total"):
        if delta.get(k):
            warn[k] = delta[k]
    if unfinished:
        warn["unfinished_users"] = f"{unfinished} user threads still busy after --drain-s"
    if logscan.get("done_lines") is not None and logscan["done_lines"] < len(ok):
        warn["done_lines"] = f"server done lines {logscan['done_lines']} < client successes {len(ok)}"
    verdict = "PASS" if all(v[0] for v in hard.values()) else "FAIL"
    summary = {"verdict": verdict, "run": a.run_name, "start": time.strftime("%FT%TZ", time.gmtime(run.t0)),
               "end": time.strftime("%FT%TZ", time.gmtime()), "minutes": a.minutes, "base_url": a.base_url, "seed": a.seed,
               "requests": len(recs), "ok": len(ok), "conversations": run.conv_n, "max_inflight": run.max_inflight,
               "max_inflight_est_tokens": run.max_inflight_tokens, "concurrency_seconds": dict(sorted(run.conc_hist.items())),
               "chars_per_token": round(run.ratio, 3), "mix": mix,
               "cached_follow": {"follow_ups": len(follow), "with_cached": len(cached_follow),
                                 "mean_cached_frac": round(statistics.mean((r.get("cached_tokens") or 0) / max(1, r.get("prompt_tokens") or 1) for r in follow), 3) if follow else None},
               "needle": [sum(needles), len(needles)], "image_correct": [sum(img_ok), len(img_ok)],
               "hard": {k: {"pass": v[0], "detail": v[1]} for k, v in hard.items()}, "warn": warn,
               "classes": classes, "trends": trends, "metrics_delta": delta, "frontdoor": fdscan}
    (run.out / "summary.json").write_text(json.dumps(summary, indent=1))
    f1 = lambda x, n=1: "-" if x is None else f"{x:.{n}f}"  # noqa: E731
    rows = []
    for k, c in classes.items():
        t, d = c["ttft_s"] or {}, c["decode_tps"] or {}
        rows.append(f"  {k:14s} n={c['requests']:4d}  ttft p50/p99 {f1(t.get('p50'), 2):>7s}/{f1(t.get('p99'), 2):>7s} s  "
                    f"decode p50/p01 {f1(d.get('p50')):>6s}/{f1(d.get('p01')):>6s} tok/s  prompt max {(c['prompt_tokens'] or {}).get('max', '-')}")
    tr = []
    for k, t in trends.items():
        if t:
            tr.append(f"  {k:28s} first {t['first'] / gib:8.2f}  last {t['last'] / gib:8.2f}  fit growth {t['growth_fit'] / gib:+7.2f} GiB")
    txt = (f"release_traffic {a.run_name}: {verdict}\n"
           f"window {summary['start']} -> {summary['end']} ({a.minutes} min), {len(recs)} requests ({len(ok)} ok) in {run.conv_n} conversations, "
           f"max in-flight {run.max_inflight}, max in-flight est tokens {run.max_inflight_tokens}\n"
           f"mix: {json.dumps(mix)}\n"
           f"cached on follow-ups: {len(cached_follow)}/{len(follow)} (mean cached/prompt {summary['cached_follow']['mean_cached_frac']}); "
           f"needle {sum(needles)}/{len(needles)}; image correct {sum(img_ok)}/{len(img_ok)}\n"
           "HARD:\n" + "\n".join(f"  {'PASS' if v[0] else 'FAIL'} {k} {json.dumps(v[1])[:300]}" for k, v in hard.items()) + "\n"
           "WARN:\n" + ("\n".join(f"  {k}: {v}" for k, v in warn.items()) or "  none") + "\n"
           "per class:\n" + "\n".join(rows) + "\n"
           "memory (GiB, after warm-up):\n" + ("\n".join(tr) or "  no samples") + "\n"
           f"metrics delta: {json.dumps(delta)}\n")
    (run.out / "summary.txt").write_text(txt)
    print(txt, flush=True)
    return verdict


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--minutes", type=float, default=60)
    p.add_argument("--run-name", default=time.strftime("rel-%Y%m%dT%H%MZ", time.gmtime()))
    p.add_argument("--out", help="default RESULTS/<run-name>")
    p.add_argument("--base-url", default="http://127.0.0.1:18080/v1")
    p.add_argument("--api-key-file", default="/mnt/vm_8tb/b70/secrets/dd_api_key")
    p.add_argument("--metrics-url", default="http://127.0.0.1:18124/metrics")
    p.add_argument("--server-log", default="/mnt/vm_8tb/b70/logs/hotschmoe-dd/tf-server.log")
    p.add_argument("--frontdoor-log", default="/mnt/vm_8tb/b70/logs/hotschmoe-dd/frontdoor.log")
    p.add_argument("--server-match", default="tensorfold-native", help="cmdline substring of the server process (RSS, VRAM)")
    p.add_argument("--no-vram-fdinfo", dest="vram_fdinfo", action="store_false",
                   help="skip VRAM from /proc/<server>/fdinfo (xe DRM client stats; a CPU-side read, no GPU submission)")
    p.add_argument("--corpus-dir", action="append",
                   help="repeatable; default: repo docs tensorfold vllm sglang strata bin kernels evals + frozen TF zig/src")
    p.add_argument("--max-concurrency", type=int, default=8)
    p.add_argument("--context", type=int, default=200000)
    p.add_argument("--max-prompt-tokens", type=int, default=92000)
    p.add_argument("--sample-s", type=float, default=30)
    p.add_argument("--stall-s", type=float, default=900, help="socket inactivity timeout per request")
    p.add_argument("--drain-s", type=float, default=600, help="wait for in-flight requests after --minutes")
    p.add_argument("--think-scale", type=float, default=None, help="multiplies think/idle times (default 1, dry-run 0.1)")
    p.add_argument("--min-requests", type=int, default=20)
    p.add_argument("--warn-follow-ttft-s", type=float, default=30)
    p.add_argument("--warn-decode-tps", type=float, default=15)
    p.add_argument("--seed", type=int, default=20261122)
    p.add_argument("--dry-run", action="store_true", help="serve release_traffic_mock.py in-process and run against it")
    p.add_argument("--mock-faults", type=float, default=0.0, help="dry-run: fault probability per request (gate self-test)")
    a = p.parse_args()
    a.out = a.out or str(RESULTS / a.run_name)
    if not a.corpus_dir:
        a.corpus_dir = [str(REPO / d) for d in ("docs", "tensorfold", "vllm", "sglang", "strata", "bin", "kernels", "evals")]
        a.corpus_dir.append("/mnt/vm_8tb/b70/build/tensorfold-dd/c6346c82/zig/src")
    if a.dry_run:
        import release_traffic_mock as mock
        Path(a.out).mkdir(parents=True, exist_ok=True)
        keyf = Path(a.out) / "mock_api_key"
        keyf.write_text("mock-" + hashlib.sha256(os.urandom(16)).hexdigest()[:16])
        port = mock.serve_background(keyf.read_text().strip(), faults=a.mock_faults, seed=a.seed)
        a.base_url, a.api_key_file = f"http://127.0.0.1:{port}/v1", str(keyf)
        a.metrics_url = f"http://127.0.0.1:{port}/metrics"
        a.server_log = a.frontdoor_log = None
        a.server_match, a.vram_fdinfo = "<dry-run: no server process>", False  # never sample a real server in a dry run
        a.think_scale = 0.1 if a.think_scale is None else a.think_scale
    a.think_scale = 1.0 if a.think_scale is None else a.think_scale
    sys.exit(0 if Run(a).go() == "PASS" else 1)


if __name__ == "__main__":
    main()
