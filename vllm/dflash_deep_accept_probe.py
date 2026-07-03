#!/usr/bin/env python3
"""Measure DFlash acceptance length at DEEP context depths (does accept hold when the
coding context is tens of thousands of tokens, not just a few hundred?).

Builds a realistic long coding context by concatenating real source files, truncates it
to a series of target token depths, and for each depth sends one deterministic coding
continuation request while scraping the server's spec-decode /metrics counters before and
after. Prints accept_len (accepted/drafts + 1) per depth.

Dependency-free (urllib). Usage:
  dflash_deep_accept_probe.py <base_url> <model> [depths_ktok=4,16,40,100] [max_tokens=300]

Token depth is approximated as chars/3.6 (code is denser than prose); the actual prompt
token count is read back from the server usage field and printed, so the depth label is
exact after the fact.
"""
import sys, os, json, time, glob, urllib.request

BASE = sys.argv[1].rstrip("/"); MODEL = sys.argv[2]
DEPTHS = [int(x) * 1000 for x in (sys.argv[3].split(",") if len(sys.argv) > 3 else ["4", "16", "40", "100"])]
MAXTOK = int(sys.argv[4]) if len(sys.argv) > 4 else 300
API_KEY = os.environ.get("API_KEY", "") or os.environ.get("B70_API_KEY", "")
CHARS_PER_TOK = 3.6

METRICS = {
    "drafts": "vllm:spec_decode_num_drafts_total",
    "draft_tokens": "vllm:spec_decode_num_draft_tokens_total",
    "accepted": "vllm:spec_decode_num_accepted_tokens_total",
}


def scrape():
    req = urllib.request.Request(f"{BASE.rsplit('/v1',1)[0]}/metrics")
    if API_KEY: req.add_header("Authorization", f"Bearer {API_KEY}")
    out = {}
    with urllib.request.urlopen(req, timeout=30) as r:
        for line in r.read().decode("utf-8", "ignore").splitlines():
            if line.startswith("#"):
                continue
            for k, name in METRICS.items():
                if line.startswith(name + " ") or line.startswith(name + "{"):
                    try: out[k] = out.get(k, 0.0) + float(line.rsplit(" ", 1)[1])
                    except Exception: pass
    return out


def build_corpus():
    files = []
    for pat in ("vllm/*.py", "sglang/patches/*.py", "bin/*.sh", "kernels/*.hpp", "*.md"):
        files += sorted(glob.glob(os.path.join(os.path.dirname(__file__), "..", pat)))
    buf = []
    for f in files:
        try:
            with open(f, encoding="utf-8", errors="ignore") as fh:
                buf.append(f"\n\n# ===== FILE: {os.path.basename(f)} =====\n" + fh.read())
        except Exception:
            pass
    corpus = "".join(buf)
    while len(corpus) < 500_000:   # ensure enough material for the deepest probe
        corpus += corpus
    return corpus


def one(depth_tok, corpus):
    nchars = int(depth_tok * CHARS_PER_TOK)
    context = corpus[:nchars]
    user = ("Below is a large excerpt of a real codebase. Study it, then write a single new, "
            "well-typed Python helper function `summarize_kv_cache(logs: list[str]) -> dict` that "
            "parses vLLM KV-cache init log lines and returns a dict with keys "
            "available_gib, kv_tokens, max_concurrency. Return only the function.\n\n"
            "```\n" + context + "\n```\n")
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": user}],
        "max_tokens": MAXTOK, "temperature": 0, "stream": False,
    }).encode()
    hdr = {"content-type": "application/json"}
    if API_KEY: hdr["Authorization"] = f"Bearer {API_KEY}"
    before = scrape()
    t0 = time.perf_counter()
    req = urllib.request.Request(f"{BASE}/chat/completions", data=body, headers=hdr)
    with urllib.request.urlopen(req, timeout=600) as r:
        resp = json.loads(r.read())
    dt = time.perf_counter() - t0
    after = scrape()
    usage = resp.get("usage", {})
    ptok = usage.get("prompt_tokens", 0); ctok = usage.get("completion_tokens", 0)
    d_drafts = after.get("drafts", 0) - before.get("drafts", 0)
    d_acc = after.get("accepted", 0) - before.get("accepted", 0)
    d_dtok = after.get("draft_tokens", 0) - before.get("draft_tokens", 0)
    acc_len = (d_acc / d_drafts + 1) if d_drafts else float("nan")
    acc_rate = (d_acc / d_dtok) if d_dtok else float("nan")
    tps = ctok / dt if dt else 0
    msg = resp["choices"][0]["message"]
    txt = (msg.get("content") or msg.get("reasoning") or "")
    coherent = "def summarize_kv_cache" in txt or "def " in txt
    return dict(ptok=ptok, ctok=ctok, drafts=int(d_drafts), acc_len=acc_len,
                acc_rate=acc_rate, tps=tps, coherent=coherent)


def main():
    corpus = build_corpus()
    print(f"corpus chars: {len(corpus):,}  model: {MODEL}")
    print(f"{'depth_req':>9} {'prompt_tok':>10} {'gen':>4} {'drafts':>6} {'accept_len':>10} {'acc_rate':>8} {'tok/s':>6} {'coherent':>8}")
    print("-" * 74)
    for d in DEPTHS:
        try:
            r = one(d, corpus)
            print(f"{d:>9} {r['ptok']:>10} {r['ctok']:>4} {r['drafts']:>6} "
                  f"{r['acc_len']:>10.3f} {r['acc_rate']:>8.3f} {r['tps']:>6.1f} {str(r['coherent']):>8}")
        except Exception as e:
            print(f"{d:>9}  ERROR: {e}")


main()
