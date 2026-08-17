#!/usr/bin/env python3
"""Long-decode collapse probe for a live OpenAI-compatible serve.

Streams one or more long generations, reports per-window tok/s + garbage
signals (!!!!, bang-runs, n-gram loops, unique-char collapse) and optional
vLLM spec-decode metric deltas.

Usage:
  python3 vllm/dflash/probe_decode_collapse.py \
      --base-url http://127.0.0.1:8078/v1 \
      --model qwen3.8-27b-fp8-dspark \
      --out 2048 --windows 8 --reps 2
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request

BANG_RE = re.compile(r"!{4,}")
WS_RE = re.compile(r"\s+")


def scrape_spec(metrics_url: str) -> dict[str, float]:
    out = {
        "drafts": 0.0,
        "draft_tok": 0.0,
        "accepted": 0.0,
        "pos": {},
    }
    try:
        with urllib.request.urlopen(metrics_url, timeout=10) as r:
            text = r.read().decode("utf-8", "replace")
    except Exception:
        return out
    for line in text.splitlines():
        if line.startswith("#") or not line.startswith("vllm:spec_decode"):
            continue
        if "num_drafts_total{" in line:
            out["drafts"] = float(line.rsplit(" ", 1)[-1])
        elif "num_draft_tokens_total{" in line:
            out["draft_tok"] = float(line.rsplit(" ", 1)[-1])
        elif "num_accepted_tokens_total{" in line and "per_pos" not in line:
            out["accepted"] = float(line.rsplit(" ", 1)[-1])
        elif "num_accepted_tokens_per_pos_total{" in line:
            m = re.search(r'position="(\d+)"', line)
            if m:
                out["pos"][int(m.group(1))] = float(line.rsplit(" ", 1)[-1])
    return out


def spec_delta(a: dict, b: dict) -> dict:
    d_drafts = b["drafts"] - a["drafts"]
    d_tok = b["draft_tok"] - a["draft_tok"]
    d_acc = b["accepted"] - a["accepted"]
    pos = {}
    keys = set(a.get("pos", {})) | set(b.get("pos", {}))
    for k in sorted(keys):
        pos[k] = b.get("pos", {}).get(k, 0.0) - a.get("pos", {}).get(k, 0.0)
    acc_len = (d_acc / d_drafts + 1.0) if d_drafts else float("nan")
    acc_rate = (d_acc / d_tok) if d_tok else float("nan")
    pos0 = (pos.get(0, 0.0) / d_drafts) if d_drafts else float("nan")
    return {
        "drafts": d_drafts,
        "draft_tok": d_tok,
        "accepted": d_acc,
        "accept_len": acc_len,
        "accept_rate": acc_rate,
        "pos0": pos0,
        "pos": pos,
    }


def window_stats(text: str) -> dict:
    n = len(text)
    bangs = len(BANG_RE.findall(text))
    bang_chars = sum(len(m.group(0)) for m in BANG_RE.finditer(text))
    uniq = len(set(text)) if n else 0
    words = WS_RE.split(text.strip()) if text.strip() else []
    loop = 0
    if len(words) >= 8:
        tail = " ".join(words[-8:])
        loop = text.count(tail)
    # 4-gram word repeat ratio
    rep = 0.0
    if len(words) >= 12:
        grams = [" ".join(words[i : i + 4]) for i in range(len(words) - 3)]
        if grams:
            rep = 1.0 - (len(set(grams)) / len(grams))
    return {
        "chars": n,
        "uniq": uniq,
        "bangs": bangs,
        "bang_chars": bang_chars,
        "loop": loop,
        "rep4": rep,
        "preview": text[:80].replace("\n", "\\n"),
        "tail": text[-80:].replace("\n", "\\n"),
    }


def stream_one(url: str, model: str, prompt: str, out: int, thinking: bool):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": out,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": thinking},
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"content-type": "application/json"},
    )
    t0 = time.time()
    first = None
    chunks = []
    usage = {}
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                obj = json.loads(data)
            except Exception:
                continue
            if obj.get("usage"):
                usage = obj["usage"]
            ch = obj.get("choices") or []
            if not ch:
                continue
            d = ch[0].get("delta") or {}
            piece = (
                d.get("content")
                or d.get("reasoning_content")
                or d.get("reasoning")
                or ""
            )
            if piece:
                if first is None:
                    first = time.time()
                chunks.append((time.time(), piece))
    end = time.time()
    return {
        "t0": t0,
        "first": first,
        "end": end,
        "chunks": chunks,
        "usage": usage,
        "text": "".join(p for _, p in chunks),
    }


def split_windows(chunks, nwin: int):
    if not chunks:
        return []
    text = "".join(p for _, p in chunks)
    if not text:
        return []
    # Prefer usage-aligned char windows of roughly equal size.
    n = len(text)
    size = max(1, n // nwin)
    out = []
    pos = 0
    t_start = chunks[0][0]
    # map char offset -> time by walking chunks
    times = []
    off = 0
    for ts, p in chunks:
        times.append((off, ts))
        off += len(p)

    def t_at(cidx):
        last = t_start
        for o, ts in times:
            if o > cidx:
                break
            last = ts
        return last

    for i in range(nwin):
        a = i * size
        b = n if i == nwin - 1 else min(n, (i + 1) * size)
        if a >= n:
            break
        piece = text[a:b]
        ta, tb = t_at(a), t_at(max(a, b - 1))
        dt = max(1e-6, tb - ta)
        # rough token estimate: 4 chars/tok fallback; caller may override
        est_tok = max(1, len(piece) / 4.0)
        st = window_stats(piece)
        st["i"] = i
        st["dt"] = dt
        st["est_tps"] = est_tok / dt
        out.append(st)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8078/v1")
    ap.add_argument("--model", default="qwen3.8-27b-fp8-dspark")
    ap.add_argument("--out", type=int, default=2048)
    ap.add_argument("--windows", type=int, default=8)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--thinking", action="store_true")
    args = ap.parse_args()
    base = args.base_url.rstrip("/")
    url = f"{base}/chat/completions"
    metrics = base[: -len("/v1")] + "/metrics" if base.endswith("/v1") else base + "/metrics"

    prompt = (
        "Write a long, detailed technical essay on how a hybrid Gated DeltaNet "
        "plus sparse attention transformer schedules KV memory during decode. "
        "Cover kernels, block sizes, prefix cache, and speculative verification. "
        "Keep writing until you hit the length limit. No short summary."
    )

    print(
        f"model={args.model} out={args.out} wins={args.windows} "
        f"reps={args.reps} thinking={args.thinking}"
    )
    collapsed = False
    for rep in range(args.reps):
        before = scrape_spec(metrics)
        t_req = time.time()
        try:
            res = stream_one(url, args.model, prompt, args.out, args.thinking)
        except urllib.error.HTTPError as e:
            print(f"rep{rep} HTTP {e.code}: {e.read()[:200]!r}")
            return 2
        except Exception as e:
            print(f"rep{rep} FAIL {type(e).__name__}: {e}")
            return 2
        after = scrape_spec(metrics)
        wall = res["end"] - res["t0"]
        ct = (res["usage"] or {}).get("completion_tokens")
        tps = (ct / wall) if ct and wall else float("nan")
        sd = spec_delta(before, after)
        full = window_stats(res["text"])
        print(
            f"rep{rep} wall={wall:.1f}s ct={ct} tps={tps:.2f} "
            f"chars={full['chars']} uniq={full['uniq']} bangs={full['bangs']} "
            f"rep4={full['rep4']:.2f} loop={full['loop']}"
        )
        if sd["drafts"]:
            print(
                f"  spec drafts={sd['drafts']:.0f} acc_len={sd['accept_len']:.3f} "
                f"rate={sd['accept_rate']:.3f} pos0={sd['pos0']:.3f}"
            )
        print(f"  head: {full['preview']}")
        print(f"  tail: {full['tail']}")
        wins = split_windows(res["chunks"], args.windows)
        for w in wins:
            flag = ""
            if w["bangs"] or w["uniq"] <= 8 or w["rep4"] >= 0.55 or w["loop"] >= 3:
                flag = "  COLLAPSE"
                collapsed = True
            print(
                f"  w{w['i']:02d} chars={w['chars']:4d} uniq={w['uniq']:3d} "
                f"bangs={w['bangs']} rep4={w['rep4']:.2f} "
                f"est_tps={w['est_tps']:.1f}{flag}"
            )
            print(f"       tail={w['tail']}")
        # cheap inter-request pause so metrics settle
        time.sleep(0.2)
        _ = t_req
    print("VERDICT", "COLLAPSE" if collapsed else "NO_COLLAPSE")
    return 1 if collapsed else 0


if __name__ == "__main__":
    sys.exit(main())
