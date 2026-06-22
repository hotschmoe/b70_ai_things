#!/usr/bin/env python3
"""
75_localmaxxing.py -- pull community Intel Arc Pro B70 LLM-inference benchmarks
from the localmaxxing.com API (https://www.localmaxxing.com/en/api-docs).

This is a READ-ONLY puller. GET endpoints on localmaxxing are public (no auth),
so by default no key is needed. The site is a crowd-sourced local-inference
benchmark/leaderboard -- it carries other people's B70 numbers (the same data
we curate by hand in docs/COMMUNITY_CONFIGS.md), including full reproducible
`vllm serve` / `llama-server` command snippets in each row's engineFlags.

ASCII only, stdlib only (urllib). No third-party deps.

Usage:
  python3 scripts/75_localmaxxing.py                 # summary: best output tok/s per model on B70
  python3 scripts/75_localmaxxing.py leaderboard     # ranked leaderboard rows (tokSOut desc)
  python3 scripts/75_localmaxxing.py configs          # full serve command snippets, ranked
  python3 scripts/75_localmaxxing.py raw              # dump all raw benchmark JSON
  python3 scripts/75_localmaxxing.py save [DIR]      # write raw JSON + markdown summary to DIR

Options (apply to summary/leaderboard/configs):
  --gpu NAME        GPU name to filter on (default: "Intel Arc Pro B70")
  --engine NAME     only this engine (vllm | llama.cpp | sglang | ...)
  --top N           limit rows shown (default 40)
  --min-toks F      drop rows below this output tok/s
  --json            machine-readable JSON instead of a table

Auth (optional, only needed for WRITE endpoints we don't use here):
  export LOCALMAXXING_API_KEY=bhk_<40 hex>   # sent as: Authorization: Bearer <key>
  Keys are created in the localmaxxing dashboard (POST /api/keys needs a browser session).
"""

import argparse
import json
import os
import sys
import urllib.parse
import urllib.request
import urllib.error

BASE = "https://www.localmaxxing.com"
DEFAULT_GPU = "Intel Arc Pro B70"
PAGE = 100  # API page cap
TIMEOUT = 45


# Cloudflare (error 1010) bans the default Python-urllib User-Agent, so send a real one.
UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/126.0.0.0 Safari/537.36")


def _get(path, params):
    url = BASE + path + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"Accept": "application/json", "User-Agent": UA})
    key = os.environ.get("LOCALMAXXING_API_KEY")
    if key:
        req.add_header("Authorization", "Bearer " + key)
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        sys.exit("HTTP %s on %s\n%s" % (e.code, url, e.read().decode("utf-8", "replace")[:500]))
    except urllib.error.URLError as e:
        sys.exit("network error on %s: %s" % (url, e.reason))


def fetch_benchmarks(gpu):
    """Page through GET /api/benchmarks for one GPU; return every record."""
    out, offset = [], 0
    while True:
        d = _get("/api/benchmarks", {"gpuName": gpu, "limit": PAGE, "offset": offset})
        rows = d.get("benchmarks", [])
        out.extend(rows)
        total = d.get("total", len(out))
        offset += PAGE
        if not rows or len(out) >= total or offset > 5000:
            return out, total


def fetch_leaderboard(gpu, engine=None, limit=200):
    p = {"hardwareName": gpu, "limit": limit}
    if engine:
        p["engineName"] = engine
    d = _get("/api/leaderboard", p)
    return d.get("rows", [])


def _f(x):
    return x if isinstance(x, (int, float)) else None


def flatten(b):
    """Normalize a benchmark or leaderboard row into a flat dict."""
    hw = b.get("hardware", {}) or {}
    e = b.get("engine", {}) or {}
    ef = b.get("engineFlags", {}) or {}
    m = b.get("model", {}) or {}
    u = b.get("user", {}) or {}
    return {
        "id": b.get("id"),
        "hfId": m.get("hfId"),
        "params": m.get("params"),
        "isMoE": m.get("isMoE"),
        "engine": e.get("engineName"),
        "engineVersion": e.get("engineVersion"),
        "quant": e.get("quantization"),
        "gpu": hw.get("gpuName"),
        "gpuCount": hw.get("gpuCount"),
        "vramGb": hw.get("vramGb"),
        "cpu": hw.get("cpu"),
        "os": hw.get("os"),
        "tp": ef.get("tensorParallel"),
        "cmd": (ef.get("commandSnippet") or "").strip(),
        "tokSOut": _f(b.get("tokSOut")),
        "tokSTotal": _f(b.get("tokSTotal")),
        "tokSPrefill": _f(b.get("tokSPrefill")),
        "ttftMs": _f(b.get("ttftMs")),
        "peakVramGb": _f(b.get("peakVramGb")),
        "batchSize": b.get("batchSize"),
        "contextLength": b.get("contextLength"),
        "user": u.get("username"),
        "verified": u.get("verified"),
        "createdAt": b.get("createdAt"),
        "notes": (b.get("notes") or "").strip(),
    }


def best_per_model(rows):
    """Keep the highest output-tok/s record per (hfId, engine, quant, gpuCount)."""
    best = {}
    for r in rows:
        if r["tokSOut"] is None:
            continue
        k = (r["hfId"], r["engine"], r["quant"], r["gpuCount"])
        if k not in best or r["tokSOut"] > best[k]["tokSOut"]:
            best[k] = r
    return sorted(best.values(), key=lambda r: r["tokSOut"], reverse=True)


def _trunc(s, n):
    s = s.replace("\n", " ")
    return s if len(s) <= n else s[: n - 3] + "..."


def print_summary(rows, top):
    rows = rows[:top]
    print("%-46s %-10s %-22s %4s %8s %8s %-14s" %
          ("model (hfId)", "engine", "quant", "gpus", "tok/s", "ttft_ms", "by"))
    print("-" * 120)
    for r in rows:
        print("%-46.46s %-10.10s %-22.22s %4s %8.1f %8s %-14.14s" % (
            r["hfId"] or "?", r["engine"] or "?", r["quant"] or "?",
            ("x%s" % r["gpuCount"]) if r["gpuCount"] else "?",
            r["tokSOut"],
            ("%.0f" % r["ttftMs"]) if r["ttftMs"] is not None else "-",
            r["user"] or "?",
        ))


def print_configs(rows, top):
    """Show ranked rows WITH their full serve command snippet -- the reproducible bit."""
    shown = 0
    for r in rows:
        if shown >= top:
            break
        shown += 1
        print("=" * 100)
        print("[%2d] %s  |  %s %s  |  %s  |  x%s  |  %.1f tok/s out  (by %s, %s)" % (
            shown, r["hfId"], r["engine"], r["engineVersion"] or "", r["quant"],
            r["gpuCount"], r["tokSOut"] or 0, r["user"], (r["createdAt"] or "")[:10]))
        if r["ttftMs"] is not None or r["peakVramGb"] is not None:
            print("     ttft=%s ms  peakVram=%s GB  batch=%s  ctx=%s" % (
                ("%.0f" % r["ttftMs"]) if r["ttftMs"] is not None else "-",
                r["peakVramGb"], r["batchSize"], r["contextLength"]))
        if r["cmd"]:
            print("     cmd: %s" % r["cmd"])
        if r["notes"]:
            print("     notes: %s" % _trunc(r["notes"], 300))


def to_markdown(rows, gpu):
    out = []
    out.append("# localmaxxing.com -- %s community benchmarks" % gpu)
    out.append("")
    out.append("Auto-pulled from `GET /api/benchmarks?gpuName=%s` (best output tok/s per"
               " model/engine/quant/gpuCount). Regenerate with"
               " `python3 scripts/75_localmaxxing.py save`." % gpu)
    out.append("")
    out.append("| Model | Engine | Quant | GPUs | tok/s out | TTFT ms | By |")
    out.append("|---|---|---|---|---|---|---|")
    for r in rows:
        out.append("| %s | %s | %s | x%s | %.1f | %s | %s |" % (
            r["hfId"], r["engine"], r["quant"], r["gpuCount"], r["tokSOut"],
            ("%.0f" % r["ttftMs"]) if r["ttftMs"] is not None else "-", r["user"]))
    out.append("")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(add_help=True, description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", nargs="?", default="summary",
                    choices=["summary", "leaderboard", "configs", "raw", "save"])
    ap.add_argument("savedir", nargs="?", default="data/localmaxxing")
    ap.add_argument("--gpu", default=DEFAULT_GPU)
    ap.add_argument("--engine", default=None)
    ap.add_argument("--top", type=int, default=40)
    ap.add_argument("--min-toks", type=float, default=None, dest="min_toks")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    raw, total = fetch_benchmarks(args.gpu)
    rows = [flatten(b) for b in raw]
    if args.engine:
        rows = [r for r in rows if (r["engine"] or "") == args.engine]
    if args.min_toks is not None:
        rows = [r for r in rows if (r["tokSOut"] or 0) >= args.min_toks]

    sys.stderr.write("pulled %d benchmark records for '%s' (server total=%d)\n"
                     % (len(raw), args.gpu, total))

    if args.cmd == "raw":
        print(json.dumps(raw, indent=2))
        return

    if args.cmd == "leaderboard":
        lb = [flatten(b) for b in fetch_leaderboard(args.gpu, args.engine)]
        lb = [r for r in lb if r["tokSOut"] is not None]
        lb.sort(key=lambda r: r["tokSOut"], reverse=True)
        if args.json:
            print(json.dumps(lb[:args.top], indent=2))
        else:
            print_summary(lb, args.top)
        return

    best = best_per_model(rows)

    if args.cmd == "configs":
        if args.json:
            print(json.dumps(best[:args.top], indent=2))
        else:
            print_configs(best, args.top)
        return

    if args.cmd == "save":
        d = args.savedir
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "b70_benchmarks_raw.json"), "w") as f:
            json.dump(raw, f, indent=2)
        with open(os.path.join(d, "b70_leaderboard_raw.json"), "w") as f:
            json.dump(fetch_leaderboard(args.gpu, args.engine), f, indent=2)
        with open(os.path.join(d, "b70_summary.md"), "w") as f:
            f.write(to_markdown(best, args.gpu))
        sys.stderr.write("wrote raw JSON + b70_summary.md to %s/\n" % d)
        return

    # default: summary
    if args.json:
        print(json.dumps(best[:args.top], indent=2))
    else:
        print_summary(best, args.top)


if __name__ == "__main__":
    main()
