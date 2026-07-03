#!/usr/bin/env python3
# dflash_accept_probe.py -- accept-length telemetry for spec-decode serves (MTP or DFlash).
#
# Drives a running OpenAI-compatible vLLM server with a REALISTIC, cumulative multi-turn
# coding conversation (deterministic, temperature 0), and scrapes the Prometheus /metrics
# spec-decode counters BEFORE and AFTER each turn to compute, per turn and cumulative:
#   - decode tok/s (completion_tokens / wall_time)
#   - acceptance_length  = accepted/drafts + 1  (mean tokens emitted per verify step)
#   - accepted_per_draft = accepted/drafts       (mean spec tokens accepted per step)
#   - acceptance_rate    = accepted/draft_tokens (fraction of PROPOSED tokens accepted)
#
# Works identically against an MTP (NEXTN) serve and a DFlash serve: both use the same
# vllm:spec_decode_* counter family (DFlash is NOT on the diffusion metric path -- verified
# in vllm/v1/spec_decode/metrics.py: is_diffusion is a *target* dLLM flag, not set by the
# dflash draft method). A/B by pointing --model at each serve's model id in turn.
#
# Metric names (vLLM v0.24.0, prometheus_client appends _total to Counters):
#   vllm:spec_decode_num_drafts_total
#   vllm:spec_decode_num_draft_tokens_total
#   vllm:spec_decode_num_accepted_tokens_total
#   vllm:spec_decode_num_accepted_tokens_per_pos_total{position="i"}   (per-position accepts)
#
# Stdlib only (urllib/json) so it runs on the host python with no deps.
#
# Usage:
#   python3 vllm/dflash_accept_probe.py \
#       --base-url http://192.168.10.5:18080/v1 \
#       --model /models/qwen3.6-27b/w8a8 \
#       [--api-key KEY] [--max-tokens 400] [--turns 8]
#
# ASCII only. No GPU touch -- pure HTTP client.

import argparse
import json
import sys
import time
import urllib.error
import urllib.request

# --- The cumulative coding workload -------------------------------------------------------
# Sequential turns of a realistic agentic coding session. Each user turn builds on the
# assistant's prior code (the whole conversation is resent every turn, as a real client does),
# so drafts are conditioned on genuine code context -- the regime where DFlash/MTP accept high.
SYSTEM_PROMPT = (
    "You are a precise senior Python engineer. When asked for code, reply with a single "
    "fenced python code block and a one-sentence explanation. Keep prior code consistent "
    "across turns."
)

USER_TURNS = [
    "Write a Python function parse_iso8601(s) that parses an ISO-8601 timestamp string "
    "into a datetime, supporting an optional trailing 'Z' for UTC. No third-party libraries.",
    "Now add error handling: raise a ValueError with a clear message on malformed input, "
    "and add a type check that the argument is a str.",
    "Refactor the whole thing into a class TimestampParser with a parse method and a "
    "configurable default timezone passed to __init__.",
    "Add a classmethod from_env that reads the default timezone offset from an environment "
    "variable TZ_OFFSET (like '+05:30') and constructs the parser.",
    "Write pytest unit tests covering: a UTC 'Z' timestamp, an offset timestamp, malformed "
    "input, a non-string argument, and the from_env constructor.",
    "There's a bug: fractional seconds like '2026-07-03T12:00:00.123Z' fail to parse. Fix the "
    "parse method to handle optional fractional seconds, and add a test for it.",
    "Add a method to_unix(self, s) that parses and returns an integer Unix timestamp in "
    "seconds, and make parse reuse a single private _parse helper to avoid duplication.",
    "Now write a short module-level docstring and a __main__ block that reads one timestamp "
    "from sys.argv and prints its Unix time, then show the final complete file.",
    "Add __slots__ to the class and a __repr__, and explain in one sentence why __slots__ "
    "helps here.",
    "Package it: write a minimal pyproject.toml (setuptools) exposing a console script "
    "'isoparse' that calls the __main__ logic.",
]

SPEC_METRICS = {
    "drafts": "vllm:spec_decode_num_drafts_total",
    "draft_tokens": "vllm:spec_decode_num_draft_tokens_total",
    "accepted": "vllm:spec_decode_num_accepted_tokens_total",
}
PER_POS_METRIC = "vllm:spec_decode_num_accepted_tokens_per_pos_total"


def _metrics_url(base_url):
    # base_url is like http://host:port/v1 ; metrics live at http://host:port/metrics
    b = base_url.rstrip("/")
    if b.endswith("/v1"):
        b = b[: -len("/v1")]
    return b + "/metrics"


def _http_get(url, api_key, timeout=30):
    req = urllib.request.Request(url)
    if api_key:
        req.add_header("Authorization", "Bearer " + api_key)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", "replace")


def _http_post_json(url, payload, api_key, timeout=600):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    if api_key:
        req.add_header("Authorization", "Bearer " + api_key)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8", "replace"))


def scrape_metrics(base_url, api_key):
    """Return dict of summed spec-decode counters + per-position accept vector.

    Sums each metric family across all label sets (engines / model_name labels).
    Returns None if the server exposes no spec-decode counters (non-spec serve).
    """
    try:
        text = _http_get(_metrics_url(base_url), api_key)
    except urllib.error.URLError as e:
        print("WARN: /metrics fetch failed: %s" % e, file=sys.stderr)
        return None
    out = {k: 0.0 for k in SPEC_METRICS}
    per_pos = {}
    seen_any = False
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        # format: name{labels} value   OR   name value
        if "{" in line:
            name = line[: line.index("{")]
            rest = line[line.rindex("}") + 1 :].strip()
        else:
            parts = line.rsplit(" ", 1)
            if len(parts) != 2:
                continue
            name, rest = parts[0], parts[1]
        try:
            val = float(rest.split()[0])
        except (ValueError, IndexError):
            continue
        for key, mname in SPEC_METRICS.items():
            if name == mname:
                out[key] += val
                seen_any = True
        if name == PER_POS_METRIC:
            seen_any = True
            pos = None
            if "position=" in line:
                seg = line.split('position="', 1)[1]
                pos = seg.split('"', 1)[0]
            per_pos[pos] = per_pos.get(pos, 0.0) + val
    if not seen_any:
        return None
    out["per_pos"] = per_pos
    return out


def _diff(before, after):
    if before is None or after is None:
        return None
    d = {k: after[k] - before[k] for k in SPEC_METRICS}
    return d


def _accept_stats(d):
    """From a counter-delta dict compute acceptance metrics (guard divide-by-zero)."""
    drafts = d["drafts"]
    dtoks = d["draft_tokens"]
    acc = d["accepted"]
    apd = (acc / drafts) if drafts > 0 else 0.0
    return {
        "drafts": drafts,
        "draft_tokens": dtoks,
        "accepted": acc,
        "accepted_per_draft": apd,
        "acceptance_length": apd + 1.0,
        "acceptance_rate": (acc / dtoks) if dtoks > 0 else 0.0,
    }


def main():
    ap = argparse.ArgumentParser(description="Spec-decode accept telemetry (MTP/DFlash A/B).")
    ap.add_argument("--base-url", required=True, help="OpenAI base, e.g. http://host:18080/v1")
    ap.add_argument("--model", required=True, help="served model id (query /v1/models)")
    ap.add_argument("--api-key", default=None, help="optional Bearer API key")
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--turns", type=int, default=8, help="number of coding turns (max %d)" % len(USER_TURNS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--label", default="run", help="tag for the printed table")
    args = ap.parse_args()

    n_turns = max(1, min(args.turns, len(USER_TURNS)))
    chat_url = args.base_url.rstrip("/") + "/chat/completions"

    spec_available = scrape_metrics(args.base_url, args.api_key) is not None
    if not spec_available:
        print("NOTE: server exposes no vllm:spec_decode_* counters -- spec columns will be blank "
              "(non-spec serve, or spec metrics disabled).", file=sys.stderr)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    rows = []
    total_completion_tokens = 0
    total_wall = 0.0
    cum_before = scrape_metrics(args.base_url, args.api_key)

    for i in range(n_turns):
        messages.append({"role": "user", "content": USER_TURNS[i]})
        payload = {
            "model": args.model,
            "messages": messages,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "stream": False,
            "seed": args.seed,
        }
        m_before = scrape_metrics(args.base_url, args.api_key)
        t0 = time.time()
        try:
            resp = _http_post_json(chat_url, payload, args.api_key)
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "replace")[:500]
            print("ERROR turn %d HTTP %s: %s" % (i + 1, e.code, body), file=sys.stderr)
            return 2
        except urllib.error.URLError as e:
            print("ERROR turn %d: %s" % (i + 1, e), file=sys.stderr)
            return 2
        dt = time.time() - t0
        m_after = scrape_metrics(args.base_url, args.api_key)

        choice = resp["choices"][0]
        content = choice["message"]["content"] or ""
        messages.append({"role": "assistant", "content": content})
        usage = resp.get("usage", {}) or {}
        ctoks = usage.get("completion_tokens") or 0
        total_completion_tokens += ctoks
        total_wall += dt

        delta = _diff(m_before, m_after)
        stats = _accept_stats(delta) if delta else None
        rows.append({
            "turn": i + 1,
            "ctoks": ctoks,
            "sec": dt,
            "toks_per_s": (ctoks / dt) if dt > 0 else 0.0,
            "stats": stats,
        })

    cum_after = scrape_metrics(args.base_url, args.api_key)
    cum_delta = _diff(cum_before, cum_after)
    cum_stats = _accept_stats(cum_delta) if cum_delta else None

    # ---- Print compact table ----
    print()
    print("=== spec-decode accept telemetry :: %s ===" % args.label)
    print("model=%s  base=%s  turns=%d  max_tokens=%d  temp=%.1f"
          % (args.model, args.base_url, n_turns, args.max_tokens, args.temperature))
    hdr = "%-5s %8s %8s %9s %8s %10s %10s %10s" % (
        "turn", "ctoks", "sec", "tok/s", "drafts", "acc_len", "acc/draft", "acc_rate")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        s = r["stats"]
        if s:
            print("%-5d %8d %8.2f %9.2f %8d %10.3f %10.3f %10.3f" % (
                r["turn"], r["ctoks"], r["sec"], r["toks_per_s"],
                int(s["drafts"]), s["acceptance_length"], s["accepted_per_draft"],
                s["acceptance_rate"]))
        else:
            print("%-5d %8d %8.2f %9.2f %8s %10s %10s %10s" % (
                r["turn"], r["ctoks"], r["sec"], r["toks_per_s"], "-", "-", "-", "-"))
    print("-" * len(hdr))
    agg_tps = (total_completion_tokens / total_wall) if total_wall > 0 else 0.0
    if cum_stats:
        print("%-5s %8d %8.2f %9.2f %8d %10.3f %10.3f %10.3f" % (
            "ALL", total_completion_tokens, total_wall, agg_tps,
            int(cum_stats["drafts"]), cum_stats["acceptance_length"],
            cum_stats["accepted_per_draft"], cum_stats["acceptance_rate"]))
        pp = cum_after.get("per_pos") if cum_after else None
        if pp:
            def _k(x):
                try:
                    return int(x)
                except (TypeError, ValueError):
                    return 1 << 30
            order = sorted(pp.keys(), key=_k)
            base = cum_before.get("per_pos", {}) if cum_before else {}
            vec = ["p%s=%d" % (p, int(pp[p] - base.get(p, 0.0))) for p in order]
            print("per-position accepts (cumulative delta): " + "  ".join(vec))
    else:
        print("%-5s %8d %8.2f %9.2f  (no spec-decode counters exposed)" % (
            "ALL", total_completion_tokens, total_wall, agg_tps))
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
