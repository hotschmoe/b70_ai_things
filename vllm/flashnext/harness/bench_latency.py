#!/usr/bin/env python3
"""Latency harness for the Flash-Next vLLM XPU server.

Latency is the number this measures: per-token decode ITL at C1 and C2,
and TTFT / prefill tok/s at 1K, 8K, and 32K. Throughput is reported only
as the reciprocal of those latencies.

The server must already be up. This script talks HTTP from the host and
does not take a GPU lease.

  python3 vllm/flashnext/harness/bench_latency.py \
      --base-url http://127.0.0.1:18081 \
      --out /mnt/vm_8tb/b70/build/vllm-flashnext/logs/a3-bench-eager.json

Strata night-final-v3 reference (strata/flash-next/dd/README.md, fixed
prompts at deploy): C1 ~141 tok/s, C2 ~92 tok/s per stream, prefill
4K ~2200 and 32K ~2150 tok/s. Those are a different model (Flash-Next
IQ3_XXS, two cards, MTP on). They are a comparison, not a matched pair.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Any

# Deploy numbers from strata/flash-next/dd/README.md (2026-10-16).
STRATA_DD = {
    "source": "strata/flash-next/dd/README.md night-final-v3",
    "model": "qwen3.8-flash-next ISTA GSQ-RCO IQ3_XXS",
    "cards": 2,
    "c1_tok_s": 141.0,
    "c1_itl_ms": 1000.0 / 141.0,
    "c2_tok_s_per_stream": 92.0,
    "c2_itl_ms": 1000.0 / 92.0,
    "prefill_4k_tok_s": 2200.0,
    "prefill_32k_tok_s": 2150.0,
}

DECODE_PROMPT = (
    "The capital of France is Paris. List three short facts about Paris, "
    "one fact per line, with no extra commentary."
)
COHERENCE_PROMPT = "The capital of France is"


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * p
    lo = int(rank)
    hi = min(lo + 1, len(ordered) - 1)
    frac = rank - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:800]
        raise RuntimeError(f"HTTP {exc.code} {url}: {detail}") from exc
    return json.loads(body)


def get_text(url: str, timeout: float = 30.0) -> str:
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8")


def get_json(url: str, timeout: float = 30.0) -> Any:
    return json.loads(get_text(url, timeout=timeout))


def garbage_verdict(text: str) -> str:
    compact = text.replace(" ", "")
    if len(compact) < 12:
        return "SHORT"
    counts: dict[str, int] = {}
    for ch in compact:
        counts[ch] = counts.get(ch, 0) + 1
    if max(counts.values()) / len(compact) > 0.55:
        return "GARBAGE"
    return "OK"


def stream_completion(
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    timeout: float,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        base_url.rstrip("/") + "/v1/completions",
        data=data,
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        method="POST",
    )
    t0 = time.perf_counter()
    arrivals: list[float] = []
    pieces: list[str] = []
    usage: dict[str, Any] = {}
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:800]
        raise RuntimeError(f"HTTP {exc.code} completions: {detail}") from exc
    with resp:
        for raw in resp:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            blob = line[5:].strip()
            if blob == "[DONE]":
                break
            try:
                obj = json.loads(blob)
            except json.JSONDecodeError:
                continue
            if obj.get("usage"):
                usage = obj["usage"]
            choices = obj.get("choices") or []
            if not choices:
                continue
            piece = choices[0].get("text") or ""
            if piece == "":
                continue
            arrivals.append(time.perf_counter())
            pieces.append(piece)
    text = "".join(pieces)
    ttft_s = (arrivals[0] - t0) if arrivals else None
    itls_ms: list[float] = []
    for prev, cur in zip(arrivals, arrivals[1:]):
        itls_ms.append((cur - prev) * 1000.0)
    decode_s = (arrivals[-1] - arrivals[0]) if len(arrivals) >= 2 else None
    completion_tokens = usage.get("completion_tokens")
    prompt_tokens = usage.get("prompt_tokens")
    # Prefer the server token count. Chunk count can be lower when a
    # chunk carries more than one token (MTP or streamed batches).
    n_out = completion_tokens if isinstance(completion_tokens, int) else len(arrivals)
    decode_tok_s = None
    if decode_s and decode_s > 0 and n_out >= 2:
        decode_tok_s = (n_out - 1) / decode_s
    prefill_tok_s = None
    if ttft_s and ttft_s > 0 and isinstance(prompt_tokens, int) and prompt_tokens > 0:
        prefill_tok_s = prompt_tokens / ttft_s
    return {
        "text": text,
        "n_chunks": len(arrivals),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "ttft_ms": None if ttft_s is None else ttft_s * 1000.0,
        "itl_ms": {
            "n": len(itls_ms),
            "p50": percentile(itls_ms, 0.50),
            "p90": percentile(itls_ms, 0.90),
            "p99": percentile(itls_ms, 0.99),
            "mean": statistics.fmean(itls_ms) if itls_ms else None,
        },
        "decode_tok_s": decode_tok_s,
        "prefill_tok_s": prefill_tok_s,
        "usage": usage,
    }


def tokenize_count(base_url: str, model: str, prompt: str) -> int:
    body = post_json(
        base_url.rstrip("/") + "/tokenize",
        {"model": model, "prompt": prompt, "add_special_tokens": True},
        timeout=60,
    )
    count = body.get("count")
    if count is None and "tokens" in body:
        count = len(body["tokens"])
    if not isinstance(count, int):
        raise RuntimeError(f"tokenize response missing count: {list(body)[:8]}")
    return count


def make_prompt(base_url: str, model: str, n_tokens: int, nonce: str) -> tuple[str, int]:
    unit = "alpha bravo charlie delta echo foxtrot golf hotel. "
    prefix = f"bench {nonce} "
    text = prefix + unit * max(1, n_tokens // 8)
    count = tokenize_count(base_url, model, text)
    # Grow or shrink. A few linear steps are enough; the recorded
    # prompt_tokens from the completion is the measurement length.
    for _ in range(12):
        if count == n_tokens:
            break
        if count < n_tokens:
            need = n_tokens - count
            text += unit * max(1, need // 8)
        else:
            # Drop from the end, keeping the unique prefix.
            extra = count - n_tokens
            cut = max(len(unit), int(len(text) * extra / max(count, 1)))
            text = text[:-cut]
            if len(text) <= len(prefix):
                text = prefix + unit
        count = tokenize_count(base_url, model, text)
    return text, count


def identity(base_url: str, expect_first: str, expect_alias: str) -> dict[str, Any]:
    body = get_json(base_url.rstrip("/") + "/v1/models")
    ids = [row.get("id") for row in body.get("data") or []]
    return {
        "ids": ids,
        "first_ok": bool(ids) and ids[0] == expect_first,
        "alias_ok": expect_alias in ids,
        "expect_first": expect_first,
        "expect_alias": expect_alias,
    }


def coherence(base_url: str, model: str, timeout: float) -> dict[str, Any]:
    runs = []
    for _ in range(2):
        row = stream_completion(base_url, model, COHERENCE_PROMPT, 16, timeout)
        runs.append(row)
    texts = [row["text"] for row in runs]
    verdicts = [garbage_verdict(t) for t in texts]
    same = texts[0] == texts[1]
    paris = all("paris" in t.lower() for t in texts)
    ok = same and paris and all(v == "OK" or v == "SHORT" for v in verdicts)
    # A 16-token answer that names Paris is SHORT under the 12-char rule
    # only when the body is tiny. SHORT is acceptable if both match and
    # mention Paris. GARBAGE is not.
    if any(v == "GARBAGE" for v in verdicts):
        ok = False
    return {
        "ok": ok,
        "same": same,
        "mentions_paris": paris,
        "verdicts": verdicts,
        "texts": texts,
    }


def summarize_itl(row: dict[str, Any]) -> dict[str, Any]:
    itl = row["itl_ms"]
    return {
        "ttft_ms": row["ttft_ms"],
        "prompt_tokens": row["prompt_tokens"],
        "completion_tokens": row["completion_tokens"],
        "n_chunks": row["n_chunks"],
        "itl_p50_ms": itl["p50"],
        "itl_p90_ms": itl["p90"],
        "itl_p99_ms": itl["p99"],
        "itl_mean_ms": itl["mean"],
        "decode_tok_s": row["decode_tok_s"],
        "prefill_tok_s": row["prefill_tok_s"],
    }


def decode_c1(base_url: str, model: str, max_tokens: int, timeout: float) -> dict[str, Any]:
    warm = stream_completion(base_url, model, DECODE_PROMPT, 8, timeout)
    measured = stream_completion(base_url, model, DECODE_PROMPT, max_tokens, timeout)
    return {"warmup_ttft_ms": warm["ttft_ms"], "measured": summarize_itl(measured)}


def decode_c2(base_url: str, model: str, max_tokens: int, timeout: float) -> dict[str, Any]:
    # Distinct suffixes so the two streams do not share a cached decode,
    # while the shared stem still exercises concurrent decode.
    prompts = [
        DECODE_PROMPT + " Focus on the river.",
        DECODE_PROMPT + " Focus on the museums.",
    ]
    with ThreadPoolExecutor(max_workers=2) as pool:
        futs = [
            pool.submit(stream_completion, base_url, model, p, max_tokens, timeout)
            for p in prompts
        ]
        rows = [f.result() for f in futs]
    return {"streams": [summarize_itl(row) for row in rows]}


def prefill_sweep(
    base_url: str,
    model: str,
    lengths: list[int],
    timeout: float,
) -> list[dict[str, Any]]:
    out = []
    for n in lengths:
        nonce = f"n{n}-{time.time_ns()}"
        prompt, built = make_prompt(base_url, model, n, nonce)
        row = stream_completion(base_url, model, prompt, 1, timeout)
        out.append(
            {
                "target_tokens": n,
                "built_tokens": built,
                "measured": summarize_itl(row),
            }
        )
    return out


def spec_metrics(base_url: str) -> dict[str, Any] | None:
    try:
        text = get_text(base_url.rstrip("/") + "/metrics", timeout=15)
    except Exception as exc:
        return {"error": str(exc)}
    drafts = accepted = draft_tokens = 0
    found = False
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or not line.startswith("vllm:spec_decode"):
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            continue
        name = parts[0].split("{", 1)[0]
        if not name.endswith("_total"):
            continue
        found = True
        try:
            val = float(parts[-1])
        except ValueError:
            continue
        if "num_drafts" in name:
            drafts += int(val)
        elif "num_draft_tokens" in name:
            draft_tokens += int(val)
        elif "num_accepted_tokens" in name and "per_pos" not in name:
            accepted += int(val)
    if not found:
        return None
    rate = (accepted / draft_tokens) if draft_tokens else None
    return {
        "num_drafts": drafts,
        "num_draft_tokens": draft_tokens,
        "num_accepted_tokens": accepted,
        "acceptance_rate": rate,
    }


def fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def ascii_summary(report: dict[str, Any]) -> str:
    lines = []
    if report.get("tag"):
        lines.append(f"tag {report['tag']}")
    ident = report["identity"]
    lines.append(
        f"identity first={ident.get('ids', [None])[:1]} "
        f"first_ok={ident['first_ok']} alias_ok={ident['alias_ok']}"
    )
    coh = report.get("coherence")
    if coh:
        lines.append(
            f"coherence ok={coh['ok']} same={coh['same']} "
            f"paris={coh['mentions_paris']} verdicts={coh['verdicts']}"
        )
        for text in coh["texts"]:
            lines.append(f"  text: {text!r}")
    c1 = report.get("decode_c1", {}).get("measured")
    if c1:
        lines.append(
            "decode C1 "
            f"itl_p50={fmt(c1['itl_p50_ms'])} ms "
            f"p90={fmt(c1['itl_p90_ms'])} "
            f"p99={fmt(c1['itl_p99_ms'])} "
            f"tok/s={fmt(c1['decode_tok_s'])} "
            f"ttft_ms={fmt(c1['ttft_ms'])} "
            f"n={c1['completion_tokens']}"
        )
    c2 = report.get("decode_c2", {}).get("streams") or []
    for i, row in enumerate(c2):
        lines.append(
            f"decode C2 stream{i} "
            f"itl_p50={fmt(row['itl_p50_ms'])} ms "
            f"p90={fmt(row['itl_p90_ms'])} "
            f"p99={fmt(row['itl_p99_ms'])} "
            f"tok/s={fmt(row['decode_tok_s'])} "
            f"ttft_ms={fmt(row['ttft_ms'])}"
        )
    for row in report.get("prefill") or []:
        measured = row["measured"]
        lines.append(
            f"prefill target={row['target_tokens']} built={row['built_tokens']} "
            f"prompt_tokens={measured['prompt_tokens']} "
            f"ttft_ms={fmt(measured['ttft_ms'])} "
            f"prefill_tok/s={fmt(measured['prefill_tok_s'])}"
        )
    mtp = report.get("mtp")
    lines.append(f"mtp {mtp}")
    ref = report["strata_dd"]
    lines.append(
        "strata DD ref "
        f"C1 {ref['c1_tok_s']} tok/s (itl {ref['c1_itl_ms']:.2f} ms) "
        f"C2 {ref['c2_tok_s_per_stream']}/stream "
        f"(itl {ref['c2_itl_ms']:.2f} ms) "
        f"prefill 4k {ref['prefill_4k_tok_s']} 32k {ref['prefill_32k_tok_s']}"
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="vLLM XPU latency harness")
    parser.add_argument("--base-url", default="http://127.0.0.1:18081")
    parser.add_argument("--model", default="hotschmoe-dev")
    parser.add_argument(
        "--alias",
        default="ornith-1.5-35b-a3b-GPTQ-int4-mixedcal-v2",
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--decode-tokens", type=int, default=64)
    parser.add_argument("--prefill", default="1024,8192,32768")
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument(
        "--tag",
        default="",
        help="Short label stored in the JSON (eager, mtp, graph).",
    )
    parser.add_argument(
        "--skip-prefill",
        action="store_true",
        help="Identity, coherence, and decode only.",
    )
    args = parser.parse_args()
    lengths = [int(x) for x in args.prefill.split(",") if x.strip()]

    report: dict[str, Any] = {
        "tag": args.tag,
        "base_url": args.base_url,
        "model": args.model,
        "strata_dd": STRATA_DD,
    }
    report["identity"] = identity(args.base_url, args.model, args.alias)
    if not report["identity"]["first_ok"]:
        print(ascii_summary(report))
        print("identity check failed; not benchmarking", file=sys.stderr)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
            fh.write("\n")
        return 2
    report["coherence"] = coherence(args.base_url, args.model, args.timeout)
    mtp_before = spec_metrics(args.base_url)
    report["decode_c1"] = decode_c1(
        args.base_url, args.model, args.decode_tokens, args.timeout
    )
    report["decode_c2"] = decode_c2(
        args.base_url, args.model, args.decode_tokens, args.timeout
    )
    if not args.skip_prefill:
        report["prefill"] = prefill_sweep(
            args.base_url, args.model, lengths, args.timeout
        )
    mtp_after = spec_metrics(args.base_url)
    if mtp_before is None and mtp_after is None:
        report["mtp"] = {"enabled": False}
    elif isinstance(mtp_after, dict) and "error" not in mtp_after:
        before_acc = (mtp_before or {}).get("num_accepted_tokens") or 0
        before_draft = (mtp_before or {}).get("num_draft_tokens") or 0
        acc = mtp_after["num_accepted_tokens"] - before_acc
        draft = mtp_after["num_draft_tokens"] - before_draft
        report["mtp"] = {
            "enabled": True,
            "accepted_during_bench": acc,
            "draft_tokens_during_bench": draft,
            "acceptance_rate": (acc / draft) if draft else None,
            "after": mtp_after,
        }
    else:
        report["mtp"] = {"enabled": False, "before": mtp_before, "after": mtp_after}

    text = ascii_summary(report)
    print(text)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
        fh.write("\n")
    print(f"wrote {args.out}")
    if not report["coherence"]["ok"]:
        return 3
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
