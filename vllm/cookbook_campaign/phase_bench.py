#!/usr/bin/env python3
"""Phase-separated client post-first bench (cookbook methodology, item 5).

Matches the SergiioB B70 cookbook contract:

  * Client post-first tok/s =
      (completion_tokens - 1) / (request_end - first_generated_token)
  * Cold prefill proxy =
      actual_prompt_tokens / TTFT  (not isolated engine prefill)
  * Unique entropy-first cold prefixes (zero intentional cache hit)
  * n repeats after one same-shape warmup; report median

Also records vLLM /metrics accept counters when available so MTP accept_len
is comparable to our older benches.

Usage:
  python3 phase_bench.py --base http://127.0.0.1:8000 --model M \\
    --prompt-tokens 512 --gen-tokens 128 --n 5 --out results.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import time
import urllib.error
import urllib.request
from typing import Any


def _post_json(url: str, payload: dict, timeout: float = 600.0) -> tuple[dict, float, float, float]:
    """Return (final_json_or_stream_agg, t_send, t_first, t_end) monotonic."""
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY', 'EMPTY')}"},
        method="POST",
    )
    t_send = time.monotonic()
    t_first = None
    body_parts: list[bytes] = []
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        if payload.get("stream"):
            completion_tokens = 0
            text_bits: list[str] = []
            finish_reason = None
            for raw in resp:
                if t_first is None and raw.strip():
                    t_first = time.monotonic()
                line = raw.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                payload_s = line[5:].strip()
                if payload_s == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload_s)
                except json.JSONDecodeError:
                    continue
                choice = (chunk.get("choices") or [{}])[0]
                delta = choice.get("delta") or {}
                if "content" in delta and delta["content"]:
                    text_bits.append(delta["content"])
                    completion_tokens += 1  # approx; corrected by usage if present
                if choice.get("finish_reason"):
                    finish_reason = choice.get("finish_reason")
                usage = chunk.get("usage")
                if usage:
                    # some servers send cumulative usage on last chunk
                    pass
            t_end = time.monotonic()
            # Prefer usage from a final non-stream request fallback: re-count
            text = "".join(text_bits)
            return (
                {
                    "stream": True,
                    "text": text,
                    "approx_completion_tokens": completion_tokens,
                    "finish_reason": finish_reason,
                },
                t_send,
                t_first or t_end,
                t_end,
            )
        else:
            body = resp.read()
            t_end = time.monotonic()
            t_first = t_end  # non-stream: no TTFT separation
            return json.loads(body), t_send, t_first, t_end


def _stream_chat(
    base: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    timeout: float,
) -> dict[str, Any]:
    """SSE chat completion with true first-token timing."""
    url = base.rstrip("/") + "/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY', 'EMPTY')}",
        },
        method="POST",
    )
    t_send = time.monotonic()
    t_first = None
    text_bits: list[str] = []
    usage: dict | None = None
    finish_reason = None
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            if not raw.strip():
                continue
            if t_first is None:
                t_first = time.monotonic()
            line = raw.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            payload_s = line[5:].strip()
            if payload_s == "[DONE]":
                break
            try:
                chunk = json.loads(payload_s)
            except json.JSONDecodeError:
                continue
            if chunk.get("usage"):
                usage = chunk["usage"]
            choices = chunk.get("choices") or []
            if not choices:
                continue
            choice = choices[0]
            delta = choice.get("delta") or {}
            if delta.get("content"):
                text_bits.append(delta["content"])
            if choice.get("finish_reason"):
                finish_reason = choice["finish_reason"]
    t_end = time.monotonic()
    if t_first is None:
        t_first = t_end
    text = "".join(text_bits)
    # Prefer server usage; fall back to whitespace-ish estimate is wrong for BPE.
    # If usage missing, count streamed chunks as completion tokens (1 delta ~ 1 token
    # is imperfect but better than nothing; we force include_usage).
    if usage:
        prompt_tokens = int(usage.get("prompt_tokens") or 0)
        completion_tokens = int(usage.get("completion_tokens") or 0)
    else:
        prompt_tokens = 0
        completion_tokens = max(len(text_bits), 1)
    ttft = t_first - t_send
    e2e = t_end - t_send
    # Client post-first: (completion - 1) / (end - first)
    if completion_tokens > 1 and (t_end - t_first) > 0:
        post_first = (completion_tokens - 1) / (t_end - t_first)
    else:
        post_first = float("nan")
    prefill_proxy = (prompt_tokens / ttft) if ttft > 0 and prompt_tokens > 0 else float("nan")
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "ttft_s": ttft,
        "e2e_s": e2e,
        "post_first_tok_s": post_first,
        "prefill_proxy_tok_s": prefill_proxy,
        "finish_reason": finish_reason,
        "text_hash": hashlib.sha256(text.encode()).hexdigest()[:16],
        "text_preview": text[:160].replace("\n", "\\n"),
    }


def _entropy_prompt(n_tokens_target: int, salt: str) -> str:
    """Build a unique long prompt; exact token count is server-side.

    We over-generate English-ish pseudo-random content so the tokenizer lands
    near the target; the reported prompt_tokens from usage is authoritative.
    """
    # ~0.75 words/token ballpark for English BPE; overshoot then trim by words.
    words_needed = int(n_tokens_target * 1.35) + 64
    # deterministic from salt
    h = hashlib.sha256(salt.encode()).digest()
    vocab = [
        "alpha", "bravo", "cache", "delta", "echo", "foxtrot", "gamma", "helix",
        "ion", "jade", "kilo", "lemma", "monad", "nexus", "omega", "prism",
        "quark", "rho", "sigma", "tensor", "ultra", "vector", "wave", "xenon",
        "yield", "zeta", "arc", "battlemage", "prefill", "decode", "spec",
    ]
    out = [f"SALT={salt}", "Count the unique tokens in this entropy block and then answer: what is 2+2?"]
    state = int.from_bytes(h[:8], "big")
    for i in range(words_needed):
        state = (state * 6364136223846793005 + 1) & 0xFFFFFFFFFFFFFFFF
        out.append(vocab[state % len(vocab)])
        if i % 32 == 31:
            out.append(f"#{i}")
    return " ".join(out)


def _get_metrics(base: str) -> str:
    try:
        with urllib.request.urlopen(base.rstrip("/") + "/metrics", timeout=10) as r:
            return r.read().decode("utf-8", errors="replace")
    except Exception:
        return ""


def _parse_accept(metrics: str) -> dict[str, float]:
    """Best-effort accept counters from vLLM /metrics."""
    out: dict[str, float] = {}
    for line in metrics.splitlines():
        if line.startswith("#"):
            continue
        if "spec" in line.lower() or "accept" in line.lower() or "draft" in line.lower():
            parts = line.split()
            if len(parts) >= 2:
                try:
                    out[parts[0]] = float(parts[-1])
                except ValueError:
                    pass
    return out


def _median(xs: list[float]) -> float:
    return float(statistics.median(xs)) if xs else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt-tokens", type=int, default=512)
    ap.add_argument("--gen-tokens", type=int, default=128)
    ap.add_argument("--n", type=int, default=5, help="timed repeats after warmup")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument("--out", default="")
    ap.add_argument("--label", default="")
    ap.add_argument("--skip-warmup", action="store_true")
    args = ap.parse_args()

    # discover model if needed
    model = args.model
    if model == "auto":
        with urllib.request.urlopen(args.base.rstrip("/") + "/v1/models", timeout=30) as r:
            models = json.loads(r.read())["data"]
            model = models[0]["id"]

    meta = {
        "base": args.base,
        "model": model,
        "prompt_tokens_target": args.prompt_tokens,
        "gen_tokens": args.gen_tokens,
        "n": args.n,
        "label": args.label,
        "methodology": {
            "post_first": "(completion_tokens - 1) / (request_end - first_token)",
            "prefill_proxy": "prompt_tokens / TTFT",
            "prefixes": "unique entropy-first cold",
            "warmup": "one same-shape then n timed",
        },
    }

    metrics_before = _parse_accept(_get_metrics(args.base))

    runs: list[dict] = []
    # warmup
    if not args.skip_warmup:
        warm_prompt = _entropy_prompt(args.prompt_tokens, salt=f"warmup-{args.label}-{args.prompt_tokens}")
        print(f"[warmup] p~{args.prompt_tokens} g{args.gen_tokens}", flush=True)
        try:
            w = _stream_chat(args.base, model, warm_prompt, args.gen_tokens, args.temperature, args.timeout)
            print(f"[warmup] done ttft={w['ttft_s']:.3f}s post_first={w['post_first_tok_s']:.2f}", flush=True)
            meta["warmup"] = w
        except Exception as e:
            print(f"[warmup] FAILED: {e}", file=sys.stderr)
            meta["warmup_error"] = str(e)

    for i in range(args.n):
        salt = f"{args.label}-p{args.prompt_tokens}-g{args.gen_tokens}-i{i}-{time.time_ns()}"
        prompt = _entropy_prompt(args.prompt_tokens, salt=salt)
        print(f"[run {i+1}/{args.n}] salt={salt[:24]}...", flush=True)
        try:
            r = _stream_chat(args.base, model, prompt, args.gen_tokens, args.temperature, args.timeout)
            r["i"] = i
            r["salt"] = salt
            runs.append(r)
            print(
                f"  prompt={r['prompt_tokens']} comp={r['completion_tokens']} "
                f"ttft={r['ttft_s']:.3f}s post_first={r['post_first_tok_s']:.2f} "
                f"prefill_proxy={r['prefill_proxy_tok_s']:.1f}",
                flush=True,
            )
        except Exception as e:
            print(f"  FAILED: {e}", file=sys.stderr)
            runs.append({"i": i, "error": str(e)})

    metrics_after = _parse_accept(_get_metrics(args.base))
    ok = [r for r in runs if "error" not in r]
    summary = {
        **meta,
        "runs": runs,
        "n_ok": len(ok),
        "median_post_first_tok_s": _median([r["post_first_tok_s"] for r in ok]),
        "median_prefill_proxy_tok_s": _median([r["prefill_proxy_tok_s"] for r in ok]),
        "median_ttft_s": _median([r["ttft_s"] for r in ok]),
        "median_prompt_tokens": _median([float(r["prompt_tokens"]) for r in ok]),
        "median_completion_tokens": _median([float(r["completion_tokens"]) for r in ok]),
        "metrics_before": metrics_before,
        "metrics_after": metrics_after,
    }
    # delta counters that look numeric
    deltas = {}
    for k, v in metrics_after.items():
        if k in metrics_before:
            deltas[k] = v - metrics_before[k]
        else:
            deltas[k] = v
    summary["metrics_delta"] = deltas

    text = json.dumps(summary, indent=2, sort_keys=True)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            f.write(text + "\n")
        print(f"[wrote] {args.out}", flush=True)
    print(
        f"[summary] median post_first={summary['median_post_first_tok_s']:.2f} "
        f"prefill_proxy={summary['median_prefill_proxy_tok_s']:.1f} "
        f"ttft={summary['median_ttft_s']:.3f}s n_ok={summary['n_ok']}/{args.n}",
        flush=True,
    )
    return 0 if summary["n_ok"] == args.n else 1


if __name__ == "__main__":
    raise SystemExit(main())
