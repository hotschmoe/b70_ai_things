#!/usr/bin/env python3
"""Self-contained reproduction of Steve Seguin's Qwen3.6 smoke protocol.

The natural-chat prompt construction and corrected after-first-chunk metric are
adapted from ``b70-optimization-lab/scripts/measure-openai-endpoint-metrics.py``
(Steve Seguin, Unlicense). This focused copy has no runtime dependency on that
repository.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
import urllib.request
from pathlib import Path

from transformers import AutoTokenizer


METRIC_RE = re.compile(
    r"^([A-Za-z_:][A-Za-z0-9_:]*)(?:\{[^}]*\})?\s+([-+0-9.eE]+)$"
)


def get_json(url: str):
    with urllib.request.urlopen(url, timeout=30) as response:
        return json.loads(response.read())


def get_text(url: str) -> str:
    with urllib.request.urlopen(url, timeout=30) as response:
        return response.read().decode("utf-8", errors="replace")


def parse_metrics(text: str) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for line in text.splitlines():
        match = METRIC_RE.match(line)
        if match:
            name, value = match.groups()
            metrics[name] = metrics.get(name, 0.0) + float(value)
    return metrics


def delta(before: dict[str, float], after: dict[str, float], name: str) -> float:
    return after.get(name, 0.0) - before.get(name, 0.0)


def fit_natural_chat_prompt(tokenizer, target_tokens: int) -> str:
    prefix = (
        "You are helping tune an Intel XPU inference server. "
        "Write a concise engineering analysis with concrete next steps.\n\n"
    )
    filler = (
        "Recent observations include stable baseline decoding, prompt-sensitive "
        "speculative acceptance, graph capture bucket sensitivity, and the need "
        "to preserve exact output quality while improving single-request speed. "
    )
    suffix = (
        "\n\nQuestion: summarize the likely bottlenecks and propose an ordered "
        "plan. Keep the answer technical and avoid marketing language. Write at "
        "least eight dense numbered paragraphs.\n"
    )
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    filler_ids = tokenizer.encode(filler, add_special_tokens=False)
    suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)
    body_budget = max(0, target_tokens - len(prefix_ids) - len(suffix_ids))
    repeats = (body_budget + len(filler_ids) - 1) // len(filler_ids)
    ids = prefix_ids + (filler_ids * repeats)[:body_budget] + suffix_ids
    return tokenizer.decode(ids[:target_tokens], skip_special_tokens=True)


def stream_completion(
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    seed: int,
) -> dict[str, object]:
    payload = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0,
            "stream": True,
            "stream_options": {"include_usage": True},
            "seed": seed,
            "ignore_eos": True,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        base_url.rstrip("/") + "/v1/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    first_at = None
    first_text = ""
    chunks: list[str] = []
    usage = None
    request_id = None
    with urllib.request.urlopen(request, timeout=max(180, max_tokens * 5)) as response:
        for raw in response:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            event = json.loads(data)
            request_id = event.get("id") or request_id
            usage = event.get("usage") or usage
            choices = event.get("choices") or []
            text = (choices[0].get("text") or "") if choices else ""
            if text and first_at is None:
                first_at = time.perf_counter()
                first_text = text
            if text:
                chunks.append(text)
    finished = time.perf_counter()
    return {
        "request_id": request_id,
        "text": "".join(chunks),
        "usage": usage,
        "elapsed_s": finished - started,
        "ttft_s": None if first_at is None else first_at - started,
        "after_first_s": None if first_at is None else finished - first_at,
        "first_text": first_text,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:18080")
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    prompt = fit_natural_chat_prompt(tokenizer, 512)
    prompt_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))

    stream_completion(args.base_url, args.model, prompt, 64, seed=-1)
    before = parse_metrics(get_text(args.base_url.rstrip("/") + "/metrics"))
    result = stream_completion(args.base_url, args.model, prompt, 512, seed=0)
    after = parse_metrics(get_text(args.base_url.rstrip("/") + "/metrics"))

    usage = result.get("usage") or {}
    output_tokens = int(usage.get("completion_tokens") or 0)
    if not output_tokens:
        output_tokens = len(
            tokenizer.encode(str(result["text"]), add_special_tokens=False)
        )
    first_tokens = len(
        tokenizer.encode(str(result["first_text"]), add_special_tokens=False)
    )
    elapsed = float(result["elapsed_s"])
    after_first = result["after_first_s"]
    ttft = result["ttft_s"]
    corrected = (
        None
        if not after_first
        else max(0, output_tokens - first_tokens) / float(after_first)
    )
    artifact = {
        "protocol": {
            "prompt": "natural-chat",
            "prompt_tokens_requested": 512,
            "warmup_output_tokens": 64,
            "output_tokens_requested": 512,
            "repeats": 1,
            "temperature": 0,
            "ignore_eos": True,
            "stream": True,
        },
        "model": args.model,
        "server_model_record": get_json(
            args.base_url.rstrip("/") + "/v1/models"
        )["data"][0],
        "prompt_tokens_actual": prompt_tokens,
        "output_tokens_actual": output_tokens,
        "first_chunk_tokens": first_tokens,
        "elapsed_s_client": elapsed,
        "ttft_ms_client": None if ttft is None else float(ttft) * 1000.0,
        "tok_s_out_client_e2e": output_tokens / elapsed,
        "tok_s_out_client_after_first_chunk_corrected": corrected,
        "vllm_metric_deltas": {
            "prompt_tokens": delta(before, after, "vllm:prompt_tokens_total"),
            "generation_tokens": delta(
                before, after, "vllm:generation_tokens_total"
            ),
            "ttft_sum_s": delta(
                before, after, "vllm:time_to_first_token_seconds_sum"
            ),
            "e2e_sum_s": delta(
                before, after, "vllm:e2e_request_latency_seconds_sum"
            ),
            "decode_sum_s": delta(
                before, after, "vllm:request_decode_time_seconds_sum"
            ),
        },
        "text": str(result["text"]),
        "text_sha256": hashlib.sha256(
            str(result["text"]).encode("utf-8")
        ).hexdigest(),
        "output_token_ids": tokenizer.encode(
            str(result["text"]), add_special_tokens=False
        ),
        "output_token_ids_source": "retokenized_text",
        "text_preview": str(result["text"])[:400],
        "printable_ascii_fraction": sum(
            character == "\n" or 32 <= ord(character) <= 126
            for character in str(result["text"])
        )
        / max(1, len(str(result["text"]))),
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(artifact, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
