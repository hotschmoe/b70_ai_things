#!/usr/bin/env python3
"""Qualify forced-length concurrent OpenAI-compatible serving capacity."""

import argparse
import concurrent.futures
import hashlib
import json
import statistics
import time
import urllib.request


def post_json(url, body, timeout=600):
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("ascii"),
        headers={"content-type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def get_json(url, timeout=30):
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.load(response)


def message_text(response):
    message = response["choices"][0]["message"]
    return (message.get("reasoning_content") or "") + (message.get("content") or "")


def run_determinism(url, model):
    body = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": "Explain Rayleigh scattering in three concise sentences.",
            }
        ],
        "max_tokens": 96,
        "temperature": 0,
    }
    texts = [message_text(post_json(url, body)) for _ in range(2)]
    if not texts[0] or texts[0] != texts[1]:
        raise RuntimeError("repeated greedy response was not byte-identical")
    print(
        "DETERMINISM -> bytes={} sha256={}".format(
            len(texts[0].encode("utf-8")),
            hashlib.sha256(texts[0].encode("utf-8")).hexdigest(),
        )
    )


def run_canaries(url, model, concurrency):
    cases = [
        ("Reply with only the integer: 19 + 26.", "45"),
        ("Reply with only the integer: 34 + 44.", "78"),
        ("Reply with only the integer: 31 * 3.", "93"),
        ("Reply with only the integer: 21 * 9.", "189"),
        ("Reply with only the capital of France.", "Paris"),
        ("Reply with only the largest planet in our solar system.", "Jupiter"),
        ("Reply with only the chemical symbol for gold.", "Au"),
        ("Reply with only the number of sides in a square.", "4"),
    ][:concurrency]

    def one(case):
        prompt, expected = case
        response = post_json(
            url,
            {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 64,
                "temperature": 0,
            },
        )
        text = message_text(response)
        return expected, text, expected.lower() in text.lower()

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        results = list(pool.map(one, cases))
    for expected, text, passed in results:
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        print(
            "CANARY expected={} pass={} bytes={} sha256={}".format(
                expected, int(passed), len(text.encode("utf-8")), digest
            )
        )
    if not all(result[2] for result in results):
        raise RuntimeError("concurrent coherence canary failed")


def stream_one(url, model, prompt, output_tokens):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": output_tokens,
        "temperature": 0,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("ascii"),
        headers={"content-type": "application/json"},
    )
    start = time.perf_counter()
    first = None
    end = None
    completion_tokens = None
    prompt_tokens = None
    finish_reason = None
    chunks = []
    with urllib.request.urlopen(request, timeout=600) as response:
        for raw in response:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            event = json.loads(data)
            usage = event.get("usage")
            if usage:
                completion_tokens = usage.get("completion_tokens")
                prompt_tokens = usage.get("prompt_tokens")
            choices = event.get("choices") or []
            if not choices:
                continue
            if choices[0].get("finish_reason"):
                finish_reason = choices[0]["finish_reason"]
            delta = choices[0].get("delta") or {}
            text = (
                delta.get("reasoning_content")
                or delta.get("reasoning")
                or delta.get("content")
                or ""
            )
            if text:
                if first is None:
                    first = time.perf_counter()
                chunks.append(text)
    end = time.perf_counter()
    if first is None or completion_tokens is None or prompt_tokens is None:
        raise RuntimeError("stream omitted timing or usage fields")
    return {
        "start": start,
        "first": first,
        "end": end,
        "completion_tokens": completion_tokens,
        "prompt_tokens": prompt_tokens,
        "finish_reason": finish_reason,
        "text_sha256": hashlib.sha256("".join(chunks).encode("utf-8")).hexdigest(),
    }


def run_batch(base, model, concurrency, output_tokens, prompt_repeat, batch):
    paragraph = (
        "Modern accelerators combine memory hierarchies, matrix engines, collective "
        "communication, graph replay, and low precision arithmetic. "
    )
    prompts = [
        "Batch {} stream {} nonce {}. Analyze these notes carefully.\n{}".format(
            batch, index, 104729 * (batch + 1) + index, paragraph * prompt_repeat
        )
        for index in range(concurrency)
    ]
    url = base + "/chat/completions"
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [
            pool.submit(stream_one, url, model, prompt, output_tokens)
            for prompt in prompts
        ]
        results = [future.result() for future in futures]
    if not all(result["completion_tokens"] == output_tokens for result in results):
        raise RuntimeError("a stream did not return the forced output length")
    if not all(result["finish_reason"] == "length" for result in results):
        raise RuntimeError("a stream did not finish because of the length limit")

    decode_tokens = sum(result["completion_tokens"] - 1 for result in results)
    decode_window = max(result["end"] for result in results) - min(
        result["first"] for result in results
    )
    total_tokens = sum(result["completion_tokens"] for result in results)
    total_window = max(result["end"] for result in results) - min(
        result["start"] for result in results
    )
    per_stream = [
        (result["completion_tokens"] - 1) / (result["end"] - result["first"])
        for result in results
    ]
    ttft = [(result["first"] - result["start"]) * 1000 for result in results]
    metric = {
        "batch": batch,
        "concurrency": concurrency,
        "streams_ok": len(results),
        "prompt_tokens_min": min(result["prompt_tokens"] for result in results),
        "prompt_tokens_max": max(result["prompt_tokens"] for result in results),
        "completion_tokens_each": output_tokens,
        "aggregate_post_first_tok_s": decode_tokens / decode_window,
        "aggregate_including_ttft_tok_s": total_tokens / total_window,
        "median_stream_post_first_tok_s": statistics.median(per_stream),
        "median_ttft_ms": statistics.median(ttft),
        "response_hashes": [result["text_sha256"] for result in results],
    }
    print("RESULT -> " + json.dumps(metric, sort_keys=True))
    return metric


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--output-tokens", type=int, default=128)
    parser.add_argument("--prompt-repeat", type=int, default=260)
    parser.add_argument("--batches", type=int, default=3)
    parser.add_argument("--json-out")
    args = parser.parse_args()
    base = args.base.rstrip("/") + "/v1"
    model_ids = [item["id"] for item in get_json(base + "/models")["data"]]
    if model_ids != [args.model]:
        raise RuntimeError("served model identity mismatch: {}".format(model_ids))
    print("IDENTITY -> " + json.dumps(model_ids))
    run_determinism(base + "/chat/completions", args.model)
    run_canaries(base + "/chat/completions", args.model, args.concurrency)
    print("COMMAND -> same-shape warmup")
    run_batch(base, args.model, args.concurrency, args.output_tokens, args.prompt_repeat, -1)
    metrics = [
        run_batch(
            base,
            args.model,
            args.concurrency,
            args.output_tokens,
            args.prompt_repeat,
            batch,
        )
        for batch in range(args.batches)
    ]
    summary = {
        "model": args.model,
        "concurrency": args.concurrency,
        "batches": args.batches,
        "median_aggregate_post_first_tok_s": statistics.median(
            metric["aggregate_post_first_tok_s"] for metric in metrics
        ),
        "median_aggregate_including_ttft_tok_s": statistics.median(
            metric["aggregate_including_ttft_tok_s"] for metric in metrics
        ),
        "metrics": metrics,
    }
    print("VERDICT -> " + json.dumps(summary, sort_keys=True))
    if args.json_out:
        with open(args.json_out, "w", encoding="ascii") as output:
            json.dump(summary, output, indent=2, sort_keys=True)
            output.write("\n")


if __name__ == "__main__":
    main()
