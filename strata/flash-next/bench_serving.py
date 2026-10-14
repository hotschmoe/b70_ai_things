#!/usr/bin/env python3
"""Measure a leased Strata endpoint; save identity, raw SSE, and engine timings.

Run while the server holds bin/gpu-run's pair lease. Fresh case/repeat prefixes
are identical across arms. Read prompt_n/cache_n before comparing prefill.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import statistics
import time
import urllib.request


def get(base, path):
    with urllib.request.urlopen(base + path, timeout=10) as response:
        return json.load(response)


def send(base, model, messages, cap, *, spec_min_p=None):
    body = dict(model=model, messages=messages, max_tokens=cap, temperature=0,
                stream=True, stream_options={"include_usage": True},
                chat_template_kwargs={"enable_thinking": False})
    if spec_min_p is not None:
        body["strata_tune"] = {"spec_min_p": spec_min_p}
    request = urllib.request.Request(base + "/v1/chat/completions",
                                     json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
    start = time.perf_counter()
    first = None
    chunks = []
    usage = timings = None
    with urllib.request.urlopen(request, timeout=1800) as response:
        for line in response:
            if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
                continue
            chunk = json.loads(line[6:])
            if chunk.get("error"):
                raise RuntimeError(chunk["error"])
            chunks.append(chunk)
            usage = chunk.get("usage") or usage
            timings = chunk.get("timings") or timings
            for choice in chunk.get("choices", []):
                delta = choice.get("delta", {})
                if first is None and (delta.get("content") or delta.get("reasoning_content")):
                    first = time.perf_counter()
    end = time.perf_counter()
    if not usage or first is None or not timings:
        raise RuntimeError("Missing usage, first token, or engine timings")
    text = "".join(choice.get("delta", {}).get("content") or ""
                   for chunk in chunks for choice in chunk.get("choices", []))
    return dict(request=body, ttft_s=first-start, wall_s=end-start,
                client_decode_tps=(usage["completion_tokens"]-1)/(end-first),
                usage=usage, timings=timings, text=text,
                output_sha256=hashlib.sha256(text.encode()).hexdigest(), chunks=chunks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:18085")
    parser.add_argument("--model", default="hotschmoe-dd")
    parser.add_argument("--alias", default="qwen3.8-flash-next-ISTA-GSQ-RCO-IQ3_XXS-strata")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--spec-min-p", type=float)
    parser.add_argument("--cases", nargs="+", choices=["coding", "p4096", "p32768"],
                        default=["coding", "p4096", "p32768"])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    base = args.base_url.rstrip("/")
    identity = get(base, "/v1/models")
    (args.output / "models.json").write_text(json.dumps(identity, indent=2) + "\n")
    ids = [row["id"] for row in identity["data"]]
    if ids[0] != args.model or args.alias not in ids:
        raise RuntimeError("Live stable name or research alias mismatch")
    registry = (Path(__file__).resolve().parents[2] / "evals/configs/models.yaml").read_text()
    if args.alias not in registry:
        raise RuntimeError("Research alias absent from eval model registry")
    def record(name, messages, cap):
        result = send(base, args.model, messages, cap, spec_min_p=args.spec_min_p)
        (args.output / (name + ".json")).write_text(json.dumps(result, indent=2, ensure_ascii=True) + "\n")
        print(json.dumps(dict(name=name, usage=result["usage"], timings=result["timings"],
                              client_decode_tps=result["client_decode_tps"],
                              output_sha256=result["output_sha256"])), flush=True)
        return result
    checks = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(record, "coherence-" + str(i),
                               [{"role": "user", "content": f"Return only the integer: {a} * {b}."}], 32)
                   for i, (a, b) in enumerate([(17, 19), (23, 29)])]
        checks = [f.result() for f in futures]
    if [r["text"].strip() for r in checks] != ["323", "667"]:
        raise RuntimeError("Concurrent deterministic arithmetic failed")
    unit = "The history of computing is long and full of surprising turns. "
    summaries = {}
    for case in args.cases:
        rows = []
        for repeat in range(args.repeats):
            prefix = f"[Flash-Next serving benchmark {case} repeat {repeat}] "
            if case == "coding":
                prompt = prefix + ("Implement a Python asyncio bounded worker pool for processing file paths. "
                    "Requirements: fixed worker count, bounded queue for backpressure, preserve input order "
                    "in results, propagate the first worker exception, cancel and await all outstanding tasks "
                    "on failure, and handle caller cancellation without leaked tasks. Explain the shutdown "
                    "protocol, provide complete typed code using only the standard library, and include "
                    "three meaningful tests for ordering, worker failure, and cancellation.")
                cap = 1500
            else:
                target = int(case[1:])
                prompt = prefix + unit * ((target-73)//12) + "\nWrite a detailed explanation of GPU memory hierarchies."
                cap = 256
            result = record(f"{case}-{repeat}", [{"role": "user", "content": prompt}], cap)
            if result["usage"]["prompt_tokens_details"]["cached_tokens"] > 64:
                raise RuntimeError("Fresh-prompt benchmark reused more than 64 tokens")
            rows.append(result)
        summaries[case] = {key: statistics.median(r["timings"][key] for r in rows)
                           for key in ["prompt_per_second", "predicted_per_second",
                                       "prompt_ms", "predicted_per_token_ms"]}
        summaries[case]["client_decode_tps"] = statistics.median(r["client_decode_tps"] for r in rows)
        summaries[case]["ttft_s"] = statistics.median(r["ttft_s"] for r in rows)
        summaries[case]["wall_s"] = statistics.median(r["wall_s"] for r in rows)
    (args.output / "health-end.json").write_text(json.dumps(get(base, "/health"), indent=2) + "\n")
    (args.output / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    print(json.dumps(summaries), flush=True)


if __name__ == "__main__":
    main()
