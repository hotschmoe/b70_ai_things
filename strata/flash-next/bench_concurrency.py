#!/usr/bin/env python3
"""Measure two overlapping requests through an already pair-leased server."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import statistics
import threading
import time

from bench_serving import get, send


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:18085")
    parser.add_argument("--model", default="hotschmoe-dd")
    parser.add_argument("--template", type=Path, required=True,
                        help="Saved bench_serving coding response JSON")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    base = args.base_url.rstrip("/")
    identity = get(base, "/v1/models")
    ids = [r["id"] for r in identity["data"]]
    alias = "qwen3.8-flash-next-ISTA-GSQ-RCO-IQ3_XXS-strata"
    registry = (Path(__file__).resolve().parents[2]/"evals/configs/models.yaml").read_text()
    if ids[0] != args.model or alias not in ids or alias not in registry:
        raise RuntimeError("Model identity mismatch")
    (args.output/"models.json").write_text(json.dumps(identity, indent=2)+"\n")
    source = json.loads(args.template.read_text())["request"]
    prompt = source["messages"][0]["content"].split("] ", 1)[1]
    cap = source["max_tokens"]
    windows = []
    for repeat in range(args.repeats):
        barrier = threading.Barrier(2)

        def request(slot):
            content = f"[Flash-Next concurrent coding repeat {repeat} slot {slot}] " + prompt
            barrier.wait(timeout=10)
            result = send(base, args.model, [{"role": "user", "content": content}], cap)
            if result["timings"]["cache_n"] > 64:
                raise RuntimeError("Fresh concurrency prompt reused more than 64 tokens")
            result["request_sha256"] = hashlib.sha256(
                json.dumps(result["request"], sort_keys=True).encode()).hexdigest()
            (args.output/f"coding-{repeat}-{slot}.json").write_text(
                json.dumps(result, indent=2, ensure_ascii=True)+"\n")
            return result

        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(request, slot) for slot in range(2)]
            rows = [future.result() for future in futures]
        wall = time.perf_counter()-start
        window = dict(repeat=repeat, wall_s=wall,
                      aggregate_output_tps=sum(r["usage"]["completion_tokens"] for r in rows)/wall,
                      ttft_s=[r["ttft_s"] for r in rows],
                      timings=[r["timings"] for r in rows],
                      hashes=[r["output_sha256"] for r in rows])
        windows.append(window)
        print(json.dumps(window), flush=True)
    summary = dict(windows=windows,
                   median_aggregate_output_tps=statistics.median(
                       w["aggregate_output_tps"] for w in windows))
    (args.output/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    (args.output/"health-end.json").write_text(json.dumps(get(base,"/health"),indent=2)+"\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
