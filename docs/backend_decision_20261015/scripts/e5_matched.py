#!/usr/bin/env python3
"""E5: Strata's own C1/C2 coding harness against llama-server (matched prompts, caps, metrics).

C1 = bench_serving.py "coding" case (3 fresh repeats, 1500 tokens, greedy).
C2 = bench_concurrency.py logic (one warm pair, then 3 fresh pairs, aggregate output tok/s).
Only the identity check differs: the research alias names the llama.cpp engine, not Strata.
Usage: e5_matched.py BASE_URL OUTDIR
"""
import hashlib, json, statistics, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, "/mnt/vm_8tb/github/b70_ai_things/strata/flash-next")
from bench_serving import get, send  # noqa: E402

BASE, OUT = sys.argv[1].rstrip("/"), Path(sys.argv[2])
MODEL, ALIAS = "hotschmoe-dd", "qwen3.8-flash-next-ISTA-GSQ-RCO-IQ3_XXS-llamacpp-sycl"
TEMPLATE = Path("/mnt/vm_8tb/b70/build/backend-decision/e2-coding-template.json")


def save(name, r):
    r = {k: v for k, v in r.items() if k != "chunks"}
    (OUT / f"{name}.json").write_text(json.dumps(r, indent=1, ensure_ascii=True) + "\n")


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    ident = get(BASE, "/v1/models")
    (OUT / "models.json").write_text(json.dumps(ident, indent=2) + "\n")
    ids = [m["id"] for m in ident["data"]] + [a for m in ident["data"] for a in m.get("aliases", [])]
    if ids[0] != MODEL or ALIAS not in ids:
        raise RuntimeError(f"identity mismatch {ids}")
    with ThreadPoolExecutor(max_workers=2) as pool:
        checks = list(pool.map(lambda ab: send(BASE, MODEL, [{"role": "user", "content":
                      f"Return only the integer: {ab[0]} * {ab[1]}."}], 32), [(17, 19), (23, 29)]))
    if [c["text"].strip() for c in checks] != ["323", "667"]:
        raise RuntimeError("concurrent arithmetic failed: " + repr([c["text"] for c in checks]))
    coding = ("Implement a Python asyncio bounded worker pool for processing file paths. "
              "Requirements: fixed worker count, bounded queue for backpressure, preserve input order "
              "in results, propagate the first worker exception, cancel and await all outstanding tasks "
              "on failure, and handle caller cancellation without leaked tasks. Explain the shutdown "
              "protocol, provide complete typed code using only the standard library, and include "
              "three meaningful tests for ordering, worker failure, and cancellation.")
    c1 = []
    for rep in range(3):
        r = send(BASE, MODEL, [{"role": "user", "content": f"[Flash-Next serving benchmark coding repeat {rep}] " + coding}], 1500)
        save(f"c1-coding-{rep}", r)
        c1.append(r)
        print("C1", rep, round(r["client_decode_tps"], 2), r["usage"]["completion_tokens"],
              (r["timings"] or {}).get("draft_n_accepted"), "/", (r["timings"] or {}).get("draft_n"), flush=True)
    src = json.loads(TEMPLATE.read_text())["request"]
    prompt, cap = src["messages"][0]["content"].split("] ", 1)[1], src["max_tokens"]
    windows = []
    for rep in [99, 0, 1, 2]:  # 99 = warm pair, excluded from the median
        bar = threading.Barrier(2)

        def req(slot):
            bar.wait(timeout=10)
            return send(BASE, MODEL, [{"role": "user", "content": f"[Flash-Next concurrent coding repeat {rep} slot {slot}] " + prompt}], cap)
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=2) as pool:
            rows = list(pool.map(req, range(2)))
        wall = time.perf_counter() - t0
        for s, r in enumerate(rows):
            save(f"c2-{rep}-{s}", r)
        w = dict(repeat=rep, wall_s=wall,
                 aggregate_output_tps=sum(r["usage"]["completion_tokens"] for r in rows) / wall,
                 client_decode_tps=[r["client_decode_tps"] for r in rows],
                 completion_tokens=[r["usage"]["completion_tokens"] for r in rows],
                 cache_n=[(r["timings"] or {}).get("cache_n") for r in rows],
                 hashes=[r["output_sha256"] for r in rows])
        windows.append(w)
        print("C2", json.dumps(w), flush=True)
    timed = [w for w in windows if w["repeat"] != 99]
    summary = dict(ids=ids,
                   c1_client_decode_tps=[r["client_decode_tps"] for r in c1],
                   c1_median=statistics.median(r["client_decode_tps"] for r in c1),
                   c1_ttft_s=[r["ttft_s"] for r in c1],
                   c2_median_aggregate_output_tps=statistics.median(w["aggregate_output_tps"] for w in timed),
                   c2_per_stream=[w["client_decode_tps"] for w in timed], windows=windows)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("SUMMARY", json.dumps({k: v for k, v in summary.items() if k != "windows"}), flush=True)


if __name__ == "__main__":
    main()
