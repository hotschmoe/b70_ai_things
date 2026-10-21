#!/usr/bin/env python3
"""One HTTP workload for all three retained B70 recipes; no GPU access."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import statistics
import threading
import time
import urllib.request


def save(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=True) + "\n", encoding="ascii")


def get(base, path):
    with urllib.request.urlopen(base + path, timeout=20) as response:
        return json.load(response)


def request(base, payload, barrier=None):
    if barrier:
        barrier.wait(timeout=30)
    start = time.perf_counter()
    req = urllib.request.Request(base + "/v1/chat/completions",
                                 json.dumps(payload).encode(), {"Content-Type": "application/json"})
    events, text, usage, timings, finish = [], [], None, None, None
    with urllib.request.urlopen(req, timeout=600) as response:
        for raw in response:
            if not raw.startswith(b"data:"):
                continue
            data = raw[5:].strip()
            if data == b"[DONE]":
                break
            row = json.loads(data)
            if row.get("error"):
                raise RuntimeError(row["error"])
            usage = row.get("usage") or usage
            timings = row.get("timings") or timings
            for choice in row.get("choices", []):
                finish = choice.get("finish_reason") or finish
                delta = choice.get("delta", {})
                piece = delta.get("content") or delta.get("reasoning_content") or delta.get("reasoning")
                if piece:
                    text.append(piece)
                    events.append({"t": time.perf_counter(), "text": piece})
    end = time.perf_counter()
    if not usage or not events:
        raise RuntimeError("Missing server usage or content arrivals")
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
    if timings:
        cached = max(cached or 0, timings.get("cache_n", 0))
    return dict(start=start, end=end, first=events[0]["t"], last=events[-1]["t"],
                events=events, text="".join(text), usage=usage, timings=timings,
                cached_tokens=cached or 0, finish_reason=finish,
                ttft_s=events[0]["t"] - start,
                decode_tps=(usage["completion_tokens"] - 1) / (end - events[0]["t"]))


def body(prompt, cap):
    return dict(model="hotschmoe-dd", messages=[{"role": "user", "content": prompt}],
                temperature=0, max_tokens=cap, stream=True,
                stream_options={"include_usage": True},
                chat_template_kwargs={"enable_thinking": False})


def run(base, fixture, out, alias):
    out.mkdir(parents=True, exist_ok=True)
    models = get(base, "/v1/models")
    save(out / "models.json", models)
    ids = [x["id"] for x in models["data"]]
    assert ids[0] == "hotschmoe-dd" and alias in ids, ids
    assert alias in Path('/mnt/vm_8tb/github/b70_ai_things/evals/configs/models.yaml').read_text()
    checks = []
    for prompt, expected in [("Return only the integer: 17 * 19.", "323"),
                             ("Return only the integer: 23 * 29.", "667")]:
        row = request(base, body(prompt, 16))
        row["expected"] = expected
        row["pass"] = row["text"].strip() == expected
        checks.append(row)
    save(out / "coherence.json", checks)
    assert all(x["pass"] for x in checks), "Arithmetic coherence failed"
    rounds = []
    for batch in fixture["batches"]:
        barrier = threading.Barrier(4)
        with ThreadPoolExecutor(4) as workers:
            futures = [workers.submit(request, base, body(p, 512), barrier) for p in batch["prompts"]]
            rows = [f.result() for f in futures]
        start, first, lastfirst, end = (min(r["start"] for r in rows), min(r["first"] for r in rows),
                                       max(r["first"] for r in rows), max(r["end"] for r in rows))
        total_in = sum(r["usage"]["prompt_tokens"] for r in rows)
        uncached = sum(r["usage"]["prompt_tokens"] - r["cached_tokens"] for r in rows)
        total_out = sum(r["usage"]["completion_tokens"] for r in rows)
        counts_ok = all(r["usage"]["completion_tokens"] == 512 for r in rows)
        cold_ok = all(r["cached_tokens"] / r["usage"]["prompt_tokens"] < .01 for r in rows)
        expected_cached = batch.get("expected_cached", False)
        cache_ok = (all(r["cached_tokens"] / r["usage"]["prompt_tokens"] > .99 for r in rows)
                    if expected_cached else cold_ok)
        result = dict(batch=batch["name"], warmup=batch["warmup"], streams=rows,
                      total_prompt_tokens=total_in, uncached_prompt_tokens=uncached,
                      total_output_tokens=total_out,
                      pp_aggregate_effective_tps=uncached / (lastfirst - start),
                      tg_aggregate_decode_window_tps=(total_out - 4) / (end - first),
                      output_aggregate_wall_tps=total_out / (end - start),
                      wall_s=end - start, time_to_all_first_tokens_s=lastfirst - start,
                      per_stream_decode_tps=[r["decode_tps"] for r in rows],
                      ttft_s=[r["ttft_s"] for r in rows],
                      all_512_tokens=counts_ok, cold_prefix=cold_ok,
                      expected_cached=expected_cached, cache_condition_met=cache_ok,
                      texts_distinct=len({r["text"] for r in rows}),
                      garbage_heuristic_pass=all(len(r["text"]) > 200 and
                          max(r["text"].count(c) for c in set(r["text"])) / len(r["text"]) < .55 for r in rows))
        rounds.append(result)
        save(out / (batch["name"] + ".json"), result)
        print(json.dumps({k: v for k, v in result.items() if k != "streams"}), flush=True)
        # Do not spend another long prefill after malformed/short/cached output.
        if not counts_ok or not cache_ok or not result["garbage_heuristic_pass"]:
            raise RuntimeError("Invalid workload; raw round saved")
    measured = [r for r in rounds if not r["warmup"]]
    summary = {"alias": alias, "fixture_sha256": hashlib.sha256(json.dumps(fixture, sort_keys=True).encode()).hexdigest(),
               "rounds": len(measured), "definitions": {
                   "PP": "uncached input tokens / batch start to all four first content arrivals; includes scheduling and overlapped decode",
                   "TG": "sum(output tokens - 1) / earliest first content to final response end; includes any remaining prefill",
                   "wall": "all output tokens / first request start to final response end"}}
    for key in ["pp_aggregate_effective_tps", "tg_aggregate_decode_window_tps", "output_aggregate_wall_tps", "wall_s"]:
        summary[key] = statistics.median(r[key] for r in measured)
        summary[key + "_runs"] = [r[key] for r in measured]
    summary["per_stream_decode_tps_median"] = statistics.median(v for r in measured for v in r["per_stream_decode_tps"])
    save(out / "summary.json", summary)
    print("SUMMARY " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--alias", required=True)
    args = parser.parse_args()
    run(args.base, json.loads(args.fixture.read_text()), args.out, args.alias)
