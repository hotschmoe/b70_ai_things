#!/usr/bin/env python3
"""Authenticated mixed-load coherence soak for the OBLITERATED DP=2 endpoint."""

import argparse
import collections
import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request


def is_degenerate(text):
    compact = "".join(text.split())
    if not compact:
        return "empty"
    if len(compact) < 12:
        return None
    run = 1
    for index in range(1, len(compact)):
        if compact[index] == compact[index - 1]:
            run += 1
            if run >= 8:
                return f"run({compact[index]})"
        else:
            run = 1
    char, count = collections.Counter(compact).most_common(1)[0]
    if count / len(compact) > 0.55:
        return f"dominant({char})"
    return None


def integers(text):
    return [int(value) for value in re.findall(r"-?\d+", text)]


def build_case(case_index, worker, request_index):
    choice = case_index % 6
    if choice == 0:
        return (
            "paris",
            "What is the capital of France? Answer in one short sentence.",
            48,
            lambda text: "paris" in text.lower(),
        )
    if choice == 1:
        return (
            "multiply",
            "What is 17*23? Answer with just the integer.",
            32,
            lambda text: integers(text) == [391],
        )
    if choice == 2:
        return (
            "fibonacci",
            (
                "List the first 12 Fibonacci numbers starting with 0, 1. "
                "Use comma-separated integers only."
            ),
            96,
            lambda text: integers(text)
            == [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89],
        )
    if choice == 3:
        return (
            "hashmap",
            (
                "In one concise paragraph, explain how a hash map handles keys and "
                "collisions and state its average lookup complexity."
            ),
            160,
            lambda text: (
                "hash" in text.lower()
                and "collision" in text.lower()
                and ("o(1)" in text.lower() or "constant" in text.lower())
            ),
        )
    if choice == 4:
        marker = f"B70-CHECK-{worker}-{request_index}"
        document = "The quick brown fox jumps over the lazy dog. " * 180
        return (
            "long_exact",
            (
                f"Read this document: {document}\nIgnore its content and reply with "
                f"exactly {marker} and nothing else."
            ),
            48,
            lambda text: text.strip() == marker,
        )
    return (
        "squares",
        (
            "Give the squares of the integers 1 through 12 as comma-separated "
            "integers only."
        ),
        96,
        lambda text: integers(text)
        == [1, 4, 9, 16, 25, 36, 49, 64, 81, 100, 121, 144],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:18080")
    parser.add_argument("--model", default="hotschmoe-dd")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--duration", type=int, default=300)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY is required", file=sys.stderr)
        return 2

    deadline = time.monotonic() + args.duration
    started = time.monotonic()
    lock = threading.Lock()
    stats = {
        "requests": 0,
        "ok": 0,
        "coherence_failures": 0,
        "degenerate": 0,
        "errors": 0,
        "completion_tokens": 0,
        "upstreams": {},
        "case_counts": {},
        "samples": [],
    }

    def record_failure(kind, case_name, detail, text=""):
        stats[kind] += 1
        if len(stats["samples"]) < 12:
            stats["samples"].append(
                {
                    "kind": kind,
                    "case": case_name,
                    "detail": detail,
                    "text": text[:240],
                }
            )

    def worker(worker_id):
        request_index = 0
        while time.monotonic() < deadline:
            case_name, prompt, max_tokens, validator = build_case(
                request_index + worker_id, worker_id, request_index
            )
            payload = {
                "model": args.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": 0,
                "seed": 1,
                "chat_template_kwargs": {"enable_thinking": False},
            }
            request = urllib.request.Request(
                args.base.rstrip("/") + "/v1/chat/completions",
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=args.timeout) as response:
                    body = json.loads(response.read().decode("utf-8"))
                    upstream = response.headers.get("X-B70-Upstream") or "direct"
                text = (
                    (((body.get("choices") or [{}])[0].get("message") or {}).get("content"))
                    or ""
                )
                completion_tokens = int(
                    (body.get("usage") or {}).get("completion_tokens") or 0
                )
                degenerate = is_degenerate(text)
                coherent = validator(text)
                with lock:
                    stats["requests"] += 1
                    stats["completion_tokens"] += completion_tokens
                    stats["upstreams"][upstream] = stats["upstreams"].get(upstream, 0) + 1
                    stats["case_counts"][case_name] = (
                        stats["case_counts"].get(case_name, 0) + 1
                    )
                    if degenerate:
                        record_failure("degenerate", case_name, degenerate, text)
                    elif not coherent:
                        record_failure(
                            "coherence_failures", case_name, "answer check failed", text
                        )
                    else:
                        stats["ok"] += 1
            except (OSError, ValueError, KeyError, urllib.error.HTTPError) as exc:
                with lock:
                    stats["requests"] += 1
                    record_failure(
                        "errors", case_name, f"{type(exc).__name__}: {exc}"
                    )
            request_index += 1

    threads = [
        threading.Thread(target=worker, args=(worker_id,), daemon=True)
        for worker_id in range(args.concurrency)
    ]
    for thread in threads:
        thread.start()

    while time.monotonic() < deadline:
        remaining = deadline - time.monotonic()
        time.sleep(min(30, max(0, remaining)))
        with lock:
            elapsed = time.monotonic() - started
            print(
                f"[soak {elapsed:5.0f}s] requests={stats['requests']} ok={stats['ok']} "
                f"coherence_failures={stats['coherence_failures']} "
                f"degenerate={stats['degenerate']} errors={stats['errors']} "
                f"upstreams={stats['upstreams']}",
                flush=True,
            )
    for thread in threads:
        thread.join(timeout=args.timeout + 5)

    elapsed = time.monotonic() - started
    bad = stats["coherence_failures"] + stats["degenerate"] + stats["errors"]
    result = {
        "base": args.base,
        "model": args.model,
        "concurrency": args.concurrency,
        "requested_duration_s": args.duration,
        "elapsed_s": elapsed,
        "passed": bad == 0,
        **stats,
    }
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    verdict = "PASS" if bad == 0 else "FAIL"
    print(
        f"[{verdict}] requests={stats['requests']} ok={stats['ok']} bad={bad} "
        f"completion_tokens={stats['completion_tokens']} wrote={args.out}"
    )
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
