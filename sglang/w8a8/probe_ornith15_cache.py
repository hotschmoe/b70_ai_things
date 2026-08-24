#!/usr/bin/env python3
"""Measure cold versus repeated-prefix TTFT on the live Ornith endpoint."""

import argparse
import json
import time
import urllib.request


def post(url: str, payload: dict) -> tuple[float, dict]:
    body = json.dumps(payload).encode()
    request = urllib.request.Request(
        url,
        data=body,
        headers={"content-type": "application/json"},
        method="POST",
    )
    start = time.perf_counter()
    with urllib.request.urlopen(request, timeout=300) as response:
        result = json.load(response)
    return time.perf_counter() - start, result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://localhost:18080/v1")
    parser.add_argument("--model", default="ornith-1.5-35b-a3b-W8A8-rtn-mtp-shisa")
    parser.add_argument("--repetitions", type=int, default=1024)
    args = parser.parse_args()

    unique_prefix = "B70 cache qualification 20260824. " + "alpha beta gamma delta " * args.repetitions
    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": unique_prefix + "\nReply with only OK."}],
        "max_tokens": 1,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    cold, cold_result = post(args.endpoint + "/chat/completions", payload)
    warm, warm_result = post(args.endpoint + "/chat/completions", payload)
    cold_usage = cold_result.get("usage") or {}
    warm_usage = warm_result.get("usage") or {}
    summary = {
        "cold_seconds": cold,
        "warm_seconds": warm,
        "warm_over_cold": warm / cold if cold else None,
        "prompt_tokens": cold_usage.get("prompt_tokens"),
        "cold_completion_tokens": cold_usage.get("completion_tokens"),
        "warm_completion_tokens": warm_usage.get("completion_tokens"),
        "byte_identical": cold_result["choices"][0]["message"] == warm_result["choices"][0]["message"],
    }
    print(json.dumps(summary, indent=2))
    if not summary["byte_identical"] or warm >= cold * 0.8:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
