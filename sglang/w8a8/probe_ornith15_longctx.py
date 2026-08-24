#!/usr/bin/env python3
"""Near-native-context retrieval and repeated-prefix cache gate for Ornith."""

import argparse
import json
import time
import urllib.request


def request_json(url: str, payload: dict, timeout: int) -> tuple[float, dict]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"content-type": "application/json"},
        method="POST",
    )
    start = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        result = json.load(response)
    return time.perf_counter() - start, result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://localhost:18080")
    parser.add_argument("--model", default="ornith-1.5-35b-a3b-W8A8-rtn-mtp-shisa")
    parser.add_argument("--filler-tokens", type=int, default=250000)
    parser.add_argument("--timeout", type=int, default=1200)
    args = parser.parse_args()

    needle = "B70-ORANGE-4917"
    content = (
        f"Memorize this secret exactly: {needle}.\n"
        + " alpha" * args.filler_tokens
        + "\nWhat was the secret? Reply with the secret only."
    )
    messages = [{"role": "user", "content": content}]
    _, tokenized = request_json(
        args.endpoint + "/tokenize",
        {"model": args.model, "messages": messages},
        args.timeout,
    )
    prompt_tokens = int(tokenized["count"])
    max_model_len = int(tokenized["max_model_len"])
    if prompt_tokens + 32 > max_model_len:
        raise SystemExit(f"prompt too long: {prompt_tokens} + 32 > {max_model_len}")

    payload = {
        "model": args.model,
        "messages": messages,
        "max_tokens": 32,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    cold, cold_result = request_json(args.endpoint + "/v1/chat/completions", payload, args.timeout)
    warm, warm_result = request_json(args.endpoint + "/v1/chat/completions", payload, args.timeout)
    cold_text = cold_result["choices"][0]["message"].get("content") or ""
    warm_text = warm_result["choices"][0]["message"].get("content") or ""
    summary = {
        "configured_context": max_model_len,
        "prompt_tokens": prompt_tokens,
        "cold_seconds": cold,
        "warm_seconds": warm,
        "warm_over_cold": warm / cold if cold else None,
        "needle_cold": needle in cold_text,
        "needle_warm": needle in warm_text,
        "byte_identical": cold_text == warm_text,
        "cold_text": cold_text,
    }
    print(json.dumps(summary, indent=2))
    if not all((summary["needle_cold"], summary["needle_warm"], summary["byte_identical"])):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
