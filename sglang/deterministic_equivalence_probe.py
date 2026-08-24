#!/usr/bin/env python3
"""Save canonical greedy responses for cross-arm byte equivalence."""

import argparse
import json
import urllib.request


PROMPTS = [
    "Return only the capital of Senegal.",
    "Compute 37 * 43. Return only the integer.",
    "Write a Python function named clamp(x, lo, hi). Return only code.",
    "Explain in two sentences why the sky appears blue.",
    "Sort these words alphabetically and return one comma-separated line: pear, apple, fig, banana.",
    "A train travels 180 km in 2.5 hours. Give its average speed and one line of arithmetic.",
    "Rewrite without changing meaning: Reliable systems make failures visible and recoverable.",
    ("Read the repeated context and answer with the marker only. " + "alpha beta gamma " * 800 + " Marker: ORCHID"),
]


def post(url, payload, timeout):
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"content-type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()

    results = []
    for index, prompt in enumerate(PROMPTS):
        payload = {
            "model": args.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "top_p": 1,
            "seed": 1234,
            "max_tokens": args.max_tokens,
            "ignore_eos": True,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        response = post(
            args.base.rstrip("/") + "/v1/chat/completions", payload, args.timeout
        )
        choice = response["choices"][0]
        message = choice["message"]
        usage = response.get("usage", {})
        canonical = {
            "index": index,
            "reasoning_content": message.get("reasoning_content"),
            "content": message.get("content"),
            "finish_reason": choice.get("finish_reason"),
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
        }
        text = (canonical["reasoning_content"] or "") + (canonical["content"] or "")
        if not text.strip() or canonical["completion_tokens"] in (None, 0):
            raise SystemExit(f"degenerate deterministic response at prompt {index}")
        results.append(canonical)
        print(
            f"PROMPT -> index={index} completion_tokens={canonical['completion_tokens']} "
            f"finish={canonical['finish_reason']}"
        )

    with open(args.out, "w", encoding="ascii") as handle:
        json.dump(results, handle, ensure_ascii=True, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"OUTPUT -> {args.out}")


if __name__ == "__main__":
    main()
