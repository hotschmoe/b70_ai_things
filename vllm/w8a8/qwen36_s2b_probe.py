#!/usr/bin/env python3
"""Small semantic/repetition gate for the Qwen3.6 S2B control."""

from __future__ import annotations

import argparse
import itertools
import json
import re
import urllib.request


def complete(base_url: str, model: str, prompt: str, max_tokens: int) -> str:
    payload = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0,
        }
    ).encode("ascii")
    request = urllib.request.Request(
        base_url.rstrip("/") + "/v1/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        body = json.loads(response.read().decode("utf-8"))
    return body["choices"][0]["text"]


def repeated_word(text: str) -> tuple[str, int] | None:
    words = re.findall(r"[A-Za-z]{3,}", text.lower())
    longest = max(
        ((word, sum(1 for _ in group)) for word, group in itertools.groupby(words)),
        key=lambda item: item[1],
        default=("", 0),
    )
    return longest if longest[1] >= 4 else None


def printable_ascii_fraction(text: str) -> float:
    if not text:
        return 0.0
    printable = sum(character == "\n" or 32 <= ord(character) <= 126 for character in text)
    return printable / len(text)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()

    cases = (
        ("The capital of France is", "paris"),
        ("Compute 17 times 19. The answer is", "323"),
    )
    failures: list[str] = []
    records: list[dict[str, object]] = []
    for prompt, expected in cases:
        text = complete(args.base_url, args.model, prompt, 48)
        fraction = printable_ascii_fraction(text)
        repeated = repeated_word(text)
        record = {
            "prompt": prompt,
            "expected": expected,
            "text": text,
            "printable_ascii_fraction": fraction,
            "repeated_word": repeated,
        }
        records.append(record)
        if expected not in text.lower():
            failures.append(f"missing expected answer {expected!r}")
        if fraction < 0.98:
            failures.append(f"printable ASCII fraction {fraction:.3f} below 0.98")
        if repeated is not None:
            failures.append(f"repeated word {repeated[0]!r} count={repeated[1]}")

    print(json.dumps({"records": records, "failures": failures}, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
