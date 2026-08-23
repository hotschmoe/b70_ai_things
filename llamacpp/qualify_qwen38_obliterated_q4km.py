#!/usr/bin/env python3
"""Deterministic coherence and MTP equivalence gate for hotschmoe-dd."""

import argparse
import hashlib
import json
import os
import re
import sys
import urllib.error
import urllib.request


TESTS = [
    {
        "id": "paris",
        "prompt": "What is the capital of France? Answer in one short sentence.",
        "max_tokens": 32,
    },
    {
        "id": "multiply",
        "prompt": "What is 17*23? Answer with just the integer.",
        "max_tokens": 32,
    },
    {
        "id": "modular",
        "prompt": "Compute (12345*6789) mod 97. Answer with just the integer.",
        "max_tokens": 32,
    },
    {
        "id": "fibonacci",
        "prompt": (
            "List the first 12 Fibonacci numbers starting with 0, 1. "
            "Use comma-separated integers only."
        ),
        "max_tokens": 96,
    },
    {
        "id": "sort_unique",
        "prompt": (
            "Sort these integers and remove duplicates: 17, 3, 17, 9, -2, 3, 12. "
            "Answer with comma-separated integers only."
        ),
        "max_tokens": 64,
    },
    {
        "id": "logic",
        "prompt": (
            "All zorps are blue. No blue thing is green. Can any zorp be green? "
            "Answer Yes or No, then give one short sentence of explanation."
        ),
        "max_tokens": 64,
    },
    {
        "id": "squares_24",
        "prompt": (
            "Write exactly 24 lines. For each integer N from 1 through 24, line N "
            "must be formatted as 'N: N*N' with N*N replaced by its square. Do not "
            "add a title, prose, bullets, or code fences."
        ),
        "max_tokens": 384,
    },
]


def integers(text):
    return [int(value) for value in re.findall(r"-?\d+", text)]


def validate(test_id, text):
    clean = text.strip()
    if test_id == "paris":
        return "paris" in clean.lower(), "contains Paris"
    if test_id == "multiply":
        return integers(clean) == [391], "exact integer 391"
    if test_id == "modular":
        return integers(clean) == [71], "exact integer 71"
    if test_id == "fibonacci":
        expected = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89]
        return integers(clean) == expected, "exact first 12 Fibonacci numbers"
    if test_id == "sort_unique":
        return integers(clean) == [-2, 3, 9, 12, 17], "exact sorted unique list"
    if test_id == "logic":
        return clean.lower().startswith("no"), "starts with No"
    if test_id == "squares_24":
        lines = [line.strip() for line in clean.splitlines() if line.strip()]
        expected = [f"{n}: {n*n}" for n in range(1, 25)]
        return lines == expected, "exact 24-line square table"
    return False, "unknown test"


def request_completion(base, model, api_key, test):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": test["prompt"]}],
        "max_tokens": test["max_tokens"],
        "temperature": 0,
        "seed": 1,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    request = urllib.request.Request(
        base.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        body = json.loads(response.read().decode("utf-8"))
        upstream = response.headers.get("X-B70-Upstream")
    choice = (body.get("choices") or [{}])[0]
    text = ((choice.get("message") or {}).get("content") or "")
    usage = body.get("usage") or {}
    return text, choice.get("finish_reason"), usage, upstream


def load_reference(path):
    if not path:
        return {}
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    return {item["id"]: item["text"].strip() for item in data.get("results", [])}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--model", default="hotschmoe-dd")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--reference")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY is required", file=sys.stderr)
        return 2
    reference = load_reference(args.reference)
    results = []
    all_ok = True

    for test in TESTS:
        error = None
        text = ""
        finish_reason = None
        usage = {}
        upstream = None
        try:
            text, finish_reason, usage, upstream = request_completion(
                args.base, args.model, api_key, test
            )
            coherent, expectation = validate(test["id"], text)
        except (OSError, ValueError, KeyError, urllib.error.HTTPError) as exc:
            coherent = False
            expectation = "request succeeds"
            error = f"{type(exc).__name__}: {exc}"

        exact_reference = None
        if test["id"] in reference:
            exact_reference = text.strip() == reference[test["id"]]
        if reference:
            passed = error is None and exact_reference is True
        else:
            passed = coherent and error is None
        all_ok = all_ok and passed
        digest = hashlib.sha256(text.strip().encode("utf-8")).hexdigest()
        result = {
            "id": test["id"],
            "passed": passed,
            "coherent": coherent,
            "expectation": expectation,
            "exact_reference": exact_reference,
            "sha256": digest,
            "finish_reason": finish_reason,
            "usage": usage,
            "upstream": upstream,
            "error": error,
            "text": text,
        }
        results.append(result)
        status = "PASS" if passed else "FAIL"
        comparison = ""
        if exact_reference is not None:
            comparison = f" reference_exact={exact_reference}"
        preview = " ".join(text.split())[:120]
        print(
            f"[{status}] {test['id']} sha256={digest[:16]}{comparison} "
            f"upstream={upstream or '-'} text={preview}"
        )

    output = {
        "tag": args.tag,
        "base": args.base,
        "model": args.model,
        "reference": args.reference,
        "passed": all_ok,
        "results": results,
    }
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"[{'PASS' if all_ok else 'FAIL'}] wrote {args.out}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
