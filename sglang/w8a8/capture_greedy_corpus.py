#!/usr/bin/env python3
"""Capture and compare a deterministic greedy serving corpus.

This is an identity/coherence gate, not a quality evaluation. Run it once on
the target-only endpoint, then pass that JSON with --reference for an MTP or
kernel candidate. Any changed completion fails before performance testing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path
from typing import Any


PROMPTS = (
    "Explain Rayleigh scattering in three concise sentences.",
    "A shop has 17 boxes with 13 bolts each. How many bolts are there? Answer with only the integer.",
    "Continue the sequence 2, 6, 12, 20, 30 with the next two values and briefly state the rule.",
    "Name the largest planet in the Solar System and give one distinguishing physical property.",
    "Write a Python function that returns True exactly when a string is a palindrome. Keep it under eight lines.",
    "A train travels 180 km in 2.5 hours. Compute its average speed in km/h and show one equation.",
    "In one paragraph, distinguish a mutex from a semaphore in concurrent programming.",
    "If all ravens are birds and no birds are mammals, can any raven be a mammal? Explain in one sentence.",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--max-tokens", type=int, default=96)
    return parser.parse_args()


def request_json(url: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["content-type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(request, timeout=300) as response:
        return json.load(response)


def complete(base: str, model: str, prompt: str, max_tokens: int) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "seed": 20260828,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    response = request_json(base + "/chat/completions", payload)
    choice = response["choices"][0]
    text = choice["message"]["content"]
    return {
        "text": text,
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "finish_reason": choice.get("finish_reason"),
        "usage": response.get("usage"),
    }


def validate_reference_contract(
    reference: dict[str, Any], model: str, max_tokens: int
) -> dict[str, bool]:
    expected_prompt_hashes = [
        hashlib.sha256(prompt.encode("utf-8")).hexdigest() for prompt in PROMPTS
    ]
    reference_prompt_hashes = [
        item.get("prompt_sha256") for item in reference.get("samples", [])
    ]
    contract = {
        "model": reference.get("model") == model,
        "max_tokens": reference.get("max_tokens") == max_tokens,
        "repeat_exact": reference.get("repeat_exact") is True,
        "sample_count": len(reference.get("samples", [])) == len(PROMPTS),
        "prompt_hashes": reference_prompt_hashes == expected_prompt_hashes,
    }
    if not all(contract.values()):
        raise RuntimeError(
            "reference corpus contract mismatch: " + json.dumps(contract, sort_keys=True)
        )
    return contract


def main() -> None:
    args = parse_args()
    base = args.base.rstrip("/") + "/v1"
    ids = [item["id"] for item in request_json(base + "/models")["data"]]
    if ids != [args.model]:
        raise RuntimeError(f"served model identity mismatch: {ids}")

    reference = None
    reference_contract = None
    if args.reference is not None:
        reference = json.loads(args.reference.read_text(encoding="ascii"))
        reference_contract = validate_reference_contract(
            reference, args.model, args.max_tokens
        )

    samples = []
    for index, prompt in enumerate(PROMPTS):
        first = complete(base, args.model, prompt, args.max_tokens)
        second = complete(base, args.model, prompt, args.max_tokens)
        if first["text"] != second["text"]:
            raise RuntimeError(
                f"prompt {index} is not repeat-exact: "
                f"{first['text_sha256']} != {second['text_sha256']}"
            )
        samples.append(
            {
                "index": index,
                "prompt": prompt,
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                **first,
            }
        )

    result = {
        "model": args.model,
        "max_tokens": args.max_tokens,
        "repeat_exact": True,
        "samples": samples,
    }
    if reference is not None:
        reference_hashes = [item["text_sha256"] for item in reference["samples"]]
        candidate_hashes = [item["text_sha256"] for item in samples]
        result["reference"] = str(args.reference)
        result["reference_contract"] = reference_contract
        result["target_exact"] = candidate_hashes == reference_hashes
        result["mismatched_indices"] = [
            index
            for index, (expected, actual) in enumerate(
                zip(reference_hashes, candidate_hashes, strict=True)
            )
            if expected != actual
        ]
        if not result["target_exact"]:
            args.output.write_text(
                json.dumps(result, indent=2, sort_keys=True) + "\n",
                encoding="ascii",
            )
            raise RuntimeError(
                "candidate is not target-exact: "
                f"mismatched prompts {result['mismatched_indices']}"
            )

    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    print(
        "GREEDY_CORPUS_OK model={} samples={} target_exact={}".format(
            args.model,
            len(samples),
            result.get("target_exact", "reference-not-supplied"),
        )
    )


if __name__ == "__main__":
    main()
