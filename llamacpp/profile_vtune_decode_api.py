#!/usr/bin/env python3
"""Warm or measure one fixed deterministic llama.cpp decode request."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from profile_qwen38_api import DECODE_PROMPT, stream_chat


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="http://127.0.0.1:18080")
    parser.add_argument("--model", default="hotschmoe-dd")
    parser.add_argument("--mode", choices=("warmup", "measure"), required=True)
    parser.add_argument("--tokens", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=1200)
    args = parser.parse_args()

    expected = 32 if args.mode == "warmup" else 512
    if args.tokens != expected:
        parser.error(f"{args.mode} requires exactly {expected} tokens")
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("API_KEY", "")
    try:
        result = stream_chat(
            args.base,
            args.model,
            api_key,
            DECODE_PROMPT,
            args.tokens,
            True,
            args.timeout,
        )
        passed = (
            result.get("completion_tokens") == expected
            and result.get("finish_reason") == "length"
            and result.get("post_first_tok_s") is not None
        )
        payload = {
            "passed": passed,
            "mode": args.mode,
            "expected_completion_tokens": expected,
            "methodology": {
                "prompt": "profile_qwen38_api.DECODE_PROMPT",
                "temperature": 0,
                "top_p": 1,
                "seed": 1234,
                "ignore_eos": True,
            },
            "result": result,
        }
    except Exception as exc:  # noqa: BLE001 - preserve the exact mechanism failure
        payload = {
            "passed": False,
            "mode": args.mode,
            "expected_completion_tokens": expected,
            "error": f"{type(exc).__name__}: {exc}",
        }

    args.out.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="ascii",
    )
    print(
        f"VTUNE_API mode={args.mode} tokens={args.tokens} "
        f"passed={payload['passed']} out={args.out}"
    )
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
