#!/usr/bin/env python3
"""Fixed one-prompt decode probe for the XL quant timing campaign."""

from __future__ import annotations

import argparse
import json
import os
import statistics
from pathlib import Path

from profile_qwen38_api import DECODE_PROMPT, stream_chat


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="http://127.0.0.1:18080")
    parser.add_argument("--model", default="hotschmoe-dd")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reps", type=int, default=1)
    parser.add_argument("--gen-tokens", type=int, default=256)
    parser.add_argument("--warmup-tokens", type=int, default=32)
    parser.add_argument("--timeout", type=int, default=900)
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("API_KEY", "")
    warmup = stream_chat(
        args.base,
        args.model,
        api_key,
        DECODE_PROMPT,
        args.warmup_tokens,
        True,
        args.timeout,
    )
    runs = []
    errors = []
    for repetition in range(args.reps):
        try:
            result = stream_chat(
                args.base,
                args.model,
                api_key,
                DECODE_PROMPT,
                args.gen_tokens,
                True,
                args.timeout,
            )
            result["repetition"] = repetition
            runs.append(result)
            print(
                f"RUN rep={repetition} completion={result['completion_tokens']} "
                f"ttft_s={result['ttft_s']:.4f} "
                f"post_first_tok_s={result['post_first_tok_s']}"
            )
        except Exception as exc:  # noqa: BLE001 - persist exact campaign failure
            error = {"repetition": repetition, "error": f"{type(exc).__name__}: {exc}"}
            runs.append(error)
            errors.append(error)
            print(f"ERROR rep={repetition} error={error['error']}")

    speeds = [run["post_first_tok_s"] for run in runs if run.get("post_first_tok_s") is not None]
    output = {
        "tag": args.tag,
        "base": args.base,
        "model": args.model,
        "methodology": {
            "prompt": "profile_qwen38_api.DECODE_PROMPT",
            "temperature": 0,
            "top_p": 1,
            "seed": 1234,
            "ignore_eos": True,
            "gen_tokens": args.gen_tokens,
            "warmup_tokens": args.warmup_tokens,
        },
        "warmup": warmup,
        "runs": runs,
        "summary": {
            "n_ok": len(speeds),
            "median_post_first_tok_s": statistics.median(speeds) if speeds else None,
        },
        "passed": not errors and len(speeds) == args.reps,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(output, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="ascii",
    )
    return 0 if output["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
