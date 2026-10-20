"""Record exact greedy token IDs for the matched MTP on/off gate."""

from __future__ import annotations

import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import Request, urlopen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:18121")
    parser.add_argument("--out", required=True)
    parser.add_argument("--baseline")
    parser.add_argument("--respect-eos", action="store_true")
    args = parser.parse_args()
    with urlopen(args.url + "/v1/models") as response:
        identity = json.load(response)
    assert identity["data"][0]["id"] == "hotschmoe-dd"
    registry = Path(__file__).resolve().parents[3] / "evals/configs/models.yaml"
    registered = set(
        re.findall(r"^\s+served_model_id: (.+)$", registry.read_text(), re.MULTILINE)
    )
    assert any(row["id"] in registered for row in identity["data"][1:])
    prompts = [
        "The capital of France is",
        'def fibonacci(n):\n    """Return the first n Fibonacci numbers."""\n',
        "Count the integers in order, separated by commas: 1, 2, 3,",
        "Explain how a hash table resolves collisions in three sentences.",
    ]

    def complete(prompt):
        body = dict(
            model="hotschmoe-dd",
            prompt=prompt,
            temperature=0,
            max_tokens=96,
            seed=0,
            ignore_eos=not args.respect_eos,
            return_token_ids=True,
        )
        req = Request(
            args.url + "/v1/completions",
            json.dumps(body).encode("ascii"),
            headers={"Content-Type": "application/json"},
        )
        with urlopen(req, timeout=180) as response:
            completion = json.load(response)
        choice = completion["choices"][0]
        tokens = choice.get("token_ids")
        if not tokens or (not args.respect_eos and len(tokens) != 96):
            raise ValueError(f"Missing exact token IDs: {choice.keys()}")
        return dict(prompt=prompt, token_ids=tokens, text=choice["text"])

    rows = [complete(prompt) for prompt in prompts]
    result = dict(identity=identity, rows=rows)
    if args.respect_eos:
        with ThreadPoolExecutor(max_workers=2) as pool:
            result["concurrent_rows"] = list(pool.map(complete, prompts[:2]))
    if args.baseline:
        recorded = json.loads(Path(args.baseline).read_text())
        base = recorded["rows"]
        result["identical"] = all(
            a["token_ids"] == b["token_ids"] for a, b in zip(base, rows, strict=True)
        )
        result["first_mismatch"] = [
            next(
                (
                    i
                    for i, (x, y) in enumerate(zip(a["token_ids"], b["token_ids"]))
                    if x != y
                ),
                None,
            )
            for a, b in zip(base, rows, strict=True)
        ]
        if args.respect_eos:
            result["concurrent_identical"] = all(
                a["token_ids"] == b["token_ids"]
                for a, b in zip(
                    recorded["concurrent_rows"], result["concurrent_rows"], strict=True
                )
            )
            result["identical"] &= result["concurrent_identical"]
    Path(args.out).write_text(json.dumps(result, indent=2, ensure_ascii=True) + "\n")
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in ("rows", "concurrent_rows")},
            ensure_ascii=True,
        )
    )
    return 0 if result.get("identical", True) else 3


if __name__ == "__main__":
    raise SystemExit(main())
