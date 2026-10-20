"""Teacher-forced perplexity through a live vLLM /v1/completions.

Uses the 4x512 wikitext windows at
/mnt/cache/b70/devmodel/quant/_work/qwen36/ppl_s512_n4.pt
(the same slice as quant/run_quant.py --mode eval). The first prompt
position has no logprob. PPL is exp(mean NLL) over the rest, which is
the shift-by-one cross entropy that run_quant reports. int3 on the
unpack path was 9.19.
"""

from __future__ import annotations

import argparse
import json
import math
import urllib.error
import urllib.request
from pathlib import Path


DEFAULT_PT = (
    "/mnt/cache/b70/devmodel/quant/_work/qwen36/ppl_s512_n4.pt"
)


def load_windows(path: Path) -> list[list[int]]:
    import torch

    ids = torch.load(path, map_location="cpu", weights_only=False)
    if ids.ndim != 2:
        raise SystemExit(f"expected [windows, seq], got {tuple(ids.shape)}")
    return ids.to(torch.int64).tolist()


def _chosen_logprob(entry: dict, token_id: int) -> float:
    # JSON object keys are strings. vLLM's Logprob is
    # {logprob, rank, decoded_token}.
    for key in (str(token_id), token_id):
        raw = entry.get(key)
        if isinstance(raw, dict) and "logprob" in raw:
            return float(raw["logprob"])
        if isinstance(raw, (int, float)):
            return float(raw)
    # Fall back to the rank-1 row when the chosen id was the only one
    # returned under a different key type.
    best = None
    for raw in entry.values():
        if isinstance(raw, dict) and "logprob" in raw:
            rank = raw.get("rank")
            if rank == 1 or best is None:
                best = float(raw["logprob"])
                if rank == 1:
                    break
    if best is None:
        raise KeyError("no logprob in prompt position")
    return best


def nll_for_window(base_url: str, model: str, tokens: list[int], timeout: float) -> tuple[float, int]:
    body = {
        "model": model,
        "prompt": tokens,
        "max_tokens": 1,
        "temperature": 0,
        "prompt_logprobs": 1,
        "stream": False,
    }
    req = urllib.request.Request(
        base_url.rstrip("/") + "/v1/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.load(resp)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:800]
        raise SystemExit(f"HTTP {exc.code}: {detail}") from exc
    choice = payload["choices"][0]
    rows = choice.get("prompt_logprobs")
    if not rows:
        raise SystemExit(f"no prompt_logprobs in {list(choice)}")
    # Position 0 is null. Positions 1..L-1 score tokens[1:].
    nll = 0.0
    counted = 0
    for pos in range(1, len(tokens)):
        entry = rows[pos] if pos < len(rows) else None
        if not isinstance(entry, dict):
            raise SystemExit(f"missing prompt logprob at {pos}")
        lp = _chosen_logprob(entry, tokens[pos])
        nll += -lp
        counted += 1
    return nll, counted


def main() -> None:
    parser = argparse.ArgumentParser(description="Served-model wikitext perplexity")
    parser.add_argument("--base-url", default="http://127.0.0.1:18084")
    parser.add_argument("--model", default="hotschmoe-dev")
    parser.add_argument("--windows", default=DEFAULT_PT)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    windows = load_windows(Path(args.windows))
    nll_sum = 0.0
    tok_sum = 0
    per = []
    for i, tokens in enumerate(windows):
        nll, counted = nll_for_window(args.base_url, args.model, tokens, args.timeout)
        nll_sum += nll
        tok_sum += counted
        per.append({"window": i, "tokens": counted, "nll": nll / counted})
        print(
            f"window {i} tokens={counted} nll={nll / counted:.4f}",
            flush=True,
        )
    mean = nll_sum / tok_sum
    ppl = math.exp(mean)
    report = {
        "model": args.model,
        "base_url": args.base_url,
        "windows": args.windows,
        "n_windows": len(windows),
        "seqlen": len(windows[0]) if windows else 0,
        "tokens": tok_sum,
        "nll": mean,
        "ppl": ppl,
        "per_window": per,
        "reference_int3_unpack_ppl": 9.19,
    }
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    print(f"ppl {ppl:.4f} nll={mean:.4f} tokens={tok_sum}", flush=True)


if __name__ == "__main__":
    main()
