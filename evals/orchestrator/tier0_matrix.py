#!/usr/bin/env python3
"""Build the Tier-0 divergence matrix from cached per-quant token dumps (one-card friendly).

Each `run_evals --tiers 0` writes a tier0_tokens.json (per-token argmax + actual logprob) and a
tier0_divergence.json (standalone ppl). This tool collects them across all runs, picks an anchor
(the highest-precision servable quant — bf16 > fp8 > w8a8 ...), and reports each quant's:
  - ppl (absolute, lower=better)
  - top1_agreement vs anchor (fraction of positions with the same argmax token)
  - nll_gap vs anchor (mean |Δ actual-token logprob|)

Usage:  tier0_matrix.py <results_dir> [anchor_quant]
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import tier0_divergence

_ANCHOR_PREF = ["bf16", "fp16", "fp8", "w8a16", "w8a8", "w4a16", "w4a8"]


def _latest_per_quant(results_dir: str) -> dict[str, dict]:
    """Map quant -> {dump, ppl} using the newest run for each quant."""
    out: dict[str, dict] = {}
    for dump in sorted(glob.glob(str(Path(results_dir) / "*" / "tier0_tokens.json"))):
        j = json.load(open(dump))
        q = j.get("quant")
        div = Path(dump).with_name("tier0_divergence.json")
        ppl = json.load(open(div)).get("ppl") if div.exists() else None
        out[q] = {"dump": dump, "ppl": ppl}  # sorted() => newest wins
    return out


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: tier0_matrix.py <results_dir> [anchor_quant]", file=sys.stderr)
        return 2
    runs = _latest_per_quant(sys.argv[1])
    if not runs:
        print("# no tier0_tokens.json dumps found", file=sys.stderr)
        return 1
    if len(sys.argv) >= 3 and sys.argv[2] in runs:
        anchor = sys.argv[2]
    else:
        anchor = next((q for p in _ANCHOR_PREF for q in runs if p in q), list(runs)[0])

    print(f"# Tier-0 divergence matrix (anchor = **{anchor}**)\n")
    print("| quant | ppl | top1-agree vs anchor | nll-gap vs anchor |")
    print("|---|---|---|---|")
    anchor_dump = runs[anchor]["dump"]
    for q in sorted(runs, key=lambda x: _ANCHOR_PREF.index(next((p for p in _ANCHOR_PREF if p in x), "w4a8"))
                    if any(p in x for p in _ANCHOR_PREF) else 99):
        ppl = runs[q]["ppl"]
        if q == anchor:
            agree, gap = "(anchor)", "(anchor)"
        else:
            c = tier0_divergence.compare(anchor_dump, runs[q]["dump"])
            agree = f"{c['top1_agreement']:.4f}" if c["top1_agreement"] is not None else "—"
            gap = f"{c['nll_gap_mean']:.4f}" if c["nll_gap_mean"] is not None else "—"
        ppls = f"{ppl:.4f}" if isinstance(ppl, (int, float)) else "—"
        print(f"| {q} | {ppls} | {agree} | {gap} |")
    print("\n> top1-agree = fraction of tokens where this quant's argmax matches the anchor's "
          "(1.0 = identical next-token decisions). nll-gap = mean |Δ logprob| of the actual token.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
