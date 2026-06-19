#!/usr/bin/env python3
"""Roll up all per-run summary.json files into one 'retention vs reference' markdown table.

Usage:  report.py evals/results/  [> evals/results/SUMMARY.md]

Finds every results/<stamp>__<model>__<quant>/summary.json, picks the run whose quant looks like the
reference (label contains 'bf16' / 'fp16', or marked reference), and expresses each quant's headline
metrics as a delta / % of that reference. Deltas SMALLER than your measured noise floor are not real
(README §4) — this table doesn't know your noise floor, so eyeball it against the bf16-vs-bf16 run.
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path


def _load(results_dir: str) -> list[dict]:
    runs = []
    for sp in glob.glob(str(Path(results_dir) / "*" / "summary.json")):
        try:
            runs.append({"path": sp, **json.load(open(sp))})
        except Exception as e:  # noqa: BLE001
            print(f"# skip {sp}: {e}", file=sys.stderr)
    return runs


def _metric(run: dict, tier: str, key: str):
    t = run.get("tiers", {}).get(tier) or {}
    return t.get(key)


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: report.py <results_dir>", file=sys.stderr)
        return 2
    runs = _load(sys.argv[1])
    if not runs:
        print("# no summary.json found", file=sys.stderr)
        return 1

    ref = next((r for r in runs if any(k in (r.get("quant", "")) for k in ("bf16", "fp16"))), None)
    ref_ppl = _metric(ref, "0", "ppl") if ref else None

    print("# Quant retention vs reference\n")
    if ref:
        print(f"Reference: **{ref.get('quant')}** ({ref.get('model')})  ·  ref ppl = {ref_ppl}\n")
    print("| quant | tier0 ppl | top1-agree vs ref | nll-gap | tier1 pass@1(+) | tier2 score | tier3 renders-clean |")
    print("|---|---|---|---|---|---|---|")
    for r in sorted(runs, key=lambda x: x.get("quant", "")):
        ppl = _metric(r, "0", "ppl")
        agree = _metric(r, "0", "top1_agreement")
        gap = _metric(r, "0", "nll_gap_mean")
        t1 = _metric(r, "1", "pass@1")
        t1s = (t1 or {}).get("plus", (t1 or {}).get("base")) if isinstance(t1, dict) else None
        t2 = _metric(r, "2", "score")
        t3 = _metric(r, "3", "renders_clean_rate")
        def fmt(x, n=4):
            return f"{x:.{n}f}" if isinstance(x, (int, float)) else "—"
        print(f"| {r.get('quant')} | {fmt(ppl)} | {fmt(agree)} | {fmt(gap)} | {fmt(t1s)} | {fmt(t2)} | {fmt(t3)} |")
    print("\n> Reminder: trust a delta only if it exceeds your bf16-vs-bf16 noise floor (README §4).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
