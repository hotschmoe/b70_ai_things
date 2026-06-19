#!/usr/bin/env python3
"""Build a side-by-side Tier-3 gallery comparing the SAME creative prompt across quants.

Discovers the latest tier3_creative run per quant under <results_dir>, then writes
<results_dir>/gallery.html: one row per prompt, one column per quant, each cell = the rendered
screenshot (click -> the live HTML). Serve <results_dir> with `python -m http.server` to view.

Usage:  gallery.py <results_dir> [quant1 quant2 ...]   (default quants: fp8 w8a8 w4a16)
"""
from __future__ import annotations

import glob
import os
import sys


def main() -> int:
    results = sys.argv[1] if len(sys.argv) > 1 else "evals/results"
    quants = sys.argv[2:] or ["fp8", "w8a8", "w4a16"]
    runs = {}
    for q in quants:
        cands = sorted(glob.glob(f"{results}/*__{q}/tier3_creative"))
        if cands:
            runs[q] = cands[-1]
    ids = sorted({os.path.basename(p)[:-5] for r in runs.values() for p in glob.glob(f"{r}/*.html")})

    rows = []
    for pid in ids:
        cells = []
        for q in quants:
            r = runs.get(q)
            png = f"{r}/{pid}.png" if r else None
            html = f"{r}/{pid}.html" if r else None
            relp = os.path.relpath(png, results) if png and os.path.exists(png) else None
            relh = os.path.relpath(html, results) if html and os.path.exists(html) else None
            inner = f'<a href="{relh}" target="_blank"><img src="{relp}"></a>' if relp else "(none)"
            cells.append(f'<td><div class="q">{q}</div>{inner}</td>')
        rows.append(f"<tr><th>{pid}</th>{''.join(cells)}</tr>")

    doc = (
        "<!DOCTYPE html><html><head><meta charset='utf-8'><title>Quant creative gallery</title>"
        "<style>body{font-family:sans-serif;background:#111;color:#eee;margin:20px}"
        "img{width:320px;height:240px;object-fit:contain;border:1px solid #333;background:#fff}"
        "td{vertical-align:top;padding:8px}th{text-align:left;color:#aaa}.q{font-weight:bold;color:#6cf;margin-bottom:4px}"
        "h1{color:#fff}</style></head><body>"
        "<h1>Tier-3 creative — Qwen3-14B quant comparison</h1>"
        "<p>Same prompt across quants. Click an image to open the live HTML. Screenshots captured @2.5s "
        "(animations may be mid-intro).</p><table>" + "".join(rows) + "</table></body></html>"
    )
    out = f"{results}/gallery.html"
    with open(out, "w") as f:
        f.write(doc)
    print(f"wrote {out}  ({len(ids)} prompts × {len(runs)} quants: {list(runs)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
