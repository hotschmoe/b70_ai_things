#!/usr/bin/env python3
"""Aggregate results/<config>/<harness>.json -> a markdown scoreboard + scores.json.

Builds the campaign table (rows = the 4 configs in canonical order, columns = each harness score
plus total wall-clock and total tokens), writes results/scores.json and results/SUMMARY.md, and
(if --readme given) injects the table between the <!-- RESULTS:START/END --> markers in the README.
Stdlib only.
"""
import argparse, glob, json, os

CONFIG_ORDER = ["27b-int4", "27b-w8a8", "35b-int4", "35b-w8a8"]
HARNESS_ORDER = ["aider", "bfcl", "tau2", "swe"]
HARNESS_LABEL = {"aider": "Aider (codegen control)", "bfcl": "BFCL multi-turn (tool isolator)",
                 "tau2": "tau2 (multi-turn tool)", "swe": "SWE/mini (agentic coding)"}
ARCH_OF = {"27b-int4": "dense", "27b-w8a8": "dense", "35b-int4": "moe", "35b-w8a8": "moe"}


def load(results_dir):
    data = {}  # config -> harness -> result
    for path in glob.glob(os.path.join(results_dir, "*", "*.json")):
        try:
            with open(path) as f:
                r = json.load(f)
        except Exception:
            continue
        c, h = r.get("config"), r.get("harness")
        if c and h:
            data.setdefault(c, {})[h] = r
    return data


def _fmt_pct(x):
    return f"{x*100:.1f}%" if isinstance(x, (int, float)) else "--"


def _fmt_int(x):
    return f"{x:,}" if isinstance(x, (int, float)) else "--"


def _fmt_secs(x):
    if not isinstance(x, (int, float)):
        return "--"
    m, s = divmod(int(x), 60)
    return f"{m}m{s:02d}s" if m else f"{s}s"


def build_table(data):
    configs = [c for c in CONFIG_ORDER if c in data] + [c for c in data if c not in CONFIG_ORDER]
    harnesses = HARNESS_ORDER
    head = ["config", "arch"] + [HARNESS_LABEL[h] for h in harnesses] + ["total wall", "total tokens", "gen tok/s"]
    rows = [head, ["---"] * len(head)]
    for c in configs:
        hs = data[c]
        row = [f"`{c}`", ARCH_OF.get(c, "?")]
        wall = 0.0
        toks = 0
        gen = 0
        for h in harnesses:
            r = hs.get(h)
            if r and r.get("score") is not None:
                sn = r.get("score_name", "")
                row.append(f"{_fmt_pct(r['score'])} ({sn} n={r.get('n_tasks','?')})")
            else:
                row.append("--")
            if r:
                if isinstance(r.get("wall_s"), (int, float)):
                    wall += r["wall_s"]
                if isinstance(r.get("tokens_total"), (int, float)):
                    toks += r["tokens_total"]
                if isinstance(r.get("tokens_gen"), (int, float)):
                    gen += r["tokens_gen"]
        row.append(_fmt_secs(wall))
        row.append(_fmt_int(toks) if toks else "--")
        row.append(f"{gen/wall:.1f}" if wall > 0 and gen else "--")
        rows.append(row)
    return "\n".join("| " + " | ".join(str(x) for x in r) + " |" for r in rows)


def within_arch_deltas(data):
    """int4 - w8a8 per architecture per harness (the load-bearing contrast)."""
    pairs = [("dense", "27b-int4", "27b-w8a8"), ("moe", "35b-int4", "35b-w8a8")]
    lines = []
    for arch, a, b in pairs:
        if a not in data or b not in data:
            continue
        for h in HARNESS_ORDER:
            ra, rb = data[a].get(h), data[b].get(h)
            if ra and rb and ra.get("score") is not None and rb.get("score") is not None:
                d = (ra["score"] - rb["score"]) * 100
                lines.append(f"- {arch} {h}: int4 {_fmt_pct(ra['score'])} - w8a8 {_fmt_pct(rb['score'])} "
                             f"= **{d:+.1f} pp**  (int4 {'better' if d>=0 else 'WORSE'})")
    return "\n".join(lines) if lines else "_(no comparable pairs yet)_"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--readme", default=None)
    args = ap.parse_args()
    data = load(args.results)
    table = build_table(data)
    deltas = within_arch_deltas(data)

    scores = {c: {h: {k: data[c][h].get(k) for k in
                      ("score", "score_name", "n_tasks", "n_passed", "wall_s", "tokens_total", "tokens_gen")}
                  for h in data[c]} for c in data}
    with open(os.path.join(args.results, "scores.json"), "w") as f:
        json.dump(scores, f, indent=2)

    body = (f"## Scoreboard\n\n{table}\n\n"
            f"### Within-architecture quant deltas (int4 - w8a8)\n\n{deltas}\n\n"
            f"_Greedy (temp=0). Scores are concurrency-invariant; wall-clock/tokens are at the fixed "
            f"eval concurrency. Read int4-vs-w8a8 within an arch; do not read dense-vs-moe as a quant effect._\n")
    with open(os.path.join(args.results, "SUMMARY.md"), "w") as f:
        f.write("# Agentic-eval scoreboard (generated)\n\n" + body)
    print(body)

    if args.readme and os.path.exists(args.readme):
        with open(args.readme) as f:
            txt = f.read()
        s, e = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"
        if s in txt and e in txt:
            new = txt[:txt.index(s) + len(s)] + "\n" + body + "\n" + txt[txt.index(e):]
            with open(args.readme, "w") as f:
                f.write(new)
            print(f"[summarize] injected scoreboard into {args.readme}")


if __name__ == "__main__":
    main()
