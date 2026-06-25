#!/usr/bin/env python3
"""agentic-eval shared helpers: vLLM token accounting + standard result emission.

Stdlib only (urllib, json) so the SYSTEM python runs it -- the per-harness uv venvs are
isolated and need not share this. Two subcommands:

  evallib.py snap <port>
      Print "PROMPT GEN" -- cumulative vllm:{prompt,generation}_tokens_total counters summed
      across all model labels, or "NA NA" if /metrics is unreachable. A harness run.sh snaps
      before + after and the delta is that run's token cost for the config under test (valid
      because exactly one config is served at a time and harnesses run serially).

  evallib.py emit --config L --harness H --subset S --served NAME \
        --parsed parsed.json --tok-before "P G" --tok-after "P G" \
        --start <epoch> --end <epoch> --out OUT [--meta k=v ...]
      Write the canonical per-(config,harness) result JSON (schema in docs/HARNESS_CONTRACT.md).

The 'parsed.json' a harness produces must contain at least:
  {"score": <float 0..1>, "score_name": "<str>", "n_tasks": <int>,
   "per_task": [{"task_id": "<str>", "passed": <bool>}, ...], "extra": {<any>}}
"""
import argparse, json, os, sys, urllib.request


def _fetch(url, timeout=5):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read().decode("utf-8", "replace")


def snapshot_tokens(port):
    """(prompt_total, gen_total) ints, or (None, None) if metrics unavailable."""
    try:
        text = _fetch(f"http://localhost:{port}/metrics")
    except Exception:
        return (None, None)
    p = g = 0.0
    pseen = gseen = False
    for line in text.splitlines():
        if not line or line[0] == "#":
            continue
        # lines look like:  vllm:prompt_tokens_total{model_name="x"} 1234.0
        if line.startswith("vllm:prompt_tokens_total"):
            try:
                p += float(line.rsplit(" ", 1)[-1]); pseen = True
            except ValueError:
                pass
        elif line.startswith("vllm:generation_tokens_total"):
            try:
                g += float(line.rsplit(" ", 1)[-1]); gseen = True
            except ValueError:
                pass
    return (int(p) if pseen else None, int(g) if gseen else None)


def _parse_pair(s):
    """'P G' -> (int|None, int|None); 'NA NA' -> (None, None)."""
    a, b = (s.split() + ["NA", "NA"])[:2]
    f = lambda x: None if x in ("NA", "None", "") else int(float(x))
    return f(a), f(b)


def _delta(after, before):
    if after is None or before is None:
        return None
    d = after - before
    return d if d >= 0 else None  # counter reset -> unknown


def cmd_snap(args):
    p, g = snapshot_tokens(args.port)
    print(f"{p if p is not None else 'NA'} {g if g is not None else 'NA'}")


def cmd_emit(args):
    with open(args.parsed) as f:
        parsed = json.load(f)
    pb, gb = _parse_pair(args.tok_before)
    pa, ga = _parse_pair(args.tok_after)
    tp, tg = _delta(pa, pb), _delta(ga, gb)
    tt = (tp + tg) if (tp is not None and tg is not None) else None
    wall = round(float(args.end) - float(args.start), 2)
    per_task = parsed.get("per_task", [])
    meta = {}
    for kv in (args.meta or []):
        if "=" in kv:
            k, v = kv.split("=", 1)
            meta[k] = v
    # auto-stamp the run conditions from the environment (no per-harness wiring needed)
    for k, env in (("thinking", "EVAL_THINKING"), ("max_len", "EVAL_MAXLEN"),
                   ("max_tokens", "AE_MAX_TOKENS"), ("concurrency", "AE_CONCURRENCY")):
        if env in os.environ and k not in meta:
            meta[k] = os.environ[env]
    out = {
        "config": args.config,
        "harness": args.harness,
        "subset": args.subset,
        "served_model_id": args.served,
        "score": parsed.get("score"),
        "score_name": parsed.get("score_name"),
        "n_tasks": parsed.get("n_tasks", len(per_task)),
        "n_passed": sum(1 for t in per_task if t.get("passed")),
        "per_task": per_task,
        "extra_scores": parsed.get("extra", {}),
        "wall_s": wall,
        "tokens_prompt": tp,
        "tokens_gen": tg,
        "tokens_total": tt,
        "throughput_tok_s": (round(tg / wall, 1) if (tg is not None and wall > 0) else None),
        "temperature": float(meta.pop("temperature", 0.0)),
        "meta": meta,
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    sc = out["score"]
    print(f"[emit] {args.config}/{args.harness}: {out['score_name']}="
          f"{sc if sc is not None else 'NA'} n={out['n_tasks']} "
          f"wall={wall}s tok_total={tt} -> {args.out}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("snap"); s.add_argument("port", type=int); s.set_defaults(fn=cmd_snap)
    e = sub.add_parser("emit")
    e.add_argument("--config", required=True); e.add_argument("--harness", required=True)
    e.add_argument("--subset", default="standard"); e.add_argument("--served", default="")
    e.add_argument("--parsed", required=True)
    e.add_argument("--tok-before", default="NA NA"); e.add_argument("--tok-after", default="NA NA")
    e.add_argument("--start", required=True); e.add_argument("--end", required=True)
    e.add_argument("--out", required=True); e.add_argument("--meta", action="append")
    e.set_defaults(fn=cmd_emit)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
