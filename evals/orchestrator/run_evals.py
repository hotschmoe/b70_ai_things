#!/usr/bin/env python3
"""Quant eval orchestrator — single control point for one (model, quant) run.

Brings the tiers together with consistent provenance + output. The GPU box just serves vLLM
(OpenAI API); this runs on the dev box and hits it over the LAN.

Examples:
  # smoke the endpoint(s)
  run_evals.py --endpoint http://192.168.10.5:18080/v1 --check

  # Tier 0 (divergence) + Tier 2 (gsm8k) for W8A8 vs the bf16 reference
  run_evals.py --endpoint http://192.168.10.5:18080/v1 --model Qwen3-14B-W8A8-INT8 --quant w8a8 \
               --reference-endpoint http://192.168.10.5:18080/v1 --reference-model Qwen3-14B \
               --tiers 0,2 --limit 200

  # noise floor: bf16 against itself
  run_evals.py --endpoint .../v1 --model Qwen3-14B --quant bf16-runA --tiers 0,2 --limit 200
  run_evals.py --endpoint .../v1 --model Qwen3-14B --quant bf16-runB --tiers 0,2 --limit 200
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # make `import common`, `import tierN` work

import common  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CORPUS = REPO_ROOT / "evals" / "prompts" / "tier0_corpus.txt"
DEFAULT_SUITE = REPO_ROOT / "evals" / "prompts" / "tier3_creative.yaml"


def main() -> int:
    sys.stdout.reconfigure(line_buffering=True)  # so tier prints show up live in redirected logs
    ap = argparse.ArgumentParser(description="Quant eval orchestrator")
    ap.add_argument("--endpoint", required=True, help="OpenAI-compatible base url, e.g. http://host:18080/v1")
    ap.add_argument("--model", help="served-model-id to test")
    ap.add_argument("--quant", help="quant label, e.g. w8a8 (used in output paths + report)")
    ap.add_argument("--reference-endpoint", help="reference (bf16) endpoint for Tier 0 agreement/KLD")
    ap.add_argument("--reference-model", help="reference served-model-id (the bf16 ceiling)")
    ap.add_argument("--tiers", default="0", help="comma list of tiers to run, e.g. 0,1,2,3")
    ap.add_argument("--corpus", default=str(DEFAULT_CORPUS), help="Tier 0 corpus file")
    ap.add_argument("--suite", default=str(DEFAULT_SUITE), help="Tier 3 prompt suite yaml")
    ap.add_argument("--task", default="gsm8k", help="Tier 2 lm-eval task")
    ap.add_argument("--limit", type=int, default=None, help="cap items (tier1/tier2/tier3) for quick runs")
    ap.add_argument("--tier1-dataset", default="humaneval", help="Tier 1 EvalPlus dataset: humaneval|mbpp")
    ap.add_argument("--tier1-think", action="store_true",
                    help="Tier 1: generate with thinking ON (default off, matching tiers 2/3)")
    ap.add_argument("--tier1-image", default="evalplus-sandbox:0.3.1",
                    help="Tier 1 sandbox image (build via evals/sandbox/build.sh)")
    ap.add_argument("--allow-code-exec", action="store_true",
                    help="Tier 1: permit UNSANDBOXED host execution if the Docker sandbox is unavailable")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--check", action="store_true", help="just health-check the endpoint(s) and exit")
    args = ap.parse_args()

    if args.check:
        for name, ep in [("endpoint", args.endpoint), ("reference", args.reference_endpoint)]:
            if ep:
                r = common.check_endpoint(ep)
                print(f"[{name}] {ep} -> ok={r['ok']} models={r['models']} err={r['error']}")
        return 0

    if not args.model or not args.quant:
        ap.error("--model and --quant are required unless --check")

    ctx = common.RunContext(
        model_id=args.model, quant=args.quant, endpoint=args.endpoint,
        reference_model_id=args.reference_model, reference_endpoint=args.reference_endpoint,
        sampling={"temperature": 0.0, "top_p": 1.0, "seed": args.seed,
                  "max_tokens": args.max_tokens, "concurrency": 1},
    )
    ep = common.check_endpoint(args.endpoint)
    if not ep["ok"]:
        print(f"FATAL: endpoint not reachable: {ep['error']}", file=sys.stderr)
        return 2
    if args.model not in ep["models"]:
        print(f"WARNING: '{args.model}' not in served models {ep['models']} — check --model / served-name")

    ctx.write_config()
    print(f"== run {ctx.quant} -> {ctx.out_dir}")

    tiers = [t.strip() for t in args.tiers.split(",") if t.strip()]
    summary: dict = {"model": args.model, "quant": args.quant, "tiers": {}}

    if "0" in tiers:
        import tier0_divergence
        summary["tiers"]["0"] = tier0_divergence.run(ctx, args.corpus)
    if "1" in tiers:
        import tier1_code
        summary["tiers"]["1"] = tier1_code.run(
            ctx, dataset=args.tier1_dataset, limit=args.limit, think=args.tier1_think,
            image=args.tier1_image, allow_host_exec=args.allow_code_exec)
    if "2" in tiers:
        import tier2_reasoning
        summary["tiers"]["2"] = tier2_reasoning.run(ctx, task=args.task, limit=args.limit)
    if "3" in tiers:
        import tier3_creative
        summary["tiers"]["3"] = tier3_creative.run(ctx, args.suite, limit=args.limit)

    common.write_json(ctx.out_dir / "summary.json", summary)
    print(f"== done. summary -> {ctx.out_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
