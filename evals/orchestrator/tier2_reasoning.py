"""Tier 2 — reasoning, exact-match (GSM8K). Thin wrapper around EleutherAI lm-evaluation-harness.

We shell out to the maintained grader (don't reinvent GSM8K parsing) but keep the orchestrator as the
single control point: it owns provenance + the normalized summary written next to the other tiers.

Per codex/lm-eval docs: use `local-completions` (NOT chat) for a quant-delta study — identical raw
prompts across all quants, and it can return logprobs; chat-completions adds template behavior that can
swamp the small delta we're trying to measure (README §6/§7).
"""
from __future__ import annotations

import glob
import json
import shutil
import subprocess

from common import RunContext, write_json


def run(ctx: RunContext, task: str = "gsm8k", limit: int | None = None) -> dict:
    if shutil.which("lm-eval") is None and shutil.which("lm_eval") is None:
        return {"tier": 2, "skipped": True,
                "error": "lm-eval not installed: pip install 'lm_eval[api]'"}
    binary = shutil.which("lm-eval") or shutil.which("lm_eval")
    out_dir = ctx.out_dir / "tier2_lm_eval"
    base = ctx.endpoint.rstrip("/") + "/completions"
    model_args = (
        f"model={ctx.model_id},base_url={base},num_concurrent={ctx.sampling.get('concurrency', 1)},"
        f"max_retries=3,tokenized_requests=False"
    )
    cmd = [
        binary, "--model", "local-completions",
        "--model_args", model_args,
        "--tasks", task,
        "--gen_kwargs", "temperature=0",
        "--output_path", str(out_dir),
        "--seed", str(ctx.sampling.get("seed", 1234)),
    ]
    if limit:
        cmd += ["--limit", str(limit)]
    print(f"[tier2] $ {' '.join(cmd)}")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    (out_dir).mkdir(parents=True, exist_ok=True)
    (out_dir / "stdout.log").write_text(proc.stdout + "\n---STDERR---\n" + proc.stderr)
    if proc.returncode != 0:
        return {"tier": 2, "task": task, "error": f"lm-eval exit {proc.returncode}; see stdout.log",
                "returncode": proc.returncode}

    # lm-eval writes results_*.json somewhere under out_dir; grab the newest.
    hits = sorted(glob.glob(str(out_dir / "**" / "results_*.json"), recursive=True))
    metrics = {}
    if hits:
        data = json.loads(open(hits[-1]).read())
        metrics = data.get("results", {}).get(task, {})
    # normalize: prefer strict exact_match, then any exact_match/acc
    score = None
    for key in ("exact_match,strict-match", "exact_match,flexible-extract", "exact_match", "acc,none", "acc"):
        if key in metrics:
            score = metrics[key]
            break
    result = {"tier": 2, "task": task, "score": score, "all_metrics": metrics, "limit": limit}
    write_json(ctx.out_dir / "tier2_reasoning.json", result)
    print(f"[tier2] {task} score={score}")
    return result
