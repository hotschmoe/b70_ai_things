"""Tier 1 — execution-graded code (HumanEval+ / MBPP+) via EvalPlus.

Thin wrapper around `evalplus.evaluate --backend openai --base-url ... --greedy` (the maintained native
tool for the + datasets). The orchestrator owns provenance + the normalized summary.

SANDBOX WARNING: this *executes model-generated code* to grade it. EvalPlus recommends its Docker path,
especially on shared machines. Run this from a sandbox/VM or use EvalPlus's container. We surface, never
hide, that risk. Pass require_docker=True (default) to refuse running ungated.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from common import RunContext, write_json


def run(ctx: RunContext, dataset: str = "humaneval", require_docker: bool = True) -> dict:
    if shutil.which("evalplus.evaluate") is None and shutil.which("evalplus") is None:
        return {"tier": 1, "dataset": dataset, "skipped": True,
                "error": "evalplus not installed: pip install evalplus"}
    if require_docker:
        return {"tier": 1, "dataset": dataset, "skipped": True,
                "error": "Tier 1 executes generated code. Re-run with --allow-code-exec once you are in "
                         "a sandbox/VM, or use EvalPlus's Docker path. Refusing to run ungated."}

    out_dir = ctx.out_dir / "tier1_evalplus"
    out_dir.mkdir(parents=True, exist_ok=True)
    binary = shutil.which("evalplus.evaluate") or shutil.which("evalplus")
    cmd = [
        binary,
        "--model", ctx.model_id,
        "--dataset", dataset,
        "--backend", "openai",
        "--base-url", ctx.endpoint,
        "--greedy",
    ]
    print(f"[tier1] $ {' '.join(cmd)}")
    env_note = "set OPENAI_API_KEY=EMPTY before running"
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(out_dir))
    (out_dir / "stdout.log").write_text(proc.stdout + "\n---STDERR---\n" + proc.stderr)
    if proc.returncode != 0:
        return {"tier": 1, "dataset": dataset, "error": f"evalplus exit {proc.returncode}; see stdout.log",
                "hint": env_note}

    # EvalPlus prints pass@1 for base + plus; also writes an eval_results.json near the samples.
    scores = _parse_pass1(proc.stdout)
    hits = list(Path(out_dir).rglob("*eval_results.json"))
    result = {"tier": 1, "dataset": dataset, "pass@1": scores,
              "result_files": [str(h) for h in hits]}
    write_json(ctx.out_dir / "tier1_code.json", result)
    print(f"[tier1] {dataset} pass@1={scores}")
    return result


def _parse_pass1(stdout: str) -> dict:
    """EvalPlus prints lines like 'humaneval (base tests) pass@1: 0.71' and '(base + extra ...)'."""
    out = {}
    for line in stdout.splitlines():
        low = line.lower()
        if "pass@1" in low:
            try:
                val = float(low.split("pass@1")[1].split(":")[1].strip().split()[0])
            except (IndexError, ValueError):
                continue
            key = "plus" if "plus" in low or "extra" in low else "base"
            out[key] = val
    return out
