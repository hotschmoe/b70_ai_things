"""Tier 2 — reasoning, exact-match (GSM8K). Self-contained grader (our harness, no lm-eval/torch).

Long reasoning chains compound per-token quant error, so GSM8K is a sensitive quant-delta signal. We
load GSM8K test via `datasets`, prompt the served model (chat, greedy, seed-pinned), extract the final
number, and exact-match against the gold answer. Keeping it in-repo means we control the prompt + answer
extraction (vs lm-eval's heavier, torch-pulling install) — better for ironing out OUR orchestrator.

(For the broader task zoo — MMLU-Pro, GPQA, etc. — lm-evaluation-harness with `--model
local-completions` against this same endpoint is the drop-in heavier alternative; see evals/README.)
"""
from __future__ import annotations

import re

from common import RunContext, write_json

_SYS = "You are a careful math problem solver. Show brief step-by-step reasoning."
_INSTR = ("\n\nSolve the problem. End your response with the final answer on its own line in EXACTLY "
          "this format:\n#### <number>")
_HASH = re.compile(r"####\s*\$?(-?[0-9][0-9,]*(?:\.[0-9]+)?)")
_NUM = re.compile(r"-?\$?\d[\d,]*(?:\.\d+)?")


def _to_num(s: str):
    try:
        return float(s.replace(",", "").replace("$", ""))
    except ValueError:
        return None


def _extract(text: str):
    """Predicted answer: prefer the #### line, else the last number in the text."""
    m = list(_HASH.finditer(text))
    if m:
        return _to_num(m[-1].group(1))
    nums = _NUM.findall(text)
    return _to_num(nums[-1]) if nums else None


def _gold(answer: str):
    m = _HASH.search(answer)
    return _to_num(m.group(1)) if m else None


def run(ctx: RunContext, task: str = "gsm8k", limit: int | None = None) -> dict:
    try:
        from datasets import load_dataset
    except ImportError:
        return {"tier": 2, "task": task, "skipped": True,
                "error": "datasets missing: pip install datasets"}
    from common import make_client
    client = make_client(ctx.endpoint)

    split = f"test[:{limit}]" if limit else "test"
    ds = load_dataset("openai/gsm8k", "main", split=split)  # canonical repo id (datasets>=5 requires ns/name)
    print(f"[tier2] gsm8k {len(ds)} items; model={ctx.model_id}")

    # Thinking OFF: bounded/fast outputs, and MORE sensitive to quant degradation (no long reasoning
    # to recover from per-token error). Qwen3 honors chat_template_kwargs.enable_thinking=False.
    extra = {"chat_template_kwargs": {"enable_thinking": False}}
    correct, items, raws = 0, [], []
    for i, ex in enumerate(ds):
        gold = _gold(ex["answer"])
        resp = client.chat.completions.create(
            model=ctx.model_id,
            messages=[{"role": "system", "content": _SYS},
                      {"role": "user", "content": ex["question"] + _INSTR}],
            temperature=0.0, seed=ctx.sampling.get("seed", 1234), max_tokens=640,
            extra_body=extra,
        )
        out = resp.choices[0].message.content or ""
        finish = resp.choices[0].finish_reason
        pred = _extract(out)
        ok = pred is not None and gold is not None and abs(pred - gold) < 1e-4
        correct += int(ok)
        items.append({"i": i, "gold": gold, "pred": pred, "ok": ok, "finish": finish, "out_len": len(out)})
        if i < 3:  # keep a few raw generations for debugging prompt/extraction
            raws.append({"i": i, "q": ex["question"], "out": out, "gold": gold, "pred": pred})
        if (i + 1) % 10 == 0:
            print(f"[tier2] {i+1}/{len(ds)} acc={correct/(i+1):.3f}", flush=True)
    write_json(ctx.out_dir / "tier2_raw_sample.json", raws)

    acc = correct / len(ds) if len(ds) else None
    result = {"tier": 2, "task": task, "n": len(ds), "score": acc, "correct": correct,
              "metric": "exact_match", "items": items}
    write_json(ctx.out_dir / "tier2_reasoning.json", result)
    print(f"[tier2] gsm8k acc={acc} ({correct}/{len(ds)})")
    return result
