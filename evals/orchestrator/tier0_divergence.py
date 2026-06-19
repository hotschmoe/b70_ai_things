"""Tier 0 — the canary. Distribution divergence of a quant vs the bf16 reference.

Fully deterministic, no grader. For a FIXED corpus we score every token under both models and report:
  - ppl:            perplexity of this model on the corpus (lower = better LM)
  - top1_agreement: fraction of positions where this model's argmax == the reference's argmax
  - nll_gap:        mean |actual-token logprob(this) - actual-token logprob(reference)|

top1_agreement and nll_gap need a reference endpoint; ppl is standalone.

Method (vLLM-native, robust): we get the exact actual-token logprobs via `prompt_logprobs`, and we
disambiguate which dict key is the *actual* token by tokenizing the passage through vLLM's /tokenize
endpoint (same tokenizer as the model, so quant and reference align position-for-position).

Caveats baked in: skip position 0 (BOS) for ppl/agreement; align by token IDs, not decoded strings
(README §7). API top-k truncates the distribution, so nll_gap/agreement are exact for the actual &
argmax tokens but a *full-vocab* KLD needs the offline forward-pass script (roadmap).
"""
from __future__ import annotations

import json
import math
import urllib.request
from pathlib import Path

from common import RunContext, write_json


def _tokenize(base_url: str, model: str, text: str) -> list[int]:
    """vLLM /tokenize (served at server root, not under /v1) -> list of token ids."""
    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        root = root[: -len("/v1")]
    req = urllib.request.Request(
        root + "/tokenize",
        data=json.dumps({"model": model, "prompt": text, "add_special_tokens": True}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read())["tokens"]


def score_passage(client, base_url: str, model: str, text: str, top_k: int) -> dict | None:
    """Per-position actual-token logprob + argmax token id, aligned to the tokenized prompt."""
    ids = _tokenize(base_url, model, text)
    resp = client.completions.create(
        model=model, prompt=text, max_tokens=0, temperature=0,
        echo=True, extra_body={"prompt_logprobs": top_k},
    )
    dump = resp.model_dump()
    plp = dump["choices"][0].get("prompt_logprobs")
    if not plp:
        return None  # build doesn't return prompt_logprobs -> caller reports a clear error
    # plp[i] is a dict {token_id: {logprob, rank, decoded_token}} (or None for BOS).
    actual_lp: list[float | None] = []
    argmax_id: list[int | None] = []
    for i, entry in enumerate(plp):
        if entry is None or i == 0:
            actual_lp.append(None)
            argmax_id.append(None)
            continue
        # keys may be ints or strings depending on json parsing
        aid = ids[i] if i < len(ids) else None
        lp = None
        if aid is not None:
            cell = entry.get(str(aid), entry.get(aid))  # type: ignore[arg-type]
            if cell is not None:
                lp = cell["logprob"]
        # argmax = the rank-1 entry
        top = min(entry.items(), key=lambda kv: kv[1]["rank"])
        actual_lp.append(lp)
        argmax_id.append(int(top[0]))
    return {"token_ids": ids, "actual_logprob": actual_lp, "argmax_id": argmax_id}


def _ppl(actual_logprobs: list[float | None]) -> tuple[float, int]:
    vals = [lp for lp in actual_logprobs if lp is not None]
    if not vals:
        return float("nan"), 0
    return math.exp(-sum(vals) / len(vals)), len(vals)


def load_corpus(path: str | Path) -> list[str]:
    """Corpus file split into passages on lines of exactly '---' (blank-padded ok)."""
    raw = Path(path).read_text()
    passages = [p.strip() for p in raw.split("\n---\n")]
    return [p for p in passages if len(p) > 40]


def run(ctx: RunContext, corpus_path: str, top_k: int = 5) -> dict:
    from common import make_client
    client = make_client(ctx.endpoint)
    ref_client = make_client(ctx.reference_endpoint) if ctx.reference_endpoint else None
    passages = load_corpus(corpus_path)
    print(f"[tier0] {len(passages)} passages; model={ctx.model_id} ref={ctx.reference_model_id}")

    tot_lp, tot_n = 0.0, 0
    agree_hits, agree_n, nll_gap_sum, nll_gap_n = 0, 0, 0.0, 0
    per_passage = []
    for j, text in enumerate(passages):
        s = score_passage(client, ctx.endpoint, ctx.model_id, text, top_k)
        if s is None:
            return {"error": "endpoint returned no prompt_logprobs; this vLLM build/flags don't "
                             "support prompt scoring. Try a newer image or the echo+logprobs route."}
        vals = [lp for lp in s["actual_logprob"] if lp is not None]
        tot_lp += sum(vals); tot_n += len(vals)
        rec = {"passage": j, "n_tokens": len(vals)}

        if ref_client is not None:
            r = score_passage(ref_client, ctx.reference_endpoint, ctx.reference_model_id, text, top_k)
            if r is not None and r["token_ids"] == s["token_ids"]:
                for a, b, qlp, rlp in zip(s["argmax_id"], r["argmax_id"],
                                          s["actual_logprob"], r["actual_logprob"]):
                    if a is not None and b is not None:
                        agree_n += 1
                        agree_hits += int(a == b)
                    if qlp is not None and rlp is not None:
                        nll_gap_sum += abs(qlp - rlp); nll_gap_n += 1
            else:
                rec["ref_misaligned"] = True
        per_passage.append(rec)

    ppl = math.exp(-tot_lp / tot_n) if tot_n else float("nan")
    result = {
        "tier": 0,
        "corpus": str(corpus_path),
        "n_passages": len(passages),
        "n_tokens_scored": tot_n,
        "ppl": ppl,
        "top1_agreement": (agree_hits / agree_n) if agree_n else None,
        "nll_gap_mean": (nll_gap_sum / nll_gap_n) if nll_gap_n else None,
        "vs_reference": ctx.reference_model_id,
        "per_passage": per_passage,
    }
    write_json(ctx.out_dir / "tier0_divergence.json", result)
    print(f"[tier0] ppl={ppl:.4f}  top1_agreement={result['top1_agreement']}  "
          f"nll_gap={result['nll_gap_mean']}")
    return result
