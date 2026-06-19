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
    token_dump = []  # full per-position data so a quant can be compared vs the cached reference (one card)
    for j, text in enumerate(passages):
        s = score_passage(client, ctx.endpoint, ctx.model_id, text, top_k)
        if s is None:
            return {"error": "endpoint returned no prompt_logprobs; this vLLM build/flags don't "
                             "support prompt scoring. Try a newer image or the echo+logprobs route."}
        vals = [lp for lp in s["actual_logprob"] if lp is not None]
        tot_lp += sum(vals); tot_n += len(vals)
        rec = {"passage": j, "n_tokens": len(vals)}
        token_dump.append({"passage": j, "token_ids": s["token_ids"],
                           "argmax_id": s["argmax_id"], "actual_logprob": s["actual_logprob"]})

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
    # Always persist the per-token dump (the reference cache for one-card campaigns).
    dump_path = ctx.out_dir / "tier0_tokens.json"
    write_json(dump_path, {"model": ctx.model_id, "quant": ctx.quant, "corpus": str(corpus_path),
                           "top_k": top_k, "passages": token_dump})
    result["token_dump"] = str(dump_path)
    write_json(ctx.out_dir / "tier0_divergence.json", result)
    print(f"[tier0] ppl={ppl:.4f}  top1_agreement={result['top1_agreement']}  "
          f"nll_gap={result['nll_gap_mean']}  dump={dump_path}")
    return result


def compare(ref_dump_path: str, quant_dump_path: str) -> dict:
    """One-card path: compare a quant's cached token dump against the reference's.

    Serve bf16, run tier0 (caches tier0_tokens.json); serve the quant, run tier0 (its own cache);
    then `python tier0_divergence.py compare <bf16 dump> <quant dump>` -> top1_agreement + nll_gap.
    Aligns passages by index and asserts identical token_ids (same tokenizer) before comparing.
    """
    ref = json.loads(Path(ref_dump_path).read_text())
    qt = json.loads(Path(quant_dump_path).read_text())
    rp = {p["passage"]: p for p in ref["passages"]}
    agree_hits = agree_n = 0
    gap_sum = 0.0; gap_n = 0
    misaligned = 0
    for p in qt["passages"]:
        r = rp.get(p["passage"])
        if r is None or r["token_ids"] != p["token_ids"]:
            misaligned += 1
            continue
        for a, b in zip(p["argmax_id"], r["argmax_id"]):
            if a is not None and b is not None:
                agree_n += 1; agree_hits += int(a == b)
        for qlp, rlp in zip(p["actual_logprob"], r["actual_logprob"]):
            if qlp is not None and rlp is not None:
                gap_sum += abs(qlp - rlp); gap_n += 1
    return {"reference": ref.get("quant"), "quant": qt.get("quant"),
            "top1_agreement": (agree_hits / agree_n) if agree_n else None,
            "nll_gap_mean": (gap_sum / gap_n) if gap_n else None,
            "n_positions": agree_n, "misaligned_passages": misaligned}


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 4 and sys.argv[1] == "compare":
        print(json.dumps(compare(sys.argv[2], sys.argv[3]), indent=2))
    else:
        print("usage: tier0_divergence.py compare <ref tier0_tokens.json> <quant tier0_tokens.json>")
