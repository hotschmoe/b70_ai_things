"""Fixed local text perplexity diagnostic; no external benchmark claim."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from urllib.request import Request, urlopen

# Original small prose/code fixture, identical bytes and tokenizer across quants.
CORPUS = [
    "A hash table stores values under keys. A hash function maps each key to a bucket. "
    "When two keys map to the same bucket, the implementation must resolve the collision. "
    "Separate chaining keeps a list of entries in each bucket. Open addressing searches "
    "for another available slot. Resizing the table reduces the load factor, but requires "
    "moving entries to new buckets. A good hash function distributes ordinary keys evenly.",
    "Water evaporates from lakes and oceans as the sun warms the surface. The vapor rises "
    "and cools in the atmosphere. Small droplets form clouds. When the droplets grow heavy "
    "enough, rain falls to the ground. Some water enters rivers, while some soaks into soil. "
    "Plants absorb water through their roots and release it through their leaves. This "
    "cycle moves water between the atmosphere, the land, and the sea.",
    "def fibonacci(n):\n    values = []\n    a, b = 0, 1\n    for _ in range(n):\n"
    "        values.append(a)\n        a, b = b, a + b\n    return values\n\n"
    "# The function returns the first n Fibonacci numbers.\n"
    "assert fibonacci(7) == [0, 1, 1, 2, 3, 5, 8]\n",
    "The railway station was quiet before dawn. A worker checked the departure board "
    "while the first passengers waited on the platform. At six o'clock a train arrived "
    "from the coast. Its doors opened, and a family stepped down with two small bags. "
    "The conductor checked the platform before giving the signal to leave. Soon the "
    "train disappeared around the bend, and the station became quiet again.",
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--corpus", default="/mnt/vm_8tb/b70/build/vllm-flashnext/x5/wikitext2-prefix.json")
    args = p.parse_args()
    corpus_path = Path(args.corpus)
    if corpus_path.exists():
        fixture = json.loads(corpus_path.read_text())
        prompts = [row["token_ids"] for row in fixture["passages"]]
        label = fixture["label"]
        corpus_sha = hashlib.sha256(corpus_path.read_bytes()).hexdigest()
    else:
        fixture = None
        prompts = CORPUS
        label = "X5 original four-passage diagnostic, not benchmark perplexity"
        corpus_sha = hashlib.sha256("\n".join(CORPUS).encode("ascii")).hexdigest()
    rows = []
    for prompt in prompts:
        body = dict(model="hotschmoe-dd", prompt=prompt, max_tokens=0, echo=True,
                    logprobs=1, temperature=0, return_token_ids=True)
        req = Request(args.url + "/v1/completions", json.dumps(body).encode("ascii"),
                      headers={"Content-Type": "application/json"})
        with urlopen(req, timeout=180) as r:
            response = json.load(r)
        choice = response["choices"][0]
        values = choice["logprobs"]["token_logprobs"]
        scored = [x for x in values if x is not None]
        assert len(scored) > 10 and all(math.isfinite(x) for x in scored)
        if isinstance(prompt, list):
            assert len(values) == len(prompt), (len(values), len(prompt))
            assert len(scored) == len(prompt) - 1, (len(scored), len(prompt))
        rows.append(dict(prompt=prompt, n=len(scored), nll=-sum(scored), response=response))
    n = sum(r["n"] for r in rows)
    nll = sum(r["nll"] for r in rows)
    out = dict(label=label, fixture=fixture, corpus_sha256=corpus_sha,
               scored_tokens=n, mean_nll=nll/n, perplexity=math.exp(nll/n), rows=rows)
    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=True) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k not in ("rows", "fixture")}))


if __name__ == "__main__":
    main()
