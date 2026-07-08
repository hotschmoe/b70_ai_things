#!/usr/bin/env python3
"""Drive a diverse calibration set through a running (calib-mode) nvfp4 server to
record per-full-attention-layer max|K|,max|V|. Run the server with
  B70_EXTRA_ENV="NVFP4_KV_CALIB_OUT=/tmp_ssd/kv_amax.json" ... GRAPH=0 KV_FP8=1
then run this inside the container:  python3 /opt/nvfp4_shim/../kv_calibrate.py
(or docker cp it in). It hits PROBE_HOST (default http://127.0.0.1:8079).

The forward hook (sitecustomize block 10) captures the post-RoPE/post-norm K,V that
are actually written to the fp8 cache, so amax here == the clipping-relevant amax.
"""
import json, os, urllib.request, urllib.error, time, sys

HOST = os.environ.get("PROBE_HOST", "http://127.0.0.1:8079")
MAXTOK = int(os.environ.get("CALIB_DECODE", "48"))
LONG_N = int(os.environ.get("CALIB_LONG", "6"))


def models():
    return json.load(urllib.request.urlopen(HOST + "/v1/models", timeout=15))["data"][0]["id"]


def call(mid, prompt, maxtok):
    body = {"model": mid, "prompt": prompt, "max_tokens": maxtok, "temperature": 0.7,
            "top_p": 0.95, "seed": 0}
    r = urllib.request.Request(HOST + "/v1/completions", data=json.dumps(body).encode(),
                               headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=600) as x:
        return json.load(x)


CODE = [
    "def quicksort(a):\n    if len(a) <= 1:\n        return a\n",
    "import torch\nclass Attention(torch.nn.Module):\n    def __init__(self, dim, heads):\n",
    "package main\nimport \"fmt\"\nfunc main() {\n    ch := make(chan int)\n",
    "SELECT u.name, COUNT(o.id) FROM users u JOIN orders o ON o.user_id = u.id GROUP BY",
    "const std = @import(\"std\");\npub fn main() !void {\n    var gpa = std.heap.GeneralPurposeAllocator",
    "fn fibonacci(n: u64) u64 {\n    if (n < 2) return n;\n    return fibonacci(n - 1)",
    "class Solution:\n    def twoSum(self, nums, target):\n        seen = {}\n",
    "#include <vector>\ntemplate<typename T> class RingBuffer {\n    std::vector<T> buf;\n",
    "async function fetchAll(urls) {\n    const results = await Promise.all(urls.map",
    "resource \"aws_instance\" \"web\" {\n  ami = \"ami-0abcd\"\n  instance_type = \"t3.micro\"\n",
]
PROSE = [
    "The industrial revolution began in the late eighteenth century in Britain, transforming",
    "In the quiet hours before dawn, the old lighthouse keeper climbed the spiral stairs,",
    "Photosynthesis is the process by which green plants convert light energy into chemical",
    "The French Revolution of 1789 fundamentally reshaped European political thought and",
    "Quantum entanglement describes a phenomenon in which two particles become correlated",
    "The Silk Road was an ancient network of trade routes connecting the East and West,",
    "Machine learning models learn patterns from data by minimizing a loss function through",
    "The human immune system defends the body against pathogens using both innate and adaptive",
    "Climate scientists measure atmospheric carbon dioxide concentrations at observatories",
    "Shakespeare's tragedies explore themes of ambition, betrayal, madness, and the fragility",
]
MATH = [
    "Prove that the square root of 2 is irrational. Proof: Suppose for contradiction that",
    "To compute the derivative of f(x) = x^3 sin(x), we apply the product rule:",
    "The eigenvalues of a symmetric matrix are always real because",
    "Solve the system: 3x + 2y = 12 and x - y = 1. First, from the second equation",
    "The probability of drawing two aces from a standard deck without replacement is",
]
CHAT = [
    "<|im_start|>user\nExplain how a transformer attention head works.<|im_end|>\n<|im_start|>assistant\n",
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nWhat causes the seasons?<|im_end|>\n<|im_start|>assistant\n",
    "<|im_start|>user\nWrite a haiku about the ocean.<|im_end|>\n<|im_start|>assistant\n",
]
MULTI = [
    "La inteligencia artificial esta transformando la manera en que trabajamos y vivimos.",
    "人工知能は私たちの働き方を大きく変えています。以下にその理由を説明します。",
    "Die kuenstliche Intelligenz veraendert die Art und Weise, wie wir arbeiten.",
    "L'intelligence artificielle transforme notre facon de travailler et de vivre.",
]
LONG_SEED = (
    "The distributed system maintains consistency through a consensus protocol. "
    "Each node in the cluster participates in leader election and log replication. "
    "When a client submits a write, the leader appends it to its log and replicates "
    "to followers before committing. Read requests may be served from any replica "
    "depending on the configured consistency level. Network partitions are handled "
    "by requiring a quorum for progress. The state machine applies committed entries "
    "in order, guaranteeing linearizability. Snapshots compact the log periodically. "
)


def main():
    mid = models()
    print("served id:", mid, flush=True)
    prompts = CODE + PROSE + MATH + CHAT + MULTI
    # duplicate the base set a couple times with light variation for more samples
    base = []
    for rep in range(3):
        for p in prompts:
            base.append((p + (" " if rep else ""), MAXTOK))
    # long-context prompts to capture position-dependent amax (fill ~6-7k tokens)
    long_prompt = (LONG_SEED * 90)[:26000]  # ~6-7k tokens
    for i in range(LONG_N):
        base.append(("Summarize the following in one sentence.\n\n" + long_prompt + f"\n\n(variant {i}) Summary:", 32))
    print(f"total calibration requests: {len(base)}", flush=True)
    t0 = time.time()
    for i, (p, mt) in enumerate(base):
        try:
            call(mid, p, mt)
        except Exception as e:
            print(f"  req{i} error: {type(e).__name__}: {e}", flush=True)
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(base)} done ({time.time()-t0:.0f}s)", flush=True)
    print(f"calibration drive complete in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
