#!/usr/bin/env python3
"""N1 bounded same-load C1/C2 determinism gate (not a latency benchmark).

Use the coordinated sidecar launch environment from O5/F1, adding the N1
plugin to PYTHONPATH and B70_GDN_BA_DETERMINISTIC=1 after primitive PASS.
N1_OUT, N1_TAG, N1_MTP (default0), N1_REPEATS (default2), N1_EAGER=1
(optional isolation). Full token arrays support separate-load/MTP parity.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
import torch
from vllm import LLM, SamplingParams

OUT = Path(os.environ["N1_OUT"])
OUT.mkdir(parents=True, exist_ok=True)
k = int(os.environ.get("N1_MTP", "0"))
tag = os.environ.get("N1_TAG", f"repeat{k}")
kwargs = dict(
    model="/mnt/vm_8tb/b70/build/backend-decision/hf-flashnext",
    served_model_name=["hotschmoe-dd", f"qwen3.8-flash-next-iq3-xxs-sidecar-n1-det-mtp{k}"],
    tensor_parallel_size=1, dtype="bfloat16", max_model_len=2048,
    max_num_batched_tokens=2048, max_num_seqs=4,
    gpu_memory_utilization=0.98, language_model_only=True,
    load_format="b70_gguf", quantization="b70_gguf", trust_remote_code=True,
    attention_backend="FLASH_ATTN", enable_prefix_caching=False,
    mamba_cache_mode="align", enforce_eager=os.environ.get("N1_EAGER") == "1",
    compilation_config=dict(mode=0, cudagraph_mode="FULL_DECODE_ONLY",
                            cudagraph_capture_sizes=sorted({1, 2, 3, 4, 8, k+1, 2*(k+1)})),
)
if k:
    kwargs["speculative_config"] = dict(method="mtp", num_speculative_tokens=k)
    if os.environ.get("B70_MTP_DRAFT_VOCAB"):
        kwargs["speculative_config"]["use_local_argmax_reduction"] = True
print(json.dumps(dict(tag=tag, kwargs=kwargs, flags={key: value for key, value in os.environ.items() if key.startswith("B70_")})), flush=True)
llm = LLM(**kwargs)
sp = SamplingParams(temperature=0.0, max_tokens=16)
gates = []
for prompt, ref in (("Return only the integer: 17 * 19.", "323"), ("Return only the integer: 23 * 29.", "667")):
    out = llm.chat([dict(role="user", content=prompt)], sp, chat_template_kwargs={"enable_thinking": False}, use_tqdm=False)
    text = out[0].outputs[0].text.strip()
    gates.append(dict(ref=ref, text=text, match=text == ref))
assert all(g["match"] for g in gates), gates
prompts = [
    "Write a long, detailed essay about the history of the city of Paris.",
    "Explain step by step how a hash map works, then implement one in Python with open addressing.",
    "List the first forty prime numbers and explain the sieve of Eratosthenes.",
    "Write a short story about a lighthouse keeper who discovers a message in a bottle.",
]
sp = SamplingParams(temperature=0.0, max_tokens=128, ignore_eos=True)
batches = dict(c1=[[p] for p in prompts], c2=[prompts[:2], prompts[2:]])
records = {}
for repeat in range(int(os.environ.get("N1_REPEATS", "2"))):
    phase_rows = {}
    for phase, requests in batches.items():
        rows = []
        for batch in requests:
            out = llm.generate(batch, sp, use_tqdm=False)
            rows.append([dict(ids=list(row.outputs[0].token_ids), text=row.outputs[0].text) for row in out])
        phase_rows[phase] = rows
    records[str(repeat)] = phase_rows
    print(json.dumps(dict(repeat=repeat, identical_to_first=phase_rows == records["0"])), flush=True)
    (OUT / f"tokens-{tag}.json").write_text(json.dumps(dict(tag=tag, gates=gates, kwargs=kwargs, records=records), indent=2)+"\n")
cross_batch = {}
for repeat, row in records.items():
    c1 = [batch[0]["ids"] for batch in row["c1"]]
    c2 = [stream["ids"] for batch in row["c2"] for stream in batch]
    cross_batch[repeat] = [a == b for a, b in zip(c1, c2)]
print(json.dumps(dict(c1_vs_c2=cross_batch)), flush=True)
assert all(row == records["0"] for row in records.values()), "same-load C1/C2 determinism failed"
print("N1_REPEAT_OK", flush=True)
# Explicit engine shutdown must precede the orchestrator's post-health.
engine = getattr(llm, "llm_engine", None)
core = getattr(engine, "engine_core", None)
if core is not None:
    core.shutdown()
del llm
sys.stdout.flush()
