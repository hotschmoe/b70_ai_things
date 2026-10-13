#!/usr/bin/env python3
"""Long-context decode probe: measure decode t/s AT a given context length (the perf_probe decode is
short-context). Decode reads the weights (fixed) + the WHOLE KV cache each step, so decode degrades as
context grows -- this charts that, and (vs an fp16-KV serve) shows fp8-KV's growing win. Usage:
  longctx_probe.py <base_url> <model> <ctx_tokens> [out=96]
"""
import os, sys, time, uuid
from openai import OpenAI

BASE, MODEL = sys.argv[1], sys.argv[2]
CTX = int(sys.argv[3])
OUT = int(sys.argv[4]) if len(sys.argv) > 4 else 96
client = OpenAI(base_url=BASE, api_key=os.environ.get("OPENAI_API_KEY", "x"), timeout=600.0)

UNIT = "The history of computing is long and full of surprising turns. "  # ~12 tokens
# unique nonce first: defeats prefix caching across runs/streams (token cost is negligible)
prompt = f"[probe {uuid.uuid4().hex}] " + UNIT * max(1, CTX // 12)

t0 = time.perf_counter(); first = None; n = 0
s = client.completions.create(model=MODEL, prompt=prompt, max_tokens=OUT, temperature=0, stream=True,
                              stream_options={"include_usage": True})
usage = None
for ch in s:
    if getattr(ch, "usage", None):
        usage = ch.usage
    if ch.choices and ch.choices[0].text:
        if first is None:
            first = time.perf_counter()
        n += 1
end = time.perf_counter()
# Chunks are not tokens under speculative decoding. Require server usage.
if usage is None:
    raise RuntimeError("Missing server-reported token usage")
n_tok = usage.completion_tokens
tg = (n_tok - 1) / (end - first) if (first and n_tok > 1) else 0.0
ttft = 1000 * (first - t0) if first else 0.0
pt = usage.prompt_tokens if usage else None
pp = (pt / (ttft / 1000)) if pt and ttft else 0.0
print(f"ctx~{CTX:<6} prompt_tok={pt} pp={pp:7.1f} t/s | decode={tg:6.1f} t/s | ttft={ttft:7.0f} ms | out_tok={n_tok}", flush=True)
