#!/usr/bin/env python3
"""Long-context decode probe via the CHAT endpoint (the raw-completions longctx_probe returns 0 tokens on
reasoning models). Builds an ~CTX-token user prompt, streams the chat response, reports decode t/s + TTFT
AT that context length. Usage: longctx_chat_probe.py <base_url> <model> <ctx_tokens> [out=64]
"""
import sys, time
from openai import OpenAI

BASE, MODEL = sys.argv[1], sys.argv[2]
CTX = int(sys.argv[3])
OUT = int(sys.argv[4]) if len(sys.argv) > 4 else 64
client = OpenAI(base_url=BASE, api_key="x", timeout=600.0)

UNIT = "The history of computing is long and full of surprising turns. "  # ~12 tokens
filler = UNIT * max(1, CTX // 12)
msg = filler + "\n\nIn one short sentence, what is the above text about?"

t0 = time.perf_counter(); first = None; n = 0
s = client.chat.completions.create(
    model=MODEL, messages=[{"role": "user", "content": msg}],
    max_tokens=OUT, temperature=0, stream=True)
for ch in s:
    if ch.choices and ch.choices[0].delta and ch.choices[0].delta.content:
        if first is None:
            first = time.perf_counter()
        n += 1
end = time.perf_counter()
tg = (n - 1) / (end - first) if (first and n > 1) else 0.0
ttft = 1000 * (first - t0) if first else 0.0
print(f"ctx~{CTX:<6} decode={tg:6.1f} t/s | ttft={ttft:8.0f} ms | out_tok={n}", flush=True)
