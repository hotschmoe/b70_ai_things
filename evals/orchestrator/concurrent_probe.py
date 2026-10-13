#!/usr/bin/env python3
"""Concurrent serving-throughput probe: fire N parallel streaming completions and measure the
AGGREGATE decode t/s (the real serving capacity, vs perf_probe's single-stream latency).
At batch N the decode GEMM reads the weights ONCE for all N sequences -> aggregate should scale
above single-stream until compute/SLM-bound. Usage:
  concurrent_probe.py <base_url> <model> <N> [max_tokens=128]
"""
import asyncio, os, sys, time, uuid
from openai import AsyncOpenAI

BASE, MODEL = sys.argv[1], sys.argv[2]
N = int(sys.argv[3])
M = int(sys.argv[4]) if len(sys.argv) > 4 else 128
PROMPT = ("Write a thorough, detailed technical essay about the history and architecture of modern "
          "GPUs, covering memory hierarchies, systolic arrays, and quantization. Be comprehensive.")
client = AsyncOpenAI(base_url=BASE, api_key=os.environ.get("OPENAI_API_KEY", "x"), timeout=300.0)

async def one(idx):
    t0 = time.perf_counter(); first = None; usage = None
    stream = await client.chat.completions.create(
        model=MODEL, messages=[{"role": "user", "content": f"[probe {uuid.uuid4().hex}] " + PROMPT}],
        max_tokens=M, temperature=0.7, stream=True,
        stream_options={"include_usage": True})
    async for ch in stream:
        if getattr(ch, "usage", None):
            usage = ch.usage
        if ch.choices:
            delta = ch.choices[0].delta
            if delta and (delta.content or getattr(delta, "reasoning", None) or getattr(delta, "reasoning_content", None)):
                if first is None:
                    first = time.perf_counter()
    end = time.perf_counter()
    if usage is None or first is None or usage.completion_tokens <= 1:
        raise RuntimeError("Missing server usage or generated tokens")
    return t0, first, end, usage.prompt_tokens, usage.completion_tokens

async def main():
    res = await asyncio.gather(*[one(i) for i in range(N)])
    total_decode_tok = sum(r[4] - 1 for r in res)
    window = max(r[2] for r in res) - min(r[1] for r in res)
    agg = total_decode_tok / window
    per = [(r[4] - 1) / (r[2] - r[1]) for r in res]
    mean_per = sum(per) / len(per)
    print(f"N={N:<2} streams_ok={len(res):<2} | AGGREGATE decode={agg:6.1f} t/s | "
          f"mean_per_stream={mean_per:5.1f} t/s | total_tok={total_decode_tok} window={window:.2f}s", flush=True)
    for i, r in enumerate(res):
        ttft = r[1] - r[0]
        print(f"stream={i} prompt_tok={r[3]} pp={r[3]/ttft:.1f} t/s | decode={per[i]:.1f} t/s | "
              f"ttft={ttft*1000:.0f} ms | out_tok={r[4]}", flush=True)

asyncio.run(main())
