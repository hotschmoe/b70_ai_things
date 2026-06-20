#!/usr/bin/env python3
"""Concurrent serving-throughput probe: fire N parallel streaming completions and measure the
AGGREGATE decode t/s (the real serving capacity, vs perf_probe's single-stream latency).
At batch N the decode GEMM reads the weights ONCE for all N sequences -> aggregate should scale
above single-stream until compute/SLM-bound. Usage:
  concurrent_probe.py <base_url> <model> <N> [max_tokens=128]
"""
import asyncio, sys, time
from openai import AsyncOpenAI

BASE, MODEL = sys.argv[1], sys.argv[2]
N = int(sys.argv[3])
M = int(sys.argv[4]) if len(sys.argv) > 4 else 128
PROMPT = ("Write a thorough, detailed technical essay about the history and architecture of modern "
          "GPUs, covering memory hierarchies, systolic arrays, and quantization. Be comprehensive.")
client = AsyncOpenAI(base_url=BASE, api_key="x", timeout=300.0)

async def one(idx):
    first = None; n = 0
    stream = await client.chat.completions.create(
        model=MODEL, messages=[{"role": "user", "content": PROMPT}],
        max_tokens=M, temperature=0.7, stream=True)
    async for ch in stream:
        if ch.choices and ch.choices[0].delta and ch.choices[0].delta.content:
            if first is None:
                first = time.perf_counter()
            n += 1
    return first, time.perf_counter(), n

async def main():
    res = await asyncio.gather(*[one(i) for i in range(N)])
    res = [r for r in res if r[0] is not None and r[2] > 1]
    if not res:
        print(f"N={N} FAILED (no streams)"); return
    total_decode_tok = sum(r[2] - 1 for r in res)        # exclude each stream's first token
    window = max(r[1] for r in res) - min(r[0] for r in res)   # earliest first-tok -> latest done
    agg = total_decode_tok / window
    per = [(r[2] - 1) / (r[1] - r[0]) for r in res if r[1] > r[0]]
    mean_per = sum(per) / len(per)
    print(f"N={N:<2} streams_ok={len(res):<2} | AGGREGATE decode={agg:6.1f} t/s | "
          f"mean_per_stream={mean_per:5.1f} t/s | total_tok={total_decode_tok} window={window:.2f}s", flush=True)

asyncio.run(main())
