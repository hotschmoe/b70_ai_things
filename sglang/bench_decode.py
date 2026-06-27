#!/usr/bin/env python3
# bench_decode.py -- single-stream decode bench against an OpenAI-compatible server (sglang/vllm).
# Measures TTFT and steady-state decode tok/s on ONE streaming request (matches how the vLLM int4
# baseline ~30 t/s was captured). Run on a QUIET server (no other load).
#   usage: bench_decode.py [port] [maxtok] [model]
import json, sys, time, urllib.request
PORT  = int(sys.argv[1]) if len(sys.argv) > 1 else 30000
MAX   = int(sys.argv[2]) if len(sys.argv) > 2 else 256
MODEL = sys.argv[3] if len(sys.argv) > 3 else 'qwen36-27b-bf16-sglang'
body = {
    'model': MODEL,
    'messages': [{'role': 'user', 'content': 'Write a detailed 400-word explanation of how a four-stroke engine works.'}],
    'max_tokens': MAX, 'temperature': 0.7, 'stream': True,
}
req = urllib.request.Request('http://127.0.0.1:%d/v1/chat/completions' % PORT,
    data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
t0 = time.time(); ttft = None; n = 0; t_last = None
with urllib.request.urlopen(req, timeout=300) as resp:
    for raw in resp:
        line = raw.decode('utf-8', 'ignore').strip()
        if not line.startswith('data:'):
            continue
        data = line[5:].strip()
        if data == '[DONE]':
            break
        try:
            d = json.loads(data)
        except Exception:
            continue
        delta = (d.get('choices') or [{}])[0].get('delta', {})
        if delta.get('content'):
            now = time.time()
            if ttft is None:
                ttft = now - t0
            n += 1
            t_last = now
decode_s = (t_last - t0 - ttft) if (t_last and ttft is not None) else 0
toks_after_first = max(n - 1, 0)
rate = toks_after_first / decode_s if decode_s > 0 else float('nan')
print('port=%d model=%s' % (PORT, MODEL))
print('  content chunks: %d   TTFT: %.3f s' % (n, ttft if ttft else float('nan')))
print('  steady-state decode: %.2f tok/s  (%d tok over %.2f s)' % (rate, toks_after_first, decode_s))
