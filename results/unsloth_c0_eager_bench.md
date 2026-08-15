# Card 0 EAGER Unsloth baseline (2026-08-15)

Live serve was already up. This run did not restart it, did not take card 1,
and did not call gpu-run.

## config

- container: unsloth_c0
- image: vllm-xpu-env:int8g-v0260
- host: http://127.0.0.1:8078
- served id: qwen3.8-27b-NVFP4-unsloth
- weights: /models/qwen3.8-27b/nvfp4-unsloth
- TP=1 CARD=0 MODE=fused GRAPH=0 (enforce_eager) MTP off MAXLEN=8192 max_num_seqs=4
- sitecustomize (1e): channel-FP8 -> INT8-XMX int8_gemm_w8a16 (B70_FP8_CHANNEL_INT8XMX=1)
- NVFP4 MLP: fused nvfp4_gemm_w4a16
- bench: IN=2048 OUT=128 CONC=1 then CONC=4 via bin/35_sweep_bench.sh
- vLLM 0.26 bench flags accepted; no curl fallback needed

Serve args (from docker logs, APIServer non-default):

```
model=/models/qwen3.8-27b/nvfp4-unsloth
host=0.0.0.0 port=8078
dtype=bfloat16 max_model_len=8192 enforce_eager=True
served_model_name=qwen3.8-27b-NVFP4-unsloth
reasoning_parser=qwen3 gpu_memory_utilization=0.9
enable_prefix_caching=False skip_mm_profiling=True max_num_seqs=4
```

## command

1. Health / models (no restart):

```
curl -sS -m 10 -o /tmp/health_8078.out -w 'http=%{http_code} bytes=%{size_download}\n' \
  http://127.0.0.1:8078/health
curl -sS -m 10 http://127.0.0.1:8078/v1/models | python3 -m json.tool
```

2. Coherence:

```
PROBE_HOST=http://127.0.0.1:8078 python3 vllm/nvfp4/kv_gate.py
```

Chat thinking-off Paris:

```
curl -sS -m 180 http://127.0.0.1:8078/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.8-27b-NVFP4-unsloth",
    "messages": [{"role": "user", "content": "What is the capital of France? Answer with one word."}],
    "max_tokens": 32,
    "temperature": 0,
    "chat_template_kwargs": {"enable_thinking": false}
  }'
```

3. Bench against the live container (c1 first, then a clean c4 rerun after a
   wrapper timeout orphaned an earlier c4 client):

```
cd /mnt/vm_8tb/github/b70_ai_things
IN=2048 OUT=128 PORT=8078 CONC="1 4" \
  bash bin/35_sweep_bench.sh unsloth_c0 qwen3.8-27b-NVFP4-unsloth unsloth-c0-eager-int8xmx \
  /models/qwen3.8-27b/nvfp4-unsloth
```

c1 completed in that first sweep. The parent was SIGTERM'd at the 5 min
wrapper limit while c4 was still inside the container; that leftover plus one
more orphaned c4 were allowed to drain (not stacked). Official c4:

```
IN=2048 OUT=128 PORT=8078 CONC="4" \
  bash bin/35_sweep_bench.sh unsloth_c0 qwen3.8-27b-NVFP4-unsloth unsloth-c0-eager-int8xmx \
  /models/qwen3.8-27b/nvfp4-unsloth
```

unsloth_c0 was left running.

## result

### health / models

- GET /health -> HTTP 200, 0-byte body
- GET /v1/models id = qwen3.8-27b-NVFP4-unsloth, max_model_len = 8192, root = /models/qwen3.8-27b/nvfp4-unsloth
- docker ps: unsloth_c0 Up, image vllm-xpu-env:int8g-v0260, 0.0.0.0:8078->8078
- after benches: still HTTP 200, same served id, container still Up (not stopped)

### coherence

kv_gate.py (PROBE_HOST=http://127.0.0.1:8078), served: qwen3.8-27b-NVFP4-unsloth

```
[PASS] capital-of-France -> 'Paris.'
[PASS] 17+26=43 -> ...' + 2 + 1 = 4.\n3. Combine the results: 43.\n\nSo, the final answer is 43.'
[PASS] gold=Au -> 'Au, derived from the Latin'
GATE: 3/3 PASS
```

Chat thinking-off Paris (enable_thinking false, temp 0, max_tokens 32):

- message.content exact string: Paris
- reasoning: null
- finish_reason: stop
- usage: prompt_tokens=24 completion_tokens=2
- system_fingerprint: vllm-0.26.0-fe7b3f84

### docker logs: weights / KV

From EngineCore at boot (container start ~20:41 UTC):

- Checkpoint size: 21.81 GiB
- Loading weights took 7.12 seconds
- Model loading took 24.7 GiB memory and 9.801045 seconds
- Available KV cache memory: 1.9 GiB
- GPU KV cache size: 34,588 tokens
- Maximum concurrency for 8,192 tokens per request: 4.22x
- Actual usage: 24.7 GiB for weight, 0.32 GiB peak activation, 0.34 GiB non-torch, 0.0 GiB CUDAGraph
- Current kv cache memory in use: 1.9 GiB

Shim (matches the claimed 1e + fused path):

```
[nvfp4-shim] MODE=fused: uniform NVFP4 (W4A4) now uses fused nvfp4_gemm_w4a16 (act fake-quant skipped)
[nvfp4-shim] (1e) channel-FP8 -> INT8-XMX int8_gemm_w8a16 (... B70_FP8_CHANNEL_INT8XMX=1)
```

### bench (vllm bench serve, random, ignore-eos)

PP derived as IN / (mean_ttft_ms/1000). c4 mean TTFT includes queueing under
max_num_seqs=4, so do not treat c4 PP as a prefill-rate number.

| conc | req/s | out tok/s (agg TG) | mean TTFT ms | mean TPOT ms | per-stream TG | PP from TTFT |
| ---- | ----- | ------------------ | ------------ | ------------ | ------------- | ------------ |
| 1    | 0.06  | 7.80               | 1571.71      | 116.85       | 8.56          | 1303.0       |
| 4    | 0.19  | 24.23              | 4115.82      | 133.85       | 7.47          | n/a (queued) |

CSV paths:

- combined: /mnt/vm_8tb/b70/results/sweep_unsloth-c0-eager-int8xmx_combined.csv
- c1 only: /mnt/vm_8tb/b70/results/sweep_unsloth-c0-eager-int8xmx_20260815_204703.csv
- c4 only: /mnt/vm_8tb/b70/results/sweep_unsloth-c0-eager-int8xmx_20260815_205605.csv

Headline:

- c1 TTFT = 1571.71 ms
- c1 PP = 1303.0 tok/s (2048 / 1.57171 s)
- c1 TG = 8.56 tok/s per stream (agg out 7.80 tok/s)
- c4 TG = 24.23 tok/s aggregate (7.47 tok/s per stream)

### docker logs: generation throughput during bench

c1 window (20:47-20:49, Running: 1, KV usage 13.5%):

- decode-only ticks (prompt throughput 0.0): gen 8.3 / 8.4 / 8.6 / 8.6 / 8.1 tok/s
- mixed ticks (prompt ~204.8 = one 2048-token prefill in the 10s bucket): gen 7.2-7.5 tok/s
- matches bench per-stream 8.56 and agg 7.80

official c4 window (20:56:21-20:59:11, Running: 4, KV usage 54.1%):

- decode-only ticks (prompt 0.0, 4 reqs): gen 31.8 / 33.6 / 34.0 / 34.0 / 34.0 tok/s
- mixed prefill+decode ticks: gen 13.6-22.6 with prompt 614-819 tok/s
- bench agg TG 24.23 sits between those mixed and decode-only 10s buckets (includes prefills)

Note: two unofficial/orphaned c4 clients also ran ~20:49:51-20:52:41 and
~20:53:21-20:56:11 after the first wrapper kill. Server-log gen there was the
same shape (decode-only ~33-34 tok/s at 4-wide). Official numbers above are
from the captured 35_sweep_bench CSVs only.

## verdict

Card 0 EAGER Unsloth NVFP4 is coherent and measurable on the already-up
serve. kv_gate 3/3. Thinking-off chat content is exactly "Paris". Weights
24.7 GiB resident, KV 34588 tokens / 1.9 GiB, 4.22x at 8192. IN=2048/OUT=128
eager baseline: c1 TTFT 1571.71 ms, PP 1303 tok/s, TG 8.56 tok/s; c4 agg TG
24.23 tok/s (~3.1x c1 agg, per-stream 7.47). Server decode-only peaks ~8.6
tok/s at c1 and ~34 tok/s at c4. Serve left running on :8078.
