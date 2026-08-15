# Unsloth NVFP4 card-1 GRAPH=1 A/B (PIECEWISE)

Date: 2026-08-15
Owner: card 1 only. unsloth_c0 on card 0 / :8078 was not touched.
No JOURNAL.md edit. No commit.

## config

- card: 1 (gpu-run --card 1)
- container: unsloth_c1_graph
- port: 8079
- image: vllm-xpu-env:int8g-v0260
- model: models/files/qwen3.8-27b/nvfp4-unsloth
- MODE=fused GRAPH=1 CGMODE=PIECEWISE IGP=false
- CAPSIZES=1,2,4,8
- MAXLEN=4096 UTIL=0.85 MAXSEQS=8 PREFIXCACHE=0
- MTPTOK= (empty; spec off)
- REASONPARSER=qwen3
- SERVED requested: qwen3.8-27b-NVFP4-unsloth-graph
- SERVED actual: qwen3.8-27b-NVFP4-unsloth-graph-graph
  (serve_nvfp4_27b.sh appends -graph when GRAPH=1)

## command

Launched via gpu-run --card 1 (lease held by docker wait):

```
cd /mnt/vm_8tb/github/b70_ai_things
TP=1 CARD=1 PORT=8079 NAME=unsloth_c1_graph MODE=fused GRAPH=1 \
  CGMODE=PIECEWISE IGP=false CAPSIZES=1,2,4,8 \
  MAXLEN=4096 UTIL=0.85 MAXSEQS=8 PREFIXCACHE=0 \
  IMG=vllm-xpu-env:int8g-v0260 \
  MODEL_REL=qwen3.8-27b/nvfp4-unsloth \
  SERVED=qwen3.8-27b-NVFP4-unsloth-graph \
  MTPTOK= REASONPARSER=qwen3 \
  ./bin/gpu-run --card 1 bash -c '
    TP=1 CARD=1 PORT=8079 NAME=unsloth_c1_graph MODE=fused GRAPH=1 \
    CGMODE=PIECEWISE IGP=false CAPSIZES=1,2,4,8 \
    MAXLEN=4096 UTIL=0.85 MAXSEQS=8 PREFIXCACHE=0 \
    IMG=vllm-xpu-env:int8g-v0260 \
    MODEL_REL=qwen3.8-27b/nvfp4-unsloth \
    SERVED=qwen3.8-27b-NVFP4-unsloth-graph \
    MTPTOK= REASONPARSER=qwen3 \
    bash vllm/nvfp4/serve_nvfp4_27b.sh
    docker wait unsloth_c1_graph
  '
```

Health wait: http://127.0.0.1:8079/health -> 200 in ~2.5 min.
No OOM / UR err 40. No MAXLEN/UTIL retry.

Gate:

```
PROBE_HOST=http://127.0.0.1:8079 python3 vllm/nvfp4/kv_gate.py
```

Chat (thinking-off):

```
POST http://127.0.0.1:8079/v1/chat/completions
model=qwen3.8-27b-NVFP4-unsloth-graph-graph
messages=[{role:user, content:"What is the capital of France? Answer with one word."}]
max_tokens=32 temperature=0
chat_template_kwargs.enable_thinking=false
```

Bench (coherent, so ran):

```
IN=2048 OUT=128 PORT=8079 CONC="1 4" \
  bash bin/35_sweep_bench.sh unsloth_c1_graph \
  qwen3.8-27b-NVFP4-unsloth-graph-graph \
  unsloth-c1-graph-int8xmx \
  /models/qwen3.8-27b/nvfp4-unsloth
```

csv: /mnt/vm_8tb/b70/results/sweep_unsloth-c1-graph-int8xmx_20260815_205012.csv

## result

### boot / logs

- [nvfp4-shim] (1e) channel-FP8 -> INT8-XMX int8_gemm_w8a16 (wrapped
  ['_ChannelAwareXPUW8A16FP8LinearKernel', '_ChannelAwareXPUW8A8FP8LinearKernel'];
  B70_FP8_CHANNEL_INT8XMX=1); per-tensor FP8 stays on fp8_gemm_w8a16
- Using _XPUW4A4FusedAsW4A16Kernel for NVFP4 GEMM
- cudagraph_mode=PIECEWISE, cudagraph_capture_sizes=[1, 2, 4, 8]
- Graph capturing finished in 2 secs, took 0.22 GiB
- Model loading took 24.7 GiB
- Available KV cache memory: 0.3 GiB
- GPU KV cache size: 4,096 tokens
- Actual usage: 24.7 GiB weight, 0.56 GiB peak activation, 0.19 GiB non-torch,
  0.22 GiB CUDAGraph. Desired util 0.85 = 25.75 GiB of 30.3 GiB card.
- speculative_config=None (MTP off)
- no OOM, no UR err 40, no DEVICE_LOST

### kv_gate

served: qwen3.8-27b-NVFP4-unsloth-graph-graph
- [PASS] capital-of-France -> 'Paris.'
- [PASS] 17+26=43 -> '...Combine the results: 43. So, the final answer is 43.'
- [PASS] gold=Au -> 'Au, derived from the Latin'
GATE: 3/3 PASS

### chat thinking-off Paris

finish_reason: stop
reasoning: None
usage: prompt=24 completion=2
exact content: Paris

Coherent: yes. "Paris" with no !!!! loop.

### IN=2048 OUT=128 bench

| conc | req/s | out tok/s | mean TTFT ms | mean TPOT ms | per-stream decode t/s |
|-----:|------:|----------:|-------------:|-------------:|----------------------:|
| 1    | 0.15  | 18.87     | 1388.85      | 42.48        | 23.54                 |
| 4    | 0.15  | 18.86     | 20465.83     | 42.54        | 23.51                 |

c4 aggregate equals c1: KV is only 4096 tokens, so 4x IN=2048 cannot run
in parallel (one 2048+128 request already uses 2176 tokens). TTFT at conc 4
is queue delay, not a faster prefill.

### leftover state

- unsloth_c1_graph left running on :8079 (healthy, http 200 after bench)
- unsloth_c0 still Up on :8078 (untouched)
- card 1 lease still HELD by gpu-run docker wait

## verdict

HEALTHY. GRAPH=1 PIECEWISE capture succeeded first try at MAXLEN=4096 UTIL=0.85.
INT8-XMX channel-FP8 path and fused W4A16 kernel both active. Coherent
("Paris", kv_gate 3/3). No OOM.

c1 IN=2048: TTFT 1.39 s, PP implied ~1475 tok/s (2048/1.389), TG 23.54 t/s.
c4 IN=2048: TTFT 20.47 s (KV-serialized), TG 23.51 t/s, no concurrency win.

Caveat: 24.7 GiB weights + 0.22 GiB graphs leave only 0.3 GiB / 4096 KV tokens
at UTIL=0.85. Fine for single-stream 2048, not for concurrent 2048.
Served id is ...-graph-graph because the recipe already included -graph and
the script appends another.
