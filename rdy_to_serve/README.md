# rdy_to_serve -- verified B70 serving shelf

This is the golden path for serving. Each directory contains exactly one best
measured configuration for a `(backend, model, quant)` tuple. Backend-specific
code stays under its backend root; shared lifecycle helpers live in
`_common/lib.sh`.

Do not reconstruct a serve from old JOURNAL entries. Start from the matching
`serve.sh`, and do not promote an unmeasured option.

## Layout

```text
rdy_to_serve/
  _common/lib.sh
  llamacpp/<model-quant>/serve.sh
  sglang/<model-quant>/serve.sh
  vllm/<model-quant>/serve.sh
```

The shelf currently has entries for:

| backend | entries | role |
|---|---|---|
| llama.cpp | Qwen3.8-27B OBLITERATED Q4_K_M | Current large-context DP=2 daily driver |
| vLLM | 27B int4, NVFP4, W4A16, W4A8, W8A8; 35B-A3B int4 and W8A8 | Maintained paused baselines |
| sglang | 27B int4, W4A8, W8A8; 35B-A3B W8A8 | Research backend for true-W8A8 / long-prefill work |

## Current headline entry

**Daily driver (2026-08-23):**
`llamacpp/qwen38-27b-obliterated-q4km/serve.sh`, two independent one-card
replicas behind nginx, Q8_0 KV, 245760 context per replica, and the V3 GGUF's
embedded MTP head at draft max 3. It serves `hotschmoe-dd` on port 18080.
Measured aggregate decode is 81.86 tok/s; a five-minute c4 mixed-load soak was
338/338 coherent with zero errors or degenerate responses. A real 152289-token
cold request completed coherently with no context shift or truncation.

The previous vLLM Qwen3.6 W8A8 and NVFP4 entries remain maintained baselines.

**NVFP4 shelf** (`vllm/qwen36-27b-nvfp4/serve.sh`) remains the measured high-agg alternative:

- Default `TP=1`: vLLM 0.25.1, one 100,352-token replica (DP=2 behind nginx was the prior DD).
  Captured MTP5, calibrated fp8 KV, embed INT8, ~64.6 code tok/s per card.
- `TP=2`: vLLM 0.26.0, one 200,000-token server, graph push-AR, 640k KV tokens.
  Qualified 2026-07-27 (18/18 + 36/36, exact 190k retrieval).

## Usage

Run locally from the repository and hold the appropriate GPU lease for the
entire serve/test/stop sequence:

```bash
cd /mnt/vm_8tb/github/b70_ai_things

# One-card default on card 0.
./bin/gpu-run --card 0 bash -c '
  CARD=0 PORT=18079 NAME=test_nvfp4 \
    bash rdy_to_serve/vllm/qwen36-27b-nvfp4/serve.sh start
  # Run probes here.
  docker stop -t 60 test_nvfp4'

# One-request 200K TP=2 mode.
./bin/gpu-run bash -c '
  TP=2 PORT=18079 NAME=test_nvfp4_tp2 \
    bash rdy_to_serve/vllm/qwen36-27b-nvfp4/serve.sh start
  # Run probes here.
  docker stop -t 60 test_nvfp4_tp2'
```

Common knobs include `PORT`, `NAME`, `MAXLEN`, `MAXSEQS`, `UTIL`, and
backend-specific settings documented in each wrapper. Served IDs must
identify the method and scheme unless the daily-driver orchestrator
deliberately forces the stable `hotschmoe-dd` alias.

## Promotion gate

A shelf change must be at least as fast and coherent under concurrent
prefill plus decode. Validate model identity, KV scheme, a mixed-load
coherence sweep, performance, and a bounded soak before promotion.

Any change to `bin/` or `rdy_to_serve/_common/` additionally requires:

```bash
bin/serve-sweep --smoke
```

The current TP=2 NVFP4 qualification driver is
`vllm/nvfp4/tp2_longctx_qualify.sh`; profiler capture is
`vllm/nvfp4/profile_tp2_v0260.sh`.
