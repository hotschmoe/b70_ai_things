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
  sglang/<model-quant>/serve.sh
  vllm/<model-quant>/serve.sh
```

The shelf currently has entries for:

| backend | entries | role |
|---|---|---|
| vLLM | 27B int4, NVFP4, W4A16, W4A8, W8A8; 35B-A3B int4 and W8A8 | Current NVFP4 daily driver and measured baselines |
| sglang | 27B int4, W4A8, W8A8; 35B-A3B W8A8 | Primary backend for new true-W8A8 serving research |

## Current headline entry

`vllm/qwen36-27b-nvfp4/serve.sh` contains two measured settings in one
shelf entry:

- Default `TP=1`: vLLM 0.25.1, one 100,352-token replica. Run one per
  card behind nginx for the DP=2 daily driver. Each replica uses captured
  MTP5, calibrated fp8 KV, embed INT8, native E4M3 decode scales, and
  working prefix reuse. Measured 64.6 code tok/s per card.
- `TP=2`: vLLM 0.26.0, one 200,000-token server across both cards.
  It adds graph push-all-reduce and a 16,384-token prefill chunk.
  Qualified 2026-07-27 with 18/18 plus two 36/36 coherence gates, exact
  190,048-token retrieval cold and warm, and a 52K-token concurrent soak.

Use DP=2 for aggregate throughput and fault isolation. Use TP=2 when one
request needs more than 100,352 tokens.

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
