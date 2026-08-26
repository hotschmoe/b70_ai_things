# rdy_to_serve -- retained B70 serving controls

Each directory preserves exactly one best measured pre-refresh configuration
for a `(backend, model, quant)` tuple. Backend-specific code stays under its
backend root; shared lifecycle helpers live in `_common/lib.sh`.

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

After the 2026-08-26 cleanup, the retained shelf has entries for:

| backend | entries | role |
|---|---|---|
| vLLM | Qwen3.6-27B NVIDIA NVFP4; Qwen3.8-27B RadixArk NVFP4 | Maintained measured baselines |

## Current headline entry

No daily-driver server was active during the 2026-08-26 cleanup. The retained
Qwen3.6 NVFP4 shelf (`vllm/qwen36-27b-nvfp4/serve.sh`) is the measured
high-aggregate baseline:

The old ABI-specific native libraries were quarantined, so neither retained
entry is currently a runnable production shelf. Rebuild against the refreshed
stack and pass the promotion gate before use.

- Default `TP=1`: vLLM 0.25.1, one 100,352-token replica (DP=2 behind nginx was the prior DD).
  Captured MTP5, calibrated fp8 KV, embed INT8, ~64.6 code tok/s per card.
- `TP=2`: vLLM 0.26.0, one 200,000-token server, graph push-AR, 640k KV tokens.
  Qualified 2026-07-27 (18/18 + 36/36, exact 190k retrieval).

The Qwen3.8 NVFP4 shelf is
`vllm/qwen38-27b-nvfp4/serve.sh`. Qwen3.8 W8A8 and the Ornith family remain
active research weights but do not yet have sweep-qualified shelf entries.

## Use after rebuild

Hold the appropriate `bin/gpu-run` lease for the full start, test, and stop
sequence. Common knobs include `PORT`, `NAME`, `MAXLEN`, `MAXSEQS`, and `UTIL`.
Served IDs must identify the method and scheme.

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
