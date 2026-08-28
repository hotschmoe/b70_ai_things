# Current-stack serving speed targets - 2026-08-27

## Outcome

All three requested aggregate serving-capacity targets have current-stack
measurements. The requested single-stream TP=2 objective is stricter and is
tracked separately below. These are research controls, not interchangeable
workload claims or live-shelf promotions.

| Model and scheme | Target | Qualified result | Measured regime |
| --- | ---: | ---: | --- |
| Qwen3.8-27B W8A8 GPTQ | 25 tok/s | 37.6 tok/s | SGLang TP2, c4, supported 300-second soak |
| Qwen3.8-27B RadixArk NVFP4 | 40 tok/s | 78.07 tok/s | vLLM 0.28, TP1, c4, 32 forced p512/o512 requests |
| Ornith-1.5-35B-A3B W8A8 RTN | 65 tok/s | 87.7954 tok/s median | SGLang TP2, c4, 12 forced p515/o512 batches |

The numbers above are aggregate output throughput. Prompt/output shape,
concurrency, tensor parallelism, and backend differ, so they show that each
capacity objective is met in its qualified regime; they are not a model-to-model
speed ranking.

## Single-stream TP=2 status

| Model and scheme | Target | Current median | Matched request shape | Status |
| --- | ---: | ---: | --- | --- |
| Qwen3.8-27B W8A8 GPTQ | 25 tok/s | 22.4513 tok/s | p515/o512, c1 | 10.2% short |
| Qwen3.8-27B RadixArk NVFP4 plus FP8 W8A16 decode | 40 tok/s | 32.6206 tok/s | p879/o512, c1 | 18.4% short |
| Ornith-1.5-35B-A3B W8A8 RTN | 65 tok/s | 62.0426 tok/s | p837/o512, c1 | 4.5% short |

The Qwen NVFP4 row is the current SGLang FULL-graph TP=2 winner. It improves
the matched stock static-FP8 activation route from 30.1665 tok/s by 8.14%.
The Ornith row is the tuned-MoE arm; its 62.0426 tok/s median is statistically
indistinguishable from the 61.9480 tok/s stock-MoE control. None of these
single-stream rows meets its requested target yet.

## Ornith sustained-speed fix

The refreshed target-only breakable graph initially measured above 80 tok/s,
then degraded monotonically to 56.7261 tok/s by batch 12. This matched the
known Level Zero executable command-list accumulation under repeated
`torch.xpu.XPUGraph.replay()`.

The retained vLLM workaround was deliberately ported into
`sglang/refresh/b70_xpu_w8a8.py`. With `B70_XPU_CG_RECLAIM=500`, every graph
keeps its modifiable representation and re-instantiates the executable at the
500-replay boundary. The matched candidate measured:

- 12 batch rates in the 86.8321-89.3556 tok/s range;
- 87.7954 tok/s median post-first aggregate throughput;
- 86.1822 tok/s median aggregate throughput including TTFT;
- 88.6186 first batch and 88.1472 final batch, a -0.53 percent delta;
- 48/48 successful streams and exactly 24,576 forced output tokens;
- byte-identical repeated greedy output and 4/4 concurrent arithmetic canaries.

The launcher defaults reclaim to 500 only for its breakable graph route. Set
`CG_RECLAIM=0` to reproduce the rejected control.

## Boundaries

- Ornith p4172/o128 at c8 measured only 39.2342 tok/s median post-first because
  long chunked prefill serialized the batch. Do not generalize the p515/o512
  number to long-prefill traffic.
- Ornith Shisa MTP remains rejected because its greedy output is not
  target-exact on the refreshed stack.
- The 78.07 tok/s Qwen NVFP4 capacity result is TP1 at c4. The later SGLang
  FULL-graph TP2 single-stream route is coherent and reaches 32.6206 tok/s,
  but still misses the 40 tok/s single-stream objective.
- Qwen NVFP4 concurrent continuations can diverge after coherent common
  prefixes at 24 and 64 tokens. Only the measured 8-token serial/concurrent
  canary was byte-identical on all four streams.
- Every accepted GPU run used `bin/gpu-run`, exact model identity, coherence,
  graceful teardown, and applicable post-card/compiled-collective health.

Full command and artifact evidence is in JOURNAL.md entries 2026-08-26v,
2026-08-27b, 2026-08-27c, and 2026-08-28a through 2026-08-28c.
