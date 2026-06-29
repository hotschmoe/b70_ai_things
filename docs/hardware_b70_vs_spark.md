# Hardware decision: 4x Intel Arc Pro B70 vs 1x NVIDIA DGX Spark

Date: 2026-06-27
Verdict: STAY ON B70, scale to 4 cards. Do not buy the DGX Spark.

## Question

For ~$4,000, serving Qwen3.6-27B (quantized) for a mixed workload -- single-user
interactive Hermes agent (decode/latency-bound) plus occasional concurrent
eval/harness load (pi.dev, omp.sh) -- is the money better spent on:

- 4x Intel Arc Pro B70 (32 GB GDDR6, 608 GB/s each; own 2, returnable $950/card), or
- 1x NVIDIA DGX Spark (GB10, 128 GB unified LPDDR5X, 273 GB/s, ~1 PFLOP FP4)?

## Verified numbers (community-reported, 3-0 adversarial verification)

Dense Qwen3.6-27B single-stream decode, the exact target model:

| Platform / stack              | decode t/s (b1) | concurrent decode     | prefill t/s | stable?      |
|-------------------------------|-----------------|-----------------------|-------------|--------------|
| Spark NVFP4 / vLLM            | 22.2            | 46.9 (c2), 102.7 (c5) | ~820        | yes          |
| Spark Q4_K_M / llama.cpp      | 11.85           | --                    | 823         | yes          |
| Spark Mistral-24B FP8 (b1)    | 8.8             | 319 (b128)            | --          | yes          |
| B70 int4+MTP / sglang         | ~15             | TBD (early)           | --          | yes-ish      |
| B70 int4+MTP / vLLM PIECEWISE | 55              | 134 (C8)              | 1520        | NO (crashes) |

- Spark hard ceiling: 273 GB/s caps dense-27B single-stream near 18-22 t/s. It will
  never decode dense 27B fast; that is locked by LPDDR5X. Strengths: 128 GB unified
  (runs 70B-120B), strong FP4 prefill, excellent batching (Llama-8B FP8: 20.5 -> 368
  t/s, b1 -> b32).
- B70 ceiling: 608 GB/s per card, dense-27B bandwidth ceiling ~30-65 t/s/card. Real
  silicon confirmed (vLLM PIECEWISE hit 30-55) but not yet stable in production; our
  bankable number today is ~15 t/s (int4+MTP on sglang).

Sources: spark-arena.com/leaderboard, github/nabe2030/dense-27b-31b-dgx-spark,
lmsys.org 2025-10-13, storagereview.com DGX Spark review, intuitionlabs.ai,
vllm.ai/blog 2025-11-11 (Arc Pro B-series). B70 rows: this repo + owner report.

## Decision factors (2026-06-27)

1. B70 cards confirmed 32 GB / 608 GB/s, brand new; we are leading public
   testing/benchmarking on them (near-zero community data exists -- we are it).
2. Target is dense 27B with LONG CONTEXT, good prefill, decent accuracy -- NOT
   70B-120B. The Spark's one unique advantage (128 GB unified) is not needed.
3. Owner enjoys the research/tinkering and intends to contribute community tooling
   and numbers -- the Intel-stack immaturity is accepted cost, not a blocker.
4. Used 3x RTX 3090 (the decode-per-dollar king at 124 t/s) is out: scarce,
   $1500-1900 each, often faulty from resellers.
5. push_allreduce patch already made TP=2 much more viable; large optimization
   surface remains.

Given (1)-(5), the Spark's edge (mature stack, ships today at 22 t/s stable) loses
to the B70's higher realizable ceiling on a path the owner wants to walk. The Spark
would be correct only if the goal were "ship fast, stop tinkering" or "need 70B+ in
one box." Neither holds.

## Verdict

Stay on B70. Add 2 cards (-> 4x B70, ~$3,800). Do not buy the Spark.

## Optimization roadmap

Goal: W8A8 TP=2 to ~20+ t/s decode -- hits accuracy + long-context + prefill levers
at once. Current W8A8-sqgptq TP=2: 18.10 captured no-MTP, 26.10 captured + MTP
spec=5 (both vLLM benchmark; port to stable sglang).

Why W8A8 TP=2 hits all levers:

- accuracy: 8-bit weights+activations > int4.
- prefill: INT8 XMX fast path (1.24-1.61x compile-fused).
- long context: TP=2 splits KV across 2 cards -> ~2x context capacity.
- capacity: W8A8 ~35 GB does not fit one 32 GB card -> TP=2 mandatory. Note: this
  forfeits DP replicas, which are an int4/W4A8-only (fits-one-card) pattern.
- decode: the weak lever. The 608 GB/s ceiling is high, but TP=2 decode is
  COLLECTIVE-LATENCY bound (~128 host-staged allreduces/token, ~0.29 ms floor each,
  no GPU P2P). push_allreduce and MTP attack exactly this.

Decode-to-20 levers, priority order:

1. MTP (speculative): proven 1.44x on TP=2 (18.10 -> 26.10). Single biggest lever.
   Get it stable on sglang (port the all_gather capture shim from the vLLM path).
2. Captured collectives / graph capture: eager 4.1 -> captured 18.10 (4.5x). The
   sycl_graph + all_gather shim that fixed vLLM must land on sglang.
3. push_allreduce: keep cutting collective latency/count. Overlap allreduce with
   compute; batch collectives; shrink payload (reduce in int8/fp16).
4. fp8 KV: at long context, KV reads dominate decode bandwidth; fp8 KV widens both
   the decode ceiling and the context window.

PP=2 verdict (asked 2026-06-27): secondary experiment, not the primary path.

- Pro: near-zero comms (1 hidden-state handoff/token vs ~128 allreduces) --
  structurally suited to a no-P2P box; already beat TP=2 eager (6.11 vs 4.18).
  Bonus: far smaller oneCCL wedge surface -> potential RELIABILITY win.
- Con: pipeline bubbles at batch=1 -> ~50% util, hurts single-stream interactive
  latency. Only fills under concurrency / microbatching.
- Use: evaluate PP=2 as a THROUGHPUT/concurrency endpoint and as a wedge-avoidance
  path, with continuous batching. Do NOT expect it to beat TP=2+MTP for the
  interactive agent.

4-card layout recommendation:

- Primary: 2x [W8A8 TP=2] replicas across the 4 cards. Each replica = accuracy
  (W8A8) + long context (TP=2 KV split) + no single-stream bubbles + strong INT8
  prefill; two replicas give concurrency without the TP=4 collective collapse.
- Avoid TP=4 for 27B: 4x the allreduce tax, decode collapses. TP=4 only for a model
  too big for 2 cards.
- Alternative to weigh: W4A8 single-card (fits one card -> DP replicas, decode ~42
  with MTP, INT8 prefill, 8-bit activations) trades some weight precision and
  single-card KV headroom for higher decode + cleaner concurrency. Pick W8A8-TP2 if
  long-context KV and max accuracy dominate; W4A8-single if decode and concurrency do.

## Data caveats

- No public dense-27B single-stream B60/B70 number exists; this repo is the only source.
- Spark NVFP4 + speculative-decode numbers are thin; EAGLE/MTP on CUDA vLLM is mature
  and could push the Spark's 22 toward 30-40 single-stream (would narrow but not close
  the gap, and still under the B70 ceiling).
- Deep-research auto-synthesis aborted on a spend limit; 18 claims verified 3-0, 7 left
  unverified (abstain), none refuted.
