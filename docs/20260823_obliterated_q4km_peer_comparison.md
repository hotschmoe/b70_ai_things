# Qwen3.8-27B OBLITERATED Q4_K_M peer comparison

Status: point-in-time comparison after updating the clean Steve and Sergio
clones on 2026-08-23. These are not matched A/B results. Model identity,
quantization, runtime, context, topology, prompt set, and metric differ.

## Local result under review

The local candidate is the fixed V3
`OBLITERATUS/Qwen3.8-27B-OBLITERATED` Q4_K_M GGUF. It runs as two independent
one-card llama.cpp replicas behind nginx, with embedded MTP3, Q8_0 KV, one
245760-token slot per card, and model id `hotschmoe-dd`.

- Card 0 simultaneous decode median: 40.35 tok/s.
- Card 1 simultaneous decode median: 41.51 tok/s.
- Two-stream sum: 81.86 tok/s.
- No-MTP two-stream sum: 47.69 tok/s; MTP3 adds 71.7%.
- Coherence: 338/338 c4 soak requests passed with no degenerate output.
- Long context: one real 152289-token prompt completed without truncation or
  context shift; the live slot is 245760 tokens.

The short benchmark prompts actually tokenized to about 1150 tokens and forced
128 output tokens. The two-stream sum is useful capacity, not one-request TP2
speed. Because each replica has one slot, c4 at the proxy queues behind two
active generation lanes rather than forming a four-request device batch.

## Steve Seguin lab

Updated clean clone:
`/mnt/vm_8tb/b70/research/b70-lab-agent` at
`0107f278a1486b6177fc5d4e6b7b44e04f14bc52`.

Closest verified llama.cpp point:

| Config | Cards | Metric | Result | Local relationship |
|---|---:|---|---:|---|
| Standard Qwen3.8 Q4_K_M, no speculation, 8K/F16 KV | 1 | final capture | 27.81 tok/s | Local mean 40.93 is 47.2% higher |

This is the most useful runtime/quant family comparison, but it is still not
matched. Steve used the standard GGUF, a different file SHA, no MTP, 8K
context, and F16 KV. The local result uses the OBLITERATED fixed V3 GGUF,
MTP3, 245760 context, and Q8 KV.

Steve's pinned certified vLLM target-only AutoRound INT4 W4A16 frontier is
30.26 tok/s at TP1 and 48.95 tok/s at TP2. The local per-card mean is 35.3%
above the TP1 point and 16.4% below the TP2 one-request point. The local
two-stream sum is 67.2% above 48.95, but that is DP capacity versus TP2
single-request latency and must not be described as a matched win.

Correction to the old headline: Steve now marks the published 101.922 tok/s
MTP5 result as refuted. Its greedy margin changed emitted text on 18/25 prompts
and the quality baseline used the same margin. The honest margin-free working
anchor is 101.17 tok/s, but three arms agree on only 21-22/25 prompts, so Steve
keeps it research-only and non-promotable. The local 81.86 two-stream sum is
19.1% below that research anchor, while the local lane has passed its stated
coherence and long-context gates. Those facts do not make the metrics matched.

## Sergio B70 inference cookbook

Updated clean clone:
`/mnt/vm_8tb/b70/community_repos/intel-arc-pro-b70-inference-cookbook` at
`dca0249684769b0a945a8d702352fdeea658852a`.

Sergio's Qwen3.8 model is a GPTQ-INT4 target in vLLM on one B70 with MTP4, not
the local GGUF or an obliterated checkpoint.

| Sergio config | Cards | Result | Local relationship |
|---|---:|---:|---|
| BF16 draft, p512/g128, cache off | 1 | 81.20 tok/s | Local per-card mean is 49.6% lower; two-stream sum is 0.8% higher |
| Optional draft-INT4, p512/g128, cache off | 1 | 112.65 tok/s | Local per-card mean is 63.7% lower; two-stream sum is 27.3% lower |
| Current greedy C1 stack | 1 | 106.7 tok/s | Local per-card mean is 61.6% lower; two-stream sum is 23.3% lower |
| Realistic C5 coding sessions | 1 | 127.4 sum-stream tok/s, 25.5 per user | Local two-lane sum is 35.7% lower; concurrency is not matched |

The apparent 81.86 versus 81.20 tie is not a win: it compares two local cards
and two simultaneous requests against one Sergio card and one request.
Sergio's optional draft-INT4 overlay is the clear raw decode leader. His
matched p8192/g1 cold prefill is about 1691-1696 tok/s. The local short-prompt
prefill was about 568-589 tok/s, and its real 152289-token prompt averaged
128.55 tok/s. Prompt lengths, engines, and metrics differ, so these prefill
figures are orientation only.

The local configuration has the stronger demonstrated production-context
story for this comparison. Sergio describes about 100K as the practical
Qwen3.8 serving context; his isolated 128K point left about 870 MiB free and
is explicitly not a serving headline. The local service exposes 245760 per
replica and completed a real 152289-token prompt. It also gives two isolated
request lanes without TP collectives.

## Verdict

Performance is good for the intended daily-driver shape: fixed obliterated
model identity, coherent speculative decoding, two independent users at about
41 tok/s each, and unusually large tested context. It is not the best raw
single-card decoder. Sergio's vLLM plus draft-INT4 work is roughly 2.75 times
the local per-card rate on its p512/g128 cell. The next meaningful performance
campaign would be a matched prompt/output/cache suite, not a comparison of the
headline numbers above.

Sources are the two local, updated repositories named above. Steve evidence:
`CURRENT.md`, `claims/lab-qwen38-27b-q4km-tp1.json`, and
`claims/lab-qwen38-27b-int4-mtp5-tp2.json`. Sergio evidence:
`docs/qwen38-27/QWEN38-VLLM-XPU.md`.
