# Combined-winner qualification proposal (source only, 2026-10-16)

Stopped at root/user direction to finish the current stage and pause for the
night. No new experiment, implementation or configuration was created. S148
GDN result was still pending when this note was frozen.

One minimal next combined configuration: retain the current all-candidate-OFF
S148 baseline unchanged except STRATA_Q5_HEAD_RUNTIME_ONES=01010101,
STRATA_Q5_HEAD_RUNTIME_ONES_C6=01010101, and
STRATA_ATTN_VALUE_PIPELINE=1. This combines the two disjoint explicit Q5 head
column branches and the separate eligible INT8 prefill attention entrypoint.
It is a proposed new configuration, not an already measured combined winner.
No individual percentages may be added or multiplied into a prediction.

| Existing candidate | Evidence and source compatibility | Proposed first combination |
| --- | --- | --- |
| Q5 C2 runtime ones | S130 +0.658692% aggregate medians; explicit type13 N2560/M248320/C2/RPW1 head, may include verifier T2 as well as paired MTP | ON |
| Q5 C6 runtime ones | S139 +1.281458% aggregate medians; sibling C6 branch, corrected distinct host marker | ON |
| Attention value pipeline | S113 +0.3321% at4K/+2.6922% at32K native prefill medians; guarded nq32/cap2051/INT8/page4 entrypoint, decode/verify/MTP retain original entry | ON |
| Q6 center | S87 +1.1546% aggregate medians; separate Q6 C6/PF0 path, source-compatible with Q5 heads, but measured under an earlier policy/source context | Keep OFF for minimal first combination |
| IQ2_XS multi | S100 +0.307757% aggregate medians; separate GU17/LN8/N2560/FF640 path, source-compatible but small and another interaction to qualify | Keep OFF |
| GDN commit layers | S148 serving in progress at cutoff | No selection before root closure |

All changes above remain default OFF in source. Disjoint eligibility does
not prove combined numerics, scheduling, latency or memory behavior. Q5 head
RPW2 must stay0 because its branch precedes both runtime-word branches.
Retain WIDE1/RPW1/A2/PF0, no owner queues/whole-pair graph/probability skip,
no parallel argmax/Q6A4/packed/QSA-tiled or other unselected knobs. Preserve
analytic RoPE policy as the current baseline requires. Do not silently import
S30's old spec-min-p0.5/suffix expansion into the current fixed native policy
suffix0/lookup0/spec4/MTPmax4/min-p0. S113 explicit HTTP override was only
spec_min_p0. This is not a claim that old and new work policies match.

Reuse and minimum gates, if root later authorizes:

1. On one frozen current executable, compare all-OFF versus the combination
   using the existing finite serving controllers: one256-output warm and
   three fresh107-prompt/1500-output C2 pairs, immediate complete text/SSE
   content/usage/native work/capture parity, identity/argv/env/loader and
   distinct HOST-marker gates; concurrent math, thinking and tool roundtrip.
   Keep all samples and no profiler. Existing stages cost roughly several
   minutes for two fresh processes including normal closure/full health.
2. Only after that passes, use one combined candidate server for existing
   S30 cheap4K then full four-chat oracle. The reusable entry is tracked
   strata/flash-next/bench_context_four.py, with --config NEW_CONFIG,
   --output NEW_OUTPUT and default --target199600/--context-size200000.
   Four distinct prompts actually tokenized199501 in S30. Sequence is
   initialAB -> residentAB -> initialCD -> residentCD -> exchangeAB ->
   exchangeCD. Two near-full active replies must each emit at least128
   tokens, overlap>.1s and report mean rows>depth+1 (depth2 means>3), not
   merely two overlapping HTTP requests. Both slots must retain near-full
   histories. Require all needles/no cross-chat codes; four forced RAM
   returns with nonzero payloads, cache reuse of held history, new prompt
   work<=128 and prior-tail replay0. Keep sampled RAM/swap/OOM and draft-KV
   restored-byte verification. The2s cached TTFT value is diagnostic only.
   Historical S30 full run occupied about13minutes; plan15-20minutes with
   startup/cheap gate/normal health, not a forecast of improved performance.
3. Reuse the actual current normal QUIT/full CID+NAME removal and per-card
   plus compiled collective post-health lifecycle. Do not relaunch the
   historical S30 shell wrapper unchanged: it hardcodes old binary SHA,
   historical image/observer and container name. The four-chat script itself
   expects container strata-ab-serve, the exact supplied config, live200k
   slots and configured32768MiB/4parking entries. Current controllers use
   other arm-specific names; a bounded name/config adaptation is necessary
   before use. No new harness framework is required.

S30 evidence remains valid for its frozen152055a6 configuration; it is not
qualification of this combination. Its snapshot observer makes full-context
rates correctness diagnostics, separate from clean decode measurements.
The existing tracked serve-iq3xxs-200k.json is an older split27/prefill4096/
16384MiB-cache daily-control config, not the S30 EP149/363/12288/32768MiB
qualification template. Start from the current retained baseline and carry
S30's full-cache gates, not the old control file. No additional speculative
kernel probe is proposed.

Inspected primary local sources: docs/20261016_strata_q5_runtime_ones_serving.md;
docs/20261016_strata_q5_c6_serving.md;
docs/20261016_strata_attention_value_fixed_policy.md;
docs/20261015_strata_qsa_tiling_and_q6_centering.md S87;
docs/20261015_strata_routed_iq2_reuse.md S100;
docs/20261014_flashnext_strata_optimization.md S30;
tracked bench_context_four.py and bench_serving.py;
R/draft-carry-s30/{config.json,launch_oracle.sh,oracle.py,full-cache-harness-sha256.json,
s30-report.json,full-cache-verdict.json}; current native_mmvq.dp.cpp head
branches at4028..4085, iq_kernels.dp.cpp GU guard at4372..4409 and
qsa_decode_attn.dp.cpp value guard at794..823. R denotes the retained
flashnext_ista_gsq_rco_iq3xxs_strata_cachefix_20261014 result root.
