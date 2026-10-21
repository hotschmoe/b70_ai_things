# X5 -- Qwen3.6-35B-A3B on one B70

Fixed-kernel quant measurements are complete. IQ3 is fastest at 8.12 ms
C1 p50; Q4_K_M is the preferred 4-bit recipe at 10.37 ms, with lower
matched prefix PPL than IQ4_XS. All three pass identity,
coherence, exact repeat, source, normal exit and native pre/post health.
Both MTP depths pass every gate. Depth 3 gives 5.67 ms effective C1
latency; depth 2 gives lower C2 latency on the short matched benchmark.
No submission or rank-win claim.

Raw root: /mnt/vm_8tb/b70/build/vllm-flashnext/x5/.
The Flash-Next MTP implementation and Strata sidecar findings are in
reports/1X1_mtp.md; that larger-model work passed to O5. The measurements
below are the official35B proxy, not Flash-Next or Ornith/Shisa.

## Recipe and identity

[Recipe](../harness/x5_recipe_35b.sh) runs a bounded health/RAM-guarded
qualification window. [Server arguments](../harness/x5_serve_proxy.sh)
are separate. Default QUANT=UD-Q4_K_M, MTP_TOKENS=0; select
QUANT=UD-IQ3_XXS for the fastest lower-bit alternative.

```bash
# CPU builds, disk scratch, MAX_JOBS=4; no GPU touched by these builds.
bin/cpu-run bash vllm/flashnext/harness/x5_build_gdn_ragged.sh
bin/cpu-run bash vllm/flashnext/harness/x5_build_eventless.sh
bin/cpu-run bash vllm/flashnext/harness/x5_build_q8.sh

TAG=q4-fixed-nospec bash vllm/flashnext/harness/x5_recipe_35b.sh
```

Physical card1, bin/gpu-run lease plus entrypoint ZE_AFFINITY_MASK=1;
logical XPU0 inside the container. TP1/world1, BF16, FULL_DECODE_ONLY
capture sizes1/2/3/4/6/8, installed FLASH_ATTN, prefix cache disabled,
maxlen4096, maxseq2, batch1024, GPUutil0.90. Stable first model name
hotschmoe-dd; quant/depth research alias second and registered in
evals/configs/models.yaml. This does not start the DD service.

Flags: B70_FUSE_SHARED=1; X5_GDN_BF16, X5_EVENTLESS,
X5_EVENTLESS_LAZY, X5_PACK_LINEAR, X5_Q8_LINEAR, X5_CANONICAL,
X5_Q8_FAST, X5_GDN_MIXED, X5_GDN_RAGGED all1; D3_GDN_SPEC1;
X5_BATCH_LINEAR0. Old D3 FP16 wrapper is off. Periodic graph
reinstantiation is off for the eventless path. B70_PREFILL0 and
SYCL_CACHE_PERSISTENT0. No CCL topology override.

Kernel7.1.0-070100, UMD26.22, loader1.28.2, Torch2.14+xpu08187d9e,
oneAPI2026.1.1. Image vllm-flashnext-xpu:py312; Q4 container records
image ID ea234fbf7123cba58b387f4046a0c32192dd5b6febead3684dbd8690882bf572.
Fork b70/flashnext HEAD0a6170da8ca9991972322fa1ca1d27674f0e75cd plus
shared dirty changes. Per-window manifests pin actual files/binaries;
HEAD alone does not describe the measured source. X5 changes are
opt-in overlays and an isolated plugin, without core fork edits.

| Component | SHA256 |
| --- | --- |
| O3g MoE | 38f7176ac290cb264268185c864da2dea1fcc36d07535fff021fb636d6538dcc |
| Aligned Q8 | 51f304b431db82cd6cde1b6591c5caa19dd4184766a018eef615dac2068ce17c |
| Fixed BF16/ragged GDN R3 | 84dad4e82fa110fae57849d370fd3151ab2c09552bb83bf11b37dc94c613d130 |

Replay helper/stub hashes are in each manifest; this helper is specific
to the pinned Torch ABI and must be rebuilt after a Torch change.

## Corrected no-spec comparison

SSE inter-chunk ITL p50; same corrected stack and card. These short
completion benchmarks stop at configured EOS/128tokens. They are
separate from the canonical512-token submission protocol.

| GGUF | C1 p50 ms | C2 p50 ms/stream | Prefix PPL | Full gate |
| --- | ---: | --- | ---: | --- |
| UD-IQ3_XXS | 8.1166 | 10.2861 /10.3282 | 9.055633 | PASS |
| UD-IQ4_XS | 10.9075 | 11.6587 /11.6468 | 8.764780 | PASS |
| UD-Q4_K_M | 10.3716 | 11.5693 /11.5573 | 8.508228 | PASS |

Both concurrent responses match their solo outputs in every final
quant window. Exact C1+C2 repeat, identity, coherence and source gates
pass. Every serve returns0 normally. All native pre/post health checks
pass. Each source-before/after SHA map
matches. Between IQ3 and later windows only metadata caption/collection
logic changed; model/kernel hashes match. No long-duration,200K
context, RAM eviction or arbitrary-concurrency qualification.

Raw tags end fastq8-canon-mixed-ragged-mtp0, prefixed
qwen36-35b-ud-iq3_xxs, qwen36-35b-ud-iq4_xs, or qwen36-35b-udq4km.

HF config/tokenizer revision995ad96eacd98c81ed38be0c5b274b04031597b0;
GGUF repository unsloth/Qwen3.6-35B-A3B-MTP-GGUF revision
5bc3e238d916f48a861bac2f8a1990a0e9b7e98d. Files live under
/mnt/cache/b70/devmodel/unsloth/Qwen3.6-35B-A3B-MTP-GGUF/.
Each has753tensors, including20MTP tensors in block40 and one
nextn prediction layer. Native Qwen3.5 MTP loads19parameter slots;
one decoder head is reused for successive draft positions.

| Format | File bytes | Trunk | Routed gate/up | Routed down |
| --- | ---: | --- | --- | --- |
| IQ3_XXS | 14069266720 | mostly Q6_K | mostly IQ2_S | mostly IQ3_S |
| IQ4_XS | 18209036576 | mostly Q8_0 | mostly IQ3_S | mostly IQ4_XS |
| Q4_K_M | 22663387424 | mostly Q8_0 | mostly Q4_K | Q5_K/Q4_K |

Dynamic mixed formats, not uniform3/4-bit weights. Heads are Q6_K.
quant-inventory.json records each tensor. The PPL diagnostic scores
the first2048 WikiText-2 raw-test tokenizer tokens as four independent
512-token chunks,2044 scored tokens. It is not full test-set PPL.
Dataset revisionb08601e04326c79dfdd32d625aee71d232d685c3;
corpus SHAefd86268d53e85efd67b67fcdf4846c7a4c56bc57cff41d00e3d7e4b57e17c46.

## MTP qualification

Corrected depths 2 and 3 each match all 576 raw token IDs across
4 C1 + 2 C2 responses against noSpec, and pass exact same-load repeat.
Source SHA maps match noSpec exactly; identity/coherence/source gates
and normal exit 0 pass. Native pre/post health passes for both depths.

| Draft depth | C1 effective ms/token | C2 effective ms/token/stream | Bench acceptance | Gate |
| --- | ---: | --- | --- | --- |
| 0 | 10.1616 | 11.2885 /11.2765 | n/a | PASS |
| 2 | 6.6182 | 9.9137 /8.5302 | 91/120=75.83% | PASS |
| 3 | 5.6736 | 11.0375 /9.0111 | 98/159=61.64% | PASS |

Effective latency is1000/decode_tok_s. Depth2 wire-chunk p50 is
17.9341ms C1 and23.0351/23.0406ms C2. Depth 3 wire-chunk p50 is
17.8525 ms C1 and 27.5076/27.5768 ms C2; one chunk carries multiple
accepted tokens. The short latency benchmark stops at EOS/128tokens.
Acceptance is specific to these prompts, not a universal estimate.
The serving log proves target FULL T3/T6 and native draft capture,
plus real ragged T1/T2 verification using the fused kernel. Depth 3
also completes FULL_DECODE_ONLY with the same graph-size list.

Spec windows also collect unranked512-token canonical packets.
Depth 2 reasoning/code rates are 147.1/141.8 tok/s, with acceptance
85.714/80.203% and mean accepted length 2.714/2.604. Depth 3 reaches
181.7/163.5 tok/s, acceptance 79.912/67.653% and mean accepted length
3.397/3.030. All four local/API dry-run validations pass;
ranking_eligible=false and no submission.
These are separate workloads from the short completion benchmark.
Choose depth 3 for measured C1 workloads; depth 2 is the measured C2
choice. These finite prompt samples do not establish a universal depth.

For a fresh matched reproduction, retain the new noSpec token file:

```bash
MTP_TOKENS=0 TAG=q4-fixed-nospec \
  bash vllm/flashnext/harness/x5_recipe_35b.sh
X5_BASELINE=/mnt/vm_8tb/b70/build/vllm-flashnext/x5/q4-fixed-nospec/greedy.json \
  MTP_TOKENS=2 TAG=q4-fixed-mtp2 \
  bash vllm/flashnext/harness/x5_recipe_35b.sh
X5_BASELINE=/mnt/vm_8tb/b70/build/vllm-flashnext/x5/q4-fixed-nospec/greedy.json \
  MTP_TOKENS=3 TAG=q4-fixed-mtp3 \
  bash vllm/flashnext/harness/x5_recipe_35b.sh
```

The final comparison uses raw tags ending
qwen36-35b-udq4km-fastq8-canon-mixed-ragged-mtp{0,2,3}.

## Implemented fixes and profiling evidence

- Compatible qkv+z, q+k+v and b+a projections merge output rows before
  device placement (harness/x5_pack_linear.py:11).
- Known dense/shared Q8 GGUF payloads stay packed; embeddings/router/
  conv keep their original path (harness/x5_q8_linear.py).
- Complete routed+shared TP1 MoE chunks at8, keeping decode arithmetic
  when another request prefills (harness/x5_canonical_moe.py:14).
- Aligned Q8 codes/half-scales, SG16/four output rows, DP4A, fixed
  activation quantization/reduction and BF16 epilogue reuse weights
  acrossM1..8 (plugins/b70_x5_xpu/csrc/q8_linear.cpp).
- BF16 D3 supports compact ragged verification, actual-only state
  writes and inactive groups. Three history rows are staged before
  shared q/k writeback; per-layer scratch is owned before capture
  (harness/patches/x5_gdn_ragged.py).
- Mixed eager GDN batches split prefill/decode metadata and rows;
  prefill stays native, decoded rows keep fused arithmetic. Metadata
  is restored in finally. Known one-token decodes retain decode math,
  fresh one-token prefills still initialize state
  (harness/x5_gdn_mixed.py:21).

The inherited reduce_128 returned parts[0] after a barrier, then reused
that shared location for the k sum before every subgroup finished
reading the q sum. R2 showed exact SSM/conv/canary but mismatching
query-dependent core/output in3-sequence cases. The read-complete
barrier is the R3-only change from R2. R3 passes all76 eager/rejection/
padded-page/inactive/8-replay cases exactly against sequential fused
decode, including previously failing cases. Stock approximation
3seeds/32replays also passes at unchanged tolerances. Pre/post native
health and normal exit0 pass. Raw gdn-ragged/r1,r2,r3 retain failures.
Persistent scratch alone did not fix the R2 mismatch. Shared D3 source
is unchanged; JOURNAL requests a deliberate port by its owner.

The earlier Q4 unitrace control q4-lazy-trace-r2 has23 decode-like
spans,1185kernels/span,11.92ms median span/10.70ms busy. The trace
is instrumented; its host gap includes GPU waiting, not pure idle.
Ordered attribution validates10GEMMs per GDN and9 per full-attn layer:

| Dense region | GEMMs/token | Kernel sum ms/token |
| --- | ---: | ---: |
| GDN projections | 150 | 3.942 |
| Full-attention projections | 40 | 1.080 |
| Shared expert/gate | 160 | 0.995 |
| Router | 40 | 0.171 |

Cats110/.304ms, casts~.232ms, fused GDN30/.338ms, Q6_K head1/.709ms.
This historical trace selected the projection work; it is not a trace
of the corrected final stack. The aligned-Q8 kernel passes10actual-
weight M1/2/3/4/8 cases: FP32 maxabs0 versus old op, exact BF16,
batch invariance and capture/replay. The historical matched generic-
Q8 control15.65ms became10.22ms C1, with identical576token IDs and
PPL. Those old full-model rows used the inherited D3 helper and are
superseded for final qualification. No isolated GDN speed claim.

## LocalMaxxing packets

Corrected Q4 noSpec packets: raw Q4 tag above, under
localmaxxing/no-spec-{reasoning,code}-v1/.

| Suite | Median tok/s (rounded payload) | TTFT of median-rate sample ms | Output tokens |
| --- | ---: | ---: | ---: |
| reasoning-v1 | 94.8 | 567.35 | 512 |
| code-v1 | 94.9 | 454.39 | 512 |

Each uses one warmup and3iterations, temperature0, leading nonce,
prefix cache off, authoritative usage and isolated request/token/spec
counters. Live speculative_config=null, specDecoding=false and
mtpEnabled=false. Both local validation and API dry-run return
valid=true, verification.verified=true, issues=[]. All launch flags,
model revision, actual4096context and source/library identity are
recorded. No submission or public verified badge is claimed.
The [API docs](https://www.localmaxxing.com/en/api-docs) specify that
dry-run validates without creating a database entry. Historical packets
in the old lazy-GDN tag are superseded and must not be the final recipe.
Public Arc entries mix card counts, context and short/unverified output;
this report does not claim a leaderboard win from unlike methods.
Only the noSpec packets are eligible for the requested non-speculative
ranking. MTP packets remain separate diagnostic variants.

## Lifecycle and limits

Early replay preloads completed requests but exited139 during Torch/
SYCL finalization. A libc-only preload plus late Torch helper fixes
normal API exit and passes2066replays/RNG advancement. Explicit
nonvirtual replay dispatch avoids recursion; preserve ordinary SYCL
symbol interposition. Original reclaim500 control stalled; its exact
cause is unproven. Eventless mode is a finite measured fix, not a soak
claim. The c7b Flash-Next FA overlay is excluded for this proxy because
its missing paged config falls back to a capture-invalid D2H reference.

A scoped native batch-invariant dense control passed at11.30ms C1,
PPL8.52856, but did not fix C2 MTP by itself and is off in the final
fast recipe. Historical MTP C1-only parity/speed rows are unqualified.
All five final windows closed with HEALTHY native pre/post checks,
unchanged source manifests and normal serve exit 0. The detached
controller reports X5_FINAL_ALL_DONE at 2026-10-21 03:37:12 UTC.
No shelf promotion, commits/pushes, DD start or
GPU reset. Other agents' dirty changes are preserved.

Claude fable-5.1 and claude-5.5 CLI consultations returned
unrecognized_model and Not logged in. No responding model IDs or
independent advice were obtained; none is invented in this report.
