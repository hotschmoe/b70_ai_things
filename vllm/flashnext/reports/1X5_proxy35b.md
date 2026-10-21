# X5 -- Qwen3.6-35B-A3B on one B70

Final requalification is in progress. Historical noSpec Q4 measured
10.225ms C1 and IQ3 8.047ms, but a subsequent larger primitive probe
located a missing shared-memory read-complete barrier in the inherited
D3 reduction helper. Those finite measurements and dry-run packets are
preserved as history; the final recipe will use the corrected kernel.
MTP C2 remains unqualified. No submission or rank-win claim.

Raw root: /mnt/vm_8tb/b70/build/vllm-flashnext/x5/.
Flash-Next MTP ownership passed to O5; X1's implementation, Strata
sidecar inventory and original failed gates are in reports/1X1_mtp.md.
This report qualifies the smaller official proxy separately.

## Recipe

Run the bounded selected window, including health/RAM/teardown gates:

```bash
TAG=qwen36-35b-q4-final-nospec \
  bash vllm/flashnext/harness/x5_recipe_35b.sh
```

The three isolated libraries can be rebuilt on CPU (not while timing):

```bash
bin/cpu-run bash vllm/flashnext/harness/x5_build_gdn.sh
bin/cpu-run bash vllm/flashnext/harness/x5_build_eventless.sh
bin/cpu-run bash vllm/flashnext/harness/x5_build_q8.sh
```

These scripts pin MAX_JOBS=4 and disk-backed scratch. Re-run their
GPU numeric/replay and model gates after a rebuild or runtime change.

Set QUANT=UD-IQ3_XXS for the fastest lower-bit alternative.
Default MTP_TOKENS=0. x5_serve_proxy.sh contains the actual server
arguments. Physical card1 only, explicitly leased and pinned; logical
XPU0 inside the container. TP1/world1, BF16, FULL_DECODE_ONLY capture
sizes1/2/3/4/6/8, installed FLASH_ATTN, no prefix cache, maxlen4096,
maxseq2, batch1024, GPUutil0.90. Stable first alias hotschmoe-dd;
second alias qwen3.6-35b-a3b-<quant>-x5-mtp<depth> is registered in
evals/configs/models.yaml. This does not start the DD service.

Flags: FUSE_SHARED1, X5_GDN_BF16=1, X5_EVENTLESS=1,
X5_EVENTLESS_LAZY=1, X5_PACK_LINEAR=1, X5_Q8_LINEAR=1,
X5_CANONICAL=1, X5_Q8_FAST=1, X5_BATCH_LINEAR=0.
Old D3 FP16 wrapper and periodic graph reinstantiation are disabled.
Native spec GDN is used unless B70_D3_GDN_SPEC=1 is explicitly selected.
B70_PREFILL=0, SYCL_CACHE_PERSISTENT=0. No CCL topology override.

Host kernel7.1.0-070100, UMD26.22, loader1.28.2, Torch2.14+xpu
08187d9e, oneAPI2026.1.1; image vllm-flashnext-xpu:py312.
Fork b70/flashnext HEAD0a6170da8ca9991972322fa1ca1d27674f0e75cd,
shared dirty changes preserved. Per-window source-before/after manifests
pin the actual code and binaries; HEAD alone does not describe it.
X5 adds opt-in overlays and an isolated plugin, without core fork edits.

Pinned O3g: 38f7176ac290cb264268185c864da2dea1fcc36d07535fff021fb636d6538dcc.
Aligned-Q8 SO: 51f304b431db82cd6cde1b6591c5caa19dd4184766a018eef615dac2068ce17c.
BF16 D3 SO: 2bb3661f2b445ba57b3af0db5dbca192550bb22ebefac24d2a7f0a1953aae328.
Replay helper/stub hashes are in every final manifest. It is specific
to this Torch ABI and must be rebuilt after a Torch change.

## Quant comparison

Same final stack, card1, no MTP; SSE inter-chunk ITL p50 below.
These short completion benchmarks are separate from the canonical
512-token submission protocol. C2 may include another request's prefill.

| GGUF | C1 p50/p90/p99 ms | C2 p50 ms/stream | Prefix PPL | Gates |
| --- | --- | --- | ---: | --- |
| UD-IQ3_XXS | 8.047/8.258/8.374 | 10.226 /10.273 | 9.055633 | PASS |
| UD-IQ4_XS | 10.854/11.090/11.446 | 11.529 /11.511 | 8.764780 | PASS |
| UD-Q4_K_M | 10.225/10.386/10.551 | 11.500 /11.491 | 8.508228 | PASS |

Raw tags: qwen36-35b-ud-iq3_xxs-fastq8-canon-mtp0,
qwen36-35b-ud-iq4_xs-fastq8-canon-mtp0,
qwen36-35b-udq4km-fastq8-canon-mtp0.

HF config/tokenizer revision995ad96eacd98c81ed38be0c5b274b04031597b0;
GGUF repository unsloth/Qwen3.6-35B-A3B-MTP-GGUF, revision
5bc3e238d916f48a861bac2f8a1990a0e9b7e98d. Files under
/mnt/cache/b70/devmodel/unsloth/Qwen3.6-35B-A3B-MTP-GGUF/.
Each contains753 tensors, including20 tensors in MTP block40 and
nextn_predict_layers1. Native Qwen3.5 MTP head, not Ornith/Shisa.
One draft layer is reused for successive speculative positions.

| Format | File bytes | Trunk | Routed gate/up | Routed down |
| --- | ---: | --- | --- | --- |
| IQ3_XXS | 14069266720 | mostly Q6_K | mostly IQ2_S | mostly IQ3_S |
| IQ4_XS | 18209036576 | mostly Q8_0 | mostly IQ3_S | mostly IQ4_XS |
| Q4_K_M | 22663387424 | mostly Q8_0 | mostly Q4_K | Q5_K/Q4_K |

These are dynamic mixed formats, not uniform 3/4-bit weights. All heads
are Q6_K. quant-inventory.json records the actual tensor formats.
The score uses the first2048 tokenizer tokens of WikiText-2 raw test,
four independent512-token chunks,2044 scored prompt-logprob tokens.
PPL=exp(meanNLL), no MTP. This is a matched prefix diagnostic, not
full test-set perplexity. Corpus hash:
efd86268d53e85efd67b67fcdf4846c7a4c56bc57cff41d00e3d7e4b57e17c46.
Dataset revisionb08601e04326c79dfdd32d625aee71d232d685c3;
parquet SHA5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91.

## Trace and implemented cuts

Q4 control q4-lazy-trace-r2, o4_trace.py-style unitrace capture:
23 decode-like spans,1185kernels/span, median11.92ms span and10.70ms
busy. Profiling overhead inflates the traced step. Untraced profiler
C1 step/replay11.748/11.093ms; C2 12.959/11.864ms. The8-9ms host gap
includes waiting for GPU completion and cannot be called idle time.
Offline profiler intentionally exits after trace flush; its exit is
not API lifecycle evidence. Native pre/post health and arithmetic pass.

| Dense region | GEMMs/token | Kernel sum ms/token | Representative N,K |
| --- | ---: | ---: | --- |
| GDN projections | 150 | 3.942 | qkv8192x2048,z4096x2048,out2048x4096,BA32x2048 |
| Full-attention projections | 40 | 1.080 | q8192x2048,KV512x2048,o2048x4096 |
| Shared expert/gate | 160 | 0.995 | GU512x2048,down2048x512,gate1x2048 |
| Router | 40 | 0.171 | 256x2048 |

Ordered role attribution validates10GEMMs per GDN layer and9 per
full-attention layer; roles.json/trace-summary.txt preserve it.
Cats110/.304ms, casts~.232ms, routed GU40/.864ms, down~.631ms,
fused GDN30/.338ms, Q6_K head1/.709ms.

Implemented:

- Merge compatible qkv+z, q+k+v, b+a output rows before placement.
- Preserve known dense projection/shared Q8 GGUF payloads rather than
  unconditional BF16 expansion. Routers/embedding/conv stay unchanged.
- Chunk the complete routed+shared TP1 MoE at8 to keep decode arithmetic
  when a decode row shares a batch with prefill. Routed-only chunking
  left the shared-expert fused/unfused boundary inconsistent.
- b70_x5::q8_linear separates existing int8 codes and half scales into
  aligned arrays, SG16/4rows per workgroup, DP4A, activation-Q8 matching
  O3, fixed reduction, BF16 epilogue, weight reuse acrossM1..8.
- BF16 instantiation of D3 and a late-loaded eventless replay helper.

The generic-Q8 control passes all gates but costs15.650ms C1 and
17.570/17.564ms C2. Aligned Q8 reduces these to10.225 and11.500/11.491:
34.7%/34.5% less latency, with identical fork hashes, prefixPPL and
all576 compared output token IDs. It does not isolate the gain from
packing or canonicalization versus the earlier unstable stack.

Actual Q4 merged projection weights at M1/2/3/4/8 pass10/10 probes:
FP32 maxabs0 against the old op, exact BF16 epilogue, batch invariance,
captured replay, exit0 and native pre/post health. Hot-loop matrix
latencies are cache-resident diagnostics, not model throughput claims.

BF16D3 primitive checks pass3seeds/32replays, exact conv/replay,
approximate stock core/state/output. State maxabs1.50e-4..2.45e-4;
tolerance is1e-4 PLUS0.005relative, not an absolute1e-4 bound.
Core maxabs<=6.10e-5, output cosine>0.999998. Full model gates remain
mandatory. The original FP16 D3 rejects BF16 after projections and
would duplicate work on fallback; the selected stack disables it.

## MTP gates

Q4 MTP block40 loads19 parameter slots from20 GGUF tensors. Target
and native Qwen3.5 drafter capture FULL graphs. One head is reused
at every depth. No Shisa substitution or separate proxy MTP file.
Every comparison respects configured EOS and compares raw token IDs.

| Stage | C1 effective ms/token | C2 gate | Verdict |
| --- | ---: | --- | --- |
| Selected BI0, stock spec GDN, depth2 | 6.597 | repeat passes, on/off FAIL at34 | rejected; C1 on/off FAIL at57 |
| BI0, fused spec GDN, depth2 | 6.632 | repeat and on/off FAIL at34 | all4C1 on/off comparisons pass |
| BI1, fused spec GDN, depth2 | 7.647 | repeat FAIL at66, on/off FAIL at34 | all4C1 on/off comparisons pass |
| BI1 + mixed GDN, depth0 control | 11.138 | repeat + solo/C2 parity PASS | PASS control |
| BI1 + mixed GDN, depth2 | 7.646 | repeat FAIL43, on/off FAIL41 | ragged fallback observed |

Every failed model stage exits0 normally, has matching before/after
source hashes and HEALTHY native pre/post checks. These remain failed
whole-variant gates; do not promote their diagnostic speed numbers.
The first stage acceptance is90/120=75%; fused spec91/120=75.83%;
BI1 stage93/128=72.66%, all for the short latency benchmark subset.
Wire-chunk timing differs from effective token latency under MTP.

BF16 fused verification passes21/21 exact sequential-equivalence
cases: widths3/4, every acceptance position,3seeds,8captured replays.
Core/output/intermediate SSM/final conv all maxabs0. Native health
and normal exit pass. Raw gdn-bf16/spec-r1.

The remaining C2 failure survives native batch-invariant dense linears.
A mixed batch made D3 fall back to stock GDN for the running decode
when another request prefills. x5_gdn_mixed.py splits eager GDN rows
and metadata: native prefill for prefill rows, fused recurrence for
decode rows. It restores metadata in finally, copies no recurrent
state, leaves pure FULL graph capture unchanged and keeps safe ragged
fallback. Known one-token decodes reclassified as prefill by vLLM
retain decode math; fresh one-token prefills still initialize state.
CPU row/state/reorder/known-fresh/ragged checks pass. Scope: TP1,
SYCL prefill, proxy maxseq2; no arbitrary-concurrency qualification.

The mixed noSpec control passes full graphs, identity/coherence,
exact C1+C2 repeat, unchanged source, exit0 and native pre/post health.
BOTH concurrent responses now match their solo outputs, whereas the
old control's Paris differs. All4solo outputs and prefixPPL8.52856158
match the prior BI1 control. C1 p50=11.3582ms,C2=12.7837/12.7484ms.
The real (prefill1,decode1,spec0) split is logged. Raw
qwen36-35b-udq4km-fastq8-bi-mixed-mtp0. MTP2 still fails C2; depth3 pending.

BI1 is a correctness control and is slower than selected BI0 noSpec.
The noSpec submission packets use BI0. MTP comparisons use the fresh
matched BI1+mixed control; do not imply parity with the BI0 recipe.

The mixed MTP2 retry exposes a real ragged fallback: ntok4,
slot(2,3), token indices4. The old fused kernel assumes equal width;
its guard safely routes ragged batches to stock GDN. The isolated
new BF16 kernel uses cu bounds and compact indices, writes only real
candidates and skips inactive groups. It stages three small history
rows before shared q/k writeback; no failure attribution is made to
that ordering change. Original libraries remain intact.

Ragged primitive R1 passes eager checks but fails8-replay equality at
3sequences with lengths[3,1,2]; HEALTHY pre/post and normal error exit1.
R2 uses caller-owned persistent scratch, is built (SHA248e6b78d150ef59f79ca0b726a696a10d39c6c9c3b9fd13c1fd26445c77b4be),
and is queued for the same probe with componentwise diagnostics.
No model result or causal lifetime claim yet. Raw gdn-ragged/r1,r2.

## LocalMaxxing

Canonical no-spec collection is complete, raw tag
qwen36-35b-udq4km-fastq8-canon-lmx-mtp0. Exact repeat/coherence,
source hashes, normal exit0 and native pre/post health all pass. Its
greedy tokens match the selected Q4 baseline.

| Suite | Median tok/s | Effective ms/token | TTFT of median-rate sample ms | Output tokens |
| --- | ---: | ---: | ---: | ---: |
| reasoning-v1 | 96.059 | 10.410 | 561.07 | 512 |
| code-v1 | 96.255 | 10.389 | 450.31 | 512 |

Both validate_dry.sh calls pass local validation and API dry-run
valid=true, verification.verified=true, issues=[]. No submission or
public verified badge is claimed. Payloads and dry-run.txt are under
localmaxxing/no-spec-{reasoning,code}-v1 in that raw directory.

x5_lmx_local.py wraps
X4's collector: one warmup plus3iterations of reasoning/code suites,
max512tokens, leading nonce, temperature0, prefix cache disabled,
authoritative usage and isolated request/token/spec counters. A
read-only config route proves speculative_config=null, without
exposing environment variables or enabling dev operations. Packets
carry exact model revision, context, dirty-build identity and flags.
The wrapper never submits. Run validate_dry.sh only on valid packets.
The [API docs](https://www.localmaxxing.com/en/api-docs) specify that
dry-run validates without creating a database entry.

Public API snapshot2026-10-21 in localmaxxing-public/ contains short
and unverified Arc entries; card counts/context/quant differ. A top
claimed one-B70 GPTQ4bit entry reports248.3tok/s with32outputtokens,
not the canonical256minimum. Do not claim a rank win from these
short proxy harness numbers or silently compare unlike methods.

## Retained failures and limitations

Early eventless API runs completed requests but exited139 during
Torch/SYCL finalization; not qualified. The libc-only preload plus
late Torch helper passes2066replays/RNG/normal exit and final API
teardown. An initial helper bridge used virtual replay and recursed;
qualified nonvirtual dispatch fixes it. This is a measured lifecycle
fix, not proof of the exact UMD constructor-order cause.

Earlier Q4 late-helper control repeated C1 but C2 Paris failed at34;
not qualified. Original reclaim500 control stalled after initial
requests; no completed latency/PPL, healthy stop. Correlation with
periodic replay is not causal proof. The c7b Flash-Next FA overlay
fails proxy capture and is excluded; installed FLASH_ATTN succeeds.
First unitrace tag failed because session names require alphanumerics;
r2 sanitizes them. First BF16D3 state tolerance failure is retained.

Claude fable-5.1 and claude-5.5 consults returned unrecognized_model
and Not logged in; no responding model IDs or advice were obtained.
No X5 commits/pushes, DD start, reset or other-agent termination.
Finite request/health gates do not qualify200K context or soak behavior.
