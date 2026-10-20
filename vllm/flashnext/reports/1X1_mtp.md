# X1 -- Flash-Next MTP on vLLM

Status: full Flash-Next target and MTP graphs run, with 77.03% bench draft
acceptance and a diagnostic C1 rate of 33.04 versus 23.98 tok/s. The
full-model exact-token gate FAILED; this is not a qualified speedup.
Proxy outputs match through configured EOS; forced post-EOS continuation
has a divergence described below. Numeric/state diagnosis is ongoing.
No promotion, commit, or daily-driver service change.

## Strata findings

The main ISTA IQ3_XXS GGUF does not contain the daily-driver MTP head.
`strata/flash-next/dd/hotschmoe-dd.json:43` selects
`/mnt/vm_8tb/b70/build/strata-ab/mtp-bf16/rt`. This directory contains
116099072-byte dense.bin, dense.txt (29 tensor records), and 707788800-byte
experts.bin. Source pack: `mtp-bf16/mtp-q2_0.gguf` (889014272 tensor bytes).
Its 31 tensors came from the original BF16 HF MTP checkpoint. The experts
are pinned by mtp-inventory.json to Qwen/Qwen3.8-Flash-Next revision
de4b8e4d43b917e7706784d8bb445c9af86a3540; mtp-manifest.json records
individual extracted tensor hashes. The packed experts
are Q2_0. Runtime dense projections are rounded to Q8_0; HC/router stay
BF16 and norms are F32 folded gamma. `src-ep/tools/mtp_rt.py:3` documents
the expert relayout and folded norms. The model-lock json pins the main
ISTA model, not this separate MTP sidecar.

Default solo verification is a four-row window (`--spec 4`,
`--mtp-max-t 4`): one already sampled target token plus up to three
drafts, not vLLM num_speculative_tokens=4. Batch MTP depth is two drafts
for concurrent requests. See `src-ep/src/program/generate.cpp:4495` and
`docs/20261016_night_lab.md:267`. One native MTP decoder layer is reused
across draft steps. Pooled RMS is over all four HC streams, fc_hidden
projects each stream independently, and fc_embedding is broadcast onto
them (`src-ep/src/core/mtp.cpp:906`). Mean-collapse before this projection
is a different computation. The Unsloth eh_proj sidecar is not the
daily-driver draft graph (G3 report).

Acceptance is prompt dependent. Historical engine.log lines 383554 and
383614 report 12526/17457 (71.75%) and 108/159 (67.92%) accepted drafts;
these are examples, not a matched benchmark or an average for the 141
tok/s reference. Strata also uses a restricted draft vocabulary
`src-ep/data/draft_vocab_en.bin`; vLLM currently uses the whole target head.

## Implementation

Fork: `/mnt/vm_8tb/b70/build/vllm-flashnext/vllm`, branch b70/flashnext.
Existing dirty changes from other agents are preserved.

- New `vllm/models/qwen4_exp/gguf/mtp_loader.py` isolates draft placement
  from target layers. The target loader dispatches to it for Qwen4ExpMTP
  and Qwen3_5MoeMTP. Shape and payload audits reject incomplete loads.
- Flash-Next selects `B70_MTP_RT=<rt>` plus
  `B70_MTP_GGUF=<mtp-q2_0.gguf>`. Dense weights use the actual rounded rt
  bytes; expert weights use the original GGUF bytes, avoiding the runtime
  code/scale-plane relayout. They remain packed for b70_gguf_xpu.
- `xpu/mtp.py` attaches quant methods to shared vocab and to the two fc
  projections when loading the rt sidecar. It then uses the existing wide
  fc path, bypassing eh_proj. Other quantization paths retain their former
  unquantized vocab construction. No new host reads or synchronization
  are introduced inside forward/capture.
- Proxy nextn block 40 is loaded into the native Qwen3.5 MTP module,
  including Q2_K/Q3_K experts supported by the current GGUF kernels.
  Target blocks never load into draft layer zero.
- `harness/x1_serve_proxy.sh`, `x1_proxy_window.sh`, and
  `x1_compare_greedy.py` provide FULL_DECODE_ONLY on/off qualification,
  exact token-ID comparison, acceptance, C1/C2 timing, health, RAM watchdog,
  and teardown. Raw evidence is in build/vllm-flashnext/x1.

## CPU evidence

CONFIG -> CPU-only torch, Gloo TP=1, meta model constructed from the real
Flash-Next text config. B70_MTP_RT/GGUF as above. No GPU allocation.

COMMAND -> `bin/cpu-run uv ... python harness/x1_check_sidecar.py`.
Also invoked the existing test_mtp_sidecar_every_tensor_once and
test_meta_qwen4exp_constructs directly because pytest is absent.

RESULT -> Existing checks pass (cpu-mapping.log). sidecar-cpu-r2.log:
FLASHNEXT_PAYLOAD_AUDIT_OK and WIDE_FC_REFERENCE_OK [2,4,2560], exit 0,
34 seconds after CPU lease. An initial CPU attempt failed because the
test initialized parallel groups outside set_current_vllm_config; fixed.
sidecar-cpu-r3.log also passes the actual _fc_stage forward against an
independent pooled RMS and Q8 dequant matmul reference, exit 0 in 25 s.

VERDICT -> Correctly sized, complete Flash-Next draft payload placement
is verified on CPU. This does not establish GPU graph or serving behavior.

## GPU evidence

First proxy baseline: card 0 health passed, target GGUF loaded, full graphs
captured, HTTP ready. X1 edited its active shell harness and caused a shell
read-offset failure (line 42 leanup: command not found), followed by clean
server teardown. No greedy or timing requests ran; this is not a baseline.
Preserved as proxy-mtp0-harness-error.log and window-mtp0-harness-error.out.
The frozen-harness baseline retry passed as recorded below.

The harness checks `/v1/models` against diagnostic entries added to
`evals/configs/models.yaml`. The stable first ID is hotschmoe-dd;
secondary IDs encode the model, GGUF method, and MTP depth.

Frozen baseline retry passed. Card-0 health before/after HEALTHY,
clean teardown, identity/coherence pass, four 96-token completions saved.
`bench-mtp0.json`: C1 p50 9.27 ms, 108.24 tok/s (9.24 effective ms/token);
C2 p50 12.48/12.38 ms, 82.34/74.93 tok/s (12.15/13.35 effective
ms/token, including a prefill transition in one stream). No MTP enabled.
This is the 35B proxy, not Flash-Next.

Matched proxy MTP depth2: identity and coherence pass, full graphs captured,
health before/after HEALTHY, clean teardown. `bench-mtp2.json`:

| Proxy config | C1 chunk p50 ms | C1 effective ms/token | C2 chunk p50 ms | C2 effective ms/token |
| --- | --- | --- | --- | --- |
| MTP off | 9.27 | 9.24 | 12.48 / 12.38 | 12.15 / 13.35 |
| MTP 2 | 18.29 | 6.99 | 28.66 / 28.42 | 10.94 / 11.86 |

Effective times are 1000 / reported token/s, not SSE chunk gaps. MTP emits
several tokens per chunk. Token/s: C1 108.24 -> 143.12; C2
82.34/74.93 -> 91.38/84.28. Bench acceptance 114/148 = 77.03%; aggregate
metrics after greedy + bench 373/454 = 82.16%. This is one short matched
run, not a stability claim, and not the two-card Flash-Next comparison.

Exact output evidence: three forced 96-token sequences are identical.
The fourth differs first at token 82; both have configured EOS 248044 at
token 78. Generation config eos_token_id is [248046,248044]. All four
match through their first configured EOS / length limit (367 tokens
including EOS). `greedy-eos-gate.json` derives this from the recorded
IDs. The original forced-tail GREEDY_RC=3 is retained; no claim of
identical generation after EOS is made.

## Flash-Next PLE correctness work

The existing PLE reads only non_spec metadata, skipping all-spec batches.
Its rolling token/conv history also commits rejected candidates, and its
graph convolution captures CPU slot indices. These prevent using the
proxy result as Flash-Next proof.

New isolated `gguf/mtp_ple.py` uses a ring of history + verify width,
indexed by logical position. Rejected future rows never overwrite past
rows needed after rejection. Both speculative and ordinary metadata groups
are staged outside the graph; live token positions and physical slots are
device buffers read by graph replay. The small V2 runner hook executes
preparation for both eager and graph steps. Enabled only with B70_MTP_RT;
the no-MTP matched Flash-Next control uses the same ring with width one.
Prefix caching is explicitly rejected until the PLE rings participate in
cache state transfer; both X1 serving scripts disable prefix caching.

CPU fixture `x1_check_ple.py` passes exact BF16 comparisons after long
prefill ring wrap, differing rejection lengths, slot reordering, and slot
reuse (ple-cpu-r3.log). A real-width (10240) XPU capture/replay probe is
passed: X1_PLE_XPU_GRAPH_REJECTION_REORDER_OK, conv-only graph
0.095748 ms at real width 10240, 50 device-event replays. Per-card health
before and after HEALTHY. This is a convolution-only graph measurement.

First full Flash-Next control placed all weights (23.97 GiB/card) but
segfaulted in SYCL PersistentDeviceCodeCache/getSortedImages during XCCL
all-gather, before ready. Post-health on both cards HEALTHY; no reset.
The passing C7b recipe had disabled persistent SYCL cache; the X1 control
left it enabled. Prepared retry disables it and uses a private cache dir,
after compiled XCCL health in the campaign runtime with the P2P override
unset. Preserved raw evidence: flash-mtp0-sycl-cache-crash.log and
window-flash-mtp0-sycl-cache-crash.out. The compiled collective retry
passed world_size=2, shape=4x5120, ten compiled iterations; both-card
health passed. The first collective diagnostic needed the already
installed ocloc on PATH; no GPU reset was required.

The next control reached graph warmup but found an int32 token-ID versus
int64 PLE history index_put mismatch. Host staging now converts IDs and
positions explicitly to int64. The CPU fixture also exercises int32
inputs (ple-cpu-int32.log, pass). That failed server's post-health passed
on both cards. Preserved evidence: flash-mtp0-int32.log and matching out.

The final no-MTP control reached ready, recorded four ordinary-EOS C1
completions and two concurrent completions, and passed identity and
coherence. Full decode graphs and the PLE position ring are active.
Configuration: BF16, IQ3_XXS main GGUF, TP=1/DP=2/EP, max length 2048,
batch limit 512, two sequences, no prefix cache, C7b FA overlay,
SYCL persistent cache disabled, B70_PREFILL=0, B70_FUSE_SHARED=0,
and pinned _C.o3.so. C1 p50 ITL 40.23 ms, 23.98 tok/s; C2 p50
44.35/44.18 ms, 22.74/22.74 tok/s. No target speed or long-context
qualification is claimed. Raw evidence: flash-mtp0.log,
flash-greedy-mtp0.json, flash-bench-mtp0.json and window-flash-mtp0.out.

Full Flash-Next MTP2 reached READY17:35:42. Both EP ranks completed the
draft payload audit (30 slots); target graphs used 0.24/0.26 GiB and
draft graphs 0.03 GiB/rank. Identity and deterministic short coherence
passed, but GREEDY_RC=3: the first C1 prompt diverges at zero-based token
17; the remaining three 96-token C1 outputs match. Concurrent prompts
diverge at tokens 4 and 72; these divergences precede EOS. The original
ordinary-EOS gate is retained in flash-greedy-mtp2.json. Post-health
HEALTHY on both cards, clean teardown, no reset.

| Flash-Next config | C1 chunk p50 ms | C1 effective ms/token | C2 chunk p50 ms | C2 effective ms/token |
| --- | --- | --- | --- | --- |
| MTP off | 40.23 | 41.70 | 44.35 / 44.18 | 43.98 / 43.98 |
| MTP 2, unqualified | 85.20 | 30.27 | 83.37 / 83.53 | 32.89 / 32.01 |

Effective time is 1000 / decode_tok_s. C1 token/s 23.98 -> 33.04;
C2 22.74/22.74 -> 30.40/31.24. MTP benchmark acceptance 171/222 =
77.03%; aggregate after the greedy and benchmark requests 528/700 =
75.43%. Speed is diagnostic until the exact gate passes; these short
2048-context runs establish neither 2x200K behavior nor stability.
Raw: flash-mtp2.log, flash-bench-mtp2.json, window-flash-mtp2.out.

The first card-0 primitive diagnosis (shape-probe.log) is bit-identical
for sampled actual rt HC/router BF16 weights at rows 1 versus 2/3/6,
and native full-shape GDN three-row verification versus three sequential
decodes, including final FP32 SSM state. This narrows but does not locate
the full-model divergence. The extension (shape-probe-r2.log) is also
bit-identical for packed Q8 GEMV at rows 1 versus 2/3/6 and the C7b FA
speculative path versus sequential decode at past lengths
16/64/128/512/1536, full 24Q/2KV heads of width 256. Both probes exited
zero in 21/16 seconds after the lease. These are sampled primitive
checks, not proof that the entire model is batch invariant. A same-load
no-MTP repeat gate is queued before another full-model workaround.

The first repeat attempt passed its own pre-health but acquired the serve
lease after X3's separate collective diagnostic timed out. X1 read the
new X3 warning during startup and stopped the load before READY, releasing
both cards. No repeat token requests ran. X2 subsequently passed compiled
two-rank health; X1 per-card post-health is pending. Preserve this attempt
as a scheduling abort, not a model crash or repeatability result.

Final CPU sidecar audit also passes after splitting the fused MTP expert
gate/up on CPU, avoiding the generic post-load hook's redundant fused GPU
copy (sidecar-cpu-final.log). This saves a derived 225 MiB per EP=2 rank;
it is not yet a measured end-to-end VRAM saving.

## Review and scope

Host baseline is 7.1.0-070100-generic. Fork HEAD at measurement is
0a6170da8ca9991972322fa1ca1d27674f0e75cd, with the shared campaign's dirty
changes preserved; runtime version string is 0.32.1rc1.dev94+g5fd62688f.
The X1 source capsule is x1/source-manifest.json, mtp_loader.py,
mtp_ple.py and fork-tracked.patch. The tracked patch includes pre-existing
shared changes and is not a standalone clean upstream patch.

Both requested Claude CLI consultations (fable-5.1 and claude-5.5)
returned unrecognized_model and Not logged in. No actual responding model
ID or review advice was received. Evidence: x1/fable51.txt, claude55.txt.
No extension kernels were rebuilt for X1. The compiled collective-health
JIT used MAX_JOBS=4 and TORCHINDUCTOR_COMPILE_THREADS=4. Existing GGUF
library is pinned to _C.o3.so, sha256 be76661bd120aa2f6c7523ae3c08813ba56d6114660f3d7f31183c32e5e861eb.
