# Pi bang investigation, 2026-09-10

Status: investigation active; production hotschmoe-dd remains offline.
No shelf promotion or production stability claim has been made.

## Answer to the GDN and TP questions

Steve's GDN phase finding explains a reproduced local bang mechanism. The
R276 builder treated a fresh one-token prefill as recurrent decode and could
read stale/uninitialized state. A one-call-site phase guard fixes that case.
It is not inherently TP2: stock fails on each physical card at TP1, while
phase-only candidates pass on both cards in a crossover. Native libraries
are unchanged. This does not establish that the original long Pi requests
hit the same internal phase; their exact upstream bodies are absent.

CPU execution of the actual scheduler further narrows that connection: a
cached or chunked one-token suffix normally has positive computed context
and existing recurrent state. In aligned mode, a fresh long prompt clipped
to a one-token first chunk is deferred at the reviewed block/budget settings.
Long history plus a singleton suffix therefore does not prove the reproduced
zero-state initialization defect occurred. Request-level trace correlation
or independent invalid-cache evidence is still needed.

CONFIG -> R276 stock versus phase-only, TP1 on both cards,8K/c4/batch512,
eager, MTP0, prefix off, FP16 KV. COMMAND ->24-case pinned tiny-prefill screen,
then swap the images between cards. RESULT -> Each stock run2 bangs / 24;
each patched run0 / 24, all 24 patched token arrays match between cards, and
all 22 nonfailed stock arrays match patched. VERDICT -> Causal bounded
reproduction, with normal teardown and the then-current selected-card health
checks. Later strict health is stronger than that old sentinel gate.

CONFIG -> Current official vLLM0.29 XPU,8K/c1/batch8192, other isolated
settings as above. COMMAND -> Stock and phase-only24-case screens on card 0.
RESULT -> Stock has 6 singleton loops, patched passes 24/24; patched arrays match both
R276 patched runs. Normal teardown and strict selected-card health pass.
VERDICT -> Updating alone does not fix this defect. This is a bounded
screen, not concurrent-batch or long-session qualification.

## Full-feature and harness evidence

R276 phase-only TP1 with MTP3, calibrated FP8 KV, prefix caching and decode
graphs completed the nine reconstructed Pi requests without bangs. Two
agent2 responses still fail broad quality checks; that supplied history
already contains corruption. Previously looping agent1 target requests now
finish without repetition flags. This temperature 0.7 replay is not a causal
proof or a semantic pass. Normal teardown and strict post-health passed.
The phase-patched TP2 full-feature run also passes all four stages: nine Pi
reconstructions, 24 tiny-prefill cases, three earlier clean-history requests,
and 32 concurrent tool checks. Tiny-prefill sequential and mixed token arrays
match exactly; server logs confirm four simultaneous requests. Teardown,
strict per-card health and compiled pair post-health pass. This is a bounded
100K-context qualification, not production promotion or a long soak.

The selected affected bundle contains 41 bangs among 317 non-429 messages; this
is not an unbiased production rate. All 36 recorded bang-recovery messages
immediately hit the harness's per-agent429 admission guard, 0-3 ms after the
aborted message was persisted. That is a separate recovery/admission race.
Gateway source and exact failed wire bodies are still needed to fix and
correlate that layer. No external tool calls from replayed histories execute.

## SGLang is viable, but not yet an equivalent replacement

The latest published Intel baseline, with a deliberate source-only XPU
pointer/device port, passes 26/26 bounded text/tiny/mixed requests. Explicit
Mamba extra_buffer with radix caching passes 32/32 tool checks across four
growing histories, with four requests running and measured cache reuse.
Both lifecycles exit0 and strict selected-card post-health passes.
These results use TP1, FP16 KV, eager mode and MTP0.

Current pinned SGLang source has been rebuilt from source with GPTQ and GDN,
explicit BMG, and Triton attention. Unused native FMHA/MoE/MLA kernels are
excluded; this is not a native vision-FMHA recipe. No quarantined binary is
copied into the stack. The CPU installed/API gate passes: all 63 installed
native artifacts match the rebuilt wheels and the held PyTorch/UMD/oneCCL
identities are unchanged. The current-source image passes strict per-card and
compiled pair health. Its first serve failed when automatic image warmup
called the deliberately excluded native vision FMHA. A text-only retry using
the supported skip-server-warmup flag passes all 26 real text checks,
including four-request mixed batches, with normal teardown and strict card1
post-health. This is TP1/FP16 KV/eager/MTP0/prefix-off only. Image requests are
not supported by this reduced recipe's default vision backend.

Two material parity gaps need qualification: native Intel attention prefill
omits FP8 descales, and Qwen lacks the required calibrated-scale loader. A
strict source-only loader and mapping are prepared for Triton. In addition,
current XPU MTP verification selects greedy tokens even for temperature 0.7.
A bounded unseeded sampling candidate passes CPU checks but is not activated.
Seeded sampling needs a shared ordinary/speculative mapping; the optional
FP32 prototype changes the legacy seed mapping and is not a drop-in parity
fix. Graph/MTP/FP8/TP2 parity
has not been demonstrated by the simpler passing screens.

## Calibration identity finding

All eight current safetensors match the publisher's pinned revision. Seventeen
model files were rehashed; only shard 2 differs from the old calibration
provenance. Ordinary and no-fallback direct disk reads both match the publisher.
The old freeze utility copied an existing manifest without rehashing weights.
This does not establish why its recorded shard 2 hash differs, or which bytes
the old calibration actually loaded. The old artifact is preserved unchanged.

The new loader refuses the mismatch. Fresh calibration completed against a
verified complete manifest: 256 short cases, four long prompts through 96K
tokens and two 4096-token continuations. All 17 model-file hashes and stat
identities match before and after collection. Teardown, strict per-card and
compiled pair post-health pass. The fresh 17-layer artifact is frozen with
SHA256 be02d915a8ac188341870cc9f642d77665b744235e142330a7a37b8f4c711062.
The recorder requires eager/prefixoff; graph and prefix-on FP16-versus-FP8
qualification must be separate. Current collection is limited to a 100K
configuration and does not qualify the previous 200K configuration.

The 256 shared short cases have 247 exact text-plus-reasoning matches with the
old collection; nine differences are ordinary wording/formatting changes.
Twenty-five of 34 K/V scales are exactly equal; remaining fresh/old ratios
range from 0.95085 to 1.01898. Calibration configuration and long prompts
differ, so these comparisons do not resolve the historical weight identity.

CONFIG -> Phase-only vLLM R276, synthetic distinct K/V scales, 67 tokens,
permuted 64-token cache blocks, each physical card. COMMAND -> Actual cache
write and attention-read oracle, including exact representable bytes and
untouched slots. RESULT -> Both cards pass; attention relative L2 is below
0.000227 against the dequantized CPU reference. Strict per-card and compiled
pair pre/post-health pass, with owned containers removed. VERDICT -> This
bounded numeric test validates scale consumption, not full-model FP8 quality.
The corresponding SGLang oracle initially fails its exact-byte comparison on
card0. A diagnostic isolates all differences to XPU's scalar divisor being
rounded to FP16 before division: 156 K bytes differ from the original CPU
reference; identical-input casts and scatter agree exactly. Independent
power-of-two stores pass. A revised independent reference preserves this
distinction, matches all bytes, then reaches a failing attention comparison
(48/6144 elements outside the unchanged tolerance). Grouped decode source
also quantizes Q and softmax P to FP8, beyond cache quantization. That compute
policy is being isolated; no FP8 attention pass is claimed. Both failed
lifecycles preserve logs and pass the applicable strict post-health checks.

## Evidence and reproducible sources

- ../vllm/int4/20260910_bang_isolation_campaign.md
- ../vllm/int4/20260910_runtime_freshness.md
- ../vllm/int4/20260910_cache_correctness_review.md
- ../vllm/int4/diagnostics/README.md
- ../sglang/cache800k/20260910_feature_parity_review.md
- ../sglang/cache800k/calibrated_kv/README.md

Raw immutable plans, requests, responses, logs, identities, comparisons and
health results are under /mnt/vm_8tb/b70/results/bang_isolation_20260910.
The extracted redacted bundle is under results/bang_investigation_20260910.
Host kernel/drivers are unchanged. Experiments use per-card or paired gpu-run
leases with matching pins and P2P0. Failed lifecycles retain their evidence.
