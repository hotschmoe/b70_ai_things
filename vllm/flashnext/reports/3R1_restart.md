# R1 -- post-outage campaign restart (2026-10-21)

CONFIG: unchanged host kernel 7.1.0-070100-generic and runtime. User
resumed the Flash-Next vLLM campaign and authorized startup serving down
for testing. Sol 6.1 high owners: D1 decode, N1 GDN determinism, Q1 QSA.
Strata remains the daily driver; no model or shelf promotion.

COMMAND: stop startup serving; `bin/gpu-run bin/xpu-health`, then
`bin/gpu-run bin/xpu-collective-health`. Run merged O4+F1 with
`TAG=restart-r1 MAX_LEN=131072 MAX_BATCHED=12288 DECODE=64 PREFILL=1024
F1_LENGTHS=8192,12288 F1_COHERENCE_LEN=8192 F1_REPEAT=1 F1_DECODE=0
RUN_PROMO_GATE=1 SKIP_HEALTH_BEFORE=1 bash
vllm/flashnext/harness/f1_merged_serve.sh`.

RESULT: native per-card health and ten compiled two-rank collectives
passed with P2P disabled. Baseline ready 16:15:33 UTC, normal teardown,
post-health HEALTHY. Raw evidence is under
`/mnt/vm_8tb/b70/build/vllm-flashnext/restart-20261021/` and
`f1/merged-restart-r1/` in the same build root. Effective O4 GGUF, F1 EP,
F1 bucket and O4 helper libraries retained their SHA256s. Tracked fork
diff before/after the serve was identical. The fork remains dirty;
this is not a frozen submission engine.

| Baseline check | Result |
|---|---|
| First ID / registered research alias | hotschmoe-dd / qwen3.8-flash-next-ista-gsq-rco-iq3-xxs-o4 |
| Arithmetic / coherence | 323, 667, repeated Paris pass |
| 8K / 32K needles | pass / pass |
| C1 ITL p50, 64 outputs | 15.1669 ms |
| C2 ITL p50 per stream, 64 outputs | 19.5596 / 19.6108 ms |
| 8186-token prefill, first measured request | 4327 ms, 1892 tok/s |
| 12283-token prefill, request-completion proxy | 5048 ms, 2433 input tok/s |
| Strict exact-repeat promotion gate | FAIL |

All C1 exact repeats match. C2 differs from C1 for multiple prompts;
the hash-table prompt also differs between the two C2 rounds. This is
both batch-shape variance and a repeatability concern. N1 addresses one
identified projection source, not every possible source of divergence.
The single 8K timing is not a matched warm regression measurement.
The 12K row emitted no content event (`n_chunks=0`); its `ttft_ms` is the
client's request-wall fallback, not observed first-content TTFT. The 8K
row did observe content. Neither rate isolates engine-only prefill time.

The attempted 512-token canonical collector stopped before measurement:
stock `vllm serve` lacks its required `/server_info` endpoint. No suite
rate was produced or submitted. A shim and an integrated callback are
prepared for the next matched window. The original harness printed a
zero shell exit despite its recorded gate RC3; it now propagates requested
gate failure after collecting diagnostics and completing teardown/health.

VERDICT: the short decode baseline reproduced the handoff after reboot;
strict promotion remains rejected. Run bounded component gates before
another full model load. Future suite collection must run inside the
serve window with the metadata shim and before teardown.

## Recovered work and setup cost

The handoff understates the completed work. D1 recovered Ld's successful
trace and post-health; Q1 recovered a synthetic integration failure and
a subsequent CPU binding repair. See their individual reports.

F1 q2 also completed before the outage: exit0, pre/post HEALTHY, arithmetic
and long coherence pass. Parsing real T12283, E374 rows (48 layers) gives
median dequant13.35 ms, combine9.80, bmm1 9.43, bmm2 5.405, gather4.415;
mean stage sum47.271 ms. T8186 card1 mean40.542 ms. These instrumented
stages synchronize after every operation and do not establish clean
end-to-end latency. Synthetic T12288 profile rows are excluded. Raw
summary: `restart-20261021/f1-q2-recovered-summary.json`. No repeat of q2
is necessary; dequant and combination are concrete prefill candidates.

Consultations with `claude-fable-5-1` and `claude-opus-5-5` each timed out
after35s (exit124, no responding model ID). No advice is attributed to
them. All new owners actually use gpt-6.1-sol.

## Matched control and rejected attention treatment

The integrated metadata shim and callback produced a complete control
on the same unchanged recipe: reasoning61.642817 tok/s, code61.800831,
three measured 512-token requests per suite. Source/SO checks, isolated
request counters, zero draft activity, identity, coherence, normal engine
teardown and both-card post-health pass. Strict C1/C2 repeatability fails.
The independent audit recomputed rates from raw event arrival times.

The first plan16 serving attempt was discarded after an integration guard
was found to reject the real server's scale tensors. After a corrected
wrapper gate passed and activation was proven in the real server, r2
failed with native level_zero error40/OUT_OF_RESOURCES during a mixed
20-token prefill plus one decode token. The traceback surfaced at a PLE
ring gather; this does not establish the cause. No candidate canonical
suite completed. Shell SERVE_RC0 describes cleanup and does not negate
the engine failure. Both-card post-health and source/SO checks passed.

Plan16 stays disabled and unqualified. There is no new headline speed
result. Full-model GDN validation proceeds on the original attention path.
See reports/3A1_audit.md and reports/3D1_decode.md for the finite evidence.

## Subsequent bounded work

N1's full original-attention serve passed arithmetic, short coherence and
8K/32K needles, but retained a C2 repeat mismatch and then stalled on an
arithmetic request after prefill. EngineCore required a forced kill.
Sources/SOs were unchanged and both-card post-health passed. The 128-token
repeat packet and canonical suites did not complete. Keep the N1 flag off.

A bounded original/temporary-plan/persistent-plan stress test passed
1501 B1 plus501 B2 replays per arm through graph re-instantiation every500,
including a PLE8K gather and mixed21-token PLE convolution. It used low
VRAM and no EP sidecar; it does not clear loaded-context resource pressure
or queue interactions. No speculative lifetime fix is justified by it.
Tracked reproduction: harness/d1_fa_lifetime_window.sh (requires D1_TAG).

P1's selected-expert dequant primitive passes20 real-weight BF16 cases,
including all five gate types, two down types and three store layouts.
The measured mode0 calls improve6.9-23.3 percent versus gather+dequant,
with32 reordered experts; this is component timing, not serving speed.
Measured native sources are in plugins/b70_p1_xpu and scripts in
harness/p1_*. No serving hook or automatic registration was added.

Strata references for4096/16384/32768 tokens are now measured using the
canonical dd-nomtp-c1 config (research alias/log only changed). All three
produce64 tokens, with drafts_offered=0/drafts_accepted=0, matching native
tokenizer IDs, unchanged sources and healthy pre/post checks. Raw rows are
q1/strata-references.json in the build root. Full QSA fidelity is pending.

The lifecycle now rejects explicit native/engine failures and forced
EngineCore shutdown even when cleanup returns0. The repeat client keeps
primary HTTP errors and marks missing rounds incomplete instead of raising
a secondary KeyError. Neither change alters model computation.

## Full QSA closes; next work is controlled localization

The first full QSA serve completed all 4K/16K/32K fixtures. All 12 real
pooled bindings, identity, arithmetic and graph initialization passed.
EngineCore drained and exited normally; ENGINE_ERROR_RC0, SERVE_RC0,
both-card native post-health HEALTHY, sources unchanged. The overall
window correctly returns1 for failed exact-repeat/reference gates.

Strata text matches, ordered solo1/solo2/C2a/C2b, are [true,true,false,true]
at4K and allfalse at16K/32K. At16K solo repeats differ; at32K all four
vLLM texts agree with each other. These are exact comparisons of64-token
reasoning prefixes, not token-ID equality or completed-answer quality.
The32K C2 first stream has p50=77.10ms but p99=8972.91ms and mean386.78ms,
versus the other stream's mean76.49ms. No causal timing explanation or
serving stability claim is established. See reports/3Q1_qsa.md and the
independent audit in reports/3A1_audit.md.

VERDICT: QSA integration works in this bounded full-model attempt; fidelity
and stable serving remain unqualified. Compare the same fixtures against
QSA-disabled vLLM before attributing drift. Loaded-context/EP liveness
and repeatability remain prerequisites for enabling N1 or D1 plan16.
P1 has a measured primitive and reproducible source, ready for a separate
default-off serving hook and matched control/treatment. No speed promotion
or external submission.

At17:19 UTC the user directed continued vLLM testing without restoring
Strata between tests. Its restoration had started, but the frontdoor
remained502 before readiness; no identity/arithmetic smoke completed.
Stop the service again and keep research ownership. Q1 prepares the same
long-fixture control with QSA disabled; P1 prepares a default-off serving
hook; D1 prepares loaded-context/EP liveness localization on CPU.

## Continued campaign: original-FA control and P1 integration

Q1's QSA-disabled control completed at17:30:10 with normal EngineCore
drain, unchanged sources and both-card native post-health HEALTHY. It
also fails solo exact repeats and16K/32K exact Strata text parity; these
failures are not unique to QSA. Individual32K C2 arrivals show the first
stream's long gaps while the other stream awaits first content. This is
an observed overlap, not a scheduling root cause. Q1 records the complete
per-output matrix and diagnostic latency comparison in3Q1_qsa.md.

P1's isolated native bucket overlay then passes44 bit-exact outputs
against the original operation, covering original repeat and flags0/1,
five real quant pairs and a T12283 prefill. Body51s, exit0, unchanged
sources/SOs, native pre/post HEALTHY. Its full serving control starts
next, followed by treatment only if control/source/fixture gates pass.
Both arms use the same binary and maxlen65536/chunk12288; sources stay
frozen across the pair. The new client records observed first-content
TTFT, wall time and individual arrival/ITL samples separately. It does
not substitute request completion for missing content.

P1's pair subsequently passes in full. Warm-excluded first-content
medians control->selected are3341.403->3198.826ms at8189 prompt tokens,
4876.350->4678.923ms at12285, and13949.494->13509.475ms at32765:
latency reductions4.27/4.05/3.15percent. All24 requests across both arms
produce16 content events and16 usage tokens. Long arithmetic passes in
both, actual config is no-speculation/prefix-off, source/SO/fork/fixture
identities match, and EngineCore teardown plus native health pass. Short
decode means remain effectively unchanged. This is one fixed-order pair;
continuations are not strict repeats in either arm. Keep the optimization
opt-in and distinguish this measured diagnostic gain from promotion.

Before the next N1 reproduction, root review caught two monitor defects:
the real GDN call passes hidden_states by keyword, and main/card1 threads
cannot share a single active-phase LIFO. CPU checks now cover both, keep
original results/exceptions when diagnostics fail, and avoid coercions
that could touch GPU data. Sources are ported and reviewed before load.
The next finite run uses the original N1 sequence with phase/stack capture;
QSA's minimal integer-reduction candidate/profile follows it.

N1 instrumentation ran successfully but the engine SIGSEGV occurred
earlier, during promotion repeat: last executeT6 scheduled[1,5]. Saved
/proc main-thread wait status11 identifies SIGSEGV before DockerTERM;
container143 is cleanup only. No native PC/cause is localized. There was
no >=8s no-return witness and the historical F1/callback point was never
reached. Sources445/N1/private hashes and both-card post-health pass.
The original offline classifier confused shell-body RC0 with successful
requests; a separate corrected classification preserves that raw error.
N1 remains disabled. Further reproduction needs native fault capture.

QSA profile-r1 rejected the trace label before workload execution; a
label-only fix and fresh profile-r2 then pass exact key/canary/replay
gates and native pre/post health. The actual captured32K M1 chain spends
about4.02/4.64ms in top-k. A group reduction saves only4-5us versus a
copied tree compiled identically; the larger native-to-copy difference
changes compiler settings, so it cannot be credited to the reduction.
The next bounded candidate uses exact upstream int64 selection/sort.

## Native-capture startup failure and recovery

CONFIG: original-FA/N1-on diagnostic, two GPUs, debug-only derived image
with unchanged existing package versions. GDB attaches only after READY.
COMMAND: TAG=n1-native-r1 bash harness/d1_native_capture/native_window.sh;
after full closure, bin/xe-reset (auto, both leases, P2P0 health).
RESULT: before READY, card0 CCS memory CATerror18 and DEVICE_LOST20 during
EP initialization. GDB never attached; no model requests ran. N1 installs
its hook at construction but loads its SO/allocates/launches only on first
apply, so this is not evidence of an N1 kernel fault. Container137 is
cleanup; actual primary failure is DEVICE_LOST. Sources471/N1/image/private
guards pass; both-card post-health passes. Kernel events preserved; root
authentication was unavailable for the Xe devcoredump. Reset rebind then
passes both native cards and compiled two-rank allreduce10 iterations,
P2P0, unchanged boot ID;66s total. No reboot or driver change.
VERDICT: no native PC for the earlier SIGSEGV, no further blind N1 retry.
Keep N1 off and continue the independently prepared QSA work.

## QSA selection critical path

CONFIG: physicalcard1/logical0, native health, six real M1/M2/context
shapes,147456 score columns, exact packed-key ties and ascending IDs.
COMMAND: isolated q1_torch, q1_fused and separate q1_fused_r2 component
windows, each quiet CPU/GPU lease and original native comparator.
RESULT: Torch int64 topk/sort passes exactness/canaries/changed graph
replay and reduces32K M1 chain from about4.66ms to1.50ms. Serial fused
compaction also passes exactness but regresses to about4.33ms; its linear
scan dominates. Parallel bitonic ordering of selected IDs passes the
next component gate with about1.36ms32K M1 chain. All native pre/post
health and source guards pass. These are populated components without
loaded-model VRAM pressure; not serving speed claims.
VERDICT: preserve all measured variants. Before serving r2, independently
reviewed hook must select only persistent decode-owned buffers, retaining
one workspace per batch and keeping native prefill. Watchdog must allow
the historical139s full32K fixture phase. Measure fresh native and r2
serving arms with identical sources/configuration, SOs and fixture IDs.
