# strata/flash-next

Live: model-lock-ista-iq3.json (ISTA GSQ-RCO IQ3_S and IQ3_XXS artifact hashes).
The serving build and configs are outside the repo under
/mnt/vm_8tb/b70/build/strata-ab; launch notes are in JOURNAL.md (2026-10-13).

Retained clean control config: serve-iq3xxs-200k.json, stock upstream
296b59d2/v0.1.44.1. Split27, two200k, MTP4, int8 KV, all main experts in
VRAM, parking16384 MiB/4 entries. Strata spec4 includes the seed and offers
at most three MTP drafts. Bounded capacity, forced RAM restoration,
concurrent coherence, reasoning/tool checks and teardown health passed.
Evidence: docs/20261014_flashnext_strata_optimization.md. The soak observer
is stopped at user request; multi-day stability remains unqualified.

Start persistently after the previous endpoint has stopped and GPU health
has passed:

```sh
systemd-run --user --unit=strata-optimized --collect \
  --property=WorkingDirectory=/mnt/vm_8tb/github/b70_ai_things \
  /bin/bash /mnt/vm_8tb/github/b70_ai_things/strata/flash-next/launch.sh
```

Stop: `docker stop -t 30 strata-ab-serve`. Endpoint: :18085/v1,
`hotschmoe-dd` first, detailed IQ3_XXS alias retained.
Switch to 27B: `bash strata/flash-next/fallback-27b.sh` (:18080/v1).
The wrapper passes the existing prerequisite check and shell syntax check;
its complete model switch has not been exercised in this stage.

Probe scripts operate through the already-leased server:

- bench_serving.py: raw SSE, live identity, matched fresh prompts/timings;
  --spec-min-p varies the per-request draft-confidence cutoff.
- bench_concurrency.py --template <saved-coding-response.json>: matched two
  overlapping requests, aggregate output, per-stream timings and hashes.
- bench_context.py --parking: near-full concurrent retrievals, overwrite both
  GPU slots with short chats, then verify concurrent cached-prefix returns.
- soak.py: HTTP metadata, container RAM, per-card DRM fdinfo, and error lines;
  no test prompts, GPU queries or automatic restarts. Stop its persistent
  observer with `systemctl --user stop strata-soak.service`.

Optional telemetry follow-up: `patches/batch-leg-draft-carry.patch` applies
after `patches/batch-mtp-restore-accounting.patch`. It preserves optional
draft totals when a batched request finishes through solo decoding. S30's
CPU red/green and live native-sum/API oracle pass; see
`docs/20261015_strata_batch_draft_carry.md` from the repository root.
This does not repair historical S29 draft counters or change kernel work.

Experimental EP: serve-iq3xxs-ep-experiment.json uses the temporary upstream
PR2205/1798 build. Decode improves, but prefill regresses to about580 tok/s;
this is not the daily-driver config. EP/spec3 also failed a new tool
continuation that completes on layer/spec4. Two-stream EP total improves
about10%, but both measured paths offer no concurrent MTP drafts.
launch.sh adds SYS_PTRACE only for EP
to permit Level Zero IPC with Docker's default seccomp profile. Upstream:
https://github.com/Niko1221/Strata/pull/2468. ep_ipc_probe.cpp records import
status; ep_transfer_probe.cpp failed before large-transfer measurement and
was revised to retain full-sized native mappings across modes; see S5.
The fixed-lifetime S5 transfer probe passes24 returned-byte checks, but DMA
is slower than the uncached kernel (1.65 vs3.32 GB/s on the large buffer).
The S3 failure remains recorded; its cause is not isolated. Build/run and
recovery evidence are in the S3/S5 document/raw logs.

The paused UD-Q4_K_XL RAM-spill campaign code that used to live here was
moved to the gitignored archive/flashnext-udq4xl-paused-20261014/ on
2026-10-14 and remains in git history before that commit. Its docs are in
docs/flashnext_udq4xl_paused/.

Opt-in FP32 peer reduction: patches/ep-prefill-reduce.patch applies after
upstream PR2205 on296b59d2; serve-iq3xxs-ep-reduce-experiment.json points
to the frozen tested binary.26-case two-card numerical probe source is
ep_prefill_reduce_probe.cpp. Matched32K prefill576 ->1106,8192 chunk1265;
text can differ due to FP32 regrouping. Functional and teardown checks pass,
but no full-context candidate qualification or daily-driver promotion.
Upstream contribution targets PR2205's author branch:
https://github.com/recutita/Strata/pull/1. S6 records full evidence.


Optional EP PUSH return experiment: `serve-iq3xxs-ep-push-experiment.json`
uses frozen `strata-push-s9`, exact workspace12288 and both
STRATA_EP_PREFILL_REDUCE=1 / STRATA_EP_PREFILL_PUSH=1. Same-binary pull/PUSH
32K medians1308.4/1530.9 tok/s and25.077/21.444s client TTFT; nine matched
responses exactly equal. Reduction changes FP32 addition order relative
to full-row combine. This configuration has short-prompt/functional and
teardown-health evidence, not full200k or shelf qualification. See S9.
Build only the `strata` target when editing this patch; the optional
standalone numerical probe also accepts `--push`.


For operation latency use Intel unitrace -d -h -s with Chrome kernel/API
logging around the leased native engine. Stop/flush the session and run a
canary before native shutdown (_Exit bypasses profiler destructors). Capture
SYCL graphs with collection ON before pausing, or replayed kernels are
missing. Match the actual serving config, not just the stable model alias.
`summarize_trace.py trace.json --output <directory>` exports counts, GPU and
host operation median/p95/range plus per-device busy interval unions. Use
--start-us/--end-us for retained request windows and an explicitly calibrated
--timestamp-shift-us when the profiler clock differs from the client clock.
Retain raw trace, submission report and calibration. Kernel copies require
source geometry to annotate transfer bytes; profiled host copy calls include
queue waits. Measure speed separately without tracing. See S10 for complete
prefill/decode census, the clock issue and matched timings.


S13 rejects an isolated attention value-transpose/four-row preload:36 INT8
cases are numerically exact, with no useful latency gain. The rejected diff
is in experiments/attn-value-gather-s13.patch; attn_value_probe.cpp compares
baseline and renamed implementations without model weights. It is not a
production patch.

Optional concurrent MTP: serve-iq3xxs-ep-batch-mtp-experiment.json uses the
same frozen PUSH-s9 plus STRATA_BATCH_MTP=1. Solo spec4 offers at most three
drafts; upstream's batch path offers ONE proposal per slot. CPU-quiet paired
coding improves total68.72 ->93.89 output tok/s, all experts remain resident,
and functional/teardown health passes. Extra MTP state changes expert
placement. Cross-arm texts differ, no candidate full200k/parking promotion.
Native batch output proves accepted proposals, while HTTP draft counters
remain incorrectly zero because BDONE lacks the slot counts. Do not use
those zeros to infer that this opt-in path is not drafting. S14 has evidence.

bench_mixed.py starts a fresh32K prefill after a second stream begins decode;
bench_serving.py retains SSE content-event arrival times and p50/p95 gaps.
These are client-visible event gaps, potentially several tokens per event.
bench_context_four.py requires32768MiB parking, samples native RSS and
container anonymous/file/swap RAM at50ms, and preflights token budgets for
four distinct near-full conversations. Long resident replies prove two
active decode slots before RAM exchange pairs. It saves completed phases;
use --resume with the SAME --output, --config, live engine PID and container
to continue after a harness failure. Keep a healthy server/cache running
while inspecting failures. A restart destroys RAM snapshots and requires
fresh prefills. Classify actual native restores: a return intended to remain
resident can move through RAM during its partner's slot handoff. Unknown
mixed attribution stays unknown. Cached-return p95 TTFT 2s including RAM
restore is a diagnostic reference, not a hard gate (user, 2026-10-15).
Once true RAM recovery avoids recompute, prioritize concurrency/decode/MTP.

patches/batch-mtp-restore-accounting.patch (upstream Strata PR2511) adds
optional slot accepted/offered counts to BDONE and merges them with admission
counts. Legacy five-field lines remain supported. It also logs snapshot
restore payload bytes, using buffer sizes and the actual restore API copy
arguments. These are not physical PCIe traffic: host-authoritative pools
and separate residency refills have different transfer behavior. Apply on
main296b59d2; runtime validation uses the existing EP/PUSH overlay. Frozen
PUSH-s9 still has the old native counters. S16 records the failed full-cache
control; S20 records successful full recovery after the cache fixes.


Fast cache correctness oracle: use --target4096 --context-size8192 with a
matching8192-slot config on the already leased server. Full qualification
keeps default --target199600 --context-size200000. S17 reproduces both
retained-tail replay and total cache loss with15988 fresh input tokens in
42s of test phase (startup/health excluded); run it before huge prefills.

patches/batch-slot-cache-preserve.patch parks a completed slot before either
overwrite path, using its own session/draft state, and carries its checkpoint
chain back to main. S19 binaryb09c9bf3 passes the fast oracle: all four RAM
returns reuse4335 held tokens with zero replay, four draft readbacks pass,
and all10 available baseline response hashes match. Functional and teardown
health pass. Full200k is untested for this patch; no shelf promotion. Measured
scope is text-only EP with batch-MTP on. CUDA/pipeline are unexecuted; image
slots and MTP-without-batch-MTP skip slot parking. A covering continuation
does not guarantee exact restoration of arbitrary shorter branch endpoints.


S20 adds patches/batch-slot-image-reuse.patch (one image buffer per stage,
not per layer) on top of S19. Full4x199501 qualification PASS: all four
forced RAM returns reuse199839 held/read31 new/zero retained-tail replay;
two-active proof, needles, functional/teardown health pass.32GiB/four-entry
cache budget; peak native RSS16.19GiB,0swap/OOM. Return TTFT2.758-6.423s
is diagnostic only. No routine daily restore; clean control remains retained.

Experimental concurrent depth: patches/batch-mtp-depth.patch applies after
batch-mtp-restore-accounting.patch. Set STRATA_BATCH_MTP_DEPTH=2 with
spec4 to offer up to two proposals per slot; default1, cap3. S21 matched
coding median93.064 ->98.901 aggregate tok/s (three pairs), all six hashes
per arm match; two9707-token outputs also match. Depth3 screen93.156,
no extra repeats. Context-end guards prevent invalid proposed rows. This
is a bounded SYCL greedy test, not depth2 full-cache or shelf qualification.
bench_concurrency.py --start-repeat N extends a promising screen with
fresh prompt IDs. Next profile main verifier and serial drafting costs.


S22 rejects STRATA_BATCH_MTP_ASYNC copy-only/submit-finish prototypes:
same-binary99.462/99.422/99.357 aggregate tok/s, both hashes each match,
emit7.02/6.99/6.99ms. Two hardware traces show no draft-head overlap in
8 interior cycles; this is not a hardware-wide concurrency claim. Only
rejected patch evidence retained; runtime source returned exactly to S21.
Do not repeat without a new mechanism. GPU teardown/collectives pass.


Operation/data-path reference: [interactive offline map](data-map/index.html),
[Markdown guide](data-map/README.md), [readiness probes](lowlevel_probe/README.md).
S24 maps 40 stages and 170 S22 operation types; CPU-readiness and hardware
counter gaps stay explicit. Single-card probes are diagnostics, not serving gains.

S25 replays the actual Q5_K vocabulary head on both cards: allC1/2/3/4/6/8
bitwise/argmax/finite/guard checks pass. C2 costs1.31-1.32ms versus
twoC1 calls about2.01ms; same-depth slot-head batching is a source-only
candidate, not a serving gain. Existing production PLE logging measures
21 miss-ticket waits over24 batch windows (median667.9us,sum13.294ms).
See lowlevel_probe/README.md and stageS25; per-card/collective health PASS.

S26 adds experimental-batch-mtp-pair-head-s26.patch (default off, SYCL
only). CPU syntax, extracted host binding checks and one AOT build pass;
GPU serving is UNTESTED and held. Standalone EP freshness tests reproduce
current inbox poison after native flag waits; completed host waits and
uncached payload hints do not cure it. The latter patch is retained as
rejected negative evidence. No production output corruption is established.
Copy control and final per-card/compiled-collective health pass. See stage
S26 and lowlevel_probe/README.md; use frozen binaries for prior measurements.

S28 resolves the bounded standalone EP freshness oracle with one explicit
system memory scope on ep_push's workgroup barrier. Default mirror fails;
scoped mirror and actual patched EP TU each65/65 PASS. Frozen engine
strata-ep-scope-s28 SHA152055a6 passes concurrent64-token S16 prefixes,
draft counts, math/reasoning/tools, normal stop and final GPU health.
Head-pair option remains OFF and unqualified. No speed/fullcache/shelf claim.
Native BMG ISA was present in hidden files; S28 explicitly corrects S27's
contrary statement. Minimal patch: patches/ep-push-system-barrier.patch;
[evidence and limits](../../docs/20261015_strata_ep_publication_scope.md).

S29 tests STRATA_BATCH_MTP_PAIR_HEAD=1 on that same frozen152055a6 engine:
three fresh C2 coding pairs97.881OFF ->102.279ON aggregate tok/s (+4.493%),
all six ON/OFF output hashes match. Client per-stream medians53.399/55.620;
all24576 experts resident, functional/normalstop/both-card+collective health
PASS. This is a bounded candidate, not full depth2/headpair cache or shelf
qualification. Recoverable experimental config:
/mnt/vm_8tb/b70/build/strata-ab/serve-head-pair-mode1-s29.json.
An existing batch->solo accounting hole drops earlier draft totals; retain
raw counters but do not compare full-request acceptance until fixed. Tiny
in-serve C2 trace is queued, not measured. See stageS29 and data map.
