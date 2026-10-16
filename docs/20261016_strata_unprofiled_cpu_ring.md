# S142: remove unitrace for a matched CPU-ring diagnostic

CONFIG -> Same S133 executable (`fe57222e...`), same all-OFF flags, EP149/363,
paired MTPdepth2/headON and fixed native work policy as S136. One fresh ordered
C2 pair,107prompt/64output tokens per request. HOST_BOUNDARY and TRACE remain1;
unitrace injection, pause/resume/stop, raw GPU trace and profiler flush are absent.
No kernel, driver, runtime adapter, model or arithmetic changed. Queueprofiling
property work stays parked.

COMMAND -> Reuse S136 controller/client/ring parser with a direct engine wrapper.
Preserve exact current executable/process identity, output/nativework gates,
normal QUIT, fullCID/NAME disappearance, per-card and compiled two-rank health.
One corrected setup continuation and ONE actual GPU request pair; no benchmark
retry, generation warmup, functionalsweep or rawtrace parse.

The FIRST startup failed before native EXEC: server Popen raised PermissionError
because trace-engine.sh was mode0664. No native engine/model load/measurements
started. Original start exit1, wrapper1, logs/CID/rootreview remain under
first-startup-failure/. Root confirmed fullCID/NAMEgone and changed ONLY mode to
0775, retaining byte SHA `0d3f02e58b37fee7e26d0a801e9f4b3ee0614e1697f42fc479fdcb3a83d64304`.
The failed run was moved aside before the explicit corrected setup continuation.
No GPUcrash/reset/recovery or extra prehealth followed that permission error.
Future directly executed wrappers get `test -x` in preparation checks.

Root reused prehealth with exact kernel/boot/no-intervening-GPU attestation:
latest card0 S141 normal/posthealth0; card1+collective S139 ON fullposthealth0,
with only healthy-closed S140/S141 card0 work since. The receipt lived OUTSIDE
run/ before start (controller creates run/), then was copied inside after start.
This resolves the historical prep README's before-start run/ path contradiction;
no controller source change was made for it. Exact receipt is retained.

RESULT -> Corrected start0/30.464s, collect0/3.893s, stop0/59.166s,
CPUanalyze0/0.182s. Full request/text/concatenated SSE content, usage, perrequest
nativework, nativewindows/paired/solo/capturelayouts match S136. Raw SSE chunk
segmentation/IDs/timestamps are not equality claims. Both rings have820declared,
820actual records, dropped0, window0=0 and complete terminal window_end. Both
parsers select SAME16 replaywindows5..16 and19..22; all cold/solo/tail records
remain. Native normalQUIT, wrapper0, fullCID/NAMEgone, prestopRunning/notOOM,
cardhealth0 and collectivehealth0 pass. No final OOM state is invented afterrm.

| CPU steady-clock bracket (n16) | S136 unitrace median ms | S142 without unitrace median ms |
| --- | ---: | ---: |
| Verifier submission | 3.2213425 | 0.067378 |
| Pair submission | 1.9506695 | 0.046123 |
| Commit | 2.928889 | 1.585553 |
| Stage PLE | 0.709177 | 0.6686695 |
| Pair polling/PLE | 1.0989555 | 2.7100315 |
| Pair queue wait | 2.9238945 | 2.617585 |
| Whole window, diagnostic only | 57.1811135 | 46.4703525 |

The verifier bracket covers peer and primary graph submissions, error handling
and existing ext_oneapi_empty AFTER input staging. Its sample ranges are
3.062910..3.532445ms versus0.062059..0.103747ms. This supports profiler-bundle
perturbation of the observed CPU brackets, not attribution to one LevelZero API
or event reset. Pair polling starts earlier and includes dependency waits and
prefetch callbacks; its increase is not independently a PLE regression.

Direct per-window pair_submit_begin->pair_return intervals yield median
5.978394->5.370319ms. This combines actual endpoints BEFORE taking medians;
summing or subtracting the separate bracket medians would misstate the path.
The single fixed-order diagnostic is not gained serving performance, statistical
stability, a CPU/GPU clock join or proof that S136's3.317ms GPU gap is removable.

VERDICT -> Stop treating the instrumented3.221ms as baseline production host
submission cost. Keep S136 shapes/names/order with explicit timing-perturbation
limits, use S142 for the current CPU brackets, and prioritize real QSA/kernel
GPU experiments. No synchronization removal, runtime override, main queue
change, serving promotion or repeat campaign follows this result. The legacy
closure field trace_JSON_validated=false describes no new rawtrace; the generic
stop record's functional_status=NOT_COMPLETED_FAILED_OR_INVALID reflects that
this diagnostic deliberately did not run a functional campaign.

Evidence: [exact controllers, failure, matched pair, parsed ring and health](../strata/flash-next/lowlevel_probe/unprofiled_cpu_ring_s142/README.md),
[independent source-bracket interpretation](../strata/flash-next/lowlevel_probe/unprofiled_cpu_ring_s142/independent/findings.md),
[S136 instrumented profile](20261016_strata_current_mtp_profile.md).
