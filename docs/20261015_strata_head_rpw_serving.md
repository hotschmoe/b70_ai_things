# S55: head-only Q5_K RPW2 serving (2026-10-15)

CONFIG -> Matched serving and correctness gates PASS; keep the head flag default
OFF. Three finite pair walls favor ON, but median31.600ms difference is smaller
than control32.818ms range. This is a small favorable diagnostic observation,
not a resolved serving gain, promotion or stability result. No standalone retry.

The preceding [S52 primitive screen](20261015_strata_head_rpw_probe.md) passed
strict full bytes and graph checks with real437MB head weights and two seeded
finite activation cases. Its C2/C6 graph improvements16.197%/31.454% motivated
this bounded serving discriminator. S52 did not capture production activations;
its primitive percentages were not a whole-model forecast.

## Exact candidate and controls

[The exact experimental patch](../strata/flash-next/patches/experimental-head-q5k-rpw2-s55.patch)
has SHA256 `ba6a43c558d5e07e96e5e690ec9b9580ccea03d1a265fdf541a4f887bfe6a596`.
It adds an explicit SYCL full-head entry at verifier, non-subset MTP and paired
MTP callers. STRATA_HEAD_Q5K_RPW2=1 selects existing RPW2 only for Q5_K/type13,
N2560/M248320/C2 orC6 with existing WIDE_K enabled. Absent/0/other strings are
OFF. Other shapes/types and subset heads use original native_mmvq; CUDA and main
dense/PLE callers are unchanged. Global STRATA_MMVQ_WIDE_K_RPW semantics remain
unchanged. RPW2 shares activations across two rows and retains each row's original
arithmetic/reduction expression; source reuse is not a physical traffic or
compiler/ISA identity claim.

Two fresh servers use the SAME new executable, flag0 then1. HC LO_ONCE from
[S51](20261015_strata_hc_serving.md) is common, S49 PLE prefetchOFF, globalRPW1
and WIDE_K1, exact new NVMe native/sibling paths, TRACE1 and buffered CPU host
markers1. Pair-headON/depth2, spec4, fixed12288 prefill, configured200k context,
32GiB/four-entry RAM cache,149/363 expert split and all24576 experts resident.
Stable hotschmoe-dd is first, research alias/registry and image checked. Launcher
binds127.0.0.1:18085. This stage does not exercise full200k RAM recovery again.

| Artifact | SHA256 |
| --- | --- |
| SAME executable, both arms | bb0669002bff48f28340cd413b576fe905ffbc327a332bebd63ff12e3066272f |
| Native kernels archive | f2b53eb482814b2c9a046e524a1f76dee717fddc6fc45073a492880bb153e9a8 |
| Engine archive | 77a18ca2e4fd4cd35414a74e8386b760a904476022b24ecc7203339fd00180b7 |
| Unchanged core archive | 34e91c40f01b7aa0e652e195228dfc1fadba93e40006c12788c81426567b0286 |
| sycl/src/kernels/cuda/native_mmvq.dp.cpp | 81558e95f721413a75c3d3ec6ed7cb5eee766e4f662080cf500c7985090f03e2 |
| sycl/src/core/verify.cpp | e97ce93a3434fa1d4e9a3a3dce99a3fd41d86aac8662fbd3c959a3c78ce0d4b2 |
| sycl/src/core/mtp.cpp | fcd5d2e4b029985a6e7768707a5aa489e4b5724a78db50fc2313d9c64e67ec31 |
| include/strata/kernels/native_head_mmvq.hpp | 397ebadf1fbb9b479ec928a22b1bfd1d37f93f6097b28deee14cf1d7aa3fdb5f |

The source rows above are after patch application. Common HC source is
`d62bcb252eecdbcaae81095907c2241e39c4798b7854bb499fb5643fecb24e99`.
Root closure pins all source/member/library/CMake identities, including unchanged
Python, generate, cache and remaining libraries. Pinned image is
`sha256:e41d59c0940e1bd2f6f64c9d5d2d7452b77812b0a8407fac8d67cacec5d7fdd2`.
Final AOT relink produces new device images; no previously qualified GPU-binary
identity is claimed. Dispatch binding here is source/API/build plus actual C2
paired/C6 native work, not an independent serving GPU dispatch/ISA trace.

## Commands and loop cost

COMMAND -> Actual source reviews by claude-opus-5-5 and claude-fable-5-1 precede
root application. Root uses bin/cpu-run around UID1000 CPU-only Docker, no DRI or
network, jobs4/linkjobs4 and disk TMP/cache; oneAPI setvars precedes strict shell.
Intended incremental delta is native_mmvq/verify/mtp, two archive updates and AOT.

The first build fails after71s before GPU: root-owned verify.cpp.o prevents
clang-offload-bundler overwrite. Native and MTP compiles succeeded and are
preserved. Root repairs ONLY that object's UID/GID, with byte identity unchanged;
no source, numerical, flags or configuration fix. A new preserved retry plan
compiles only verify, updates both archives and links,177s successful. Do not
rerun successful TUs or overwrite attempt1. Future builds should preflight write
access to every concrete planned output, not only its parent directory.

Historical recipes under raw P use build-root.sh / build-retry1-root.sh, then
harness-prep/make_configs.py --exe NEW_EXE --sha256 AUDITED_SHA. New mode0/1
configs bind the SAME exe/hash, differing only head flag and validated log path.
Comparator normalizes precisely those two fields; it retains executable equality
and all other config/work gates. No prefetch-treatment or binary normalization.

For each arm, root starts under bin/cpu-run until loaded, then runs client warm
and timed under bin/cpu-run --quiet; functionals follow. Systemd's detached
service inherits no CPU fd7 and retains the pair GPU lease. Timed clients never
nest a GPU lease. stop_and_health.py runs CPUquiet OUTSIDE its health GPU calls.
Normal TERM Python -> native QUIT, owned container removal and server lease0
precede per-card plus compiledTP10 post-health. No outer kill/retry/reset.
Readiness observation is not permission to restart a healthy process.

No redundant prehealth: root verifies the exact latest S51 candidate post-health
reference/manifest, no intervening GPU use or stack change; offline no-DRI build
and metadata repair are not GPU touches. Both NEW per-arm post-healths remain
mandatory and pass. Startup observations52.167/44.144s are fixed-order readiness
with2s polling/cache differences, not causal head/HC/NVMe speed evidence.

## Matched finite serving results

RESULT -> One untimed128x2 warm pair plus three fresh128x2 pairs per arm; native
flushed prompt-start orders logical request0 before request1. Raw clocks preserve
actual overlap, not a fixed physical slot assumption. Warm128-token prefixes match retainedS16;
all four pair groups match request/fulltext/SSE content/usage, native draft counts,
pair/solo counts, window/capture/copy work and complete marker frames. No sorting
or work normalization. Every coding request has107 prompt tokens, cache0,128 completions.
Native paired/solo/window counts in BOTH arms: warm45/2/48; r1 46/2/49;
r2 45/2/48; r3 46/1/48. Functional concurrent323/667, reasoning391 and real tool
roundtrip pass. All final diagnostics and normal-QUIT/health closures pass.

| Timed pair | Flag0 wall s | Flag1 wall s | Change |
| --- | ---: | ---: | ---: |
| repeat1 | 4.791661103 | 4.756362743 | -0.736662% |
| repeat2 | 4.758842952 | 4.754691226 | -0.087242% |
| repeat3 | 4.787963032 | 4.766246908 | -0.453557% |
| Median | 4.787963032 | 4.756362743 | -0.659994% |
| Observed range | 0.032818151 | 0.011555682 | |

Median difference31.600289ms is below control range32.818151ms. All three signs
favor ON, but one process/arm, fixed OFFthenON order and nested repetitions leave
this unresolved. There was no declared2% serving gate. Repeat1's35.298360ms
difference DOES exceed the control range; the reviewer statement that EACH pair
delta was below that range is false. The primary median conclusion is unchanged.
Aggregate256-token output divided by pair-wall median is53.4674/53.8226tok/s;
these are descriptive pair aggregates, not per-stream target attainment.

Per-logical-stream median [observed min,max] across the three timed requests:

| Flag/stream | TTFT s | Completion wall s | Client decode tok/s |
| --- | ---: | ---: | ---: |
| 0/0 | 0.902 [0.902,0.913] | 4.747 [4.716,4.775] | 33.026 [32.889,33.292] |
| 0/1 | 2.066 [2.045,2.073] | 4.571 [4.563,4.593] | 50.694 [49.849,51.010] |
| 1/0 | 0.903 [0.896,0.909] | 4.712 [4.712,4.753] | 33.275 [33.039,33.340] |
| 1/1 | 2.056 [2.030,2.062] | 4.554 [4.549,4.558] | 50.934 [50.234,50.955] |

SSE content-event gaps are retained per request (109 intervals each), not token
ITL. Across both logical streams/repeats, p95 gaps range45.787..46.277ms OFF and
45.352..45.635ms ON; maximum gaps include admission/prefill overlap. Client decode
intervals likewise include the partner's prefill/admission. No token-level timing
alignment or clean isolated head timing follows from these values.

Hot replay whole-window CPU medians r1/r2/r3 are45.780/45.702/45.477ms OFF and
45.267/45.269/45.088ms ON. CPU queue-wait spans overlap GPU/software work; they
are not the head's device service, hardware work reduction, occupancy, starvation
or physical traffic evidence. Do not add CPU spans to GPU/service budgets or use
them to explain why the primitive gain did not translate into a clear wall signal.

Both normal QUIT/container removal/lease exits are0. Card/compiled-collective
post-health elapsed seconds: OFF26.764/27.607, ON27.568/26.859, all0; native
compiled collective world2/4x5120/10iterations/P2P0 PASS. No GPU crash/reset.

## Decision, evidence limits and archive

VERDICT -> Keep STRATA_HEAD_Q5K_RPW2 defaultOFF, preserve opt-in source and bounded
observation. HC remains common retained candidate, S49OFF. No default/daily/shelf
promotion, standalone short retry, arm-reversal campaign, profiler or full-context
rerun. Retained qualified152055a6 and daily settings are unchanged. Keep headOFF
in BOTH arms of later unrelated experiments; enabling it as a free rider would
confound that mechanism. No new test campaign follows automatically.

Actual resumed post-result IDs claude-opus-5-5 (13.247s) and claude-fable-5-1
(12.528s), both exit0; root read both full texts. Both accept matching and default
OFF/no promotion. Root rejects Fable's each-delta/spread error above, causal
head-fraction/less-hardware-work explanations from CPU waits, and any automatic
ON free rider or retry. Model advice does not replace the measured evidence.

Actual production-activation full logits were NOT captured. Strict S52 results
remain real weights with seeded inputs; final text/SSE/usage/native-work matching
is not proof of full logits, arbitrary activations, KV-state identity, general
model quality, stability or Flash-Next target attainment. Current serving has no
independent dispatch/ISA trace. Historical170 map operations and rollups are
unchanged; S55 is a separate bounded observation.

Raw R is `/mnt/vm_8tb/b70/results/flashnext_ista_gsq_rco_iq3xxs_strata_cachefix_20261014`.
P is R/decode-head-serving-s55; source prep is R/decode-head-serving-s55-prep;
reviews are R/decode-head-{reviews,postreviews}-s55-20261015. Primary records:
P/{s55-root-report,comparison,root-build-closure,root-prewindow-decision,
root-postreview-decision}.json; mode0/1 coding/raw events, complete CPU analyses,
functionals, normal-stop and post-health. Original failed build and narrow retry
remain distinct. Reproduction must bind frozen identities in NEW output paths,
not launch historical live-path recipes unchanged against later libraries.

Frozen external `head-serving-s55-manifest.json`:216files/13729315bytes,
SHA256 `42d67d6dd37ab4cf391f845b6eb05fe0c1b5180deddb7d65d52d9e06cd3aafbb`.
No raw outputs were changed by this curated report.
