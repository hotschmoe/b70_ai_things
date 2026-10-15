# Strata data readiness probes

These standalone probes complement the [data map](../data-map/README.md).
They measure transfer preparation and dependencies; they do not claim a
serving optimization. Actual Claude Fable 5.1 authored the Level Zero
capability query; Claude 5.5 authored the readiness probe, reviewed and
corrected before GPU execution.

Build offline, without a GPU:

    bash build.sh

Every GPU run uses the lease and matching device pin. From this directory:

    ZET_ENABLE_METRICS=1 ZE_AFFINITY_MASK=0 ONEAPI_DEVICE_SELECTOR='level_zero:*' ../../../bin/gpu-run --card 0 build/ze_metric_enum --json build/card0-metrics.json
    bash run_readiness.sh card 0 2 full
    bash run_readiness.sh card 1 2 full

The launcher pins CPU 2, checks the expected PCI BDF and binary hash, and
runs the AOT binary inside strata-ab-build:ocloc. Card selection is physical;
the masked device is logical 0. K_USED=10 is confirmed from served GGUF
qwen4exp.expert_used_count; rows=6, N=2560, hc=4. Exact payload sizes:
24-byte tokens, 17280-byte Q8_1 activations, 245760-byte residual,
17520-byte ids+activation and 614400-byte expert return. EP sizes are
single-card transfer stand-ins here, not P2P transport measurements.

Each timed arm includes fresh salted bytes, destination poison/guards and
exact verification. The finite producer has an observable reduction checksum
verified on the host; an earlier removable busy loop was caught in review.
Host allocation/touch/synthetic pack/enqueue/wait/consume times are separate.
Device start/end timestamps measure service and producer-copy-consumer gaps.
Host and device clocks are never subtracted. command_submit versus start
was inconsistent (including negative queued differences); preserve those raw
fields as diagnostic only and exclude ALL queued-time values from conclusions.
Synthetic pack/consume/checksum costs are not Strata's quant/pack costs.

S24: one full run per card, 680 samples each, all byte/guard/checksum checks
passed; both-card post-health passed. Sample 0 is excluded from summaries.
Fresh pageable versus preallocated pinned D2H at 614400 bytes had median
producer-end to copy-start gaps 121.20 versus 1.41 us (GPU0), 116.93 versus
1.41 us (GPU1), 11 warm samples per arm. This warrants serving-path readiness
instrumentation, not a claimed model speedup or a bandwidth limit.
Both host metric-group queries returned ERROR_DEPENDENCY_UNAVAILABLE despite
timestamp support; no driver, kernel, sysctl or package was changed.

Raw provenance, exact commands, frozen binaries, hashes and summaries:
/mnt/vm_8tb/b70/results/flashnext_ista_gsq_rco_iq3xxs_strata_cachefix_20261014/lowlevel-s24/

No server restoration, duplicated smoke matrix or new whole-model build was
needed. The GR no-root-sync experiment passed 32 bitwise cases but was neutral;
its rejected source patch is retained separately. No serving defaults changed.

The retained metric enumerator fixes a display-only review finding: timer
units depend on properties.stype, not the driver API version. The installed
ze_api.h documents DEVICE_PROPERTIES as ns, DEVICE_PROPERTIES_1_2 as Hz.
Raw original values are retained; no performance conclusion used that label.


## Real vocabulary head replay (S25)

    bash build_head.sh

This separate CPU-only AOT build links the recorded runtime archives and
prints source/archive/object/binary hashes. It writes under build/head-cols.
Run inside the matching build container with bin/gpu-run --card N,
ZE_AFFINITY_MASK=N and ONEAPI_DEVICE_SELECTOR=level_zero:*, and mount the
GGUF read-only. Arguments:

    head_cols_probe --iters 30 --expect-bdf 0000:0b:00.0 --weights FILE 11024256 --json OUT

FILE is the first served IQ3_XXS GGUF; output.weight has Q5_K type13,
shape2560x248320,437043200bytes, SHA256
93f1c7cb7a7cf8134ce9f8cf80c5416df968f7f1ddcc16f2159a9521080826a5.
The offset is specific to this model file. Re-derive it from GGUF metadata
before using another artifact. C=1/2/3/4/6/8 use deterministic normal
activations, every column compared bitwise with its independent C1 result,
plus argmax, finiteness, inactive rows and tail guards. Actual read weight
FNV-1a64 must be da8746c3dcc2469e for this tensor.

Both cards passed all six cases. Card0/card1 minimum-of-three mean spans
in us: C1 1006.0/1006.6; C2 1322.0/1307.9; C3 1721.7/1676.5;
C4 2043.2/1978.0; C6 2700.5/2620.7; C8 3376.6/3302.6. Quantization is
timed separately, about4.2-4.6us. These event-barrier spans include queue
and launch gaps. Tiny unitrace per-kernel medians on card0: C1 1016.666us,
C2 1309.010us, C6 2681.146us, grid62080x1x1, workgroup32x4x1, SIMD32.
This is a standalone replay; cache/frequency/context differ from serving.

Only two slots at the SAME draft depth may be paired. Four serial draft
heads cannot become one C4 call because successive depths are autoregressive.
Ideal head-only saving at depth2 is about1.38ms/window before extra graph
launches and coordination; no serving gain is claimed.

## Existing production PLE trace (S25)

STRATA_PLE_TRACE=1 on the frozen S21 binary changes timer/logging only;
keep STRATA_VERIFY_PROFILE unset (it changes shared-stream scheduling).

    python3 -I parse_ple_trace.py ENGINE_LOG --window START:END --json SUMMARY

The existing trace omits all-cache-hit tickets (jobs0) and tickets with
200+ I/O jobs. It reports same-clock differences without absolute anchors:
issue->submit, submit->last, last->return and collect wait. Large
last->return can mean useful prefetch slack, not a stall. The collect wait
includes caller locking/scheduling. It cannot be joined exactly to GPU
events and submit->last is not device-only NVMe latency.
S25 had21 miss-ticket records between batch capture and the24-window summary:
collect median667.9us, p90 817.4us, sum13.2942ms; jobs8-64.
Two107-input/64-output replies match S16 prefixes, active slots offer46
and accept40 drafts each; arithmetic canary and both-card/10 compiled
collective post-health pass. No clean throughput claim from this diagnostic.
Raw commands, frozen hashes and trace summaries: cachefix_20261014/head-profile-s25/.


## EP publication diagnostics (S26-S28)

    bash build_ep.sh
    bash run_ep.sh copy producer 0 20
    bash run_ep.sh flag producer 0 64

The launcher takes the pair lease, pins both visible devices, gates their
PCI BDFs, checks the binary hash and enables the same IPC capability as
the serving launcher. It never runs outside the lease. The probe links
its recorded kernel archive; native means that archive's EP implementation.

Messages contain17520-byte ids/activation pushes and614400-byte
upper-bound returns, with fresh nonzero salts, poison and4096-byte guards
every sample. IPC mappings remain alive throughout each process.
The spin and host status-poll bounds are finite. Host-sync1 waits on each
event only after its bounded completion poll. Device times use START/END
on the same card; inter-card clock alignment is unavailable.

Copy mode passed21 messages including first use. Flag mode failed fresh
data checks: owner poison sample10 (1068 bad push words); producer poison
rev2 sample3 (1158). Rev3 separates stage and inbox poison: poll-only failed
sample8 (1296 inbox-poison words), completed-event waits still failed
first-use sample0 (720 inbox-poison words). Guards, return and spin codes
passed; producer source and all later inbox rereads were correct. Post-run
matching counters do not prove what the wait kernel observed at entry.

A two-line probe-local change to use uncached payload stores also failed
sample9 (440 inbox-poison words), so it is retained as rejected evidence.
No timing or speed conclusion is drawn from these failed arms. This has
not demonstrated serving-output corruption. Freshness diagnostics now
take priority over evaluating the head-pair prototype. See the stage report
and raw head-pair-s26/ep-readiness/ for exact preserved binaries/variants.
The next tests inspect consecutive consumer reads, the wait's observed
epoch, and actual BMG fence/store ISA.

Prototype source: patches/experimental-batch-mtp-pair-head-s26.patch.
It is CPU-built and unqualified on GPUs. Slot-order binding, allocation
context and error propagation were corrected in review; its syntax and
CPU argument-binding tests pass. The serving control remains frozen S21.

S27 extends the same probe with `--extra-reads 2`: after the first receiver
copy, read the same inbox twice into distinct poisoned private buffers on
the same in-order queue, with no intervening host wait. `--wait debug`
records wanted/first/last epochs and spins from the actual device wait.
These are diagnostic arms; neither extra reads nor debug wait is a fix.
The S27 source26e26f96 and binary49cfed4f are preserved in raw
ep-ordering-s27/fable-dynamic/. Tracked source now matches S28 sourcea73dc02e.

`run_ep.sh flag producer 0 64 native 2` selects native wait plus extra
reads; change argument5 to `debug` for the observed-epoch arm. The wrapper
retains maxfail1; measured raw dispatch used maxfail2. Wrapper bash syntax
is checked; measurements used the frozen raw dispatch launcher, not a
redundant rerun of this adapted wrapper.

S27 native first/second/third read bad-word counts956/0/0 and1428/0/0;
debug1404/0/0 and1406/0/0. Only current inbox poison was consumed. Debug
wanted3/4 and observed flags2->3/3->4 (198/203 spins) match intended epochs.
This excludes a wrong wanted epoch for THOSE debug failures; native
internal values remain unobserved. It does not settle incoming payload
completion versus receiver visibility. Later views and return/guards are
clean. All queues drained, no crash/timeout/reset, both-card and ten
compiled two-rank collectives pass after EACH arm. All failed-arm latency
claims remain excluded; no production output-corruption claim.

S28 corrects S27's no-native-ISA statement: ocloc emitted hidden
`.text.*.asm` files, found by actual Opus5.5 and reviewed by root. The
default kernel has a group-scope UGM fence before flag publication;
explicit system scope on the workgroup barrier changes that to decoded
`fence.ugm.invalidate.gpu`. Compiler causality/legality remains unproven.

The source now includes native, split payload/flag, default mirror and
explicit-scope mirror producer paths, prepared by actual Fable5.1:

    bash run_ep.sh flag producer 0 64 debug 2 mirror
    bash run_ep.sh flag producer 0 64 debug 2 scoped
    bash run_ep.sh flag producer 0 64 native 0 split

S28 same-binary mirror fails112/648 first-read poison words; scoped65/65
passes. Split debug/native waits each65/65 pass. Actual patched EP TU with
the original S27 probe object and native wait/no extra reads passes65/65.
Frozen full engine152055a6 passes both concurrent64-token S16 prefixes,
draft accounting, math/reasoning/tools, normal teardown and final health.
No speed, long-run stability or full-cache requalification is implied.

Patch only the ep_push barrier, retaining the existing release fence:
`../patches/ep-push-system-barrier.patch`. Do not infer the old native
failure is absent by linking a new fixed archive. Use retained frozen
binaries for old controls. Tracked source is verbatim measured S28;
adapted wrapper is syntax checked only, not redundantly rerun. Measured
raw dispatch stopped after two failures; wrapper remains maxfail1.
See [S28 evidence](../../../docs/20261015_strata_ep_publication_scope.md)
and cachefix_20261014/ep-ordering-s28/s28-report.json.

S29 finally serves the default-OFF head-pair prototype on frozen S28
152055a6, toggling only the environment option. Three fresh C2 coding pairs
measure97.881OFF ->102.279ON aggregate tok/s,6/6 matched output hashes.
Functionals, both normal teardowns and both health closures PASS. Native
emit7.033->5.380ms/window is measured; new paired C2 per-kernel service is
not. S25's microprobe and S22's old four-C1 trace remain separate scopes.
No repeated head microprobe or new engine build was needed. Full-cache
depth2/headpair qualification remains pending, as does a small fix for
draft counters lost when an API request returns from batch to solo.

S35 adds `peer_gather_oracle.cpp`, copied byte-for-byte from the compiled
and tested source. It compares production per-expert activation gathering
against one gather into FP16-aliased output storage followed by reverse
expert processing. Two card-1 cases cover ten real quant format pairs,
363 synthetic-routed experts and 297,369,600 bit-identical FP32 outputs.
Gather bytes, finiteness, nine buffer guards, unused output tail and all
per-card pre/post health checks pass. This is ordinary device USM, not
production IPC or serving qualification; cold/warm order prevents a speed
claim. See the [S35 report](../../../docs/20261015_strata_peer_gather_alias.md)
for identities, original build/run recipes and raw closure locations.


S39: `grf_oracle.cpp` calls the actual fused_gr_read_multi from current
libstrata_kernels.a + libstrata_core.a, no replacement math/setter. Frozen
7365c788 compares fresh GRF0/1 processes with fixed slicedDOWN/cols16UP,
T1/2/6 apply/inject,48 full snapshots of R/lo/rs/inject/mixed/xn and
T2/6 changed-input graph replay. All35,893,248 compared bytes/guards/finites/
tails PASS. Actual trace names/property table show128vs256 registers,
private/spill0 both; GRF256 slowerall12 profiler-sensitive cases. Keepdefault,
no serving test. Build29s; card1quiet windows39/47s include pre/posthealth;
normalteardown PASS. Current production archive/member/flags and API ownership
checks are recorded in raw grf-oracle-s39-prep/. Reuse that offline build
recipe against its pinned archives; do not use the old no-op fast setter or
S24 root-sync binary as a cross-GRF gate. Three raw timing rounds/case, not
best-only. No clean timing, dynamic occupancy or production regression claim.
See [S39 report](../../../docs/20261015_strata_grf_resource_probe.md).


S41: `q6_dense_oracle.cpp` is the exact compiled0a768982 source, not a new
kernel. Frozena09b4c32 links productionnativeMMVQ/Q8quantizer/Q6pack-unpack;
one103s offlineCPUbuild/36squietcard1window including health. Firstactual
6144->2560 C6 pairfailsstrictbits, guards/finites/tails/readonlyinputsPASS;
no remainingpackedcolumns/shapes or graph replay. Bothhealth0/normalnumeric
exit5/containerremoved. Savedactualinputs/outputs permit boundedCPUFP64/fsum
reference withoutGPUrerun (19rows/114outputs/0.141s). OldstrictgateFAILED;
no arbitrarytolerance or speedclaim. Current production source untouched.
See [S41 report](../../../docs/20261015_strata_q6_dense_packed_probe.md)
and raw q6-dense-packed-s41-prep/{build-closure,s41-report}.json, build-v2.sh,
run-prep/ROOT_RUN.sh, run-profiled/ and cpu-reference/. Exactbuild requires
its pinned production archives; do not substitute qualified152055a6's
unavailable old archive or assume current unfrozen engine is promoted.


S42 `grouped_peer_oracle.cpp` is exact compiledf69603f2/frozen64586e43.
One139s build/44s card1 window covers16 real experts/1748 rows. GU fails
strict equality before candidate SiLU/down/timing; prior memory/readonly
checks and both health checks pass. Later source paths are unqualified.
Exact recipes: raw peer-grouped-gemm-s42-prep/v2/build-oracle.sh and
root-level ROOT_RUN.sh -> leased_run.sh; closure in run-clean/.

S43 `q6_cpu_reference/{check_first_c6.py,checker-pins.json}` is copied
exactly from executed7831f155/e3ad41b2. CPU-only NumPy; run under bin/cpu-run
with a NEW output directory. Pins require S41's retained evidence and
source identity. Full first C6 diagnosis passes analytical L15/18 bounds
in1.9545s, with rational and wrong-row checks. This does not replace S41's
strict failure or qualify other cases, model quality or production.
[S42 report](../../../docs/20261015_strata_grouped_gemm_probe.md),
[S43 report](../../../docs/20261015_strata_q6_cpu_reference.md).


S44 `q6_dense_collect.cpp` is exact compiled40ba97c1/frozen93f8bfc9.
One101s offlinebuild/42s card1window saves96snapshots overfourrealQ6shapes,
six Cs and initial/changed ownC6 direct/graphs; strict memory/ownreplay and
health/teardown PASS. `q6_collection_reference/` copies the exact executed
CPUchecker/helper/pins and timing analyzer. Run from the retained raw bundle
with its required earlier sources/dependencies, not this inspection copy.
Full72case/2,623,488output arithmetic diagnosisPASS11.43s; S41bitgate stays
FAILED. WarmC6 three16-call batchmeans reject integration asconfigured,
no serving orall52/quality claim. Repeated-weight cache regime isunmeasured.
[S44 report](../../../docs/20261015_strata_q6_collection_and_warm_probe.md)
contains commands, hashes, configuration and limits.


S47 `ple_prefetch_cpu.cpp` is exact executed8415719e, real PleTable/host
reader/DirectFile with a finite synthetic IQ4_NL GGUF and Mmap oracle.
Both44case arms pass bytes/guards/reader-request/logical-byte counts under
cache0/256 and io_thread0/1. It includes the pinned external production
`ple_reader_test.cpp` helper; use the retained raw bundle/container recipe.
The key-match patch is source-only; batch prefetch wiring is separately
unapplied and no serving benefit is measured. No broad inherited selftest
is run. [S47 report](../../../docs/20261015_strata_ple_prefetch_readiness.md).


S48 `lo_once_oracle.cpp` is exact executed67377dd9/binary7e3f1345. Private
`hc_lo_once_probe_s48.patch` selects existingLO_ONCE atsamecols16, preserving
production sources/archives. One44s offlinebuild/45s card1window,36 full
strictpairs/S39hashanchors PASS; warmT6 failsdeclaredspreadscreen. Analyzer
`analyze_hc_lo_once_s48.py` is exact executedsource; runitfromtheretainedraw
bundlewithS39data, not this inspectioncopy. No serving/integration claim.
[S48 report](../../../docs/20261015_strata_hc_lo_once_probe.md).


S50 `lo_once_graph_oracle_s50.cpp` is exact executed2e45e683/newbinaryc92233f7.
SameS48 IR archive/member, finalAOTimages regenerated. All26fullS48anchors
PASS; actualDOTsingle4/batch128kernel nodes identify capturedC16UPbranches.
Eightbalancedpaired32-read single-submission graph batchmeans improve
59.359->55.625us/read(-6.290percent), declaredscreenPASS. No serving/physical
traffic/per-readp95 claim. Exactanalyzer`analyze_hc_lo_once_graph_s50.py`
requires retainedS48/rawdependencies; use retained bundle, inspectioncopies
are not standalone. [S50 report](../../../docs/20261015_strata_hc_graph_probe.md).

S52 existing Q5_K head RPW2 passes strict C2/C6 primitive screens, with
16.197/31.454 percent lower quantize+head graph medians. These are seeded
activation diagnostics, not serving gains. Exact inspection copies:
head_rpw_leaf_s52.cpp and analyze_head_rpw_s52.py;
[evidence and limits](../../../docs/20261015_strata_head_rpw_probe.md).


## HC MORE_GROUPS graph primitive (S73)

`hc_more_groups_s73/` holds exact source/comparator/build/controller/host-closure
and anchor record copies. Historical absolute paths are retained; do not run
them in place or overwrite frozen external outputs. Current HC/shared objects,
supported additional HC+verify prelink wrapper and ordinary hostlink bind
newbinarybd904cc0; no kernel TU rebuild. All12 synthetic T1/2/6 apply/inject
cases/ownchangedgraphs,276 full-state files/60DOTs/312checks and36S39anchors
perprocess PASS. Four fresh ABBA processes, one pre/postcard1health pair and
all ownCID+name removals/wrappers PASS. MORE1 graph-batch completed averages
are slower16.08/19.10/27.35percent T1/2/6 (two process medians/mode, three
nested samples each). Actual Opus/Fable accept keepMORE0/no serve/retry/sweep.
No wholemodel regression, isolated-device service, helper-ODR or cause proof.
[S73 report](../../../docs/20261015_strata_hc_more_groups_probe.md).

## Signed INT8 matrix static capability (S74)

`int8_xmx_capability_s74/` retains exact source/offline recipes and decode
records. Sourcefcb7e5f8 compiles to82,648B executable37728fb9; selected896B
main ISA contains one signed-byte DPAS-to-INT32 atSG16/SIMD16. Offline build
and disassembly do not execute the binary; no runtime128-output, coordinate,
IQdecode/scale/nativeFP32-order or latency gate is claimed. Do not run the
optional --run path without a separately approved leased gate.
[S74 report](../../../docs/20261015_strata_int8_xmx_capability.md).
