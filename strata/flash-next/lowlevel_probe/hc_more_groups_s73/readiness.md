# S73 source preparation: current HC MORE0/1

CONFIG -> N2560/HC4/LR320, exact S39v2 seed7/RNG/distribution/BF16 fixture.
Current frozen f2b kernel archive; HC fat member27d0ab6a (current source d62b),
verify-kernel support584e694f; common8f5a + core34e unchanged. No model payload.
MORE0/1 is the only treatment, four fresh processes a1/b1/b2/a2; LARGE_GRF0,
sliced1,normsplit1,UP_COLS16,ILP0,V30 fixed; HCQ8/QFUSE absent. Static selectors
preclude same-process toggling. Current HC LO_ONCE stays common; no kernel edit.

COMMAND -> ROOT ONLY after exact source/model reviews: bash build.sh; review
ordinary host link/map/hash closure, then ROOT_RUN.sh after announced card1
quiet window. Nothing compiled/run/tested by preparer. CPU build uses cpu-run,
pinned e41 image, no DRI/network, UID1000, disk TMP, <=4link jobs. Extract ONLY
existing HC and verify_kernels fat members into own build, check exact SHA,
compiler-supported -fsycl-link for additional HC-support wrapper plus adapter
wrapper; ordinary final hostlink without -fsycl. Common8f5a excludes HC AND
verify_kernels. S39 actual link.map3-6 shows HC's gpu_stamp host dependency
pulls verify_kernels; no stub/fake registration. Stamp buffer remains nullptr.
No kernel TU compile or production archive edit. New AOT images from unchanged
IR, NOT same old GPU binaries. Final map/registration/CPU build audit is pending;
setup/link failure remains a real failure, no unsupported fallback.

RESULT -> PREPARATION ONLY. hc_more_oracle.cpp is a176-line host adapter of
S39; source diff included. All12 T1/2/6 x apply0/1 x inject0/1 initial/changed
own direct/single-graph full-byte checks finish before any timing. All3
T1/2/6 A1I1 captured20-read graphs also compare against actual same-mode20-direct
initial AND changed output before timing. The3 graphs remain alive on one
in-order queue/shared immutable weights and stable allocations; kernel arguments
are captured by value. Reset and all reads/checks drain the queue. No graph
outlives its buffers; all batch graphs are explicitly destroyed after drain.

Before every sample reset canonical initial R/bo/inj and output poison outside
timing. Each batch applies residual20times by design, compared against OWN
20-direct final reference, not a single-call output. Three warm replays and
three raw timed samples for T1/2/6 A1I1 only; one graph submission/sample holds
20 complete four-stage HC reads. Host enqueue/completed wall separate; optional
same-queue barrier span is a host-submission-inclusive graph interval, not
isolated device service/per-call p95. Source geometry from frozen launchers;
DOT records captured class/order, not runtime dispatch or NDRange measurement.

Each snapshot checks all12 buffer guards, finites, active R apply0/1 semantics,
inactive R, every public scratch tail/untaken inject poison, full bo/inj and
full norm/down/up/inject weight bytes. Hidden production scratch is not exposed
as separately guarded. Snapshot layout EXACT S39: R,lo,rs,inject,mixed,xn,
all8allocatedrows,747776bytes. Changed mixed must be observable. 69bin/process:
48single-call anchors +12batch20 anchors +9timed final states;9warm snapshots
are checked but not saved. 78snapshot records,15DOT/process (12single+3batch20).
36unchanged phase hashes must match frozen S39 anchors; do not compare old
32-call timed-final with new20-call state. All four process artifacts must
full-byte match before timings are accepted descriptively. Stopfirst mismatch,
no tolerance/math-policy relaxation. Synthetic HC, not production activation.

compare.py requires all4complete terminal records, all cases and exact20call/
9timing counts, guard/readonly reports, all69full-byte SHAs,36oldSHA anchors,
and15actual DOTs/process. DOT must be one complete directed four-stage chain
(single4nodes or batch80nodes), with correct normal vs MORE BIGfalse classes
and cols16 baseline. STD_Layout addresses are NOT decoded USM pointers; no
false crossprocess pointer gate. Geometry is source/class binding: normal4T,
down80->480, reduceceil(324T/128), UP160->320. No unitrace for this stage.

Four-process numerical/capture gate is external AFTER collection, before any
performance acceptance. Per-process three rounds are nested within two fresh
processes/mode; no six-independent-trials/significance/serving claim. Preserve
all round order/trends and raw host/optional barrier values. Root decides
whether any whole-read signal merits serving; no target milliseconds forecast.

Lifecycle: ROOT_RUN takes cpu-run --quiet OUTSIDE gpu-run --card1 and mask1/
selectorlevel_zero:*. Adapter checks one visible GPU and PCI0000:44:00.0.
One card1 prehealth and one posthealth for the whole normal ABBA window. No
TP2/collective or extra servers. Unique names + createdCID per process; success-
checked fullID/name enumeration must prove both absent before next process or
health. Unconfirmed removal retains BOTH leases; cleanup touches createdCID
only, never another container. No retry/reset. Each300s leaf cap is a workload
watchdog, not an exec-yield deadline; no outerkill. Numerical5 still posthealth
only AFTER confirmed removal, then failure propagates. First failure stops
later leaves. Normal four-process cleanup and posthealth are required.

Root must create root-build-closure.json with actual binary_sha256,
adapter_source_sha256,frozen_kernel_archive_sha256,common_wrapper_sha256,
HC_member_sha256,verify_support_member_sha256 after checking build hash/map/
ordinary zero-AOT hostlink; host_closure.py binds that exact file pre/post.
Current root upstream refresh is separately owned at hc-more-groups-s73-root/.
No app/source/server/default change. Expected graph work648HCreads/process;
normal runtime budget requires root estimate after build, roughly a short
single-card window plus two health calls, not a clean serving benchmark.

VERDICT -> READY FOR EXACT SOURCE REVIEWS/ROOT AUDIT ONLY. No build/GPU/model
review launched. First actual gate is strict all-case equality/guards and
actual DOT binding under current code; no S15 UP8 or S39 GRF retry claim.
