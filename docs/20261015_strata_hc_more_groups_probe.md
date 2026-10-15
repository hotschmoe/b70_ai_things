# Strata HC MORE_GROUPS probe, 2026-10-15

Keep `STRATA_GR_MORE_GROUPS=0`. S73 passed its full numerical, capture and
lifecycle gates, but MORE1 was slower for every tested token count. No serving
arm, retry or sweep is warranted. This is a synthetic HC primitive result,
not a full-model regression or a diagnosis of hardware behavior.

CONFIG -> One binary, four fresh processes in ABBA order (0,1,1,0), physical
card1/PCI0000:44:00.0. Current frozen production HC with common LO_ONCE; exact
S39 seed7 synthetic BF16 weights/inputs, N2560/HC4/LR320. All12 T1/2/6 x
apply0/1 x inject0/1 cases; changed-input own direct/graph checks. LARGE_GRF0,
sliced1, normsplit1, UP_COLS16, ILP0, V30 fixed; HCQ8/QFUSE absent. No model
payload or serving process. S72's exact S15 matrix did not test MORE; its
slower UP8-only arm lacked the480-group down/eight-load-reduce combination.
S39 also fixed MORE0. This was a distinct existing mechanism, not a GRF retry.

COMMAND -> Root ran the reviewed source adapter using existing frozen fat
objects and compiler-supported extra HC/support and adapter prelink wrappers,
then an ordinary host link. CPU-only build:30.5976s; HC/support AOT8.178s,
adapter compile9.428s, adapter images0.339s, final hostlink0.350s. No kernel TU
recompile or source/archive modification. The new HC AOT is not the old GPU
binary. Root then ran ROOT_RUN.sh with cpu-run --quiet outside gpu-run --card1,
mask1/level_zero:*, no unitrace, one pre/posthealth pair for the ABBA window.
Each sample resets canonical inputs outside timing, then submits ONE captured
20-read graph. Three warm and three timed samples per T1/2/6 apply1/inject1;
all12 single-read and all3 initial/changed batch gates pass before timing.

RESULT -> Binary SHA-256:
`bd904cc03792988313a5e7ccfea1f4f175c34119de9bbdd538cab0f7d6e2e47a`.
Actual source SHA-256:
`08d15004117114a2e04ee549b28ea96c12c15bec2377d24ec8de6ad6f3f41375`.
The ordinary link used1ld/0AOT; map selected only frozen fused_gr and
verify_kernels host members. All three wrappers retain constructor/destructor
registration hooks. Target normal/MORE HC classes occur in HC-support/final,
not common. General SYCL/DPCT helper names overlap; registration/name metadata
alone neither proves equivalence nor eliminates ODR risk. Runtime checks are
separate evidence. Exact identities and overlaps are retained in the audit.

Across four processes, all276 full-state files (206386176 bytes) and312 checked
snapshots passed. EACH process also matched36 retained S39 anchors. Files contain
all8 allocated rows of
R/lo/rs/inject/mixed/xn, with guards, finite values, active/inactive/untaken
write checks and full bo/inj/weight immutability (R is intentionally updated
when apply1). Own initial/changed single and
20-read graph outputs match direct controls; all four process outputs match
in full bytes. All60DOTs passed1152-node/1092-edge complete-chain and expected
normal/MORE kernel-class checks. DOTs show capture classes/order, not measured
NDRange execution or actual dispatch; geometry is source-bound separately.

Host-completed graph-batch time divided by20, microseconds per read:

| T | MORE0 median of two process medians | MORE1 median of two process medians | Change |
| --- | ---: | ---: | ---: |
| 1 | 27.480400 | 31.898075 | +16.075730% |
| 2 | 33.278675 | 39.635075 | +19.100520% |
| 6 | 56.179375 | 71.545650 | +27.352164% |

Each process median uses three nested samples. These are two fresh processes
per mode, not six independent trials, pooled-six timing estimates, per-call
p95 or significance claims. At each T, all six candidate observations exceed
all six controls. Raw rounds, process medians, enqueue/completed and optional
same-queue barrier spans remain available; do not add these clocks together
or call the barrier span isolated GPU service. Repeated synthetic weights and
20 sequential residual updates differ from streaming model activations. No
cache, occupancy, bandwidth, scheduling or kernel-specific cause is established.

The whole leased window closed in80s: all four oracle exits0, valid own CIDs
AND names confirmed absent, one card1 pre/posthealth PASS, wrapper/tee exits0.
Removal confirmation precedes next process/health; no reset or retry. Host
pre/post pins and final artifact comparison bind the same binary/source/input
records. Other containers were preserved. No full-context qualification was
needed for rejecting this primitive.

VERDICT -> KEEP_MORE0; no serve/retry/sweep. Source/math equivalence, compiled
image/registration metadata and actual numerical/timing/cleanup results have
distinct scopes. S31's7.199ms HC service pool is not a recoverable budget, and
this probe supplies no full-model latency forecast. Root recorded source-review
approval and read both complete actual post-result reviews. Opus5.5 (30.268s)
and Fable5.1 (35.469s) both accept KEEP_MORE0 and no serve/retry/sweep.
They reviewed the supplied packet, without raw-file tools. No universal ranking
of future mechanisms or hardware bottleneck follows from their agreement.

Exact record copies: [HC MORE_GROUPS S73](../strata/flash-next/lowlevel_probe/hc_more_groups_s73/README.md).
They preserve historical external paths and are not runnable in-place templates.

External evidence root:
`/mnt/vm_8tb/b70/results/flashnext_ista_gsq_rco_iq3xxs_strata_cachefix_20261014/`.

- `hc-more-groups-s73-prep/manifest.json` (source preparation only):
  `18959a4ece02a5380049ce8999d06e66cea4f09b1cd1070c3d9719fbaa6f68da`.
- `hc-more-groups-s73-prep/root-build-closure.json`:
  `31f3f9a71f545ba06b2cefac5e3e38abba82f0eb16cd4bcc67cb5cdcba61f673`.
- `hc-more-groups-s73-root/subagent-link-audit.json`:
  `d7cc29410982f3d4c09ccb2468f7ce93688f505a4dd8e76d42180dc064931b87`.
- `hc-more-groups-s73-prep/run-clean/comparison.json`:
  `17caa534cb0887150427c070ca8e644764fb3c6e4d200db103909c2fc4a3f8a1`.
- `hc-more-groups-s73-prep/run-clean/closure.json`:
  `6b3cc4d113cea8f73b3639fd227af2bf5ea7a5584e3c2cd894df1693a9392203`.
- `hc-more-groups-s73-root/root-runtime-decision.json`:
  `44cd576613c1709b59114b97972aba6ef9dadd6d76e54add208cd83c674979f4`.

Actual source reviews used fresh sessions: claude-opus-5-5
`8e75a3cf-a7e7-4c11-b17a-edd80c51aec8` (47.458s) and claude-fable-5-1
`0fa44131-0a41-43bb-bc94-9b9226d27f86` (141.215s). Post reviews resumed
those exact sessions. Root verified all36 actual anchor keys before runtime;
the generic comparator alone would not reject missing future anchor keys.

Frozen external manifests (relative to the evidence root above):

- Completed build/runtime: `hc-more-groups-s73-manifest.json`,434 files,
  278453985 bytes; SHA256
  `c87c62e4b9fe9f6d39bbae9ca8d9e116408e0d654f073822e3d03a4d40908310`.
- Source consultations: `hc-more-groups-source-reviews-s73-manifest.json`;
  `0d710aa9e2b7cf029f5a47bb4ca7f9094be6eef693b618768a3f0ef60521229c`.
- Post consultations: `hc-more-groups-postreviews-s73-manifest.json`;
  `55547052d4e4979690d3eff1d721359191937e88bfb34fa024b130fd09870129`.

Dev-loop decision: reject this mechanism at the primitive gate. The80-second
lease avoided a full server startup and redundant context/cache qualification.
Keep the reusable supported wrappers; prepare a distinct integer-matrix kernel
in parallel, with strict public-byte gates before conditional timing.
