# S50: captured HC graph discriminator (2026-10-15)

CONFIG -> Restricting the duplicate global lo store to group0 reduces this
synthetic captured whole-read graph's median time by6.290%. All eight paired
rounds improve and pass the screen declared before execution. This warrants
a small serving candidate; it does not qualify end-to-end model speed.
The qualified152055a6/headpairON/depth2 configuration remains recoverable.

S48's direct-submit timing was inconclusive because its timing spread was
larger than the median difference. S50 changes the measurement mechanism:
each variant captures32 consecutive whole HC reads and submits that graph
once per timed batch. Kernel source, IR archive and replaced member are
unchanged. Final AOT linking regenerates device images, so this is a new GPU
binary and receives bounded direct/replay numerical anchors.

T6/apply1/inject1 uses seed7 BF16,2560 dimensions,hc4,rank320, default C16,
GRF0/ILP0, sliced DOWN and norm split, one in-order profiling queue, physical
card1 PCI44:00.0. Apply updates the residual on each of the32 reads, equally
in both modes. Both modes use the same allocations and private scratch.
Single-read and32-read graphs coexist; queue drains precede graph destruction
and all graphs die before their USM buffers. The host selector stays fixed
at0 during replay, after both branches have been captured.

COMMAND -> Actual claude-opus-5-5 and claude-fable-5-1 reviewed the exact v2
adapter before execution (140.55/151.41s). One33s CPU-only offline adapter
build reused S48's private archive; no kernel TU recompile, DRI or network.
Root verified compiler/link flags, archive/member and link-map identities.
One45s quiet card1 window ran strict checks then eight balanced paired32-read
batch timings, orders AB/BA/BA/AB/AB/BA/BA/AB, after four batch warmups/mode.
There was no external profiler, model, server, card0 work, collective or reset.
Pre-health, oracle, post-health, driver and tee all exit0; container removed.

Raw P:
`/mnt/vm_8tb/b70/results/flashnext_ista_gsq_rco_iq3xxs_strata_cachefix_20261014/hc-lo-once-graph-s50-prep/v2/`.
Actual recipes: `bash P/build.sh`, `bash P/ROOT_RUN.sh`; leased run uses
cpu-run --quiet outside gpu-run --card1 and the matching device mask.
Analysis: `python3 P/analyze.py P/run-clean S48_ROOT > P/root-analysis.json`.
Use retained raw dependencies; repository source copies are inspection copies.

| Artifact | SHA256 |
| --- | --- |
| Executed adapter | 2e45e683a4957bbc9595421f9e10aa3f59b95d8732a40b367730b53466f2163e |
| Executable | c92233f7f556a135d8ca4157d33d1e3302560de05f1c802991d2e018c9ecdd36 |
| Reused private kernel archive | 83e726418f6b91e00d7af1ce4983baf71db2ac461affa1206514609e48ebbe0e |
| Reused replaced member | 1ff40f83b7c9deeaf63bed829829f0c5bf8040330728174e17d59b68ced222c1 |
| Actual core archive input | 34e91c40f01b7aa0e652e195228dfc1fadba93e40006c12788c81426567b0286 |
| Executed analyzer | 94aab3d155475cc7ce5a59d349b8cfc1c9697830e9d98e52e0ab767ef1825c03 |

RESULT -> All26 full snapshots match exact S48 outputs: direct initial and
changed, own single-graph initial and changed,32-read initial, and every
paired timed final. This is19,442,176 compared bytes over10 distinct earlier
anchors, not26 independent old fixtures. Guards, finites, tails, changed-input
observability and immutable inputs/weights pass. Repeating all12 S48 cases
was unnecessary for the unchanged kernel IR and this selected T6 fixture.

Actual verbose DOT node inventories strengthen branch binding: each single
has4 kernel nodes/3 edges; each batch128 nodes/127 edges. The ordered motif
is norm -> down_sliced -> down_reduce -> UP, repeated1 or32 times. Control
contains only gr_up_multi_cols/scalar16; candidate only
 gr_up_multi_cols_lo_once_s48/scalar16. Node NAME fields are parsed once and
demangled; baseline substring counting cannot overcount the candidate.
This proves captured contents and ordering, not independent GPU dispatch.
Bit-identical numerical outputs alone cannot identify the mode.

| Round | Order | Control complete us/read | Candidate complete us/read | Paired change |
| --- | --- | ---: | ---: | ---: |
| 0 | AB | 58.963 | 55.613 | -5.681% |
| 1 | BA | 59.354 | 55.662 | -6.220% |
| 2 | BA | 59.321 | 55.620 | -6.239% |
| 3 | AB | 59.296 | 55.668 | -6.118% |
| 4 | AB | 59.380 | 55.614 | -6.343% |
| 5 | BA | 59.364 | 55.636 | -6.279% |
| 6 | BA | 59.426 | 55.631 | -6.387% |
| 7 | AB | 59.434 | 55.599 | -6.451% |

These are eight batch means per mode. Primary host-complete medians are
59.358797 ->55.625281us/read (-6.289743%), difference3.733516us, versus maximum
same-mode range0.470906us. All eight paired signs agree and change exceeds2%:
all three predeclared screen conditions PASS. This is a conservative screen,
not a confidence interval, per-read p95, stability or production promotion.

Control first4/last4 medians59.308/59.403; candidate55.641/55.622us/read.
All positions and chronological rounds are retained. Same-queue barrier-span
medians59.364->55.640us/read support the primary observation but are not
isolated device service. Single-submission enqueue costs amortize to
0.644/0.635us/read medians; dividing by32 does not measure an individual
HC host launch. Enqueue, completion and event spans overlap and cannot be
added as a latency budget. Optional graph-event timestamps are null because
this adapter does not enable the graph profiling property; no claim of an
unsupported runtime API follows from those nulls.

VERDICT -> Graph microbenchmark signal passes; serving and broader geometry
benefits remain unqualified. The S31 HC pool is a ranking observation, not
an additive or removable latency budget. Repeated synthetic weights,
fixture state and graph shape differ from a model pass. No isolated UP,
physical stores, bandwidth, L2 capacity or occupancy measurement is claimed.
The default-off PLE serving experiment stays separate to preserve attribution.
No full-context repeat, direct-submit rerun, broad knob sweep or daily-server
restore is needed before a bounded HC serving comparison.

Loop: one adapter-only build and one45s card window resolved the S48 timing
question, with existing snapshots reused for correctness. Actual post-result
review recommendations and raw manifest are recorded below after closure.

Post-result actual reviewers: claude-opus-5-5 (30.96s) and
claude-fable-5-1 (42.86s), both exit0, favor one isolated matched serve after
S49 is decided, and hold promotion. Opus stresses the regenerated production
binary and suggests ISA/numerical checks; Fable stresses model scheduling and
suggests later HC-pool profiling. These are recommendations, not automatic
new campaigns or tolerance changes. Root accepts the serving discriminator,
keeps exact parity/health/source binding, and does not infer physical cache
behavior, a forecast model delta, or a complete causal decomposition from
host fractions. Captured contents remain distinct from a dispatch trace.

Frozen raw manifest `hc-lo-once-graph-s50-frozen-manifest.json` covers
113files/27,015,685bytes, SHA256
`7cd3d4e4405ec75ed161ca2937e236bdf2263141a52d7415f1b712b6144d102d`.
