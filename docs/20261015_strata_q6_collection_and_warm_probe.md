# S44: Q6 arithmetic coverage and warmed C6 decision (2026-10-15)

CONFIG -> Do not integrate the existing packed Q6 path. It passes the
bounded independent arithmetic diagnosis on four real matrices, but shows
no useful warmed C6 benefit. Three shapes are substantially slower in this
microprobe. Serving remains frozen152055a6/headpairON/depth2; S46's owner
and packer changes remain unapplied. No new serving target is achieved.

The source-only capacity cost remains803.90625MiB for52 eligible main dense
matrices. No extra main-engine build, residency campaign or serving arm is
justified by these measurements. S41's strict cross-algorithm bitwise
failure remains FAILED; S44 uses the separately declared source-derived
arithmetic contract, not a tolerance chosen from observed differences.

COMMAND -> Actual Claude Opus5.5 and Fable5.1 review the complete collector,
CPU checker and narrow warm-timing delta before the first build. One offline
CPU-only build takes101s. One42s quiet card1 window collects96 snapshots,
checks memory and same-algorithm graphs, and measures warmed C6 batches in
the same process. A separate CPU-only11.42785s reference check follows
release of the GPU. There is no second build or GPU run for timing.

Exact identity and commands are retained under
`/mnt/vm_8tb/b70/results/flashnext_ista_gsq_rco_iq3xxs_strata_cachefix_20261014/q6-dense-reference-s44-prep/`
(called P below). Upstream remains296b59d2; PR2511,recutita PR1 and PR2
remain open with unchanged heads. No production source/archive is changed.

| Artifact | SHA256 |
| --- | --- |
| Compiled v3 collector | 40ba97c13f1e4737543be73cf5605ae05bfb9b2691586f297bea6eb90bfa0b6f |
| Frozen executable | 93f8bfc9a2c69ed900e969f8b58be009b341b468e50bc7a3b0a46cce2601eab8 |
| Executed CPU checker | 9def6b1a828e6aa6d4500910cc0d7b24529ea45d0e45ebfbede6a30da0fee679 |
| Decoder/reference helper | 1291c87fbfa21fa8ddc27d81bbbdf5d9cd3fcb8bc1193c4c510f9383dad7004b |
| Executed pins | f8f65c20b0cae340a9d78ecaa4b16261edc96ccf331638f7f41650fbe681e170 |

The actual build is `P/build-v3.sh`; actual GPU lifecycle is
`P/ROOT_RUN.sh -> P/leased_run.sh`, with CPU-quiet outside the card1 lease.
Device masks and executable PCI assertion pin physical44:00.0. Source and
library identities are checked before execution. Container health runs
before and after the numerical process, including failure paths; it exits0
and is removed. Both health checks exit0. No card0, collective, server,
reset, unitrace or external profiler is involved. Queue profiling remains
enabled for the explicit event spans. The build has no GPU device mount.

CPU command, from the repository, with NEW_OUTPUT_DIR not already present:

```
B70_CPU_LOCK_TIMEOUT=0 B70_AGENT=S44-full-reference bin/cpu-run \
  python3 "$P/check_collection_v3.py" "$P/run-clean" NEW_OUTPUT_DIR
```

RESULT -> Four actual Q6_K tensors use the saved native Q8_1 operands.
Columns6,1,2,3,4,8 are checked for both paths; each path also has initial
and changed-input C6 graph replay and changed-input direct comparison.
The original graph is captured before packing; the packed graph afterward.
Both are destroyed before packed storage is released. Own direct/graph equality,
finites, all guards, inactive tails and complete FP32/Q8/weight immutability
are fatal gates. Cross-algorithm differences are recorded without aborting
collection. The collector exits COLLECTED_UNVALIDATED; CPU analysis decides
the arithmetic diagnosis afterward.

| Tensor | K -> rows | Checked outputs, both paths/cases | Largest absolute reference error |
| --- | --- | ---: | ---: |
| blk.3.attn_output.weight | 6144 -> 2560 | 215040 | 7.528e-7 |
| blk.4.attn_qkv.weight | 2560 -> 10240 | 860160 | 5.769e-7 |
| blk.27.attn_q.weight | 2560 -> 12288 | 1032192 | 1.017e-6 |
| blk.1.attn_gate.weight | 2560 -> 6144 | 516096 | 5.080e-7 |

All72 cases and2,623,488 output comparisons pass their predeclared bounds.
Graph/repeated-prefix cases reuse a reference only after byte checks prove
actual Q8 and own outputs consistent. These are not2.6million independent
random trials. CPU analysis reads all73,113,600 payload bytes, verifies
payload FNV identities and records SHA256, checks19 earlier scalar rows,
new-shape scalar rows and exact rational spots. Peak RSS is67,580KiB;
requested BLAS threads1, block128. Explicit array budget64MiB excludes
external BLAS workspace. No full47GB shard read is required.

The storage decoder computes the real dot of stored Q6 and saved Q8,
not original FP32 activations or model output quality. The unchanged
[S43 proof](20261015_strata_q6_cpu_reference.md) extends to K2560 with
source depths12/15 for original/packed (K6144 stays15/18). Each bound is
`(gamma_L32 + gamma_(K-1)64) * A_upper`, outward rounded, with
`A=sum(abs(w*x))`. Finite-half dyadic grids, exact FP64 products, actual
half-scale finiteness and precise AOT flags support the assumptions.
CPU FE_TONEAREST is checked; GPU RNE is a build assumption, not newly read
hardware state. Wrong-row controls detect gross misassociation only.

Configuration is fixed to packed GRID640, original WIDE/A2/RPW1, PF0,
wide columns through8 and minimum0; these are not tuned per shape.
The optional timing phase uses a separate guarded original weight copy
which is never registered as packed. It matches its own original output;
the packed path matches its own packed output. Initial input/Q8 is restored,
16 calls per path warm up, then three paired16-call batches run AB/BA/AB.
Guards, outputs, inactive tails and immutable buffers are checked outside
each batch. Full weight checks surround the phase. No registry toggle or
live graph mutation is needed.

| Warm C6 matrix | Original, us/call | Packed, us/call | Change in median |
| --- | ---: | ---: | ---: |
| 6144 -> 2560 | 63.609 | 63.466 | -0.225% |
| 2560 -> 10240 | 74.907 | 142.584 | +90.349% |
| 2560 -> 12288 | 90.790 | 162.209 | +78.664% |
| 2560 -> 6144 | 47.342 | 87.863 | +85.592% |

These are medians of three batch means, host submission through final
completion, with16 calls per batch. They are not48 independent latency
samples or per-operation p95 values. `warm-report.json` retains every batch,
paired changes and separate host-submit, post-submit wait and event spans.
Same-queue barrier spans include submission bubbles and barrier cost; they
are not isolated GPU kernel service. Tiny first-shape differences have no
significance claim. All shapes share one fixed collection order on one
card; this is a warmed microprobe, not a full-model regression benchmark.
Repeated identical weights may behave differently from serving that streams
multiple matrices. No cache occupancy or physical bandwidth is measured;
S41's cold observation remains unexplained rather than refuted.

VERDICT -> Reject integration for this work item. The earlier S41 single
cold profiled observation cannot establish speed; warm paired data supplies
the missing discriminator. Bounded arithmetic success does not establish
all52 matrix semantics, production/model quality, concurrent serving,
stability or end-to-end performance. None is needed to reject this candidate.
Do not spend a full serving/cache campaign on it. Larger dense, HC and
synchronization pools remain the source/trace audit priority.

Actual post-result reviewers report model IDs `claude-opus-5-5` and
`claude-fable-5-1`, exits0 in34.79s and59.24s. Both agree to skip integration
and serving, request explicit configuration/cache-regime/batch-mean limits,
and favor source audit before another measurement. Root does not adopt
their inference that event-span agreement proves isolated device service,
or their unmeasured bandwidth/cache-capacity estimates. Those require
separate counters and joined readiness evidence.

The development loop used one build and one GPU window for numerical
coverage and warm timing, followed by saved-output CPU analysis. Reusing
that process avoided another startup/health cycle. No routine daily-server
restore, broad knob sweep or repeated200k campaign was run.

Raw evidence: `build-closure.json`, `root-build-decision.json`,
`run-clean/{identity,closure}.json`, `run-clean/oracle.{log,stderr}`,
`cpu-full/gate.json`, `cpu-full/fixture*-gate.json`, `warm-report.json` and
both actual-model review directories. Source copies under
`strata/flash-next/lowlevel_probe/` are exact inspection copies. The checker
and timing analyzer run from the retained P bundle, whose relative pins
also require preserved earlier versions, prior S41/S43 evidence, archives
and build files. They are not standalone portable launchers.

Frozen raw manifest `q6-dense-reference-s44-frozen-manifest.json`:200files,
92,289,057bytes, SHA256
`ccb3ef69b9faaa19207de4cb6b33ec9725c275ad10a49dc178045d5453645e27`.
