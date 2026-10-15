# S48: HC duplicate-store predicate, bounded probe (2026-10-15)

CONFIG -> Arithmetic and graph checks pass, but warmed performance remains
inconclusive under the rule declared before execution. Keep the predicate
unintegrated and qualified152055a6/headpairON/depth2 unchanged. A separate
graph-replay discriminator is justified; a serving run is not yet justified.

Default HC UP has160 workgroups computing the same lo[T][320] values from
immutable partials. Every group uses its own SLM copy, yet all write the same
global lo output. Existing LO_ONCE=true restricts that global store to group0.
The candidate selects it for the existing C16/non-ILP/GRF0 path, keeping
geometry, dots, reductions and epilogue unchanged. No other launch branch
changes. Consumer/source audit finds no in-kernel reader of global lo and
separate caller workspaces. This is a source-level redundancy finding;
logical duplicate stores are not measured physical traffic or saved time.

S31 records109 such UP calls/cycle and3.598ms summed service, not a removable
wall budget. The inferred mix is97T6/10T1/2T3. The new timing covers T6 only.

COMMAND -> Actual Claude Opus5.5/Fable5.1 exact source/adapter/build reviews
return READY. Root matches production Ninja flags, including version define,
precise math, target, subgroup, split and includes. One44s offline CPU build
compiles a private copied TU and S39-derived adapter, replacing only one
member in a copied archive; production source/archives remain read-only.
One45s quiet card1 window runs both variants, strict gates then warm timing.
Both card health checks and the oracle exit0; the named container is removed.
No server, model load, card0, collective, reset or external profiler is used.

Raw root P:
`/mnt/vm_8tb/b70/results/flashnext_ista_gsq_rco_iq3xxs_strata_cachefix_20261014/hc-lo-once-s48-prep/`.
Actual recipes: `P/build-root.sh -> P/build-container-root.sh`, and
`P/ROOT_RUN.sh -> P/leased_run.sh`. Both use disk scratch; GPU execution uses
cpu-run --quiet outside gpu-run --card1 plus device mask and executable
physical-PCI44:00.0 assertion. The offline build has no DRI/network. All
oracle exits receive post-health and owned-container closure checks.

| Exact artifact | SHA256 |
| --- | --- |
| Production candidate patch, unapplied | b3b4a4c925cf10c1448b53ab957351b6c991cee125b70b2e5b3305cee9e2239f |
| Private probe TU | 26edf194bd79418c061f27bc9f08605e371a1676864096164d375b606b71351f |
| Executed adapter | 67377dd91fbe3eaa0fa9b23fa32673fb7a7f490e661b8e47d9305b97c1059a5c |
| Executable | 7e3f1345f7bf8d0a8a567f3f32c52393e42d5ce69deb1bdfd3938b8b388c6de0 |
| Private archive | 83e726418f6b91e00d7af1ce4983baf71db2ac461affa1206514609e48ebbe0e |
| Replaced member | 1ff40f83b7c9deeaf63bed829829f0c5bf8040330728174e17d59b68ced222c1 |

The private TU adds a default-off host selector and a distinct candidate
kernel name. Root verifies one actual fused_gr_read_multi and selector
symbol, one replacement member, and its link-map binding. These are source/
build/host-path evidence, not an independent GPU dispatch trace or a proof
of all DPCT ODR properties. Bit-identical outputs cannot identify which
variant executed. A future graph dump may strengthen binding evidence.

RESULT -> T1/2/6 xapply0/1 xinject0/1 gives12 fixture cases for both variants.
All36 full snapshot pairs match byte-for-byte:26,919,936bytes, including lo,
R, rs, inject, mixed and xn. Own initial/changed T2/T6 graph replays match
their direct outputs; graphs are destroyed before the next selector change.
Guards, finite active values, inactive tails and immutable inputs/weights
pass. All36 control phases also match the preserved S39 mode0 snapshots and
their recorded per-file SHA256 values. No old profiled timing is used as a
performance comparison. Three additional timed-final pairs match exactly;
78 snapshots are retained in total.

Only after these internal strict cases pass, T6/apply1/inject1 warms8 calls
per variant, then runs three paired32-call batches in AB/BA/AB order. All
batches reset identically. Repeated apply intentionally updates R on every
call, equally in both variants. Measurements are direct enqueue of the whole
HC read, not graph replay or isolated UP kernel service.

| Round | Order | Original complete us/call | Candidate complete us/call | Paired change |
| --- | --- | ---: | ---: | ---: |
| 0 | original then candidate | 66.855 | 59.591 | -10.866% |
| 1 | candidate then original | 59.539 | 57.999 | -2.587% |
| 2 | original then candidate | 61.714 | 55.523 | -10.031% |

These are three batch means per variant, not96 independent latency samples.
The ratio of medians is61.714->57.999us/call (-6.019%). Before the run, root
required all three paired signs to agree, at least2% absolute change, and
the median absolute difference to exceed the larger within-variant range.
The3.715us median difference is smaller than the7.316us control range;
the declared screen therefore FAILS. It is not a qualified6% improvement.

Host enqueue means fall53.0->37.5->28.1us/call in control and
45.0->34.4->25.8 in candidate. Larger apparent gains coincide with candidate
running second. The median enqueue/event-span ratios are0.611/0.596;
these overlapping intervals are not an additive decomposition or proof of
host dominance, physical starvation or device-store savings. Submission
backpressure and temporal effects remain unresolved. Same-queue barrier
spans include submission gaps and barrier cost. No bandwidth, L2 behavior,
physical store count, occupancy or isolated device delta is measured.

VERDICT -> Both actual post-result reviewers, model IDs claude-opus-5-5 and
claude-fable-5-1, agree to call benefit inconclusive and skip integration.
They favor a graph-only discriminator that matches production dispatch and
reduces host submission noise. Root retains those recommendations but does
not adopt their quantitative position-bias, host-dominance or cache-traffic
inferences. Source/build evidence is distinguished from independent observed
kernel dispatch identity.

The next adapter can reuse the same relocatable IR member/private archive,
while final linking regenerates AOT device images. It must not be described
as the same tested GPU binary. Recheck bounded T6 initial/changed and timed
final anchors; repeating all12 unchanged-kernel fixture cases is unnecessary.
No direct-enqueue rerun, knob sweep, main-engine build, serve or200k campaign
is justified by this result. The default-off PLE wiring candidate is prepared
independently, with minimal aggregate reuse proof, and remains unapplied.

Raw evidence: `root-build-decision.json` (predeclared screen),
`build-closure.json`, `run-clean/{identity,closure}.json`,
`run-clean/oracle.log`, `s48-analysis.json`, `s48-report.json`, and actual
review event streams. Repository copies preserve the exact adapter, private
TU delta, production candidate and analysis script. Use the retained P bundle
for relative source/archive/data dependencies; these are inspection copies.

Frozen raw manifest `hc-lo-once-s48-frozen-manifest.json`:149files,
85,892,469bytes, SHA256
`d76c95ee913b3d986a520b3ef37917a256b76fa6814eca60d301310d37b6170d`.
