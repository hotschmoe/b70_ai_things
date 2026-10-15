# S57/S58: pruning the next Strata candidates (2026-10-15)

CONFIG -> Queue only a narrow IQ4_XS numerical-oracle preparation. Defer Q4/Q5
primary RPW2 and expert-ownership serving experiments. These are source/history/
compiled-resource decisions, with no new GPU run or serving-speed measurement.

COMMAND -> Independent source audits by strata_ep_s27 (WideK history/header join),
ple_upstream (existing AOT resources) and mtp_q4_share (EP balance/controls).
Read retained experiments and existing model headers; inspect existing binary
metadata. No model-weight payload, compiler rebuild or live source change.

RESULT -> Generic WideK RPW2 is distinct from the rejected Q6 knob. S15 tested
only synthetic Q5_K C1..4: corrected median latency ratio0.946101, mixed by shape;
tiny K/V regressed. RPW4 ratio1.310344 was rejected. Q4_K/IQ4_XS real C6 was not
covered by that audited history. Source preserves per-row arithmetic/order, but
that is not a new compiled numerical pass.

S31's Q4_K/IQ4_XS/non-head Q5_K pool totals5.384218ms/126calls per reviewed cycle.
Header metadata matches the format/shape/count groups; it does not identify each
trace event's layer. The largest per-format representative groups yield real
Q4 qkv2560x10240, IQ4_XS qkv2560x10240 and Q5 out6144x2560 tensors. Exact slices
are retained in the audit; no weights were read to reach this decision.

The existing S55 executable's six C6 resource records narrow that list:

| Format | RPW1 spill/scratch fields | RPW2 spill / scratch bytes |
| --- | --- | ---: |
| Q4_K | absent | 2112 / 4096 |
| IQ4_XS | absent | absent |
| Q5_K | absent | 1728 / 2048 |

All six declare128GRF/SIMD32/eu_thread_count8. The successful metadata-only scan
reads2.631MB in0.03957s. These are compiler declarations, not observed occupancy,
traffic, latency or measured runtime zeros. Spill presence alone does not prove
slower execution or negate the separate Q5 head results. It is enough to
prioritize one IQ4_XS C6 actual-weight discriminator over a three-format sweep.
A later oracle must still pass full bytes, changed-input graph, memory guards
and actual branch binding before interpreting warm timing. No global serving
RPW toggle follows. The selected IQ4_XS representative is blk.0.attn_qkv.weight,
type23, N2560/M10240,13,926,400bytes, shard1 [741126016,755052416).

The proposed ownership lever is weaker than raw149/363 counts suggest. S31's
routed-expert service is7.738ms primary versus8.066ms peer,144calls each. Peer
27.1ms wait precedes new input; it cannot execute fixed primary dense/HC/head
work through a quota knob. Separate-device times do not establish physical idle
or an additive savings budget. Hottest profile ranks are already snake-dealt
approximately evenly until primary capacity fills.

Lowering primary149->144 would move only ranks288,291,292,295,296 per layer,
whose actual traffic is unobserved. It keeps kernel counts fixed. The existing
--expert-cache5181 control is a largest-blob BYTE BUDGET, not144slots/layer;
source predicts144/368 only if current free memory is not the binding cap.
Keep reserve1024: increasing reserve also raises the peer requirement.

Five extra peer bundles cost399.658MiB. Same-scope saved final peer free2010MiB
would hypothetically become1610MiB; the earlier31.84GiB-before-experts number
omits later workspace. This is allocation arithmetic, not a measured new free
sample. Any future arm would need actual both-card final>=1024MiB, all24576
resident, exact intended counts and no rescued context/prefill/cache reduction.
Prefill ownership changes local/peer weighted-sum grouping and may alter rounding,
text and acceptance. A matched speed comparison cannot assume those gates pass.

VERDICT -> IQ4_XS-only C6 oracle preparation is justified; no new build/run here.
Q4/Q5 primary probes are deferred by compiled resource risk. Expert ownership
stays149/363: no measured moved-rank traffic or expert-phase imbalance justifies
another pair of load/health cycles. Revisit with new evidence, without a reserve
sweep or a new routing collector by default. No user performance target is newly
met; [S55](20261015_strata_head_rpw_serving.md) separately keeps head RPW2 OFF.

Raw evidence is under
`/mnt/vm_8tb/b70/results/flashnext_ista_gsq_rco_iq3xxs_strata_cachefix_20261014/`:
`primary-widek-next-s57-prep/`, `primary-widek-resources-s57-20261015/`, and
`ep-balance-next-s58-prep/`. Findings, source/header joins, exact resource YAML,
image offsets and source/binary identities remain preserved there.

Frozen s57: 16files/110817bytes; external candidate-audit-s57-manifest.json
SHA256 `4343f0014a99825810d8537b5d216d19a5c8d4be4762808db31f6bdba1688e27`.

Frozen s58: 4files/13478bytes; external candidate-audit-s58-manifest.json
SHA256 `589d9e1f2dcc4b804deda9255a90d1884574a664eae9c1578d22315405af0eff`.
