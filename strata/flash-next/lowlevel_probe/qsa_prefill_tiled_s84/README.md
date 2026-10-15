# S84 private QSA prefill tiled scorer records (2026-10-15)

Exact small historical records; record-copies.json binds each original path,
bytes and SHA256. Large40data arrays275523200B, executables, object/image
wrappers, archives and full linker logs remain external. Do not execute these
copies in place: absolute controller paths name completed raw runs, and no
binary or fresh root build closure is supplied here.

CONFIG -> S81 source lead, private8-query tiling of existing multi scorer.
Five synthetic deterministic shapes: nq256 ending n_kv256/4096/32768/200000,
and ragged nq9 ending200000. All four n_kv mod4 boundaries, score stride50002,
pooled50001rows including spare tail. Public frozen QSA scorer and unchanged
public top-k anchor every active score/ID. Private arithmetic body is copied
bytewise: original float4 expression, XOR/head order, causal/dead/+1e9 rules.

COMMAND -> v1 offline build stops after public prelink: noncopyable device_ext
assigned by value. v2 changes host handle to const reference plus paths/pins;
v3 removes unused private launch property after actual conditional source
reviews. Complete textual deltas preserve these changes. No kernel arithmetic
or oracle data/protocol change; v2/v3 AOT wrappers are byte-identical. The
co-residency explanation is not accepted as an observed local runtime fact.
No main QSA TU recompile or native-common AOT regeneration. Final build
passes in28s; public prelink7.450s, adapter prelink0.434s, ordinary host link
0.189s. Two wrappers contain31images, unique private/public scorer/top-k IDs;
eight generic memcpy/fill IDs overlap, so no blanket ODR claim. Final executable
SHA dd63db5a427f2670e59bd2e1f17d4ab6471076299d9667bef83aaf2bcd32832e.

One CPUquiet/card0/physical0b:00.0 eager process: all five correctness cases
before3warm/arm+four AB/BA/BA/AB paired scorer+unchanged-topk samples. No graph,
DOT, external profiler, model/GGUF or serving run. Reset/readbacks/checks stay
outside timing; completed wall includes host submission and final queue wait.

RESULT -> Quiet window92s, leaf66s. All full valid score/ID bytes, repeats,
changed/restored, known observable1->0 score, inactive poison, guards/read-only
PASS. Forty exported binaries275523200B match. Both health checks0, own fullCID
and NAME absent, wrappers0, no retries/reset. Source review records contain
actual Opus5.5/Fable5.1 IDs and conditional source recommendations; root's
closure resolves the versioned deltas before GPU. Later ongoing combined
S85-source/S84-post consultations are deliberately outside this completed
manifest and may be linked by root separately.

| Case | Median of four paired completed-wall changes |
| --- | ---: |
| near0, nq256 | +177.194% |
| near4k, nq256 | +15.045% |
| near32k, nq256 | -7.472% |
| near200k, nq256 | -17.677% |
| near200k, nq9 | -2.587% |

All raw timing pairs remain in comparison.json and leaf.log. Four nested
pairs in one process are descriptive; near200k varies across pairs. No
isolated scorer service, bandwidth, occupancy, causal hardware, full-prompt
TTFT/throughput or serving-quality inference follows. Static128GRF/SIMD32/
SLM16512/barrier1 is compiled metadata, not measured occupancy/runtime spill.

VERDICT -> Retain PRIVATE long/full256-query lead only; short cases regress.
No production integration, serving arm, retry, threshold tuning or promotion.
S30's growing select bucket is not wholly scorer service or a removable budget.

Whole completed evidence:187files282165114B, manifest SHA
33047e7ca2aaacf7793ad02209ea08f63c3ab4a2fc5601cd634ac49f5255adb3.
Raw roots under
/mnt/vm_8tb/b70/results/flashnext_ista_gsq_rco_iq3xxs_strata_cachefix_20261014/:
qsa-prefill-tiled-s84-prep[-v2|-v3]/ and
qsa-prefill-source-reviews-s84-20261015/.
