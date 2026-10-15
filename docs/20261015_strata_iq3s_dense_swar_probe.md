# IQ3_S dense C6 packed-sign probe, 2026-10-15

S65 passes the strict primitive checks but gives mixed latency results. Keep
global SWAR unintegrated; no immediate serving test or repeat is justified.
The gate projection improved in all eight nested pairs and ssm_out regressed
in all eight. This is one process with synthetic activations and real weights,
not serving qualification or a model throughput forecast.

CONFIG -> Upstream base296b59d2, unchanged main source and frozen archives.
The private original mirrors public IQ3_S NC6/EXACT_N; the candidate changes
only sign reconstruction to existing packed SWAR helpers. Original weight
loads, FP arithmetic, K schedule, subgroup32 and local(1,4,32) remain.
S66 removed the proposed aligned-load change before reviews or build because
the existing compiler already emits three reconstruction d32 messages.

Two actual IQ3_S tensors, each6758400 bytes:

| Tensor | N | M | C | GGUF offset |
| --- | ---: | ---: | ---: | ---: |
| blk.0.attn_gate.weight | 2560 | 6144 | 6 | 734367616 |
| blk.1.ssm_out.weight | 6144 | 2560 | 6 | 2545748608 |

COMMAND -> Actual fresh Opus5.5 and Fable5.1 exact-source reviews both permit
one bounded build. Root read both full responses. One CPU-only adapter build
reuses S60 common wrapper8f5a7efe:23s total,10.025s compile,.511s adapter
prelink,.397s ordinary host link, no common AOT or kernel-TU rebuild. These
are different adapters from S60; no matched generic build-speedup claim.

Root versioned the host execution controller before GPU use. Docker-created
full CID AND container name must be absent in a successful enumeration before
post-health. Unresolved cleanup retains CPU/GPU leases and never starts
health beside the workload. A native peer read and checked that correction.
The original preparation runners were not executed. Added host checks compare
final saved output/Q8 bytes and bind controller hashes before/after execution.

One quiet CPU/card0 lease, physical0000:0b:00.0, one process/two shapes,
41s lease/12s leaf, no retries. Both shapes complete correctness before timing.
Public eager, private original and candidate match full output bytes for initial
seed7 and changed finite inputs. Changed output/Q8 actually differ. Single and
batch20 graphs replay both inputs. Per shape59 checked snapshots cover
finiteness,4096-byte guards,512-byte inactive output/Q8 tails and immutable
weights/input/Q8. Saved92 binaries total6155904 bytes; final four files also
match the saved public/canonical anchors. Ten DOT files contain86 nodes and
76 edges with expected classes/chains and the same three USM resources within
each shape. DOT bindings describe captured metadata, not independent dispatch.

RESULT -> All byte/graph/memory/identity/host-pin gates PASS, pre/post-health0,
own container removed before health, wrapper exits0. No card1 or server use.
Projection-only timing uses canonical Q8 prepared outside graphs, three checked
warm replays per arm immediately before each shape, eight alternating OCCO/COOC
rounds with two samples/arm/round. Each sample is a completed barrier span for
20 captured projections, divided20. Samples are nested in one process.

| Shape | Original median us/call | Candidate median us/call | Median paired change | Original range | Candidate range |
| --- | ---: | ---: | ---: | --- | --- |
| gate | 70.263000 | 68.489575 | -2.514711% | 70.039075-70.536475 | 68.429700-68.559900 |
| ssm_out | 84.160150 | 85.213538 | +1.274878% | 83.929675-84.359375 | 85.015625-85.354175 |

Ranges and medians use eight round means. They are not independent process
samples, p95, significance or stability evidence. Graph repetition also differs
from the model's working set. S59 included quantization; do not compare its
absolute timings to these projection-only measurements.

VERDICT -> Numerical primitive PASS; keep global/default implementation.
Gate-only use remains an isolated seeded-input primitive lead; no K/M-keyed
main dispatch was exercised and other IQ3_S shapes remain unmeasured. No main-source integration, server build, cache campaign,
standalone retry or shelf promotion follows. S68 separately confirms the
intended compiled mechanism but cannot explain the timing causally. Absolute
prefill/decode targets remain open; correct full RAM recovery stays qualified
only within the retained S30 scope.

Evidence root:
`/mnt/vm_8tb/b70/results/flashnext_ista_gsq_rco_iq3xxs_strata_cachefix_20261014/`.
Preparation/build/runtime: `iq3s-dense-oracle-s65-prep/`, especially
`run-clean/{closure,comparison,extra-host-post}.json`.
Actual execution and root timing reports: `iq3s-dense-s65-execution/`.
Exact recorded sources/controllers:
`strata/flash-next/lowlevel_probe/iq3s_dense_s65/`.

Source SHA256457f3dbc306b8f5484eb5cd5a4e408802e677831154b456203bfdc770c5c1931;
private headercb729c6ec4eccca931ed7eeb8f907686fc23f886713697bec1a71cf190e4684a;
executable4a81f24dcbb2dee280cfe009035e58f8c605a8f47cf970c5d080e8242ca30902.
Source preparation manifestf56a5754b425760251f1703dfcd32d550435a5207e1435d77e5555982dd52fa9.
Fresh actual source reviews: Opus63.760s/session8d53576a-99fb-448d-a00a-5141458efa83;
Fable202.552s/sessionb0d463ad-9081-499a-8963-5f01983ec5ef. Both exit0 and
responded under the requested model IDs. Root corrected review wording about
mask positions, cleanup ordering and identity preflight; model agreement is
not a substitute for the actual checks. Post-result reviews resumed those same S65 sessions: actual Opus13.511s and
Fable29.486s, both requested IDs/exit0/ACCEPT keeping global unintegrated and
no new GPU run. Root read both full responses. Balanced orders do not causally
exclude slot effects; measured magnitudes do not establish population/serving
magnitude; the public anchor covers measured cases, not universal extraction
correctness. No private/public timing equivalence or static-ISA cause is claimed.
Raw post reviews: `iq3s-dense-postreviews-s65-20261015/`.

Final S65 preparation/build/runtime/root-execution manifest:
`iq3s-dense-s65-manifest.json`,185 files/79038276 bytes, SHA256
`5aa9dcbdd3fec3d0693ad17fdf5414807d85e7c88f1effb3305db2710089bbc7`.
Post-review manifest `iq3s-postreviews-s65-manifest.json`,17 files/73884 bytes,
SHA256 `6002d7329a71741d3aa7050491dab70f1a9ee2640e2561967adbd44958b5dae3`.
Both manifests are external to their frozen roots and exclude themselves.
