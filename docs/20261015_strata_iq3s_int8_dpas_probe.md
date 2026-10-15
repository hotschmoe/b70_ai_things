# S75: canonical-Q8 IQ3_S C6 INT8 DPAS probe (2026-10-15)

The tested DPAS mapping passed strict numerical and lifecycle gates but was
about 7.7-7.9 times slower than its anchored original in this bounded primitive.
Keep production unchanged; no serving arm, retry or knob sweep follows.

CONFIG -> COMMAND -> RESULT -> VERDICT

CONFIG

The private prototype uses two real IQ3_S/type21 matrices and six seeded finite
activation columns: blk.0.attn_gate.weight N2560/M6144 and blk.1.ssm_out.weight
N6144/M2560. Each payload is 6,758,400 bytes, read from the locked NVMe shard
at offsets 734367616 and 2545748608 respectively. The exact tensor hashes,
FNV identities, model lock and file metadata are retained in weight-identity.json
and the runtime identity record. Initial seed7 and the retained changed-input
transformation feed the actual public native Q8_1 quantizer. These are real
weights with synthetic inputs, not captured production activations.

An unchanged private ControlSplit original must bit-match the public native
IQ3_S C6 kernel before checking the candidate. The candidate uses one SG16
workgroup, 16 output rows, an 8-token tile (six active, two zero) and K32 signed
INT8 matrix products. Original decoding supplies B; canonical Q8 bytes supply
A. INT32 accumulators reset each segment, then receive the odd integer scale.
The bound 32*15*127*31=1,889,760 fits INT32 and exact FP32 integer conversion.
A 32-slot per-output/token ledger preserves native lane partial order; the
lane-c-permuted final tree preserves the XOR16/8/4/2/1 source association.
Three uniform local barriers per segment order staging, tile publication and
storage reuse. Source payload is 13,708B before compiler/runtime alignment.

Native preparation was by /root/mtp_q4_share. Actual claude-opus-5-5 source
review completed exit0 in 116.901s with conditional build GO after a statement
split; actual claude-fable-5-1 completed exit0 in 255.334s with bounded build GO.
Root read both full responses. V2 introduces `const float v=d*sumi;` then
`partial+=v;`, matching native statement separation as a source precaution.
V1 remains source-only. There is no v1 compiled comparison or demonstrated
v1 contraction bug. No numerical tolerance or compiler policy was relaxed.

COMMAND

Root built v2 once under bin/cpu-run in pinned e41d59c0 Intel2026.1, with no
network or /dev/dri: adapter compile, supported adapter AOT prelink, ordinary
host-only final link. Existing common8f5a7efe, kernels archivef2b53eb4 and
core34e91c40 were reused read-only. No common AOT or kernel TU rebuild ran.
The CPU lease was 23s; compiler phases were 10.319s/.589s/.385s. Final host
link had one linker and zero AOT invocations. Exact header/leaf/binary hashes:

| Artifact | SHA256 |
| --- | --- |
| V2 private header | 8a490cc2b46b19e5d0a6bd37141f615e267852067126364e374d4bd89e8ab863 |
| Leaf | e26056959edc34e95928efd11157d0a53585bdc115c2992b6df9bc8eee470d94 |
| Executable | 18045cced6ce5539851a8e9db1e52a40bff6660a5e3aa9af4f0d019dcdae032f |

Root then ran one 38s quiet CPU/card0 leased window, physical mask0/selector
level_zero:* and PCI0000:0b:00.0. The leaf took 10s. No external profiler ran.
Public/original/candidate initial and changed eager/repeat/own single-call and
20-call graphs completed before timing. Three warm graph replays per arm
preceded eight balanced nested OCCO/COOC rounds per shape, four20-call samples
per round. Canonical Q8 quantization, reset and full-byte checks were outside
timing. Actual scripts preserve commands and CID-owned cleanup semantics.

RESULT: source, compiled image and runtime scopes

The bounded offline audit indexed 1088 embedded images. Exact public,
private-original and candidate full IDs were unique; public zebin bad2ccde is
byte-identical to S65. Candidate _ZTSN3s756DpasC6E has 17664B main text versus
16128B private original. Its image contains one static signed INT8-to-INT32
`dpas.8x8`, separate scale multiply, local tile store and barrier sequence.
The observed first token ledger path has two FP32 mul instructions then add.
This is v2 emitted evidence, not a v1 comparison or runtime dispatch trace.

Metadata reports SG16/SIMD16/GRF128/has_dpas, seven dynamic local pointers and
three global pointers. barrier_count1 is a resource count; four static
signal/fence/sync sites are not dynamic invocation counts. Total SLM, scratch
and private allocation are unreported. Shared helper name overlaps were
classified, not exhaustively proven body-identical; target math IDs and actual
numerical gates were independently checked. No occupancy or latency cause is
inferred from instruction/resource counts.

The actual oracle passed 118 checked snapshots and 92 saved binaries totaling
6,155,904B. Full outputs bit-match public anchors for both inputs and all
replays; finiteness, 4KiB guards, nonempty 512B poisoned tails, immutable input,
weight and Q8 bytes pass. Changed Q8/output observability passes. Ten DOTs
contain 86 kernel nodes/76 edges with expected classes, dependency chains and
global resource bindings. Final saved outputs/Q8 equal their initial public/
canonical anchors. Capture metadata is not an independent hardware dispatch
trace. Root confirmed created CID and container name absent; pre/post card0
health, leaf, analyzer and wrappers all exit0. No crash/reset/retry occurred.

Primary timing is host-completed wall per call, summarized as medians of eight
nested round paired means. The percent column is the median paired change,
not a ratio recomputed from rounded medians:

| Projection | Original us | DPAS us | Median paired change |
| --- | ---: | ---: | ---: |
| attn_gate | 70.8408125 | 557.928425 | +687.510385% |
| ssm_out | 84.632 | 655.79565 | +674.970007% |

All eight paired observations were slower for each shape. Optional queue
barrier-span medians were 70.97135->557.989575us and 84.7474->655.83985us,
respectively; these are separate interval measurements, not isolated kernel
service. One process supplies nested samples, not eight independent process
trials, per-call p95, physical bandwidth or full-model slowdown estimates.

VERDICT

STRICT_PASS_REJECT_THIS_MAPPING_LATENCY. This exact local-matrix/scale/ledger
implementation works for the two tested real-weight seeded-input shapes and
changed graphs. Its substantial primitive regression rejects integration here.
Conservative staging, barriers, decoding, padding and ledger work are possible
cost sources, not measured bottleneck attribution. This is not an all-DPAS
verdict, a hardware bottleneck proof, production-logit qualification or a serving
result. Production sources, daily configuration and numerical policy remain
unchanged. Root read both complete actual post-result reviews: Opus5.5
47.575s and Fable5.1 59.961s both accept this bounded rejection and no serving,
retry, sweep or barrier-elision GPU followup. They reviewed the supplied packet
without raw-file tools. Their causal explanations remain hypotheses.

Evidence and record copies

External root:
`/mnt/vm_8tb/b70/results/flashnext_ista_gsq_rco_iq3xxs_strata_cachefix_20261014/`.
The canonical v2 build/run is iq3s-int8-dpas-s75-prep-v2; static evidence is
in iq3s-int8-dpas-s75-isa; root-runtime-decision.json and root-build-host-audit.json
are in iq3s-int8-dpas-s75-root. Historical source-v2 manifest SHA is
3a89e80869a67b488aa932f2d5aad1eec9e0b303370d2da59c4619b9fb544eaa;
it is not the final whole-stage manifest. The static report's pending-runtime
status is preserved historically and superseded by the completed root decision.
Small exact records and their origin/hash bindings are at
[strata/flash-next/lowlevel_probe/iq3s_int8_dpas_s75](../strata/flash-next/lowlevel_probe/iq3s_int8_dpas_s75/README.md).
They are review copies with historical absolute paths, not runnable in place.

Actual consultations and final evidence

Source and post calls used actual claude-opus-5-5, session
`3be4cc81-16c4-4492-8151-fe48210fa00e`, and claude-fable-5-1, session
`3d339367-d3da-4437-ac8b-3f7704756eeb`. Original model responses are retained.
Opus corrected its earlier definite-contraction claim: v1 was never compiled.
Fable's post response says32pairs/shape; the actual design has8 paired round
means and32 timing samples per shape. Similar complete/span values do not
isolate a physical cause. Native traffic/spill absence and a universally best
next direction are not established. cpu-run timeout0 means wait forever,
not immediate failure; model consultations held no CPU lease.

Frozen external manifests (relative to the evidence root above):

- Completed source/build/runtime: `iq3s-int8-dpas-s75-completed-manifest.json`,
  198 files/79269638 bytes; SHA256
  `ff730eec68f5f4bc74220c58c29e330f5f4aac0fa75ebbb08a22a5b8b4ab4287`.
- Offline ISA: `iq3s-int8-dpas-s75-isa-manifest.json`;
  `357f7821b5c599715e2f65c36df798cff2d26d4c9d1fc35975dcace3024410cf`.
- Source consultations: `iq3s-int8-dpas-source-reviews-s75-manifest.json`;
  `11870f34f389b0b7230752df683c857fb31a3fa031aa11ce9a6f318b80efa822`.
- Post consultations: `iq3s-int8-dpas-postreviews-s75-manifest.json`;
  `3f9c5c55de068785074c4c750e9365f99b7dc46cc67aad31b03a222a3212ddab`.

Dev-loop decision

One23s build and one38s measurement window settle this exact implementation;
no standalone S74 runtime, server startup or context/cache repeat was needed.
For future reversible offline leaf builds, root source checks and exact pins
permit compilation under cpu-run in parallel with actual model reviews.
Resolve findings and bind any versioned rebuild before GPU execution. Build
success remains separate from source/numerical review. Future review packets
will include all timing rows and compact gate/resource summaries, with full
artifact hashes, instead of repeating97KB of snapshot/pointer records.
The next investigation is source cost/parallelism accounting; no new GPU
variant is authorized or predicted to win.
