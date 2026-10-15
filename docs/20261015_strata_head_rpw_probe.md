# S52: existing Q5_K head RPW2 discriminator (2026-10-15)

CONFIG -> Existing two-row-per-subgroup Q5_K head kernels reduce this bounded
captured quantize+head graph's median time by16.197% at C2 and31.454% at C6.
Full-byte and predeclared shape screens pass. A head-specific serving candidate
is justified after the separate HC campaign; no whole-model gain is qualified.

The S31 current159-operation CSV ranks primary non-head dense at13.899ms/
468calls per reviewed cycle, paired C2 heads at2.416ms summed service/two
calls, and main C6 head at2.262ms/one call. These are ranking observations,
not additive latency savings. S52 targets the full-vocabulary head only:
N2560/M248320, Q5_K type13,437043200 actual output.weight bytes. 62080 is
RPW1's work-group count, not the vocabulary size. RPW2 uses31040groups;
both use four SIMD32 subgroups/work-group and the same original quant bytes.

Public native_quantize_q8_1/native_q5_k_mmvq APIs dispatch the already compiled
WideQ5K branch. RPW2 shares activation loads between two rows; each row keeps
the original reduction order/expression and XOR tree. That source mechanism
does not establish actual traffic, compiler instruction identity or occupancy.
The production STRATA_MMVQ_WIDE_K_RPW env is latched per process and affects
other K-format projections. It is unsuitable for an isolated head-only serve.

S25's real-weight head curve covered RPW1, not this RPW2 branch. S52 reuses
seed7 normal finite activations plus -0.7*x+0.001*(i%17), with saved actual
input bytes. These are seeded inputs, not captured serving activations.
C2 uses the production paired tail's two C1 quantizers into adjacent Q8
columns; C6 uses one C6 quantizer. All calls run on one in-order queue.

COMMAND -> Actual claude-opus-5-5 and claude-fable-5-1 reviewed the exact
leaf/protocol (105.873/297.243s), both ACCEPT. Both missed a C++ ADL collision:
unqualified equal(Bytes,Bytes,"literal") could select std::equal. First offline
build fails after14s before GPU. Root preserves v2, makes v3 with only the
helper/calls renamed strict_equal, and proves inverse rename reconstructs the
reviewed source. One101s corrected CPU-only adapter build succeeds, with no
native kernel TU recompile, DRI or network. AOT relinking generates new device
images under audited bmg-g31/precise/per_kernel flags and four parallel jobs;
this is unchanged kernel IR, not an identical previously qualified GPU binary.

One89s quiet physical card0 PCI0b:00.0 window contains four sequential fresh
processes,61s combined leaf wall time: A1 RPW1, B1 RPW2, B2 RPW2, A2 RPW1.
Each shape captures a single-call graph and a20-call graph, checks correctness,
warms the batch three times and records three timed batch replays. Fixed shape
order is C2 then C6. The primary metric is same-queue barrier span/20; begin is
waited before host t0. Host complete/submit spans are retained separately.
No reset/copy/host wait is captured inside the graphs. No external profiler,
server, collective, second-card work, broad width scan or routine server restore.

Raw P:
`/mnt/vm_8tb/b70/results/flashnext_ista_gsq_rco_iq3xxs_strata_cachefix_20261014/decode-next-s52-prep/`.
Historical build/window recipes are P/v3/{build,ROOT_RUN}.sh; cpu-run owns the
offline build, cpu-run --quiet wraps gpu-run --card0 with mask0 for the finite
window. One pre/post health surrounds all four processes; each container is
removed. No automatic retry/reset or later arm after a failed arm.

| Artifact | SHA256 |
| --- | --- |
| Executed v3 leaf | ae4deafcf6eb5ec50e1deffcdbff07a2e6b812d8b4c6af152f5edd4fea3f2533 |
| Executable | 59e485b4f947af7cd5cc3cb807650fffe3f89664cf89c2f1176497683822303c |
| Native archive | 498e0d1949f062d288ed92dfce441d95256e72f1c18e91f19943b600a9c4b992 |
| Core archive | 34e91c40f01b7aa0e652e195228dfc1fadba93e40006c12788c81426567b0286 |
| Executed comparator | 625c400a45f0c34b339ab6247af2a657c8c8a2f46738e79342600137d4709101 |
| Actual link map | 21749ccd1d0a11e9b503ccd96d4cc3a2a0f5eeeacd8dc50416a5bc121f565669 |

Build/runtime image ID is
`sha256:e41d59c0940e1bd2f6f64c9d5d2d7452b77812b0a8407fac8d67cacec5d7fdd2`.
The exact head slice at shard1 offset11024256 has SHA256
`93f1c7cb7a7cf8134ce9f8cf80c5416df968f7f1ddcc16f2159a9521080826a5`.
Root freezes both archives under P/frozen-inputs with identity.json. Future
reproduction uses NEW scratch and mounts those exact archive files read-only
over the recipe's container link paths, plus pinned headers/source and image.
Do not restore frozen archives into the live build tree or run the historical
live-path recipe unchanged against later archives. Repository
[leaf](../strata/flash-next/lowlevel_probe/head_rpw_leaf_s52.cpp) and
[comparator](../strata/flash-next/lowlevel_probe/analyze_head_rpw_s52.py) are exact
inspection copies; retained raw recipes/dependencies establish the executed run.

RESULT -> All four leaf exits, full-byte comparator, driver/tee and pre/post
health exit0. All containers removed. Quantizer-only Q8 is retained before
eager head; full initial/changed eager logits are saved before graph invocation.
Own initial/changed graph, batch-before-warm and each timed-final full outputs
match. Float inputs, Q8 and weights remain unchanged; guards, tails, finiteness,
argmax and changed-input observability pass. Numerical/structural checks precede
strict comparison; streamed full-weight readbacks occur outside timing.
All96 binary exports total128675840bytes; cross-process comparison uses72
files/96506880bytes against A1, not72 independent fixtures.

Independent parsing/demangling of actual16 verbose DOTs finds420 kernel nodes:
252 quantizers,168 heads,404 edges, every graph a single linear chain. C2 single
is2quantizers+1head; batch40+20=60nodes/59edges. C6 single is1+1; batch20+20=
40nodes/39edges. A1/A2 capture original native_mmvq_wide; B1/B2 capture explicit
native_mmvq_wide_rpw<WideQ5K,C,2>. C2 actual input offsets are10240bytes and Q8
offsets2880bytes; head reads the first Q8 base. Batch weight/Q8/output pointers
stay fixed. C1/C6 quantizer scalar arguments are source-bound, not decoded from
opaque DOT fields. This proves captured contents/order, not an independent GPU
dispatch trace, execution count, ISA, physical stores, cache hits or bandwidth.

| Shape | RPW1 median us/call | RPW2 median us/call | Change | Absolute delta / max same-mode range us |
| --- | ---: | ---: | ---: | ---: |
| C2 | 1223.192725 | 1025.067700 | -16.197368% | 198.125025 / 77.130200 |
| C6 | 2525.101575 | 1730.843750 | -31.454490% | 794.257825 / 152.984400 |

| Process | RPW | C2 three batch means us/call | C6 three batch means us/call |
| --- | ---: | --- | --- |
| A1 | 1 | 1294.005,1222.724,1223.662 | 2605.250,2517.698,2457.995 |
| B1 | 2 | 1046.385,1025.521,1024.615 | 1750.781,1730.464,1693.729 |
| B2 | 2 | 1049.188,1020.984,1020.083 | 1754.516,1731.224,1694.213 |
| A2 | 1 | 1295.313,1218.182,1218.208 | 2610.979,2532.505,2475.271 |

Both candidate process medians are below both control process medians; pooled
absolute median changes exceed maximum within-variant ranges and2%: all three
predeclared screens PASS independently at both shapes. Every candidate batch
mean is also below every control batch mean. These six means are nested in TWO
fresh processes/variant, not six independent trials or same-process pairs.
Within-process first-to-last times trend downward despite warming. Init/load,
readback and process clock/cache state differ from serving; no causal warming
explanation, isolated GEMV service, individual-call p95 or stability is inferred.

VERDICT -> Source-aware strict primitive and signal gates pass. Actual resumed
post-result reviewers claude-opus-5-5 (8.839s) and claude-fable-5-1 (9.244s), both
exit0, recommend a default-OFF exact head-only serving candidate AFTER isolated
HC testing, no extra primitive rerun. They recommend production-activation
logit parity and matched real serving evidence before default changes. Opus's
phrase "two real inputs" is corrected to real weights plus seeded inputs;
Fable's original missing-rg failure-direction explanation was incorrect, though
the tiny grep dependency cleanup is retained. Reviews guide decisions; they do
not replace source/compiled/runtime evidence or create an automatic new campaign.

No full-model delta, serving qualification, physical traffic/occupancy conclusion
or global RPW serving change follows. Existing170 reference map operations and
rollups remain historical; S52 is a separately scoped measurement. Parent retains
qualified configuration recovery and records the whole-stage raw manifest.

Loop: one cheap failed compile, one corrected adapter-only build and one card0
window; no extra isolated rerun. A small offline C++ frontend pass before long
remote source reviews could catch overload errors that bash/Python syntax checks
cannot. Post-review reused the original sessions and sent only an11KB result
packet; both finished in about9s while separate HC work continued.

Frozen whole-stage raw manifest `decode-next-s52-manifest.json` covers240files/
222195943bytes, SHA256
`f301c25c1121b7c14203ad8ed5ca7ab05752e63a97ec492ee8f8e3e3ea8a2f5f`.
