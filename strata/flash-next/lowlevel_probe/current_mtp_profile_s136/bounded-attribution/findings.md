# S136 first bounded head-free step attribution

CONFIG -> Actual one107prompt/64output C2 diagnostic on frozen S133 main,
all candidate flags0. Root closed native/normal teardown/fullhealth; native
paired>0/client admission overlap/S16prefix/820CPUring dropped0 passed. GPU
raw trace was parsed ONCE; useful cycles/census/CSV saved, then root SIGINTed
only redundant global-gap CPU analysis. Status remains PARTIAL_ANALYSIS,
original analyzer-2/controller1 preserved. No full-parser PASS or C6 ON result.

COMMAND -> Read ONLY saved cycles.json5255606B, census.json, operations.csv
366770B and root partial receipt. No raw279MB trace, model/main/archive read,
new GPU/serve/profile/compiler or repeated parser. Source raw trace hash reused
f5650d4daa36dc19fde1af631d19c434bab83ac6f04fb1b5ab509be5123baef6.
Timestamp-sort the retained array: original compact event order is NOT strictly
chronological. All5151events in FIRST source-compatible cycle3 retained;
56.73325ms, twoC2heads +oneC6head,48EP. No fastest-cycle choice.

RESULT -> Narrow lower anchor is completion of first tail's two mtp_selects
IDs140990/140991, trace_ts1792125388347254.0; upper is secondC2head141196 start
1792125388348565.8. Span1311.75us;204 fully-contained primary-GPU events,
no crossing event found at these endpoints. Mapped copies140992/141093 and
repeated body motifs fit two sequential owner pre_step(T1) bodies plus shared
tail input Q8. OWNER/TENSOR pointers are not logged: roles are source-conditioned.
Source local graph order supports the interpretation, not exact buffer identity.

| Disjoint recorded name family | Calls | Service sum us |
| --- | ---: | ---: |
| Q8 dense/shared projections, all shapes | 18 | 330.824 |
| HC norm/down/reduce/UP, six ordered bundles | 24 | 261.762 |
| Attention chunk/merge | 4 | 183.539 |
| Other named kernels | 40 | 128.214 |
| Routed GU/down | 4 | 92.186 |
| Q8 input quantization | 12 | 29.682 |

Q8 q_proj-compatible M12288/C1 grid3072 IDs141010/141111 total121.666us.
Position-compatible o_proj IDs141017/141118 after attention/Q8 total73.437us.
All six M2560/C1 Q8 kernels total119.684us ALSO include FC/shared operations;
calling all of them o_proj would overstate its cost. Exact original names/IDs
and relative timestamps retained in204-row headfree-events.csv/facts.json.

HC consists six4-kernel bundles: norm6/40.520us, down6/91.456us,
reduce6/21.039us, UP6/108.747us. Source matches attention/MLP/final HC for each
T1 owner. This directly scopes sibling's three-role paired-HC idea: actual
head-free HC pool is261.762us in this interval, NOT whole-cycle historical
6.718ms. Pairing doubles per-call tokens; fewer launches does not predict saved
wall time, new kernel resources or floating-point equivalence.

Named-kernel service sum1026.207us; named-kernel coverage union1008.084us
(overlap exists). Another102 GPU-lane intervals named eventReset/barrier total
125.376us; these are NOT CPU API timing, and their engine/profiler contribution
is unproved. Union over all204records1095.440us. Residual216.310us is merely
missing collected coverage within this interval, not physical idle, data
starvation or a removable host wait. First mapped copy starts23us after the
select anchor; mapped device-copy service is not a CPU staging duration.
Large epoch timestamps have finite start resolution; tiny gaps are diagnostic.

The surrounding56.73325ms cycle includes verifier/commit/other work. Its first
pre-round boundary uses different multirow inputs; this j>=1 result is not all
MTP depths. Whole-collection CSV rows and source shape compatibility supply
context only; do not multiply counts into a serving budget. Sibling owns the
separate retained verifier-phase ranking, avoiding double counting.

VERDICT -> Evidence supports a bounded paired-HC private SOURCE experiment
before broader T2 projection/attention changes: six bundles are visible and
existing per-token FusedGrArgs accepts separate owner pointers. Q8 projections
also exceed HC service as a family; input quantization alone is a smaller
recorded pool, and attention cannot be naively batched over disjoint KV pools.
This ranks source leads, not predicted gains. One instrumented nested cycle,
no clean latency/population/critical-path savings claim. CPU steady/client and
peer clocks remain UNJOINED; GPU families/API spans/CPU waits are nonadditive.
