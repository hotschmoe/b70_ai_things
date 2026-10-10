# V2 host HC F32 arithmetic adapter guards

CONFIG -> unchanged frozen V1 C++ host helper source
cf12835a10659b997446f87419d771396846a5305943778a33e1b7ddb6cbb9bf;
new PythonV2 adapter,tests and sourceplan. Original model and quantizer unchanged.

COMMAND -> PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s
strata/flash-next -p 'test_hc_f32_arithmetic35_host_cpu_v2.py'.

RESULT -> 18 CPU tests PASS through existing host libm fmaf and actual tiny
synthetic bytes; subprocess adapter positive remains mocked. This preparing
agent did not compile or execute C++/GPU/Docker/model payload/runtime workloads.
Root's separately reported CPU-only pinned-container compile and compiled
analytic tests are separate receipts, not retroactively written into V1 history.

VERDICT -> frozen V2 source ready. Device intrinsics, ordinary HC mix contraction,
full HC/native/model arithmetic and quality remain unqualified; no tolerance.
V1's historical source/evidence is preserved, including its RNE-only Python
arithmetic limitation. Use V2 instead for direct Python arithmetic validation.

RNE alone does not establish FTZ/DAZ behavior. Before each public Python operation
and each host_fma call, V2 requires normal input2^-126*0.5+0 -> subnormal2^-127,
and subnormal input2^-149*2^24+0 -> normal2^-125. A failure refuses operation
before accepting computed arithmetic. Results are not cached across calls or
threads. These are behavioral probes, not direct MXCSR flags or native GPU
property queries. Simulated FTZ/DAZ tests verify failclosed behavior; a previously
positive probe cannot authorize a later flushing call. The C++ helper remains
unchanged and already explicitly clears/checks FTZ/DAZ in its own process.

The XOR-order negative uses32 lanes with A=2^24 atlane0,-A atlane16,1 atlane1.
Exactsum1. Source XOR16 first cancels A,-A before adding1, yielding1; reversed
XOR1 first rounds A+1 toA and ends at0. Unlike a dyadicalllanes sum, this catches
a changed tree without fitting observed native outputs or moving thresholds.
Other known FMA cancellation,tie-even,subnormal,onehot,Q8half/signedcode and
lane-strided cancellation negatives remain unchanged.

V2 run_helper pins the explicitly supplied ELF SHA and frozen source bindings,
checks host metadata and probes gradual behavior before/after root execution.
It accepts caller-supplied byte operands only. Root owns actual original-data
admission and any fresh SYCL fixture; no native values enter the fulloriginal.
The optional SYCL primitive/capturepoint plan and limits remain in V1 design.
