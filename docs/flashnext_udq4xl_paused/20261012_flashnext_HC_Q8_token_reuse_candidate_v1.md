# Default-OFF HC Q8 token reuse candidate

CONFIG -> Actual source52 HC Q8_0 weights, unquantized F32 activation, exact per-output lane FMA column order and sum32 XOR16/8/4/2/1. Target down K10240/M320 and up K320/M10240. This is token reuse, not source46 row batching, GEMM, F16 re-quantization or altered HC arithmetic. Frozen52 is untouched.

COMMAND -> Source patch candidate-default-off-hc-q8-token-reuse-v1.patch plus standalone kernels/hc_q8_token_reuse_fixture_v1.cpp. Root must compile fresh patched source in C388/39992 NoDri before any operator GPU run. A minimal generic compile command after patching a new source copy (no original source edit) is:

```
icpx -fsycl -std=c++17 -O2 -I/src/sycl/include -I/src/include /fixture.cpp /src/sycl/src/kernels/hc_native_projection.cpp -o /build/hc_q8_token_reuse_fixture
```

No AOT/device-target option, model mount or GPU exposure for compilation. Actual compile ABI/header compatibility remains untested by author. Root operator execution uses bin/gpu-run --card0 and ONEAPI_DEVICE_SELECTOR=level_zero:gpu ZE_AFFINITY_MASK=0, exact fresh ELF/source, pre/post health and normal owned retirement. No old SDK52 runtime proof transfers; model integration requires fresh all8 rebuild/actual fixture first.

RESULT -> One32-lane subgroup owns one output row and2/4 consecutive tokens. Q8 half scale and signed code decode once per lane/column and feed independent token accumulators, each visiting lane,lane+32,... in original order. Each token uses the SAME sum32 helper; token validity is subgroup-uniform. Row-major output[token,m] unchanged, pointers/geometry captured by value. No new allocations, local memory, barriers, atomics, GPU queries or host/device waits. Distinct input values still need loads: input load count is NOT claimed reduced. F32 injection kernel is unchanged. Tile absent/0 uses old kernel;2/4 only changes multi-token Q8 calls; T1 production stays old. Invalid flag/tile refuses before kernel submission.

Actual source52 projection shape bound is tokens<=8 (hc_native_projection.hpp). Actual prefill.cpp2867 splits chunk T into batches<=8. Recorded launch names include up groups81920 and down2560, consistent with T8 for M10240/320 under this exact kernel's group-per-output mapping. Do not interpret aggregate kernel counts as T512. Production caller/descriptor bound remains unchanged in this first candidate, so launch count is not claimed reduced. Explicit operator-only entry accepts T1..512 for fixture: old scalar reference launches each<=8 slice, candidate evaluates same entireT512. Production T512 coalescing is a separate follow-up if this reuse fixture wins.

Fixture makes deterministic finite random original Q8_0 scale/code rows and F32 activations. Shapes both directions, T1/3/7/8/512, tiles2/4, exact byte comparison against unchanged old scalar. T512 graph capture/replay twice with changed owned input contents tests by-value argument/storage lifetime; graph objects retire before buffers. Invalid tile3 rejects. No performance timer/threshold or old fixture-count authority.

VERDICT -> SOURCE CANDIDATE ONLY; no compile/GPU/model/provider by author, no bitwise or speed claim. Need actual operator fixture to establish compiler/device arithmetic parity and performance before model control. Production weights/quantization/math and decode T1 remain unchanged; all existing model identity/coherence/health/full4/teardown gates apply to later matched validation. The observed UNADMITTED trace's HC dominance is a lead, not a causal latency proof.
