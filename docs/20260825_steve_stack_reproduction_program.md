# Steve B70 XPU Stack Reproduction Program

Date: 2026-08-25

## Mission

Understand the complete software mechanism behind Steve Seguin's B70 W8A8
results, reproduce it from source, and own independent vLLM and sglang paths in
this repository. Steve's repositories are evidence and attributed prior art,
not runtime dependencies.

The program is complete only when a fresh checkout can build, serve, benchmark,
and explain every selected path without access to Steve's working trees or
prebuilt binaries.

## Current Evidence

- Steve's accepted Qwen3.6 TP=2 smoke reported 85.87 output tok/s and passed
  its JSON and color canaries. It used a natural-chat 512-token prompt, a
  64-token warmup, one 512-token measured generation, and ignored EOS.
- The largest accepted lever was the usable PIECEWISE graph/correctness stack:
  about 12.83 to 76.48 tok/s. Disabling custom collectives reduced 76.48 to
  70.79 tok/s. A prefill-safe GDN configuration reached 93-95 tok/s before the
  TP=2 result. P2P is useful but is not the 3-6x explanation.
- The public launcher overlays `/home/steve/src/vllm` and
  `/home/steve/src/vllm-xpu-kernels` through `PYTHONPATH`, and loads native
  libraries through `LD_LIBRARY_PATH`. The container/image environment alone
  is not the complete stack.
- The exact 2026-06-15 record tree was dirty and is not pinned. The first
  surviving 2026-06-16 vLLM checkpoint changes 84 files and adds about 17,148
  lines. The current pinned S2B image is a later 2026-08-18 snapshot.
- The later image is not standalone for Quark W8A8 XPU. It contains
  `_xpu_C::int8_gemm_w8a8` and native activation quantization, but its vLLM
  registry has no `PlatformEnum.XPU` INT8 candidate. Restoring the June class
  makes vLLM select `XPUInt8ScaledMMLinearKernel`.
- The later image also has a partial shared-expert API merge: `FusedMoE` passes
  `shared_expert_gate`, while `MoERunner` does not accept it. A narrow adapter
  is required before model allocation.
- A third narrow repair restores the no-spec uniform PIECEWISE decode
  descriptor. With all three repairs, the exact Qwen model loads on TP=2,
  compiles, captures, allocates 755,153 KV tokens at maxlen 8192, and passes
  semantic canaries with native Quark W8A8 INT8 math.
- The matched P2P-off, split-collective control produced 17.06 corrected output
  tok/s: 498 actual prompt tokens, 512 output tokens, 624.29 ms client TTFT,
  and 30.010 s decode time. Steve's accepted run used the same request shape
  and decoded in 5.963 s. The current gap is therefore a 5.0x decode-step
  runtime/graph-ownership gap, not a model-identity or raw INT8-selection gap.
- In Steve's accepted configuration, direct-P2P oneCCL collectives remained in
  the forced graph. P2P-off host-staged oneCCL requires explicit per-layer
  graph splits locally. The next safe transaction is our capturable Level Zero
  IPC push all-reduce inside replay, still with P2P access disabled.
- The image warns that no tuned B70 INT8 MoE config exists for `E=256,N=256`.
  This is a later kernel lever, not the present request-time graph failure.

## Stack Map

### 1. Base environment

Own the image recipe and lock:

- OS libraries and Python;
- torch and Intel XPU extension versions;
- Intel compute runtime and Level Zero loader;
- oneDNN and oneCCL versions;
- compiler and oneAPI toolchain;
- vLLM or sglang source revision;
- native kernel build revision.

The image digest is necessary but insufficient when live source and SO paths
shadow installed files.

### 2. Python source overlay

`PYTHONPATH` places a source checkout before installed packages. Python imports
the checkout's `vllm` modules while still using compiled extension modules and
dependencies from the virtual environment. This permits rapid iteration but
can silently combine source and native code from different revisions.

Required reproduction:

- enumerate `sys.path` and `module.__file__` for all critical modules;
- hash imported module files at serve startup;
- replace the live checkout with pinned patches baked into our image;
- reject an unexpected module origin before touching the GPUs.

### 3. Native kernel overlay and `_xpu_C`

The XPU kernel build exposes PyTorch custom operators from a shared object.
The conceptual path is:

```text
Python import
  -> dynamic loader resolves `_xpu_C.abi3.so`
  -> TORCH_LIBRARY declares operator schemas
  -> TORCH_LIBRARY_IMPL binds XPU implementations
  -> fake/meta functions describe output shapes to Dynamo/AOT
  -> dispatcher selects the XPU implementation at runtime
  -> implementation calls oneDNN/SYCL/Level Zero or a custom kernel
```

For each used op, record schema, fake/meta contract, layouts, dtype/scale
contract, mutation/alias behavior, scratch ownership, stream behavior, and
capture safety. A symbol existing in the SO does not make it reachable: vLLM
must also register/select the corresponding Python kernel class.

Required local artifact:

- source under `kernels/`;
- backend-specific build recipe;
- exact SO hash manifest;
- import/dispatch smoke test;
- numeric reference test;
- capture/replay test;
- no mount or link to Steve's checkout.

### 4. Model math paths

Trace and attribute separately:

- Quark/compressed-tensors loader and scale interpretation;
- dynamic per-token INT8 activation quantization;
- dense INT8 XMX GEMM;
- routed-expert fused INT8 MoE;
- shared-expert linears and combination;
- Qwen GDN projections, recurrent prefill, and decode kernel;
- attention, RMSNorm, LM head, and sampler fallbacks.

Every fallback must identify whether it changes correctness, capacity, graph
coverage, or speed. Do not infer that an INT8 checkpoint executes INT8 for
every linear.

### 5. Graph and scheduler paths

Trace:

- Dynamo graph creation and AOT cache identity;
- PIECEWISE split operation list;
- uniform versus non-uniform batch descriptors;
- warmup capture and later replay selection;
- GDN prefill filtering and decode descriptors;
- collective custom-op mutation/alias contract;
- input clone and stable-address handoff;
- async scheduling and host synchronization.

The late-capture failure was caused by a missing no-spec uniform decode
descriptor in the later snapshot and is repaired narrowly. The active target
is now the 5.0x decode gap between the coherent split-collective control and
Steve's graph-owned-communication result. Attribute the cost of every
collective boundary before changing math kernels.

### 6. Collective paths

Treat these as different implementations, not one P2P switch:

- oneCCL host-staged P2P-off;
- oneCCL direct P2P;
- Steve's custom-op collective handoff;
- our Level Zero IPC push all-reduce;
- graph-owned in-place versus cloned output.

Measure small-message latency, bulk bandwidth, graph boundaries, host wall per
step, device time, rank imbalance, and end-to-end decode. The working hypothesis
is that graph integration and boundary count dominate after the raw collective
falls below about 40 us.

## Portability Matrix

| Target | TP=1 | TP=2 | PP=2 | DP=2 | Questions |
| --- | --- | --- | --- | --- | --- |
| Qwen3.6 35B-A3B Quark W8A8 | capacity diagnostic | exact Steve control | stage control | replica control | Which June wins reproduce exactly? |
| Ornith 1.5 35B-A3B W8A8 | target speed | target plus collectives | current coherent baseline | product concurrency | Which changes survive different calibration and MTP? |
| Qwen3.8 27B W8A8 | full serve | communication cost | stage balance | two replicas | Which wins are dense-model generic? |
| Additional 27B CT W8A8 | full serve | communication control | optional | two replicas | Does artifact format change routing? |

For each cell: c1, c2/c4 aggregate, TTFT, TPOT, prefill, prefix hit, capacity,
graph count, collective count, determinism, mixed-load coherence, and health.
MTP is a second axis and is added only after target-only correctness.

## SGLang Transfer

Do not transplant vLLM monkeypatches. Identify the equivalent SGLang seams:

- quantization registry and linear method selection;
- custom-op/fake registration for torch compile;
- model forward and GDN implementation;
- graph capture and replay manager;
- tensor-parallel collective abstraction;
- scheduler and prefix-cache state handling.

Keep native op source shared under `kernels/`, then compile a separate SGLang
binary against its torch ABI and write an SGLang-native adapter. Prove numeric
equivalence before performance tests.

## Ordered Campaign

1. Freeze and hash the currently inspected Steve artifacts.
2. DONE: make the exact Qwen P2P-off target-only graph control coherent.
3. DONE: reproduce Steve's benchmark request exactly and add semantic canaries.
4. Attribute graph boundaries and the 5.0x decode-step gap.
5. Attribute INT8 dense, MoE, GDN, sampler, and scheduler one factor at a time.
6. Integrate our push collective into the winning graph contract.
7. Perform one guarded kernel-7.1 direct-P2P transaction.
8. Rebuild every native SO from local source and remove external mounts.
9. Implement and gate the SGLang-native route.
10. Execute the model and parallelism portability matrix.
11. Transfer only proven wins to Ornith, then add MTP and prefix caching.
12. Run Pi plus local Terminal-Bench and promote only a coherent winner.

## Definition Of Done

- Fresh-cache build and serve are scripted and deterministic.
- Runtime imports and SO hashes match a committed manifest.
- No dependency on Steve's checkout or prebuilt binary remains.
- Adapted source carries clear attribution and compatible license notices.
- Every selected feature has mechanism, correctness, and A-B-B-A evidence.
- vLLM and SGLang use separate ABI-correct builds.
- TP=1, TP=2, PP=2, and DP=2 conclusions are explicit.
- The final shelf path passes concurrency, cache, long-context, eval, and health
  gates and remains coherent under Pi/Terminal-Bench load.

## Related Evidence

- `docs/20260825_steve_qwen36_w8a8_forensics.md`
- `docs/20260823_tp2_inference_profile.md`
- `docs/20260823_tp2_optimization_campaign.md`
- `P2P_GPU.md`
- `MTP_TODO.md`
