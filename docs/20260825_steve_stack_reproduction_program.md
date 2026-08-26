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
  Older repetitive-prompt TP2 artifacts reached 91.35-91.59 tok/s under
  weaker gates.
- The operator-supplied LocalMaxxing result `cmq9ifq0500b0r8012f27j1xl` is
  TP4 at 99.77 tok/s. It is not the TP2 target. Steve's current-program TP4
  strict/TP2 smoke values are 93.55/85.87 tok/s, and his older values are
  99.77/about 91.35 tok/s. Four cards add about 9 percent, not the missing
  local 5x.
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
- An early adapter incorrectly added a no-spec uniform PIECEWISE decode key.
  The coherent P2P-off control happened to use maxlen 8192 and smaller capture
  sizes, so it did not expose the scheduling error. The exact 32K control used
  vLLM's default sizes through 48 with maxseqs 24 and failed while capturing
  sizes above 24. June ordinary decode used the relaxed general PIECEWISE key;
  the extra local key is removed and an off-device contract now preserves that
  behavior for every default size.
- The matched P2P-off, split-collective control produced 17.06 corrected output
  tok/s: 498 actual prompt tokens, 512 output tokens, 624.29 ms client TTFT,
  and 30.010 s decode time. Steve's accepted run used the same request shape
  and decoded in 5.963 s. A later off-device schema census and the preserved
  server log proved this control JIT-ran Triton's routed-MoE kernel because the
  installed August `_xpu_C` does not register the grouped W8A8 operator. The
  5.0x gap therefore combines a routed-MoE dispatch mismatch with graph and
  runtime differences; it was not an all-native-math control.
- In Steve's accepted configuration, direct-P2P oneCCL collectives remained in
  the forced graph. P2P-off host-staged oneCCL requires explicit per-layer
  graph splits locally. The capturable Level Zero push all-reduce is correct in
  its standalone graph harness but cannot import rank 0's IPC allocation from
  rank 1 inside the loaded vLLM process.
- The pinned image contains the current preserved `_xpu_C` and GDN binaries,
  but this does not establish June-record identity:
  his accepted controls used a restored 67 MB `_xpu_C`, while the surviving
  extension is 116706992 bytes. The first guarded direct-P2P model run failed
  on the first compiled all-reduce, but source comparison found that the
  August vLLM tree had removed Steve's required inner clone. That clone is
  restored locally. The raw oneCCL oracle subsequently passed 256/256 direct
  and 512/512 XPUGraph checks with exact loaded hashes under unset/default IPC
  identity.
- The clone-correct vLLM integration oracle then exported only the local op and
  passed eager/compiled exact shapes plus 256 single-op XPUGraph replays on
  both ranks without mismatch, mutation, or alias. Its synthetic unrolled
  81-profile-collective arm was not model-representative: Inductor retained 81
  independent 32 MiB outputs, emitted five 16-output pointwise fan-out kernels,
  and rank 1 DEVICE_LOST while autotuning the second fan-out before the
  81-collective graph stage. Correct that stress arm before reuse; the exact
  interleaved model is the decisive gate.
- The complete locally rebuilt June kernel package passes an off-device import
  and dispatch gate for activation quantization, dense W8A8, grouped W8A8,
  SiLU, remap, and gather. The exact model launcher now mounts this package as
  one unit and rejects unexpected SO hashes, module origins, or missing
  schemas.
- The first coherent exact-package endpoint captured all 9/9 PIECEWISE graphs,
  passed the exact p498/o512 metric and both 16/16 canaries, and tore down
  healthy at 47.5448 corrected tok/s. This is 2.788x the earlier control but
  only 55.37% of Steve. The route gate correctly rejected it: the pinned
  image's Quark source SHA256 `7e4c13d2...` unconditionally calls generic
  Triton `fused_experts`. Registering the June grouped operator did not select
  it.
- Per-rank instrumentation localized the exact-model TP=2 boundary. With
  P2P off, both ranks completed prior work and the input clone, entered the
  same first `[8192,2048]` BF16 all-reduce 245.077 ms apart, and neither
  returned; zero MoE calls had begun. This is an in-oneCCL deadlock after
  matched entry, not a missing rank, dense kernel, clone allocation, or MoE
  failure. With direct P2P and synchronous instrumentation, both ranks
  completed all 81 profile all-reduces and all 40 native MoE calls. The only
  failure was the diagnostic wait itself at command-graph recording, where
  PyTorch correctly rejects synchronizing a recording queue.
- A shape-bounded clone fence is the minimal working repair. For direct-P2P
  profile tensors with at least 8192 rows, synchronize only after the input
  clone and immediately before oneCCL. No pre-collective rank fence and no
  post-collective wait are required. The fence is inactive for graph capture
  and decode shapes. This exact native-MoE route captured all 9/9 graphs,
  served coherently, passed both 16/16 canaries, and tore down with both health
  layers green at 45.3649 corrected tok/s. It is 52.83% of Steve and 4.58%
  slower than the generic-Triton-MoE exact-package control. The compiled TP=2
  collective boundary is therefore closed; the remaining 40.5042 tok/s gap is
  elsewhere in graph/runtime/kernel behavior, and the current untuned native
  grouped-MoE route is not itself a speed win.
- The closest surviving June vLLM source at `e190923b` now passes a
  12-component source-origin/hash contract and the full exact endpoint. The
  public June source exposed one concrete ABI seam: its MoE path passes
  persistent `scratch`, while the reconstructed June-9 Python interface does
  not accept it. Mounting the recovered scratch-aware interface from kernel
  commit `2dd55f38` with the same rebuilt binaries closed the seam. The run
  completed 81/81 clone fences per rank, captured 9/9 graphs, selected native
  dense and MoE INT8, passed both 16/16 canaries, and passed both post-health
  layers at 48.5315 tok/s. This is +6.98 percent over the August-adapter native
  control and leaves a 1.7693x gap. Full June source is not the missing lever.
- Steve's directly comparable fresh-54/restored-67 `_xpu_C` results were
  87.2888 and 89.9613 tok/s, about a 3 percent file-class difference. This does
  not compare the June-9 reconstruction with exact recovered June-16
  checkpoint `122b698b`, which adds native quantization output-buffer operators. File size
  is still not an identity; source/operator A/B is the required test.
- The GPU model matches, but the host does not: Steve's June Qwen35 system was
  an EPYC 9015 PCIe 5 host, while this system is a Threadripper 1950X with the
  cards under separate PCI domains on a PCIe Gen3-era platform. Steve also
  proved `pidfd` on a later Threadripper PRO 5955WX two-card oracle. Host
  topology is therefore a live direct-P2P compatibility variable, not the
  explanation for the graph/no-graph 5x.
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

The exact-control capture failure was caused by a locally added no-spec uniform
decode descriptor, not a missing descriptor in the later snapshot. June
ordinary decode reused the relaxed general PIECEWISE key. The local addition
made capture sizes 32, 40, and 48 impossible to schedule with maxseqs 24. It is
removed, and a no-device contract guards the June behavior without narrowing
the default capture-size list. The next run reached endpoint health but exposed
an August capture filter under June's eager-prefill variable. That filter
removed the relaxed general graphs ordinary decode reuses, so first inference
selected an uncaptured graph and returned HTTP 500. The adapter now retains all
nine general captures while restoring the variable before runtime dispatch. A
v2 no-device contract guards both halves. Direct P2P plus a profile-only clone
completion fence now crosses the compiled collective boundary and captures all
nine graphs. The true-June source control raises the stable endpoint to
48.5315 tok/s. Instrumented replay observes the same 41 graph pieces as Steve,
while synchronized rank-0 model-forward is 22.6748 ms versus 5.6946 ms. The
active target is therefore native/runtime execution inside the matched
topology. Exact checkpoint `122b698b` native siblings then measured 50.3706
tok/s versus 48.5315 tok/s for the matched June-9 endpoint (+3.79 percent),
with both canaries and post-health green. Native scratch output helps but is
not the missing 1.7x. The next controlled toggle is synchronized timing on the
same native checkpoint, followed by integrated graph-collective timing.

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
4. Attribute graph boundaries and the remaining 1.7693x decode-step gap.
5. Attribute INT8 dense, MoE, GDN, sampler, and scheduler one factor at a time.
6. DONE: integrate and validate our push collective graph contract; loaded
   vLLM IPC import remains asymmetric and is not the exact-Steve route.
7. DONE: the guarded kernel-7.1 direct-plus-XPUGraph oneCCL oracle passed 256
   direct and 512 graph iterations per rank with zero mismatch. This clears the
   hardware, topology, current oneCCL binary, direct P2P, and raw graph replay.
8. PARTIAL: the locally owned clone-correct GroupCoordinator oracle passed
   exact runtime identities, export, eager/compiled real shapes, and single-op
   XPUGraph replay. Its independent 81-way profile fan-out triggered an
   artificial 16-output Triton autotune DEVICE_LOST, so the 81-op graph stage
   did not run. Replace this arm with a sequential low-live-buffer chain before
   treating the oracle as a complete volume gate.
9. DONE for the collective boundary: the native Quark route plus direct P2P
   and clone-only profile fence captured all 9/9 graphs, passed the exact
   metric and both 16/16 canaries, and tore down healthy at 45.3649 tok/s. The
   strict evidence gate passed, including native XPU MoE selection and absence
   of request-time generic MoE JIT. Continue the source-overlay bisect and
   per-step profile from this control; do not spend another transaction on the
   now-cleared compiled-collective boundary unless a different model shape
   needs a new fence threshold.
10. DONE: the locally owned minimal June 9 source reconstruction built
   a 55,523,648-byte B70-AOT `_xpu_C`, both Xe2 siblings, and a complete
   pinned-image runtime package with all required XPU dispatch registrations.
   This matches Steve's 54 MB fresh-build class, not the unavailable accepted
   67 MB binary. The June/August quant and dense A-B-B-A harness is ready; the
   pinned August package cannot be the grouped arm because that schema is
   absent. It passed full-model dispatch/capture/coherence gates, but it lacks
   the later quantization output-buffer schemas.
11. DONE: exact native checkpoint `122b698b` passes schema/import, endpoint,
   coherence, graph-capture, and teardown-health gates at 50.3706 tok/s, a
   matched +3.79 percent over June-9.
12. IN PROGRESS: repeat synchronized timing on `122b698b`, then isolate
   integrated graph collective/runtime cost.
13. Implement and gate the SGLang-native route.
14. Execute the model and parallelism portability matrix.
15. Transfer only proven wins to Ornith and dense 27B, then add MTP and prefix
   caching.
16. Run Pi plus local Terminal-Bench and promote only a coherent winner.

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
- `vllm/w8a8/qwen36_oneccl_graph_oracle.py`
- `vllm/w8a8/run_qwen36_oneccl_graph_oracle.sh`
- `P2P_GPU.md`
- `MTP_TODO.md`
