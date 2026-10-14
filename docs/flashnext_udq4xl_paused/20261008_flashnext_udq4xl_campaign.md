# Flash Next UD Q4 K XL dual B70 research plan

## Decision and authorized scope

On 2026-10-08 the user selected Unsloth UD-Q4_K_XL for Qwen3.8-Flash-Next,
authorized downtime of the current GPU service, and authorized work on any
backend, custom kernels, profiling and optimization. This campaign expands the
older backend/model scope in AGENTS.md deliberately. It does not restore code
or ABI-specific binaries from the cleanup archive.

The objective is a correct, reproducible Flash-Next service using two 32 GB
B70s, system RAM and SSD-backed storage where needed. Start from current
GGUF-native backends. Compare newest llama.cpp SYCL and Strata SYCL before
committing to substantial engine work. Preserve every target expert and the
selected checkpoint's quantization. Do not silently substitute Coder, a pruned
model, another publisher, FP8, NVFP4, or a lower-bit checkpoint.

No performance, capacity, correctness or shelf qualification is established
by this plan. The first release is text-only research at one request at a time;
vision, MTP, longer context and concurrency each require separate qualification.
The publisher vision and MTP artifacts are retained for those later stages.

## Model and host identity

- Publisher: `unsloth/Qwen3.8-Flash-Next-GGUF`.
- Revision: `766911a6b7369840a91dbcd95f9f997acaab6cd6`.
- Target: four `UD-Q4_K_XL` shards, 111,334,654,784 bytes (103.69 GiB).
- Additional artifacts: Unsloth Q4_K_M MTP sidecar, BF16 vision projector,
  publisher README and MTP README. Complete selected intake: 115,028,467,617 bytes.
- Exact files, sizes and hashes: [model lock](../strata/flash-next/model-lock.json).
- Target shard LFS hashes match Strata's documented `38bb39e` revision.
- Storage: `models/files/qwen3.8-flash-next/unsloth-ud-q4-k-xl/`.
- Client-facing identity remains `hotschmoe-dd`; research alias is
  `qwen3.8-flash-next-Unsloth-UD-Q4_K_XL`, with engine/config in result paths.
- Host: Threadripper 1950X, 16 cores/32 threads, AVX2; approximately 121 GiB
  usable RAM and 104 GiB available at initial inspection; two 32 GB B70s.
- Kernel baseline: `7.1.0-070100-generic`. Host Compute Runtime
  `26.22.38646.4-0`, Level Zero loader `1.28.2-2`.
- Existing service image: `sha256:d55637b3353eaf470677627dc1627c3dda3ec6a6abb0298451aed34b73937067`;
  PyTorch `2.13.0+xpu`, vLLM `0.27.2rc1.dev77+gac7509e2b.xpu`, Triton XPU
  `3.7.2`, oneCCL `2022.0.0`. This is a rollback/health identity, not a new backend.
- Raw initial inventory and all run output:
  `/mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/`.

UD-Q4_K_XL is mixed precision: Q4_K/Q5_K gate/up experts, Q5_1/Q8_0 down
experts, and higher-precision exceptions. Strata documents about 77 GB of
expert storage and a 28.8 GB IQ4_NL PLE table. These are sizing inputs,
not runtime peak measurements. Parse actual GGUF tensor inventory before
selecting a split. GPU memory must also hold dense weights, KV/state,
prefill workspace, cache metadata and any MTP state. Host estimates must count
pinned copies, file cache, loading transients and driver-accounted memory.

## Research sources and findings

### Strata

Source: [Niko1221/Strata](https://github.com/Niko1221/Strata), fetched at
`fb58e0dbc8399662c0e47c76578c6e878b14f6cf`, local
`/mnt/vm_8tb/github/strata`. The `b70` remote retains the author's branches;
`intel-arc-0.1.40` is older than the selected upstream correctness fixes.

- SYCL layer split already exists. Community dual-B70 IQ3_XXS runs establish
  feasibility of that mechanism, not our Q4-plus-RAM configuration.
- The helper-GPU expert tier is stubbed; it is distinct from layer splitting.
- `sycl/src/program/generate.cpp` constructs and binds the pinned expert
  mirror only for stage 0. Stage 1 has no equivalent mirror binding. Balanced
  dual-card host misses therefore need implementation and validation.
- A proposed smaller step is an asymmetric split: keep every later-stage expert
  on GPU1, put remaining layers plus RAM misses on GPU0. Choose boundaries
  from actual allocation receipts; 32/16 is only an initial sizing hypothesis.
- Native SYCL kernels cover the chosen mixed expert formats. Prompt execution
  uses dequantization plus FP16/oneMKL; CUDA MMQ speed claims do not transfer.
- Unsloth packing with `--compat-bf16` rounds some Q8 hyper-connection weights.
  The documented AMD comparison reports 6-9% higher perplexity versus the same
  GGUF in llama.cpp. The Q8 compute path is explicitly missing in
  `sycl/src/kernels/cuda/fused_gr.dp.cpp`. Preserve native Q8 arithmetic or
  prove a numerically faithful alternative before promoting this import.
- Preserve the large-allocation alias checks, ring-wait fixes, stage accounting
  and latest opt-in stage-pinning policy. The documented B70 RAM route requires
  `STRATA_VERIFY_NO_HOST=1` and complete mirror coverage. An environment switch
  is not a substitute for proving pointer residency and queue completion.
- Generate reviewed explicit settings: the Intel setup wrapper does not fully
  reconcile Unsloth RAM-budget settings with the SYCL path.

References: [Intel](https://github.com/Niko1221/Strata/blob/fb58e0dbc8399662c0e47c76578c6e878b14f6cf/docs/INTEL.md),
[Unsloth formats and fidelity](https://github.com/Niko1221/Strata/blob/fb58e0dbc8399662c0e47c76578c6e878b14f6cf/docs/UNSLOTH_Q4.md).

### Newest llama.cpp

Fresh source at `/mnt/vm_8tb/github/llama.cpp-flashnext`, upstream
`de7fa0a3c6a2e1b4cd9f22eb8d6bf5b12dbdb63b` on 2026-10-08.
This is a new deliberate source intake, not an archived runtime restoration.

- Upstream qwen4exp model, PLE lazy loading, MTP architecture and SYCL expert
  kernels exist. Do not blindly apply Sergio's historical patches again.
- New `--cpu-moe --moe-cache-mib N` implements a GPU cache for host experts,
  assigned across GPUs by layer placement and split proportions. It rejects
  tensor-parallel split mode; start with `--split-mode layer`.
- This can make llama.cpp a candidate optimized engine as well as an
  independent GGUF fidelity reference. Verify SYCL execution of this path on
  the actual build before calling it supported locally.
- Source review confirms SYCL supports its I32 slot maps, GET_ROWS, uploads
  and synchronization. Expert-ID reads synchronize each MoE layer; large-batch
  copies can use a blocking fallback. Profile these costs instead of assuming
  asynchronous overlap. Mixed quant layouts have separate cache groups.
- Start with conservative measured cache capacity, lazy mmap PLE, FP16 KV,
  no speculation, no prefix reuse, no optional graphs and one request.

References: [cache implementation](https://github.com/ggml-org/llama.cpp/blob/de7fa0a3c6a2e1b4cd9f22eb8d6bf5b12dbdb63b/src/llama-moe-cache.cpp),
[qwen4exp](https://github.com/ggml-org/llama.cpp/blob/de7fa0a3c6a2e1b4cd9f22eb8d6bf5b12dbdb63b/src/models/qwen4exp.cpp).

### Sergio

Cookbook revision `1a76136ea1b3f96bba3b20e42041a6fcc95f56fd` demonstrates
dual-B70 llama.cpp layer splitting, CPU PLE and selective CPU expert blocks.
The measured AtomicChat and Signal artifacts are not our chosen Unsloth model.
Its larger Q4 placement shows why real padded allocation and headroom matter.

The raw MTP evidence says the published decode medians used pre-fix expert
indexing. A corrected smoke exists, but the original medians are not a new
correctness-qualified target. Rebuild and remeasure any adopted source delta.
Its external draft artifact also must not silently replace the selected
Unsloth sidecar. MTP format compatibility must be established separately.

References: [recipe](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook/blob/1a76136ea1b3f96bba3b20e42041a6fcc95f56fd/docs/qwen38-flash-next/QWEN38-FLASH-NEXT-LLAMACPP.md),
[raw MTP caveat](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook/blob/1a76136ea1b3f96bba3b20e42041a6fcc95f56fd/results/qwen38-flash-next-mtp-kernel-v1/summary.json),
[larger placement](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook/blob/1a76136ea1b3f96bba3b20e42041a6fcc95f56fd/docs/qwen38-flash-next/AP-Q4KXL-DUAL-PLACEMENT.md).

### Steve

Lab revision `6c5d2eafcb148f4f9d56279cedfbb847d992007e` contains four-card
FP8 exact-output work, not a qualified dual-card Unsloth Q4 route. Transfer
its placement/address-table ideas, bounded loaders, PLE caching, primitive
tests, GDN/MTP state checks and evidence discipline, not its speed claims.

October 8 attempt 7 faulted all four cards during the initial 64-token dummy
forward, before readiness and any user generation. Post-fault probes passed,
but a fault latch still blocks launches at that revision. Earlier CURRENT.md
entries describing the attempt as running are superseded by the fault receipts.
Host-buffer residency/addressing is a hypothesis, not an established cause.
The driver-shadow explanation was withdrawn: GPUActive remained about 73 GB
with deferred backing disabled; pinned rows dominated. Do not copy driver
settings on the assumption that they solve memory fit.

References: [closeout](https://github.com/steveseguin/b70-optimization-lab/blob/6c5d2eafcb148f4f9d56279cedfbb847d992007e/results/qwen38-flash-next-fp8-b70/CLOSEOUT-20260913.md),
[fault analysis](https://github.com/steveseguin/b70-optimization-lab/blob/6c5d2eafcb148f4f9d56279cedfbb847d992007e/experiments/qwen38-flash-next-fp8-b70/notes/2026-10-08-attempt7-gpu-fault-analysis.md).

### vLLM and SGLang current feasibility

Checked on 2026-10-08, not assumed from historical installed images:

| Backend | Release and source | Selected artifact on XPU |
| --- | --- | --- |
| vLLM | v0.31.0; `ab905a885dfbfc60a2c02286cc9c608c93884de3` | qwen4exp explicitly rejects XPU; separate GGUF plugin documents CUDA/ROCm |
| SGLang | v0.5.21; `71584b62c0f6c9f1a8dbad1b0a9191f775c0bcd8` | GGUF execution kernels lack XPU support; Flash-Next GGUF mapping/offload combination unqualified |

Do not start a full-model trial by merely bypassing a platform guard. Either
backend remains allowed if later profiling justifies the loader/kernel port.
Update source/image identity at the start of a new experimental generation,
then freeze it across comparisons. No unrelated host driver upgrade is needed
to satisfy the request for current backends.

References: [vLLM guard](https://github.com/vllm-project/vllm/blob/ab905a885dfbfc60a2c02286cc9c608c93884de3/vllm/models/qwen4_exp/__init__.py),
[SGLang GGUF](https://github.com/sgl-project/sglang/blob/71584b62c0f6c9f1a8dbad1b0a9191f775c0bcd8/python/sglang/srt/layers/quantization/gguf.py).

## Research and correctness contract

Follow [neural.download](https://neural.download/) and its
[optimization guide](https://github.com/steveseguin/b70-optimization-lab/blob/6c5d2eafcb148f4f9d56279cedfbb847d992007e/docs/model-optimization-guide.md),
using this repository's lease, host and recovery rules.

1. Freeze artifact, tokenizer/template, full configuration, source, compiler,
   runtime, image and hardware identities. Hash every conversion/patch and
   disclose whether it changes decoded weights or arithmetic.
2. Maintain an independent artifact/math reference: same GGUF with no MTP,
   CPU dequantization checks and trusted primitive results. Cross-engine output
   need not be bit-identical; preregister numeric tolerances, logit errors,
   top-token margins, teacher-forced perplexity and task checks before scoring.
   No arbitrary post-hoc tolerance widening or unexplained corruption passes.
3. After validating an engine, freeze its deterministic optimization authority:
   prompt token IDs, outputs, logits and recurrent/KV state fixtures. Require
   exact tokens and repeats for changes claiming unchanged behavior. A
   changed-arithmetic oracle is a new generation and cannot erase a regression
   against the old oracle. Steve's FP8 hashes are not this Q4 model's answers.
4. Before server bisection, census relevant quantized GEMV/GEMM, GDN, attention,
   routing and hyper-connection shapes on each card. Check mixed types, large
   offsets, noncontiguous expert IDs, padding and host/device ownership.
5. Use a frozen varied suite: prose, executable code, arithmetic, instructions,
   structured JSON, multilingual text, long-context retrieval and conversation.
   Test repeated deterministic requests, fresh starts, cancellation and then
   concurrent requests against their own solo references.
6. Performance attempts use cold requests, no prefix/KV/response reuse and no
   history-based draft advantage. Resident weights and compiled kernels are
   allowed. Use a 512-output-token cap; preserve short natural completions.
   Primary decode metric is 99 divided by time between tokens 1 and 100,
   aggregated within each class then across class medians. Stream chunks are
   not token timestamps: instrument engine timings or label another metric.
   Missing 100-token measurements stay unavailable; do not extrapolate.
7. Also record TTFT, actual prompt/output counts, prefill with defined timing,
   full-output decode and wall throughput, p10/mean, peak RAM/VRAM, page faults,
   host/device traffic and MTP proposal/acceptance counts where applicable.
8. Establish baseline noise with three independent launches. Finalist trials
   use at least three balanced paired A/B blocks with reversed order. Freeze
   workload, clocks/settings, context, KV, cache state and host conditions.
   Choose a promotion threshold from measured baseline dispersion before
   examining a candidate's score; retain every pair, including negatives.
9. Performance runs exclude profiling instrumentation. Use profiles to identify
   a bottleneck, make one bounded change, rerun primitive/correctness gates,
   then measure the release path. No speed claim from an unqualified screen.
10. Introduce MTP only after target correctness: unchanged verifier, greedy
    no-MTP parity, rejection rollback, GDN/KV advancement and deterministic
    repeats. Quantized KV is a separately labeled quality choice, not free speed.
11. Promotion needs complete identity, coherence, determinism, fresh-start,
    matched performance, applicable concurrency, teardown and post-health
    evidence. Store CONFIG -> COMMAND -> RESULT -> VERDICT for every experiment.

## Execution stages and exit gates

| Stage | Work | Exit gate |
| --- | --- | --- |
| F00 intake | Download/hash files; parse tensor/PLE/MTP metadata; inventory live stack; fetch current sources | Exact model/source lock; explicit memory budget; reproducible new build recipe |
| F01 reference | Fresh llama.cpp SYCL, no-MTP FP16-KV reference; CPU and per-card primitive checks | No unsupported tensors; explained numerics; deterministic short outputs; clean teardown/health |
| F02 placement | Dual layer split, lazy CPU PLE, fixed CPU expert placement versus new GPU expert cache | Measured per-card/host peaks; all experts reachable; bounded loading; no faults/aliasing; exactness against authority |
| F03 Strata | Audit/repair native Q8 HC path; explicit asymmetric split; then per-stage mirrors if worthwhile | Same-artifact fidelity gates and deterministic engine authority; healthy two-card lifecycle |
| F04 profile | Attribute decode/prefill to kernels, launch/waits, expert misses, PCIe, PLE and CPU work | Measured bottleneck and preregistered single-change hypothesis |
| F05 optimize | Cache placement, quant kernels, batched expert work, copy overlap, PLE cache, fusion, graphs as justified | Primitive and exact-output gates; repeatable paired improvement outside noise |
| F06 speculation | Pinned publisher-compatible MTP, depth sweep after rollback tests | Target-verified acceptance; no-MTP parity and measured wall benefit |
| F07 service | Context ladder 8K/32K/64K/128K then higher only if supported; C1 then C2/C4; cancel/reuse | Frozen retrieval/coherence cases at exact lengths; solo/concurrent parity; clean repeated lifecycle |
| F08 package | Rebuildable recipe, source patches, fixtures, evidence index, qualified shelf entry | Independent replay plus all applicable service gates |

The stages are evidence gates, not a commitment to optimize every engine.
If current llama.cpp meets the goal, Strata remains a measured comparison;
if its expert cache is bottlenecked, port only the source mechanisms supported
by profiles. Custom SYCL/Triton/Level Zero kernels are in scope. Prioritize
fidelity and two-card memory access before additional speculative features.

## GPU lifecycle and stack changes

The parent owns GPU operations. Agents may perform source audits and CPU work;
each actual GPU operation, including runtime-compiling probes, is under
`bin/gpu-run`. Pair single-card leases with explicit device selection.

Before downtime, capture the actual installed service, container/image and
model identities. The live native5802 service already owns both leases and
has a supervised stop marker plus post-health; use that lifecycle instead of
force-removing its container. Verify its exit/post-health, then acquire the
pair lease for new work. Preserve monitoring and unrelated containers.

Record the host/image kernel, UMD, Level Zero, oneCCL, compiler, PyTorch and
backend before changes. Use a new pinned build image; do not overwrite the
rollback image. Change one stack layer at a time and rebuild ABI-specific
extensions from source. No power/clock changes as an incidental cleanup step.

Require strict per-card health and compiled two-rank P2P0 health before
serving and after risky work. Fail closed on inconclusive health. Layer split
is not vLLM tensor parallel; do not introduce P2P1 TP experiments casually.
Watch explicit GPU fault signatures during loading and generation; stop the
experiment on a fault, record it, and use the repository recovery ladder.
No unattended crash/restart loop. Each launch has a bounded deadline and
owned cleanup. Preserve failed receipts and never overwrite a prior result.

Rollback target is the saved native5802 service/image/config, not a rebuilt
approximation. Service restoration must use its existing owner/readiness and
identity checks; GPU research authorization permits downtime but does not
make an unqualified Flash-Next endpoint the daily driver.

## Initial commands and artifact locations

The compiler base is the official image
`intel/oneapi:2026.1.0-devel-ubuntu24.04@sha256:e9db518398753434ee5aab9740a25f1d3134396a30be1569cfad8f8b0d90740c`.
Its pull is recorded separately from the derived build. The source build recipe
is [build.sh](../llamacpp/flash-next/build.sh), with a
[Dockerfile](../llamacpp/flash-next/Dockerfile) and
[unexecuted initial settings](../llamacpp/flash-next/initial-recipe.json).
It self-acquires both leases and compiles in a container without GPU device
access. Initial build settings are SYCL ON, oneDNN ON, and SYCL_F16 OFF
(the upstream compute default), distinct from explicitly FP16 KV storage.
Image ID, package versions, compiler, CMake settings and binary hashes are
recorded. No old native libraries enter the new image.

The download runs as user unit `b70-flashnext-model-intake.service`, with a
12 GiB memory limit and two file workers. It verifies every selected file's
publisher hash before reporting success. It is CPU/network-only.

```sh
python3 strata/flash-next/fetch_model.py \
  --receipt /mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/model-intake.json
systemctl --user status b70-flashnext-model-intake.service
```

Model metadata and reproducible tooling live in `strata/flash-next/`;
backend-independent curated findings live here; subsequent llama.cpp
experiments belong under `llamacpp/flash-next/` and Strata experiments under
`strata/flash-next/`. Large outputs and builds stay under `/mnt/vm_8tb/b70/`.
Append JOURNAL.md after each coherent checkpoint; do not edit old evidence.

The header-only inspector is `strata/flash-next/inspect_gguf.py`. It reports
per-tensor format/shape/storage, per-layer expert/dense/PLE bytes, split
completeness and tokenizer/template hashes without scanning weight payloads.
It references the separate full-file hash receipt; it does not replace it.
The first target shard is metadata-only, so its completion does not establish
the backbone inventory. Missing shards produce a partial report and exit 2.

The build runs as user unit `b70-flashnext-llamacpp-build.service`, with logs at
`/mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/llamacpp-build.log`.
It acquires both GPU leases and builds the server, benchmark, perplexity tool
and backend/quantization correctness probes. No serving job is chained to it:
verified model intake, primitive tests and explicit runtime health must pass first.

## Status

- Research and source audit complete for the initial decision.
- Target and sidecar identities pinned; supervised download started.
- Current live stack inventory saved before any stack mutation.
- Fresh Strata and llama.cpp upstream source checkouts available.
- Previous service stopped through its owned lifecycle: service/backend exit 0,
  both-card post-health healthy and compiled P2P0 collective post-health healthy
  for 10 iterations. Workers completed XPU shutdown; the executor subsequently
  sent SIGTERM after its worker grace period. Preserve this detail rather than
  claiming a signal-free exit. The service is inactive and both leases released.
- Official compiler image pull and supervised derived build started.
- Model parsing, new-backend GPU qualification and F01 remain pending.
- No candidate has passed F01 or been promoted.
