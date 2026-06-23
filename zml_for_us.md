# ZML For Our Dual B70 Work

Date: 2026-06-23

Repos checked:

- B70 repo: `/home/hotschmoe/github/b70_ai_things`
- ZML repo: `/home/hotschmoe/github/zml`
- ZML upstream head reviewed: `51314672` (`zml/attention: tune triton kernel for oneAPI performance (#588)`)
- Upstream PRs/branches reviewed through GitHub API on 2026-06-23.

## Verdict

ZML is worth testing directly on the dual Arc Pro B70s, but it should be treated
as an experimental compiler/TP path today, not as the replacement for our vLLM
serving stack.

The important split is:

- Use ZML directly to validate oneAPI PJRT multi-device enumeration, Shardy/SPMD
  tensor parallelism, direct sharded loading, and compiler-visible collectives.
- Keep vLLM as the primary production/research serving path for compressed-tensors
  W8A8/W4A8 and B70 quantized serving, because ZML does not currently provide our
  compressed-tensors W8A8/W4A8 loader/kernel stack.
- Mine ZML aggressively for design: tensor sharding annotations, placement of
  all-reduces, delayed replication, direct per-shard loading, and oneAPI paged
  attention kernel choices.

There is a real contradiction in upstream ZML right now. The oneAPI README still
says oneAPI is early, sharding is not supported, and only one GPU is supported
(`platforms/oneapi/README.md:6-20`). The code on master, however, has generic
multi-device SPMD plumbing for oneAPI: power-of-two device validation, addressable
device discovery, Shardy by default, SPMD compile options, device assignment, and
per-device buffers. The most relevant open upstream PR is #592, which explicitly
sets oneAPI PJRT create options and says it updates the PJRT plugin with a
collective fix. That PR is the first thing to test on the B70 pair.

## What ZML Has Today

### oneAPI Platform

ZML enables oneAPI at build time through `ZML_RUNTIME_ONEAPI`, checks for
`/dev/dri/renderD*`, sets oneCCL defaults, and loads `libpjrt_oneapi.so` from
Bazel runfiles (`platforms/oneapi/oneapi.zig:12-63`).

Master's oneAPI create options are nearly empty (`zml/platform.zig:629-650`).
The named-value creation path only applies XLA GPU allocator options to CUDA and
ROCm, not oneAPI (`zml/platform.zig:707-719`). PR #592 changes that to pass the
same XLA GPU create options to oneAPI, which is why the PR matters for B70
collectives and OOM behavior.

ZML `Platform.auto` includes oneAPI before CPU fallback (`zml/platform.zig:344-355`,
checked locally), but the oneAPI backend itself requires a Linux Intel GPU and
currently depends on external device selection in the README.

### Sharding And Compilation

The sharding system is already target-agnostic enough to include oneAPI:

- The default partitioner for oneAPI is Shardy (`zml/Sharding.zig:44-52`).
- Partitioning computes partitions, replicas, and device assignment
  (`zml/Sharding.zig:77-105`, `zml/Sharding.zig:1577-1592`).
- ZML always keeps replicated sharding as a fallback (`zml/module.zig:126-135`).
- It emits `mhlo.num_partitions` and `mhlo.num_replicas`
  (`zml/module.zig:231-242`).
- It emits Shardy mesh ops (`zml/module.zig:273-292`).
- It sets XLA compile options for SPMD, Shardy/GSPMD choice, and explicit device
  assignment (`zml/module.zig:574-625`).
- The oneAPI compile override disables autotune, command buffers, and cublasLt
  flags (`zml/module.zig:648-651`).

The weak spot is physical topology. `PhysicalMesh.auto` maps oneAPI to the CPU
mesh builder (`zml/Sharding.zig:900-907`), which creates a simple bus/tree mesh
(`zml/Sharding.zig:924-935`). That is reasonable for two PCIe B70s as a first
model, but it is not a B70-aware topology. For TP=2, the bigger question is
whether oneAPI PJRT plus CCL can execute the collectives correctly and keep them
inside compiled graphs.

### Dense LLM Tensor Parallel Layout

ZML's dense Llama/Qwen layout matches the usual correct TP split:

- Q/K/V and MLP gate/up are output-channel sharded.
- Attention O projection and MLP down projection are input-channel sharded.
- Residual-visible hidden state is replicated after the row-parallel projection.
- KV cache heads are sharded on the model axis.

Local references:

- Llama MLP: `examples/llm/models/llama/model.zig:420-422`
- Llama attention projections: `examples/llm/models/llama/model.zig:460-463`
- Llama hidden replication before Q/K/V: `examples/llm/models/llama/model.zig:508-510`
- Llama O projection replication after row-parallel output:
  `examples/llm/models/llama/model.zig:557-559`
- Llama KV cache head sharding: `examples/llm/models/llama/model.zig:570-582`
- Qwen3.5 MLP and attention: `examples/llm/models/qwen3_5/model.zig:431-498`
- Common model/expert mesh axes: `examples/llm/models/common.zig:45-70`

This is directly useful for our vLLM audit: verify that XPU TP does not
all-reduce after column-parallel QKV/gate/up, and that it performs exactly one
reduce/replication after O/down before residual-visible hidden state.

### Loading

ZML's loader is one of the most interesting parts for us. It registers tensors
with semantic axis tags and partition hints, picks an explicit sharding at load
time, and for CUDA/oneAPI streams directly into per-device buffers:

- Tensor registry and creation path: `zml/io.zig:136-145`
- oneAPI uses `DirectMemoryWriter`: `zml/io.zig:303-305`
- direct writer builds per-device shard writers and PJRT buffers:
  `zml/io.zig:811-834`
- sharding is picked from explicit axis binding, otherwise replicated:
  `zml/io.zig:1152-1156`
- host buffer upload and uninitialized buffers iterate canonical devices:
  `zml/buffer.zig:121-142`, `zml/buffer.zig:211-223`

This maps cleanly to our compressed-tensors pain point. Our W4A8 prepack audit
shows that compressed-tensors W4A8 stores 4-bit weights unpacked as int8, then
vLLM packs on load, creating a large transient that can OOM a 32 GiB B70
(`docs/kernel/16_model_prepack_audit.md:12-21`). ZML's direct per-shard loading
is the design we should move toward for vLLM compressed-tensors: stream only the
local shard into the final packed/device format when possible.

### Attention

There are two attention paths to separate:

- Generic attention auto-selects vanilla XLA SDPA on oneAPI
  (`zml/attention/attention.zig:17-33`, `zml/attention/attention.zig:129-140`).
- Paged attention auto-selects Triton on oneAPI
  (`zml/attention/paged_attention.zig:18-24`) and master has a oneAPI-specific
  unified attention kernel after #588.

The oneAPI paged-attention work is worth comparing to our vLLM `TRITON_ATTN`
path. Our current B70 Triton-XPU note says the key gate is whether the engine
process sees `torch.xpu.is_available()` before Triton selection, and that the
fallback problem is a real `TritonPlaceholder` in kernels if the gate is false
(`docs/kernel/13_triton_xpu_enable.md:14-31`, `docs/kernel/13_triton_xpu_enable.md:80-85`).
ZML is useful here as a source of Intel-specific split-K/cached-KV choices, not
as a drop-in replacement for vLLM attention today.

### MoE

ZML's MoE structure is useful, but not yet a oneAPI quantized serving answer:

- `Backend.auto` does not include oneAPI; oneAPI falls to
  `error.UnimplementedMoEBackend` (`zml/moe/moe.zig:20-46`).
- When expert weights are sharded, ZML maps global experts to local experts,
  runs local fused experts, and all-reduces output
  (`zml/moe/moe.zig:164-211`).

For our B70 path, copy the expert-axis sharding and collective structure. Do not
expect ZML to replace our vLLM Quark/compressed-tensors W8A8 MoE path yet.

## Open Upstream PRs And Branches

Open PRs checked on 2026-06-23:

- #592 `zml/onapi: set defaults for oneAPI PjRT create options`
  (`kevin/oneapi-oom`): highest priority for us. Body says it is the same as
  #579 and updates the PJRT plugin with the collective fix. Diff changes the
  oneAPI PJRT artifact from `manual-2026-06-18T20-14-00Z` to
  `manual-2026-06-23T00-20-00Z`, and changes `CreateOptions.toNamedValues` so
  oneAPI receives the XLA GPU allocator/create options (`zml/platform.zig` diff).
- #589 `zml/attention: add attention Triton kernel`
  (`corendos/triton-mha-kernel`): relevant to Intel attention kernel design, not
  specifically B70 multi-GPU.
- #586 `zml/io: rework zml.io.load`
  (`corendos/loader-toolbox`): highly relevant to our loader/prepack direction.
  The PR body says it introduces a reusable `Loader` and load-time tensor
  transformations.
- Other open PRs checked were tokenizer, ROCm allocator, xet/VFS, zml-smi,
  Bazel 9, metrics, LTX, ZLS, and troubleshooting work; none are directly B70 TP
  support.

Relevant branches found by `git ls-remote --heads origin`:

- `kevin/oneapi-oom`: PR #592; first branch to test.
- `kevin/oneapi-2026`, `kevin/oneapi-2026-SAVE`, `raphael/oneapi-2026`,
  `steeve/oneapideb`: likely oneAPI runtime/debug branches.
- `corendos/llama-sharded-tests`, `gw/sharding-post-review`: sharding test or
  review branches worth scanning if #592 works.
- `corendos/paged-attention-2-triton`,
  `corendos/paged-attention-2-flashattn-sharded`,
  `corendos/triton-mha-kernel`: attention branches to compare against vLLM
  `TRITON_ATTN`.
- `louis/moe_triton_sharding`, `louis/moe-ep`, `louis/moe-laguna`: MoE sharding
  branches, useful for expert-parallel ideas.
- `tmp/comparison_vllm`: worth a later scan for explicit vLLM comparisons.

Inference: if the founder has TP working on 2x B70s, it is likely either on top
of the #592 oneAPI PJRT artifact/options branch or a nearby unpublished/local
branch, not plain master as documented in `platforms/oneapi/README.md`.

## Direct ZML Test Plan For Our B70s

Do not touch the GPUs without the lease.

First checks:

```bash
gpu-run --status
ssh root@192.168.10.5 'ls -l /dev/dri/renderD*'
```

Build/runtime prerequisites:

- Bazelisk installed locally at `/home/hotschmoe/tools/bazelisk`.
- Local Bazelisk reports `bazel 8.7.0`.
- The cloned repo is `/home/hotschmoe/github/zml`.
- If testing on the GPU host, mirror or clone ZML under `/mnt/vm_8tb/b70/` or
  another host-visible path.

Recommended first GPU test is not an LLM. It is the sharding example under
oneAPI, first on master, then on PR #592:

```bash
gpu-run bash -lc '
  cd /mnt/vm_8tb/b70/zml &&
  export ONEAPI_DEVICE_SELECTOR=level_zero:gpu &&
  export ZE_FLAT_DEVICE_HIERARCHY=FLAT &&
  /home/hotschmoe/tools/bazelisk run //examples/sharding \
    --config=release \
    --@zml//platforms:cpu=false \
    --@zml//platforms:oneapi=true \
    -- \
    --partitioner=shardy \
    --mesh=auto
'
```

What to look for:

- PJRT sees exactly two addressable B70 devices.
- ZML creates a oneAPI platform, not CPU fallback.
- The compile uses `num_partitions=2`.
- The device assignment contains both devices.
- The run completes without CCL/PJRT collective failures.

Then test PR #592:

```bash
git fetch origin kevin/oneapi-oom
git checkout --detach origin/kevin/oneapi-oom
```

Run the same sharding command. If master fails and #592 works, #592 is mandatory
for B70 TP. If both work, #592 still likely matters for memory preallocation and
collective stability.

Only after that, try small LLMs:

```bash
gpu-run bash -lc '
  cd /mnt/vm_8tb/b70/zml &&
  export ONEAPI_DEVICE_SELECTOR=level_zero:gpu &&
  export ZE_FLAT_DEVICE_HIERARCHY=FLAT &&
  /home/hotschmoe/tools/bazelisk run //examples/llm \
    --config=release \
    --@zml//platforms:cpu=false \
    --@zml//platforms:oneapi=true \
    -- \
    --model=hf://meta-llama/Llama-3.2-1B-Instruct \
    --topk=1 \
    --prompt="Say hello in one sentence."
'
```

Start with small dense models. Then try a dense Qwen/Llama model whose bf16/f16
weights fit across two B70s. Do not start with compressed-tensors W8A8/W4A8;
ZML's current path does not consume our compressed-tensors quantized artifacts or
our oneDNN/XPU int8 scaled-mm kernels.

## Lessons To Apply To Our vLLM Patches

### P0: Audit TP Layout Against ZML

For dense Qwen/Llama on vLLM XPU TP=2, enforce the ZML layout:

- no all-reduce after column-parallel QKV/gate/up
- one reduce/replication after row-parallel O/down
- residual-visible hidden state replicated exactly where needed
- KV heads sharded consistently with model TP

This is the highest-signal code audit because B70 PCIe/collective cost makes
extra all-reduces expensive.

### P0: Move All-Reduces Later And Keep Them Captured

Our `docs/P2P_GPU.md` notes that the B70 win is all-reduce surgery, not oneCCL
knob tuning: graph breaks around collectives dominate raw CCL latency
(`docs/P2P_GPU.md:99-128`). ZML's compiler-visible collectives show the desired
shape: keep communication in the graph/IR and replicate only at semantic
boundaries. For vLLM, prioritize delayed O/down all-reduce and custom-op/captured
regions over transport knobs.

### P0: Port Direct Sharded Loading Ideas

ZML's direct sharded loader should inform our compressed-tensors loader work:

- choose sharding before materialization
- stream one shard per rank/card
- write final packed/device format directly when possible
- avoid full CPU/GPU materialization of transient unpacked tensors

This directly targets W4A8 prepack/OOM and future TP=2 compressed-tensors
checkpoints.

### P1: Compare oneAPI Paged Attention Choices

ZML's oneAPI paged-attention work should be compared against vLLM `TRITON_ATTN`
on B70. The useful pieces are Intel split-K/cached-KV/kernel layout choices. The
generic ZML LLM attention path still defaults oneAPI to vanilla XLA SDPA, so do
not assume the LLM path automatically exercises the oneAPI paged kernel.

### P1: Borrow MoE Sharding Structure, Not MoE Kernels

ZML's expert-axis sharding and local-expert map/all-reduce pattern is useful for
Qwen/MiniMax-style MoE TP/EP thinking. But ZML's oneAPI MoE backend is not
implemented today, and its current MoE path is not our quantized B70 kernel path.
Keep vLLM 0.23 plus our XPU int8/W4A8 work as the serving path.

### P1: Keep W8A8/W4A8 Kernel Research In vLLM

Our B70 INT8 microbench shows real int8-XMX wins:

- GEMM/prefill int8 is 1.06-2.13x bf16, peak 250.9 INT8 TFLOP/s
  (`docs/kernel/19_int8_microbench_results.md:12-14`).
- Decode/GEMV int8 is bandwidth-bound and shape-dependent, with large-N around
  2x and small-N often near 1.1x (`docs/kernel/19_int8_microbench_results.md:15-20`).
- The remaining kernel headroom is small-N/KV projection layout
  (`docs/kernel/19_int8_microbench_results.md:55-58`).

ZML does not replace that. It gives us TP/loader/attention architecture to steal
while vLLM remains the compressed-tensors W8A8/W4A8 execution stack.

## Validation Run

No GPU workload was run during this investigation.

Local setup completed:

```bash
git clone https://github.com/zml/zml /home/hotschmoe/github/zml
/home/hotschmoe/tools/bazelisk --version
# bazel 8.7.0
```

CPU-only validation started:

```bash
cd /home/hotschmoe/github/zml
/home/hotschmoe/tools/bazelisk run //examples/sharding -- --partitioner=shardy --mesh=auto
```

This first Bazel build compiled XLA/MLIR/Zig dependencies on the local aarch64
host and had not touched any GPU. It was interrupted manually after 910.195s at
5388/5470 actions because it was only a cold-cache CPU validation and was still
compiling large MLIR/XLA objects. Bazel reported `build interrupted`, not a
source compile error.

## Immediate Next Actions

1. Test `//examples/sharding` under oneAPI on the B70 pair with `gpu-run`.
2. Repeat on `origin/kevin/oneapi-oom` / PR #592.
3. If #592 is required, track it as the B70 TP ZML baseline until it lands.
4. Audit vLLM XPU TP linears against ZML's QKV/gate/up and O/down placement.
5. Start a vLLM compressed-tensors loader design note based on ZML direct
   per-shard loading.
6. Compare ZML oneAPI paged-attention kernel choices to vLLM `TRITON_ATTN` once
   our Triton-XPU gate is confirmed healthy in the engine process.
