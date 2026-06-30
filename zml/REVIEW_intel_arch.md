# ZML On Dual B70 -- Re-Validation Review

Date: 2026-06-30
Author: re-validation pass over the prior note `zml_for_us.md` (2026-06-23).

Clone reviewed: `/mnt/vm_8tb/b70/zml`
Current HEAD: `89b0908c` ("zml/attention/flashattn: make fa2 window_size_left configurable (#595)", Mon Jun 29 2026)
Prior note HEAD: `51314672` ("zml/attention: tune triton kernel for oneAPI performance (#588)", Mon Jun 22 2026)

This document re-verifies every claim in `zml_for_us.md` against the current clone,
states what changed, and answers the six investigation points. GPUs were NOT touched
(reads/grep/web only).

NOTE ON STALE PATHS: `zml_for_us.md` references `/home/hotschmoe/github/zml` and
`/home/hotschmoe/tools/bazelisk`. Both are gone after the box consolidation. The repo
is now a single clone at `/mnt/vm_8tb/github/b70_ai_things` and the ZML clone is at
`/mnt/vm_8tb/b70/zml`. There is no bazel/bazelisk on the box currently.

---

## TL;DR / Verdict

- **PR #592 (`kevin/oneapi-oom`) MERGED.** It is commit `a1acbd36` ("zml/onapi: set
  defaults for oneAPI PjRT create options (#592)", Kevin Barre, Jun 23 2026) and is in
  the current HEAD. So the "first thing to test, possibly mandatory for B70 TP" from the
  prior note is now plain master. You no longer need to check out a branch for it.
- **TP=2 on two B70s is *plausible but unproven* on current master.** All the generic
  SPMD plumbing is present and oneAPI is wired into every path that matters (create
  options, partitioner default, device assignment, direct per-shard loading, MoE Triton
  backend). The two real unknowns are unchanged: (a) whether oneAPI PJRT + oneCCL can run
  the collectives correctly inside compiled graphs on this hardware, and (b) the B70
  TP=2 firmware/BCS wedge we already document in `CLAUDE.md`. ZML's README still says
  "sharding is not yet supported / one GPU", which directly contradicts the code -- so
  treat TP=2 as "expected to compile and attempt, must be empirically validated".
- **Qwen3.6-27B is NOT loadable as-is, but the backbone architecture is closer than the
  prior note implied.** ZML already ships a `qwen3_5` model that IS a hybrid
  GatedDeltaNet (GDN) linear-attention + full-attention dense model (the Qwen3.5 /
  Qwen3-Next family). The missing pieces for our checkpoint are: (1) the vision tower,
  (2) the MTP head, and (3) `model_type` string detection. zml runs HF safetensors in
  bf16/f16 via XLA; it does NOT consume our compressed-tensors W8A8/W4A8/W4A16 artifacts
  and has no oneDNN int8 XPU kernel. "W8A8 TP=2 / W4A16 DP=2" maps onto zml only as
  "bf16/f16 dense, sharded TP=2 / replicated DP=2".
- **Realistic first milestone:** `//examples/sharding` (CPU then oneAPI), then a small
  dense Llama-3.2-1B TP=2 smoke test. Qwen3.6-27B is a multi-week Zig model port
  (vision + MTP), not a config drop-in.

---

## 1. oneAPI Platform Status

### README still contradicts the code (UNCHANGED)
`platforms/oneapi/README.md` still says, verbatim:
- "EXPERIMENTAL: oneAPI ... currently in an early stage of development" (line 1, 6)
- "The sharding is not yet supported, so only models that fit in the GPU memory can be
  run." (line 8)
- "One GPU is supported on Linux." (line 19)

The code contradicts this (generic multi-device SPMD is present -- see sections 2 and 3).
This is the same contradiction the prior note flagged; the README was not updated.

### `oneapi.zig` CHANGED since prior review
Prior note cited `platforms/oneapi/oneapi.zig:12-63`. The file was rewritten (commit
`4cd71394` "#582 add hermetic oneAPI 2026 runtime deps") and now:
- `isEnabled()` gates on `ZML_RUNTIME_ONEAPI` (oneapi.zig:12-14) -- unchanged in spirit.
- `hasOneApiDevices()` checks `/dev/dri/renderD*` (oneapi.zig:16-28) -- unchanged.
- Loads `libpjrt_oneapi.so` from the bazel runfiles `libpjrt_oneapi/sandbox/lib`
  (oneapi.zig:49-63) -- unchanged in spirit.
- **NEW: `setupOneAPIEnv()` (oneapi.zig:30-34)** sets oneCCL defaults at load time:
  `CCL_LOG_LEVEL=error`, `CCL_ATL_TRANSPORT=ofi`, and **`CCL_TOPO_P2P_ACCESS`**.

WARNING / LIKELY BUG to watch (oneapi.zig:33):
```zig
_ = c.setenv("CCL_TOPO_P2P_ACCESS", c.getenv("CCL_ATL_TRANSPORT") orelse "1", 1);
```
The default value is read from `CCL_ATL_TRANSPORT` (not `CCL_TOPO_P2P_ACCESS`), and
line 32 has already set `CCL_ATL_TRANSPORT=ofi`. Net effect: unless you pre-export
`CCL_TOPO_P2P_ACCESS`, ZML sets `CCL_TOPO_P2P_ACCESS=ofi` (a non-numeric, effectively
garbage value). The intent was clearly to default P2P access to `1`. Two consequences
for us:
1. This is the exact knob our `CLAUDE.md` / `P2P_GPU.md` flags as wedge-prone in vLLM
   TP serves. ZML drives oneCCL through PJRT/XLA (a different code path than vLLM's
   multiproc workers), and our own notes say the raw allreduce path works with P2P=1 --
   so ZML may be on the "good" side. But the *firmware/BCS copy-engine TP=2 wedge*
   (reboot-only) is hardware-level and could still bite a ZML TP=2 run.
2. Because of the bug, **explicitly export `CCL_TOPO_P2P_ACCESS=0` (or `1`) before any
   ZML oneAPI run** so the collective transport is not configured from a garbage value.

### `zml/platform.zig` create options -- #592 LANDED
Prior note: master's oneAPI create options were nearly empty and `toNamedValues` only
applied XLA GPU allocator options to CUDA/ROCm, not oneAPI; #592 was to fix that.

Current HEAD (`zml/platform.zig`):
- `CreateOptions.toNamedValues` now routes oneAPI through the XLA GPU path:
  `.cuda, .rocm, .oneapi, .metal => self.xla_gpu.writeNamedValues(&values)`
  (platform.zig:714). **This is the #592 change, now on master.**
- The default GPU allocator is BFC with `preallocate=true, memory_fraction=0.90`
  (platform.zig:648, 679-683). This now applies to oneAPI -- so a oneAPI B70 run will
  preallocate ~90% of the 32 GiB and use BFC, which is the OOM/collective-stability
  reason #592 mattered.
- `oneapi: struct {} = .{}` is still an empty per-target struct (platform.zig:651); the
  GPU options come from the shared `xla_gpu` field, not a oneAPI-specific one.
- `Platform.auto` order is `tpu, neuron, rocm, cuda, oneapi, metal, cpu`
  (platform.zig:344-353) -- oneAPI is still tried before CPU fallback (matches prior).

### PJRT plugin artifact -- bumped by #592, now pinned in HEAD
`platforms/oneapi/oneapi.bzl:4-7`:
```
PJRT_ONEAPI_RELEASE = "manual-2026-06-23T00-20-00Z"
PJRT_ONEAPI_ARTIFACT_SHA256 = "97e0892e7d3815118c897d4f4134004f8bca177332de8be2384f71f6f55e954f"
PJRT_ONEAPI_ARTIFACT_URL = ".../pjrt-oneapi_linux-amd64.tar.gz"
```
This is exactly the artifact #592 bumped to (`manual-2026-06-23T00-20-00Z`, the "collective
fix" build). `git log -S` confirms `a1acbd36` (#592) is the only commit that introduced
this string. So the new collective-fixed PJRT plugin is what current master pulls.

---

## 2. PR / Branch Status

### Merged since the prior review (`51314672..HEAD`, 21 commits)
oneAPI / sharding / attention / MoE relevant:
- `a1acbd36` **#592** oneAPI PjRT create options + collective-fix plugin -- **MERGED**
  (this is THE branch the prior note told us to test; `kevin/oneapi-oom` is now deleted
  from `origin` heads).
- `33ced8fa` **#603** "add oneAPI as Triton backend selector for MoE" -- **MERGED**.
  This flips the prior note's MoE conclusion (see section 6 / MoE below).
- `c023e26f` **#600** "enable moe for any compute capability" -- **MERGED**.
- `c2cbcb82` **#605** "Optimize heuristic and parameters of triton attention" -- MERGED.
- `89b0908c` **#595** flashattn `window_size_left` configurable -- MERGED (current HEAD).
- `33a3b379` **#593** + `62519536` **#613** + `a4993854` **#598**: Metal platform + Metal
  MoE added. Not B70-relevant, but it explains why `.metal` now appears alongside
  `.oneapi` in many switch arms (auto ordering, create options, mesh builder).
- Infra: `aa90f928` **#601** bump to Bazel **9.1.1**; `f9ec63fd` **#599** XLA bump;
  `1dbe39d3`/`d19949e2` parallel GCS/S3/HF reads; `1d71ee7b` safetensors 1GB read cap.

### Still-open branches relevant to us (from `git ls-remote --heads origin`)
- oneAPI dev branches persist: `kevin/oneapi-2026`, `kevin/oneapi-2026-SAVE`,
  `raphael/oneapi-2026`, `steeve/oneapideb`. These are the live oneAPI runtime/debug
  lines; if master TP=2 misbehaves on B70, these are the branches to diff against.
  (`kevin/oneapi-oom` is gone -- merged as #592.)
- Loader: `corendos/loader-toolbox` (prior note's #586 "rework zml.io.load") -- still
  open / unmerged.
- Sharding: `corendos/llama-sharded-tests`, `gw/sharding-post-review` -- still open.
- Attention: `corendos/triton-mha-kernel`, `corendos/paged-attention-2-triton`,
  `corendos/paged-attention-2-flashattn-sharded`, plus many new `paged-attention-2-*`
  variants (cudnn/flashinfer/fp8) -- still open.
- MoE: `louis/moe_triton_sharding`, `louis/moe-ep`, `louis/moe-laguna`, plus new
  `louis/gpt_oss_moe_backend`, `louis/nemotron-moe` -- still open.

### Recommended way to get multi-B70 TP working today
**Plain master (`89b0908c`).** #592 is merged, the artifact is the collective-fix build,
oneAPI is wired into create options / partitioner / device assignment / direct loading /
MoE-Triton. No branch checkout is required to attempt TP=2 anymore. Keep
`raphael/oneapi-2026` / `kevin/oneapi-2026` in your back pocket as the debug lines.

### Public Intel/Arc/B70 multi-GPU signal (web, June 2026)
No public zml/zml GitHub issue specifically about B70/Arc oneAPI sharding surfaced. The
broader ecosystem signal matches our own `CLAUDE.md`: multi-GPU TP=2 on Intel GPUs is a
known pain point -- vLLM issues #6701 and intel/ipex-llm #13131 report
`UR_RESULT_ERROR_DEVICE_LOST` / GPU HANG at `tensor_parallel_size=2` on Intel GPUs, and
Puget's June-2026 4x B70 re-benchmark runs `intel/llm-scaler-vllm` with
`CCL_TOPO_P2P_ACCESS=0` and `VLLM_WORKER_MULTIPROC_METHOD=spawn`. The DEVICE_LOST class
of TP=2 failure is an Intel-platform-wide pattern, not a zml-specific one -- another
reason to validate ZML TP=2 empirically and to pin `CCL_TOPO_P2P_ACCESS` explicitly.

---

## 3. Sharding / TP=2 Plumbing (re-verified)

All prior claims hold at current HEAD (line numbers shifted; content intact):
- Default partitioner for oneAPI is **Shardy**: `Partitioner.fromTarget` returns
  `.shardy` for `.cpu, .cuda, .rocm, .tpu, .oneapi, .neuron, .metal`
  (`zml/Sharding.zig:48-52`).
- Replicated sharding is always kept as a fallback (`zml/module.zig:126-133`).
- `mhlo.num_partitions` / `mhlo.num_replicas` emitted (`zml/module.zig:231-242`).
- Shardy mesh ops emitted (`zml/module.zig:280-298`).
- SPMD compile options set: `use_spmd_partitioning=true`, `use_shardy_partitioner`
  toggled by partitioner, explicit `DeviceAssignment` with replica/computation counts
  and per-partition device ids (`zml/module.zig:579-620`).
- oneAPI compile override disables autotune + command buffers + cublasLt:
  `xla_gpu_autotune_level=0`, `xla_gpu_enable_command_buffer=""`,
  `xla_gpu_enable_cublaslt=false` (`zml/module.zig:648-651`).

### The weak spot PERSISTS: oneAPI uses the CPU mesh builder
`PhysicalMesh.auto` still maps oneAPI (and now Metal) to the simple CPU mesh:
`.oneapi, .metal => cpu(allocator, platform_devices)` (`zml/Sharding.zig:906`). The
`cpu()` builder makes a single `.bus`/`.tree` axis over the devices
(`zml/Sharding.zig:924-935`). The real GPU topology builder `gpu()` (link /
point-to-point, used by CUDA/ROCm) is NOT used for oneAPI (`zml/Sharding.zig:903`,
938-959).

NEW nuance since prior review: `shardableAxes` now lists `.oneapi => &.{ .link, .bus }`
(`zml/Sharding.zig:701`), distinct from `.cpu, .metal => &.{.bus}`. So the sharding layer
*knows* oneAPI could shard on a `.link` axis -- but the auto mesh only creates a `.bus`
axis, so a 2-device auto run gets a 2-wide `.bus` axis. For TP=2 that is logically fine
(one 2-wide model axis), it is just not a B70-PCIe/link-aware topology. A custom
`PhysicalMesh` (the `--mesh` hook in the sharding example) could build a `.link` axis if
needed.

### Is two-B70 TP=2 expected to work on current HEAD?
Compiler/IR side: yes -- the graph will compile with `num_partitions=2`, a 2-device
assignment, Shardy SPMD, and collectives in-graph. Execution side: UNPROVEN on this
hardware. The open risks are (a) oneAPI PJRT + oneCCL executing the all-reduce/all-gather
collectives correctly across two B70s, and (b) the documented B70 TP=2 firmware/BCS
wedge. This must be measured with `//examples/sharding` first (section: First GPU Test).

---

## 4. Model Support / Qwen3.6-27B Gap

### What the model zoo has (current HEAD)
`examples/llm/models.zig` registers exactly four model types (models.zig:16-20):
`lfm2`, `llama`, `qwen3_5`, `qwen3_5_moe`. Detection is by exact `config.json`
`model_type` string -> enum name (`detectModelType`, models.zig:272-277): an unknown
`model_type` returns `error.UnknownModelType`.

(Correction to prior note: `qwen3_5_moe` was ALREADY present at the prior HEAD
`51314672` -- `git ls-tree` confirms -- the 2026-06-23 note just did not mention it.
So MoE Qwen support is not new; what is new is oneAPI being a valid MoE backend, below.)

### The big positive: `qwen3_5` is a hybrid GDN model
`examples/llm/models/qwen3_5/model.zig` is a hybrid linear-attention + full-attention
DENSE model -- the Qwen3.5 / Qwen3-Next family:
- Layers are either `full_attention` (SelfAttn) or `linear_attention` (GatedDeltaNet)
  (model.zig:35-39, 305, 400-420).
- It carries `GatedDeltaNet` with `conv_state` + `recurrent_state` KV cache
  (model.zig:404-405; `qwen3_5/inference.zig:457-471, 649-660`).
- Config is VLM-shaped: nested `text_config` (model.zig:11-13) and `mrope_section[3]`
  multimodal RoPE (model.zig:42, 505) -- i.e. it is the *text backbone of a Qwen3.5-VL*
  config, with vision deliberately not implemented.
- TP layout is the standard-correct split (verified): q/k/v `.dout=.model`, o_proj
  `.dout=.replicated, .d=.model`, MLP gate/up intermediate-sharded, down row-parallel,
  residual replicated, KV heads `.h=.model` (model.zig:432-443, 495-498, 543-558,
  370-375).

Web cross-check: HF `model_type "qwen3_5"` inherits from Qwen3-Next (`qwen3_next`): ~75%
GatedDeltaNet linear-attn layers, ~25% full-attn (every 4th layer), and the full Qwen3.5
is additionally a VLM with MoE MLPs. ZML's `qwen3_5` implements the dense text backbone
of that family; `qwen3_5_moe` adds the MoE MLP variant.

### The concrete gap for OUR `qwen3.6-27b/bf16`
Our checkpoint is described as a DENSE VLM: hybrid GDN+full attention + a **vision
tower** + an **MTP head**. Mapping to zml:
1. **GDN hybrid backbone: ALREADY SUPPORTED** by `qwen3_5` (dense). This is the hard part
   and zml has it. This is a major upgrade vs. the prior note, which only discussed the
   generic Llama/Qwen TP layout.
2. **`model_type` detection: MISSING.** If our config says `qwen3_6` (or `qwen3_5_vl`,
   etc.), `detectModelType` returns `error.UnknownModelType`. If the architecture is
   tensor-compatible, this is a small alias add; if tensor names/config keys differ, the
   `Config`/`TextConfig` struct (model.zig:11-33) and weight prefixes need adjusting.
3. **Vision tower: NOT IMPLEMENTED.** There is zero vision/visual code in `examples/`
   (grep for `vision|visual` finds nothing). The model loads only `text_config`. A VLM
   serve path (image patch embed + ViT + projector + mRoPE wiring) is a brand-new Zig
   model component.
4. **MTP head: NOT IMPLEMENTED.** No `mtp|nextn|multi_token` anywhere. zml has no
   speculative/MTP decode path; you would lose the MTP speedup entirely.

So Qwen3.6-27B is **not a config drop-in**. Even text-only it needs at minimum a
`model_type` alias plus config/tensor reconciliation against zml's `qwen3_5`; full
parity (vision + MTP) is a multi-week Zig port. And remember: it would run in **bf16/f16
dense**, not from our W8A8/W4A8 compressed-tensors checkpoint, with no oneDNN int8 XPU
kernel -- so even a successful port would be a quality/architecture experiment, not a
replacement for the sglang W8A8 daily driver.

### What zml CAN run today on oneAPI for a first TP=2 smoke test
- `llama` (Llama-3.2-1B-Instruct / Llama-3.1-8B) -- the README's own example; smallest,
  cleanest first dense TP=2 target.
- `qwen3_5` dense (if you have a real Qwen3.5 dense checkpoint whose `model_type` is
  literally `qwen3_5`) -- exercises the GDN hybrid path on oneAPI.
- `lfm2` (hybrid conv+attention) and `qwen3_5_moe` (MoE, now valid on oneAPI via the
  Triton backend) -- secondary targets.

---

## 5. Build Requirements On THIS Box (x86_64, Ubuntu 26.04, no bazel yet)

### Bazel
- Repo pins **Bazel 9.1.1** (`.bazelversion` = `9.1.1`, set by #601). Install bazelisk
  (it reads `.bazelversion` and fetches 9.1.1) -- the prior note's local "bazel 8.7.0"
  is now too old.
- The repo uses bzlmod (`MODULE.bazel` + `MODULE.bazel.lock` present; there is a `bazel`
  wrapper and `bazel.sh`).

### Architecture match (POSITIVE DELTA vs prior review)
The oneAPI PJRT artifact is **x86_64/amd64 only**: `pjrt-oneapi_linux-amd64.tar.gz`
(oneapi.bzl:6) and all oneAPI deb packages are read for `["amd64"]` (oneapi.bzl:181,191).
The `platforms:oneapi.enabled` config_setting is gated to `@platforms//cpu:x86_64`
(`platforms/BUILD.bazel`). The prior review was attempted on an **aarch64 laptop** where
this artifact does not exist -- it could not have built oneAPI at all. The current box is
**x86_64**, so the artifact arch now MATCHES. This removes a hard blocker the prior note
did not call out.

### Build flags
From `platforms/oneapi/README.md` and `.bazelrc`:
- `--config=release` => `--compilation_mode=opt --copt=-Ofast --strip=always`
  release_safe Zig (`.bazelrc:107-110`).
- `--@zml//platforms:cpu=false --@zml//platforms:oneapi=true` to select the oneAPI PJRT
  plugin instead of the CPU one (defaults: `cpu=True`, `oneapi=False`, `platforms/BUILD.bazel`).
- Optional `--config=native` for `-march=native` if you want host-CPU tuning of the
  frontend.

### Runtime env
- `ONEAPI_DEVICE_SELECTOR=level_zero:gpu` (select all L0 GPUs) or `level_zero:0,1` to pin
  exactly the two cards. The README example uses `level_zero:0` (single GPU); for TP=2
  you want both cards visible.
- `ZE_FLAT_DEVICE_HIERARCHY=FLAT` so each B70 is one PJRT device (avoid tile/sub-device
  composite enumeration).
- **`CCL_TOPO_P2P_ACCESS=0`** (or `1`) -- set explicitly to override the garbage default
  from the `oneapi.zig:33` bug AND to align with the wedge guidance in `CLAUDE.md`.
- ZML also auto-sets `CCL_LOG_LEVEL=error` and `CCL_ATL_TRANSPORT=ofi` (oneapi.zig:31-32).

### How the oneAPI PJRT plugin is obtained
Hermetic, fetched by bazel as a pinned http_archive: release tag
`manual-2026-06-23T00-20-00Z`, sha256 `97e0892e...`, from `zml/pjrt-artifacts`
(oneapi.bzl:4-7, 173-178). It is sandboxed together with the oneAPI 2026.0 runtime debs
(CCL 2022.0, MPI 2021.18, MKL/SYCL 2026.0, compiler runtime, UMF 1.1, TCM/hwloc 1.5) into
a runfiles `sandbox/lib` dir (`libpjrt_oneapi.BUILD.bazel:39-83`), and `oneapi.zig:49-63`
loads `libpjrt_oneapi.so` from there at runtime. No system oneAPI install is required.
A `dlopen` shim (`zmlxoneapi.zig`) rewrites un-versioned `.so` names to the sandboxed
versioned ones. Note this is a **manual-dated** artifact (not a semver release), so it
moves whenever the oneAPI team rebuilds the plugin -- #592 was exactly such a bump.

---

## 6. Loader / Attention / MoE Lessons -- Still Valid To Steal?

### Loader (direct per-shard) -- STILL VALID
`zml/io.zig` still streams oneAPI (and CUDA) directly into per-device PJRT buffers:
`MemoryWriter.init` selects `DirectMemoryWriter` for `.cuda, .oneapi`, and
`BufferedMemoryWriter` for the rest (`zml/io.zig:304-305`). `DirectShardWriter` builds a
per-device transfer-manager buffer (`zml/io.zig:377-433`). The design lesson for our
compressed-tensors loader (choose sharding before materialization; stream one shard per
card; avoid full transient unpacked tensors -> targets the W4A8 prepack OOM) holds
unchanged. `corendos/loader-toolbox` (#586) is still the unmerged "reusable Loader"
branch to watch.

### Attention -- STILL VALID, with the same caveat
- Generic LLM attention still auto-selects **vanilla XLA SDPA** on oneAPI:
  `attention.Backend.auto` returns `.vanilla` for `.cpu, .rocm, .tpu, .oneapi`
  (`zml/attention/attention.zig:34`; vanilla path = `zml.nn.sdpa`, line 161-169).
- Paged attention still auto-selects **Triton** on oneAPI:
  `paged_attention.Backend.auto` returns `.triton` for `.oneapi` (and cuda/rocm)
  (`zml/attention/paged_attention.zig:24`). #605 tuned the triton heuristics since the
  prior review.
- Caveat unchanged: the plain LLM example path uses vanilla SDPA, so it does NOT
  automatically exercise the oneAPI Triton paged kernel. Still a good source of
  Intel split-K / cached-KV layout ideas to compare against our vLLM `TRITON_ATTN`.

### MoE -- CHANGED (prior note's conclusion is now STALE)
Prior note: "`Backend.auto` does not include oneAPI; oneAPI falls to
`error.UnimplementedMoEBackend`." This is **no longer true**. Per #603/#600,
`moe.Backend.auto` now returns `.triton` for `.cuda, .rocm, .oneapi` on bf16/f16/f32
weights (`zml/moe/moe.zig:22-38`). So oneAPI MoE now maps to the Triton MoE backend
(unimplemented is only for truly unknown targets). The expert-axis sharding + local
expert map + in-graph all-reduce structure to steal is still present
(`zml/moe/moe.zig:172-221`, with a prefill/decode pair around 250-280); #577 moved the
allReduce into the manual computation block. For us this means zml is now a *better*
reference for Qwen/MiniMax-style MoE EP on oneAPI -- but it is still bf16 Triton, NOT our
quantized Quark/compressed-tensors W8A8 MoE kernel path. Keep vLLM/sglang for serving.

### W8A8/W4A8 kernel research -- STILL in vLLM/sglang
Unchanged: zml has no int8 XPU scaled-mm / oneDNN path and does not consume our
compressed-tensors artifacts. It contributes TP/loader/attention/MoE *architecture* to
steal; our INT8-XMX kernel research stays in the vLLM/sglang stack.

---

## First GPU Test (exact command)

Do NOT bypass the lease. First check devices, then run the sharding example on CPU, then
on oneAPI. Bazelisk must be installed (it will fetch Bazel 9.1.1).

```bash
gpu-run --status
ls -l /dev/dri/renderD*
```

CPU sanity (no GPU touch, validates the build + Shardy SPMD path):
```bash
cd /mnt/vm_8tb/b70/zml
bazelisk run //examples/sharding -- --partitioner=shardy --mesh=auto
```

oneAPI sharding smoke test on the two B70s (the real first GPU test):
```bash
gpu-run bash -lc '
  cd /mnt/vm_8tb/b70/zml &&
  export ONEAPI_DEVICE_SELECTOR=level_zero:gpu &&
  export ZE_FLAT_DEVICE_HIERARCHY=FLAT &&
  export CCL_TOPO_P2P_ACCESS=0 &&   # override the oneapi.zig:33 garbage default
  bazelisk run //examples/sharding \
    --config=release \
    --@zml//platforms:cpu=false \
    --@zml//platforms:oneapi=true \
    -- \
    --partitioner=shardy \
    --mesh=auto
'
```
Notes:
- The sharding example's runtime flags are only `--partitioner=shardy|gspmd` and
  `--mesh=auto|mock` (`examples/sharding/main.zig:19-22, 71-109`). `--mesh=mock` needs
  >=8 devices (`buildMockMesh`, main.zig:116) so it is NOT usable on 2 cards; use
  `--mesh=auto`.
- `--mesh=auto` builds the oneAPI auto mesh (= CPU `.bus` builder, section 3) over the
  visible devices and uses the default `CreateOptions` (BFC, 90% preallocate, P2P via
  the #592 path).

What to look for:
- PJRT enumerates exactly two addressable B70 devices (check the `platform.fmtVerbose`
  log, main.zig:172).
- ZML creates a oneAPI platform, not CPU fallback.
- Compile reports `num_partitions=2` and a 2-device assignment.
- Run completes with no oneCCL/PJRT `DEVICE_LOST` / collective failure and no box wedge
  (run `bin/xpu-health` after).

Only after the sharding example is green, try a small dense LLM:
```bash
gpu-run bash -lc '
  cd /mnt/vm_8tb/b70/zml &&
  export ONEAPI_DEVICE_SELECTOR=level_zero:gpu &&
  export ZE_FLAT_DEVICE_HIERARCHY=FLAT &&
  export CCL_TOPO_P2P_ACCESS=0 &&
  bazelisk run //examples/llm \
    --config=release \
    --@zml//platforms:cpu=false \
    --@zml//platforms:oneapi=true \
    -- \
    --model=hf://meta-llama/Llama-3.2-1B-Instruct \
    --topk=1 \
    --prompt="Say hello in one sentence."
'
```

---

## What Changed Since 2026-06-23 (delta vs `zml_for_us.md`)

| Area | Prior note (HEAD 51314672) | Current HEAD 89b0908c | Verdict |
|---|---|---|---|
| PR #592 oneAPI create options | open branch `kevin/oneapi-oom`, "first to test" | **MERGED** as `a1acbd36`; branch deleted | On master now |
| `toNamedValues` for oneAPI | only CUDA/ROCm got XLA GPU opts | `.oneapi` included (platform.zig:714) | Fixed |
| PJRT oneAPI artifact | `manual-2026-06-18T20-14-00Z` | `manual-2026-06-23T00-20-00Z` (collective fix) | Bumped |
| `oneapi.zig` | thin loader | + `setupOneAPIEnv` sets CCL/P2P (with a bug, line 33) | Watch |
| MoE on oneAPI | `error.UnimplementedMoEBackend` | oneAPI -> Triton MoE (#603/#600), moe.zig:24 | Newly supported |
| Qwen GDN model | (only generic TP layout discussed) | `qwen3_5` = hybrid GDN dense model confirmed; `qwen3_5_moe` also present | Backbone exists |
| Bazel | local 8.7.0 | repo pins **9.1.1** (#601) | Upgrade required |
| Host arch | aarch64 laptop (oneAPI artifact unavailable) | x86_64 box; artifact is amd64-only | Now buildable |
| Metal platform | absent | added (#593/#598/#613) -- shares switch arms w/ oneAPI | Cosmetic for us |
| README oneAPI status | "early / no sharding / one GPU" | UNCHANGED (still contradicts code) | Same contradiction |
| PhysicalMesh oneAPI | CPU mesh builder (weak spot) | UNCHANGED (Sharding.zig:906); `.link` now in shardableAxes | Same weak spot |
| Sharding/SPMD/loader/attention | as documented | all re-verified, line numbers shifted only | Holds |

---

## Immediate Next Actions

1. Install bazelisk; confirm it fetches Bazel 9.1.1 for this repo.
2. `bazelisk run //examples/sharding -- --partitioner=shardy --mesh=auto` on CPU to
   validate the build end-to-end (no GPU).
3. Under the lease, run the oneAPI `//examples/sharding` smoke test on the 2x B70 with
   `CCL_TOPO_P2P_ACCESS=0` set explicitly. Capture PJRT device count, `num_partitions`,
   and whether collectives complete without DEVICE_LOST / box wedge. Run `bin/xpu-health`
   after.
4. If sharding is green, run Llama-3.2-1B TP=2 as the first dense-LLM smoke test.
5. Audit our vLLM/sglang XPU TP linears against the re-verified ZML layout (no all-reduce
   after column-parallel QKV/gate/up; one reduce after row-parallel O/down; residual
   replicated; KV heads on the model axis).
6. Treat Qwen3.6-27B as a scoped Zig-port investigation: start from `qwen3_5` (text
   backbone exists), then decide whether vision + MTP are worth porting given it would
   run bf16-dense only (no W8A8) -- i.e. an architecture experiment, not a daily-driver
   replacement.
