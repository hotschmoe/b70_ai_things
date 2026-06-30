# zml/ -- ZML (Zig + MLIR/XLA/PJRT-oneAPI) backend for the dual B70

Fourth serving backend, added 2026-06-30. ZML is an experimental compiler/TP path, NOT a drop-in
replacement for the sglang/vLLM serving stack. It runs HF safetensors in **bf16/f16 via XLA**; it does
NOT consume our compressed-tensors W8A8/W4A8/W4A16 artifacts and has no oneDNN int8 XPU kernel. So the
request "W8A8 TP=2 / W4A16 DP=2" maps onto zml only as **bf16/f16 dense, sharded TP=2 / replicated DP=2**.
Its value is the architecture to mine: compiler-visible collectives, Shardy/SPMD tensor sharding, and
direct per-shard loading. The deep re-validation (vs the 2026-06-23 `../zml_for_us.md`) is in
`REVIEW_intel_arch.md`.

## Status (current HEAD 89b0908c)

- **oneAPI multi-device is mainline.** PR #592 ("set defaults for oneAPI PjRT create options" +
  collective-fix PJRT plugin `manual-2026-06-23T00-20-00Z`) MERGED into master -- no branch checkout
  needed to attempt TP=2. MoE on oneAPI is now valid too (Triton backend, #603/#600).
- **TP=2 on two B70s: plausible but UNPROVEN.** SPMD plumbing is intact (Shardy default, 2-device
  assignment, in-graph collectives, direct per-shard load). Unknowns: oneAPI PJRT+oneCCL executing the
  collectives correctly on this hardware, and the documented B70 TP=2 firmware/BCS reboot-wedge. Must be
  measured with the sharding example first.
- **Qwen3.6-27B is NOT a config drop-in.** zml ships `qwen3_5` -- the hybrid GatedDeltaNet (GDN) +
  full-attention DENSE text backbone of the Qwen3.5/Next family -- so the hard recurrent backbone EXISTS.
  Missing for our checkpoint: (1) `model_type` detection (exact-string match -> UnknownModelType today),
  (2) the vision tower (zero vision code), (3) the MTP head. Full parity is a multi-week Zig port, and it
  would run bf16-dense only. Treat as an architecture experiment, not a daily-driver candidate.

## Layout

- `REVIEW_intel_arch.md`  -- the re-validation review (read first); what changed since 2026-06-23.
- `build.sh`              -- `bazelisk build` the oneAPI examples (GPU-free; heavy XLA/MLIR compile).
- `test_sharding.sh`      -- THE first GPU test: `//examples/sharding` SPMD across both cards (TP=2 validation).
- `serve_llama_tp2.sh`    -- first dense-LLM milestone: small Llama TP=2 (after sharding is green).

Upstream source clone (git-ignored runtime, NOT repo content): `/mnt/vm_8tb/b70/zml` (HEAD 89b0908c).
Toolchain: `bazelisk` at `~/.local/bin` (reads `.bazelversion` -> Bazel 9.1.1); `zig` at `~/.local/bin`.
The oneAPI PJRT plugin + 2026.0 runtime are fetched hermetically by bazel (amd64-only; matches this box).

## Bring-up order

```sh
# 1. build the oneAPI examples (GPU-free; tens of minutes cold). Fetches the hermetic oneAPI PJRT plugin.
bash zml/build.sh

# 2. CPU sanity (no GPU): validates the build + Shardy SPMD path
cd /mnt/vm_8tb/b70/zml && ~/.local/bin/bazelisk run //examples/sharding -- --partitioner=shardy --mesh=auto

# 3. oneAPI sharding smoke on the 2x B70 (NEEDS the lease):
./bin/gpu-run bash zml/test_sharding.sh

# 4. only if green: small dense Llama TP=2 (NEEDS the lease):
./bin/gpu-run bash zml/serve_llama_tp2.sh
```

## Critical env (do not skip)

- `CCL_TOPO_P2P_ACCESS=0` -- set EXPLICITLY. zml's `oneapi.zig:33` bug defaults this knob from
  `CCL_ATL_TRANSPORT` (= "ofi", garbage); =0 also matches our P2P-wedge discipline. The test/serve scripts
  set it for you.
- `ZE_FLAT_DEVICE_HIERARCHY=FLAT`, `ONEAPI_DEVICE_SELECTOR=level_zero:gpu`.

## Lessons to STEAL into vLLM/sglang (not code -- architecture)

- TP linear layout (verified in zml's `qwen3_5`/`llama`): no all-reduce after column-parallel QKV/gate/up;
  exactly one reduce after row-parallel O/down; residual replicated; KV heads on the model axis. Audit our
  XPU TP linears against this.
- Compiler-visible collectives: keep communication in-graph, replicate only at semantic boundaries
  (zml's structural answer to the graph-break-around-collective cost we pay on vLLM).
- Direct per-shard loading (`zml/io.zig` DirectMemoryWriter): choose sharding before materialization,
  stream one shard per card -- the design target for our compressed-tensors W4A8 prepack/OOM work.
