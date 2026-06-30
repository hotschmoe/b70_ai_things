# Intel Arc B70 support per serving backend

How each of our four serving backends supports the dual Intel Arc Pro B70 (Battlemage / Xe2, 32 GB
each), what runs stock vs. what we had to patch, how each maps the W8A8 / W4A16 quant + TP/DP targets,
and whether each can serve our headline model qwen3.6-27b (`Qwen3_5ForConditionalGeneration` -- a hybrid
GatedDeltaNet linear-attention + full-attention DENSE VLM with a vision tower and a 1-layer MTP head).

Created 2026-06-30 when llama.cpp + zml were added alongside vLLM + sglang. Sources: the per-backend
reviews (`llamacpp/REVIEW_intel_arch.md`, `zml/REVIEW_intel_arch.md`), the patch audit
(`docs/patch_applicability_matrix.md`), `docs/P2P_GPU.md`, and the shelf (`rdy_to_serve/`).

## The fundamental split

There are TWO different worlds here, and most of our hard-won kernel/patch work lives in only one:

- **torch + compressed-tensors world (vLLM, sglang).** int8/int4 WEIGHTS *and* int8 ACTIVATIONS
  (W8A8/W4A8) via our custom oneDNN int8 GEMM ops + per-backend shims; collectives via oneCCL with the
  Battlemage stability env we reverse-engineered; graph capture; MTP/NEXTN; vision grafts. This is where
  the B70 INT8-XMX fast paths actually get exercised.
- **everything-else world (llama.cpp, zml).** llama.cpp = weight-only GGUF quant with fp16/fp32 compute
  and its OWN SYCL kernels; zml = bf16/f16 via XLA with compiler-visible collectives. Neither consumes our
  compressed-tensors checkpoints, neither links our oneDNN int8 ops, and neither uses oneCCL the way vLLM
  does. So almost none of our int8/oneCCL CODE transfers (see `patch_applicability_matrix.md`); what
  transfers is the LESSONS (below).

This is why "serve qwen3.6-27b in W8A8 (tp=2) and W4A16 (tp=1,dp=2)" does not map cleanly onto the two new
backends -- there is no true W8A8 outside vLLM/sglang. The mappings below are the honest closest analogs.

## At-a-glance matrix

| Capability | vLLM (paused) | sglang (primary) | llama.cpp (new) | zml (new) |
|---|---|---|---|---|
| Stock Intel XPU support | yes (xpu platform) | yes (xpu device) | yes (SYCL/GGML backend) | yes (oneAPI PJRT plugin) |
| Build environment | `vllm-xpu-env` images | `sglang-xpu` images | inside oneAPI image (SYCL) | hermetic bazel (oneAPI PJRT) |
| Compute / quant model | torch; W8A8/W4A8/W4A16/int4/fp8/bf16 | torch; W8A8/W4A8/int4woq/bf16 | weight-only GGUF (Q8_0/Q4_K_M/...) | bf16/f16 only (XLA) |
| int8 ACTIVATIONS (true W8A8) | yes (our oneDNN kernels) | yes (our oneDNN kernels) | NO (weight-only) | NO |
| Consumes our compressed-tensors | yes | yes | no | no |
| Multi-GPU TP | TP=2 (oneCCL, wedge-prone) | TP=2 (oneCCL) | `--split-mode tensor` (own all-reduce) | Shardy/SPMD (XLA collectives) |
| Multi-GPU DP | DP=2 + nginx | DP=2 + nginx | 2x server + nginx | replicated mesh |
| Collective stack | oneCCL + our Battlemage env | oneCCL + our Battlemage env | own SYCL ring all-reduce | XLA/PJRT in-graph |
| qwen3.6-27b arch | yes (qwen3_5 + our patches) | yes (qwen3_5 + our patches, daily driver) | YES (stock `qwen35` + GDN SYCL kernel) | backbone only (`qwen3_5` dense; no vision/MTP) |
| vision tower | yes (grafted) | yes (grafted, served) | yes (mmproj `qwen3vl`) | NOT implemented |
| MTP / spec decode | yes | yes (NEXTN, daily driver) | yes (`--mtp`) | NOT implemented |
| OpenAI server + api-key + metrics | yes | yes | yes (`--api-key` `--metrics`) | example only (no server) |
| Daily-driver-ready today | yes (shelf) | YES (current default) | candidate (pending GPU bring-up) | NO (multi-week port) |

## Per-backend detail

### vLLM (paused baseline)
- Default image `vllm-xpu-env:v0230`; W8A8 research image `:int8g` carries `XPUInt8ScaledMMLinearKernel`.
- Patched heavily: our oneDNN int8/int4 GEMM ops (`kernels/`), compressed-tensors W8A8/W4A8/W4A16 loaders,
  graph-capture fake registrations, the Battlemage multi-GPU stability env (`rdy_to_serve/_common/lib.sh`
  `MGPU=(...)`), the wedge guard.
- Paused because it batches concurrent prefill+decode and emits "!!!!" garbage under load; kept as a
  maintained, sweep-gated baseline on the shelf (`rdy_to_serve/vllm/*`).
- TP=2 is the documented wedge surface: `CCL_TOPO_P2P_ACCESS=1` in a TP>1 serve deterministically wedges
  the box (reboot-only). See `docs/P2P_GPU.md`.

### sglang (primary; the daily driver)
- Image `sglang-xpu:mtp`. The production daily driver = `rdy_to_serve/sglang/qwen36-27b-w8a8` (W8A8 fused
  int8 oneDNN ops + NEXTN MTP, TP=2), behind `:18080` with Open WebUI + Prometheus/Grafana.
- Patched: fused int8 W8A8/W4A8 kernels (runtime `.so` + shims), XPU NEXTN/MTP gates, XPUGraph capture,
  GDN fused conv/gating, the qwen3_coder incremental tool-arg streaming fix, Quark int8 MoE loader.
- This is the only stack that actually serves our true W8A8 qwen3.6-27b with vision + MTP today.

### llama.cpp (new -- SYCL/GGML)
- **Build:** inside `sglang-xpu:mtp` (has oneAPI 2025.3 + icx/icpx + oneMKL + oneDNN + cmake + ninja). JIT
  SPIR-V target (`-DGGML_SYCL=ON`, icx/icpx, `-DLLAMA_CURL=OFF`). `llamacpp/build_sycl.sh`. Done 2026-06-30.
- **Stock support is strong:** the fresh HEAD has a dedicated `qwen35`/`qwen35moe` arch with a real SYCL
  gated-delta-net kernel (`ggml/src/ggml-sycl/gated_delta_net.cpp`), an mmproj `qwen3vl` vision tower, and
  bundled MTP. So qwen3.6-27b converts + runs text + vision + MTP with NO source patch.
- **Quant mapping (honest):** Q8_0 = 8-bit WEIGHTS only (fp16 activations -> ~W8A16, NOT true W8A8 -- no
  int8-activation path; B70 INT8-XMX only used for weight dequant via `dpct::dp4a`). Q4_K_M = 4-bit
  weights, fp16 compute (the B70-validated community default). On B70, Q8_0 has historically been ~4x
  slower than Q4_K_M (#21517) -- so unlike our sglang result, Q4_K_M likely WINS here.
- **TP/DP:** "W8A8 tp=2" -> Q8_0 + `--split-mode tensor` (NOT `row`; `row` is unsupported on SYCL). Real
  risk: qwen35 is a hybrid recurrent (GDN) arch and tensor-split across the GDN recurrent state is
  UNVERIFIED (every OTHER recurrent arch is excluded by the arch gate) -- must coherence-sweep-gate. Plus
  compute-runtime 26.x has a known multi-GPU bug (#21747). "W4A16 tp=1,dp=2" -> Q4_K_M, two single-card
  `llama-server` + nginx -- LOW risk, matches the daily-driver pattern, expected production default.
- **Server parity:** OpenAI `/v1/*`, `--api-key`, Prometheus `--metrics`, `--parallel`/`--cont-batching`,
  `--jinja` tool-calling, `--mmproj` vision, `--mtp` speculative -- full daily-driver parity.
- Patches needed from us: NONE for arch/serve. We reuse `bin/xpu-health`/`xe-reset` and the
  `SYCL_UR_USE_LEVEL_ZERO_V2` knob; everything int8/oneCCL-specific is N/A (its kernels + multi-GPU are its
  own). Scripts: `llamacpp/{build_sycl,convert_gguf,serve_dp2_q4km,serve_tp2_q8}.sh`.

### zml (new -- Zig + MLIR/XLA/PJRT-oneAPI)
- **Build:** hermetic bazel 9.1.1 (`bazelisk`) fetches the oneAPI PJRT plugin (`manual-2026-06-23T00-20-00Z`,
  amd64-only -> matches this x86_64 box) + the 2026.0 runtime. `zml/build.sh`. Compile is heavy (XLA/MLIR)
  but GPU-free.
- **oneAPI multi-device is mainline** (PR #592 merged; collective-fix PJRT plugin; MoE-on-oneAPI via Triton
  now valid). TP=2 plumbing is intact (Shardy default, 2-device assignment, in-graph collectives, direct
  per-shard load) but UNPROVEN on B70 hardware -- the same DEVICE_LOST / firmware-BCS-wedge class that bites
  every Intel TP=2 path could bite here too. Watch the `oneapi.zig:33` bug: it defaults `CCL_TOPO_P2P_ACCESS`
  from the wrong env var -> export `CCL_TOPO_P2P_ACCESS=0` explicitly.
- **Quant mapping:** none. zml is bf16/f16 only -> "W8A8 TP=2 / W4A16 DP=2" maps only to "bf16 dense
  sharded / replicated." No quantized serve of our checkpoints.
- **qwen3.6-27b:** NOT a drop-in. zml ships `qwen3_5`, the hybrid GDN+full-attn DENSE text BACKBONE (the
  hard part exists), but lacks `model_type` detection for our config, the vision tower, and the MTP head --
  a multi-week Zig port, bf16-only. Realistic milestone: `//examples/sharding` TP=2 -> small Llama TP=2.
- **Value:** architecture to STEAL (compiler-visible collectives, verified TP linear layout, direct
  per-shard loading), not a serving path. No serving today. Scripts: `zml/{build,test_sharding,serve_llama_tp2}.sh`.

## Portable lessons (transfer as ideas, not code)

From `docs/patch_applicability_matrix.md`. These are the hard-won B70 insights that apply regardless of
backend:

1. **No Arc P2P; route collectives over host.** Battlemage has no working GPU P2P for our collectives;
   `CCL_TOPO_P2P_ACCESS=1` in a TP>1 vLLM serve deterministically wedges the box (reboot-only). The
   host-staged fabric ceiling still bounds llama.cpp's `--split-mode tensor` all-reduce and zml's mesh.
   Both new backends: pin `CCL_TOPO_P2P_ACCESS=0` / avoid Arc P2P.
2. **Graph-break-around-collective is the first-order TP cost** (not oneCCL knob tuning). zml's
   compiler-visible collectives are the structural fix; for vLLM/sglang, delayed O/down all-reduce + kept-
   captured regions.
3. **TP across a hybrid/recurrent (GDN) state is the risk axis.** sglang W8A8 TP=2 needed `--skip-server-warmup`
   to avoid GDN-state "!!!!" poisoning; llama.cpp tensor-split on qwen35 is the analogous unverified risk.
   Always coherence-gate a multi-card serve under concurrent load.
4. **Vision-tower retention is a standing directive.** Every served qwen3.6 must keep its vision tower;
   some quants silently drop it. llama.cpp: convert `--mmproj` and pass it at serve time.
5. **Coherence-gate every serve.** A broken quant/capture/TP serve stays /health-green while emitting a
   single repeated token. The shelf gen-probe gate (in `_common/lib.sh` and each serve.sh) catches it; the
   new backends' serve scripts carry the same gate.
6. **Single-file bind-mount inode gotcha:** editing a single-file Docker bind mount needs `docker restart`,
   not a reload. The llama.cpp DP=2 nginx conf is regenerated + fresh-started each time to avoid this.
7. **CUDA-isms do not port.** Both new backends avoid the entire CUDA-shim class (cuda.libdevice unlink,
   fake op registrations) by design -- llama.cpp/zml never present a CUDA surface.

## Bottom line

- **sglang** stays the primary W8A8 daily driver (only true-W8A8 + vision + MTP path).
- **llama.cpp** is the most promising NEW backend: stock qwen3.6-27b (text+vision+MTP), full server
  parity, low-risk DP=2 Q4_K_M production path + a coherence-gated TP=2 Q8_0 experiment. Ready to test the
  moment the GPUs are idle.
- **zml** is a compiler/TP research path: validate sharding + Llama TP=2 on oneAPI, mine its architecture
  for our vLLM/sglang TP work; qwen3.6 is a future Zig port, bf16-only, not a serving candidate yet.
