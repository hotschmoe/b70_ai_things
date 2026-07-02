# SGLANG_MOE_PLAN -- W8A8 int8 MoE (Qwen3.6-35B-A3B) on sglang: Route-A loader port

Date: 2026-06-29. DRAFT from CODE READING ONLY (no GPU touched). Companion to `MOE_UNBLOCK.md`
(the 3-route survey). This doc records exactly what was confirmed from sglang's in-image source, the
loader wiring, the dense-linear reuse plan, the probe + serve steps, and the honest unknowns.

Source of truth read: the `sglang-xpu:mtp` image, package `sglang` at
`/opt/venv/lib/python3.12/site-packages/sglang` (extracted read-only via `docker create` + `docker cp`;
no container was started). All `sglang/srt/...` line numbers below are from THAT build.

---

## 1. THE CRITICAL FINDING (Route-A go question, answered from code)

**YES -- sglang ships the int8 fused-MoE Triton kernel in-tree, AND a complete int8 MoE loader.**
Two independent confirmations:

1. **The kernel.** `sglang/srt/layers/moe/moe_runner/triton_utils/fused_moe_triton_kernels.py`:
   - `:324 fused_moe_kernel(...)`, `:374 use_int8_w8a8: tl.constexpr`
   - int8 GEMM + dequant: `:560 tl.dot(a, b ...)` (int8->int32) then
     `:572/:574 accumulator += tl.dot(a, b) * a_scale[:,None] * b_scale[None,:]`,
     `:597 accumulator *= a_scale * b_scale`, `:606 accumulator.to(compute_type)`.
   - dynamic per-token activation int8 quant is done INSIDE the launcher when `a1_scale is None`:
     `invoke_fused_moe_kernel(...)` (`:714`) -> `:778 elif use_int8_w8a8:` -> `:785 per_token_quant_int8(A)`.
   - `per_token_quant_int8` is a pure Triton kernel (`sglang/srt/layers/quantization/int8_kernel.py:28-89`),
     NOT a CUDA C++ op -> eligible to run on triton-xpu. (One XPU caveat: it uses
     `tl.extra.cuda.libdevice.round` at int8_kernel.py:50 -- must lower on triton-xpu; see risks.)
   This is the SAME platform-agnostic Triton path vLLM uses; no C++/SYCL op in the hot path. So the
   port is a LOADER, not a kernel build -- consistent with MOE_UNBLOCK Route A.

2. **A ready-made loader to mirror.** sglang ALREADY has a generic int8 MoE method:
   `sglang/srt/layers/quantization/w8a8_int8.py:238-387 W8A8Int8MoEMethod` --
   - `:252 create_weights` -> `w13_weight [E,2I,H] int8`, `w2_weight [E,H,I] int8`,
     `w13/w2_weight_scale [E,N,1] f32` (CHANNEL), input scales = None.
   - `:316 process_weights_after_loading` (just re-wraps params).
   - `:329 create_moe_runner` -> `MoeRunner(MoeRunnerBackend.TRITON, cfg)`.
   - `:335 get_triton_quant_info` -> `TritonMoeQuantInfo(use_int8_w8a8=True, per_channel_quant=True,
     w13/w2_scale=..., a13/a2_scale=None)`.
   - `:347 apply` -> `self.runner.run(dispatch_output, quant_info)`.
   The runner threads `use_int8_w8a8` straight into the kernel:
   `sglang/srt/layers/moe/moe_runner/triton.py:53-71` (TritonMoeQuantInfo fields),
   `:129-166 _fused_moe_kernel_sequence(... use_int8_w8a8=...)`, `:220-240 fused_experts(... use_int8_w8a8=...)`.

**Net:** the int8 expert kernel + a generic loader both exist in sglang already. The only thing missing
is **dispatch**: getting our Quark-format int8 checkpoint to USE them.

---

## 2. THE GAP -- sglang's Quark config has no int8 branch

The 35B checkpoint is Quark format (`config.json` -> `quantization_config.quant_method == "quark"`), so
the model loader picks `sglang/srt/layers/quantization/quark/quark.py QuarkConfig`. That config only
dispatches FP8 and MXFP4:
- `quark.py:477-502 get_moe_scheme` -> only `_is_mx_fp4` / `_is_fp8_w8a8`, else
  `raise RuntimeError("Unsupported FusedMoe scheme")` -- **int8 MoE hard-fails here**.
- `quark.py:434-460 _get_scheme_from_config` (linear) -> only MXFP4 / FP8, else
  `raise NotImplementedError` -- **int8 dense linears hard-fail here**.

(For contrast, the vLLM build DOES have the int8 branches -- `quark_moe.py:116-122` dispatches
`QuarkW8A8Int8MoEMethod`, and the dense side is patched to `QuarkW8A8Int8DequantXPU` in
`rdy_to_serve/vllm/qwen36-35b-a3b-w8a8/patches/quark.py:749-771`. We mirror both on the sglang side.)

So the port = a small monkeypatch that teaches sglang's `QuarkConfig` to recognize int8 and route it to
the existing-style int8 MoE loader + a dense dequant linear.

---

## 3. THE LOADER WIRING (what was written) -- `sglang/patches/quark_moe_int8.py`

A pure-Python, mount-not-bake patch (same delivery model as `sglang/patches/w8a8_shim.py`). Contents:

- **`Int8MoEMethod(FusedMoEMethodBase)`** -- a near-verbatim mirror of sglang's native
  `W8A8Int8MoEMethod` (w8a8_int8.py:238-387) and vLLM's `QuarkW8A8Int8MoEMethod` (quark_moe.py:518-815).
  int8 expert weights, per-channel `[E,N,1]` scales, `use_int8_w8a8=True`, `per_channel_quant=True`,
  activation scale `None` (dynamic per-token in-kernel). Hooked in exactly like
  `AWQMoEMethod` (sglang/patches/awq.py:442; dispatch awq.py:138-202).
- **`Int8DequantLinear(LinearMethodBase)`** -- mirror of vLLM `QuarkW8A8Int8DequantXPU`
  (rdy_to_serve/.../patches/quark.py:109-178): int8 `[N,K]` + per-channel scale `[N,1]` -> dequant to
  bf16 ONCE at load -> plain `F.linear` (== W8A16; XPU has no int8 scaled-mm -- w8a8_int8.py:46 only
  imports `int8_scaled_mm` under `_is_cuda`). Default is dequant; an opt-in fast path (env
  `B70_XPU_W8A8_FUSED=1`) can instead route these to the built oneDNN ops via the existing
  `w8a8_shim` machinery.
- **`install()`** -- monkeypatches `QuarkConfig.get_quant_method` to intercept int8 layers BEFORE the
  stock dispatch raises: excluded -> `UnquantizedLinearMethod`; int8 `FusedMoE` -> `Int8MoEMethod`;
  int8 `LinearBase` -> `Int8DequantLinear`; everything else -> original dispatch. Detection
  (`_is_int8_w8a8`) reads the matched spec via `QuarkConfig._find_matched_config` and checks
  `weight.dtype==int8 & input.dtype==int8 & weight.symmetric` (mirror vLLM quark.py:448-474/:501-530).

### Layer routing for THIS model (from config.json)
- 256 routed experts `...mlp.experts.*` -> **`Int8MoEMethod`** (true int8 Triton fused MoE).
- `...self_attn`/`linear_attn.*` projections + `...mlp.shared_expert.{gate,up,down}_proj` ->
  **`Int8DequantLinear`** (int8->bf16 dequant). These are int8 in the ckpt (NOT in `exclude`).
- `exclude` (152 entries) -> `UnquantizedLinearMethod`: all `model.visual.*`, every
  `...mlp.shared_expert_gate`, and `lm_head`.

### Two config-detection paths (both land on the same runtime layout)
- **Quark (have it):** patch `QuarkConfig` as above.
- **Compressed-tensors W8A8 (can produce via llmcompressor):** sglang's NATIVE
  `W8A8Int8Config.get_quant_method` (w8a8_int8.py:104-120) already returns `W8A8Int8MoEMethod` for
  `FusedMoE` and `W8A8Int8LinearMethod` for `LinearBase`. The MoE works as-is on the Triton path; only
  the LINEAR `int8_scaled_mm` is CUDA-only -> reuse the shipped `w8a8_shim.py` for the dense linears.
  NOTE: in THIS image `compressed_tensors/schemes/compressed_tensors_w8a8_int8_moe.py` defines only the
  **NPU** variant (`NPUCompressedTensorsW8A8Int8DynamicMoE`), so if you go the compressed-tensors route,
  drive it through `W8A8Int8Config` (quant_method `w8a8_int8`) rather than `compressed-tensors`, or
  point the MoE scheme at `Int8MoEMethod` from this file. The Quark path is the recommended default
  since we already have that checkpoint and no local Quark quantizer is required.

---

## 4. DENSE-LINEAR REUSE PLAN (linear_attn.* / shared_expert.*)

- Default: `Int8DequantLinear` (correctness-first, fewest moving parts). Same numerical approach the
  vLLM 35B entry ships and that benched coherent.
- Optional fast path: reuse `sglang/patches/w8a8_shim.py` ops unchanged --
  decode `int8_gemm_w8a16(x_f16, B_nt, wscale)`; prefill `dynamic_per_token_int8_quant -> int8_gemm_w8a8`.
  Requires the built `_xpu_C.abi3.so` (`B70_XPU_C_SO`) + `B70_XPU_W8A8_FUSED=1`, exactly as the
  `rdy_to_serve/sglang/qwen36-27b-w8a8` entry does. Per the vLLM finding (rdy_to_serve/.../quark.py:54-55)
  int8 linear is NOT a serve win on this MoE (linears are a minority), so keep dequant until the MoE is
  proven, then A/B the fused linears.

---

## 5. STEP LIST FOR THE GPU DRIVER

1. **Run the Route-A probe (no serve, single card, minutes):**
   `research/w8a8/sglang_moe_int8_probe.py` -- builds the real E=256/top-8/I=512/H=2048 int8 experts and
   calls sglang's `fused_experts(use_int8_w8a8=True, per_channel_quant=True)` in DECODE (T=1) and
   PREFILL (T=256), comparing int8 vs bf16-dequant cosine. Run inside `sglang-xpu:mtp` (see the
   docstring for the exact `docker run` line; source oneAPI setvars + prepend the compiler lib).
   - PASS (cosine>0.99, no nan/inf, no Triton codegen error) => Route A is GO; proceed to serve.
   - NO-GO (kernel raises / mis-codegens on XPU) => fall back to Route C (fused SYCL grouped GEMM,
     MOE_UNBLOCK sec 1).
2. **Wire install() into the serve entrypoint.** Mount `sglang/patches/quark_moe_int8.py` into the
   image (like `w8a8_shim.py`) and call `quark_moe_int8.install()` BEFORE the model builds (env
   `B70_QUARK_MOE_INT8_AUTOINSTALL=1` auto-installs on import, or call it from the launch shim). If
   using the fused dense path also install `w8a8_shim` and set `B70_XPU_W8A8_FUSED=1` + `B70_XPU_C_SO`.
3. **Serve TP=2.** (TP=2 BCS/GuC wedge CURED on kernel 7.1, 2026-07-02.) int8 weights ~35 GB -> ~17.5 GB/card, fits TP=2. Start from
   a copy of `rdy_to_serve/sglang/qwen36-27b-w8a8/serve.sh` (it already does TP=2, `--device xpu
   --attention-backend intel_xpu --linear-attn-backend triton`, `--disable-cuda-graph`,
   `--mamba-ssm-dtype float32`, `--skip-server-warmup`). Point `--model-path` at
   `/models/qwen3.6-35b-a3b/quark-w8a8-int8`, served id e.g. `qwen36-35b-a3b-quark-w8a8-int8`.
   GDN note: this model has `linear_attn.*` -- carry the 27B GDN settings (skip-warmup, fp32 ssm) to
   avoid the `!!!!` GDN poison.
4. **Coherence-gate then bench** vs the vLLM Quark entry (43.1 t/s c1 / 22.2 c4) and the 35B W4A16
   (56.8 t/s). Greedy-only on XPU. Vision is excluded by the quant -> text-only serve (or graft the
   bf16 visual like the dense path, sglang/graft_vision.py).
5. **Only if Triton MoE is slow** (probe runs but PREFILL int8 << bf16, or serve TG underwhelms):
   Route C -- add `is_w8a8` to the SYCL grouped GEMM (MOE_UNBLOCK sec 1, fused_moe_interface.py:32-44,485-521).

---

## 6. OPEN RISKS / WHAT COULD NOT BE CONFIRMED FROM CODE ALONE

- **triton-xpu int8 codegen (THE gate).** `fused_moe_kernel` int8 `tl.dot`(int8->int32) and the
  `per_token_quant_int8` helper's `tl.extra.cuda.libdevice.round` (int8_kernel.py:50) must both lower on
  the B70 triton-xpu. This is precisely what the probe exists to settle; cannot be known by reading.
- **Quark expert tensor naming -> sglang FusedMoE weight_loader.** The Quark ckpt stores per-expert
  `gate_proj/up_proj/down_proj` weight + `weight_scale` (+ symmetric zero_points). sglang's shared
  FusedMoE CHANNEL weight_loader must map these to `w13/w2` and fill `[E,N,1]` scales. This plumbing is
  shared with the proven FP8/AWQ MoE paths, but it was NOT exercised for this exact int8 ckpt here.
  Risk: the loader may choke on the unused `*zero_point` tensors -- if so, add them to the model's
  skip/exclude list (the int8 kernel is symmetric; zero-points are discarded, cf. quark_moe.py:693-702).
- **Shared-expert fusion.** sglang QuarkConfig has `can_fuse_shared_expert` (quark.py:507-537) keyed on
  the shared-expert body NOT being excluded; here the shared_expert body IS quantized (only
  `shared_expert_gate` is excluded), so fusion logic may engage -- verify it doesn't misroute the
  shared expert through the routed-expert int8 path.
- **`get_supported_act_dtypes`.** Our patch leaves QuarkConfig's bf16/fp16 support intact; the GDN
  linear-attn path on B70 wants bf16 (fp16 causal_conv1d crashes, cf. awq.py:104-108). Serve in bf16.
- **TP=2 hardware wedge** (BCS copy-engine job timeout) -- CURED on kernel 7.1 (2026-07-02; AGENTS.md GPU
  Discipline; 70.54.0 pin retired). Still good hygiene: `bin/xpu-health` pre-flight, and do not chain TP=2
  worker-init crashes (that trips the SEPARATE oneCCL state-corruption wedge, untested on 7.1).
- **Sustained concurrent load / `!!!!`.** Same open question as the vLLM entry (MOE_UNBLOCK sec 3): a
  long c4+ soak is needed before calling either backend production for this GDN MoE.

---

## 7. FILES WRITTEN (this session)

- `sglang/patches/quark_moe_int8.py` -- DRAFT loader: `Int8MoEMethod` + `Int8DequantLinear` + `install()`.
- `research/w8a8/sglang_moe_int8_probe.py` -- DRAFT offline Route-A go/no-go probe.
- `research/w8a8/SGLANG_MOE_PLAN.md` -- this document.

All DRAFT / untested on GPU. Nothing here serves or benches until the probe (step 1) is green.
