# NVFP4 (Qwen3.6-27B) on sglang-XPU -- port bring-up (2026-07-16)

First time the NVFP4 quant is stood up on the **sglang** backend (all prior NVFP4 work is vLLM-only,
`vllm/nvfp4/`). Target: `nvidia/Qwen3.6-27B-NVFP4`, the ModelOpt MIXED_PRECISION checkpoint
(`models/files/qwen3.6-27b/nvfp4-modelopt/`, quant_algo `MIXED_PRECISION`). Same GDN-hybrid multimodal
arch (`Qwen3_5ForConditionalGeneration`) as the int4/w8a8 sglang shelf.

Status: **infra COMPILED + CPU-load-verified; NOT yet GPU-served.** The GPU "yolo" test is deferred to
the main session (see the exact command at the bottom). This session did not hold the GPU lease.

## Checkpoint layout (401 quantized layers)

- MLP gate/up/down (193 layers) -> `quant_algo = "W4A16_NVFP4"`: 4-bit E2M1 weight `weight` uint8 [N, K/2],
  per-16-K fp8-e4m3 block scale `weight_scale` [N, K/16], fp32 `weight_scale_2` (global) + `input_scale`.
- self_attn + GDN in_proj (208 layers) -> `quant_algo = "FP8"`: per-tensor E4M3 weight + input scale.
- norms / conv1d / embed / lm_head / vision tower / mtp -> BF16 (in `ignore`; UnquantizedLinearMethod).
- KV cache -> declared fp8 (`kv_cache_scheme` num_bits 8) but NO calibrated scales shipped.

## A. What upstream sglang already has (verified in-source, sglang 0.5.6.post3)

sglang **already ships the ModelOpt loader natively** (unlike the W4A8 compressed-tensors case):
`srt/layers/quantization/modelopt_quant.py` has `ModelOptMixedPrecisionConfig` (registered as
`modelopt_mixed`), `ModelOptFp4LinearMethod` (NVFP4 dense), `ModelOptFp8LinearMethod`,
`ModelOptFp4Config`, `ModelOptNvFp4FusedMoEMethod`, plus `nvfp4_online`, `mxfp4*`, marlin/cutlass/
flashinfer fp4 helpers. The multimodal `Qwen3_5ForConditionalGeneration` (GDN + vision + mtp) is in
the qwen3_5 EntryClass. So the model + config plumbing is native -- **no model-registry / GDN-config
hacks** (the W4A8 shim needed those only because that ckpt was text-only flattened).

**But there is NO native XPU NVFP4 kernel** -- confirmed the web roadmap claim in-source:
- All NVFP4 dense GEMM backends are CUDA-only: cutlass (`is_cuda()` gated, `sgl_kernel`),
  flashinfer trtllm (`is_sm100_supported`), marlin (`prepare_nvfp4_layer_for_marlin`, CUDA kernel).
  `ModelOptFp4LinearMethod.process_weights_after_loading` raises
  `"ModelOpt NVFP4 native dense GEMM backends require SM100+"` on anything else and `.cuda()`s the
  swizzled scales; `.apply` calls `fp4_gemm` (cutlass/flashinfer). None run on XPU.
- `cutlass_fp8_supported()` returns `False` on XPU (`not _is_cuda`), so the FP8 attention path's
  `apply_fp8_linear` has no accelerated XPU route either.
- **Routing gap**: the MLP's `quant_algo` is `"W4A16_NVFP4"`, but `ModelOptMixedPrecisionConfig.
  get_quant_method` matches only `"NVFP4"`/`"FP8"` exactly -> W4A16_NVFP4 falls through to
  `UnquantizedLinearMethod` (would load a uint8 4-bit packed weight as a plain bf16 Linear = crash).

Net: sglang gives us the loader/config/arch for free; we supply (1) the XPU kernel, (2) the routing
fix, (3) XPU process_weights/apply for the NVFP4 + FP8 methods.

## B. The kernel -- nvfp4_gemm_w4a16 built for sglang's torch 2.12 ABI

Same op as the vLLM fused path (`kernels/nvfp4_gemm_w4a16.h`,
`torch.ops._xpu_C.nvfp4_gemm_w4a16(A_bf16[M,K], B_f4e2m1[K/2,N] uint8 NT-view, bias?,
B_scale_bf16[K/16,N], group_size=16) -> bf16[M,N]`; weights stay 4-bit f4_e2m1 resident, decompressed
in the oneDNN JIT gemm; bit-exact vs the E2M1 reference, 2.85x bf16 at decode).

**Both the sglang and vLLM images are torch 2.12.0+xpu**, so the ABI matches (the W4A8 case had to
rebuild only because vLLM was torch 2.11 then). Confirmed: the existing vLLM v0240 .so
(`/mnt/vm_8tb/b70/nvfp4_fused_kernel/_xpu_C.abi3.so`) loads and registers `nvfp4_gemm_w4a16` in the
sglang image directly. Still, per the "one source, compiled per-backend" contract we build a
dedicated sglang-ABI .so:

    bash sglang/nvfp4/build_nvfp4_kernel_sglang.sh
      # source: /mnt/vm_8tb/b70/vllm-xpu-kernels-nvfp4-sglang  (isolated copy of vllm-xpu-kernels-v0240,
      #   which carries nvfp4_gemm_w4a16.h + the binding; made with:
      #   rsync -a --exclude build/ --exclude .deps/ --exclude '*.so' --exclude .git/ \
      #     /mnt/vm_8tb/b70/vllm-xpu-kernels-v0240/ /mnt/vm_8tb/b70/vllm-xpu-kernels-nvfp4-sglang/)
      # base image: sglang-xpu:woq (torch 2.12.0+xpu, DPC++ 2025.3), NO GPU needed
      # scope: XPU_SPECIFIC_KERNELS_ENABLED=ON, everything else OFF (GDN OFF -- sglang has its own
      #   triton GDN backend, unlike vLLM which needs the vendored gdn_attention_core .so)
      # output (git-ignored runtime artifact): /mnt/vm_8tb/b70/nvfp4_kernel_sglang/_xpu_C.abi3.so

Build gotcha fixed vs the vLLM recipe: `source setvars.sh` must run in the OUTER `docker -c` shell,
NOT inside a `set -u` script (setvars references unbound vars and aborts under `set -u`).

CPU-verified: `ctypes.CDLL(so, RTLD_GLOBAL)` in sglang-xpu:woq registers `nvfp4_gemm_w4a16` (and the
sibling int8/int4 ops). Op EXECUTION on XPU is untested here (needs the GPU) -- see OPEN.

## C. The shim -- sglang/patches/nvfp4_shim.py (loader + kernel wiring)

Gated on `B70_XPU_NVFP4=1`; auto-installs at import via `sglang/patches/nvfp4_shim.pth`
(`import nvfp4_shim`, a no-op unless the env gate is set). Loads the op like w4a8/w8a8_shim
(`ctypes.CDLL(B70_XPU_C_SO, RTLD_GLOBAL)` after `import torch`). Installs:

0. `get_min_capability -> 0` on ModelOptFp4Config + ModelOptMixedPrecisionConfig (spoof the XPU
   capability gate; the real kernel gates are bypassed by the method overrides).
1. `ModelOptMixedPrecisionConfig._resolve_quant_algo`: normalize `*NVFP4` (i.e. `W4A16_NVFP4`) ->
   `NVFP4` so the MLP routes to `ModelOptFp4LinearMethod` instead of Unquantized.
2. `ModelOptFp4LinearMethod.process_weights_after_loading` / `.apply` -> XPU:
   - load: fold `weight_scale(fp8).float() * weight_scale_2.max()` -> `[K/16,N]` bf16 in the op's NT
     layout; keep `weight` [N,K/2] uint8 resident; free the fp8/fp32 scale params.
   - apply: `nvfp4_gemm_w4a16(x.bf16, weight.t() NT view, bias, wscale_nt, 16)` -> bf16.
     Exact analogue of vLLM `_XPUW4A16NvFp4Kernel` (W4A16 read of the W4A4 ckpt: weights exact-dequant,
     bf16 acts -> equal-or-higher quality than the intended W4A4).
3. `ModelOptFp8LinearMethod.process_weights_after_loading` / `.apply` -> **dequant-at-load to bf16**
   (per-logical-width `weight_scale`) + plain `F.linear`. Conservative + XPU-safe (no cutlass /
   torch._scaled_mm). Attention is a small compute share; native XPU fp8 is an OPEN optimization.

KV cache: the serve script strips `quantization_config.kv_cache_scheme` from a patched `config.json`
mounted RO over the ckpt (bf16 KV), mirroring vLLM `KV_FP8=0` -- fp8 KV is unsupported on sglang-XPU
AND ships no scales.

CPU-verified: `python -m py_compile` clean; op registration + monkeypatch targets all resolve in the
image. Full install path (which imports sglang quant modules) exercised only at serve time.

## D. Serve script -- sglang/nvfp4/serve_nvfp4_27b_sglang.sh

Runtime mounts (NOT a baked image) over `sglang-xpu:mtp`: the built `_xpu_C.abi3.so` + `nvfp4_shim.py`
+ `nvfp4_shim.pth`; in-container `source setvars` + PREPEND oneAPI compiler to `LD_LIBRARY_PATH`
(required or the ctypes .so resolves but torch loses the XPU device -- W4A8_BUILD.md). Knobs:
`GRAPH` (XPUGraph, forces triton attn), `MTP` (NEXTN, ckpt has bf16 mtp.* natively), `RADIX`
(mamba extra-buffer prefix cache), `KVFP8` (default 0 = strip). Defaults = conservative bring-up
(eager, no MTP/graph/radix, bf16 KV) for the most-likely-coherent first start. `--dtype bfloat16`
(the op emits bf16). Does NOT take the GPU lease itself -- wrap in `gpu-run --card 0`.

## DONE vs OPEN

DONE (this session, no GPU):
- [x] Verified upstream sglang has the ModelOpt loader/arch but NO XPU NVFP4/FP8 kernel + the
      W4A16_NVFP4 routing gap.
- [x] Built the sglang-ABI `nvfp4_gemm_w4a16` .so; CPU-verified it loads + registers in sglang-xpu:woq.
- [x] Wrote nvfp4_shim.py (routing + NVFP4 op wiring + FP8 dequant) + .pth autoloader; py_compile clean.
- [x] Wrote the serve script + build script; bash -n clean; KV-strip verified.

OPEN (need the GPU, main session):
- [ ] Actually SERVE it: run the yolo command below; confirm /health + the coherence gate (Rayleigh).
- [ ] Op EXECUTION correctness on XPU (the .so was only load-tested on CPU; the microbench proved the
      op bit-exact on the vLLM image, but re-confirm coherence on this build).
- [ ] FP8-attention-as-bf16 is a numerics choice -- confirm coherence; if a HumanEval+ regression vs
      the vLLM NVFP4 serve appears, revisit (native XPU fp8 or W8A8-int8 attention).
- [ ] GRAPH=1 (XPUGraph) and MTP=1 (NEXTN) are wired but unproven on this quant on sglang -- bring up
      after eager is green (same order the vLLM M5->M9 path took).
- [ ] Perf vs the vLLM NVFP4 27B (38-67 t/s captured+MTP) and vs the sglang int4/w4a8 shelf.

## The GPU yolo test command (main session, coordinated -- needs a free card 0)

    # 0) ensure the sglang-ABI kernel exists (CPU build, ~12 min, no GPU) -- already built this session:
    ls -la /mnt/vm_8tb/b70/nvfp4_kernel_sglang/_xpu_C.abi3.so   # or: bash sglang/nvfp4/build_nvfp4_kernel_sglang.sh

    # 1) conservative bring-up: eager, bf16 KV, no MTP/graph/radix. Coherence-gated, stays up.
    cd /mnt/vm_8tb/github/b70_ai_things
    /mnt/vm_8tb/b70/gpu-run --card 0 bash sglang/nvfp4/serve_nvfp4_27b_sglang.sh start
    #   watch: docker logs sglang_nvfp4_27b | tail ; expect "[nvfp4-shim] NVFP4 layer ready ..." x193
    #   + "[nvfp4-shim] FP8->bf16 layer ready ..." + "GATE OK: 'Rayleigh...'". DBG=1 for per-call NaN checks.
    bash sglang/nvfp4/serve_nvfp4_27b_sglang.sh gen        # one-shot probe
    bash sglang/nvfp4/serve_nvfp4_27b_sglang.sh stop

    # 2) once eager is coherent, stack the levers (one at a time):
    GRAPH=1 /mnt/vm_8tb/b70/gpu-run --card 0 bash sglang/nvfp4/serve_nvfp4_27b_sglang.sh start   # XPUGraph
    MTP=1   /mnt/vm_8tb/b70/gpu-run --card 0 bash sglang/nvfp4/serve_nvfp4_27b_sglang.sh start   # NEXTN MTP
    GRAPH=1 MTP=1 SPEC_STEPS=5 /mnt/vm_8tb/b70/gpu-run --card 0 bash sglang/nvfp4/serve_nvfp4_27b_sglang.sh run  # +bench

If eager emits "!!!!" garbage: check the op executed (DBG=1 -> out_bad=False) and that the FP8-as-bf16
dequant is correct; if the W4A16 op is the suspect, A/B by temporarily pointing B70_XPU_C_SO at the
proven vLLM .so (/mnt/vm_8tb/b70/nvfp4_fused_kernel/_xpu_C.abi3.so). Do a GPU reset (bin/xe-reset)
between failed starts if any TP path was involved (single-card TP=1 here, so usually just `stop`).
