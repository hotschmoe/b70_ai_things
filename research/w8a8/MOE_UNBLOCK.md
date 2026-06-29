# MOE_UNBLOCK -- W8A8 int8 of the 35B-A3B MoE: the real state + the routes to sglang

Date: 2026-06-29. Synthesis of two code-level investigations + a reconciliation of docs/kernel/18
and /20. Goal: get **W8A8 int8 serving of Qwen3.6-35B-A3B (256 experts, top-8)**, ideally on sglang
(the production backend; vLLM is paused). Standing target: "W8A8 int8 of ANY model" (AGENTS.md).

## 0. TL;DR -- corrected reality

- **W8A8 MoE already WORKS on vLLM today.** `rdy_to_serve/vllm/qwen36-35b-a3b-w8a8` (image
  `vllm-xpu-env:v0230` + `patches/quark.py`) just benched **43.1 t/s c1 / 22.2 c4, coherent** on our
  2x B70 (TP=2). The 256 experts run TRUE int8 via the STOCK v0230 `quark_moe.py`
  `QuarkW8A8Int8MoEMethod`; only the dense `linear_attn.*`/`shared_expert.*` linears use a load-time
  dequant fallback (`patches/quark.py` `QuarkW8A8Int8DequantXPU`). So the int8-XMX MoE target is MET --
  on the paused backend.
- **llm-scaler is a DEAD END (correct the doc-20 header).** The pullable image
  `intel/llm-scaler-vllm:0.14.0-b8.3.1` never built the XPU int8 fused-MoE op (`vllm._moe_C` missing ->
  `topk_softmax` AttributeError on the first MoE forward; doc 20 sec 8). steve's 99.77 t/s ran on an
  UNPUBLISHED source build (vLLM 0.20.2rc1.dev2 + llm-scaler ESIMD kernels). Intel does not list int8
  W8A8 as a supported MoE quant. Nothing to pull. Do NOT chase it.
- **sglang has no W8A8 MoE yet -- BUT the port likely needs NO custom kernel.** The vLLM int8 MoE
  GEMM is the **pure-Triton `fused_moe_kernel` `use_int8_w8a8` path** (no C++/SYCL op in the hot path),
  and **sglang ships the same `fused_moe_triton` with that kernel**. So the sglang port = a LOADER +
  routing, not a kernel build.

## 1. The three routes to sglang W8A8 MoE (easiest -> hardest)

### Route A -- Triton `use_int8_w8a8` (RECOMMENDED; likely no custom kernel)
The vLLM expert kernel is platform-agnostic Triton: `fused_moe/fused_moe.py:293` `fused_moe_kernel`
with the `use_int8_w8a8` constexpr (`:342`; int8 `tl.dot`->int32, dequant by `A_scale*B_scale`
`:450,493,522`); per-token activation int8 quant is done dynamically INSIDE the kernel
(`moe_kernel_quantize_input`). sglang has its own `fused_moe_triton` with the same kernel. Port =
1. A sglang `Int8MoEMethod(FusedMoEMethodBase)` hooking in exactly like `AWQMoEMethod`
   (`sglang/patches/awq.py:442`, `get_quant_method` at `:150/:199/:350`, `create_moe_runner` `:471`).
2. Load int8 expert weights + per-channel scales into sglang's `EPMoE/FusedMoE` layout:
   `w13_weight [E,2I,H] int8`, `w2_weight [E,H,I] int8`, per-channel `w13/w2_weight_scale [E,N]`
   reshaped `[E,N,1]`. Activation scale = None (dynamic per-token). Mirror
   `quark_moe.py QuarkW8A8Int8MoEMethod` (the live v0230 one at
   `/mnt/vm_8tb/b70/build/vllm/vllm/.../quark/quark_moe.py:518-815`).
3. Route experts to sglang's int8 `fused_experts` (`use_int8_w8a8=True`,
   `per_act_token_quant=True`).
4. Dense int8 linears (`linear_attn.*`, `mlp.shared_expert.*`): load-time dequant -> bf16 plain GEMM
   (sglang has no XPU int8 scaled-mm either; mirror `QuarkW8A8Int8DequantXPU`).
- **The one risk to validate first**: does sglang's `fused_moe_triton` `use_int8_w8a8` path actually
  run on triton-xpu (B70)? If yes, this is the whole job. If the Triton int8 kernel mis-codegens on
  XPU, fall to Route C.

### Route B -- per-expert dense loop (correctness-first stopgap; not for prefill perf)
Loop the existing dense oneDNN op `kernels/int8_gemm_w8a8.h` (`dnnl_matmul_w8a8_int8`) per active
expert: gather tokens -> dense int8 GEMM -> scatter. **Correctness PROVEN**: `cosine 0.99992`,
`rel_err 1.3e-3` on the real 35B expert shapes (`scripts/int8_moe_grouped_test.py`). Hooks into the
same `Int8MoEMethod.apply` (replace the `XpuFusedMoe` call). BUT eager is dispatch-bound (~1280
launches/iter -> NO int8 win); only servable if PIECEWISE / `B70_XPU_CUDAGRAPH` capture amortizes the
launches. Use as a correctness oracle / fallback, not the production path.

### Route C -- fused SYCL grouped-GEMM (the real prefill win; hardest)
Add an `is_w8a8`/`is_B_int8` flag to `cutlass_grouped_gemm_interface`
(`sglang/_v0230_kernels/vllm_xpu_kernels/fused_moe_interface.py:485-521`; `_get_recipe` `:32-44` has
NO int8 branch today): an `s8 x s8 -> s32` MMA atom + per-token activation-scale epilogue on
`ptr_scales`. The C++ op lives in the out-of-repo `vllm-xpu-kernels` SYCL submodule; the building
blocks exist in `intel/sycl-tla` (`04_bmg_grouped_gemm`, the `XE_8x16x32_S32S8S8S32_TT` DPAS atom --
doc kernel/10:541). "W8A8 is the single missing flag" (doc 18). Real prefill win 1.4-2.0x; does NOT
help memory-bound decode (W4A16 stays the decode recipe). Build only if Route A's Triton perf is poor.

## 2. Recommendation + phased plan

1. **Probe Route A feasibility (no serve, ~mins):** confirm sglang's in-tree `fused_moe_triton` has
   the `use_int8_w8a8` kernel and that triton-xpu compiles it on B70 (a tiny offline triton int8-MoE
   microbench, like `scripts/int8_moe_grouped_test.py` but on sglang's Triton kernel). This is the
   whole go/no-go for the easy path.
2. **Write the loader (Route A):** `sglang/patches/quark_moe_int8.py` -> `Int8MoEMethod` +
   `Int8DequantLinear`, mirroring the v0230 `quark_moe.py`/`quark.py` references above. Also handle the
   compressed-tensors W8A8 MoE config (we can produce that with llmcompressor; avoids needing the Quark
   quantizer, which we do not have locally).
3. **Serve + gate (TP=2, ATTENDED -- wedge risk):** the 35B int8 (~35GB) splits ~17.5GB/card -> fits
   TP=2. Coherence-gate, then bench vs the existing vLLM Quark 43.1 t/s and the 35B W4A16 (56.8 t/s).
   Greedy-only on XPU; vision excluded by the quant (text-only serve, or graft visual like the dense path).
4. **Only if Triton is slow:** Route C (fused SYCL `is_w8a8`).

## 3. Open questions / risks

- **Does the vLLM Quark MoE survive SUSTAINED concurrent load?** The bench (short) passed, but the 35B
  has GDN `linear_attn.*` layers -- the same path that gives the dense models the `!!!!` NaN poison
  (SHORTCOMINGS / vLLM #38994). A long soak under c4+ is needed before calling the vLLM entry
  production. If it `!!!!`s, that is another reason to want the sglang path.
- **triton-xpu int8 codegen** on B70 is the Route-A gating unknown.
- **No local Quark quantizer** (`import quark` absent) -- use the existing
  `nameistoken/...Quark-W8A8-INT8` ckpt (we have it) or produce compressed-tensors W8A8 with llmcompressor.

## 4. Reference map (file:line)

- sglang dispatch: `sglang/patches/qwen3_5.py:60,599,815,1439,1461`; recipe gap
  `sglang/_v0230_kernels/vllm_xpu_kernels/fused_moe_interface.py:32-44,485-521`; quant_method hook
  `sglang/patches/awq.py:150,199,350,442,471`.
- vLLM live int8 MoE (the port reference): `/mnt/vm_8tb/b70/build/vllm/vllm/.../quark/quark_moe.py:116-122,518-815`;
  Triton kernel `.../fused_moe/fused_moe.py:293,342,450,493,522`; config helper `.../fused_moe/config.py:630-653`;
  linear fallback `rdy_to_serve/vllm/qwen36-35b-a3b-w8a8/patches/quark.py:265-273,754-771`.
- dense int8 op (Route B): `kernels/int8_gemm_w8a8.h:18-25`; proof `scripts/int8_moe_grouped_test.py`.
- Quark config: `models/files/qwen3.6-35b-a3b/quark-w8a8-int8/config.json` (global int8, W per-channel
  sym static, IN per-channel dynamic; 152 excludes = visual + shared_expert_gate + lm_head).
- superseded background: `docs/kernel/18` (build-the-kernel; moot for serve), `docs/kernel/20` (llm-scaler;
  header optimistic, sec 7-8 = the dead-end correction).
