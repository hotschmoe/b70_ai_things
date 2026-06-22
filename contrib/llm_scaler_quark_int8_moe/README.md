# Quark W8A8 INT8 MoE (Qwen3.6-35B-A3B) on 2x B70 -- WORKING on vLLM 0.23

Goal: run the Quark **W8A8 INT8** MoE serve (Qwen3.6-35B-A3B, ckpt
`nameistoken/Qwen3.6-35B-A3B-Quark-W8A8-INT8`) on **our 2x B70** at **TP=2**.

## [SOLVED 2026-06-22] It serves + generates on `vllm-xpu-env:v0230` (vLLM 0.23.0)
Use **`scripts/76_quark35b_v0230.sh`** + **`v0230/quark.py`** (one bind-mounted file).
NOT the llm-scaler 0.14.x image (dead end -- see below). Why v0230 wins:
- vLLM **0.23 already ships** `QuarkW8A8Int8MoEMethod` AND a dynamic-per-token int8 LINEAR
  dispatch (`_is_dynamic_per_token_w8a8`). So **no MoE patch and no linear-dispatch patch**.
- v0230 routes the 256 int8 experts through the **Triton `fused_moe_kernel`** on XPU (the same
  path our int4 MoE uses -- `contrib/vllm_moe_xpu`), so `_moe_C` being unbuilt does NOT matter.
- The ONE gap left: XPU has no int8 scaled-mm LINEAR kernel (`_POSSIBLE_KERNELS` KeyErrors), so
  `v0230/quark.py` reroutes ONLY the int8 linear layers (linear_attn.*, mlp.shared_expert.*) to a
  weight-only int8->bf16 **dequant** GEMM (`QuarkW8A8Int8DequantXPU`). Experts stay TRUE int8.

Verified (TP=2, enforce-eager): model load 17.54 GiB/card, KV 10.2 GiB, concurrency 89x@8192,
`backend=xccl world_size=2`, Triton `fused_moe_kernel` (E=256,N=256, int8),
gen "The capital of France is" -> " Paris, a city renowned for its rich history, culture, and iconic
landmarks." Next perf lever: graph capture (eager only so far; PIECEWISE gave +617% on the int4 MoE).

---

## Background: the llm-scaler 0.14.1 attempt (DEAD END -- kept for the diagnosis)

Original target: steveseguin's accepted Quark W8A8 INT8 run (his 4x B70 = 99.77 tok/s) on the
on-host `intel/llm-scaler-vllm:0.14.0-b8.3.1` image, via `scripts/74_quark35b_bench.sh`.

These two files are PATCHED copies of the image's vendored vLLM modules
(`vllm 0.14.1.dev0+gb17039bcc.d20260605`), bind-mounted over the originals at serve
time (no image rebuild). Re-extract + re-apply if the image version changes.

| file | image path it overrides | what we added |
|---|---|---|
| `quark_moe.py` | `.../quantization/quark/quark_moe.py` | `QuarkW8A8Int8MoEMethod` + int8 dispatch branch (stock only wires fp8-w4a8 / fp8-w8a8 / ocp-mx -> `raise "Unsupported FusedMoe scheme"` on int8). Mirrors `CompressedTensorsW8A8Int8MoEMethod` (same image) using Quark's dict config + 1-dim per-channel scales. |
| `quark.py` | `.../quantization/quark/quark.py` | `QuarkW8A8Int8DequantXPU` linear scheme + int8 dispatch branch. The image has NO XPU int8 scaled-mm kernel (`_POSSIBLE_KERNELS` = CPU/CUDA/ROCM only), so the stock `QuarkW8A8Int8` linear path can't run; we dequant int8 per-channel weights -> bf16 and run a plain GEMM (effectively W8A16; activations not quantized). Used only for the minority linear layers (`linear_attn.*`, `mlp.shared_expert.*`); the 256 routed experts stay true int8. |

## Why int8 needs patching at all (the checkpoint is GLOBAL int8)
`config.json` `quantization_config.global_quant_config`: W = int8 per-channel symmetric,
IN = int8 per-channel **dynamic**; `layer_quant_config` empty (only the vision tower is
excluded, dropped by `--language-model-only`). So every linear AND every MoE expert is
dynamic-per-token int8 -- a scheme the 0.14.1 image's Quark path never matches.

## Blocker chain (all CLEARED by the patches + `scripts/74` env), in order hit
1. **Inspect subprocess SYCL abort** "No device of requested type available" -- steve's env
   double-pins `ONEAPI_DEVICE_SELECTOR`+`ZE_AFFINITY_MASK` (his 4-card values). Fix: expose
   both cards, no pin (our proven TP>1 path).
2. **oneCCL `zeMemOpenIpcHandle ... ZE_RESULT_ERROR_INVALID_ARGUMENT`** at TP=2 collective.
   Fix: our Battlemage multi-GPU stability env (vLLM #41663): `CCL_TOPO_P2P_ACCESS=0`,
   `CCL_ZE_IPC_EXCHANGE=pidfd`, `CCL_ENABLE_SYCL_KERNELS=0`, `SYCL_UR_USE_LEVEL_ZERO_V2=0`.
3. **`RuntimeError: Unsupported FusedMoe scheme`** (MoE) -> `quark_moe.py` patch.
4. **`NotImplementedError: No quark compatible scheme was found`** (linear) -> `quark.py` patch.
5. **`Inference tensors do not track version counter`** (torch.compile) -> dequant under
   `inference_mode(False)` + `--enforce-eager` (B70 TP=2 capture is blocked anyway).

After these, the model **fully constructs, the TP=2 collective comes up (`backend=xccl,
world_size=2`), and all 7 shards load**.

## The REMAINING blocker (NOT fixable on this image) -- status: BLOCKED
6. **`AttributeError: '_OpNamespace' '_moe_C' object has no attribute 'topk_softmax'`** during
   the first eager MoE forward. `vllm._moe_C` **does not exist** in this image (the compiled
   MoE op suite -- routing `topk_softmax` AND the int8 fused-expert GEMMs -- was not built),
   and `vllm_topk_softmax` has no native fallback. `VLLM_XPU_USE_LLM_SCALER_MOE` is also not
   honored here. So generic int8 fused-MoE simply cannot EXECUTE on XPU in 0.14.1.

steve's 99.77 t/s ran on **vLLM 0.20.2rc1.dev2** -- a newer llm-scaler build that HAS the
XPU MoE kernels. **Finish path:** pull a newer `intel/llm-scaler-vllm` tag (steve's ~0.20.x)
that ships `_moe_C` / the XPU fused-MoE kernels, then re-run `scripts/74_quark35b_bench.sh`
(the patches above stay relevant only if that newer image still lacks the int8 Quark MoE
dispatch -- check first; upstream vLLM >=0.20 likely already has `QuarkW8A8Int8MoEMethod`).
