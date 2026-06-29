# vllm_moe_xpu -- int4 MoE (W4A16) on Intel Arc Pro B70

Makes **Qwen3.6-35B-A3B int4 AutoRound (256-expert MoE)** load and generate on ONE B70.
The fix is a ~16-line routing patch to vLLM's Intel Neural Compressor (INC) quant integration --
no new kernel. Verified on `vllm-xpu-env:v0230` (vLLM **0.23.0+xpu**), 2026-06-20.

## The bug

`auto-round` checkpoints map to `INCConfig` (`vllm/model_executor/layers/quantization/inc.py`).
On XPU, `INCConfig.get_quant_method()` sends **every** layer to `apply_xpu_w4a16_quant_layer()`,
which only handles `LinearBase` / `ParallelLMHead`. For a `RoutedExperts` (FusedMoE) layer it falls
through to `return None`.

A `None` quant method makes the `FusedMoE` layer fall back to `UnquantizedFusedMoEMethod`, which holds
experts in **bf16**. So the 256 int4 experts dequantize toward ~70 GB and the model
`OUT_OF_DEVICE_MEMORY`s at weight load on a 32 GB card -- even though the int4 weights are only ~21.5 GB
on disk / 19.6 GiB packed in VRAM.

On CUDA and CPU the *same* INC code routes `RoutedExperts` -> `MoeWNA16Config` (int4-preserving). The XPU
branch just never implemented it. (Intel's `llm-scaler-vllm:0.14.x` is worse: its INC returns
`UnquantizedFusedMoEMethod` for *any* MoE.) Net: no stock Intel image serves this model unquantized-free;
you must patch.

## The fix

Add the missing `RoutedExperts` branch to `apply_xpu_w4a16_quant_layer`, mirroring the proven gptq path
(`apply_gptq_quant_layer`): present the experts as a gptq config and delegate to `MoeWNA16Config`.

```python
# in apply_xpu_w4a16_quant_layer(...), after the LinearBase/ParallelLMHead block, before `return None`:
if isinstance(layer, RoutedExperts):
    from vllm.model_executor.layers.quantization.moe_wna16 import MoeWNA16Config
    config = {
        "quant_method": "gptq",   # packing_format is auto_round:auto_gptq + sym
        "bits": weight_bits,
        "group_size": group_size,
        "sym": sym,
        "lm_head": False,
    }
    return MoeWNA16Config.from_config(config).get_quant_method(layer, prefix)
```

`RoutedExperts` is already imported at the top of `inc.py`. Full patched file: **`inc.py`** here.

### Why no CUDA kernel is needed

`MoeWNA16Method.apply` -> `fused_experts` -> `dispatch_fused_moe_kernel`. For int4_w4a16 with a group size,
it calls `should_moe_wna16_use_cuda()`, which is `return current_platform.is_cuda() and ...` -> **False on
XPU** -> it dispatches to the pure-Triton `invoke_fused_moe_wna16_triton_kernel` (`fused_moe_kernel_gptq_awq`).
Triton is live on Battlemage (the model's Gated-DeltaNet attention already uses Triton/FLA), so the int4 MoE
GEMM JIT-compiles and runs. No CUDA-only op, no Marlin.

## Apply it (no image rebuild)

Bind-mount the patched file over the image's copy at run time:

```bash
-v /path/to/inc.py:/opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/quantization/inc.py:ro
```

End-to-end load + generate test: **`scripts/53_loadtest_35b_moe_xpu.sh`** (run via
`scripts/runremote.sh`; wraps the GPU touch in `gpu-run`).

## Verified result (2026-06-20, one B70)

```
quantization=inc, arch Qwen3_5MoeForConditionalGeneration
Model loading took 19.6 GiB            <- int4 experts PACKED (was ~70 GB bf16 -> OOM)
Available KV cache 6.24 GiB / 122,880 tokens / 60.00x concurrency @ 2048
fused_moe: E=256,N=512,device=Intel(R)_Graphics_[0xe223],dtype=int4_w4a16
Triton JIT: fused_moe_kernel_gptq_awq
GENERATION OK -> "The capital of France is" -> " Paris, a city renowned for its rich history, ..."
~6 t/s decode (single-stream, eager)
```

## Not done yet (load proof, not optimized)

- **Perf:** no tuned `E=256,N=512,...,dtype=int4_w4a16.json` -> "default MoE config ... sub-optimal" warning;
  ~6 t/s eager is a correctness/load proof. Expert-config tuning is still open.
  - **[RESOLVED 2026-06-20] PIECEWISE graph capture = +617% (7.17x): 7.93 -> 56.84 t/s** (served, same probe;
    the biggest capture win on the B70). Bake `:v0230moe` (= `:v0230` + this `inc.py`) and serve with
    `w4a8/30_serve_w4a8_graph.sh GRAPH=1` (its `pass_config` fix dodges the XPU compile `NameError`). The masked
    `fused_moe_kernel_gptq_awq` is routing-agnostic, so the captured graph is correct (verified coherent). The
    MoE was the MOST eager-dispatch-bound config (256-expert routing + GDN), so it gains the most. See JOURNAL
    2026-06-20 / FINDINGS / SUMMARY.
- **Accuracy eval** (gsm8k / HumanEval+): not run.
- **MTP / shared-expert** paths: the checkpoint ships an `mtp.*` Multi-Token-Prediction module (ignored here)
  and bf16 shared experts (the 240 `extra_config` bits:16 exceptions, loaded unquantized -- correct).
- **Bake into an image** (vs bind-mount) if this becomes a standing serve target.
