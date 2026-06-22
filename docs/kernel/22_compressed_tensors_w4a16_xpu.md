# compressed-tensors W4A16 on XPU (vLLM 0.23) -- serving a TEXT-ONLY Qwen3.5-27B quant

Investigation log for fixing `Qwen3.6-27B-W4A16` (compressed-tensors `pack-quantized`, int4 weight / 16-bit
act) so it serves on the B70. Goal: keep ALL models in compressed-tensors format for parity (and as the
substrate for future W4A4 research -- W4A4 = compressed-tensors int4-weight + int4-act). Status: IN PROGRESS.

Image: `vllm-xpu-env:v0230` (has the GDN/gated-delta-net kernel the Qwen3.5 hybrid LM needs). All work on
ONE card (`gpu-run --card 0`), leaving the other free. Recipe dir: `rdy_to_serve/qwen36-27b-w4a16/`.

## The checkpoint is LANGUAGE-MODEL-ONLY (key fact)
`config.json`: `architectures=["Qwen3_5ForCausalLM"]`, `model_type=qwen3_5` (with nested `text_config`
`qwen3_5_text`). ALL 1363 tensors are `model.language_model.*` + `lm_head` -- **zero vision tensors**.
The WORKING daily-driver 27B (`Lorbus_..-int4-AutoRound`) is the FULL VL model
(`Qwen3_5ForConditionalGeneration`, has vision weights). So this W4A16 quant dropped the vision tower.

`ignore` list = only `lm_head`. group_size 128, num_bits 4, symmetric int -> vLLM weight_type `uint4b8`.

## FOUR structural blockers (all FIXED) -- loaded as text-only, served HEALTHY
vLLM kept resolving the checkpoint to the VL class and building a (weightless) vision tower, then failed in
stages. Fix = a `sitecustomize` + module shim (`patches/`, on PYTHONPATH), pinned to vLLM 0.23:

1. **Vision tower built -> 4304-dim WNA16 assert.** The registry only maps the VL
   `Qwen3_5ForConditionalGeneration`; `_normalize_arch` suffix-maps our unregistered `...ForCausalLM` onto
   it (-> builds `self.visual` unconditionally; its MLP `linear_fc2` has input 4304, 4304%32!=0 AND
   4304%128!=0, so `compressed_tensors_wNa16.create_weights` asserts -- and there are no vision weights
   anyway). FIX: `ModelRegistry.register_model("Qwen3_5ForCausalLM", <text class>)` -- the real text class
   `qwen3_5:Qwen3_5ForCausalLM` exists (it is the VL model's own `.language_model`). Now resolves text-only,
   never builds the vision tower.

2. **`assert mamba_block_size is not None`** (mamba/abstract.py). The text class
   `Qwen3_5ForCausalLMBase` is NOT `IsHybrid` (bases: nn.Module, HasInnerState, SupportsEagle3, SupportsLoRA,
   SupportsPP), so `model_config.is_hybrid` is False and the GDN/mamba KV-cache setup is skipped.
   `is_hybrid(model) == getattr(model, "is_hybrid", False)` and IsHybrid is a runtime_checkable Protocol,
   so FIX: set `is_hybrid = True` on the registered subclass.

3. **`AttributeError: ... has no attribute get_mamba_state_shape_from_config`** (interface.py
   `_align_hybrid_block_size`). The GDN state shape/dtype/copy classmethods live on the VL wrapper, NOT the
   text class. They compute purely from `vllm_config` (cls unused). FIX: graft them onto the subclass
   (`classmethod(_VL.get_mamba_state_shape_from_config.__func__)` etc.).

4. **`assert supports_mrope(model)`** (gpu_model_runner `_init_mrope_positions`). The shared (VL) config
   declares M-RoPE; the text decoder uses standard 1D RoPE, but vLLM still routes through mrope position
   prep. FIX: set `supports_mrope = True` + a text-only `get_mrope_input_positions` returning
   `arange(n)` broadcast to `[3, n]`, delta 0 -- VERIFIED identical to the VL text path
   (`np.broadcast_to(np.arange(text_len), (3, text_len))`, delta = max_pos+1-len = 0).
   NOTE: trying to DISABLE mrope by stripping `rope_scaling['mrope_section']` in a MODELS_CONFIG_MAP hook
   did NOT work -- the hook is registered too late (uses_mrope already cached). Granting support is robust
   (read off the class at model build).

## OPEN: serves HEALTHY but generates garbage ("!!!!")
A numerical-correctness bug, not structural. mrope ruled out (matches VL exactly). Suspects:
- (a) the stock XPU WNA16 kernel `XPUwNa16` (`torch.ops._xpu_C.int4_gemm_w4a16`) -- its
  `process_weights_after_loading` has a gptq-marlin/compressed-tensors shared transpose dance keyed on a
  shape comparison; a layout mismatch there would corrupt the GEMM.
- (b) the GDN/hybrid setup via the grafted methods differing from the native VL path.

DEAD END: forcing our explicit dequant kernel (`contrib/vllm_wna16_xpu/xpu_wna16_dequant.py`, int4->bf16 +
dense GEMM) for ALL layers OOMs -- int4->bf16 is ~4x weight memory (~13.5 GiB int4 -> ~54 GiB bf16), does
not fit one 32 GB card. So the int4 path must be fixed IN PLACE.

ISOLATION RESULT (2026-06-23): the WORKING Lorbus 27B (auto_round) does **NOT** use `XPUwNa16` --
its serve log shows `inc.py:619 Successfully imported auto_round_kernel` (Intel Neural Compressor
auto_round int4 path) + `Using Triton/FLA GDN prefill kernel`. So:
- Lorbus linear = INC `auto_round_kernel`; GDN = Triton/FLA -> coherent.
- Our W4A16 linear = compressed-tensors -> `XPUwNa16` (`int4_gemm_w4a16`) -> garbage.
=> `int4_gemm_w4a16` (the compressed-tensors W4A16 XPU GEMM) is exercised by NO working model and is the
prime suspect. GDN is NOT the bug (Lorbus's GDN works; ours uses the same Triton/FLA path). mrope ruled out.

NEXT: a numerical unit test of `torch.ops._xpu_C.int4_gemm_w4a16` vs a reference int4 dequant-matmul, on a
small known weight, sweeping the weight/scale layout (the `process_weights` transpose dance) -- to confirm
the op is wrong and/or find the layout it actually expects. This is one-card (`gpu-run --card 0`) friendly.

VIABLE FALLBACK (if the op can't be fixed): our `xpu_wna16_dequant.py` (dequant int4->bf16 + dense GEMM) is
CORRECT-by-construction but ~4x weight memory; it does not fit ONE 32 GB card for the 27B (~54 GiB bf16) ->
would need TP=2 (both cards, ~27 GiB/card). Keep int4-in-place (XPUwNa16 fix) as the one-card goal.

[!] MULTI-AGENT NOTE: this box may have another agent on card 1. ALWAYS `gpu-run --card 0` for this work
(never the default, which locks BOTH cards). Do not touch card 1 / gpu.lock.1.
