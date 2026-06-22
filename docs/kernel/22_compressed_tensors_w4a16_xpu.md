# compressed-tensors W4A16 on XPU (vLLM 0.23) -- serving a TEXT-ONLY Qwen3.5-27B quant

Investigation log for fixing `Qwen3.6-27B-W4A16` (compressed-tensors `pack-quantized`, int4 weight / 16-bit
act) so it serves on the B70. Goal: keep ALL models in compressed-tensors format for parity (and as the
substrate for future W4A4 research -- W4A4 = compressed-tensors int4-weight + int4-act).

## STATUS: RESOLVED 2026-06-23 -- serves COHERENTLY on :v0230, one card.
Root cause was a weight-name prefix mismatch (all weights silently skipped -> random init -> "!!!!"
garbage), NOT the int4 kernel. Fix = a `load_weights` remap (`model.language_model.` -> `model.`) on the
registered text subclass, on top of the four structural fixes below. After the remap: skipped-weight
warnings 0; gen "The capital of France is Paris... most populous city of France"; "The ocean is a vast,
mysterious expanse of saltwater that covers more than 70% of the Earth's surface". The full fix lives in
`rdy_to_serve/qwen36-27b-w4a16/patches/` (sitecustomize.py + qwen35_text_hybrid.py). The XPUwNa16 int4
kernel was EXONERATED (see below). Below: the full debugging chain (kept for the research record).

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

## ROOT CAUSE FOUND + the linear kernel EXONERATED (2026-06-23)
Two numerical unit tests of `torch.ops._xpu_C.int4_gemm_w4a16` on `--card 0` (vs a reference int4 dequant
matmul) PROVED the op + the XPUwNa16 layout are CORRECT:
- the op requires the weight in **NT format** -- pass `weight_packed.t()` as a NON-contiguous VIEW (a
  `.contiguous()` there raises `RuntimeError: Int4 weight must be in NT format!`), with `scale.t().contiguous()`
  -> `[n_groups, N]`. This is exactly what `XPUwNa16.apply` does.
- synthetic (N=K=256, g128, uint4b8): maxerr 0.0156 [MATCH]. REAL layer (down_proj N=5120 K=17408 g128):
  maxerr 0.0156 [MATCH]. So compressed-tensors int4 W4A16 computes correctly on XPU. NOT the bug.

THE ACTUAL BUG = **weight-name prefix mismatch -> ALL weights silently skipped -> model runs on RANDOM
init -> garbage**. Serve log (`qwen3_5.py:420`):
```
WARNING Parameter language_model.embed_tokens.weight not found in params_dict, skip loading
WARNING Parameter language_model.layers.0.mlp.down_proj.weight_packed not found in params_dict, skip loading
... (every layer/param)
```
The checkpoint was quantized as the VL model's `.language_model`, so its keys are `model.language_model.*`
(+ `lm_head.weight`). The VL class loads fine (it HAS `self.language_model`); the standalone text class's
params are `model.*`, so the `language_model.` segment never matches and every weight is skipped (vLLM only
WARNS, does not error -> it serves HEALTHY on random weights). FIX: a weights remap on the registered text
subclass that strips the `language_model.` segment (`model.language_model.` -> `model.`). [implementing]

LESSON (for research): "serves HEALTHY + coherent-looking infra logs" != "weights loaded". Always grep the
load for `not found in params_dict` / `skip loading` when bringing up a re-homed checkpoint. The
all-same-token ("!!!!") degenerate output is the classic random-weights signature (cf. the Q8 false positive).

## PERF: compressed-tensors W4A16 vs AutoRound int4 (2026-06-23, ctx=2048, GRAPH=1, card 0)
Both 27B int4-weight, DIFFERENT kernels: w4a16 = compressed-tensors -> `int4_gemm_w4a16` (oneDNN/XMX);
int4 = Lorbus AutoRound -> INC `auto_round_kernel` + Triton/FLA GDN. `vllm bench serve` random in2048/out128,
warmup then measured. pp = C*2048/(TTFT/1000); tg = per-stream decode = 1000/TPOT.
```
  model                          C   pp(t/s)  TTFT(ms)  TPOT(ms)  tg(t/s)  agg_out(t/s)
  w4a16 (CT, int4_gemm_w4a16)    1   1676     1221.7    47.69     20.97    17.59
  int4  (AutoRound, INC)         1   1573     1302.0    33.50     29.85    23.04
  w4a16 (CT)                     4   2510     3263.5    63.57     15.73    45.12
  int4  (AutoRound)              4   2364     3464.8    51.73     19.33    50.99
```
TAKEAWAY: the two int4 paths trade off OPPOSITELY. DECODE (tg): AutoRound wins big -- +42% at C1
(29.85 vs 20.97), +23% at C4. PREFILL (pp/TTFT): compressed-tensors wins slightly (1676 vs 1573, TTFT
1222 vs 1302 at C1). So `int4_gemm_w4a16` is the better GEMM (prefill, compute-bound) but the slower GEMV
(decode, batch=1 memory-bound) -- a clear int8/int4 GEMV optimization target. Net: AutoRound is the better
GENERATION pick (decode-bound) -> stays the daily driver; compressed-tensors W4A16 is the parity/research
baseline and is competitive on prefill.
