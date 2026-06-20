# AutoRound W4A8 feasibility -- 27B dense (GDN) + 35B-A3B MoE on one B70

Date: 2026-06-20. Author: quant-eng (read-only investigation; NO GPU touched).
Goal under test: convert the two flagship W4A16 int4-AutoRound models to **W4A8**
(int4 sym group weights + per-token dynamic int8 activations) via AutoRound, to keep
the fast captured int4 decode AND gain prefill/TTFT off the int8-XMX systolic path
(on the 14B w4a8 was +51% prefill / ~-32% TTFT vs w4a16).

================================================================================
VERDICT (per model)
================================================================================

  +----------------+-----------+------------------------------------------------+
  | Model          | Verdict   | Why                                            |
  +----------------+-----------+------------------------------------------------+
  | 27B dense (GDN)| GO        | All linears (attn + GDN in/out_proj + MLP)     |
  |                | (NEEDS    | route to XPUW4A8IntLinearKernel (real XPU      |
  |                |  recipe   | int4_gemm_w4a8 op). bf16 source on host.       |
  |                |  fix-up)  | BUT: the repo's 10_*.sh uses a DEAD AutoRound  |
  |                |           | API -- see "Recipe fix" below before running.  |
  +----------------+-----------+------------------------------------------------+
  | 35B-A3B MoE    | NO-GO     | (1) NO XPU W4A8-int8 MoE kernel exists in vLLM |
  |                | (tonight) | (CPU-only, raises NotImplementedError on XPU). |
  |                |           | (2) No bf16 source on the host (only the int4  |
  |                |           | dir; AutoRound needs full precision).          |
  +----------------+-----------+------------------------------------------------+

================================================================================
[CRITICAL] Does an int8-ACTIVATION MoE path exist on XPU?  -> NO. (VERIFIED)
================================================================================

VERIFIED in image vllm-xpu-env:v0230moe (vLLM 0.23.0+xpu),
`vllm/model_executor/layers/fused_moe/oracle/w4a8_int8.py`:

  - line 38 comment:  "Currently only CPU INT4 backend is available for W4A8 INT8 MoE."
  - `_get_priority_backends()`:
        if current_platform.is_cpu():  return [W4A8Int8MoeBackend.CPU_INT4]
        return []                          # <-- XPU/CUDA get an EMPTY list
  - `select_w4a8_int8_moe_backend()` line 93:
        raise NotImplementedError("W4A8 Int8 MoE is only supported on CPU platforms")
  - the ONLY kernel class is `CPUExpertsInt4`, packing via
        torch.ops.aten._dyn_quant_pack_4bit_weight   (an aten CPU/ARM op, not SYCL)

So the class `CompressedTensorsW4A8Int8MoEMethod` EXISTS in the source tree, but its
backend oracle returns nothing on XPU and **raises at weight-load**. A W4A8-int8 MoE
checkpoint (int4 experts + int8 dynamic act) therefore has NO fused-expert fastpath on
the B70 and would crash on serve (it would NOT silently fall back to fp16 -- it raises).
Codex independently confirmed this reading.

Contrast: the W4A16 MoE path that the 35B uses TODAY is fine -- `MoeWNA16Config` ->
Triton `fused_moe_kernel_gptq_awq` (int4_w4a16, **fp16 acts**), wired by the
`inc.py` RoutedExperts patch (contrib/vllm_moe_xpu). That path has no int8 activations,
so it gives NO int8-XMX prefill win. There is no int8-act fused-expert kernel for Xe2.

=> The 35B's hoped-for W4A8 prefill fastpath DOES NOT EXIST. 35B W4A8 is a NO-GO until
   someone writes an XPU int4w x int8act fused-expert SYCL kernel (large effort; out of
   scope for an overnight quant). Do NOT spend GPU hours quantizing the 35B to W4A8.

================================================================================
The 27B dense path -- GO  (VERIFIED the kernel + arch; the RECIPE needs a fix)
================================================================================

WHY IT WORKS (all VERIFIED on the host images):
  - 27B arch = `Qwen3_5ForConditionalGeneration`, model_type qwen3_5 (dense hybrid:
    17 full_attention + 48 linear_attention/GDN layers, full_attention_interval=4).
  - Dense kernel is REAL on XPU: `XPUW4A8IntLinearKernel`
    (vllm/model_executor/kernels/linear/mixed_precision/xpu.py:125) -- its
    can_implement requires current_platform.is_xpu() and it calls
    `torch.ops._xpu_C.int4_gemm_w4a8` (line 219). Proven on the 14B w4a8-gptq.
  - GDN linears ARE quantized: in qwen3_5.py the GatedDeltaNet `in_proj_qkvz`,
    `in_proj_ba`, `out_proj` all receive `quant_config` (lines ~480/581/795), so they
    route through the same W4A8 kernel. GDN's non-linear state ops (conv1d, A_log,
    dt_bias, the delta recurrence) stay fp/bf16 -- correct, they are not GEMMs and are
    not touched by W4A8 (same as in the W4A16 int4 model that already serves).
  - The compressed-tensors scheme picker has a dedicated W4A8-int8 detector
    (compressed_tensors.py ~500-511: weight 4-bit + act 8-bit + TOKEN strategy +
    dynamic -> CompressedTensorsW4A8Int dense scheme). This matches an AutoRound
    W4A8 (sym int4 group + dynamic per-token int8 act) export.
  - bf16 source EXISTS on host: `/mnt/vm_8tb/b70/models/Qwen_Qwen3.6-27B` (72 GB,
    arch Qwen3_5ForConditionalGeneration). AutoRound consumes the bf16, not the int4.
  - The existing `Qwen3.6-27B-W4A16` already serves as quant_method=compressed-tensors,
    proving a compressed-tensors dense Qwen3_5 loads on the box; W4A8 only adds int8 acts.

GDN-specific concern: NONE that blocks. GDN routes through normal quantized Linears
(handled) + fp ops (untouched). The only known 27B w4a16 wrinkle was the XPUwNa16
"/32 dim" issue (4304 dim); if the W4A8 kernel trips the same shape check, add the
offending proj to AutoRound `layer_config` as bf16 (see recipe note). UNVERIFIED whether
W4A8 hits it -- watch the smoke run.

--------------------------------------------------------------------------------
[!] RECIPE FIX -- the repo's w4a8/10_quant_autoround_w4a8.sh uses a DEAD API
--------------------------------------------------------------------------------
VERIFIED against the installed `auto_round 0.13.1` (pip --no-deps into :v0230):

  - `AutoRound(model, tok, bits=4, ..., act_bits=8, act_dynamic=True)`  -- the public
    `AutoRound.__init__` is a LazyImport passthrough; the REAL class is
    `auto_round.compressors.BaseCompressor` with params:
        (config, model, tokenizer, platform, format, scheme, low_gpu_mem_usage,
         device_map, ..., layer_config, nsamples, seqlen, **kwargs)
    There is NO top-level `bits=`/`act_bits=` kwarg path like the draft script assumes.
  - There is NO preset scheme named "W4A8" (or "W8A8"). Presets are:
        W4A16, W8A16, INT4, INT8, INT8_W8A8, NVFP4, MXFP4, GGUF:*, ...
    -> you must BUILD the W4A8 scheme by hand (the QuantizationScheme dataclass DOES
       have act_bits/act_dynamic/act_data_type fields; a custom W4A8 constructs cleanly
       -- VERIFIED).
  - `ar.save_quantized(OUT, format="llm_compressor")` -- `save_quantized`/
    `quantize_and_save` DO exist on BaseCompressor, and `export_to_llmcompressor` is a
    real export module (so format -> compressed-tensors, routes to the dense XPU kernel).
    Use `quantize_and_save(output_dir=OUT, format="llm_compressor")` (one call).
  - `device=` -> the real kwarg is `device_map`.

So the draft 10_*.sh WILL FAIL as written. Use the corrected python below.

--------------------------------------------------------------------------------
EXACT 27B AutoRound-W4A8 command (corrected, ready to adapt into 10_*.sh)
--------------------------------------------------------------------------------
Image: vllm-xpu-env:v0230  (has XPUW4A8IntLinearKernel + int4_gemm_w4a8 op + GDN
arch Qwen3_5 + compressed-tensors W4A8Int scheme -- ALL verified present). Route the
GPU touch through scripts/gpu-run. Mount the bf16 source and the models dir.

  SRC=/models/Qwen_Qwen3.6-27B           # bf16 dense source (72 GB, on host)
  OUT=/models/Qwen3.6-27B-W4A8-autoround # output (compressed-tensors W4A8)

  docker run --rm --name w4a8_ar_27b --device /dev/dri -e ZE_AFFINITY_MASK=0 \
    -v /mnt/vm_8tb/b70/models:/models \
    -v /mnt/vm_8tb/b70/hf_cache:/hf_cache -e HF_HOME=/hf_cache \
    -e OMP_NUM_THREADS=32 --entrypoint bash vllm-xpu-env:v0230 -lc '
      pip install -q --no-deps auto-round || pip install -q auto-round
      pip install -q "transformers>=4.52" accelerate datasets || true
      python - <<PY
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from auto_round import AutoRound
from auto_round.schemes import QuantizationScheme
SRC="/models/Qwen_Qwen3.6-27B"; OUT="/models/Qwen3.6-27B-W4A8-autoround"
# W4A8 = int4 sym group-128 weights + per-token dynamic int8 activations.
W4A8 = QuantizationScheme(bits=4, group_size=128, sym=True, data_type="int",
                          act_bits=8, act_dynamic=True, act_sym=True, act_data_type="int")
tok = AutoTokenizer.from_pretrained(SRC, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(SRC, torch_dtype="auto", trust_remote_code=True)
ar = AutoRound(model, tok, scheme=W4A8, iters=200, nsamples=128, seqlen=2048,
               device_map="xpu", format="llm_compressor")   # device_map, NOT device
ar.quantize_and_save(output_dir=OUT, format="llm_compressor")
print("DONE", OUT)
PY'

  # SMOKE FIRST: run once with iters=50 (and maybe nsamples=64) to validate the
  # toolchain + that the export config is quant_method=compressed-tensors with a
  # W4A8 (4-bit weight / 8-bit dynamic-token act) group, BEFORE the full overnight run.
  # If the 27B trips the XPUwNa16 /32-dim shape gate on some proj, append that module
  #   to layer_config={"<module>": {"bits":16}} (keep it bf16) and re-run.

Serve check after quant (route through gpu-run; image with W4A8 kernel + GDN):
  IMG=vllm-xpu-env:v0230 MODEL=/models/Qwen3.6-27B-W4A8-autoround \
  SERVED=qwen36-27b-w4a8-autoround GRAPH=1 bash /mnt/vm_8tb/b70/30_serve_w4a8_graph.sh
  # served_model_id MUST encode method -> use ...-autoround (NOT a bare qwen36-27b-w4a8).
  # Confirm on serve log: quantization=compressed-tensors, XPUW4A8IntLinearKernel chosen.

================================================================================
IMAGE-COMPATIBILITY MATRIX (VERIFIED on host, 2026-06-20)
================================================================================
Surprise: the task premise (":int8/:int8g lack GDN; :v0230 lacks the W4A8 kernel")
is WRONG today. ALL four images have BOTH. The kernels + arch + MoE patch are baked
everywhere now.

  +-----------+----------------+----------------+--------------+--------------+
  | image     | XPUW4A8Int     | int4_gemm_w4a8 | Qwen3_5 GDN  | inc.py MoE   |
  |           | LinearKernel   | op + fake      | arch (27/35) | WNA16 patch  |
  +-----------+----------------+----------------+--------------+--------------+
  | v0230     | YES            | YES (op+fake)  | YES (1 / 2)  | YES (baked)  |
  | v0230moe  | YES            | YES (op+fake)  | YES (1 / 2)  | YES (baked)  |
  | int8      | YES            | YES (op+fake)  | YES (1 / 2)  | YES (baked)  |
  | int8g     | YES            | YES (op+fake)  | YES (1 / 2)  | YES (baked)  |
  +-----------+----------------+----------------+--------------+--------------+
  (grep counts: Qwen3_5ForConditionalGeneration=1, Qwen3_5MoeForConditionalGeneration=2
   in registry.py; inc.py RoutedExperts->MoeWNA16Config patch = 5 hits, all images.)

RECOMMENDATION: use **vllm-xpu-env:v0230** for both the AutoRound quant container AND
the 27B W4A8 serve -- it is the image the JOURNAL already used to serve+capture the 27B
int4 (7.84 -> 30.84 t/s) and it has the W4A8 kernel. No image is a blocker for the 27B.
(The ":int8g vs :v0230" distinction that mattered earlier was about graph-capture
pass_config wiring in 30_serve_w4a8_graph.sh, not kernel/arch presence.)

================================================================================
QUANT-TIME / COMPUTE BUDGET  (INFERRED unless noted)
================================================================================
Baseline (VERIFIED, JOURNAL): 14B GPTQ-W4A8 = 6724 s ~= 112 min on the B70, but that
was Cholesky-bound at ~15% GPU util (GPTQ, not AutoRound -- different algorithm).
AutoRound is per-block sign-SGD (iters x nsamples forward/backward per transformer
block); its cost scales ~ with total quantized params x iters, not the GPTQ Hessian.

INFERRED AutoRound iters=200 wall-clock (codex-corroborated ranges):
  - 27B dense  on B70 (xpu, device_map):  ~4 - 8 h.  Source is 72 GB bf16; will need
    block-streaming / low_gpu_mem_usage so it fits 32 GB. If XPU AutoRound is flaky,
    CPU fallback (125 GB RAM) is ~1 - 2 days -- prefer XPU, smoke first.
  - 35B-A3B MoE: ~8 - 15 h (256 expert blocks dominate; active-param count is
    irrelevant to quant time). MOOT -- no XPU serve kernel + no bf16 source. SKIP.

Sizing for the 27B output: compressed-tensors W4A8 ~= the 14B pattern (int4 weights,
likely UNPACKED i8 on disk ~ same byte/weight). Expect ~30+ GB on disk for 27B unless
the llm_compressor export packs; VRAM resident ~18 GiB (cf. 27B int4 = 17.6 GiB proven,
27B w4a16 fits 1x). 6.5 TB free on /mnt/vm_8tb -- disk is not a constraint.

AutoRound architecture support (INFERRED): AutoRound quantizes Linear modules
generically by block, so a dense Qwen3_5 (GDN) should quantize fine (GDN linears are
ordinary nn.Linear; its conv/recurrence are skipped as non-Linear). MoE (256 experts)
AutoRound export is supported in principle but IRRELEVANT here (no XPU kernel). Known
risk: AutoRound-on-XPU maturity is UNVERIFIED for this build -- the smoke run (iters=50)
is the gate; fall back to device_map="cpu" if the XPU path device-losts.

================================================================================
VERIFIED vs INFERRED summary
================================================================================
VERIFIED (read in the images/host today):
  - No XPU W4A8-int8 MoE backend; oracle is CPU-only + raises on XPU (the 35B blocker).
  - XPUW4A8IntLinearKernel is real on XPU and used by the 14B w4a8; present in all 4 images.
  - 27B arch=Qwen3_5 dense GDN; GDN linears carry quant_config; compressed-tensors W4A8Int
    scheme detector exists; bf16 source Qwen_Qwen3.6-27B (72 GB) on host; NO 35B bf16 source.
  - auto_round 0.13.1 API: NO "W4A8" preset; real class BaseCompressor; build the scheme
    by hand; device_map (not device); quantize_and_save + export_to_llmcompressor exist.
  - All 4 images carry W4A8 kernel + Qwen3_5 GDN arch + inc.py MoE-WNA16 patch.
INFERRED:
  - AutoRound wall-clock (4-8 h 27B / 8-15 h 35B), output sizes, AutoRound-XPU stability,
    whether the 27B trips the /32-dim shape gate under W4A8.

BOTTOM LINE: quant the **27B dense** to W4A8 with the corrected AutoRound recipe on
:v0230 (smoke iters=50 first). SKIP the 35B MoE entirely -- there is no XPU int8-act
expert kernel and no bf16 source; it cannot serve W4A8 and cannot be quantized from int4.
