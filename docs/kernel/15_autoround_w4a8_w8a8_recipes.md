# 15 -- AutoRound W4A8 / W8A8 recipes for Qwen3.6-27B + the 35B-A3B MoE verdict

> **[!] UPDATE 2026-06-22 -- AutoRound DOES quantize the qwen3_5 VLMs (the "MLLM-calib blocked" wall is BEATEN).**
> Q2/Q4 thought AutoRound was blocked on the VLM and fell back to GPTQ. Real cause: AutoRound 0.13.1 auto-detects the
> qwen3_5 VLM, forces MLLM mode, and `quantize()` asserts `processor should not be None` (`quant_nontext_module=False`
> alone is insufficient). **Working recipe (scripts/84_q8_qwable_int4.py, smoke-validated end-to-end):**
> `proc = AutoProcessor.from_pretrained(SRC, trust_remote_code=True)`;
> `AutoRoundMLLM(model, tokenizer, processor=proc, quant_nontext_module=False, dataset=<list[str] text>, scheme="W4A16"
> (or "W8A8"), layer_config={<vision/mtp/linear_attn/lm_head> -> {"bits":16}}, device_map="auto"/max_memory)`;
> `ar.quantize(); ar.save_quantized(output_dir=OUT, format="auto_round")` (inc-servable; NOT llm_compressor). Build
> `layer_config` by enumerating nn.Linear names matching `lm_head|visual|vision_tower|mtp|linear_attn`. This unblocks
> Q8 (W4A16) AND retroactively Q2/Q4 (W8A8-AutoRound) + RESEARCH_TODO Track 3. (auto_round 0.13.1 on vllm-xpu-env:v0230.)

Date: 2026-06-20. Author: quant-eng (read-only investigation; NO GPU touched -- only
non-GPU `docker run` package inspection + upstream WebSearch/WebFetch).
Scope: the user wants AutoRound (better accuracy than RTN) for the 27B in BOTH W4A8 and
W8A8, plus a go/no-go on the 35B-A3B MoE int8-act path.

================================================================================
TL;DR (read this, skip the rest if pressed)
================================================================================

  +----------------+------------------------------------------------------------+
  | Target         | Verdict                                                    |
  +----------------+------------------------------------------------------------+
  | AutoRound W4A8 | STILL BLOCKED. auto_round 0.13.1 (latest pip) AND main      |
  | export         | (0.14.0-dev) both hard-assert W8A8 in the llm_compressor    |
  |                | exporter for ANY int8-dynamic-act scheme. No release fixes  |
  |                | it. Graft of auto_round/gptq packing onto a compressed-     |
  |                | tensors W4A8 config does NOT match our kernel's weight       |
  |                | layout. => Use GPTQ-W4A8 (scripts/54) for the 27B W4A8.      |
  +----------------+------------------------------------------------------------+
  | AutoRound W8A8 | SUPPORTED (this is the one scheme the exporter allows:       |
  | export         | INT8_W8A8, per-channel int8 weight + dynamic per-token int8  |
  |                | act). Exact command below. 27B caveat: GDN/VLM arch needs   |
  |                | the ignore list + the wrapper-config graft to serve.        |
  +----------------+------------------------------------------------------------+
  | 35B-A3B MoE    | W4A8-int8 MoE: NO-GO on XPU (oracle raises, CPU-only).       |
  | int8-act       | W8A8-int8 MoE: the ONLY non-CPU backend is generic Triton    |
  |                | (no XPU oneDNN int8 expert kernel exists; XPUExperts covers  |
  |                | fp8/mxfp8/blockfp8/int4-WNA16/mxfp4 -- NO int8). Triton-on-  |
  |                | XPU for fused MoE is unproven/flaky here. => NO-GO for a     |
  |                | clean fast path; a small patch only routes it to Triton, not |
  |                | to the XMX systolic path. Also: no bf16 35B source on host.  |
  +----------------+------------------------------------------------------------+

The headline: **AutoRound cannot make a W4A8 checkpoint for our XPU int8 kernel.** The
user's accuracy goal for 27B W4A8 is served by **GPTQ-W4A8 (scripts/54)**, already proven
on the 14B (HumanEval+ 0.872/0.835, ties w4a16-gptq base, ~1 CI off on plus).

================================================================================
1. AutoRound W4A8 export -- RE-VERIFIED, STILL BLOCKED (0.13.1 AND main 0.14.0-dev)
================================================================================

WHAT'S INSTALLED (verified by `pip show` in non-GPU containers, 2026-06-20):
  - NO image has auto-round OR llmcompressor baked in. All four vllm-xpu-env images
    (:v0230, :v0230moe, :int8, :int8g) carry compressed-tensors 0.17.0 + vllm 0.23.0+xpu
    only (:tf has compressed-tensors 0.15.0.1). The quant scripts `pip install` the
    tooling at runtime -- so any recipe MUST pip-install auto-round/llmcompressor.
  - pip's latest auto-round is **0.13.1** (full version list checked: ...0.13.0, 0.13.1;
    nothing newer). auto-round `main` self-reports `__version__ = "0.14.0"` (UNRELEASED).

THE BLOCK -- re-read on BOTH 0.13.1 and `main` (auto_round/formats.py):
  `LLMCompressorFormat.check_and_reset_format()` -- for ANY scheme where
  `act_bits <= 8 and act_dynamic` (i.e. our int8-dynamic-act target), and not an fp8/mx/nv
  special case, falls into a bare `else` that pins and asserts:

      bits, group_size, sym, act_bits = 8, -1, True, 8
      assert (ar.bits == bits and ar.group_size == group_size and ar.sym == sym
              and ar.act_bits == act_bits and ar.act_dynamic), \
          "Currently only support to export llm_compressor format for sym dynamic "
          "quantized W{bits}A{act_bits} model with group_size=-1 ..."

  A W4 scheme has `ar.bits == 4` -> `4 == 8` is False -> **AssertionError at export**.
  The error message renders "W4A8" from the actual bits, but the REQUIRED config is W8A8.
  (This is the exact failure the smoke run hit on 2026-06-20; JOURNAL "AutoRound CANNOT
  export W4A8".)

WHY main (0.14.0-dev) does NOT fix it, even though the exporter LOOKS more flexible:
  - The low-level builder `export_to_llmcompressor/export.py::construct_ct_scheme()` WILL
    build a valid W4A8 compressed-tensors scheme (4-bit weight group + int8 dynamic-token
    act) -- it only nulls activations when `act_bits >= 16`. So the *plumbing* could emit
    W4A8. BUT it never runs for a W4A8 scheme, because:
  - `LLMCompressorFormat.support_schemes` (main) = [MXFP4, MXFP8, NVFP4, FPW8A16,
    FP8_STATIC, INT8, INT8_W8A8, FP8_BLOCK, W4A16, W8A16]. There is **no W4A8 entry**;
    `is_support_scheme()` / `check_scheme_args()` reject a 4-bit + int8-act scheme up front.
  - The int8-dynamic-act router `is_dynamic_wint8aint8(ar)` -> `is_wint8aint8(ar)` requires
    BOTH weight `bits == 8` AND act `bits == 8`. A W4 scheme fails it, so it is NOT routed
    to the INT8 backend; it falls through to the bare-`else` assertion above and dies.
  Net: on main the W8A8-only gate is the SAME hard assertion as 0.13.1. **W4A8 export is
  blocked on every released and unreleased auto-round version as of 2026-06-20.**

OPTION (a) -- newer auto-round that exports W4A8 directly:  NONE EXISTS (verified above).

OPTION (b) -- AutoRound W4 weights in auto_round/gptq format, then GRAFT a compressed-
tensors W4A8 quantization_config:  NOT VIABLE (weight layout + tensor names mismatch).
  Our serve path (CompressedTensorsW4A8Int + XPUW4A8IntLinearKernel) expects, at load:
    - `weight_packed`: an UNPACKED int8 `[out, in]` tensor holding signed int4 values
      in [-8, 7] (the XPU kernel re-packs it to int32 itself in
      `process_weights_after_loading::_pack_int4_weight`), plus
    - `weight_scale`: per-group/per-channel scales, dynamic int8 per-token activations
      computed AT RUNTIME (`ops.dynamic_per_token_int8_quant_ref`).
  auto_round's `auto_round:auto_gptq` format packs int4 into **int32 `qweight`** with
  `qzeros / scales / g_idx` tensor NAMES -- a different dtype, a different packing, and
  different parameter names. A grafted compressed-tensors W4A8 config would make vLLM's
  weight loader look for int8 `weight_packed`, which the gptq file does not contain ->
  load failure. Making it work needs a real RE-PACK pass (unpack int32 gptq -> int8
  [out,in], rename, write a compressed-tensors header) -- effectively re-implementing the
  llm_compressor export that auto-round refuses to do. Not worth it vs GPTQ-W4A8.

OPTION (c) -- AutoRound `--format` choices:  the only int8-dynamic-act export is
  `INT8` / `INT8_W8A8` (= W8A8). The `auto_round` native format serves on XPU as **W4A16**
  (the INC path ignores int8 acts) -- no int8-XMX prefill win. Neither gives W4A8-int8.

-> CONCLUSION (W4A8): **still blocked, use GPTQ-W4A8 instead.** This matches the prior
   pivot. Important nuance vs the upstream scare in vLLM issue #38064 (W4A8-INT silently
   runs W4A16 -- act never int8): that bug is in the CUDA **Marlin** path
   (`apply_gptq_marlin_linear`, `act_type=params_dtype`). It does **NOT** affect us: our
   XPUW4A8IntLinearKernel.apply_weights explicitly calls
   `ops.dynamic_per_token_int8_quant_ref(x, True, 8)` then `int4_gemm_w4a8` -- it genuinely
   runs int8 activations on the XMX path (that is why the 14B w4a8 measured +51% prefill).
   So our compressed-tensors W4A8 GPTQ checkpoints ARE real int8-act on the B70.

--------------------------------------------------------------------------------
27B W4A8 -- the recommended path (GPTQ-W4A8 via llmcompressor, scripts/54-style)
--------------------------------------------------------------------------------
GPTQ-W4A8 on the 14B already landed HumanEval+ 0.872 base / 0.835 plus (JOURNAL
2026-06-20) -- base TIES w4a16-gptq (0.872), plus within ~1 CI of 0.848. So GPTQ recovers
nearly all of AutoRound's hoped-for accuracy edge for W4A8.

27B SERVE caveat (open blocker, NOT a quant-method problem -- JOURNAL "27B-W4A8 serve
blocked by VLM odd-dims (4304)"): Qwen3.6-27B is a Qwen3_5 VLM (vision + GatedDeltaNet +
MTP). The group-128 int4 W4A8/W4A16 kernel rejects dims not divisible by 128 -- the vision
MLP `linear_fc2` (dim 4304) and some DeltaNet projections trip
`input_size_per_partition 4304 not divisible by group_size 128`. To serve a 27B W4A8 you
must (1) keep vision/DeltaNet/MTP/lm_head BF16 in the IGNORE list (scripts/49 already does
this for W8A8), AND (2) graft the VLM wrapper config (`w4a8/fix_27b_vlm_config.py`) so vLLM
builds those BF16. This is the same arch work the W8A8 path needs (section 2). The W4A8
SPEED hypothesis (int8-XMX prefill) is ALREADY proven on the dense 14B; the 27B W4A8 is a
confirmation, gated on resolving the VLM odd-dim serve issue.

  EXACT 27B GPTQ-W4A8 quant command (GPU-accelerated calibration, route via gpu-run):
    scripts/gpu-run env \
      SCHEME=W4A8 METHOD=gptq DEVICE=xpu SMOOTHQUANT=0 \
      SRC=/mnt/vm_8tb/b70/models/Qwen_Qwen3.6-27B \
      OUTNAME=Qwen3.6-27B-W4A8-gptq SAMPLES=512 SEQLEN=2048 \
      IGNORE="lm_head re:.*linear_attn.* re:.*visual.* re:.*mtp.*" \
      bash scripts/49_quantize_27b_w8a8.sh
    # scripts/49 already passes SCHEME through to GPTQModifier; SMOOTHQUANT=0 because the
    # hybrid Qwen3_5 (only 16/64 layers carry self_attn q/k/v) breaks SmoothQuant's
    # smooth-layer<->qkv pairing (ValueError before any GPU work). GPTQ-only is still a
    # strong recipe (GPTQ weight calibration + dynamic per-token int8 acts).
    # actorder=None inside scripts/49 (avoids the XPU gather device-lost).
  Then: w4a8/fix_27b_vlm_config.py on the output, then serve via
  w4a8/30_serve_w4a8_graph.sh with served_model_id qwen36-27b-w4a8-gptq (method-tagged).

================================================================================
2. AutoRound W8A8 export -- SUPPORTED. Exact recipe.
================================================================================

W8A8 is the ONE int8-dynamic-act scheme the auto-round llm_compressor exporter allows
(`is_dynamic_wint8aint8` -> INT8 backend; the assertion `bits==8,group_size==-1,sym,
act_bits==8` is exactly W8A8 and PASSES). Output is compressed-tensors: per-channel int8
weights + dynamic per-token int8 activations -> routes to our int8 W8A8 oneDNN kernel
(image vllm-xpu-env:int8 / :int8g, the canonical INT8 W8A8 path).

NOTE the 0.13.1 API (verified): public `AutoRound` is a LazyImport over
`auto_round.compressors.BaseCompressor`. There is NO bare `bits=`/`act_bits=` kwarg and NO
"W8A8" preset string; build the scheme by hand (the preset `INT8_W8A8` also works) and use
`device_map` (NOT `device`). `quantize_and_save(format="llm_compressor")` exists.

27B is a hybrid GDN VLM -> ignore the non-GEMM / odd-dim modules (same list as scripts/49):
  lm_head + the WHOLE vision tower (re:.*visual.*, incl. the 4304-dim fc2) + MTP
  (re:.*mtp.*). GDN: the GatedDeltaNet has ordinary nn.Linear in/out projections -- those
  CAN quantize, BUT some have non-/128-friendly dims and serving the VLM is fragile, so the
  SAFE first pass IGNORES linear_attn too (re:.*linear_attn.*) -- identical to the proven
  scripts/49 W8A8 list. (A later pass can try quantizing GDN linears to shrink VRAM if the
  /32-divisible shape check passes; not needed to fit 32 GB -- W8A8 27B ~= 27 GB resident,
  fits one card.)

  EXACT 27B AutoRound-W8A8 command (B70-accelerated calibration; route via gpu-run):

    SRC=/mnt/vm_8tb/b70/models/Qwen_Qwen3.6-27B         # bf16 VLM source (~72 GB, on host)
    OUT=/mnt/vm_8tb/b70/models/Qwen3.6-27B-W8A8-autoround

    scripts/gpu-run docker run --rm --name w8a8_ar_27b \
      --device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path -e ZE_AFFINITY_MASK=0 \
      --ipc=host --shm-size 32g \
      -v /mnt/vm_8tb/b70:/mnt/vm_8tb/b70 -e HF_HOME=/mnt/vm_8tb/b70/hf_cache \
      -e OMP_NUM_THREADS=32 --entrypoint bash vllm-xpu-env:v0230 -lc '
        source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1 || true
        pip install -q auto-round "transformers>=4.52" accelerate datasets 2>&1 | tail -2
        python - <<PY
import torch
from transformers import AutoTokenizer
from auto_round import AutoRound
from auto_round.schemes import QuantizationScheme
SRC="/mnt/vm_8tb/b70/models/Qwen_Qwen3.6-27B"
OUT="/mnt/vm_8tb/b70/models/Qwen3.6-27B-W8A8-autoround"
# W8A8 = per-channel (group_size=-1) sym int8 weights + per-token dynamic int8 acts.
W8A8 = QuantizationScheme(bits=8, group_size=-1, sym=True, data_type="int",
                          act_bits=8, act_dynamic=True, act_sym=True, act_data_type="int")
# Keep vision tower + MTP + lm_head (and, safe first pass, GDN linear_attn) BF16.
IGN = ["lm_head"]   # AutoRound layer_config: set these modules to 16-bit (see below)
tok = AutoTokenizer.from_pretrained(SRC, trust_remote_code=True)
# VLM: AutoModelForCausalLM may not map Qwen3_5 ForConditionalGeneration; AutoRound takes a
# preloaded model. Use the loader fallback chain from scripts/49 if needed (try CausalLM,
# then AutoModelForImageTextToText / Vision2Seq / AutoModel).
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(SRC, torch_dtype="auto", trust_remote_code=True)
# layer_config: force the ignore set to 16-bit (AutoRound keeps them unquantized).
import re
layer_config = {}
for name, mod in model.named_modules():
    if isinstance(mod, torch.nn.Linear) and (
        name.endswith("lm_head") or re.search(r"(visual|mtp|linear_attn)", name)):
        layer_config[name] = {"bits": 16}
ar = AutoRound(model, tok, scheme=W8A8, layer_config=layer_config,
               iters=200, nsamples=128, seqlen=2048,
               device_map="xpu", format="llm_compressor")   # device_map, NOT device
ar.quantize_and_save(output_dir=OUT, format="llm_compressor")
print("DONE", OUT)
PY'

  SMOKE FIRST: re-run once with iters=50 nsamples=64 to validate the toolchain + that the
  export config is quant_method=compressed-tensors with a W8A8 (8-bit weight / 8-bit
  dynamic-token act) group, BEFORE the full ~4-8 h run. If AutoRound-on-XPU device-losts,
  fall back to device_map="cpu" (slower, ~1-2 days, but 125 GB RAM holds the 72 GB bf16).

  ALTERNATIVE (lower risk, already proven on this arch): GPTQ-W8A8 via scripts/49 with the
  default SCHEME=W8A8 -- this is the EXISTING, tested 27B int8 path. AutoRound-W8A8 is the
  accuracy upgrade ONLY if AutoRound-on-XPU is stable for this VLM (UNVERIFIED -- gate on
  the smoke run). If it flakes, GPTQ-W8A8 is the safe ship.

  SERVE (route via gpu-run; int8 W8A8 oneDNN image):
    Apply w4a8/fix_27b_vlm_config.py to OUT first (VLM wrapper-config graft + processor
    files), then serve on vllm-xpu-env:int8g with served_model_id
    qwen36-27b-w8a8-autoround (method-tagged -- NEVER a bare qwen36-27b-w8a8). Confirm on
    the serve log: quantization=compressed-tensors, the W8A8 int8 kernel chosen.

  SIZE: 27B W8A8 ~= 27 GB weights resident + KV -> fits one 32 GB B70 (tight at long ctx;
  use FP8 KV / shorter MAXLEN if needed). bf16 source is the only 27B full-precision copy
  on host. Disk: ~27-30 GB output, 6.5 TB free -- not a constraint.

================================================================================
3. 35B-A3B MoE int8-act -- VERDICT + patch assessment
================================================================================

VERIFIED on the live host (vllm-xpu-env:v0230moe, vllm 0.23.0+xpu, 2026-06-20):

W4A8-INT8 MoE -- NO-GO (unchanged from prior finding):
  `fused_moe/oracle/w4a8_int8.py::_get_priority_backends()` returns [] for non-CPU;
  `select_w4a8_int8_moe_backend()` raises `NotImplementedError("W4A8 Int8 MoE is only
  supported on CPU platforms")`. The only kernel class is `CPUExpertsInt4` (aten CPU pack,
  not SYCL). A W4A8-int8 MoE checkpoint would RAISE at weight-load on the B70 -- not even
  a fp16 fallback. No XPU int4w x int8act fused-expert kernel exists.

W8A8-INT8 MoE -- effectively NO-GO for a FAST path (new detail this pass):
  `fused_moe/oracle/int8.py::_get_priority_backends()` returns `[Int8MoeBackend.TRITON]`
  with NO platform gate -- it maps int8 W8A8 MoE to the generic `TritonExperts` fused-MoE
  kernel. BUT the XPU-native expert path `experts/xpu_moe.py` (the oneDNN `xpu_fused_moe`
  op) has subclasses ONLY for: Fp8, MxFp8, BlockFp8, **WNA16 (int4 W4A16, is_int4=True)**,
  and MxFp4. There is **NO XPUExperts int8-W8A8 subclass** -- the kernel takes
  is_fp8/is_int4/is_mxfp4 booleans, no is_int8. So a W8A8-int8 MoE on XPU would NOT touch
  the XMX systolic oneDNN expert path; it would have to run on Triton-on-XPU.
  Triton-on-XPU for these fused-MoE kernels is unproven/flaky in this stack (see
  docs/kernel/13_triton_xpu_enable.md -- enabling it is itself a fragile, brittle exercise).
  => There is NO clean int8-act fused-expert fastpath for the 35B MoE on Xe2. Even if
  Triton-XPU runs, it would not be the XMX int8 win we get on the dense 27B/14B; it would
  be a generic Triton kernel of uncertain perf/correctness. NO-GO for the intended speedup.

Patch sketch (if someone insists later -- NOT recommended now):
  A *small* patch would only force the W8A8-int8 MoE to TritonExperts on XPU (the oracle
  already returns TRITON unconditionally, so mostly it is about getting Triton-XPU to JIT
  the int8 fused_moe kernel cleanly -- the docs/kernel/13 RANK-1..3 enablement). That gets
  CORRECTNESS at best, not the XMX fastpath. The REAL win needs a NEW `XPUExpertsInt8`
  subclass in xpu_moe.py + an `is_int8` path in the `xpu_fused_moe` oneDNN op (a SYCL
  int8w x int8act fused-expert kernel) -- large effort, out of scope for a quant run, and
  parallels the same gap as the dense int8 kernel we already wrote (contrib/vllm_int8_xpu).
  Today the 35B's working path is W4A16 (int4 weight, fp16 act) via MoeWNA16Config ->
  XPUExpertsWNA16 / Triton GPTQ-AWQ kernel (the inc.py patch) -- NO int8 acts, NO int8-XMX
  prefill win. That is the ceiling for the MoE without new kernel work.

ALSO BLOCKING regardless of kernel: NO bf16 35B source on host (only the int4-AutoRound
dir, 21.5 GB). AutoRound/GPTQ both need full precision -> cannot even produce a 35B int8
checkpoint without re-downloading the bf16 (and there is no kernel to serve it on anyway).

=> 35B-A3B int8-act (W4A8 or W8A8): **NO-GO.** Keep serving the 35B as W4A16-int4
   (decodes 56.8 t/s captured). Do not spend GPU hours quantizing it to int8-act.

================================================================================
VERIFIED vs INFERRED
================================================================================
VERIFIED (read today, host images + auto-round main + pip index):
  - auto-round latest pip = 0.13.1; main = 0.14.0-dev. BOTH block W4A8 llm_compressor
    export via the same `bits==8,group_size==-1,sym,act_bits==8` assertion; support_schemes
    has no W4A8; is_wint8aint8 needs weight bits==8. (auto_round/formats.py + compressors/utils.py)
  - construct_ct_scheme() on main COULD emit W4A8 but is never reached for a W4 scheme.
  - No image bakes auto-round/llmcompressor; compressed-tensors 0.17.0 + vllm 0.23.0+xpu.
  - XPUW4A8IntLinearKernel genuinely quantizes acts to int8 (dynamic_per_token_int8_quant_ref
    + int4_gemm_w4a8); the #38064 Marlin silent-W4A16 bug does NOT affect XPU.
  - compressed_tensors W4A8 weight_packed = unpacked int8 [out,in]; gptq/auto_round packs
    int32 qweight -> graft layout mismatch.
  - 35B: W4A8-int8 MoE oracle raises on XPU (CPU-only). W8A8-int8 MoE oracle -> TritonExperts
    only; XPUExperts has NO int8 subclass. No bf16 35B source on host.
  - 14B GPTQ-W4A8 = HumanEval+ 0.872/0.835 (the recommended 27B W4A8 fallback's track record).
INFERRED:
  - AutoRound-W8A8-on-XPU wall-clock (~4-8 h, 27B) and XPU stability for this VLM (gate on
    the iters=50 smoke); whether the 27B W8A8 trips any /32-dim shape gate on a quantized
    GDN linear (the safe IGNORE list avoids it); Triton-XPU int8 MoE perf if anyone tries it.

BOTTOM LINE:
  - AutoRound W4A8: NOT POSSIBLE (export blocked on all versions). Ship GPTQ-W4A8 (scripts/54
    pattern via scripts/49 for the 27B) -- after resolving the 27B VLM odd-dim serve graft.
  - AutoRound W8A8: POSSIBLE -- use the section-2 command on vllm-xpu-env:v0230 (smoke first);
    GPTQ-W8A8 (scripts/49 default) is the proven low-risk fallback for the same scheme.
  - 35B-A3B int8-act: NO-GO (no XPU int8 expert kernel + no bf16 source). Stay W4A16-int4.
