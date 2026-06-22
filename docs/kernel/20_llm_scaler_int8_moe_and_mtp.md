# 20 -- THE UNLOCK: int8 MoE serving + MTP are ALREADY SOLVED via intel/llm-scaler-vllm

Date: 2026-06-22. Source: two community B70 submissions (provided by Isaac) + an on-host capability probe of the
`intel/llm-scaler-vllm:0.14.0-b8.3.1` image (already pulled on the GPU host). This SUPERSEDES two of our earlier
conclusions: docs/kernel/18 ("no XPU int8 MoE kernel -> build it") and docs/literature/09 ("MTP not viable on B70").
Both are now WRONG for the right stack: **intel/llm-scaler-vllm has the int8 MoE kernel AND the MTP spec path.**

================================================================================
1. THE STACK -- intel/llm-scaler-vllm (Intel's XPU vLLM distribution)
================================================================================
- Image **intel/llm-scaler-vllm:0.14.0-b8.3.1** is ON THE HOST (34.3 GB). vLLM **0.14.1.dev0+gb17039bcc** (2026-06-05).
- Supported MoE archs (probed `ModelRegistry`): **OlmoeForCausalLM, Qwen3MoeForCausalLM, Qwen3_5MoeForConditionalGeneration,
  Qwen3VLMoeForConditionalGeneration, Qwen3OmniMoeForConditionalGeneration, Qwen3_5MTP, Qwen3_5MoeMTP**.
  => our 35B (qwen3_5_moe) AND its MTP variant are first-class here.
- Quant methods registered: **quark, compressed-tensors, experts_int8, rtn, moe_wna16, gptq, awq, awq_marlin, auto-round,
  fp8, gguf, bitsandbytes, ...** -- so int8 MoE can come from Quark, compressed-tensors, OR runtime `experts_int8`/`rtn`.
- Has `qwen3_5_mtp.py` + `qwen3_5.py` in model_executor/models (the MTP arch + spec wiring present in-image).
- CAVEAT: the **Quark QUANTIZER is NOT in the image** (`import quark` -> ModuleNotFoundError). The image SERVES Quark
  checkpoints (`--quantization quark`) but to PRODUCE one you `pip install amd-quark` separately (or use a community ckpt,
  or use compressed-tensors / experts_int8 / rtn instead).

================================================================================
2. int8 MoE SERVING -- the PROVEN recipe (steveseguin, accepted production run)
================================================================================
Qwen3.6-35B-A3B **Quark W8A8 INT8** on **4x B70**, **99.77 corrected tok/s** (e2e 98.27, TTFT 76.5ms; 512/512, b1, t=0).
HF ckpt: `nameistoken/Qwen3.6-35B-A3B-Quark-W8A8-INT8`. Launch (the template):
```
vllm serve <quark-w8a8-ckpt> --host 127.0.0.1 --port 18080 --trust-remote-code \
  --served-model-name qwen36-35b-a3b --dtype auto --quantization quark \
  --tensor-parallel-size 4 --pipeline-parallel-size 1 --distributed-executor-backend mp \
  --max-model-len 32768 --max-num-batched-tokens 8192 --max-num-seqs 48 \
  --gpu-memory-utilization 0.95 --kv-cache-dtype auto --no-enable-prefix-caching \
  --language-model-only --compilation-config '{"cudagraph_mode":"PIECEWISE"}' --generation-config vllm
```
Engine: vLLM 0.20.2rc1.dev (his build) / our on-host is 0.14.1.dev (newer naming). ~31.9 GiB/card at util 0.95.
NOTE Steve used **TP=4**; we have **2** cards -> TP=2 (the 35B int8 ~35GB splits to ~17.5GB/card -> fits 32GB).
Quant options for US (no Quark quantizer locally): (a) `--quantization experts_int8` or `rtn` on the BF16 model
(RUNTIME int8 -> NO offline quant, fastest validation); (b) our compressed-tensors W8A8 (llmcompressor, proven) +
`--quantization compressed-tensors`; (c) `pip install amd-quark` and reproduce Steve's Quark path exactly.

================================================================================
3. MTP -- the PROVEN recipe (user ytnszmy). Answers "how did others get MTP working on 4xB70?"
================================================================================
Qwen3.6-27B BF16 (fp16 runtime), **TP=4**, **MTP/spec-decode UNBLOCKED FROM USERSPACE**:
- **vllm_xpu_kernels v0.1.9** wheel
- **qwen3_5.py spec-wiring patch == vLLM PR #43565**
- **Half-KV**
- `num_speculative_tokens=5`, **mean accept length 4.04 (88.9% accept @ spec=3)**; prefill ~2100 tok/s; 256K ctx.
- Stack: `intel/llm-scaler-vllm:0.14.0-b8.3.x`.
=> Our doc-09 "MTP not viable / -19%" was a STACK gap (we lacked #43565 spec-wiring + the v0.1.9 kernel wheel + Half-KV),
NOT a B70 limitation and NOT a quant-precision question. Re-test on the on-host image (which already has qwen3_5_mtp.py;
verify it includes #43565 -- the image is a 2026-06-05 build so likely YES). The W8A8-vs-W4A16 "MTP receptivity" question
is moot until MTP runs; once it runs, measure acceptance, not just t/s.

================================================================================
4. WHAT THIS CHANGES (supersedes earlier docs)
================================================================================
- docs/kernel/18 (build an int8 MoE kernel): mostly MOOT for SERVING -- llm-scaler already has it. The "build our own"
  becomes a RESEARCH/port goal (study llm-scaler's fused-int8-MoE + the Quark/experts_int8 path; port to contrib/vllm_int8_xpu
  if we want it in OUR :int8g image). Not a blocker for the 35B anymore.
- docs/literature/09 (MTP): viable -> the recipe is sec 3 above.
- QUANTS_TODO Q6/Q7 (35B int8): SERVE PATH unblocked -> Quark/experts_int8 on llm-scaler TP=2. Production via Quark;
  fast validation via runtime experts_int8/rtn. Our multi-day llmcompressor-GPTQ produce is NOT needed if Quark/runtime works.

================================================================================
5. OUR PLAN (ordered)
================================================================================
1. [small MoE] Serve **OLMoE-1B-7B-0924-Instruct** (downloaded) on llm-scaler with `--quantization experts_int8` (runtime
   int8 MoE, no offline quant) TP=1 -> validate the int8 fused-MoE kernel on OUR cards -> bench ctx2048 c=1/2/4/8.
   Fallback: compressed-tensors W8A8 (llmcompressor) if experts_int8 misbehaves.
2. [35B] Serve Qwen3.6-35B-A3B on llm-scaler with experts_int8 (or a Quark W8A8 ckpt) TP=2 -> the real Q6/Q7 SERVE
   datapoint (int8 MoE on the 35B, finally). Bench vs the existing 35B W4A16 (Intel int4-AutoRound, 56.8 t/s).
3. [MTP] Re-test MTP on llm-scaler:0.14.0-b8.3.1 + Half-KV + `--speculative-config '{"method":"qwen3_5_mtp",
   "num_speculative_tokens":5}'`; confirm #43565 is in-image, else apply it. Measure accept length.
4. [our kernel] Study how llm-scaler implements the int8 fused-MoE GEMM (the "persistent W8A8 MoE layerlet"); decide
   whether to port into contrib/vllm_int8_xpu so OUR :int8g image gets it too.

================================================================================
6. PATCHES / GITHUBS to pull if needed (per Isaac)
================================================================================
- vLLM **PR #43565** -- qwen3_5.py MTP spec-wiring (verify presence in 0.14.0-b8.3.1; else cherry-pick).
- **vllm_xpu_kernels v0.1.9** -- the kernel wheel ytnszmy used for MTP (check the on-host image's version).
- **amd-quark** (pip) -- to produce Quark W8A8 ckpts ourselves (image serves but doesn't quantize Quark).
- steveseguin/b70-optimization-lab (cloned /tmp/b70-lab): `scripts/bench-qwen36-b70-single-mtp.sh`,
  `experiments/minimax-m27-reap-autoround-vllm/notes/2026-06-02-moe-micro-and-logitsws-retest.md`,
  `repro/.../configs/promoted-env.sh` -- his MoE/MTP bench harness + promoted env (reference).
- nameistoken/Qwen3.6-35B-A3B-Quark-W8A8-INT8 (HF) -- a ready Quark 35B ckpt (could pull instead of quantizing).

================================================================================
7. [!] CRITICAL llm-scaler GOTCHA + CORRECTION (2026-06-22)
================================================================================
**The image ENTRYPOINT is `["bash","-c","vllm serve"]` -- a FIXED string with NO model arg.** A normal
`docker run IMG vllm serve <ckpt> --host ... --port ...` does NOT work: docker passes your args as positional
params ($0,$1,...) to `bash -c "vllm serve"`, which ignores them -> vLLM runs bare `vllm serve` -> serves its
**DEFAULT model `Qwen/Qwen3-0.6B`** (loads ~1.12 GiB), silently. It even reports HEALTHY on :8000. This also makes
`--port` and `--served-model-name` appear "ignored" (they were positional-swallowed too).
=> **MUST override: `docker run ... --entrypoint vllm IMG serve <ckpt> --host 0.0.0.0 --port 8000 ...`** (note: CMD
starts with `serve`, not `vllm serve`). scripts/74 fixed accordingly.

**CORRECTION to sec 1/earlier JOURNAL:** the "OLMoE int8 MoE serves on B70 (200 OK)" milestone was the **0.6B default**,
NOT a real int8 MoE. The int8 MoE serve on our cards is NOT YET CONFIRMED. The tell was "Model loading took 1.12 GiB"
for a supposed 7B/35B. All those serves (OLMoE experts_int8, 35B Quark) were the 0.6B default until the entrypoint fix.

**Current status (entrypoint fixed):** 35B Quark-W8A8 now loads the REAL model at TP=2 (no longer 0.6B), but hits a
**WorkerProc initialization failure** at TP=2 (multiproc_executor.wait_for_ready) in ~90s. Root cause TBD (TP=2
collective init or 17.5GB/card+KV OOM at util 0.95). NEXT: re-run capturing the full worker traceback; try
SYCLKERNELS=1 (set) + lower util/MAXLEN; if collective-init, compare to our :int8g TP=2 (which worked host-staged).
The 35B int8 MoE serve verdict is PENDING this debug.
