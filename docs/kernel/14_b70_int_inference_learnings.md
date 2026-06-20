# 14 - What we learned optimizing int4/int8 LLM inference on the Intel Arc Pro B70

Capstone synthesis of the 2026-06-20 kernel-optimization campaign (Qwen3-14B W4A8/W8A8 + Qwen3.6 flagships
on ONE B70: Xe2/Battlemage, 32 GB GDDR6, 608 GB/s, 367 INT8 TOPS, **no native FP8**). The definitive
takeaways, newest evidence in JOURNAL.md + docs/kernel/04,10-13. ASCII only.

## The one-paragraph answer
On the B70, **graph capture -- not kernel hand-tuning -- is the decode win.** The custom int4/int8 oneDNN
GEMM is *already* at the practical bandwidth ceiling, so the high-leverage move was killing eager per-op
dispatch (PIECEWISE XPU graph capture), not rewriting the matmul. We proved this by *trying* the kernel
rewrites and measuring them flat. The best single-card stack today = **oneDNN int8/int4 GEMM + PIECEWISE
capture + fp8-KV** (the fp8-KV piece is VALIDATED: `--kv-cache-dtype fp8` (e4m3) serves coherently, +3% decode
at medium ctx growing with context, and 2x KV capacity; the unscaled `fp8_e5m2` is rejected for quantized
checkpoints). The next real gains need a toolchain bump (oneAPI 2026.0 -> FULL capture) or a 2nd card.

> NB on fp8-KV: B70 has no FP8 ALU, so this is fp8-STORAGE + dequant-on-read in attention -- it halves the KV
> bandwidth + footprint (decode win grows with context length; doubles max context / batch), at no measured
> quality loss. Use `e4m3` (alias `fp8`), NOT `e5m2`. Knob: `30_serve_w4a8_graph.sh KVDTYPE=fp8`.

## The five load-bearing learnings

1. **PIECEWISE XPU graph capture is the dominant decode lever (THE breakthrough).** Eager vLLM-XPU pays
   per-op dispatch for hundreds of ops/token; capturing the decode graph removes it. Measured: w4a16 28->55,
   **w4a8 17->48 t/s (+187%)**, w8a8 24->27; flagships **27B 7.8->30.8, 35B-MoE 7.9->56.8 t/s (4-7x)**. The
   gain scales with how dispatch-bound the path was (w4a8's unfused pure-PyTorch act-quant gained most). Recipe:
   image `:int8g`, `VLLM_XPU_ENABLE_XPU_GRAPH=1`, `cudagraph_mode=PIECEWISE`, torch 2.11+xpu, + a `pass_config`
   that disables the CUDA-only inductor fusion passes (else `NameError: MLARoPEKVCacheCatFusionPass` on XPU).

2. **The oneDNN int8/int4 GEMM is already near the practical ceiling -- the kernel is NOT the bottleneck.**
   Two "obvious" hand-tunes were implemented, validated correct, and measured PERF-NEUTRAL: **B1** (drop the
   symmetric src zero-point) and **PP-1** (`format_tag::any` weight + cached VNNI reorder, the IPEX trick).
   oneDNN v3.9's `jit:gemm:any` already handles weight-layout + zero-points internally. Prefill sits at 67-80%
   of 367 TOPS, decode at 73-95% of 608 GB/s. **A custom SYCL int4 GEMV (the "1-2 week biggest win") is futile:
   our oneDNN int4 (~73% BW) meets/beats llama.cpp's purpose-built Xe2 int4 GEMV (~67%), and llama.cpp's int8
   path is a regression vs int4.** Lesson: measure the library before assuming you can beat it -- on Battlemage,
   oneDNN is good. (Corollary: the int8-GEMM microbench has +/-30% run-to-run noise -> judge kernel changes at
   SERVE decode-t/s, never the microbench.)

3. **No native FP8 -> int8 is the real low-precision compute path; each quant has a distinct sweet spot
   (MEASURED Pareto, Qwen3-14B, capture + fp8-KV).** Xe2 has XMX INT8 (367 TOPS, s8s8s32 DPAS via oneDNN) but
   no FP8 ALU. The three quants are NOT a simple ladder -- they trade prefill vs decode:
   | quant | weights | PREFILL t/s | DECODE t/s (%BW) | sweet spot |
   |-------|---------|-------------|------------------|------------|
   | **W8A8**  | 14 GiB  | **5508** (best, int8xint8 XMX) | 26 (61.5%) | prefill/batch-heavy, accuracy |
   | **W4A8**  | 9.3 GiB | 4953 | 48 (73%) | balanced |
   | **W4A16** | 9.3 GiB | (no int8 fast-path) | **55** (best) | decode-heavy |
   The non-obvious finding: **W8A8 decode (26 t/s) trails W4A8 (48) on BOTH axes** -- more weight bytes AND a
   lower m=1 GEMM efficiency (the int4 grouped weight-decompression GEMV hits 73% of BW; the general int8
   jit:gemm:any is shape-sensitive -- wide-n MLP up/gate ~50%, tall down ~93% -> ~61% avg). PP-1 tried to lift
   the wide-n int8 layout and was perf-neutral, so this is a known oneDNN limit. **=> W8A8 is the PREFILL
   champion, not a decode upgrade; for decode pick W4A16/W4A8.**

4. **MTP (native Qwen3.6 multi-token-predict head) drafts accurately but is net-NEGATIVE on one card -- and
   it's toolchain-gated, not sampler-gated.** The head accepts 86.9% @ N=1 / 2.86 mean @ N=3 (great drafter,
   "the legs"), but wall-clock decode is -19% to -37% because the **eager-attention VERIFY** under PIECEWISE
   costs more than the accept saves. Proven by elimination: enabling the (Triton) rejection sampler changed
   nothing (25.5 vs 25.5 t/s). Net-positive needs FULL capture (captures attention) -> blocked by the SYCL-Graph
   `work_group_scratch_memory` limit (oneAPI DPC++ 2026.0 lifts it) + `TRITON_ATTN` being unwired on vLLM-XPU.

5. **Two reusable fixes worth keeping.** (a) **Triton-XPU enable:** the "0 active drivers -> Disabling Triton"
   is just `is_active() == torch.xpu.is_available()` gated per spawned process with an lru_cache; a 1-line
   `sitecustomize.py` warming `torch.xpu.device_count()` (via PYTHONPATH) fixes it -- no rebuild. (b) **vLLM-XPU
   + torch.compile** crashes on CUDA-only fusion passes -> disable them in `pass_config` (the capture recipe).

## Quant-pipeline discipline (hard-won)
- **Always verify which checkpoint is actually served** (RTN vs GPTQ silently corrupted a result once): query
  `/v1/models`, cross-check `models.yaml` -> the exact path, `served_model_id` must encode the method (`-gptq`/
  `-rtn`). Less-performant dups -> `models/archive/`.
- **AutoRound's llm_compressor export is W8A8-ONLY** (`check_and_reset_format` asserts `bits==8`) -- you cannot
  get W4A8 from it; use SmoothQuant+GPTQ (or RTN) for W4A8.
- **RTN is fast on CPU (data-free, ~90s for 27B); GPTQ/SmoothQuant calibration needs the GPU.**
- **27B is a VLM (Qwen3_5):** `AutoModelForCausalLM` collapses its config + misquantizes DeltaNet; use the
  27B-aware quantizer + a config graft to serve. group_size=128 is incompatible with its 4304-dim layers.

## What's left (all toolchain- or card-#2-gated, or low-ROI)
- **FULL graph capture (A2)** -> squeezes attention into the graph + flips MTP/spec-decode positive. Needs
  oneAPI DPC++ 2026.0 (work_group_scratch) and/or TRITON_ATTN wired on XPU. The single highest remaining lever.
- **Dual-card W8A8/W4A8 of 27B + 35B-MoE** -- the accel-quant run for card #2 (35B int8-MoE on XPU is likely
  no-go; W8A8 dense 27B is the target).
- **PP-2 hand DPAS joint_matrix GEMM** -- ~1.2x prefill for 1-2 weeks; low ROI given oneDNN is already 67-80%.
- **L1 wire the existing fused rmsnorm+int8-quant** -- a real kernel exists unwired, but under capture its
  dispatch saving is gone and its BW saving (~6 MiB/token) is negligible vs the multi-GiB weight read.
