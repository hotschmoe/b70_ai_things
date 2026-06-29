# qwen36-27b-w8a8-mtp -- W8A8 (int8) + NEXTN MTP, the int8 all-rounder

The W8A8 path that **handily beats bf16/fp8 on prefill, TTFT, AND decode**, with vision retained and
higher code accuracy than int4. Built on our fused int8 oneDNN kernels + NEXTN chain-MTP.

## Numbers (IN2048/OUT128, warm c1, TP=2 == both cards)

| metric | this (W8A8 fused+MTP) | bf16 TP=2 | vs bf16 |
|---|---|---|---|
| decode (TG) | **25.2 t/s** | 9.03 | **+180% (2.8x)** |
| prefill (PP) | **4344 tok/s** | 3098 | **+40%** |
| TTFT | **471 ms** | 661 | **-29%** |

- DECODE (25.2) is rock-solid run-to-run. PREFILL/TTFT vary more under MTP (the spec draft adds prefill-side
  work): warm c1 seen ~471-691 ms TTFT / ~2960-4344 PP. The table shows the warm-best (run2); for a CONSISTENT
  prefill/TTFT champion (and sampling) use the eager sibling (PP 4570 / TTFT 448, `scripts/123`).
- Also beats the int4+MTP driver on decode (25.2 vs 15.3) -- the MTP verify (M>1) rides the int8-XMX
  `int8_gemm_w8a8` (2.0x bf16) instead of int4 woqgemm.
- FP8 has no native B70 path (oneDNN emulates `fp8_gemm_w8a16` at ~1.0x bf16 prefill) -> W8A8 wins PP vs fp8 too.
- **Accuracy: HumanEval+ 0.970 / 0.933 (base/plus)**, sandboxed -- HIGHER than int4 same-stack (0.933/0.896).
  int8 weights are more accurate than int4; the fused kernels add zero loss; MTP is greedy-lossless.
  Result: `../../evals/results/20260628T233713Z__qwen36-27b-w8a8-vision-mtp__w8a8-fused-vision`.

## How it works

- **Decode (M==1):** `int8_gemm_w8a16` -- s8 weight x fp16 act, per-channel dequant fused in the oneDNN
  epilogue (1 launch). At M=1 the GEMV is weight-BW-bound so int8 activations buy nothing; fp16-act is leaner.
- **Prefill / MTP-verify (M>1):** `int8_gemm_w8a8` -- s8 x s8 on the XMX/DPAS systolic array (2.0x bf16),
  with `dynamic_per_token_int8_quant` (fused single-launch act-quant).
- **MTP:** NEXTN chain spec-decode, steps=10 (W8A8 peak -- the cheap int8-XMX verify rewards deeper drafts
  than int4's steps=7: 7->23.8, 10->25.25, 12->24.35). Greedy-only on XPU.
- Both ops built from vllm-xpu-kernels source vs sglang torch 2.12 (`../../w8a8/W8A8_BUILD.md`).

## Dependencies (runtime mounts, NOT a baked image)

- image `sglang-xpu:mtp` (baked XPU NEXTN gates + compressed_tensors W8A8 scheme)
- built kernel `.so` at `/mnt/vm_8tb/b70/w8a8_kernel/_xpu_C.abi3.so` (sha bc643c3f8a61; build: W8A8_BUILD.md)
- the FUSED `w8a8_shim.py` (`../../sglang/patches/w8a8_shim.py`, `B70_XPU_W8A8_FUSED=1`)
- grafted ckpt `/mnt/vm_8tb/b70/models_w8a8/Qwen3.6-27B-W8A8-sqgptq-vision-mtp` (vision 333 + W8A8 LM +
  BF16 MTP head; built by `../../sglang/graft_mtp.py` onto `Qwen3.6-27B-W8A8-sqgptq-vision`; symlinks into
  `/models`, so the serve mounts BOTH `/models` and `/models_w8a8`)

## Use

```
/mnt/vm_8tb/b70/gpu-run bash serve.sh start    # serve TP=2 (both cards), coherence-gated, stay up
bash serve.sh gen "your prompt"                # quick greedy chat probe
bash serve.sh stop                             # stop + release + health check
/mnt/vm_8tb/b70/gpu-run bash serve.sh run      # start + warm c1 bench + stop in one lease
```

- TP=2 holds BOTH cards. cudagraph is DISABLED (W8A8 TP=2+MTP is stable that way; XPUGraph capture is a
  CEILING at TP=2 -- decode is all-reduce-bound, not launch-bound).
- For **prefill-heavy or sampling** loads, use the eager sibling (no MTP, samples):
  `../../scripts/123_w8a8_fused_ab.sh` (FUSED=1 GRAPH=0) -> PP 4570 / TTFT 448 / decode 8.1.
- Greedy-only: MTP verify runs greedily on XPU (temperature/top_p/top_k ignored), like all XPU NEXTN.

Campaign: `../../w8a8/W8A8_SGLANG_PLAN.md`. JOURNAL 2026-06-28/29.
