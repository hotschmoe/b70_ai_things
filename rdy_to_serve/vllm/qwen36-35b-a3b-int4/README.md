# Qwen3.6-35B-A3B MoE int4-AutoRound -- FASTEST single-card decode

Serves `Intel_Qwen3.6-35B-A3B-int4-AutoRound` (weights at `models/files/qwen3.6-35b-a3b/int4-autoround`)
on ONE Intel Arc Pro B70. 35B total / ~3B active (A3B MoE, 256 routed experts) -> more knowledge than the
27B at higher aggregate decode.

## Run (on the GPU host)
```bash
/mnt/vm_8tb/b70/gpu-run --card 0 bash serve.sh          # start (PIECEWISE capture), wait healthy, gen-probe
bash serve.sh stop                                       # stop + release the GPU
GRAPH=1 /mnt/vm_8tb/b70/gpu-run --card 0 bash serve.sh run   # serve + concurrency sweep + stop, one lease
```
Endpoint: `http://<host>:8000/v1`. Served id: `qwen36-35b-a3b-int4`.

## [!] Image: `vllm-xpu-env:v0240` (vLLM 0.24.0, torch 2.12) -- PORTED 2026-07-04
Rollback = `IMG=vllm-xpu-env:v0230moe` (the old baked-patch leaf; drop the patch mount there).
v0.24.0 rewrote INC into a package (`inc/` + `schemes/`). The old `contrib/vllm_moe_xpu` `inc.py`
`RoutedExperts -> MoeWNA16` routing patch is re-ported as **`patches/inc_wna16_scheme.py`**, now
MOUNT-not-bake. Upstream `get_moe_method` still hard-returns `UnquantizedFusedMoEMethod` on XPU (which
bf16-inflates the 256 int4 experts to ~70 GB -> OOM on a 32 GB card); the patch routes gptq/awq-packed
experts to the pure-Triton `MoeWNA16` path, skipping the CUDA-only Marlin probes. v0.24.0 packages
`gdn_attention` + `int4_gemm_w4a16` in the stock kernel `.so`, so **no kernel mounts are needed**.

## Two capture-compat gotchas (baked into the patch + serve.sh)
1. **ARK graph-break.** v0.24.0's in-tree INC XPU int4 linear defaults to `auto_round_kernel` (ARK)
   `woqgemm`, a ctypes call that is NOT dynamo-traceable -> hard graph-break under `torch.compile`. The
   patch gates ARK behind `B70_INC_ARK=1` (eager only) and defaults to the capturable in-tree
   `torch.ops._xpu_C.int4_gemm_w4a16`. (This is slower at M=1 decode than ARK -- see the c1 note below.)
2. **IGC compile crash.** PIECEWISE capture aborted compiling a fused RMSNorm-into-router-matmul kernel
   ("IGC Internal Compiler Error: Floating point exception", ocloc err 245, in ANY GRF mode). FIX =
   `INDUCTOR={combo_kernels,benchmark_combo_kernel,prologue_fusion:false}` (baked). This int4 variant is a
   matmul-template prologue fusion, so `prologue_fusion=false` alone kills it. (The int8 MoE needs a heavier
   fix -- see that entry.)

## Recipe (baked into serve.sh)
- GRAPH=1 PIECEWISE capture. DTYPE=auto, UTIL=0.90, MAXLEN=8192, MAXSEQS=64, KVDTYPE=fp8_e5m2.
- CAPSIZES=1,2,4,8,16,32,64. TOOLCALL=1 / qwen3_coder / REASONPARSER=qwen3. Vision ON (VLLM_USE_AOT_COMPILE=0).

## Verified perf (v0.24.0, IN=2048/OUT=128, warm; per-stream decode / aggregate-out)
- c1 46.5 / 39 . c2 57.6 / 89 . c4 44.0 / 116 tok/s . TTFT 558 ms . KV **484,498 tok** (59x @ 8192). Coherent.
- **c1 (46.5) is DOWN from the historical v0230moe 67.7** -- the ARK int4-linear kernel is gated off for
  capture-compat (gotcha 1); the in-tree `int4_gemm_w4a16` is slower at M=1. c4 (44.0) matches the old 43.8.
  c1 < c2 (46.5 < 57.6) is the launch-bound MoE signature (~3B active is compute-light at batch=1).
- Spec-decode does NOT help this MoE (2026-06-22 M5: MTP = +3% flat vs +79% on the dense 27B). MoE headline
  is CAPTURE, not MTP. There is no recent lever that lifts single-stream toward 90-100; aggregate scales fine.

verified 2026-07-04: coherent gen "Paris", gate-clean start, benched captured. Re-verify via `serve.sh run`.
