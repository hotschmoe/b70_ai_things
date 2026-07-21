# Qwen3.6-27B W4A8 (int4 weights / int8 activations, SmoothQuant+GPTQ, prepacked)

Serves `qwen36-27b-w4a8-sqgptq` on ONE Intel Arc Pro B70 (default card 1, the research card).
The int8-activation / int8-XMX path on the 27B: GDN + lm_head + visual (333 tensors) + mtp (15
tensors) kept bf16 in-checkpoint; linears are offline-prepacked int32 [out, in/8] symmetric
group-128 int4 with dynamic per-token int8 activations. fp16 KV, vision ON, MTP spec=3.

> [!] 2026-07-21: PORTED to vLLM 0.25.1 (`vllm-xpu-env:int8g-v0251`, TP=1 PIECEWISE+MTP).
> **UNVERIFIED -- no GPU run yet; the coordinator gates (smoke + coherence) before this lands.**
> The old v0.23 recipe (`vllm-xpu-env:int8g`, eager-ish, fp8-KV note) is in git history of this dir.

## Run (on the GPU host)
```bash
cd /mnt/vm_8tb/github/b70_ai_things
NAME=w4a8_c1 PORT=18079 ./bin/gpu-run --card 1 bash rdy_to_serve/vllm/qwen36-27b-w4a8/serve.sh start
bash rdy_to_serve/vllm/qwen36-27b-w4a8/serve.sh stop
```
Endpoint: `http://<host>:18079/v1`. Served id: `qwen36-27b-w4a8-sqgptq`.

## How the 0.25.1 port works (what changed vs the v0.23 recipe)
- **Upstream drift in our favor:** vLLM 0.25.1 UPSTREAMED our two v0.23 patch classes. The image's
  `mixed_precision/xpu.py` ships `XPUwNa16LinearKernel` + `XPUW4A8IntLinearKernel`, and
  `linear/__init__.py` already registers `XPUW4A8IntLinearKernel` first in
  `_POSSIBLE_KERNELS[PlatformEnum.XPU]` (both verified in-image 2026-07-21). **No registry patch
  needed.** The two mounts carry only the b70 deltas on top of verbatim upstream content:
  - `patches/compressed_tensors_w4a8_int.py`: `VLLM_W4A8_PREPACKED` -> allocate weight int32
    [out, in/8] so the prepacked checkpoint loads directly (upstream would unpack-then-pack, a
    ~28 GiB GPU transient a 32 GB B70 cannot fit on the 27B).
  - `patches/xpu.py`: the matching skip-pack branch; plus the OPT-IN `B70_W4A8_HYBRID=N` small-M
    route through the quant-free fp16-act `int4_gemm_w4a16` (sglang-proven 1.83x at M==1; zero
    extra weight memory -- shares the packed-weight `.t()` NT view) and its lazy `register_fake`.
- **Kernels:** no build. `$ROOT/w8a8_kernel_v0240/_xpu_C.abi3.so` (the same torch-2.12 .so the W8A8
  v0251 shelf mounts) already carries `int4_gemm_w4a8` + `int4_gemm_w4a16` + GDN. The
  `int4_gemm_w4a8` fake for PIECEWISE capture is baked in the image's `scaled_mm/xpu_int8.py`.
- **`patches/sitecustomize.py`** (PYTHONPATH mount, blocks numbered like the W8A8 shelf's):
  (1) BF16 MTP drafter (else 0% accept), (4) XPU mamba USM-pointer fix (unblocks
  `PREFIXCACHE=1`), (6) drafter-eager fallback (opt-in), (7) NEO graph-replay reclaim
  (default ON via `CGRECLAIM`, the linear_stream.h:84 fix -- single-card captured+MTP is not
  exempt). W8A8's TP-only capture-safe all_gather block is dropped (TP=1).
- **Serve defaults:** `DTYPE=float16` (the int4 ops emit fp16; `DTYPE=auto` bf16 is the numerics
  rollback), `IGP=false` (inductor partitioner KeyErrors on the mixed quantized+BF16-GDN region),
  `UTIL=0.85 MAXLEN=8192 MAXSEQS=4 CAPSIZES=1,2,4 MTPTOK=3`, vision auto-gates
  `--skip-mm-profiling` + `VLLM_USE_AOT_COMPILE=0`.

## Checkpoint sanity (checked 2026-07-21)
`models/files/qwen3.6-27b/w4a8-sqgptq/`: `quantization_config.is_prepacked_w4a8: true`; ignore list
is the CORRECT regex form (`re:.*linear_attn.*`, `re:.*visual.*`, `re:.*mtp.*`, `lm_head`); index
maps 333 `model.visual.*` + 15 `mtp.*` tensors; linear weights are I32 [out, in/8].

verified: NOT YET on 0.25.1 (pending coordinator gate). Prior v0.23 recipe: smoke GREEN 2026-06-23.
Re-verify: `bin/serve-sweep --smoke`.
