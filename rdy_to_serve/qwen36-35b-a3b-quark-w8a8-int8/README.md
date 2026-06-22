# Qwen3.6-35B-A3B Quark W8A8 INT8 (int8 MoE) -- 2x B70, TP=2

Serves the int8 W8A8 MoE 35B (HF ckpt `nameistoken/Qwen3.6-35B-A3B-Quark-W8A8-INT8`, on the host at
`/mnt/vm_8tb/b70/models/Qwen3.6-35B-A3B-Quark-W8A8-INT8`) on two Intel Arc Pro B70s.

## Run (on the GPU host)
```bash
/mnt/vm_8tb/b70/gpu-run bash serve.sh        # start (TP=2, eager), wait healthy, gen-probe, leave serving
bash serve.sh bench                           # concurrency sweep (in 2048 / out 128, c=1 2 4)
bash serve.sh stop                            # stop + release the GPU
GRAPH=1 /mnt/vm_8tb/b70/gpu-run bash serve.sh run   # serve + bench + stop, with PIECEWISE graph capture
```
Endpoint: `http://192.168.10.5:8000/v1` (OpenAI-compatible). Served id: `qwen36-35b-a3b-quark-w8a8-int8`.

## [!] Image: `vllm-xpu-env:v0230` (vLLM 0.23.0). DO NOT use llm-scaler 0.14.x.
The 0.14.x image has no XPU MoE op suite (`vllm._moe_C` unbuilt) -> int8 MoE dies on `topk_softmax`. vLLM
0.23 routes the 256 int8 experts through the Triton `fused_moe_kernel` on XPU. See ../README.md + kernel/20.

## What `patches/quark.py` does (the one thing not already in vLLM 0.23)
vLLM 0.23 already has `QuarkW8A8Int8MoEMethod` + the dynamic-per-token int8 LINEAR dispatch. The ONLY gap on
XPU is the int8 scaled-mm LINEAR kernel (registry has no XPU entry -> `KeyError: PlatformEnum.XPU`). The patch
reroutes the int8 linear layers (`linear_attn.*`, `mlp.shared_expert.*`) to `QuarkW8A8Int8DequantXPU` --
a weight-only int8->bf16 dequant GEMM (effectively W8A16; activations not quantized -> MORE accurate than the
checkpoint's W8A8, correctness-first). The 256 routed experts stay **TRUE int8** via the Triton MoE kernel.
It is bind-mounted over the image's `quark.py` at run time (no image rebuild).

## Recipe details (baked into serve.sh)
- TP=2 (the int8 weights are ~35 GB -> 17.5 GiB/card; does NOT fit one 32 GB card).
- Battlemage multi-GPU stability env (vLLM #41663): `CCL_TOPO_P2P_ACCESS=0`, `CCL_ZE_IPC_EXCHANGE=pidfd`,
  `CCL_ENABLE_SYCL_KERNELS=0` (eager) / `=1` (graph capture), `SYCL_UR_USE_LEVEL_ZERO_V2=0`, OFI, spawn.
- Text-only VLM serve: `--limit-mm-per-prompt {image:0,video:0}` (skips the vision encoder).
- Do NOT pin `ONEAPI_DEVICE_SELECTOR`/`ZE_AFFINITY_MASK` for TP=2 (aborts the model-inspect subprocess).

## Verified
Load 17.54 GiB/card, KV 10.2 GiB, concurrency 89x@8192, `backend=xccl world_size=2`, Triton
`fused_moe_kernel` (E=256,N=256,int8). Gen: "The capital of France is" -> " Paris, a city renowned for its
rich history, culture, and iconic landmarks."

Perf (random 2048-in/128-out, TP=2):
- **eager** (default): c1 4.80 per-stream decode (e2e 4.46), c2 8.16 agg, c4 14.08 agg. Works at all conc.
- **GRAPH=1** (PIECEWISE capture; default `WARMUP=1` warms the compile cache so c>1 does not stall):
  agg / per-stream-decode t/s = c1 20.0/25.9, c2 33.0/21.3, c4 **45.7**/17.5 -> **3.2-4.5x aggregate, ~4-5x
  decode** vs eager (single-stream decode varies ~25-41 t/s run-to-run). Cold start adds a ~6 min one-time
  compile (cached in /vllm_cache). `CGMODE=FULL_DECODE_ONLY` is BLOCKED on stock v0230 (SYCL-Graph scratch).
- Open levers (../../FINDINGS.md, docs/kernel/20, RESEARCH_TODO Track 9): a tuned `E=256,N=256` MoE config
  (XPU tuner needs porting); and true-int8 linear via the XMX/DPAS Triton kernel (B70_INT8_LINEAR=triton).

verified: _common smoke=GREEN (eager, TP=2, 2026-06-23, 233s): HEALTHY, id ok, quark.py mount ok,
gen "Paris, a city renowned for its rich history, culture, and iconic landmarks." Re-verify: `bin/serve-sweep --smoke`.
