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

## [!] Image: `vllm-xpu-env:v0240` (vLLM 0.24.0, torch 2.12) -- PORTED 2026-07-04. NEVER llm-scaler 0.14.x.
Rollback = `IMG=vllm-xpu-env:v0230` (serve.sh auto-selects the v0.23 `patches/quark.py` for any non-v0240
image). The 0.14.x image has no XPU MoE op suite (`vllm._moe_C` unbuilt) -> int8 MoE dies on `topk_softmax`.
vLLM routes the 256 int8 experts through the in-tree Triton `fused_moe`.

## What `patches/quark_v0240.py` does (re-grafted from the proven v0.23 patch)
Stock `QuarkW8A8Int8` still `KeyError`s on XPU in v0.24.0 (`_POSSIBLE_INT8_KERNELS` has no XPU entry). The
patch reroutes the int8 linear layers (`linear_attn.*`, `mlp.shared_expert.*`) to `QuarkW8A8Int8DequantXPU`
-- a weight-only int8->bf16 dequant GEMM (effectively W8A16; MORE accurate than the checkpoint's W8A8,
correctness-first). The 256 routed experts stay **TRUE int8** via the in-tree `QuarkW8A8Int8MoEMethod`
(Triton `fused_moe`, unchanged upstream). `replace_parameter` moved to `utils.layer_utils` in v0.24.0.
`B70_INT8_LINEAR=native` opts into stock int8 linear (needs `IMG=int8g-v0240` + the runtime `.so`; not the
default -- int8 linear is no speed win on this MoE, linear is the minority path). Mount-not-bake.

## Capture on v0.24.0 needs the IGC-fusion workaround (the hard part of this port)
PIECEWISE capture aborted compiling a fused RMSNorm-into-matmul kernel ("IGC Internal Compiler Error:
Floating point exception", ocloc err 245) in ANY GRF mode. This int8 MoE needed BOTH levers (baked in serve.sh):
- **`IROP`** = `--ir-op-priority` forcing rms_norm/fused_add_rms_norm to `["xpu_kernels","native"]` (the
  OPAQUE custom-op impl) so inductor cannot fuse the reduction into the router matmul.
- **`INDUCTOR`** = all inductor fusion off (combo/prologue/epilogue). The int8 MoE's GDN-region kernel is a
  scheduler-level decomposed-mm fusion the template knobs do not govern; only opaque-rms_norm + fusion-off
  together compile. (The int4 MoE needed only `prologue_fusion=false` -- its variant was a template fusion.)

## Recipe details (baked into serve.sh)
- TP=2 (the int8 weights are ~35 GB -> 17.5 GiB/card; does NOT fit one 32 GB card).
- Battlemage multi-GPU stability env (vLLM #41663): `CCL_TOPO_P2P_ACCESS=0`, `CCL_ZE_IPC_EXCHANGE=pidfd`,
  `CCL_ENABLE_SYCL_KERNELS=0` (eager) / `=1` (graph capture), `SYCL_UR_USE_LEVEL_ZERO_V2=0`, OFI, spawn.
- Text-only VLM serve: `--limit-mm-per-prompt {image:0,video:0}` (skips the vision encoder).
- Do NOT pin `ONEAPI_DEVICE_SELECTOR`/`ZE_AFFINITY_MASK` for TP=2 (aborts the model-inspect subprocess).

## Verified (v0.24.0, 2026-07-04)
Coherent gen "Paris, a city renowned for its rich history, culture, and iconic landmarks." KV **774,516 tok**
(94.5x @ 8192, up from 684k on v0.23), `world_size=2`, TRUE int8 `fused_moe`. Capture completes (compilation
~119s) with the IROP + INDUCTOR fix; 0 IGC crashes.

Perf (v0.24.0, IN=2048/OUT=128, warm; per-stream decode / aggregate-out):
- c1 37.0 / 28 . c2 29.6 / 47 . c4 23.5 / 65 tok/s . TTFT **1061 ms** (down from 1502 on v0.23, -29%).
- **c1 DEGRADES run-over-run under sustained back-to-back load** (37.0 -> 25.5 -> 19.5 across three c1 benches;
  TTFT stays ~1050ms -> clock/thermal, the display-attached card1 downclocks under TP=2, NOT a leak). c1 (37)
  is also below the old v0.23 43.1 partly because dodging the IGC crash needs inductor fusion OFF (a decode cost).
- Spec-decode is architecturally ineffective on this A3B MoE (M5: MTP +3% flat); no recent lever reaches
  90-100 t/s single-stream. Aggregate scales (65 tok/s @ c4). The port's wins are TTFT + KV capacity + v0.24.0
  parity, not decode.

## Historical v0.23 perf (rollback reference, random 2048-in/128-out, TP=2):
- **eager** (default): c1 4.80 per-stream decode (e2e 4.46), c2 8.16 agg, c4 14.08 agg. Works at all conc.
- **GRAPH=1** (PIECEWISE capture; default `WARMUP=1` warms the compile cache so c>1 does not stall):
  agg / per-stream-decode t/s = c1 20.0/25.9, c2 33.0/21.3, c4 **45.7**/17.5 -> **3.2-4.5x aggregate, ~4-5x
  decode** vs eager (single-stream decode varies ~25-41 t/s run-to-run). Cold start adds a ~6 min one-time
  compile (cached in /vllm_cache). `CGMODE=FULL_DECODE_ONLY` is BLOCKED on stock v0230 (SYCL-Graph scratch).
- Open levers (../../FINDINGS.md, docs/kernel/20, RESEARCH_TODO Track 9): a tuned `E=256,N=256` MoE config
  (XPU tuner needs porting); and true-int8 linear via the XMX/DPAS Triton kernel (B70_INT8_LINEAR=triton).

verified: _common smoke=GREEN (eager, TP=2, 2026-06-23, 233s): HEALTHY, id ok, quark.py mount ok,
gen "Paris, a city renowned for its rich history, culture, and iconic landmarks." Re-verify: `bin/serve-sweep --smoke`.
