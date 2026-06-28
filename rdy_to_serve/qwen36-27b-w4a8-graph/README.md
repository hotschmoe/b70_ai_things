# qwen36-27b-w4a8-graph -- FASTEST single-stream daily driver (W4A8/W4A16 hybrid + XPUGraph)

Qwen3.6-27B served from the **proven Lorbus int4-AutoRound** checkpoint (multimodal
`Qwen3_5ForConditionalGeneration` -- **VISION retained**, full GDN+MLP int4), but with its int4 linears
dispatched to the **oneDNN `int4_gemm` ops** instead of auto_round `woqgemm`, stacked under
**`torch.xpu.XPUGraph` decode capture**:

- **decode** (M==1) -> `int4_gemm_w4a16` (fp16 act) -- numerically == woqgemm (relerr ~1e-3), but a faster oneDNN kernel
- **prefill** (M>1) -> `int4_gemm_w4a8` (per-token symmetric int8 act) -- ~1.9x faster prefill than woqgemm, lower TTFT.
  The act-quant runs as a single-launch **Triton** kernel (`w4a8_actquant_triton.py`), 8.3x faster than the
  eager chain -> cuts another ~12% off TTFT (`B70_W4A8_TRITON_AQ=0` forces the eager fallback).

Same int4 weights as `qwen36-27b-int4-graph`, faster kernel. **SAMPLING-capable** (honors temperature/top_p),
soak-stable, coherent, GDN-correct under mixed load, no wedge.

## Headline -- CLEAN same-session head-to-head (card 0, GRAPH=1, warm c1, bench2048 IN2048/OUT128; 2026-06-28)
Both serves benched back-to-back, same machine state, 1st run discarded, 2 recorded runs (see `../../sglang/W4A8_PLAN.md`).
| metric | W4A8-graph (this, Triton act-quant) | int4-woqgemm graph champion |
|---|---|---|
| decode c1 (warm) | **27.3 t/s** (int4 lm_head, `LMHEAD=1` default) | 23.5 t/s  (+16%) |
| decode c1 (warm), bf16 lm_head (`LMHEAD=0`) | 25.3 t/s | 23.5 t/s  (+7.8%) |
| decode soak (2000-tok), bf16 lm_head | 24.7 t/s (stable, 1.14x) | 23.0 t/s |
| TTFT | **~935 ms** | ~1159 ms  (-19%) |
| prefill PP (tok/s) | **2189** | 1766  (+24%) |
| eager fallback (GRAPH=0) | 9.7 t/s decode + faster TTFT | 9.4 t/s |
| VRAM | ~17.4 GB weights | 17.4 GB |

(Earlier eager-act-quant build: decode 25.2, TTFT 1054, PP 1944. The Triton act-quant cut TTFT another ~11% / +13% PP.)

**Accuracy GATE -- PASS (HumanEval+ 164, thinking-off, greedy, sandboxed):** on the SAME sglang stack +
SAME int4 weights, W4A8 hybrid scores **0.921 base / 0.896 plus** == the int4-woqgemm champion's
**0.921 / 0.896** (delta 0.000 / 0.000). The int8-act prefill (relerr ~9e-3, prompt-encoding only; decode is
fp16-act) causes ZERO coding-accuracy loss. (The absolute 0.921 is depressed by max_tokens=2048 truncation of
verbose solutions, not quant damage.) Full analysis: `../../sglang/W4A8_PLAN.md` (ACCURACY GATE 2026-06-28d),
`../../evals/results/SUMMARY.md`.

**int4 lm_head (`LMHEAD=1`, the +8% decode lever) -- accuracy HELD:** the Lorbus ckpt leaves lm_head BF16
(2.54 GB, read every decode step); RTN-quantizing it to int4 g32 and routing the logits GEMV through the same
captured `int4_gemm_w4a16` op gives **decode 27.3 t/s (+7.9%)** with TTFT/prefill UNCHANGED (lm_head is M=1
even in prefill). Same-session HumanEval+: int4 lm_head **0.933 / 0.896** vs bf16 lm_head **0.921 / 0.896** --
NO regression (base +0.012 via greedy cross-kernel non-determinism, plus identical). Despite a high ~10% naive-RTN
weight relerr (lm_head is outlier-heavy), code argmax margins are large so greedy is unaffected. Set `LMHEAD=0`
to revert to the bf16 lm_head. Details: `../../sglang/W4A8_PLAN.md` (int4 lm_head 2026-06-28g).

## How it works (the frontier win)
The int4 weights live on disk in auto_gptq packing; `woq_shim.py` (`_XpuW4A8WoqKernel`, opt-in
`B70_XPU_W4A8_WOQ=1`) converts them ONCE to the `int4_gemm` op layout (a pure relayout -- numerically gated
by `../../sglang/w4a8_from_woq_probe.py`) and routes the `GPTQLinearScheme` kernel hook to the hybrid. The
`int4_gemm_w4a16` decode op is **XPUGraph-capturable** (bs=1 decode = a single op, no data-dependent act-quant),
so the 25 t/s capture win stacks on top. The prefill `int4_gemm_w4a8` act-quant is a single-launch **Triton**
kernel (torch.compile of it HANGS serve startup -- inductor async-worker deadlock; Triton JITs in-process, no
hang). `B70_XPU_CUDAGRAPH=1` + `--attention-backend triton` enable capture (same as int4-graph).

## REQUIREMENTS (this is NOT a baked image -- it has runtime mounts)
- **Image:** `sglang-xpu:mtp` (the champion int4-graph image; the multimodal Qwen3_5 path is baked).
- **Kernel:** the built `_xpu_C.abi3.so` at `/mnt/vm_8tb/b70/w4a8_kernel/` (50 MB, **NOT in git**;
  build per `../../sglang/W4A8_BUILD.md`; sha256 `63c8be3d26c8...`). `B70_XPU_C_SO` points the shim at it.
- **Shim:** `../../sglang/patches/woq_shim.py` (mounted over the baked copy; carries `_XpuW4A8WoqKernel`)
  + `../../sglang/patches/w4a8_actquant_triton.py` (mounted; the Triton prefill act-quant kernel).
- **oneAPI LD_LIBRARY_PATH:** the container PREPENDS `/opt/intel/oneapi/compiler/2025.3/lib` (required, or the
  ctypes-loaded .so resolves but torch loses the XPU device). serve.sh does this automatically.
`serve.sh start` preflights all of the above and refuses if any are missing.

## Run (on the GPU host)
```bash
cd /mnt/vm_8tb/github/b70_ai_things/rdy_to_serve/qwen36-27b-w4a8-graph
/mnt/vm_8tb/b70/gpu-run --card 0 bash serve.sh start   # serve, capture at startup, coherence-gated probe
bash serve.sh bench                                     # warm c1 (pp/ttft/tg @ ctx2048) + soak
bash serve.sh stop
/mnt/vm_8tb/b70/gpu-run --card 0 bash serve.sh run      # start + bench + stop in one lease
```
Endpoint: `http://<host>:30000/v1`. Served id: `qwen36-27b-w4a8-graph`. Capture takes ~50s at startup; sglang
`/health` first returns 200 ~150s after launch.

## [!] Pin card 0 -- the cards are asymmetric
`xe` drives the console/display off one Arc card; `ZE_AFFINITY_MASK=1` (card 1) is downclocked
(card 0 ~25 t/s vs card 1 ~15 t/s). serve.sh defaults `DEVICE=0` (the fast compute card).

## [!] Single-stream driver -- use DP=2 for concurrency
`--max-running-requests 1` + a single captured `bs=1` graph. For >1 user run **DP=2**
(`../../sglang/serve_dp2_w4a8.sh` -> 2 single-card replicas, card0 ~25 + card1 ~15, wedge-proof: no
cross-card collective), NOT a higher `max-running-requests` here.

## Driver matrix
| driver | c1 t/s | sampling | cards | use |
|---|---|---|---|---|
| **W4A8 + XPUGraph + int4 lm_head (this)** | **~27.3** | yes | 1 | FASTEST single-stream + lowest TTFT (~935 ms); vision |
| W4A8 + XPUGraph DP=2 (`../../sglang/serve_dp2_w4a8.sh`) | ~25 + ~15 | yes | 2 | 2 users, wedge-proof (bf16 lm_head) |
| int4 + XPUGraph (`../qwen36-27b-int4-graph`) | ~23.5 | yes | 1 | prior champion (woqgemm kernel) |
| int4 + NEXTN MTP steps=7 (`../qwen36-27b-int4-mtp`) | ~15.3 | greedy only | 1 | superseded by graph |
