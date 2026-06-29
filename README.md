# Qwen3.6 on dual Intel Arc B70 -- INT8-first serving on XPU

Serving **Qwen3.6-27B** (dense VLM) and **Qwen3.6-35B-A3B** (MoE VLM) on 2x Intel Arc Pro B70
(Battlemage, `xpu`). The headline: custom **fused INT8 W8A8** kernels make 8-bit weights beat
emulated-fp8 and bf16 on prefill, TTFT, *and* decode -- vision tower + MTP head retained, zero
accuracy loss. **sglang is the production backend; vLLM is paused** (see the vLLM table).

## What we built (the load-bearing custom work)

- **Fused INT8 W8A8 oneDNN GEMM kernels** -- `int8_gemm_w8a16` (decode, fp16-act) + `int8_gemm_w8a8`
  (prefill, s8xs8 XMX), built from source + a decode/prefill-routing shim that replaces the 3-launch
  `torch._int_mm` chain. W8A8 27B: **25.6 t/s** decode, HumanEval+ **0.970/0.933** (> int4 same-stack).
  -> `kernels/`, `sglang/patches/w8a8_shim.py`, `research/w8a8/W8A8_BUILD.md`.
- **NEXTN MTP (speculative decode) on XPU** -- first stable break of the ~9.4 t/s eager decode
  ceiling, correct under concurrent load (4 CUDA/NPU-only gate fixes in one shim).
  -> `sglang/patches/mtp_tree_xpu.py`.
- **GPU-free vision + MTP graft** -- the quantizer silently dropped all 333 `visual.*` tensors; a
  pure-CPU graft restores them + adds the bf16 MTP head (no requant). -> `sglang/graft_vision.py`, `graft_mtp.py`.
- Also: first sglang-XPU decode **XPUGraph** capture (`sglang/patches/xpu_cudagraph.py`, 2.5x eager
  single-stream); **W4A8 hybrid** int4-weight kernel (1.83x decode / 1.9x prefill vs woqgemm); a custom
  **L0-IPC push-allreduce** for TP=2 + the BCS/oneCCL **wedge guard** (`bin/xpu-health`, `bin/xe-reset`).

## Serve shelf -- sglang (production)

Warm bench, IN=2048 / OUT=128. TG = per-stream decode t/s; c1 = 1 stream, c4 = 4 concurrent. KV =
engine-allocated KV cache. Each row is `rdy_to_serve/sglang/<dir>/serve.sh` at *its own* best config.

| Model | Quant (kernel) | Wt GB | TP | PP | TTFT | TG c1 | TG c4 | KV avail |
|---|---|---|---|---|---|---|---|---|
| qwen3.6-27b | int4-AutoRound (W4A16) + NEXTN MTP | 19 | 1 | 1 | 921 ms | 15.3 | 4.5 | 29.8k tok |
| qwen3.6-27b | W4A8 hybrid (int4-w / int8-a, XPUGraph) | 19 | 1 | 1 | 993 ms | 27.3 | 27.7\* | 145k tok |
| qwen3.6-27b | **W8A8 int8 fused + NEXTN MTP** | 35 | 2 | 1 | **541 ms** | **25.6** | 5.8 | 182k tok |

\* W4A8 is the single-stream XPUGraph driver (`max-running-requests=1`): at c4 the 4 requests serialize,
so per-stream decode holds ~27.7 t/s but TTFT balloons to ~8.1 s. Best for single-stream throughput.

## Serve shelf -- vLLM (PAUSED)

> **vLLM is paused.** Concurrent prefill+decode trips the XPU GDN/Mamba SSM-state NaN poison ->
> endless `!!!!` (then global until a restart). Tracking issue: **vLLM #38994** (Qwen-3.5 garbled
> output, Intel B70, open since 2026-04); the unfixed XPU piece is **vllm-xpu-kernels #172** (fp32
> `ssm_state` in `chunk_fwd_o`). Kept as a maintained baseline; resume when #172 lands. See `SHORTCOMINGS.md`.

Numbers at each entry's own production config (`GRAPH=1` PIECEWISE capture -- the ~4x decode lever).

| Model | Quant | Wt GB | TP | PP | TTFT | TG c1 | TG c4 | KV avail |
|---|---|---|---|---|---|---|---|---|
| qwen3.6-27b | int4-AutoRound (W4A16) | 19 | 1 | 1 | 1289 ms | 28.6 | 19.5 | 103k tok |
| qwen3.6-27b | W4A16 (compressed-tensors) + MTP | 26 | 2 | 1 | 3145 ms | 22.1 | 8.9 | 172k tok |
| qwen3.6-27b | W4A8-sqgptq (int8-act) | 26 | 1 | 1 | 1085 ms | 6.3\* | 5.8\* | OOM @GRAPH=1 |
| qwen3.6-27b | W8A8-sqgptq (int8) + MTP | 35 | 2 | 1 | 795 ms | 22.4 | 15.7 | 76k tok |
| qwen3.6-35b-a3b | int4-AutoRound (W4A16 MoE) | 21 | 1 | 1 | **441 ms** | **67.7** | 43.8 | 270k tok |
| qwen3.6-35b-a3b | Quark W8A8-INT8 (MoE) | 35 | 2 | 1 | 1502 ms | 43.1 | 22.2 | 684k tok |

\* W4A8: at `GRAPH=1` the capture buffers leave only 0.32 GiB for KV -> engine init OOMs (est. max len
2496); EAGER numbers shown. It is the one vLLM entry without a working captured config.

## Where to look

- `AGENTS.md` (= `CLAUDE.md`) -- standing rules, backend-split layout contract, shelf rules.
- `rdy_to_serve/<backend>/<model-quant>/serve.sh` -- the verified serve shelf (one best config each).
- `research/LESSONS.md` -- optimization ledger: what generalizes across quants + backends.
- `SHORTCOMINGS.md` -- open blockers + dead ends (the honest negatives).
- `models/` -- model registry (`manifest.yaml` + `fetch.sh`); weights in `models/files/` (git-ignored).
- `kernels/` -- shared custom gemm source; `research/w8a8/`, `research/w4a8/` -- the kernel campaigns.
- `FINDINGS.md` / `JOURNAL.md` -- raw measurements / chronological lab log.

## Environment

2x Intel Arc Pro B70 (Battlemage, 32 GB each), Ubuntu kernel 7.0, run locally as `hotschmoe`. Images:
sglang `sglang-xpu:mtp`, vLLM `vllm-xpu-env:{v0230,v0230moe,int8g}`. Every GPU touch goes through the
shared lease (`bin/gpu-run`); TP=2 = both cards. The smoke/bench gate is `bin/serve-sweep`.
