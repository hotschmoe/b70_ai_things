# Qwen3.6 on dual Intel Arc B70 -- INT8-first serving on XPU

Serving **Qwen3.6-27B** (dense VLM) and **Qwen3.6-35B-A3B** (MoE VLM) on 2x Intel Arc Pro B70
(Battlemage, `xpu`). The headline: custom **fused INT8 W8A8** kernels make 8-bit weights beat
emulated-fp8 and bf16 on prefill, TTFT, *and* decode -- vision tower + MTP head retained, zero
accuracy loss. **The daily driver is now vLLM v0.24.0** (flipped 2026-07-03): the concurrent `!!!!` garbage is
fixed on v0.24.0, and captured W8A8 decode runs ~2x the old sglang driver (35-44 tok/s vs 18) with vision +
tool-calling + reasoning + 131K context all retained. sglang stays the maintained fallback (its edge: a working
prefix cache -- vLLM's is off pending a hybrid-GDN test). See the vLLM table + `JOURNAL.md` 2026-07-03.

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

## Serve shelf -- sglang (maintained fallback)

Was the production daily driver until 2026-07-03 (now vLLM v0.24.0 -- see below); kept as the maintained
fallback, and still the backend with a working prefix cache (RADIX) + the int8-MoE 35B entry.

Warm bench, IN=2048 / OUT=128. PP = prefill (prompt-processing) throughput = IN*1000/TTFT (tok/s);
TG = per-stream decode t/s; c1 = 1 stream, c4 = 4 concurrent. KV = engine-allocated KV cache. Each row
is `rdy_to_serve/sglang/<dir>/serve.sh` at *its own* best config.

| Model | Quant (kernel) | Wt GB | TP | PP tok/s | TTFT | TG c1 | TG c4 | KV avail |
|---|---|---|---|---|---|---|---|---|
| qwen3.6-27b | int4-AutoRound (W4A16) + NEXTN MTP | 19 | 1 | 2224 | 921 ms | 15.3 | 4.5 | 29.8k tok |
| qwen3.6-27b | W4A8 hybrid (int4-w / int8-a, XPUGraph) | 19 | 1 | 2062 | 993 ms | 27.3 | 27.7\* | 145k tok |
| qwen3.6-27b | **W8A8 int8 fused + NEXTN MTP** | 35 | 2 | **3786** | **541 ms** | **25.6** | 5.8 | 182k tok |
| qwen3.6-35b-a3b | **W8A8 int8 MoE** (Route A, eager) | 35 | 2 | **7529** | **272 ms** | 7.9 | 5.6\*\* | 1.04M tok |

\* W4A8 is the single-stream XPUGraph driver (`max-running-requests=1`): at c4 the 4 requests serialize,
so per-stream decode holds ~27.7 t/s but TTFT balloons to ~8.1 s. Best for single-stream throughput.

\*\* qwen3.6-35b-a3b W8A8 is the FIRST sglang serve of the int8 *MoE* (256 experts via the in-tree Triton
`use_int8_w8a8` -- no custom kernel; see `research/w8a8/SGLANG_MOE_PLAN.md`). Its **TTFT 272 ms is the
best of any 35B entry** (int8-XMX prefill). Decode is eager-slow (~8 t/s single / agg 23.9 t/s at c4) --
graph capture / NEXTN MTP / fused int8 dense are the open decode levers. Soak: clean + stable (no
degradation), and unlike vLLM it stays coherent under sustained concurrent load.

## Serve shelf -- vLLM (UN-PAUSED on v0.24.0)

> **vLLM is un-paused (2026-07-03).** Rebased to **v0.24.0 (torch 2.12)**, whose five hybrid
> mixed-prefill+decode PRs (#44700 split mixed -> recurrent GDN, #43990/#42430/#43961/#43556) FIX
> the concurrent `!!!!` GDN/Mamba SSM-state garbage that paused it. The W8A8 daily-driver candidate
> re-benched below is now **coherent** under staggered concurrent load (40/40 + 32/32 gate), PIECEWISE
> capture is **stable** under sustained MTP (restarts=0) + **deterministic** (3/3 greedy), and
> tool-calling (`qwen3_coder`) + reasoning (`qwen3`) + vision all work. Images: `vllm-xpu-env:{v0240,int8g-v0240}`.
>
> **Vision + capture FIXED (2026-07-03):** enabling vision under `torch.compile` crashed at init
> (`'NoneType'.size` in dynamo) -- root cause was the **standalone AOT-compile** serialize/reload mishandling
> the optional (None) multimodal inputs. Fix = **`VLLM_USE_AOT_COMPILE=0`** (env, no code patch, no runtime
> cost -- the captured graph is identical). With it, vision + PIECEWISE capture + MTP run together: **44 tok/s
> usage-based WITH vision** (2.4x the sglang daily driver's 18.0), gate 40/40 coherent, restarts=0, tool-calling
> + reasoning intact. The shelf auto-applies it (+ `--skip-mm-profiling`) when vision is on and `GRAPH=1`.
> (Remaining minor: prefix caching for the hybrid GDN model is untested here -- `PREFIXCACHE=0` for now.)

Numbers at each entry's own production config (`GRAPH=1` PIECEWISE capture -- the ~4x decode lever). Only the
**W8A8 row is re-benched on v0.24.0** (the daily-driver candidate); the other rows are the last vLLM baseline.

| Model | Quant | Wt GB | TP | PP tok/s | TTFT | TG c1 | TG c4 | KV avail |
|---|---|---|---|---|---|---|---|---|
| qwen3.6-27b | int4-AutoRound (W4A16) | 19 | 1 | 1589 | 1289 ms | 28.6 | 19.5 | 103k tok |
| qwen3.6-27b | W4A16 (compressed-tensors) + MTP | 26 | 2 | 651 | 3145 ms | 22.1 | 8.9 | 172k tok |
| qwen3.6-27b | W4A8-sqgptq (int8-act) | 26 | 1 | 1888 | 1085 ms | 6.3\* | 5.8\* | OOM @GRAPH=1 |
| qwen3.6-27b | **W8A8-sqgptq (int8) + MTP -- v0.24.0 (DAILY DRIVER)** | 35 | 2 | 2711 | 755 ms | **30.0**\*\*\* | 15.4 | 320k tok |
| qwen3.6-35b-a3b | int4-AutoRound (W4A16 MoE) | 21 | 1 | **4644** | **441 ms** | **67.7** | 43.8 | 270k tok |
| qwen3.6-35b-a3b | Quark W8A8-INT8 (MoE) | 35 | 2 | 1364 | 1502 ms | 43.1 | 22.2 | 684k tok |

\* W4A8: at `GRAPH=1` the capture buffers leave only 0.32 GiB for KV -> engine init OOMs (est. max len
2496); EAGER numbers shown. It is the one vLLM entry without a working captured config.

\*\*\* 2026-07-03 "faster-DD" session (JOURNAL, incl the correction entry): **MRV2**
(VLLM_USE_V2_MODEL_RUNNER=1) lifts cold-bench decode to **32.3-33.6 c1 (+7-11%, noise +-4%)**, gate 24/24
coherent -- but it is INCOMPATIBLE with the hybrid's prefix caching (mamba align-mode assert), so the
PREFIXCACHE=1 production daily driver keeps the prefix cache (6.97x warm TTFT) and stays at the row's 30.0;
serve.sh auto-gates MRV2 on PREFIXCACHE (PREFIXCACHE=0 configs get it by default). The initially-credited
SYCL_UR_USE_LEVEL_ZERO_V2 knob was never engaged (lib.sh pins V2=0 last-wins, vllm#41663) -- untested, no
claim. Also new: vLLM ships
DFlash in-tree and it WORKS on XPU (first serve, coherent, TP=2) -- slower than NEXTN MTP at spike settings,
see `vllm/DFLASH_XPU.md`. The v0.24.0 W8A8 row is **the live daily driver's actual config** (`35_sweep_bench` IN=2048/OUT=128 against
`b70_daily_0`: vision-on, 131K ctx, PIECEWISE capture, push-AR, MTP) -- the **best coherent W8A8 decode on the
box** (TG c1 30.0 > sglang 25.6 > old-vLLM 22.4). A chat-workload usage-based probe (short prompt, higher MTP
accept) reads even higher -- single-stream **44 tok/s vs the old sglang driver's 18** (the captured ~74ms
forward pass beats sglang's eager ~267ms). Build: `vllm/build_v0240_base.sh` + `build_v0240_int8gdn_so.sh` +
`images/int8g/bake_v0240.sh`; gate `vllm/gate_concurrent_coherence.py`; flip `DD_MODEL=vllm/qwen36-27b-w8a8`. JOURNAL 2026-07-03.
Context: the daily driver runs at **248K (253952)** -- Qwen3.6 is 262144-native (no rope scaling), so 248K is
in-window with ~66K KV headroom (KV 320k -> 1.26x concurrency at full length); full 262144 also serves (1.22x).
Decode/prefill (the table's IN=2048 numbers) are independent of max-model-len; only the KV column scales.

## Where to look

- `AGENTS.md` (= `CLAUDE.md`) -- standing rules, backend-split layout contract, shelf rules.
- `rdy_to_serve/<backend>/<model-quant>/serve.sh` -- the verified serve shelf (one best config each).
- `research/LESSONS.md` -- optimization ledger: what generalizes across quants + backends.
- `SHORTCOMINGS.md` -- open blockers + dead ends (the honest negatives).
- `models/` -- model registry (`manifest.yaml` + `fetch.sh`); weights in `models/files/` (git-ignored).
- `kernels/` -- shared custom gemm source; `research/w8a8/`, `research/w4a8/` -- the kernel campaigns.
- `FINDINGS.md` / `JOURNAL.md` -- raw measurements / chronological lab log.

## Environment

2x Intel Arc Pro B70 (Battlemage, 32 GB each), Ubuntu kernel 7.1 (since 2026-07-02; cured the TP=2 wedge),
run locally as `hotschmoe`. Images:
sglang `sglang-xpu:mtp`, vLLM `vllm-xpu-env:{v0240,int8g-v0240}` (v0.24.0/torch 2.12; older `{v0230,v0230moe,int8g}`
kept as rollback). Every GPU touch goes through the
shared lease (`bin/gpu-run`); TP=2 = both cards. The smoke/bench gate is `bin/serve-sweep`.
