# Qwen3.6 on dual Intel Arc B70 -- INT8-first serving on XPU

Serving **Qwen3.6-27B** (dense VLM) and **Qwen3.6-35B-A3B** (MoE VLM) on 2x Intel Arc Pro B70
(Battlemage, `xpu`). The headline: custom **fused INT8 W8A8** kernels make 8-bit weights beat
emulated-fp8 and bf16 on prefill, TTFT, *and* decode -- vision tower + MTP head retained, zero
accuracy loss.

> **[2026-07-08 CURRENT STATE -- supersedes the dated banners below.** The daily driver is **NVFP4 27B
> TP=2, bf16 KV, captured + MTP5 + XPUGraph reclaim** (`hotschmoe-dd`, port 18080). The reclaim fix
> (`B70_XPU_CG_RECLAIM`, re-instantiate captured graphs) makes captured+MTP crash-free at full speed, so
> NVFP4 DOES keep MTP+graph. Fresh IN=2048 bench: **code decode 53 t/s** (beats W8A8's 41), KV **385k @
> 128k** (more than W8A8 TP=2's 238k) at smaller weight (24 vs 35 GB) -> the DD choice is NVFP4, not W8A8.
> A second NVFP4 config, **TP=1 fp8 KV**, fits a full **128k context on ONE card** (frees the other card;
> fp8 = 1.98x bf16 context, proven with a 118,856-token needle). The 2026-07-06 "revert to W8A8" /
> "NVFP4 returns to research" banners below are SUPERSEDED. See the two NVFP4 rows + JOURNAL 2026-07-08 +
> docs/20260708_nvfp4_prefill_int8_and_fp8kv_investigation.md.]**

> **[2026-07-16 -- BACKENDS UPGRADED TO NEWEST + caching-on bench refresh.** vLLM **0.24.0 -> 0.25.1**
> (`vllm-xpu-env:{v0251,int8g-v0251}`) and sglang **0.5.6 -> 0.5.15** rebuilt on XPU with all custom kernels
> re-grafted. vLLM 0.25.1 keeps `torch==2.12.0` (no ABI bump, custom int8/nvfp4 .so load as-is) but shipped
> FOUR XPU regressions we had to fix (all baked into the recipes): a minimal-`LD_LIBRARY_PATH`/`CCL_ROOT`
> env that hid the GPUs ("Failed to infer device type"); an `int8g` bake-anchor rename
> (`XPUFP8ScaledMMLinearKernel` -> `XPUW8A8FP8LinearKernel`+`XPUW8A16FP8LinearKernel`); a bundled oneCCL
> **2021.15** that dies at TP=2 (`ze mem_to_ipc_handle: device_fd is invalid`) -> swapped back to v0.24.0's
> **2021.17**; and a new `_attention_ops` entry (`vllm::hpc_rope_norm_forward`) that `splitting_ops` must
> include or PIECEWISE capture asserts. All fixes: `vllm/build_v0251_base.sh` + `vllm/images/int8g/bake_v0251.sh`.
> **Fresh bench (IN=2048/OUT=128, prefix caching ON, TP=2 same as prior tables):**
>
> | backend | quant | Wt GB | TP | cold PP | TTFT (warm) | code TG c1 | generic TG c1 | code agg c4 | KV @131k | prefix cache |
> |---|---|---|---|---|---|---|---|---|---|---|
> | **vLLM 0.25.1** | NVFP4 | 24 | 2 | 2181 | 290 ms | **48.3** (best 50.1) | 16.7 | 105 | 342k tok | **57% hit (now works)** |
> | **vLLM 0.25.1** | W8A8 | 35 | 2 | **2598** | 514 ms | 38.5 (best 40.7) | 13.8 | 93 | 264k tok | works |
> | sglang 0.5.15 | W8A8 | 35 | 2 | _(in progress)_ | | | | | | |
> | sglang 0.5.15 | NVFP4 | 24 | 1 | _(new port -- yolo pending)_ | | | | | | |
> | zml (bf16 wildcard) | bf16 | 54 | 2 | n/a | n/a | 11.7 (decode) | 11.7 | n/a | n/a | n/a (CLI) |
>
> Both vLLM quants coherent on 0.25.1; NVFP4 leads decode (48.3 vs 38.5 code) + more KV at smaller weight,
> W8A8 leads cold prefill (2598 vs 2181, int8-XMX) -- same story as the prior tables, now re-confirmed on the
> newest backend. **New win: prefix caching WORKS on the NVFP4 path on 0.25.1** (57% hit; the running 0.24.0
> DD has it off). NVFP4-on-sglang is a first-ever port (sglang ships the ModelOpt loader natively + our XPU
> `nvfp4_gemm` kernel; single-card TP=1) -- infra committed (`sglang/NVFP4_PORT.md`), GPU yolo bench pending.
> zml's `llm` example now implements the Qwen3.6 GDN-hybrid arch -> **bf16 wildcard RAN: full unquantized 27B,
> TP=2 sharded across both cards, coherent (thinking mode), 11.7 tok/s decode** (CLI-only; zml has no OpenAI
> server; vs the proven zml W8A8 TP=2 13.7 t/s -- bf16 slightly slower). The 0.24.0 DD is unchanged; vLLM
> 0.25.1 is validated + available to promote.]**

> **[2026-07-07 CORRECTION -- the "NVFP4-only crash / W8A8 keeps MTP+graph" claim below is REFUTED.**
> The `linear_stream.h:84` NEO abort under MTP-verify + piecewise cudagraph is NOT NVFP4-specific: the
> W8A8 daily driver hit the IDENTICAL abort after ~3h of real use (and a 6-way concurrent soak reproduced
> it at ~96k decode tokens). Root cause (binary disasm): `at::xpu::XPUGraphImpl::replay` submits the
> captured graph via `submit_with_event` and never syncs, so per-replay NEO command-list entries accumulate
> (the MTP propose loop fires ~spec x pieces replays/step) until the command buffer overflows. It needs BOTH
> MTP and capture; dropping either avoids it (which is why sglang -- never captures MTP -- never saw it, and
> why enforce-eager+MTP is stable). W8A8 is NOT "bulletproof for days" with MTP+graph; it just crashes ~50-100x
> LATER than NVFP4 (NVFP4's fused nvfp4_gemm encodes far more commands/replay -> crash at ~1-2k tokens).
> A per-step `torch.xpu.synchronize` does NOT reclaim it; recapture-every-N is racy under load. The DD
> decision below rests on the false "W8A8 keeps MTP+graph" premise and is REOPENED. Real-fix effort is now
> focused on NVFP4 (fastest repro). See `docs/20260707_dd_mtp_piecewise_neo_abort.md` + RESEARCH_TODO 11h.]**
>
> **[2026-07-06: DAILY DRIVER REVERTED TO W8A8.** NVFP4-as-DD had two NVFP4-only faults (uncalibrated
> fp8 KV -> repetition; MTP-in-cudagraph -> NEO crash). Same-harness A/B on the STABLE configs picked
> W8A8: decode 43.0/38.2 t/s (generic/coding) vs stable-NVFP4's 31.7/25.5 -- W8A8 keeps MTP+graph, NVFP4
> can't (see the DD block below + JOURNAL 2026-07-06). [2026-07-07: "W8A8 keeps MTP+graph, NVFP4 can't" is
> WRONG -- W8A8 hits the same abort, just later; see the correction banner above.] The NVFP4 material below
> is now HISTORICAL.]**
>
> **Long-context daily driver moving to NVFP4 TP=2 (2026-07-04) [SUPERSEDED -- see 2026-07-06 above].** `nvidia/Qwen3.6-27B-NVFP4` (box
> quality #1) spanned across both cards serves the **full 256K model context** (262,144) with **764k KV
> tokens (2.92x @ 256K)** and **prefix caching** (3.98x warm-TTFT reuse), warm single-stream decode
> **48-50 t/s** (beats single-card 40-44), gate 18/18. The old cold-prefill tradeoff is **SOLVED (2026-07-05)**
> by the **push-AR prefill overlay** (a hand-rolled L0-IPC posted-write all-reduce, PUSH_AR=1): cold prefill
> **PP 666 -> 2174, TTFT 3076 -> 937 ms (3.3x), decode-neutral, gate 18/18** -- NVFP4 TP=2 prefill now
> MATCHES single-card 1702. Long prefills stack a further +11.5% via MAXBATCH=16384 + PUSH_AR_MAXB=256 MiB.
> See the NVFP4 TP=2 row + footnote ◆ below.

> **[SUPERSEDED 2026-07-06 -- numbers below are INVALID for a stable DD; DD decision REOPENED.]** The
> NVFP4 config benched below (fp8 KV + MTP5 + graph) has TWO NVFP4-only faults found 2026-07-06:
> (1) the checkpoint's UNCALIBRATED fp8 KV cache (scale 1.0) accumulates precision loss over generation
> length -> fluent REPETITION collapse late in long coding sessions; (2) MTP-verify inside the piecewise
> XPU cudagraph triggers a native NEO command-stream abort that crashes the engine. The stable NVFP4
> config is **B4 = bf16 KV (KV_FP8=0) + MTP-off + graph + PUSH_AR_GRAPH=1**: decode 25-31 t/s (drops MTP,
> so BELOW the 46-50 headline), TTFT 94ms, no repeat, no crash. **RESOLUTION (2026-07-06): same-harness
> A/B on the STABLE configs picked W8A8 -> DD REVERTED TO W8A8** (`vllm/qwen36-27b-w8a8`, PIECEWISE graph
> + MTP3 + PUSH_AR_GRAPH, live as b70_daily_0):
>
> | metric (same harness, TP=2) | stable NVFP4 B4 | **stable W8A8** |
> |---|---|---|
> | decode generic / coding | 31.7 / 25.5 t/s | **43.0 / 38.2 t/s** |
> | TTFT (short) | 94 ms | 195 ms |
> | prefill (cold ~15k) | 2195 | 2395 tok/s |
>
> W8A8 wins decode +36-50% because it keeps BOTH MTP and graph capture (NVFP4 can't: MTP+graph = crash);
> plus it was bulletproof for days, is 8-bit (quality preference), keeps vision+prefix-cache. NVFP4's
> [2026-07-07 CORRECTION: "NVFP4 can't / bulletproof for days" is FALSE -- W8A8 hits the same
> linear_stream.h:84 abort under MTP+graph, just ~50-100x later; it crashed the live DD 2026-07-07 after
> ~3h. The +36-50% decode edge was measured on W8A8's crashing config vs NVFP4's stable (MTP-off) config,
> so it is apples-to-oranges. See the correction banner at the top.]
> edges (TTFT, 2.4x KV @256K) don't outweigh decode for a coding DD. NVFP4 returns to research pending
> the two fixes in RESEARCH_TODO 11h (MTP-in-graph crash) + 11i (fp8-KV calibration). See JOURNAL
> 2026-07-06 + vllm/nvfp4/bisect_probe.py. Original (invalid, fp8-KV+MTP) A/B below:
>
> **DAILY DRIVER "DECIDED" (2026-07-05): NVFP4 27B TP=2.** Head-to-head A/B vs W8A8-int8 TP=2 on a real
> coding workload (same box + robust usage-based decode bench) picked NVFP4 for the coding daily driver:
>
> | axis (coding DD) | **NVFP4 TP=2** (+push-AR +MTP5) | W8A8-int8 TP=2 (+MTP +decode-push) |
> |---|---|---|
> | HumanEval+ (base/plus) | **0.988 / 0.945** | 0.970 / 0.933 |
> | code decode c1 (interactive) | **46.7-50.3 t/s** | 36.5-39.1 t/s |
> | code decode c2 (aggregate) | 62.3 | **71.2** |
> | cold prefill PP @2K | 2174 | **2711** |
> | KV @256K | **757k tok** | 320k tok |
> | weights on-card | **24 GB** | 35 GB |
>
> NVFP4 wins the axes that matter for a single-developer coding DD: **accuracy, single-stream interactive
> decode (+28%), and long-context KV (2.4x)** -- and push-AR made its prefill competitive (prefix caching
> covers the residual). W8A8's edges (cold prefill +25%, c2 aggregate) matter less for interactive coding.
> NVFP4 also has decode headroom left (`PUSH_AR_GRAPH=1` decode-push, W8A8's default, not yet ported).
> Served via `DD_MODEL=vllm/qwen36-27b-nvfp4 DD_ENV="TP=2"` at 256K ctx (push-AR + MTP5 + prefix cache +
> tool/reasoning parsers + API key auto-on).

**The prior W8A8 daily driver is vLLM v0.24.0 W8A8 + DFlash all-sliding speculative decoding** (`DFLASH=1
DFSWA=1`, spec=8, full 253952/248K context): +26% over the MTP config on coding/agentic work (35.2 t/s vs 27.8,
accept_len 3.24) and it holds accept at depth (3.6 at 100K, +20% decode vs MTP at 40K) with full context. The
TP-worker `shm_broadcast cancelled` crash that blocked DFlash was ROOT-CAUSED + FIXED 2026-07-03 (session 4):
DFlash is in vLLM's `EagleModelTypes` so async scheduling auto-enabled, but only DFlash's
`precompute_and_store_context_kv` writes context K/V by slot_mapping every draft pass -- under async scheduling a
cancelled request leaves a stale slot_mapping -> a freed-block write -> the rare hard fault (or the soft `!!!!`
poison). Forcing `--no-async-scheduling` on the DFlash path (now baked into the shelf) removes the race by
construction at ~zero perf cost. Validated: 7-min concurrent cancellation soak clean, `gate_concurrent_coherence`
18/18. **MTP is the robust one-flag fallback** (`DFLASH=0`): accept_len ~2.9 flat 4K->100K, 12h+ proven; prefer
it for general-chat-dominated workloads (DFlash accept ~0% on non-coding prose). sglang is the maintained
backend fallback. See the vLLM table + `vllm/DFLASH_XPU.md` + `JOURNAL.md` 2026-07-03 (session 4).

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

Numbers at each entry's own production config (`GRAPH=1` PIECEWISE capture -- the ~4x decode lever). The two
**NVFP4 rows are freshly benched 2026-07-08** (IN=2048/OUT=128 warm; TG c1 = generic-summarization decode with
the coding-workload decode in **bold** -- generic is low-MTP-accept, code is the ~99%-accept number that
matters for the coding DD; c4 shows per-stream with aggregate in parens); other rows are the last vLLM baseline.

| Model | Quant | Wt GB | TP | PP tok/s | TTFT | TG c1 | TG c4 | KV avail |
|---|---|---|---|---|---|---|---|---|
| qwen3.6-27b | **NVFP4 TP=2 bf16 KV + captured MTP5 + reclaim (DAILY DRIVER)**¶ | 24 | 2 | 2201 | 926 ms | 16.3 (**53** code) | 6.6 (26 agg) | **385k tok** |
| qwen3.6-27b | **NVFP4 TP=1 fp8 KV + captured (single-card 128k ctx)**◆ | 24 | 1 | 1909 | 1068 ms | **25.9** (MTP-off) | 17.4 (**70** agg) | 150k tok |
| qwen3.6-27b | int4-AutoRound (W4A16) | 19 | 1 | 1589 | 1289 ms | 28.6 | 19.5 | 103k tok |
| qwen3.6-27b | W4A16 (compressed-tensors) + MTP | 26 | 2 | 651 | 3145 ms | 22.1 | 8.9 | 172k tok |
| qwen3.6-27b | W4A8-sqgptq (int8-act) | 26 | 1 | 1888 | 1085 ms | 6.3\* | 5.8\* | OOM @GRAPH=1 |
| qwen3.6-27b | W8A8-sqgptq (int8) + MTP -- v0.24.0 (prior DD; see 2026-07-08 banner) | 35 | 2 | 2711 | 755 ms | **30.0**\*\*\* | 15.4 | 320k tok |
| qwen3.6-27b | W8A8 + DFlash all-sliding drafter (spec=8) -- v0.24.0 (research, not DD\*\*\*\*) | 35+2 | 2 | 2566 | 798 ms | 19.6 | 10.3 | 291k tok |
| qwen3.6-35b-a3b | int4-AutoRound (W4A16 MoE) -- v0.24.0† | 21 | 1 | 3670 | 558 ms | 46.5‡ | 44.0 | **485k tok** |
| qwen3.6-35b-a3b | Quark W8A8-INT8 (MoE) -- v0.24.0† | 35 | 2 | 1930 | 1061 ms | 37.0§ | 23.5 | **775k tok** |

\* W4A8: at `GRAPH=1` the capture buffers leave only 0.32 GiB for KV -> engine init OOMs (est. max len
2496); EAGER numbers shown. It is the one vLLM entry without a working captured config.

¶ **NVFP4 TP=2 = the daily driver (2026-07-08).** `nvidia/Qwen3.6-27B-NVFP4` ModelOpt checkpoint
(W4A16_NVFP4 MLP + FP8 attn + bf16 mtp/vision) via our custom `nvfp4_gemm_w4a16` oneDNN op (4-bit
f4_e2m1 resident) across BOTH cards + PIECEWISE capture + NEXTN MTP spec=5 + push-AR all-reduce. Two
fixes make this the DD: **bf16 KV** (`KV_FP8=0`) sidesteps the uncalibrated-fp8-KV repetition, and the
**XPUGraph re-instantiation reclaim** (`B70_XPU_CG_RECLAIM`, default-on for GRAPH=1) fixes the
MTP-verify x cudagraph `linear_stream.h:84` NEO abort -> captured+MTP now runs crash-free at full speed.
Bench (IN=2048, warm): code decode **53 t/s** (spec=5, ~99% accept on code; 16.3 on a generic
summarization = low MTP accept), PP 2201, TTFT 926 ms, KV **385k tok @ 128k (2.94x concurrency)** -- MORE
KV than W8A8 TP=2 (238k) at smaller weight (24 vs 35 GB). HumanEval+ **0.988/0.945** (box #1); gate 18/18;
vision + tool-call + reasoning retained. Serve = the systemd DD config (`SERVED_FORCE=hotschmoe-dd`, port
18080) over `vllm/nvfp4/serve_nvfp4_27b.sh`. JOURNAL 2026-07-08 + docs/20260707_dd_mtp_piecewise_neo_abort.md.

◆ **NVFP4 TP=1 fp8 KV = the single-card 128k-context config (2026-07-08).** Same NVFP4 checkpoint +
`nvfp4_gemm_w4a16` kernel + PIECEWISE capture, on ONE card. **fp8 KV cache** (calibrated `amax/448` per-layer
scales injected post-load by sitecustomize block 10 -- scale=1.0 is also coherent here, no clipping) gives
**1.98x the bf16-KV context**, so a full **128k-token prompt fits on a single B70** (150k-tok pool @ UTIL=0.92;
bf16 KV caps ~71.5k -> 128k is impossible without fp8). Proven with a real 118,856-token needle retrieval. This
frees the second card entirely. MTP is OFF (the drafter + capture buffers would not leave room for 128k KV);
captured decode is **25.9 t/s generic** (workload-independent -- no MTP) and it has the **best concurrency of
the 27B configs** (c4 agg 70 t/s -- single card, no per-token TP all-reduce, no MTP-verify). fp8 KV WORKS on
the vLLM XPU FlashAttention backend (the scales are applied -- verified); `--calculate-kv-scales` is a dead
end on this hybrid GDN model (vLLM #37554 + config-disabled), so calibrate offline. ~200k context is
weight-bound (24 GB resident) -> the llama.cpp GGUF route. Serve: `CARD=N TP=1 MODE=fused GRAPH=1
MAXLEN=131072 UTIL=0.92 KV_SCALES=vllm/nvfp4/kv_scales_nvfp4_27b.json bash vllm/nvfp4/serve_nvfp4_27b.sh`.
Details: docs/20260708_nvfp4_prefill_int8_and_fp8kv_investigation.md + JOURNAL 2026-07-08.

† **Both 35B-A3B MoE rows re-benched on vLLM v0.24.0 (torch 2.12), 2026-07-04** (JOURNAL). Ported off the old
v0.23 stack (int4 = `:v0230moe` baked image; w8a8 = `:v0230` + `patches/quark.py`) to `:v0240`. The XPU MoE
patches were re-grafted onto v0.24.0's rewritten INC package (`patches/inc_wna16_scheme.py`) and drifted
Quark loader (`patches/quark_v0240.py`), both mount-not-bake. **The port's wins are TTFT and capacity, not
single-stream decode:** int4 TTFT 441->558ms is *worse* but w8a8 TTFT 1502->1061ms (**-29%**) and BOTH KV
caches grew a lot (int4 270k->485k, w8a8 684k->775k) from v0.24.0's tighter KV accounting. **The 90-100 t/s
single-stream hope did NOT materialize** -- MoE decode is launch-bound at batch=1 (~3B active is compute-light,
little to amortize) and spec-decode is architecturally ineffective on A3B (the 2026-06-22 M5 finding: MTP =
+3% flat on this MoE vs +79% on the dense 27B), so there is no recent lever that lifts it. Aggregate scales
fine (int4 116 tok/s agg @ c4). Capturing on v0.24.0 needed a new IGC/ocloc workaround: inductor fuses the
RMSNorm reduction into the small-N MoE router matmul, and IGC cannot compile that kernel in ANY GRF mode
("Floating point exception", ocloc err 245). int4 needs `prologue_fusion=false`; the harder int8 MoE needs
BOTH the opaque-rms_norm op-priority (`--ir-op-priority` xpu_kernels) AND all inductor fusion off. See the
serve.sh headers.

‡ int4 **c1 46.5 is DOWN from the historical 67.7** (v0230moe). Likely cause: v0.24.0's in-tree INC XPU int4
path defaults to the `auto_round_kernel` (ARK) `woqgemm`, which is a ctypes call that graph-breaks under
`torch.compile` -> we gate it off (`B70_INC_ARK=1` for eager-only) and use the capturable in-tree
`int4_gemm_w4a16`, which is slower at M=1 decode. c4 (44.0) matches the old 43.8, and c1<c2 (46.5<57.6) is the
launch-bound MoE signature. Aggregate out: 39/89/116 tok/s @ c1/2/4.

§ w8a8 **c1 37.0 is the first warm run and DEGRADES run-over-run under sustained back-to-back load** (37.0 ->
25.5 -> 19.5 across three c1 benches; TTFT stays ~1050ms, so it is clock/thermal -- the display-attached card1
downclocks under TP=2 -- not a leak). c1 is also below the old 43.1 partly because dodging the IGC crash
requires disabling inductor fusion (prologue+epilogue+combo), which costs some decode. Aggregate out:
28/47/65 tok/s @ c1/2/4 (first warm run).

\*\*\* 2026-07-03 "faster-DD" session (JOURNAL, incl the correction entry): **MRV2**
(VLLM_USE_V2_MODEL_RUNNER=1) lifts cold-bench decode to **32.3-33.6 c1 (+7-11%, noise +-4%)**, gate 24/24
coherent -- but it is INCOMPATIBLE with the hybrid's prefix caching (mamba align-mode assert), so the
PREFIXCACHE=1 production daily driver keeps the prefix cache (6.97x warm TTFT) and stays at the row's 30.0;
serve.sh auto-gates MRV2 on PREFIXCACHE (PREFIXCACHE=0 configs get it by default). The initially-credited
SYCL_UR_USE_LEVEL_ZERO_V2 knob was never engaged (lib.sh pins V2=0 last-wins, vllm#41663) -- untested, no
claim. The W8A8+MTP row is now the W8A8 KERNEL baseline (`35_sweep_bench` IN=2048/OUT=128 random text) -- the
best coherent W8A8 decode on that workload (TG c1 30.0 > sglang 25.6 > old-vLLM 22.4). Build:
`vllm/build_v0240_base.sh` + `build_v0240_int8gdn_so.sh` + `images/int8g/bake_v0240.sh`;
gate `vllm/gate_concurrent_coherence.py`. JOURNAL 2026-07-03.

\*\*\*\* **DFlash evaluation -- researched, NOT the daily driver (2026-07-03 session 3).** vLLM's in-tree
**DFlash block-diffusion drafter** (`z-lab/Qwen3.6-27B-DFlash`, W8A8-RTN, `spec=8`) was evaluated as the DD.
Best variant = **all-sliding mode** (`DFSWA=1`: all 5 drafter layers windowed to 2048 -> one KV group). The
table's IN=2048/OUT=128 numbers are
RANDOM text = DFlash's worst case (the drafter can't predict random tokens); real coding is much faster. The
metric that decides a spec config is **accept_len**, and the key finding is how it behaves with context DEPTH
(from `vllm/dflash_deep_accept_probe.py`, /metrics counters):

| context | MTP spec=3 (DD) | DFlash stock full drafter | DFlash all-sliding |
|---|---|---|---|
| 4K   | 2.90 | 4.46 | 3.91 |
| 16K  | 2.89 | 2.54 | 3.46 |
| 40K  | 2.72 | 2.17 | 3.14 |
| 100K | 3.03 | **1.63** (collapses) | **3.60** (holds) |

Findings: (1) the stock full-attention drafter is fastest at SHALLOW context (+75% on short coding bursts) but
its sliding layers run out-of-distribution at depth and **collapse** (accept 4.46 -> 1.63), capping ctx at
~186K. (2) Windowing every drafter layer (all-sliding) keeps them in-distribution: accept **holds ~3.6 to 100K**,
it fits the **full 253952** context (291k KV tokens, 1.15x conc), and reliable two-point decode t/s **beats MTP
at depth** (40K: **21.3 vs 17.7 tok/s**). (3) BUT all-sliding is **not production-stable**: under concurrent
bench load it crashed with a TP-worker `shm_broadcast: cancelled` **EngineDeadError** (same mode as the earlier
DFlash spec=7 spike), and a request cancelled mid deep-prefill soft-poisons the GDN state to `!!!!`
(`docker restart` / `bin/dd-watchdog` heals it). => **the DD stays MTP** until DFlash's TP-worker crash is
root-caused. DFlash remains a one-flag research option: `DFLASH=1` (serve.sh toggle; `DFSWA=1` all-sliding,
`DFSWA=0` stock full drafter). Artifacts: `vllm/patches/kv_cache_utils_gcd.py` (KV-padding patch),
`vllm/patches/qwen3_dflash_swa.py` (SWA overlay), probes `vllm/dflash_deep_accept_probe.py` +
`vllm/twopoint_decode_tps.py`; full analysis in `vllm/DFLASH_XPU.md`. Decode/prefill (the IN=2048 numbers) are
independent of max-model-len; only KV scales.

## Incoming (downloaded, NOT yet served/benched)

- **`nvidia/NVIDIA-Nemotron-Labs-3-Puzzle-75B-A9B-NVFP4`** -- a Nemotron-H hybrid (mamba2 + attention)
  MoE, 75B total / 9B active, text-only, ModelOpt NVFP4 (NVFP4 experts group_size 16 + FP8 attention,
  bf16 MTP head). On disk at `models/files/nemotron-3-puzzle-75b-a9b/nvfp4-modelopt/` (~54 GB, curl-fetched
  2026-07-08; in `models/manifest.yaml` as source:hf). New `nemotron_h_puzzle` arch -- **serve + bench is an
  open TODO** (RESEARCH_TODO Track 11j): needs the arch recognized on a backend (vLLM/sglang, trust-remote-code
  modeling + XPU mamba2/MoE port) and a TP=2 config (won't fit one 32 GB card). No numbers yet.

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
