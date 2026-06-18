# HANDOFF — for the next dev agent (Linux dev machine -> SSH to Threadripper/B70)

**Date:** 2026-06-19 (updated; was 2026-06-18). **Context:** dev is on a Linux machine SSHing into the Unraid
host with the Arc Pro B70. This is the pickup note. Read [README.md](README.md) -> [FINDINGS.md](FINDINGS.md)
-> [docs/kernel/02_int8_w8a8_status.md](docs/kernel/02_int8_w8a8_status.md) -> [JOURNAL.md](JOURNAL.md)
(newest entries at the bottom). A 2nd B70 is being added (week of 2026-06-22) -> dual-card work unblocks.

## TL;DR of where we are
We wrote the **first working INT8 W8A8 inference kernel for Intel Battlemage (B70) in vLLM** (oneDNN
`s8s8s32` + a fused per-token int8 quant). It serves a real model end-to-end and **beats FP8 ~1.6x in
prefill**, nearly matches decode (22.6 vs 29 t/s), and composes with **FP8 KV cache (2x context budget)**.
The engine config for a long-context coding server is settled: **INT8 W8A8 linear + FP8 KV cache** (+
optional W8A16 ignore-list). All committed/pushed to `github.com/hotschmoe/b70_ai_things` (main).

**Update 2026-06-19:** add **PIECEWISE XPU graph capture** to that config -> **+16.7% decode** (27.23 vs 23.33
t/s) for free. Unblocked by adding `register_fake` meta kernels for our 2 custom int8 ops (image
`vllm-xpu-env:int8g`). See "NEW serving config" below. ngram spec-decode is net-NEGATIVE on XPU (parked).

## Box access (Linux dev machine -- SET UP 2026-06-19)
- Host: `ssh b70` = `root@192.168.10.5`. **On THIS Linux machine the key is the default `~/.ssh/id_ed25519`**
  (NOT the old `b70_unraid_ed25519`). The `b70` alias is in `~/.ssh/config` (added 2026-06-19). Unraid 7.3.1,
  TR 1950X (32T), 125 GiB RAM, Docker. GPU: 1x Arc Pro B70 (Battlemage, 32 GB), `--device /dev/dri`, `ZE_AFFINITY_MASK=0`.
- ALL heavy data on the 8TB SSD: `/mnt/vm_8tb/b70/` (models, hf_cache, results, the kernel repo, caches).
- **Running scripts: use `scripts/runremote.sh`** (bash port of `runremote.ps1`, written 2026-06-19). It
  base64-transports a local `.sh` to the box and runs it under `bash -s`, with optional env vars:
  `scripts/runremote.sh scripts/NN_foo.sh KEY=VALUE [host=b70]`. All `scripts/*.sh` are plain bash that
  `docker run` the right images with `/mnt/vm_8tb/b70` mounted. (`runremote.ps1` is the old Windows version.)
- **No python on the Unraid host** -- for quick python (e.g. parsing a safetensors index), use the
  `python:3.11` image: `docker run --rm -v /mnt/vm_8tb/b70:/mnt/vm_8tb/b70 python:3.11 python -c '...'`.

## Key artifacts (on the box unless noted)
- **`vllm-xpu-env:int8`** docker image (committed) = base v0230 + our int8 kernel baked in. Serve ANY
  compressed-tensors W8A8-INT8 checkpoint with a plain `vllm serve` (no graft/patch). See
  docs/kernel/02 for the exact `docker run`. Verified: `Selected XPUInt8ScaledMMLinearKernel`.
- **Forked kernels repo:** `/mnt/vm_8tb/b70/vllm-xpu-kernels` (upstream head 11f42aa + our edits:
  `csrc/xpu/onednn/int8_gemm_w8a8.h`, `onednn_ext.h` s8_s8 dtype, `csrc/xpu/sycl/dynamic_per_token_int8_quant.cpp`,
  `torch_bindings.cpp`, `ops.h`, `CMakeLists.txt`). Built `.so` at `vllm_xpu_kernels/_xpu_C.abi3.so`.
- **vLLM Python patch:** `contrib/vllm_int8_xpu/` (in THIS repo) = `xpu_int8_kernel.py` (the
  `XPUInt8ScaledMMLinearKernel`) + `registry_patch.md` + native-source copies. Applied at image-bake by
  `scripts/45`/`47` via `apply_patches.py` (resolves the real vLLM dir via `import vllm`).
- Other images: `vllm-xpu-env:v0230` (vLLM 0.23.0, the build/serve base), `:tf` (0.20.2), `intel/llm-scaler-vllm:0.14.0-b8.3.1`.
- Checkpoints (`/mnt/vm_8tb/b70/models/`): `Qwen3-14B-W8A8-INT8` (ours, works), `Qwen3-14B-W4A8-INT`,
  `Lorbus_Qwen3.6-27B-int4-AutoRound`, `Qwen_Qwen3.6-27B-FP8`, `Qwen_Qwen3.6-27B` (BF16, 54 GB),
  **`Qwen3.6-27B-W8A8-INT8-RTNtest`** (33 GB, data-free RTN validation checkpoint -- text-only decoder, our
  W8A8 dynamic scheme; for card-#2 serveability testing). 14B BF16 at `/mnt/vm_8tb/specula-build/models/Qwen3-14B`.
- **Qwen3.6-27B is a `Qwen3_5ForConditionalGeneration` VLM** (vision tower + hybrid DeltaNet/full-attn text +
  MTP), NOT a dense text model. `AutoModelForCausalLM` loads only the **text decoder** (no vision/MTP) -- good
  for a coding server. 64 layers (48 `linear_attention` DeltaNet + 16 `full_attention`), hidden 5120, vocab 248320.

## RIGHT NOW (state as of 2026-06-19 EOD)
- **27B BF16 download COMPLETE** (54 GB, `models/Qwen_Qwen3.6-27B`). GPU FREE (no server up).
- **NEW recommended serve config = PIECEWISE XPU graph capture** (image `vllm-xpu-env:int8g`, see below) ->
  **+16.7% decode** (27.23 vs 23.33 t/s) over the old eager config, no accuracy change.
- **27B W8A8 quant pipeline VALIDATED** (data-free RTN test checkpoint `models/Qwen3.6-27B-W8A8-INT8-RTNtest`,
  33 GB). The full GPTQ+SmoothQuant quality pass is **DEFERRED until card #2 confirms the model serves on XPU**
  (33 GB > one card; no point optimizing accuracy of an unconfirmed-servable model). See JOURNAL 2026-06-19.

## NEW serving config (use this) -- PIECEWISE XPU graph capture, +16.7% decode
- Image `vllm-xpu-env:int8g` = `:int8` + `register_fake` meta kernels for our 2 custom int8 ops (built by
  `scripts/52_bake_int8_graph.sh`; source `contrib/vllm_int8_xpu/xpu_int8.py`). The fakes let torch.compile/
  dynamo trace through our ops so vLLM's XPU graph capture works.
- Serve: `scripts/runremote.sh scripts/51_serve_int8_specdecode.sh IMG=vllm-xpu-env:int8g GRAPH=1 CGMODE=PIECEWISE SPEC=0`
  (= `VLLM_XPU_ENABLE_XPU_GRAPH=1`, no `--enforce-eager`, `cudagraph_mode=PIECEWISE`, raised pid/thread ulimits).
- **FULL graph capture is BLOCKED** by Intel's SYCL Graph extension: `sycl_ext_oneapi_work_group_scratch_memory
  is not yet available for use with the SYCL Graph extension` -- hit by the **FlashAttention-v2 XPU** kernel.
  PIECEWISE splits attention OUT (eager) and captures only linear/MLP -> our int8 oneDNN GEMM IS capture-safe.
  FULL stays blocked until a newer oneAPI supports work_group_scratch under SYCL Graph (or a no-scratch XPU
  attention kernel exists).

## IMMEDIATE PICKUP STEPS (prioritized)
1. **CARD #2 is the gate for the 27B (the user's actual goal).** When it arrives (~week of 2026-06-22):
   - First de-risk serveability: try to serve `models/Qwen3.6-27B-W8A8-INT8-RTNtest` across 2 cards (`-tp 2`)
     via `:int8g`. The arch (`Qwen3_5ForConditionalGeneration`, loads text-only via AutoModelForCausalLM) IS
     registered in vLLM 0.23.0 and there's a `gdn_attention_core_xpu` op -- but DeltaNet-on-XPU is UNPROVEN.
   - If it serves: run the **GPTQ+SmoothQuant quality pass** (`scripts/49_quantize_27b_w8a8.sh`, METHOD=gptq,
     DEVICE=xpu; `actorder=None` now, NOT False) for near-BF16 accuracy, then serve that across 2 cards.
   - Also validate the int8 kernel under TP2 (per-shard matmul unchanged; all-reduce is host-staged, no P2P).
2. **DONE this session (see JOURNAL 2026-06-19):** Linux SSH workflow + `runremote.sh`; ngram spec-decode PoC
   (**net-NEGATIVE**, 23.33->21.51 t/s); **XPU graph capture unblocked via register_fake -> PIECEWISE +16.7%**;
   27B quant pipeline validated. Spec-decode is parked: it stays net-negative even WITH PIECEWISE graph
   (27.23->25.28) because attention runs eager; it needs FULL capture (blocked) to win.

## OPEN THREADS / ROADMAP (see tasks + JOURNAL for detail)
- **MoE int8 W8A8 kernel** -> unlocks Qwen3.6-35B-A3B W8A8 AND the "Quark 99 t/s" parity chase (our dense
  int8_gemm does NOT cover the fused-MoE expert path; `compressed_tensors_moe_w8a8_int8`). Needs card #2.
- **"Quark 99 t/s" (Qwen3.6-35B-A3B, 4xB70): OPEN chase, NOT debunked** (docs/COMMUNITY_CONFIGS.md). Credible
  IF custom kernel (we proved possible); different regime (MoE/4-card). Parity = MoE int8 kernel + multi-card.
- **XPU graph capture -- LARGELY RESOLVED (2026-06-19).** It's ALREADY wired in vLLM 0.23.0
  (`VLLM_XPU_ENABLE_XPU_GRAPH=1`), not unwired. We unblocked it for our int8 path with `register_fake` meta
  kernels (contrib/vllm_int8_xpu/xpu_int8.py) -> **PIECEWISE capture works, +16.7% decode**. FULL capture is
  blocked by Intel's SYCL Graph ext (work_group_scratch_memory, via flash-attn) -- a vendor-maturity wall,
  retest on newer oneAPI. Contribution: the 2 register_fake impls (upstream-worthy, small).
- **Spec-decode on XPU -- PARKED (2026-06-19).** ngram is net-negative in eager (-7.8%) AND with PIECEWISE
  graph (-7% vs graph-no-spec) because attention runs eager in PIECEWISE -> verify pays eager attn overhead
  x(N+1) at ~16% accept. Spec-decode (ngram/DFlash/MTP) needs FULL graph capture (attention included) to win.
- **DFlash spec-decode on XPU** (single-pass drafter, vLLM PR #38300): only worth revisiting once FULL XPU
  graph capture is possible (see above). Until then it'll lose for the same reason ngram does.
- **MTP on Qwen3.6-27B:** NOTE the W8A8 checkpoint loads **text-only via AutoModelForCausalLM (no MTP head)**,
  so `method:"mtp"` is NOT available on that checkpoint. Would need a VLM/MTP-aware load path to keep the head.
- **Decode gap (22.6 vs 29):** an M=1 int8 GEMV fast path / vectorized quant K-loop (diminishing returns).
- **make-it-right (#12):** asym/AZP + static schemes in int8_gemm_w8a8 (needs an asym checkpoint; lower value).
- **Upstream (#13):** PR vllm-xpu-kernels (int8_gemm + fused quant) + PR vLLM (kernel class + registry + the
  `.get()` chooser hardening, which also fixes the GDN-FP8 KeyError -- standalone-worthy).
- **TurboQuant KV: DEFER** (KV-only, L-effort SYCL, vLLM's own bench shows throughput regression). Use FP8 KV.
- **llama.cpp DeltaNet (#9):** llama.cpp now has Gated-DeltaNet (PR #16095, SYCL) -> retry Qwen3.6-27B on a
  current llama.cpp build (our "build 9680 segfaults" blocker is stale). Could unblock dense 27B on GPU simply.

## GOTCHAS / LESSONS (save yourself the pain)
- **Build only `_xpu_C`:** `pip install -e .` rebuilds flash-attn/MoE/cutlass = 1-2h. Use the minimal-target
  env (`FA2/MOE/GDN/MQA/BASIC_KERNELS_ENABLED=OFF`, `XPU_SPECIFIC_KERNELS_ENABLED=ON`) -> minutes. scripts/44 does this.
- **CMakeCache is path-pinned:** mount the repo at `/src` to match the cached `.deps` paths, or `rm -rf build/`.
  ccache makes rebuilds fast; keep CCACHE_DIR on /mnt/vm_8tb.
- **Always remove ALL vllm serving containers before serving** (vllm_int8/qwen3/w4a8/w8a8) -- a survivor holds
  port 18080 AND the GPU -> false-HEALTHY + silent VRAM contention (this bit us; scripts 36/41/42/45 now do it).
- **`import vllm` resolves the editable path** (sometimes `/workspace/vllm/vllm`, sometimes site-packages) --
  patch via `import vllm` dirname, never hardcode (apply_patches.py does this).
- **Verify ops on the GPU, not CPU:** loading `_xpu_C.so` in a no-GPU container shows 0 gemm ops even for the
  stock build -- not a real test. Load WITH `--device /dev/dri` + `source setvars.sh`.
- **GPU device-lost during XPU quant** was VRAM contention (a server was up), not (confirmed) a torch-xpu bug
  -- run quant with the GPU exclusive; use GPTQ `actorder=False` to skip the H[perm] gather just in case.
- W8A8 stock vLLM KeyError-crashes on XPU -- that's the bug our kernel fixes; don't "rediscover" it.
