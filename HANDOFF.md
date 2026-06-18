# HANDOFF — for the next dev agent (Linux dev machine -> SSH to Threadripper/B70)

**Date:** 2026-06-18. **Context:** main dev is moving to a Linux machine that SSHes into the Unraid host
with the Arc Pro B70. This is the pickup note. Read [README.md](README.md) -> [FINDINGS.md](FINDINGS.md)
-> [docs/kernel/02_int8_w8a8_status.md](docs/kernel/02_int8_w8a8_status.md) -> [JOURNAL.md](JOURNAL.md)
(newest entries at the bottom). A 2nd B70 is being added (week of 2026-06-22) -> dual-card work unblocks.

## TL;DR of where we are
We wrote the **first working INT8 W8A8 inference kernel for Intel Battlemage (B70) in vLLM** (oneDNN
`s8s8s32` + a fused per-token int8 quant). It serves a real model end-to-end and **beats FP8 ~1.6x in
prefill**, nearly matches decode (22.6 vs 29 t/s), and composes with **FP8 KV cache (2x context budget)**.
The engine config for a long-context coding server is settled: **INT8 W8A8 linear + FP8 KV cache** (+
optional W8A16 ignore-list). All committed/pushed to `github.com/hotschmoe/b70_ai_things` (main).

## Box access (NOTE: you are on Linux now, not Windows)
- Host: `ssh b70` = `root@192.168.10.5`, key `~/.ssh/b70_unraid_ed25519`. Unraid 7.3.1, TR 1950X (32T),
  125 GiB RAM, Docker. GPU: 1x Arc Pro B70 (Battlemage, 32 GB), `--device /dev/dri`, `ZE_AFFINITY_MASK=0`.
- ALL heavy data on the 8TB SSD: `/mnt/vm_8tb/b70/` (models, hf_cache, results, the kernel repo, caches).
- **Running scripts:** the repo's `scripts/runremote.ps1` is **Windows PowerShell** -- it base64-transports a
  local `.sh` to the box and runs it. On Linux, replace it with the equivalent:
  `ssh b70 'bash -s' < scripts/NN_foo.sh` (for env vars: `ssh b70 "FOO=bar bash -s" < scripts/NN_foo.sh`,
  or prepend `export`s). **First task: write a `runremote.sh` bash equivalent.** All `scripts/*.sh` are plain
  bash, already written for the box; they `docker run` the right images with `/mnt/vm_8tb/b70` mounted.

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
  `Lorbus_Qwen3.6-27B-int4-AutoRound`, `Qwen_Qwen3.6-27B-FP8`. 14B BF16 at `/mnt/vm_8tb/specula-build/models/Qwen3-14B`.

## RIGHT NOW (in flight as of handoff)
- **`qwen27b_dl` container is downloading Qwen/Qwen3.6-27B BF16** (~54 GB; was ~21 GB when I left). Detached,
  resumable. Check: `ssh b70 'docker logs --tail 5 qwen27b_dl; du -sh /mnt/vm_8tb/b70/models/Qwen_Qwen3.6-27B'`.
  If it died, resume with `scripts/50_download_27b_detached.sh` (snapshot_download resumes from cache).
- GPU is FREE (no server up).

## IMMEDIATE PICKUP STEPS (prioritized)
1. **Finish the 27B W8A8 quant** (Tier-1b). When the download completes:
   - Inspect the model's module names to FINALIZE the DeltaNet/MTP ignore-list:
     `ssh b70 'docker run --rm -v /mnt/vm_8tb/b70:/mnt/vm_8tb/b70 --entrypoint python vllm-xpu-env:v0230 -c "from transformers import AutoConfig; import json; print(json.dumps(AutoConfig.from_pretrained(\"/mnt/vm_8tb/b70/models/Qwen_Qwen3.6-27B\", trust_remote_code=True).to_dict(), indent=1)[:2000])"'`
     then grep a loaded model's `named_modules()` for `linear_attn` / `mtp` to set `IGNORE` exactly.
   - Run `scripts/49_quantize_27b_w8a8.sh` (SmoothQuant+GPTQ, **GPU-accel**, `actorder=False`, ignore-list).
     GOAL: stay near W8A16/BF16 accuracy (the user's priority). The XPU-accel calibration is a research path
     -- if GPTQ device-losts again (isolated GPU now, so contention should be gone), fall back METHOD=rtn or
     DEVICE=cpu. 27B W8A8 ~27 GB -> needs **card #2** to serve (or W8A16 fallbacks to shrink). Validate it
     serves via our kernel on 2 cards.
2. **ngram spec-decode PoC on 14B W8A8** (cheap, diagnostic): serve with
   `--speculative-config '{"method":"ngram","num_speculative_tokens":4,"prompt_lookup_max":3}'` and bench
   vs no-spec. If net-POSITIVE -> single-pass drafters (DFlash/MTP) will likely win on XPU; if NEGATIVE ->
   **wiring `torch.xpu.XPUGraph` into the vLLM XPU runner is the true prerequisite** for all spec-decode
   (see JOURNAL spec-decode entry + literature/06).
3. **Card #2 arrives:** validate our int8 kernel under TP2 (`-tp 2`; per-shard matmul unchanged, all-reduce
   separate); serve 27B W8A8 across 2 cards (the user's actual goal); then the MoE/parity work below.

## OPEN THREADS / ROADMAP (see tasks + JOURNAL for detail)
- **MoE int8 W8A8 kernel** -> unlocks Qwen3.6-35B-A3B W8A8 AND the "Quark 99 t/s" parity chase (our dense
  int8_gemm does NOT cover the fused-MoE expert path; `compressed_tensors_moe_w8a8_int8`). Needs card #2.
- **"Quark 99 t/s" (Qwen3.6-35B-A3B, 4xB70): OPEN chase, NOT debunked** (docs/COMMUNITY_CONFIGS.md). Credible
  IF custom kernel (we proved possible); different regime (MoE/4-card). Parity = MoE int8 kernel + multi-card.
- **XPU graph capture** (`torch.xpu.XPUGraph`, PyTorch 2.11) into vLLM XPU runner -> fixes decode launch
  overhead AND unlocks spec-decode/MTP. Highest strategic leverage. literature/06 contribution #3.
- **DFlash spec-decode on XPU** (single-pass drafter, already in vLLM PR #38300) -> the spec-decode method
  whose advantage survives no-graph-capture. M-L effort (non-causal attn + Triton-XPU kernels are the risk).
- **MTP on Qwen3.6-27B** (it ships MTP heads): `method:"mtp"` -- zero-effort single-pass spec-decode once 27B serves.
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
