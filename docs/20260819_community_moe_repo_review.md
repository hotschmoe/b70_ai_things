# 2026-08-19 community MoE repo review

Audience: Ornith-1.0-35B NVFP4 fused MoE apply on 2x Arc Pro B70
(vLLM-XPU int8g-v0260). Graph-capture fused apply is already capture-safe:
gather via `index_select`, Python-int M=1 gemms (T<=16), `unique()` only for
eager prefill. GRAPH TP=1 ~34.9 t/s (was 3.8). TP=2 GRAPH ~22.6/24.2.
MTP parked: NVFP4 experts require `moe_backend=emulation`; unquantized MTP
MoE rejects emulation and wants triton; triton is rejected for NVFP4.
XPU `tensor[scalar_tensor]` host-syncs inside command graphs. P2P stays off.

This is a read-only review. No GPU work. No commits. Community trees were
not modified. This agent has no git/shell, so it could not `git clone` into
`/mnt/vm_8tb/b70/community_repos/`. Review used:

- on-disk `/mnt/vm_8tb/b70/b70-optimization-lab` (git, `origin` fetch
  historically restricted to `main`)
- on-disk `/mnt/vm_8tb/b70/b70-optimization-lab-main` (full tree snapshot,
  richer than the git clone; matches the 2026-08-19 S2 digest era)
- on-disk `/mnt/vm_8tb/b70/qwen38-b70`
- GitHub API: every public branch on the listed remotes, plus 1-hop
  citations from those READMEs

intel/llm-scaler is the real repo (image `intel/llm-scaler-vllm:*`). There
is no separate public `intel/llm-scaler-vllm` source repo.

================================================================================
0. Ranked steal-list for OUR Ornith NVFP4 fused apply + MTP + TP=2 GRAPH
================================================================================

HIGH

1. Grow-only sticky MoE scratch + 4-deep output ring (llm-scaler #505/#507).
   Graph capture freezes MoE output addresses. Realloc on n_tokens change
   frees the captured storage; later replay reads garbage -> "!!!!".
   Grow-only buffers + `.narrow(0,0,n_tokens)` + a 4-slot output ring
   because the caller holds the tensor across TP all-reduce / residual /
   next layer. Direct map onto our fused apply workspaces.

2. Dual-path M==1 GEMV vs M>1 grouped expert GEMM (llm-scaler #491/#561/#584).
   Decode M=1: 1D `block_load` along K (~528 GB/s) not 16-wide 2D DPAS
   (~315 GB/s, 1/4 of a 64B cacheline). Prefill: expert-grouped TILE_M
   GEMM with a GPU tile-map (no d2h). We already split unique()/eager
   prefill vs M=1 graph decode; steal the K-major 1D load and the
   per-shape persistent buffer table (M=1..12).

3. Independent drafter vs target MoE backends (cookbook draft-INT4 +
   llm-scaler gemma MTP grouped forward). Do not share one `moe_backend`
   across NVFP4 target experts and unquantized MTP drafter MoE. Quantize
   or route the draft at `load_weights`, never lazily in `@support_torch_compile`
   forward.

MED

4. Caller-provided fused-MoE workspaces (vllm-xpu-kernels PR #392, still
   OPEN/dirty). Same shape as our persistent apply buffers: remap / GEMM1 /
   act / GEMM2 reused across decode steps.

5. TopK V2 template dispatch for E=256/top_k=8 with heap fallback
   (llm-scaler `moe-topk-dispatch-fix`, `fix-moe-int4-topk-e128`). Keep
   routing on-device; do not `unique().tolist()`.

LOW

6. Steve W8A8 full-layerlet / oneDNN resident island as an oracle, not a
   production NVFP4 path. Gate traces proved flags can lie.

7. Cookbook mixed-split v5 + GDN prefix-parity / persistent scratch zero-init
   for when we unpark GDN-MTP on Ornith. Dense 3.8, not fused NVFP4 MoE.

================================================================================
1. steveseguin/b70-optimization-lab
================================================================================

Repo: https://github.com/steveseguin/b70-optimization-lab
Branch: `main` only (GitHub API; the old `codex/*` branches from
docs/literature/10 are gone from the remote).
HEAD: `99ca073` 2026-08-19T15:03:19Z
  "bench(qwen38): zero-init fix passes established parity gates at MTP4/MTP5"

On-disk:
- git clone `/mnt/vm_8tb/b70/b70-optimization-lab` -- older, origin fetch
  was `+refs/heads/main:refs/remotes/origin/main` only
- snapshot `/mnt/vm_8tb/b70/b70-optimization-lab-main` -- full lab tree
  used for this review (includes experiments/patches after 2026-06-22)

What we already knew (literature/10, COMMUNITY_CONFIGS, kernel/20,
20260819_steve_qwen38_int4ar.md):
- Capture-safe custom all-reduce / XCCL; SYCLKERNELS is our equivalent
- n-gram spec, async+compile_sizes=[1], gpu_memory_utilization 0.90-0.97
- MiniMax QK-RMS / VLLM_MINIMAX_* fusions measured NEGATIVE
- Quark W8A8 INT8 35B-A3B serving story (later corrected / reopened)
- MTP on B70 historically catastrophic until GDN verifier/KV-position work
- llama.cpp SYCL Q4 fusion -- skip for vLLM NVFP4
- 2026-08-19: Qwen3.8 INT4-AR MTP5 ~101.9 tok/s on 2x B70 (dense, not MoE)

NEW since those reviews:
- Persistent GDN scratch zero-init as a GRAPH correctness contract
  (`experiments/qwen38-27b-b70/notes/2026-08-19-autoround-int4-gdn-scratch-zero-init-built-ab.md`
  on GitHub HEAD 99ca073). Op-level gates can pass on a fresh process even
  with uninitialized scratch because the caching allocator returns clean
  pages; bug-removal proof needs the serving-host strict-25 rerun.
- W8A8 MoE "full layerlet" is a real C++ op name
  `qwen36_moe_w8a8_full_layerlet`, gated by Python that often never
  entered the C++ path. See suggestions/findings/qwen35-b70-options.md
  ~358-397. First flag A/B was not a real layerlet test.
- oneDNN resident multi-window GEMM1/SILU/quant/GEMM2/gather sidecar
  (`tools/onednn_moe_island_resident_runner.cpp`). Strong as an oracle;
  "not a production speed path inside graph capture"
  (suggestions/findings/sources.md ~61-64).
- MiniMax Triton tuned configs are E=256, N=384 (not N=256) int4_w4a16
  on `Intel(R)_Arc(TM)_Pro_B70_Graphics`. Missing `"1"` key does NOT
  fall back; decode uses the nearest larger key
  (`experiments/minimax_moe_tuned_configs/README.md` L10-14).
- Direct-gather reuse for MiniMax local-argmax: quality OK, speed
  61.289 vs 61.404 -- rejected
  (`patches/minimax-direct-gather-reuse-no-improvement-20260517.md`).
- Community vendor tree now mirrors 0xSero / SergiioB / dominick253
  (`community/`).
- Detached dirty snapshot of `moe_layerlet.cpp` + `fused_moe_interface.py`
  (`experiments/qwen36-35b-quark-int8-b70/notes/2026-07-04-vllm-xpu-kernels-detached-dirty-snapshot.md`):
  int32/int64 topk_ids templating in the layerlet prologue.

File:line (on-disk snapshot unless noted):
- `/mnt/vm_8tb/b70/b70-optimization-lab-main/suggestions/findings/qwen35-b70-options.md:358-397`
  W8A8 layerlet / workspace / gate-trace plan
- `/mnt/vm_8tb/b70/b70-optimization-lab-main/suggestions/findings/sources.md:91-95,157-161`
  fused_moe_interface.py still allocates scratch; vllm-xpu-kernels #390/#392
- `/mnt/vm_8tb/b70/b70-optimization-lab-main/tools/onednn_moe_island_resident_runner.cpp:1-60,1021-1029`
  resident oneDNN island
- `/mnt/vm_8tb/b70/b70-optimization-lab-main/experiments/minimax_moe_tuned_configs/README.md:1-37`
  E=256 N=384 Triton tiles; decode key `"1"` required
- `/mnt/vm_8tb/b70/b70-optimization-lab-main/docs/vllm-intel-upstream-candidates.md:80-92,327-351`
  MoE route-capture patches + graph-safe collectives
- GitHub `99ca073` files under `experiments/qwen38-27b-b70/notes/` and
  `data/2026-08-19-gdn-prefix-parity-*.json`

Steal vs skip for Ornith:
- STEAL (HIGH pattern, MED code): grow-only workspaces + never
  `unique().tolist` inside capture. Steve's oneDNN sidecar uses
  `.tolist()` in capture patches -- do NOT copy that
  (`patches/vllm-xpu-qwen36-onednn-sidecar-*-20260613.patch` ~196).
- STEAL (MED): layerlet gate-trace discipline. Flags can be set and
  still miss the C++ op.
- SKIP: MiniMax QK-RMS, P2P-on recipes (`CCL_TOPO_P2P_ACCESS=1` in
  MiniMax promoted env), llama.cpp GGML, W8A8 INT8 quark layerlet as
  a drop-in for NVFP4.

================================================================================
2. SergiioB/intel-arc-pro-b70-inference-cookbook
================================================================================

Repo: https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook
Branch: `master` only
HEAD: `1378950` 2026-08-19T15:04:17Z
  "windows: 2026.08.19 draft-INT4 overlay, prefix cache on for sessions"

What we already knew (docs/20260818_qwen38_sergiioB_cookbook.md,
COOKBOOK_CAMPAIGN.md):
- 1x B70 Qwen3.8-27B GPTQ-Int4 + BF16 MTP4, 83.7 tok/s C1 p512/g128
- Image `vllm/vllm-openai-xpu@sha256:f01e24f6...` (0.27.2rc1 / kernels 0.1.12.3)
- `patch_mtp_nightly.py` then `patch_mtp_boundary.py` (we already ported)
- fp8 KV, XPU graph on, language-model-only
- MoE 35B GPTQ + BF16-preserved MTP is a real 1.3-1.7x lever on cookbook
  campaign (COOKBOOK_CAMPAIGN.md M5)

NEW since 2026-08-18 ingest:
- Image tag 2026.08.19: mixed-split v5 + draft-INT4 S+M1.
  Linux matched n=5: **112.65 vs 81.20** tok/s at p512/g128, cache off.
  Draft is requantized at start; checkpoint still ships BF16 MTP head.
- Prefix cache **on** for real sessions; off only for cold decode tests.
- `--generation-config auto`.
- Draft-INT4 is two patches:
  `patch_draft_lmhead_int4.py` (LM head INT4 g128 via
  `torch.ops._xpu_C.int4_gemm_w4a16`) and `patch_draft_mtp_int4.py`
  (five MTP linears: fc, qkv, o, gate_up, down).
- Critical compile rule (from the MTP INT4 patch docstring): hook
  `Qwen3_5MTP.load_weights`, NOT forward. Forward is
  `@support_torch_compile`; lazy construction in forward breaks Dynamo
  ("Failed to trace builtin operator print").
- 35A3 docs on this repo are quality-only
  (`docs/qwen36-35a3/QUANTIZATION-QUALITY.md`), not a fused-MoE kernel.
- Nemotron-3.5-Lightning DFlash remains a second image generation
  (official `method=dflash`); not NVFP4 MoE.

File:line (GitHub `1378950` patch text):
- `windows/Qwen38-Docker-Standalone/patches/patch_draft_lmhead_int4.py:1-40`
  (in commit 1378950) -- draft LM head INT4, target stays fp16
- `windows/Qwen38-Docker-Standalone/patches/patch_draft_mtp_int4.py` docstring
  "hook goes in Qwen3_5MTP.load_weights ... NOT in forward"
- `docs/qwen38-27/WINDOWS-STANDALONE.md` upgrade table: 112.65 vs 81.20

Steal vs skip:
- STEAL (HIGH for MTP unpark): split draft quant/backend from target.
  Build INT4/BF16 draft tensors at load time so the captured graph
  already sees the drafted method.
- STEAL (MED): prefix-cache-on is a session lever, not a cold-decode
  cheat. Do not copy `--no-enable-prefix-caching` into a shelf.
- SKIP: Windows kits, 83.7/112.65 as W8A8 or NVFP4 numbers, Nemotron
  grouped-topk / SSU, "NVFP4 unsupported on Intel" claim (false here).

================================================================================
3. 0xSero/qwen38-b70
================================================================================

Repo: https://github.com/0xSero/qwen38-b70
Branch: `main` only
HEAD: `17323a6` 2026-08-17T20:14:06Z
  "Fix: bash shell for oneAPI setvars in build"
On-disk: `/mnt/vm_8tb/b70/qwen38-b70` (same SHA)

What we already knew (llamacpp/QWEN38_B70_0XSERO.md, JOURNAL 2026-08-17):
- llama.cpp SYCL Q4_K_M TP=2, not vLLM
- Packaged steveseguin TP2 stack + mndodd/llama.cpp @4302fb5

NEW: none for MoE/vLLM. README still points at mndodd/llama.cpp and
b70-optimization-lab TP2 patches. MTP numbers are GGUF speculative
(easy 84.3 / hard 49.0 on TP2). vLLM XPU 0.27.2 is mentioned as a
reference, not a recipe.

1-hop: https://github.com/mndodd/llama.cpp

Steal vs skip:
- SKIP entire tree for Ornith NVFP4 fused apply (llama.cpp GGML/SYCL).
  Keep as the 3.8 dense Q4_K_M speed/quality ceiling only.

================================================================================
4. rmacy/vllm
================================================================================

Repo: https://github.com/rmacy/vllm (fork of vllm-project/vllm)
Default branch: `main`
HEAD (rmacy commit): `77242f1` 2026-08-16T23:35:55Z
  "fix(worker): handle list-type draft_token_ids in async scatter"
Created 2026-08-16. Fork still lists a huge inherited upstream branch
set (100+ names including `bugfix/37931-nvfp4-batched-all2all`,
`akaratza_fix_moe_bugfix`, humming CI shards). Those are upstream
vLLM branches copied at fork time, not rmacy B70 work.

rmacy-authored commits (already in vllm/dflash/DSPARK_RMACY.md):
- `72f353c` dflash SpecForge readout offset
- `d7c7222` sample all k draft slots at offsets 0..k-1
- `77242f1` pad list-type draft_token_ids before async scatter
  (`vllm/v1/worker/gpu_model_runner.py` ~1948-1960)

What we already knew: DSpark on dense Qwen3.8-27B FP8, image
`ghcr.io/rmacy/qwen38-fp8-dspark`, llm-scaler#620, SpecForge#769.

NEW for MoE: nothing. NVFP4 all2all / humming / cutlass branches are
CUDA upstream, not XPU fused apply.

1-hop: https://github.com/sgl-project/SpecForge/pull/769
       https://github.com/intel/llm-scaler/pull/620

Steal vs skip:
- STEAL (LOW for Ornith MTP later): pad variable-length draft ids
  without `.item()` if we ever do adaptive block spec on XPU graph.
- SKIP: DSpark FP8 dense path as an NVFP4 MoE kernel. SKIP CUDA
  NVFP4 batched all2all / humming.

================================================================================
5. intel/llm-scaler
================================================================================

Repo: https://github.com/intel/llm-scaler
Default branch: `main` HEAD `0c090e7` 2026-08-19T02:23:12Z
  "docs: remove obsolete oneAPI setup instructions (#627)"
Image: `intel/llm-scaler-vllm:0.21.0-b3.1` (Aug 2026) is current;
  our old kernel/20 work was 0.14.0-b8.3.1.

Remote branches checked (MoE-relevant subset of ~70):
- `add_moe_int4_kernel` 9aec602 2026-04-17
- `moe-topk-dispatch-fix` 4339446 2026-04-24
- `fix-moe-int4-topk-e128` 30ef249 2026-06-04
- `fix/qwen3-moe-xpu-graph-sticky-buffer` cdbe967 2026-06-30  [merged as #505]
- `fix/int4-moe-sticky-buffers-xpu-graph` 3da0545 2026-06-30  [merged as #507]
- `fix/moe-fp8-e4m3-dispatch` (e4m3 vs e5m2 decoder mismatch)
- `pr/fp8-block-moe-decode-kernel` 474fe96 2026-07-24  [merged as #561]
- `pr/fp8-block-moe-prefill-kernel`
- `optimize_moe_int4` / `optimize_moe_int4_kernel` / `update_moe_int4`
- `optimize_mtp` f88ec7b 2026-07-30  [merged as #579]
- `upgrade/v0.26.0`, `upgrade/vllm-xpu-v0.21.0` (sticky buffers land here)

What we already knew (kernel/20, contrib/llm_scaler_quark_int8_moe):
- 0.14.1 image had no `_moe_C` / no int8 fused MoE
- Entry-point swallowed serve args
- Quark W8A8 MoE needed contrib patches; int8 fused experts were a
  newer llm-scaler (~0.20) story
- P2P in serve wedges our box; their README still documents
  `CCL_TOPO_P2P_ACCESS=1` as a 15% large-batch win -- do not copy

NEW (this is the payload):

A. Sticky buffers = the GRAPH "!!!!" root cause for ESIMD MoE
   (`vllm/custom-esimd-kernels-vllm/csrc/moe_batch/moe.sycl` in
   commit cdbe967 / #505).
   Static `thread_local` scratch reallocated whenever
   `s_cached_ntokens != n_tokens`. Capture freezes `s_final_output`
   address. Larger eager batch reallocs and frees it. Small-batch
   graph replay reads freed memory -> NaN -> "!!!!".
   Eager always re-runs `ensure_moe_buffers`, so it looks fine.
   Fix: grow-only, return `.narrow(0,0,n_tokens)`,
   `MOE_STICKY_BUF=0` for A/B. Verified Qwen3.6-35B-A3B fp8 TP=2.
   INT4 analogue (#507, moe_int4.sycl): allocate once at cap 64,
   4-deep `s_int4_final_outputs` ring because the caller holds the
   tensor across TP AR + residual + next layer.

B. Per-shape decode buffer cache for block-FP8 MoE
   (`moe.sycl` `s_block_buffers` array, later #584 extends exact
   kernels through batch 12). "Capture and serving commonly
   alternate among 1..4 decode tokens. Keep one persistent slot
   per token count." Concurrent multi-stream use unsupported;
   one compute stream per TP worker.

C. M==1 1D GEMV vs grouped DPAS (#491 in upgrade/v0.21.0).
   16-wide `lsc_load_2d` fills 1/4 of BMG 64B cacheline.
   `block_load<uint8_t,256>` along K restores bandwidth.
   Prefill: `moe_up_fp8_grouped` / `moe_down_fp8_grouped` with
   GPU `moe_build_tile_map_kernel` (no d2h). Tile-parallel to
   kill expert-skew long tails. MG=2 amortizes fp8 weight load
   across two DPAS. Direct e4m3->fp16 bit map beats widen-to-fp32.
   Isolated M=256: 1.15 ms vs Triton fused_moe 6.9 ms (~6x).

D. TopK V2 dispatch by (E, top_k) with heap fallback
   (4339446, 30ef249). Known fast paths include (256,8) -- our
   Ornith shape -- plus (128,8)/(128,10)/(512,10). Unknown
   shapes must not TORCH_CHECK(false) on prefill.

E. MTP (#579 / optimize_mtp, 2026-07-30):
   - Qwen speculative GDN ESIMD kernel + fused conv checkpoint
     writes (unique Q/K/V owners)
   - Gemma MTP: `moe_forward_full_fp8_grouped` takes routing
     from the caller so Python does not rebuild intermediates
   - `esimd_gemv_fp16_gelu_mul` fused gate-up
   Official README 3.5 now documents
   `--speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":2}'`
   on Qwen3.6-35B-A3B. That is dense-head MTP on Intel's stack,
   not NVFP4 expert + unquantized drafter MoE coexistence.

F. e4m3 vs e5m2 dispatch (#494): feeding e4m3 weights to e5m2
   decoders silently "!!!!". Scheme must match the checkpoint.

File:line (GitHub commits; line numbers from the patches):
- `moe.sycl` ensure_moe_buffers grow-only (cdbe967, ~1452-1576)
- `moe_int4.sycl` S_INT4_MAX_NTOKENS=64 + 4-ring (3da0545, ~1373-3153)
- `moe.sycl` s_block_buffers comment (474fe96, ~1653)
- `moe_fp8_grouped.h` tile-map + MG=2 + grouped accumulate
  (f88ec7b / #579)
- `vllm/README.md` section 3.5 MTP Enable (main 0c090e7)

Steal vs skip:
- STEAL (HIGH): sticky grow-only + output ring. This is the
  exact failure mode if our fused apply ever reallocates a
  captured workspace or returns a static buffer that the next
  layer overwrites before TP AR finishes.
- STEAL (HIGH): M=1 1D K-load vs grouped prefill. Matches our
  T<=16 Python-int gemms; next win is fewer launches via a
  grouped NVFP4 kernel, not more Python unique().
- STEAL (MED): separate grouped-forward API that accepts
  precomputed routing -- useful if MTP draft MoE and target
  NVFP4 MoE must not share `moe_backend`.
- SKIP: P2P-on, `--enforce-eager` as a default, 0.14.x image,
  MiniMax-only logits paths, CUDA cutlass names in the INT4
  "tiny_cutlass_nmajor" symbols (they are ESIMD on XPU).

================================================================================
6. 1-hop citations (READMEs of the five repos only)
================================================================================

| hop | why |
|---|---|
| https://github.com/vllm-project/vllm-xpu-kernels issues/390 PR/392 | reusable fused MoE workspaces; PR still OPEN, mergeable_state=dirty, updated 2026-08-07. Author ehartford / QuixiAI. |
| https://github.com/QuixiAI/vllm-xpu-kernels | head of PR 392 |
| https://github.com/intel/intel-xpu-backend-for-triton/issues/6389 | grouped-GEMM tile vs real skewed routes (Steve sources.md) |
| https://github.com/mndodd/llama.cpp | 0xSero SYCL source -- skip GGML |
| https://github.com/sgl-project/SpecForge/pull/769 | rmacy DSpark training -- dense |
| https://github.com/intel/llm-scaler/pull/620 | rmacy DSpark on llm-scaler -- dense |
| https://github.com/sgl-project/sglang/issues/28511 and /pull/28695, https://github.com/Johnny-Liou/ReplaySSM | Steve ReplaySSM notes; GDN spec unlock, not NVFP4 MoE |

No extra B70 fused-NVFP4 MoE github appeared in those 1-hop READMEs.

================================================================================
7. Already-knew vs NEW vs skip (cross-repo)
================================================================================

Already knew (do not re-derive):
- Steve capture-safe AR / graph-with-comm; we use SYCLKERNELS=1
- Quark W8A8 INT8 35B story and the 0.14.1 `_moe_C` hole (kernel/20)
- Cookbook GPTQ-Int4 + BF16 MTP 83.7; patches we already have
- 0xSero is llama.cpp SYCL
- rmacy is DSpark readout, not MoE kernels
- P2P-on wedges this box
- `unique().tolist()` / `.item()` host-syncs graphs; we already
  replaced routing gather with `index_select` + Python-int M=1

NEW and actionable for Ornith NVFP4 GRAPH:
- Sticky grow-only MoE scratch + output ring (llm-scaler #505/#507)
- Per-M persistent decode slots (block MoE #561/#584)
- M==1 1D GEMV vs grouped prefill (llm-scaler #491)
- Draft vs target backend split, load_weights-time quant
  (cookbook 2026.08.19 draft-INT4)
- TopK V2 (256,8) on-device dispatch
- vllm-xpu-kernels #392 still unmerged workspace API
- GDN persistent scratch zero-init as GRAPH residue contract
  (Steve 99ca073) -- relevant when unparking MTP, not the fused apply

Explicit skip list:
- MiniMax-only QK-RMS / oproj / delayed-AR fusions
- CUDA-only: humming, cutlass NVFP4 all2all, DeepEP/nvshmem
- llama.cpp GGML / SYCL Q4 (0xSero, mndodd, Steve q4 patches)
- P2P-on (`CCL_TOPO_P2P_ACCESS=1`) in any TP>1 serve
- llm-scaler 0.14.x as a serve image
- Steve oneDNN sidecar `.tolist()` capture patches
- Photocopying 101.9 / 112.65 / 83.7 dense INT4+MTP onto NVFP4 MoE
- W4A4, TurboQuant Triton tiles, EP-comm-dominated layouts

================================================================================
8. Mapping onto OUR current fused apply
================================================================================

Our apply (JOURNAL 2026-08-18/19 campaign): capture-safe gather via
`index_select`, Python-int M=1 gemms, unique() eager-only. That already
matches Intel's "decode M==1, prefill grouped" split at the Python
level.

Gaps vs this review:
1. Workspace lifetime under GRAPH+TP=2. If any fused-apply buffer is
   `empty()` on shape change, we inherit the #505 "!!!!" bug. Prefetch
   max T, grow-only, ring the returned hidden if the next collective
   aliases it.
2. Launch count. Intel fused the M==1 path into one ESIMD op
   (logits in, hidden out). We still fire per-expert gemms in Python.
   Next speed step is a grouped NVFP4 kernel or a persistent layerlet,
   not more routing Python.
3. MTP. Intel documents `method=qwen3_5_mtp` on 35B-A3B **on their
   FP8/INT4 ESIMD MoE**, not on NVFP4 emulation. Cookbook proves the
   draft can be a different quant than the target if it is built at
   load_weights. Our parked MTP is a **backend chooser** bug
   (emulation vs triton), not a missing GDN kernel. Steal the split:
   target stays NVFP4 fused/emulation; drafter MoE gets its own method
   and must not call `moe_backend=triton` on NVFP4 tensors.
4. Do not turn P2P on to chase Steve MiniMax or llm-scaler README
   numbers.

Verdict: the only community code that changes our Ornith GRAPH plan
this week is llm-scaler sticky buffers / M=1 GEMV / grouped prefill,
plus cookbook's load_weights-time draft split for MTP unpark.
Everything else is corroboration or skip.

================================================================================
9. Local clones (2026-08-19, after this review)
================================================================================

Cloned/fetched under /mnt/vm_8tb/b70/community_repos/ (git-ignored runtime,
not repo content):
- intel-arc-pro-b70-inference-cookbook  master 1378950 (only branch)
- qwen38-b70  main 17323a6 (only branch)
- rmacy/vllm  main 77242f1 + inherited upstream branches
- intel/llm-scaler  main + add_moe_int4_kernel, b8.1, b8.2, ...
steveseguin/b70-optimization-lab already at /mnt/vm_8tb/b70/b70-optimization-lab
(origin/main now 99ca073; only public branch is main).

LOOP 62 measured after this review:
- map emulation->XPUExperts: boots chooser, then dies
  cutlass_grouped_gemm_interface (op lives in stock int8g-v0260 _xpu_C,
  not in our fused NVFP4 overlay).
- map emulation->TritonExperts: MTP GO. Eager 4.3 vs no-MTP 5.5.
  GRAPH MTP3 21.2 vs no-MTP GRAPH 34.9. MTP is coherent and a net loss
  until the draft is cheap (cookbook load_weights INT4).
- TP=2 GRAPH + PUSH_AR_GRAPH: 22.5/23.9 vs 22.6/24.2 decode-neutral.
)
