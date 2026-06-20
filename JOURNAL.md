# B70 / Qwen3.6-27B Optimization Journal

Chronological log of every experiment. Newest entries at the **bottom** of each
section. Keep entries factual: config, command, result, verdict. When something
fails, record the error and the suspected cause so we don't repeat it.

Legend: [OK] works | [WARN] partial / caveats | [FAIL] failed | [WIP] investigating

---

## Environment baseline

| Item | Value | Confirmed |
|------|-------|-----------|
| Host | Unraid @ 192.168.10.5, hostname `Tower`, login `root` | [OK] |
| Unraid version | 7.3.1 | [OK] |
| CPU | AMD Threadripper 1950X, 16C / **32T** | [OK] |
| RAM | **125 GiB** (≈113 GiB free) | [OK] |
| GPU | 1x Intel Arc Pro B70 — **Battlemage G31** `[8086:e223]` @ PCI `44:00.0` | [OK] |
| GPU driver | kernel **`xe`** module (GuC 70.65.0, HuC 8.2.10, DMC v2.6); DRM `xe 1.1.0` | [OK] |
| `/dev/dri` | `card0` + `renderD128` present, perms 0777, by-path → `pci-0000:44:00.0` | [OK] |
| Docker | **29.5.2**, btrfs storage driver | [OK] |
| Docker root | `/var/lib/docker` = **50 GB `docker.img` loop** (3.4 G used) (!) size cap | [OK] |
| GPU passthrough to Docker | via `--device /dev/dri` | [TODO] to validate |
| **Work SSD (8 TB VM drive)** | `/dev/sdd1` = **Samsung 870 QVO 8TB**, ROTA=0, **`/mnt/vm_8tb`**, 6.9 TB free | [OK] |
| Other SSD | NVMe `nvme0n1` SPCC 954 GB @ `/mnt/cache` (cache pool, ~half full) | [OK] |
| Array (avoid) | 3x WD 10TB HDD (ROTA=1) @ `/mnt/disk1`,`disk2`,array | [OK] |

**Pre-existing on box:** docker images incl. `specula-qairt:2.45`, `python:3.10/3.11`; `/mnt/vm_8tb` already
has `domains/` (VM disks), `specula-build/`, and a 64 GB `swapfile`. Existing containers (nextcloud,
mariadb, syncthing, clamav) run off the same docker.img — **do not disrupt** them when pruning images.

### Target hardware: Intel Arc Pro B70 (confirmed specs)
- Xe2 "Battlemage" (G31), **32 GB GDDR6**, 256-bit, **608 GB/s** bandwidth, ECC capable.
- 32 Xe-cores, **256 XMX engines**, **367 INT8 TOPS** (the INT8 fast path we want), 22.94 FP32 TFLOPS.
- PCIe 5.0 x16, 230 W TBP. Launched 2026-03-25, ~$949.
- **Decode is memory-bandwidth-bound.** Single-stream ceiling ~= 608 GB/s / model-bytes:
  - Q4 (~16 GB): ceiling ~38 tok/s   |   Q8 (~28 GB): ceiling ~21 tok/s (before KV overhead).
  - Beating this ceiling is exactly what MTP / speculative decoding is for.

### Target model: Qwen3.6-27B
- Dense 27B (post Qwen3.5/Feb-2026). Context 262K native, extensible ~1M.
- **Ships MTP layers** -> multi-token prediction / speculative decoding, ~1.5-2x faster, no acc loss.
- VRAM budget on 32 GB B70:
  - Q4_K_M ~16 GB  -> fits easily, lots of KV headroom. [primary 4-bit target]
  - Q8_0   ~28 GB  -> fits, tight KV. [primary 8-bit target; pair with MTP]
  - BF16   ~54 GB  -> does NOT fit on one card; skip/offload only.
  - NVFP4 = NVIDIA FP4, not applicable to Intel XMX -> skip.
- Candidate repos:
  - `Qwen/Qwen3.6-27B` (official full weights; source for IPEX-LLM sym_int8/sym_int4, AWQ/GPTQ)
  - `unsloth/Qwen3.6-27B-GGUF` and `unsloth/Qwen3.6-27B-MTP-GGUF` (Q4_K_M, Q8_0; MTP variant for spec-decode)
  - `RDson/Qwen3.6-27B-MTP-Q4_K_M-GGUF`, `havenoammo/Qwen3.6-27B-MTP-UD-GGUF` (community)

### Key constraints / decisions
- **C1 — docker.img is 50 GB.** Big backend images (vLLM-XPU, IPEX-LLM ~5–15 GB each) will overflow it.
  → Keep ALL heavy data (model weights, HF cache, builds) on `/mnt/vm_8tb` via bind-mounts, never in image
    layers. Prune backend images between experiments. Revisit (expand docker.img / data-root move) only if needed.
- **C2 — Battlemage is new.** Needs recent Intel compute-runtime / Level-Zero + oneAPI 2025.x in containers.
  Host `xe` driver is loaded & healthy; user-space compute stack lives inside containers (validate next).
- **D1 — All project data under `/mnt/vm_8tb/b70/`** (models, hf_cache, results). SSD, fast, 6.9 TB free.

---

## Matrix to cover

Backends:
- [ ] llama.cpp (SYCL backend)
- [ ] llama.cpp (Vulkan backend)
- [ ] vLLM upstream (XPU / SYCL)
- [ ] Intel vLLM (`vllm-xpu` / IPEX-LLM)
- [ ] IPEX-LLM (llama.cpp portable / Ollama portable)

Quant formats:
- [ ] GGUF Q8_0 (8-bit)
- [ ] GGUF Q4_K_M (4-bit)
- [ ] INT8 (XMX fast path, e.g. via IPEX-LLM `sym_int8`)
- [ ] INT4 (IPEX-LLM `sym_int4` / AWQ / GPTQ)
- [ ] AWQ / GPTQ (if backend supports on XPU)

Techniques:
- [ ] Speculative decoding (draft model)
- [ ] MTP (multi-token prediction — Qwen3.6 ships MTP layers)
- [ ] Tensor/flash attention paths on XPU
- [ ] CPU/RAM offload to 128 GB DDR4 (partial -ngl, KV offload, MoE offload)
- [ ] Concurrency / batching throughput sweep (1..32 parallel requests)
- [ ] Multi-GPU: tensor-parallel vs pipeline-parallel on 2x B70 (PCIe3 x16) [next week]

Metrics (per run): PP tok/s | TTFT ms | TG tok/s (single + aggregate) | peak VRAM | power.
Full roadmap, sweep matrix, and dual-card plan: see `STRATEGY.md`. Literature: `docs/literature/`.

---

## Log

### 2026-06-17 — Project kickoff
- Generated dedicated SSH key (`b70_unraid_ed25519`), added `b70` host alias.
- Host reachable, SSH port open. Awaiting GUI key authorization for passwordless login.
- Scaffolded repo + journal. Next: confirm GPU, driver, Docker, and 8TB SSD path on box.

### 2026-06-17 — GPU passthrough validated [OK]
- Image: `ghcr.io/ggml-org/llama.cpp:full-intel` (note: wrapper entrypoint; real bins in `/app`, NOT on PATH).
- `docker run --device /dev/dri --entrypoint sycl-ls <img>` enumerates the B70 on BOTH backends:
  - `[level_zero:0] Intel(R) Arc(TM) Pro B70 Graphics` (Level-Zero V2 [1.15.38308]) <- preferred for SYCL/IPEX
  - `[opencl:gpu] Intel(R) Arc(TM) Pro B70 Graphics OpenCL 3.0 NEO [26.18.38308.1]`
- clinfo: **B70 usable global mem = 30.3 GiB** (of 32 GB). CPU OpenCL also present (Threadripper, 125.7 GiB).
- Compute runtime is recent enough for Battlemage out of the box. No host-side driver work needed to start.
- Implication: Q4 (~16 GB) comfortable; Q8_0 (~28 GB) fits with ~2 GB KV headroom (short ctx) — tight.
- To run tools in this image use `--entrypoint /app/<binary>` (llama-bench, llama-cli, llama-server,
  llama-speculative, llama-bench, etc. all present, incl. spec-decode + lookup-decoding tools).

### 2026-06-17 — PCIe link diagnosis (important)
- Card UPLINK to system (`42:00.0`): LnkCap 32GT/s x16 (Gen5-capable), **LnkSta 8GT/s x16 "downgraded"**
  => running **PCIe 3.0 x16 (~16 GB/s)** because the TR 1950X host is Gen3. As expected.
- Internal link to GPU die (`44:00.0`, behind on-card bridge `43:01.0`): **reports Gen1 x1**
  (`max_link_width=1` in sysfs). Anomalous. Either (a) real internal link-train issue capping GPU
  traffic at ~250 MB/s, or (b) a Battlemage big-die reporting quirk (real mem path not this PCIe link).
- DISAMBIGUATION: model-load time in the next bench tells us. ~60s load for 15.66 GB => x1 real;
  few seconds => cosmetic. On-GPU prefill/decode don't cross PCIe, so throughput numbers valid either way.
- TODO before dual-card: resolve this (BIOS PCIe gen/ASPM, slot wiring, firmware) — matters for offload + TP.

### 2026-06-17 — Literature digest: multi-GPU on 2x B70 (see docs/literature/02_multigpu.md)
- **Intel Arc has NO GPU-to-GPU P2P.** TP all-reduce round-trips through host RAM over PCIe; oneCCL is
  CPU-driven. On our PCIe3 x16 this is worse than the published dual-B70 benches (which were PCIe5 x8).
- => **Prefer pipeline/layer-split (PP) over tensor-parallel (TP)** on our box. PP comm is ~1000x lighter.
- Expect only **~1.0-1.3x single-request speed** from 2 dense-model cards (layer-split efficiency ~30-37%
  of combined BW). 2 cards buy **capacity** (64 GB pooled: 70B-4bit dense, 80B MoE, longer ctx, 2 instances),
  and **aggregate throughput under concurrency** — mostly for **MoE** models.
- llama.cpp SYCL has no TP yet (layer-split works). vLLM-XPU TP fragile on Battlemage (bug #41663 open,
  needs CCL_ENABLE_SYCL_KERNELS=0 + --enforce-eager). Best homelab throughput may be **2 independent
  single-card instances** (data-parallel, ~0 cross-card penalty).
- Top dual-card configs to test: (1) llama.cpp layer-split 70B-Q4_K_M capacity check; (2) vLLM-XPU TP=2
  MoE FP8 concurrency sweep; (3) two independent instances.

### 2026-06-17 — Literature digest: backends (see docs/literature/01_backends.md)
- **Backend bet: llama.cpp SYCL first** (~2.2x Vulkan; 27B Q4_K_M ~20 t/s decode / ~700 t/s prefill reported).
- **LANDMINE:** SYCL can collapse to ~CPU speed on old DDR4/PCIe3 hosts (#22413). We ARE DDR4/PCIe3
  (1950X) => must benchmark Vulkan as fallback and watch our SYCL numbers vs the ~20 t/s reference.
- **8-bit reality:** GGUF **Q8_0 is a Xe2 decode regression (~4x slower than Q4_K_M)**; INT8 XMX fast path
  is NOT used in decode (memory-bound). XMX INT8 only helps **prefill**. => For the 8-bit goal, prefer
  **vLLM-XPU FP8**, not GGUF Q8_0. Revise Phase 1: keep Q4_K_M as the workhorse; treat Q8_0 as a
  data point, not a target.
- **IPEX-LLM is ARCHIVED (read-only since 2026-01-28)** — drop it; XPU moved into PyTorch >=2.8. Use Intel
  **LLM-Scaler vLLM** (official B70 support since 0.14.0-b8.2). Avoid vanilla vLLM 0.19.0 GPTQ on XPU (broken #39474).
- Required env baked into bench: `UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1`, immediate cmdlists,
  `SYCL_CACHE_PERSISTENT=1` (skip ~27s JIT), `-fa 1`. `GGML_SYCL_F16` gave +139% prefill upstream.
- Opportunity for upstream contribution: XMX DPAS/joint_matrix decode kernels + Q8_0 quant-layout fix.

### 2026-06-17 — [FAIL] Qwen3.6-27B Q4_K_M segfaults on llama.cpp (CPU *and* GPU)
- Image `ghcr.io/ggml-org/llama.cpp:full-intel` (libllama build 9680). Model loads + sets up fully,
  then **exit 139 (SIGSEGV) at the first forward pass** (`generate:` line) — with `-ngl 99`, `-ngl 4`,
  AND `-ngl 0` (pure CPU). So NOT a SYCL/Battlemage/passthrough issue.
- GGUF declares `general.architecture = qwen35` (Qwen3.6 reuses Qwen3.5 arch family). Per literature,
  Qwen3.6-27B is DENSE and uses **Gated-DeltaNet** (linear) attention. This build's qwen35/DeltaNet
  graph crashes on first eval -> llama.cpp arch-support bug, or GGUF needs a newer build than 9680.
- Ruled OUT: bad passthrough (sycl-ls sees B70), OOM (crash is pre-warmup, small ctx), flash-attn
  (crashes without it), interactive/stdin EOF (crashes with -no-cnv).
- Notable side-finding: model load reached device-fit in ~12s / CPU load ~5s => effective bandwidth
  ~1+ GB/s, so the PCIe "Gen1 x1" register reading is COSMETIC, not a real ~250 MB/s bottleneck. [resolved]
- NEXT: (a) prove B70 SYCL pipeline with a known-good standard-attention model (first real tok/s);
  (b) find llama.cpp build/commit that supports Qwen3.6 Gated-DeltaNet; (c) candidate upstream contribution.
- ROOT CAUSE (confirmed via Unsloth card + llama.cpp docs): Qwen3.6 Gated-DeltaNet needs the **absolute
  latest llama.cpp** for the new operators. Build 9680 in `:full-intel` is too old. Arch = 64 layers:
  `16 x (3x(GatedDeltaNet->FFN) -> 1x(GatedAttention->FFN))`; 48 linear-attn V heads / 16 QK, 128-dim.
  FIX PATH: build llama.cpp SYCL from latest source in a container (also needed to contribute kernels),
  or pull a newer intel image tag if one exists. DeltaNet support is immature across backends (vLLM XPU
  also chokes on gdn_attention) -> this is a real bleeding-edge integration + contribution target.

### 2026-06-17 — [OK] BREAKTHROUGH: B70 SYCL inference works; crashes were a POISONED CACHE
- Root cause of ALL prior segfaults: the **SYCL persistent kernel cache got poisoned** by the very first
  run (script 06) which crashed mid-JIT while `SYCL_CACHE_PERSISTENT=1` wrote to `/sycl_cache`. Every later
  run that READ `/sycl_cache` (08/09/11) then segfaulted on the corrupt cached kernel — incl. the
  known-good 7B. NOT a DeltaNet bug, NOT a passthrough/hardware bug, NOT my "optimization" env flags per se.
- Proof: 7B Q4_K_M with NO cache (script 12, configs A-D all variants) **generates correctly**:
  - **decode (eval) ~89-90 tok/s** for Qwen2.5-7B Q4_K_M on the B70 (11.1 ms/token). [first real number]
  - Level-Zero and OpenCL backends both work; immediate-cmdlists on/off both fine; SYSMAN optional.
  - First-run **JIT compile ~50 s** with no cache => a CLEAN persistent cache is essential (just never let a
    crashed run poison it; clear `/sycl_cache` after any crash before retrying).
- TOOLING BUG: `llama-bench` segfaults right after backend load on this image (before compute) — separate
  from the cache issue. Use `llama-completion` / `llama-server` for benchmarking until understood.
- Lesson baked into scripts: clear `.sycl_cache` on any crash; don't trust a cache written by a crashed run.

### 2026-06-17 — Community data point (user-supplied): dual-B70 TP2+MTP negative result
- Setup: 2x Arc Pro B70, Ubuntu 24.04, kernel 6.17.0-14; **vLLM 0.20.1**, **INT4 AutoRound W4A16**, TP=2,
  flash_attn, spec-decode+MTP (patched fallback), ctx 4096, batch 1.
- Result: **NEGATIVE** — TP2+MTP slower than TP2 non-MTP AND slower than single-B70 MTP. Latency 7.20s.
  Warmup MTP acceptance 62.6%. Note: **vLLM disables XPU graph capture for TP2 comm ops** (perf hit).
- Takeaways: (1) corroborates "MTP is a SINGLE-card win, not dual"; (2) a working B70 vLLM lane exists:
  vLLM 0.20.1 + AutoRound W4A16 + (patched) MTP; (3) AutoRound W4A16 is our vLLM 4-bit format (not GGUF);
  (4) XPU-graph-capture-disabled-on-TP2 = candidate upstream contribution. -> digging into vLLM/LLM-Scaler
  versions for B70+Qwen3.6+MTP+AutoRound (research launched).

### 2026-06-17 — Qwen3.6-27B confirmed crashing on llama.cpp SYCL b9680 (clean cache)
- Re-test with FRESH cache + minimal env: 7B works, but **Qwen3.6-27B still SIGSEGV at first eval**.
  So this is a REAL Gated-DeltaNet support gap on build 9680, independent of the (now-fixed) cache bug.
- Hypothesis (from user's Windows data point): **DeltaNet works on Vulkan/CPU but not yet SYCL** at this
  build. Testing CPU next. Fix paths ranked: (1) build llama.cpp SYCL from latest source (>b9680);
  (2) Vulkan on Linux (needs Mesa ANV for Battlemage - unverified); (3) vLLM-XPU AutoRound W4A16.

### 2026-06-17 — Community data point (user): single-B70 Vulkan on Windows = ~107 t/s
- 1x B70, Win11 native (NOT WSL - "B70 Vulkan needs native Windows"), AMD 9800X3D, 128GB DDR5.
- **upstream llama.cpp b9553, Vulkan backend, Q4_K_M, -ngl 99, ctx 262144, KV fp16, batch 2048**.
- **Median ~107.78 t/s** (this speed => almost certainly the 35B-A3B MoE class, not 27B dense ~20 ceiling).
- CRITICAL Vulkan flags/landmines:
  - `--device Vulkan2` REQUIRED to pin the B70 (else layer-splits across RTX3090+iGPU+B70; iGPU on system
    RAM tanks it 107 -> 22 t/s). (For us: single GPU, but the pinning lesson generalizes.)
  - DO NOT USE on this Vulkan build: `-fa` (crashes/degrades); `-ctk/-ctv q8_0` (107 -> 8-13 t/s,
    Vulkan KV-quant unoptimized); `--spec-type draft-mtp` (107 -> 27 t/s, 4x regression — upstream bug
    **#23769**, PR **#24312** closed/unfixed as of b9553).
- Implication: Vulkan is a strong plain-Q4 path on B70; but flash-attn/KV-quant/MTP all regress there now.
  SYCL (when DeltaNet lands) keeps flash-attn + KV-quant + MTP options open. Worth racing SYCL vs Vulkan.

### 2026-06-17 — [OK] Precise diagnosis: Qwen3.6 DeltaNet works on CPU, crashes on SYCL
- Qwen3.6-27B `-ngl 0` (CPU), clean cache: **generates correctly** ("Paris. The capital of Spain is
  Madrid.") at ~2.15 t/s, exit 0. `-ngl 99` (SYCL): SIGSEGV at first eval.
- CONCLUSION: **Gated-DeltaNet is a llama.cpp SYCL-backend gap (build 9680)** — CPU impl exists, SYCL
  kernels don't. => Upstream contribution target: implement/fix DeltaNet ops in ggml-sycl. Until then,
  llama.cpp SYCL cannot run Qwen3.6 on the GPU. (Vulkan reportedly can — user's Win data point.)

### 2026-06-17 — Community data points (user): vLLM-XPU runs Qwen3.6 on B70 (incl. INT8 + MTP)
- **[8-bit / INT8!]** Qwen3.6-**35B-A3B** Quark **W8A8 INT8**, **4x B70**, vLLM 0.20.2rc1.dev2, TP=4,
  ctx 32768: **99.77 t/s output, TTFT 76.5 ms** (512/512, b1, temp0). ~31.9 GiB/card @ util 0.95.
  -> This is the REAL INT8 8-bit path (Quark W8A8), not GGUF Q8_0. Uses XMX INT8.
- **[27B + MTP]** Qwen3.6-**27B** BF16, **4x B70**, **`intel/llm-scaler-vllm:0.14.0-b8.3`**, TP=4:
  decode **~54 t/s**, prefill **~2100 t/s**, TTFT good. MTP via **vllm_xpu_kernels v0.1.9** + qwen3_5.py
  spec-wiring patch (**vLLM #43565**) + Half-KV; num_spec=5, accept 88.9% (spec=3) / mean accept len 4.04.
  **2.9x faster than llama.cpp 27B Q8 (15.6 t/s b1).** 256K native ctx.
- DECISION: **vLLM-XPU (intel/llm-scaler-vllm) is THE path for Qwen3.6 on B70.** It runs DeltaNet on GPU,
  supports MTP and W8A8 INT8. llama.cpp = standard models + future DeltaNet-SYCL contribution.
- Single-B70 (32 GB) fit check: 27B BF16 ~54 GB (needs >=2 cards); **27B W8A8 INT8 ~27 GB (fits 1 card,
  tight)**; **27B AutoRound W4A16 ~15 GB (fits 1 card easily)**. 35B-A3B W8A8 ~35 GB (needs >=2 cards).
  -> On our single card now: target 27B **W4A16** (4-bit) and **W8A8 INT8** (8-bit). MoE waits for card #2.

### 2026-06-17 — PRIORITY set by user: 8-bit, but ONLY via the XMX INT8 fast path
- User values 8-bit accuracy/reliability, but ONLY if it leverages Intel INT8 fast paths. Correct instinct.
- THE distinction:
  - **GGUF Q8_0 = W8A16 (weight-only)**: dequant->FP16 GEMM, XMX INT8 NOT used, Xe2 regression. AVOID.
  - **W8A8 INT8 (Quark/AutoRound, vLLM-XPU)**: INT8xINT8->INT32 GEMM -> **XMX DPAS / 367 INT8 TOPS**. USE.
- => Primary 8-bit target = **Qwen3.6-27B W8A8 INT8 on vLLM-XPU**. Fits 1 card (~27 GB, tight KV).
- Fast path dominates PREFILL + batched decode; single-stream b1 decode still ~memory-bound (27 GB),
  latency between Q4 (15 GB) and FP16 (54 GB), but far more accurate. Sweet spot for reliable+fast-under-load.
- MUST VERIFY the INT8 path is truly engaged (not a silent W8A16 fallback): inspect vLLM quant-kernel
  selection in logs, prefill t/s near ~2100 class, and xpu-smi XMX/EU utilization. Bench W8A8 vs W4A16
  head-to-head (decode/prefill/TTFT + quality spot-check).
- Need a Qwen3.6-27B W8A8 checkpoint: find on HF (Quark/AutoRound) or produce with the quant tool.

### 2026-06-17 — Reality check + the MTP lever (re: the 54/2100 BF16 number)
- The 54 t/s decode / 2100 t/s prefill BF16 run was **4x B70 (TP=4)** — BF16 27B ~54 GB does not fit on
  1-2 cards. It's a multi-card CEILING, not single-card. Set expectations accordingly.
- The real hero is **MTP**: decode is bandwidth-bound (1 weight-read/token); MTP verifies K draft tokens
  per read, so effective decode ~= bandwidth_ceiling x mean_accept_len (their run: accept len 4.04).
- => MTP favors SMALLER resident footprints (higher ceiling to multiply). Projected single-card (MEASURE):
  - W4A16+MTP (~15 GB): ceiling ~40 t/s x ~2 -> ~60-90 t/s (4-bit).
  - **W8A8 INT8+MTP (~27 GB): ceiling ~22 t/s x ~2 -> ~40-55 t/s (~FP16 accuracy) <- user's sweet spot.**
- So single-card W8A8+MTP may ~match the 4-card BF16 decode at 8-bit accuracy. 2 cards push higher +
  unlock BF16. 4 cards = the full 54/2100. Prefill scales with cards (XMX-bound).
- Target ladder: 1 card = W8A8+MTP; 2 cards = faster + BF16 possible; 4 cards = BF16+MTP dream.

### 2026-06-17 — vLLM-XPU recipe (from docs/literature/04_vllm_versions.md) + checkpoint plan
- Image: pin **`intel/llm-scaler-vllm:0.14.0-b8.3.1`** (b8.3 fallback, currently pulling) = only validated
  B70 path. Upstream `vllm`/`intel/vllm:*-xpu` is the OTHER lineage (user's W8A8+patched-MTP data points).
- Supported-image quant: online **`--quantization fp8`** (best XMX path) or **`sym_int4`**. W8A8 INT8 (Quark)
  = upstream lineage. MTP is NOT a stock toggle (broken: gdn_attention spec masks, llm-scaler #386; real fix
  vLLM #43565 only in v0.23.0). 54 t/s run used a MANUAL patch. `ngram` spec-decode = low-risk first.
- Mandatory flags: `--enforce-eager`, `--dtype float16`, `--block-size 64`. Envs:
  VLLM_OFFLOAD_WEIGHTS_BEFORE_QUANT=1 (use 128GB DDR4 for online quant), VLLM_WORKER_MULTIPROC_METHOD=spawn,
  VLLM_ALLOW_LONG_MAX_MODEL_LEN=1. TP2 later: CCL_TOPO_P2P_ACCESS=1, CCL_ENABLE_SYCL_KERNELS=0.
- Pre-quantized checkpoints found (skip 54GB BF16):
  - **`Qwen/Qwen3.6-27B-FP8`** OFFICIAL, block-128, "near-identical" quality, ~27GB -> FITS 1 card.
    [FIRST TARGET: downloading now] 8-bit, supported path, fastest reliable GPU run.
  - `Minachist/Qwen3.6-27B-INT8-AutoRound` — INT8, but "48GB VRAM recommended" -> likely WON'T fit 1 card
    (ceiling case -> smaller model for INT8+MTP test now, full 27B on 2 cards per user plan).
- Plan: FP8 27B single-card first (prove vLLM-XPU pipeline + first number + verify XMX via prefill t/s &
  xpu-smi), then sym_int4, then INT8/W8A8 + MTP (upstream path, the user's north star) as a sub-project.
- User guidance: maximize single-card now (short ctx OK to prove W8A8+MTP mechanism); smaller model if we
  hit VRAM ceiling; card #2 in ~1-2 days for full 27B W8A8+MTP at long context.

### 2026-06-17 — [FAIL] Qwen3.6-27B-FP8 on llm-scaler b8.3: DeltaNet+FP8 init bug (NOT OOM)
- Downloaded official `Qwen/Qwen3.6-27B-FP8` (29 GB, per-layer safetensors). vLLM-XPU recognized arch
  **Qwen3_5ForConditionalGeneration**, quantization=fp8, cutlass FA XPU backend, mamba/DeltaNet page-size
  config — all good — then crashed in engine init (exit 1, NOT GPU OOM; crash is pre-weight-alloc):
  `AttributeError: 'MergedColumnParallelLinear' object has no attribute 'weight'`
  at `qwen3_5.py:1108  _qkvz_sz = self.linear_attn.in_proj_qkvz.weight.shape[0]`.
- Root cause: the DeltaNet linear_attn `in_proj_qkvz` reads `.weight.shape` at __init__, but the FP8 quant
  method hasn't materialized `.weight` yet. Real bug in image b8.3 (DeltaNet + FP8 combo). Never reached
  the VRAM-fit question. Bleeding-edge integration bug (candidate report/contribution).
- Config used: TP1, --dtype float16, --enforce-eager, --block-size 64, --max-model-len 2048, util 0.97.
- NEXT: research recommends **b8.3.1** (Intel's pinned version for Qwen3.6-27B); we're on b8.3 fallback.
  Plan: grow docker.img (need space + general infra), pull b8.3.1, retry FP8. If still broken, pivot to
  upstream vLLM + Quark W8A8 INT8 (the user's working data-point lineage).

### 2026-06-17 — DeltaNet+quant bug root-caused; clean env workaround found
- Same `in_proj_qkvz.weight` AttributeError on BOTH b8.3 and b8.3.1 -> not version-specific.
- ROOT CAUSE (read qwen3_5.py:1095-1135): it's an Intel **ESIMD "fused input norm"** optimization,
  enabled ONLY for `fp8`/`sym_int4`, that pre-allocates buffers from `in_proj_qkvz.weight.shape[0]`.
  For quantized linears `.weight` doesn't exist at __init__ -> crash. (BF16 has `.weight` -> why the
  user's BF16 run worked on this same image.)
- WORKAROUND (no patch): code checks `os.environ["DISABLE_ESIMD_FUSED_INPUT"] != "1"`. Set
  **`DISABLE_ESIMD_FUSED_INPUT=1`** -> fused path skipped -> model loads (lose that micro-opt only).
- BUG TO REPORT (contribution): fused path should read shape from `output_size_per_partition`/config,
  not `.weight.shape`, or be guarded for quantized layers. File against intel/llm-scaler.
- Architectural note: Qwen3.6 DeltaNet = only ~1/4 layers (full-attn) carry growing KV; the linear-attn
  layers use a small fixed recurrent state. So KV footprint at short ctx is much smaller than a normal
  27B -> 27B FP8 (29 GB) at 2K ctx may actually fit the single 32 GB card. Retrying with the env set.

### 2026-06-17 — [FAIL] Official Qwen FP8 on llm-scaler = dead end (2nd, deeper bug)
- With DISABLE_ESIMD_FUSED_INPUT=1, passed bug #1, then crashed DEEPER at
  `fp8.py:565 process_weights_after_loading: layer.weight, layer.weight_scale_inv` ->
  `AttributeError: MergedColumnParallelLinear has no attribute 'weight'`.
- => The official `Qwen/Qwen3.6-27B-FP8` (block-FP8 + weight_scale_inv) is incompatible with llm-scaler's
  ipex/fp8 weight-loading for the DeltaNet merged projections. TWO distinct bugs in this path. Even if
  patched, 29 GB weights vs 30.3 GB usable => no KV room on one card. Abandoning official-FP8-on-llm-scaler.
- Cleanup: removed b8.3 image (both b8.3/b8.3.1 share these bugs; b8.3.1 kept). See docs/CLEANUP.md.
- PIVOT options (single card now): (A) prove vLLM-XPU 8-bit pipeline on a SMALL standard model now +
  do Qwen3.6 8-bit on card #2 [low-risk, recommended]; (B) sym_int4 4-bit Qwen3.6 via BF16 online quant
  (54 GB dl; different int4 kernel, fits VRAM; ~50/50 vs DeltaNet immaturity) [gets real model running, 4-bit];
  (C) upstream vLLM + Quark W8A8 INT8 = user's proven 99 t/s lineage (8-bit, more setup, tight VRAM).

### 2026-06-17 — [OK] How to activate the B70 INT8/FP8 fast paths (llm-scaler b8.3.1)
- Registered quant methods (vllm QUANTIZATION_METHODS): fp8, sym_int4, quark, compressed-tensors, awq,
  gptq, gptq_marlin, ipex, inc, auto-round, modelopt, rtn, torchao, bitsandbytes, gguf, ... (full list logged).
- Intel multi-arc kernel present: `vllm_int4_for_multi_arc.so` (INT4). FP8 has ESIMD GEMV kernels
  (`esimd_gemv_fp8_pert[_fused2/3]`) via `XPUFp8LinearMethod`.
- **INT8 fast path mechanism (from ipex_quant.py):** IPEX WoQ with `lowp_mode = ipex.quantization.
  WoqLowpMode.INT8`: "weight de-packed to INT8; float activation dynamically quantized (per-token) to INT8;
  compute dtype INT8 to leverage instructions" => real INT8 XMX GEMM. This underlies `sym_int4` (W4 storage,
  A8 INT8 compute = W4A8) and an INT8-weight variant.
- ACTIVATION cheat-sheet:
  - FP8 (8-bit, ESIMD/XMX): `--quantization fp8` (online) or load an FP8 checkpoint.
  - sym_int4 (W4A8, INT8-XMX compute, Intel-native kernel): `--quantization sym_int4`.
  - W8A8 INT8 (true 8-bit weights + INT8 compute): Quark (`--quantization quark`, user's 99 t/s path) or
    compressed-tensors W8A8 / IPEX INT8. Verify it lands on the INT8 XMX kernel, not a fallback.
- VERIFY engaged (not a silent fp16 fallback): grep serve logs for the chosen LinearMethod/kernel; high
  prefill t/s; xpu-smi XMX/EU utilization during prefill. Plan: Gemma 4 12B -> FP8 first, then W8A8 INT8.

### 2026-06-17 — llm-scaler b8.3.1 supports Gemma3, NOT Gemma4; upstream is where quants work
- Checked arch registry: gemma support = Gemma/Gemma2/Gemma3/Gemma3n/PaliGemma. **No gemma4** (image
  predates Gemma4's 2026-06-03 release). Serving the downloaded Gemma4-12B on llm-scaler would fail.
- Community data point (user): Gemma 4 12B INT4 AutoRound W4A16, 4x B70, **upstream `vllm 0.20.2rc1.dev2`**,
  TP4, compilation (piecewise cudagraph, NOT enforce-eager), prefix-caching on, `--limit-mm-per-prompt
  {image:4}` (Gemma4 is multimodal). 119 prompt/256 gen tok, c8.
- PATTERN: ALL user working QUANTIZED runs use UPSTREAM vLLM-XPU 0.20.x (Gemma4 AutoRound, Qwen W8A8 Quark,
  Qwen W4A16). llm-scaler 0.14.x = where DeltaNet/FP8 bugs hit. Upstream supports AutoRound + Quark (the
  proven quant formats) AND gemma4. -> strong case to pivot to the upstream image for the fast-path work.
- DECISION PENDING: (A) pivot to upstream vLLM-XPU image (use downloaded Gemma4, AutoRound/Quark, aligns
  with all proven data points) vs (B) verify fast path on Gemma3-12B now on current llm-scaler image.

### 2026-06-17 — Pivot to upstream vLLM-XPU (build from source) for Gemma 4
- Decision (user): go upstream for Gemma 4. No recent pre-built `intel/vllm:*-xpu` tag with gemma4 found
  (`0.17.0-xpu` is pre-gemma4). Intel's official "Run Gemma 4 on Arc" blog = BUILD from source:
  `git checkout 3ca6ca2; docker build -f docker/Dockerfile.xpu -t vllm-xpu-env --shm-size=4g .`
  (commit 3ca6ca210 = "xpu docker: pin oneAPI to 2025.3", #41380). Build running (30-60 min) on the box.
- Gemma 4 notes: multimodal (image), 2 attn variants (sliding + full). Intel recipe:
  `--attention-backend TRITON_ATTN --enforce-eager`. Triton attn works OOB on Xe; SYCL-TLA FA optional.
- Plan once built: serve Gemma 4 12B online FP8 (8-bit fast path) -> verify XMX engaged; then BF16 baseline
  + self W8A8 INT8 comparison. Single-card upstream should avoid the #41663 TP2 GP-faults (no inter-card comm).
- Caveat: upstream B70 support is less validated than llm-scaler; bring stability env only if needed.
  Gemma 4 12B BF16 weights already on SSD (24 GB). llm-scaler b8.3.1 retained for Qwen/standard work.

### 2026-06-17 — [WARN] Gemma 4 12B FP8 on upstream vLLM-XPU: fast path VERIFIED, e2e blocked
- Built `vllm-xpu-env` from source (commit 3ca6ca2, 10m51s); +derived `:tf` (transformers 5.5.3->5.12.1)
  to recognize gemma4. Server starts HEALTHY. Key wins:
  - **FP8 fast path CONFIRMED:** log shows `Selected XPUFP8ScaledMMLinearKernel for Fp8OnlineLinearMethod`
    (XPU FP8 scaled-MM / XMX kernel, not an fp16 fallback). Activation = `--quantization fp8`.
  - Fits great: model 12.16 GiB + KV 14.35 GiB -> 39,184 KV tokens, 4.78x concurrency @ 8k ctx.
- BUT generation 500s: `RuntimeError: shape '[-1, 8, 256]' invalid for input of size 17408` in
  `vllm/model_executor/models/transformers/base.py:99 vllm_flash_attention_forward`. Prefill ok, first
  DECODE dies -> engine dead. Same on TRITON_ATTN and FLASH_ATTN backends.
- ROOT CAUSE: vLLM @3ca6ca2 has NO native gemma4 model; it uses the generic **Transformers-integration**
  path (`TransformersMultiModalForCausalLM`), whose attention reshape is wrong for Gemma4 head config
  (head_dim 256, GQA). Upstream vLLM bug; not config-fixable. (Intel's "OOB" blog likely used a newer
  vLLM with native gemma4, or the text-only variant.)
- Minor XPU gaps logged (contribution notes): "RMSNorm+quant fusion" and "Activation+quant fusion" not yet
  supported on XPU (disabled). Also tokenizer has no chat template (download missed *.jinja) -> used
  /v1/completions with manual gemma turn-tokens for benching.
- NET: fast-path ACTIVATION + verification skill = achieved (kernel selection visible). End-to-end NUMBERS
  need a natively-supported model. Gemma4 e2e -> revisit when vLLM ships native gemma4 (or newer commit).

### 2026-06-17 — Root fix for Gemma4: rebuild at user's proven commit c51df4300 (native gemma4)
- User pointed out their working data point = vllm `0.20.2rc1.dev2+gc51df4300` (commit **c51df4300**), newer
  than Intel-blog `3ca6ca2` (0.20.1). Checked c51df4300: it HAS **native gemma4** — `gemma4.py` +
  `gemma4_mm.py`, registry maps `Gemma4ForCausalLM`->gemma4, `Gemma4ForConditionalGeneration`->gemma4_mm.
  3ca6ca2 lacked these (fell back to buggy Transformers-integration -> the attention reshape crash).
  Native gemma4 landed BETWEEN the two commits. Rebuilding vllm-xpu-env at c51df4300. transformers pin
  `>=4.56,!=5.5.0` -> build pulls a gemma4-capable transformers automatically (likely no :tf patch needed).
- **Unraid vs Ubuntu (user Q):** NOT an Unraid issue. The Gemma4 crash is a Python tensor-reshape bug inside
  vLLM model code -> runs identically in the container on any host (container userspace is Ubuntu-based
  regardless). Host only provides kernel/xe-driver/dev-dri, which work fine (single-card inference proven on
  Unraid: 7B 90 t/s, Gemma4 loaded, FP8 kernel selected). Ubuntu MAY matter LATER for dual-card TP stability
  (#41663 BOM-sensitivity: Intel validates Ubuntu 25.04/kernel 6.14), but not for single-card. No OS move needed now.

### 2026-06-17 — DEFINITIVE: gemma4_unified breaks vLLM-XPU generic fallback (not version/Unraid)
- Rebuilt at user's proven commit c51df4300 (vllm 0.20.2rc1.dev2) + transformers 5.12.1. Server HEALTHY,
  FP8 fast path engaged (XPUFP8ScaledMMLinearKernel), 12.16 GiB + 12.5 GiB KV. BUT decode still 500s with
  the SAME `shape '[-1,8,256]' invalid for 17408` in transformers/base.py:99 -> so NOT a vLLM-version fix.
- ROOT CAUSE (config.json): `google/gemma-4-12B-it` is **`gemma4_unified`** (text+vision+audio). Text tower
  has MIXED head dims: `head_dim=256` (sliding layers) + **`global_head_dim=512`** (global layers), 16 heads,
  8 KV. vLLM resolves it to **TransformersMultiModalForCausalLM** (no native impl for the *unified* arch),
  whose generic `vllm_flash_attention_forward` hard-reshapes `[-1,8,256]` -> the 512-dim global layers break it.
- vLLM HAS native `gemma4.py`/`gemma4_mm.py` (Gemma4ForCausalLM / Gemma4ForConditionalGeneration) at c51df4300,
  but the *unified* checkpoint doesn't route to them. User's working run used `gemma4-12b-it-int4-autoround-intel`
  which DOES resolve native (the key difference). -> contribution: vLLM Transformers-integration mixed-head-dim
  attention reshape bug for gemma4_unified.
- BOTTOM LINE: B70 + vLLM-XPU + FP8 fast path = PROVEN working (kernel selected, loads, healthy). The block is
  this ONE multimodal checkpoint's arch routing. For e2e NUMBERS: use a natively-supported model OR a
  native-resolving Gemma4 checkpoint (the AutoRound one). Stack is sound; this is model-specific.

### 2026-06-17 — [KEY] No INT8 W8A8 kernel on Battlemage vLLM-XPU; FP8 is THE 8-bit path
- Agent verified in vLLM source (docs/literature/05_w8a8_recipe.md): XPU `scaled_mm` is **FP8-only**
  ("XPUFP8ScaledMM only support FP8 weight dtype"); INT8 W8A8 path is `is_cuda_alike()`-gated; `_POSSIBLE_
  INT8_KERNELS` has CPU/CUDA/ROCm but **no XPU**. Not on Intel's H1-2026 XPU roadmap (#37979 targets wNa16 +
  w8a16/FP8). => compressed-tensors/Quark **W8A8 INT8 silently dequant-falls-back to FP16 on B70** (no INT8
  TOPS). The "community Quark W8A8 on B70" 99 t/s was very likely that fallback.
- DECISION: **drop W8A8 INT8** for B70. **FP8 (XPUFP8ScaledMMLinearKernel) is the accurate 8-bit fast path.**
  W8A8 wouldn't beat FP8 anyway (decode bandwidth-bound; both ~1 byte/weight; INT8 TOPS only help compute-bound
  prefill/large-batch).
- THE real INT8-XMX path on B70 = **W4A8** (`XPUW4A8IntLinearKernel`: int4 weights + per-token int8 act, oneDNN).
  Revised quant matrix: **F16 | FP8 | sym_int4 (W4A16) | W4A8 (INT8 act)**. Test which int4 checkpoint/flag
  routes to XPUW4A8IntLinearKernel (grep load log). Verify-engaged greps in 05_w8a8_recipe.md.

### 2026-06-17 — Qwen3.6-27B DeltaNet+FP8 also broken on UPSTREAM vLLM-XPU (different bug)
- Tried Qwen3.6-27B-FP8 on vllm-xpu-env (c51df4300, which HAS native gdn_attention_core_xpu). Crash:
  `KeyError: <PlatformEnum.XPU: 4>` in `choose_scaled_mm_linear_kernel` (kernels/linear/__init__.py:351),
  via `GatedDeltaNetAttention.create_qkvz_proj` -> fp8.py:377 -> `init_fp8_linear_kernel`.
- ROOT CAUSE: the **GDN linear-attention FP8 projection uses the generic scaled_mm kernel-chooser, which has
  NO XPU entry** in `possible_kernels`. Regular attention FP8 correctly uses XPUFP8ScaledMMLinearKernel
  (why Qwen3-14B FP8 works); the GDN path doesn't. So Qwen3.6 DeltaNet FP8 is broken on XPU end-to-end
  (llm-scaler: ESIMD `in_proj_qkvz.weight` bug; upstream: scaled_mm chooser KeyError). Contribution target.
- CONCLUSION: single-card Qwen3.6-27B stays blocked (BF16 54GB won't fit; both 8-bit DeltaNet paths have XPU
  kernel gaps). Route = card #2 (BF16 across 2 cards) or fix the GDN FP8 XPU kernel selection. [task #6]

### 2026-06-18 — [BIG] vLLM v0.23.0 SOFTWARE-unblocks Qwen3.6-27B DeltaNet on B70 (VRAM is the only wall)
- Built vllm-xpu-env:v0230 (vllm 0.23.0, torch 2.11.0+xpu, 11m34s). Served Qwen3.6-27B-FP8:
  - Resolved `Qwen3_5ForConditionalGeneration`; **`Selected XPUFp8BlockScaledMMKernel`** (NEW block-scaled
    FP8 kernel — past the 0.20.2 `KeyError(XPU)`); **`Using Triton/FLA GDN prefill kernel`**. DeltaNet runs!
  - Model footprint **28.51 GiB**; then `ValueError: No available memory for the cache blocks` (KV = -0.22 GiB).
- CONCLUSION: **v0.23.0 fixes the Qwen3.6 DeltaNet+FP8 XPU kernel gaps.** Remaining block is PURELY VRAM
  (28.5 GiB leaves no KV on one 32 GB card). => "newest vLLM best?" = YES for Qwen3.6 (only version that
  loads it on XPU). Qwen3.6-27B is now a clean CARD #2 target (64 GB pooled) on a known-good stack [v0.23.0].
- Single-card paths to still try: Qwen3.6-27B **int4** (~15 GB, would fit w/ KV) on v0.23.0 GDN kernels —
  moonshot (needs int4 ckpt + XPU int4 is fragile). Otherwise card #2.
- NOTE: pin **vllm-xpu-env:v0230** as the new primary for DeltaNet models; keep :tf (0.20.2) for dense.

### 2026-06-18 — NIGHT CAMPAIGN SUMMARY (single B70, vLLM-XPU)
Headlines:
- **Qwen3-14B FP8 = the single-card sweet spot:** 35 t/s single-stream (62 ms TTFT w/ compile),
  **~558 t/s aggregate ceiling @ C64** (saturates there). near-lossless. Default --max-num-seqs 16 caps
  you at ~330 — raise it.
- **Qwen3.6-27B (DeltaNet) RUNS on ONE B70** via int4 AutoRound on vLLM 0.23.0 (7.89 t/s, coherent).
  Only known single-card path. FP8/8-bit needs card #2 (VRAM). Decode is GDN-kernel-limited (opt target).
- **vLLM 0.23.0 > 0.20.2** for B70: same throughput, ~7x better eager TTFT, and the only version with
  working GDN+FP8 XPU kernels. Pin v0230. (compile broke on it - torch 2.11; run eager.)
- F16 18.7 t/s (tight, dominated by FP8). W8A8 INT8 has NO XPU kernel (use FP8). Draft spec-decode is
  3.4x SLOWER (no XPU cudagraph). All documented in RESULTS.md / FINDINGS.md.
Infra: public repo github.com/hotschmoe/b70_ai_things (committing per experiment); docker.img 200GB;
images vllm-xpu-env:{tf=0.20.2, v0230=0.23.0}. Dual-card plan + scripts ready (DUALCARD.md, 43_serve_multi.sh).
Contribution targets logged: XPU GDN/DeltaNet decode-kernel speed; llama.cpp DeltaNet SYCL; gemma4_unified
vLLM fallback attention bug; W8A8 INT8 XPU kernel gap.

<!-- New entries below -->

### 2026-06-18 — [TIER-1a WIN] FP8 KV cache works on our W8A8 path: 2x context budget, no graft/patch
- Served Qwen3-14B-W8A8-INT8 from the committed `vllm-xpu-env:int8` image with `--kv-cache-dtype fp8_e4m3`
  [scripts/48]. Log: `Using fp8_e4m3 data type to store kv cache` + `Selected XPUInt8ScaledMMLinearKernel`
  -> **INT8 weights/acts + FP8 KV compose cleanly.**
- **GPU KV cache 138,880 tokens (was 71,040 with auto/fp16 KV) = ~1.96x; max concurrency 8.7x -> 17x.** Same
  10.6 GiB KV memory, double the tokens (FP8 KV = half the bytes/token). The long-context-coding win, free.
- => The engine config is settled: **W8A8 INT8 linear (ours) + FP8 KV cache** (+ optional W8A16 ignore-list
  for sensitive layers). Decode throughput ~NEUTRAL with FP8 KV: 512:128:1 = 20.4 t/s (vs 22.6 w/o; slight
  short-ctx descale cost), 512:128:32 agg = 476 (vs 465; slight gain). So FP8 KV = 2x context for ~free. [Tier-1a]

### 2026-06-18 — [RESEARCH] spec-decode for XPU: DFlash is the one that survives no-graph-capture; ngram PoC first
- Agent identified pflash/dflash: **dflash = z-lab DFlash** (arXiv:2602.06036), a **block-diffusion drafter**
  that emits a whole K-token block in ONE parallel forward pass (vs EAGLE's K sequential). **Already in vLLM**
  (PR #38300, vllm>=0.20.0): `vllm/v1/spec_decode/dflash.py` + `models/qwen3_dflash.py`. pflash = Luce-Org
  llama.cpp-fork prefill-compression -- NOT a vLLM spec-decode intrinsic; do not port.
- KEY INSIGHT for our 3.4x-slower XPU spec-decode result: it's a kernel-LAUNCH-COUNT problem. Autoregressive
  drafters (draft-model, EAGLE) do K sequential drafter forwards/step -> K launch-waves -> gated on XPU graph
  capture. **Single-pass drafters (MTP, Medusa, DFlash) do 1 forward/step -> launch multiplier ~1 -> their
  advantage SURVIVES the no-graph-capture penalty.** So DFlash is the spec-decode family worth targeting on XPU.
- MTP reality (confirmed): needs trained MTP heads. **Qwen3.6-27B HAS them** (`method:"mtp"`, ~75-79% accept
  @3) -> zero-effort single-pass spec-decode once it serves. **Qwen3-14B dense does NOT** -> a "14B MTP" PoC
  is really draft/ngram.
- PLAN: (1) cheapest diagnostic = **ngram spec-decode** (0 drafter forwards -> isolates whether verify-step
  eager overhead ALONE sinks spec-decode on graph-less XPU). If ngram net-positive -> single-pass drafters
  will likely win; if even ngram loses -> **wiring torch.xpu.XPUGraph is the true prerequisite** (lit/06 #3).
  (2) For 27B: try native MTP. (3) DFlash-on-XPU = M-L effort (non-causal attn + Triton-XPU kernels are the
  risk; the spec-decode plumbing already exists) -- the real target for dense models lacking MTP heads.

### 2026-06-18 — [RESEARCH] TurboQuant KV cache: DEFER; use FP8 KV cache instead (already on the XPU kernel)
- Agent researched TurboQuant (arXiv:2504.19874, ICLR 2026): a calibration-free KV-cache vector-quant
  (random/Hadamard rotation -> Gaussianized coords -> per-coordinate Lloyd-Max scalar quant; Q also rotated
  at attention time). KV-cache-ONLY -> orthogonal to our int8 LINEAR kernel; matmuls stay BF16.
- VERDICT: **not worth it for B70.** (1) No XPU path -- upstream is CUDA/Triton only; would need new SYCL
  kernels (rotation + sub-byte 3-bit pack/unpack + codebook dequant + paged flash-attn integration), ~L
  effort 3-5 wk, no oneDNN crutch; our per-token int8 quant op does NOT reuse (different granularity).
  (2) vLLM's OWN benchmark: TurboQuant **regresses throughput 10-68%** at accuracy-preserving bits, only wins
  under memory saturation; 3-bit (real memory savings) degrades coding accuracy (20-pt LiveCodeBench drop).
- **ANSWER to codex's KV rec + the KV-cache question: enable FP8 KV cache.** `--kv-cache-dtype fp8_e4m3` is
  **already supported in the XPU flash-attn kernel** (vllm-xpu-kernels, e5m2/e4m3 online descale, paged +
  chunked prefill) -> 2x KV capacity (~2x long-context budget on 32 GB), ~BF16 accuracy, no throughput
  penalty. Composes cleanly with our INT8 W8A8 linear path (int8 weights/acts + FP8 KV). Sources in agent
  report. [Tier-1 action: verify/bench FP8 KV on our W8A8 serve]

### 2026-06-18 — [MAKE-IT-FAST] fused per-token int8 quant kernel: decode 13 -> 22.6 t/s (1.7x)
- Wrote a fused SYCL `dynamic_per_token_int8_quant` op (csrc/xpu/sycl/, one work-group/row, sub-group absmax
  reduction, 2-pass quantize) to replace the eager `@torch.compile` `_ref` (the per-layer decode drag shared
  by W4A8 too). Agent-coded, parent-built (minimal profile, ccache-fast) + verified: q-match 100% (f16) /
  99.98% (bf16) vs ref, dequant maxerr ~0.04 (half a quant step), zp=0. [scripts/44 build + 45 serve]
- Re-bench (warm): **single-stream decode 13 -> 22.6 t/s (1.7x) = ~78% of FP8 (29)**; TTFT 142 ms; C32 decode
  agg 415 -> 465; **prefill unchanged at 6325 (still 1.6x FP8)**. So the ref quant WAS a major decode
  bottleneck. INT8 W8A8 now WINS prefill 1.6x AND nearly matches decode -> well-rounded throughput champion.
- Remaining decode gap (22.6 vs 29) = the M=1 int8 GEMM itself (oneDNN single-row) vs FP8 -- diminishing
  returns; vectorizing the quant K-loop or an M=1 GEMM path is a further (optional) optimization.

### 2026-06-18 — [VALIDATED] our INT8 W8A8 kernel beats FP8 ~1.6x in PREFILL (the native-systolic win is real)
- Prefill/large-batch bench, our W8A8 kernel vs FP8, same model/harness [scripts/46]:
  - 4096:8:1 prefill **6353 vs 3997 t/s = 1.59x**, TTFT 645 vs 1025 ms.
  - 2048:64:8 **14648 vs 9176 = 1.60x**; 4096:8:8 **10888 vs 6802 = 1.60x**.
  - Decode (batch-1): FP8 ~29 vs ours ~13 -> dragged by the eager `dynamic_per_token_int8_quant_ref`
    (per-layer reference quant). Fused SYCL quant = the fix (#11).
- CONFIRMS the literature/06 thesis: FP8 is conversion-based on Xe2 (FP16 datapath); INT8 is native
  s8s8s32 -> ~1.6x in compute-bound prefill. THIS is why the kernel matters. Decode is bandwidth-bound and
  quant-overhead-dragged. Caveats: 512:128:1 ours JIT-cold; 8192 hit --max-model-len; C32 confounded by
  --max-num-seqs (FP8 16 vs ours default). Logged RESULTS.md.

### 2026-06-18 — [MILESTONE] END-TO-END: Qwen3-14B W8A8 INT8 SERVES on the B70 via OUR kernel
- The exact `Qwen3-14B-W8A8-INT8` checkpoint that KeyError-crashed at the start of this campaign now LOADS,
  SELECTS our kernel, and GENERATES coherently on a single Arc Pro B70. "Make it work" = DONE end-to-end.
- Proof (scripts/45_patch_serve_int8.sh): grafts our int8-enabled `_xpu_C.so` over the image package, drops in
  `XPUInt8ScaledMMLinearKernel`, registers `_POSSIBLE_INT8_KERNELS[XPU]` + hardens the chooser (`.get()`), then
  serves with VLLM_LOGGING_LEVEL=DEBUG:
  - `INFO ... linear/__init__.py:620] Selected XPUInt8ScaledMMLinearKernel for CompressedTensorsW8A8Int8`
  - `Using scheme: CompressedTensorsW8A8Int8` for every layer (NO KeyError).
  - Model load 15.34 GiB + KV 10.85 GiB (71,040 tok, 8.67x conc @ 8k). `Application startup complete`, HEALTHY.
  - Smoke gen: "The capital of France is" -> " Paris. ..." (coherent, via the int8 oneDNN kernel at inference).
- INTEGRATION GOTCHAS (fixed): (1) patched `/workspace/vllm/vllm` not the runtime copy -> resolve via
  `import vllm` dirname (it IS the editable path). (2) shell `VLLMDIR=$(python -c import vllm)` got polluted by
  vllm's stdout INFO logs -> have the Python patcher copy the class file too (reliable BASE, no shell capture).
  (3) wrong model path (W8A8-INT vs W8A8-INT8).
- SIGNIFICANCE: first working INT8 W8A8 path on Intel Battlemage in vLLM (confirmed novel vs steveseguin +
  upstream RFCs #33214/#37979). The `.get()` hardening also fixes the GDN-FP8 KeyError family. Phase done:
  MAKE IT WORK. Next: make it RIGHT (asym/AZP, static schemes), make it FAST (fused per-token int8 quant;
  prefill/large-batch bench vs FP8 to cash the native-systolic INT8 win); then upstream PRs (kernels + vLLM).

### 2026-06-18 — [BIG WIN] int8_gemm_w8a8 XPU kernel: COMPILES, REGISTERS, and is NUMERICALLY CORRECT on B70
- Implemented our own INT8 W8A8 oneDNN op in vllm-xpu-kernels (head 11f42aa): new
  `csrc/xpu/onednn/int8_gemm_w8a8.h` + `joint_dtypes_t::s8_s8_{f16,bf16}` in onednn_ext.h + dispatch in
  onednn_matmul.cpp + `torch_bindings.cpp` registration. (Agent-drafted, parent-verified.) Symmetric-only
  phase 1 (per-token dynamic int8 acts x per-channel int8 weights -> f16/bf16).
- BUILD: editable `pip install -e .` rebuilt EVERYTHING (flash-attn/MoE/cutlass) -> 1-2h, wasteful. Fixed by
  building ONLY `_xpu_C` via the CMake toggles (`FA2/MOE/GDN/MQA/BASIC_KERNELS_ENABLED=OFF`,
  `XPU_SPECIFIC_KERNELS_ENABLED=ON`) -> minutes. [scripts/44_build_int8_kernel.sh]. Gotchas: stale CMakeCache
  (agent built at /src; mount repo at /src to match) and TWO concurrent build containers racing (kill all,
  wipe build/). Build RC=0, `_xpu_C.abi3.so` produced.
- VERIFY (GPU present, load .so directly; no-GPU load shows 0 gemm ops even for the stock .so -> invalid test):
  `int8_gemm_w8a8 -> OK` alongside int4_gemm_w4a8 / fp8_gemm_w8a16. **REACHABLE.**
- NUMERICAL: B=[K,N], M8/K64/N32, vs ref `(A.float()@B.float())*A_scale*B_scale`: **max_abs_err 2.4e-4**
  (ref|max| 0.80) = fp16 rounding. **CORRECT.** (B=[N,K] errors on shape, confirming the [K,N] contract.)
- STATUS: "make it work" DONE for the native op -- a working, correct INT8 W8A8 GEMM on Battlemage that
  did NOT exist anywhere (confirmed vs steveseguin repo + upstream RFCs). Remaining: vLLM Python patch
  (XPUInt8ScaledMMLinearKernel + _POSSIBLE_INT8_KERNELS[XPU] + .get() hardening) -> serve the W8A8 checkpoint
  that currently KeyError-crashes and show our kernel SELECTED (the end-to-end fast-path-activation proof).
  Patch ready in contrib/vllm_int8_xpu/. [task #10]

### 2026-06-18 — [CORRECTION] The "Quark W8A8 INT8 99 t/s" dissolves: cloned steveseguin/b70-optimization-lab
- User pointed us to `github.com/steveseguin/b70-optimization-lab` as "the quark int8 community user." Cloned +
  grepped it (`/mnt/vm_8tb/b70/b70-optimization-lab`). **No Quark, no INT8 W8A8 kernel, no `_POSSIBLE_INT8_KERNELS
  [XPU]` patch.** Their Qwen 35B paths are INT4-AutoRound (preferred) and FP8.
- The mislabel decoded: (1) this community says "**W8A8**" to mean **FP8 W8A8** ("static compressed-tensors W8A8
  FP8 safetensors"); their cited missing kernel is a native XPU **128x128 block-FP8 W8A8 GEMM** (FP8, not INT8).
  (2) The ~99 number = MiniMax-M27 INT4 MoE **99.79 TOTAL tok/s but ~33 OUTPUT** (one "99.77" is a gemma4 compile
  time). So "Quark W8A8 INT8 99.77 output t/s" conflated FP8-meaning-of-W8A8 + INT4-MoE-total-throughput + a wrong
  Quark/INT8 tag. Corrected docs/COMMUNITY_CONFIGS.md row #2 + added a repo deep-dive section.
- **NET: strengthens our finding -- nobody runs INT8 W8A8 on B70.** Our int8_gemm_w8a8 (docs/kernel/01) is first.
- BONUS: the repo is a goldmine for OTHER threads -- capture-safe XCCL all-reduce + XPU-graph (lit/06 #3), Qwen3.6
  FP8 fallback patches, a SECOND missing kernel (block-FP8 W8A8 GEMM = contribution #1b, same vllm-xpu-kernels repo),
  and llama.cpp SYCL Q4 fusion/all-reduce. Cloned on the box for reference.

### 2026-06-18 — [OK/WARN] YES the B70 accelerates quantization (SmoothQuant flies; GPTQ partial + a torch-xpu bug)
- User question: can the B70 speed up quantization? Ran llmcompressor INSIDE the XPU image (vllm-xpu-env:
  v0230, torch 2.11+xpu) with the model on `device_map=xpu`, GPTQ+SmoothQuant, 64 samples/seq512.
  [scripts/42_quant_on_xpu.sh DEVICE=xpu]
- [OK] `torch.xpu.is_available()=True`; device 0 = `Intel(R) Graphics [0xe223]` (the B70). llmcompressor
  auto-used **"Accelerator 0" (B70, 32.5 GB)** with NO code change. (The earlier "No accelerator available"
  was only the CPU-only python:3.11 image — NOT a real llmcompressor limitation.)
- [OK] **SmoothQuant calibration on XPU is the big win:** 41 blocks in ~40 s at 130-330 it/s. The CPU
  attempt of the same kind of pass was ~8-11 s/ITER per block (would be hours for the full model). This is
  the phase that made CPU calibration painful -> the GPU fixes it.
- [OK->WARN] **GPTQ ran on the B70** (Accelerator 0, ~6 s/module, real GPU matmuls) but with two issues:
  1. `torch.linalg.cholesky` -> "Aten Op fallback from XPU to CPU" (op not on XPU; perf hit, not fatal).
  2. CRASH in `_apply_activation_ordering` at `H[perm][:, perm]`:
     **`RuntimeError: level_zero backend failed with error: 20 (UR_RESULT_ERROR_DEVICE_LOST)`**.
- [CORRECTION 2026-06-18] **Cause of the DEVICE_LOST is CONFOUNDED, likely VRAM contention -- NOT confirmed
  a torch-xpu indexing bug.** Discovered afterward that the W4A8 server (`vllm_w4a8`, ~25 GiB VRAM) was
  STILL RUNNING during this experiment (scripts 41/42 only removed vllm_qwen3/vllm_w8a8, not vllm_w4a8). So
  the GPTQ run put a 28 GiB fp16 model on the SAME 32 GiB card on top of a ~25 GiB server -> ~53 GiB demand
  -> almost certainly OOM/device-lost, which happened to surface at a large-tensor gather. **Needs an
  ISOLATED re-test (no other GPU user) to attribute** (real torch-xpu gather bug vs pure OOM). [follow-up]
- VERDICT (still valid): **XPU-accelerated quant = YES for calibration/data-collection (SmoothQuant huge
  speedup, GPTQ matmuls run on the B70).** Full end-to-end GPTQ-on-XPU not yet proven (crashed, cause
  confounded). Re-run isolated with `actorder=False` for a clean time + a working checkpoint. [task #6]
- LESSON: serve scripts must remove ALL vllm serving containers (vllm_w4a8/w8a8/qwen3), not just one --
  a survivor holds port 18080 AND the GPU, giving false-HEALTHY and silent VRAM contention. Fixed in 36/41/42.

### 2026-06-18 — [OK] W4A8-INT runs on B70: INT8-XMX kernel ENGAGED (first time), but decode kernel unoptimized
- Self-quantized `Qwen3-14B-W4A8-INT` (int4 sym group-128 weights + per-token dynamic int8 acts, data-free
  RTN, ~3 min CPU) [scripts/43]. config.json verified: weights num_bits=4/group128/int, input_activations
  num_bits=8/int/dynamic/token -> matches `_is_dynamic_token_w4a8_int`.
- Served on vLLM 0.23.0 (`--dtype float16`). **`Using XPUW4A8IntLinearKernel for CompressedTensorsW4A8Int`**
  (oneDNN int4_gemm_w4a8) — the FIRST time we've lit the INT8 XMX datapath on the B70. Coherent output
  ("...is Paris..."). Load **9.3 GiB** (smallest yet), KV 15.0 GiB / 98,496 tok / 12.0x conc @ 8k.
- Bench (random 512/128): C1 16.6 t/s/stream (TTFT 155 ms), C8 16.0, C32 374 agg / 12.9 per-stream.
  **Single-stream decode 16.6 t/s = ~half FP8 (~31)** and only ~25% of the 9.3 GB bandwidth ceiling (~65)
  -> the int4_gemm_w4a8 DECODE kernel is unoptimized (same pattern as GDN). At C32 aggregate (374) slightly
  beats FP8 v0230 (333): the compute-bound regime where native-INT systolic helps.
- VERDICT: W4A8 is the lightest-VRAM, best-high-concurrency option AND the only int8-XMX path, but FP8 wins
  single-stream/interactive. Optimization target: faster int4_gemm_w4a8 decode kernel [literature/06 #2-ish].
- Quant matrix on B70 now empirically complete: F16 18.7 | FP8 35 (winner, interactive) | W8A8 HARD-FAIL |
  W4A8 16.6 (int8-XMX, lightest, high-conc). [tasks #5 done]

### 2026-06-18 — [KEY] Kernel-dev literature (agent) -> docs/literature/06_xpu_kernel_fastpaths.md + 3 CORRECTIONS
- Dedicated research agent wrote `docs/literature/06_xpu_kernel_fastpaths.md` (DPAS tile shapes, SYCL
  joint_matrix/ESIMD/oneDNN/Triton-XPU surfaces, vllm-xpu-kernels seams, ranked contribution targets).
- DPAS on Xe2: N=16, SystolicDepth=8. INT8 8x32x16->s32 (native), INT4 8x64x16->s32 (native),
  FP16/BF16 8x16x16->fp32. B must be VNNI-packed. INT8 = 2x FP16 throughput; INT4 = 2x INT8.
- **BIG: FP8 is NOT native systolic on Xe2** (conversion-based, rides FP16 path); **INT8 IS native.** So a
  true INT8 W8A8 kernel could beat FP8 in PREFILL / large-batch (up to ~2x systolic) — though batch-1 decode
  ties (both ~1 byte/wt, bandwidth-bound). This reframes "FP8 always dominates": only true for decode.
- **The W8A8 KeyError fix is small + double-duty:** oneDNN GPU matmul already does native s8xs8->s32 on
  Battlemage, so the missing piece is just an `int8_gemm_w8a8` op in vllm-xpu-kernels + `_POSSIBLE_INT8_
  KERNELS[XPU]=[XPUInt8ScaledMMLinearKernel]` (model on int4_gemm_w4a8.h). Also unblocks GDN-FP8 (same
  selector). => This is almost certainly what the community "Quark W8A8 99 t/s" person did. CONTRIBUTION #1.
- **THREE prior-assumption CORRECTIONS (agent [WELL-SOURCED], flagged to re-verify):**
  1. The "Xe2 warptile bug" is mis-stated — real bug is reorder/DPAS-sync corruption (llama.cpp #21893,
     workaround `GGML_SYCL_DISABLE_OPT=1`). (No warptile/mmq knob in the SYCL backend.)
  2. **llama.cpp DOES have Gated-DeltaNet now** (PR #16095, ~Nov 2025; GATED_DELTA_NET marked SYCL-supported).
     Our "build 9680 segfaults -> contribution target" is STALE -> retry Qwen3.6-27B on a NEWER llama.cpp
     SYCL build (could unblock the GPU path we thought was missing). [new task]
  3. **`torch.xpu.XPUGraph` exists** (PyTorch 2.11, PR #174046) — "no XPU graph capture" is now only true
     INSIDE vLLM (not wired up). => our spec-decode-negative / launch-overhead findings could flip if vLLM
     wires XPUGraph. CONTRIBUTION #3.
  Plus: Battlemage has NO GPU P2P (confirmed our SKU) -> cross-card collectives must host-stage (an ESIMD
  IPC-peer all-reduce is NOT viable); llama.cpp Vulkan DOES use coop-matrix on Xe2 ("matrix cores: none" was
  misdetection/pre-Xe2). [task #8 done]
- Top-3 kernels to write: (1) XPU INT8 W8A8 scaled-MM via oneDNN + registry fix [highest leverage, also
  fixes GDN KeyError]; (2) faster XPU GDN decode kernel (ESIMD/sycl-tla); (3) wire vLLM to torch.xpu.XPUGraph
  + host-staged capture-safe all-reduce.

### 2026-06-18 — [KEY] Quark W8A8 99 t/s "mystery" SOLVED by source: stock c51df4300 CANNOT run it (same KeyError)
- User asked to chase the community claim: Qwen3.6-35B-A3B **Quark W8A8 INT8 = 99.77 t/s** on 4x B70,
  vLLM **0.20.2rc1.dev2+gc51df4300** (== our `vllm-xpu-env:tf` image, EXACT commit), TTFT 76.5 ms.
- Inspected the quark dispatch in BOTH `:tf` (c51df4300) and `:v0230`: **`QuarkW8A8Int8.create_weights`
  calls `init_int8_linear_kernel`** — the SAME function compressed-tensors W8A8 uses. It indexes
  `_POSSIBLE_INT8_KERNELS[current_platform._enum]`.
- Read `_POSSIBLE_INT8_KERNELS` in `:tf` (kernels/linear/__init__.py:151): keys = **CPU, CUDA, ROCM only —
  NO XPU**. (Contrast: `_POSSIBLE_FP8_KERNELS[XPU]=[XPUFP8ScaledMMLinearKernel]` and `_POSSIBLE_KERNELS`
  (mixed-precision) `[XPU]=[XPUW4A8IntLinearKernel, XPUwNa16LinearKernel]` DO exist.)
- CONCLUSION (defensible): on **stock c51df4300**, BOTH Quark and compressed-tensors W8A8 INT8 hit
  `KeyError: PlatformEnum.XPU` at model load -> cannot run. So the community 99.77 t/s "Quark W8A8 INT8" was
  **NOT stock INT8 W8A8 on XPU**. It needed (a) a custom XPU int8 scaled-MM kernel + a registry patch
  (`_POSSIBLE_INT8_KERNELS[XPU]=[...]`), or (b) the Quark config actually resolved to FP8 / W4A8, or
  (c) misattribution. The image ships `vllm-xpu-kernels 0.1.7` (community used 0.1.9 for MTP) — no obvious
  int8 symbol exposed. "What enabled it" is NOT in stock vLLM or the shipped wheel.
- ACTIONABLE: writing that XPU INT8 scaled-MM kernel + the one-line registry entry is **contribution
  target #1** (a dedicated agent is researching the Battlemage kernel toolchain -> docs/literature/06).
  TODO: check vllm_xpu_kernels v0.1.9 for an int8 kernel; hunt the community patch. [task #7]
- SILVER LINING: the same grep proved **W4A8 IS reachable on XPU** (`_POSSIBLE_KERNELS[XPU]` has
  `XPUW4A8IntLinearKernel`) -> producing a W4A8-int checkpoint next. [task #5]

### 2026-06-18 — [OK->FAIL] EMPIRICAL: W8A8 INT8 hard-fails at load on B70 (KeyError XPU) — confirms paper finding
- Goal: stop guessing from source — actually RUN compressed-tensors W8A8 INT8 on the B70 and read the
  kernel selection. No Qwen3-14B W8A8 on HF (only 8B/32B; `shawnw3i/Qwen3-14B-INT8` is a real W8A8 but we
  made our own). Produced `Qwen3-14B-W8A8-INT8` (16 GB) from local BF16 via llm-compressor **data-free RTN**
  (`QuantizationModifier scheme=W8A8`, int8 per-channel weights + per-token DYNAMIC int8 activations, no
  SmoothQuant/no calibration -> `DataFreePipeline`, ~2 min CPU). [scripts/40_quantize_w8a8.sh DATAFREE=1]
- Served on `vllm-xpu-env:v0230` (vLLM 0.23.0). vLLM auto-detected `quantization=compressed-tensors`,
  picked scheme **`CompressedTensorsW8A8Int8`** for layer 0 qkv_proj, then **CRASHED at `create_weights`**:
  `init_int8_linear_kernel -> choose_scaled_mm_linear_kernel -> platform_kernels =
   possible_kernels[current_platform._enum]` => **`KeyError: <PlatformEnum.XPU: 4>`**
  (`kernels/linear/__init__.py:495`). Engine core init failed; server never came up. [scripts/41]
- VERDICT (now EMPIRICAL, not just source-read): **W8A8 INT8 does not even LOAD on Battlemage vLLM** —
  the INT8 scaled-MM kernel registry (`_POSSIBLE_INT8_KERNELS`) has **no XPU key**, so the chooser KeyErrors.
  Not a graceful "failed to find kernel", not a silent fp16 fallback — a hard crash at weight creation.
  Confirms docs/literature/05_w8a8_recipe.md.
- Cross-link: this is the **SAME `KeyError(XPU)` in the SAME `choose_scaled_mm_linear_kernel`** that killed
  the Qwen3.6 DeltaNet-FP8 projection on 06-17. Shared root cause: the generic scaled-MM chooser has no XPU
  entry for INT8 (or the GDN-FP8 path). Regular dense FP8 works because it routes to
  `XPUFP8ScaledMMLinearKernel` via a different path. => single upstream contribution: add an XPU branch (or
  a clear "unsupported on XPU" error) to `choose_scaled_mm_linear_kernel` / `possible_kernels`.
- Debunks the community "Quark W8A8 INT8 = 99 t/s on 4x B70" as an INT8-XMX run: vanilla compressed-tensors
  W8A8 KeyErrors at load on this exact chooser, so that number was NOT this path (vendor patch/fallback, or
  misattribution). Updated docs/COMMUNITY_CONFIGS.md row #2 accordingly.
- No bench possible (nothing served). FP8 remains THE accurate 8-bit path on B70. Restored FP8 baseline.
- NEXT: W4A8 (`XPUW4A8IntLinearKernel`, int4 weights + per-token int8 activations) = the ONLY upstream XPU
  path that lights the INT8 XMX datapath [task #5]. Plus experiment: XPU-accelerated quant calibration [#6].

### 2026-06-19 — [OK] Linux dev machine pickup: SSH workflow + runremote.sh (HANDOFF first task)
- Now driving the box from a Linux dev machine (not Windows). Passwordless SSH is via the DEFAULT key
  `~/.ssh/id_ed25519` (NOT the handoff's `b70_unraid_ed25519` — that key isn't on this machine).
- Added `~/.ssh/config` alias **`b70` -> `root@192.168.10.5`** (IdentityFile id_ed25519). All scripts/docs
  that say `ssh b70` now work unchanged.
- Wrote **`scripts/runremote.sh`** = bash port of `runremote.ps1` (strips CRLF/BOM, prepends `export KEY=VAL`
  for KEY=VALUE args, base64-transports the .sh, runs under `bash -s` on the box). Verified: env-var passing
  (incl. values with spaces), remote exec, GPU dri visible. This is the standard runner from now on.
- Made `scripts/38_specdecode_bench.sh` env-aware (NAME/MODEL/LABEL env with positional fallback) so
  runremote (env-only transport) can drive it.

### 2026-06-19 — [KEY] Qwen3.6-27B is a Qwen3_5 **DeltaNet VLM**, not a dense text model (changes the quant plan)
- 27B BF16 download COMPLETE (54 GB safetensors, 15 shards; `qwen27b_dl` exit 0). GPU free.
- Inspected config + the safetensors weight map (1199 tensors). The model is
  **`Qwen3_5ForConditionalGeneration`** (model_type `qwen3_5`): a **vision-language** model with
  a `model.visual.*` tower + a **hybrid linear-attention/full-attention** text model + an MTP head. DENSE
  (no MoE). 64 layers, hidden 5120, head_dim 256, vocab 248320.
  - `text_config.layer_types`: 48 `linear_attention` (DeltaNet) + 16 `full_attention` (`full_attention_interval=4`).
  - Module families: `model.language_model.layers.N.self_attn.{q,k,v,o}_proj` (16 full-attn layers, std Linear);
    `...mlp.{gate,up,down}_proj` (all 64, std Linear); `...linear_attn.{in_proj_a,in_proj_b,in_proj_qkv,in_proj_z,out_proj}`
    + conv1d/A_log/dt_bias/norm (48 DeltaNet layers); `model.visual.*` (vision tower); `mtp.*`; `lm_head`.
- **vLLM 0.23.0 (our :int8 image) registers `Qwen3_5ForConditionalGeneration` AND `Qwen3_5MTP`** in the model
  registry — arch is supported at the registry level. Open serveability risk = the DeltaNet (GDN/Triton)
  kernels on XPU (Triton-XPU), a card-#2-day question anyway (W8A8 ~30-36 GB > one 32 GB card).
- **CORRECTED the scripts/49 ignore-list.** Old default missed the vision tower -> text-only calibration
  would wreck vision. New IGNORE = `lm_head re:.*linear_attn.* re:.*visual.* re:.*mtp.*`
  (quantize only the std self_attn + MLP linears; keep DeltaNet + vision + MTP + head in BF16).

### 2026-06-19 — [OK->NEG] ngram spec-decode PoC on 14B W8A8 = NET-NEGATIVE on XPU (HANDOFF step #2)
- Served Qwen3-14B-W8A8-INT8 from `vllm-xpu-env:int8` (int8 linear + FP8 KV, eager) with and without
  `--speculative-config '{"method":"ngram","num_speculative_tokens":4,"prompt_lookup_max":3,"prompt_lookup_min":2}'`.
  Same harness (scripts/38, single-stream coherent decode). New script: `scripts/51_serve_int8_specdecode.sh`.
- Both serve HEALTHY; `Selected XPUInt8ScaledMMLinearKernel`; FP8 KV gives 138,880 tokens (16.95x conc @ 8k).
- **Baseline (no spec): 23.33 t/s** mean decode. **ngram spec: 21.51 t/s** mean decode -> **~8% SLOWER.**
  Acceptance: 48 drafts x4 = 192 draft tokens, **40 accepted (20.8%)**; per-pos 20/8/6/6 (decays, expected).
- VERDICT: spec-decode overhead (draft proposal + multi-token verify + rejection sampling) in **eager mode**
  exceeds the benefit at ~21% acceptance. Confirms the HANDOFF hypothesis: spec-decode is launch-overhead-bound
  on XPU; **graph capture is the prerequisite** before any drafter (ngram/DFlash/MTP) can win.

### 2026-06-19 — [KEY] XPU graph capture is ALREADY WIRED in vLLM 0.23.0 — blocked only by a missing fake kernel
- The serve log emits: `XPU Graph is disabled by environment variable, please set VLLM_XPU_ENABLE_XPU_GRAPH=1`.
  => Contrary to the HANDOFF ("wire torch.xpu.XPUGraph into the vLLM XPU runner", M-L effort), the runner
  support **already exists** in 0.23.0; it's just OFF by default.
- Tried `VLLM_XPU_ENABLE_XPU_GRAPH=1` (and dropped `--enforce-eager`, required for capture). Engine init
  **FAILED at graph capture** with a precise, fixable root cause:
    `torch._subclasses.fake_tensor.UnsupportedOperatorException: _xpu_C.dynamic_per_token_int8_quant.default`
    `torch._dynamo.exc.Unsupported: Operator does not support running with fake tensors`
  vLLM's XPU graph path runs `aot_compile_fullgraph` (torch.compile/dynamo), which traces with **fake
  tensors**. Our custom SYCL ops (`_xpu_C.dynamic_per_token_int8_quant`, and likely `_xpu_C.int8_gemm_w8a8`)
  have **no FakeTensor/meta registration**, so dynamo can't infer output shapes -> capture aborts.
- **This reframes the #1 strategic item from "wire XPUGraph (M-L)" to "register fake/meta kernels for 2
  custom ops (S, Python-only via `torch.library.register_fake`, no .so rebuild)."** A clean upstream
  contribution that should unblock graph capture -> potentially speeds plain decode AND could flip
  spec-decode positive. NEXT: add the fake registrations to the vLLM int8 patch and retest GRAPH=1.
- Op contracts (for the fake impls):
  `dynamic_per_token_int8_quant(x[M,K], sym=True, bits=8) -> (x_q[M,K] int8, x_s[M,1] f32, x_zp[M,1])`;
  `int8_gemm_w8a8(A[M,K] i8, A_s[M,1] f32, A_zp, B[K,N] i8, B_s[1,N] f32, azp_adj, bias[N]|None, out_dtype) -> [M,N]`.

### 2026-06-19 — [WIN] XPU PIECEWISE graph capture works on our int8 W8A8 path -> +16.7% decode (fakes unblocked it)
- Followed up the "XPUGraph already wired" finding. Added `torch.library.register_fake` meta kernels for our
  2 custom SYCL ops (in contrib_int8/xpu_int8.py: `_xpu_C.dynamic_per_token_int8_quant` ->
  (int8[M,K], scale[M,1], zp int32[M,1]); `_xpu_C.int8_gemm_w8a8` -> out_dtype[M,N]). Baked
  `vllm-xpu-env:int8g` (= :int8 + fakes; scripts/52). **Gotcha re-hit:** `import vllm` resolves to the
  editable `/workspace/vllm/vllm` in the bake but the SERVE process loads from site-packages -> the bake now
  writes the kernel to EVERY vllm copy (find under /workspace + /opt/venv).
- With fakes registered, `VLLM_XPU_ENABLE_XPU_GRAPH=1` (no --enforce-eager): dynamo traced THROUGH both
  custom ops, AOT-compiled (`saved AOT compiled function`), and proceeded to real SYCL Graph capture. So the
  fake registration **fully unblocked the vLLM/dynamo layer** — the original blocker is SOLVED.
- Two further blockers surfaced, both NON-fundamental:
  1. `cannot allocate memory for thread-local data: ABORT` at graph ~48/51 = thread/PID ceiling during
     capture. Fixed with `--pids-limit=-1 --ulimit nofile=1048576 --ulimit nproc=63556` + `OMP_NUM_THREADS=8`.
  2. FULL capture then hit: **`RuntimeError: The sycl_ext_oneapi_work_group_scratch_memory feature is not yet
     available for use with the SYCL Graph extension.`** = an Intel SYCL-Graph-extension maturity limit, hit by
     the **FlashAttention-v2 XPU** kernel (uses work-group scratch / SLM). NOT our op, NOT vLLM config.
- **PIECEWISE mode wins:** `cudagraph_mode=PIECEWISE` splits attention OUT (eager) and captures only the
  linear/MLP pieces. Captured all 12 graphs in 4 s (4.21 GiB), HEALTHY, coherent output. => our **int8 oneDNN
  GEMM IS SYCL-Graph-capture-safe**; only the flash-attn kernel isn't.
- **BENCH (single-stream coherent decode, same harness):**
    eager baseline = **23.33 t/s** ; PIECEWISE graph capture = **27.23 t/s** => **+16.7% decode**, TTFT ~55 ms.
- **Verdict / config recommendation:** add **PIECEWISE XPU graph capture** to the settled serving config
  (image `:int8g` + `VLLM_XPU_ENABLE_XPU_GRAPH=1` + `cudagraph_mode=PIECEWISE` + raised pid/thread ulimits).
  Free ~17% decode, no accuracy change. FULL capture stays blocked until Intel's SYCL Graph ext supports
  work_group_scratch_memory (or a non-scratch attention backend exists). This RESOLVES the handoff's
  "highest strategic leverage" item: prereq was NOT wiring (done) and NOT our ops (fixed) — it's the flash-attn
  kernel's scratch memory under SYCL Graph. Contribution: the 2 register_fake impls (upstream-worthy).

### 2026-06-19 — [KEY] Full 2x2 matrix: spec-decode stays NEGATIVE even with graph capture (needs FULL capture)
- Completed the grid on 14B W8A8 (single-stream coherent decode, same harness), image :int8g:
  | config | decode t/s | vs eager baseline |
  | eager, no spec                 | 23.33 | -      |
  | eager + ngram spec             | 21.51 | -7.8%  |
  | PIECEWISE graph, no spec       | 27.23 | +16.7% |  <- WINNER
  | PIECEWISE graph + ngram spec   | 25.28 | +8.4%  |
- ngram acceptance ~16% (39/244 draft tokens; per-pos 17/8/7/7). Spec-decode COSTS ~7% in BOTH eager
  (23.33->21.51) AND graph (27.23->25.28).
- WHY spec-decode is still negative WITH graph: in PIECEWISE mode **attention runs eager** (it's the
  work_group_scratch_memory kernel that SYCL Graph can't capture). Spec-decode verifies N+1 tokens/step, each
  needing an attention forward -> it still pays full eager attention launch overhead x(N+1), and the captured
  linear/MLP speedup can't offset that at only ~16% acceptance.
- **Refined verdict (updates HANDOFF):** "graph capture is the prerequisite for spec-decode" is only HALF
  right. Graph capture is a big standalone decode win (+16.7%), but ngram spec-decode needs **FULL** graph
  capture (attention included) to flip positive — and FULL capture is blocked by the Intel SYCL Graph ext
  (work_group_scratch_memory, via flash-attn). So on B70 today: **ship PIECEWISE graph capture (no spec).**
  Spec-decode (ngram/MTP/DFlash) stays parked until either FULL XPU graph capture lands or a
  no-scratch XPU attention kernel exists.

### 2026-06-19 — [OK] 27B W8A8 quant pipeline VALIDATED (data-free RTN) — text-only decoder, 33 GB, correct scheme
- Ran the corrected scripts/49 with DATAFREE=1 DEVICE=cpu (after fixing two bugs: DATAFREE env wasn't
  forwarded into the container; GPTQModifier `actorder=False` is rejected by current llmcompressor -> `None`).
- The Qwen3_5 VLM **loads via `AutoModelForCausalLM`** with no fallback needed — and importantly it loads the
  **TEXT-ONLY decoder** (`model.layers.*`): the vision tower and MTP head are NOT instantiated by the CausalLM
  auto-class. So `re:.*visual.*` / `re:.*mtp.*` ignore patterns matched 0 modules (harmless no-ops); the real
  ignores were the 48 DeltaNet `linear_attn` layers + `lm_head` (all kept BF16). Exactly **256 modules
  quantized** = 192 MLP (64x3) + 64 full-attn (16x4) std linears. Clean.
- Output `models/Qwen3.6-27B-W8A8-INT8-RTNtest` = **33 GB**, `quantization_config`: format=int-quantized,
  group_0 targets=Linear, weights 8-bit int, input_activations 8-bit **dynamic** -> the EXACT W8A8 dynamic-sym
  scheme our XPUInt8ScaledMMLinearKernel serves. ~40 s quant + ~3.5 min shard write.
- **33 GB > one 32 GB card** (weights alone, before KV/activations) -> confirms 27B W8A8 needs **card #2**.
  Implication: this is a text-only coding-server derivative (no vision, no MTP -> no MTP spec-decode path).
- Pipeline now DE-RISKED. Remaining unknown is XPU serveability of the DeltaNet (Qwen3_5) text model — vLLM
  registers the arch + ships `gdn_attention_core_xpu`, but it can't be tested until card #2 (33 GB won't fit
  one card). The RTN checkpoint is a sufficient artifact to test serveability the moment card #2 lands.
- DECISION PENDING (user): run the long GPTQ+SmoothQuant quality pass NOW (background, ready for card-#2 day)
  vs. DEFER until card #2 confirms the model serves on XPU (avoid hours of compute on an unconfirmed path).

### 2026-06-19 — [WIN] 27B W8A8 GPU-accelerated GPTQ VALIDATED on the B70 (then deferred); SmoothQuant breaks on hybrid Qwen3_5
- User asked to actually run W8A8 quant ON the B70 GPU (not CPU). Ran scripts/49 DEVICE=xpu METHOD=gptq.
- **SmoothQuant FAILS on the hybrid Qwen3_5 27B** (BEFORE any GPU work): its mapping resolver needs exactly
  one smooth-layer per balance group, but only 16/64 layers carry self_attn q/k/v (the other 48 are DeltaNet
  `linear_attn`) -> `ValueError: must match a single smooth layer ... got [all 64 input_layernorm]`. Fix:
  added **`SMOOTHQUANT=0`** env to scripts/49 -> GPTQ-only (still strong: GPTQ weight calib + dynamic
  per-token int8 acts; SmoothQuant matters most for static/per-tensor acts, which we don't use). Gotcha:
  the python heredoc lives inside `bash -c '...'` -> an apostrophe in a comment ("SmoothQuant's") closed the
  quote = bash syntax error. Keep that heredoc apostrophe/single-quote free.
- **GPU-accelerated GPTQ calibration WORKS end-to-end on the B70** (a first — prior validated run was CPU +
  data-free RTN). Measured live mid-run: compute engine (CCS) ~76->100% busy, copy engine (BCS) ~78% busy
  (host<->device layer streaming), ~7.6 GiB VRAM resident; llmcompressor's own log: `Accelerator 0 | total
  memory: 32.5 Gb` = the B70. Container CPU only ~8.6/32 cores (NOT the ~3200% a CPU forward would peg) ->
  confirms the matmuls are on the GPU. PCIe 3.0 x16 (board-limited from the card's Gen5) makes the streaming
  a real tax -> a 2nd card (resident model in 64 GB) would cut that, but GPTQ is sequential so not a 2x.
- **Progressive slowdown:** per-module GPTQ time grew to ~5 min on the back-half layers (`GPTQ METRIC time
  299s`), and container RAM crept 24 -> 42 GiB (calibration-cache accumulation). Full-attn layers (q/k/v/o +
  MLP = 7 modules) are the slow ones; DeltaNet layers (MLP-only, 3 modules) are fast. ETA ballooned to ~5h.
- **DECISION (user): killed at layer 47/65 to free the GPU for the 14B eval campaign.** Rationale: the 33 GB
  27B W8A8 can't be SERVED on one card anyway (needs card #2 ~Jun 22), so finishing it tonight banks an
  unusable-until-next-week artifact while the GPU is the bottleneck for tonight's evals. The risky part
  (GPU-GPTQ on B70) is now PROVEN; re-run the full artifact later, lighter (SAMPLES=128 to dodge the slowdown).
- Survived a dev-box restart mid-run: the `docker run` container kept going under dockerd (the SSH `tee` died);
  monitor via `docker logs` (dockerd capture), not the frozen host logfile.

### 2026-06-19 — [BUILD] evals/ quant-quality harness scaffolded (+ docs/storage.md)
- New **`evals/`** dir to measure quant DEGRADATION of a fixed base model (small-delta vs a bf16 reference).
  Tiers: 0 divergence (ppl + top-1 token agreement + nll-gap, the deterministic canary), 1 code (EvalPlus
  HumanEval+/MBPP+), 2 reasoning (lm-eval gsm8k), 3 creative (headless-rendered HTML, "renders-clean" +
  pairwise vs bf16). Orchestrator runs on the dev box, hits vLLM over the LAN; box only serves. README has
  intent / DOs-DONOTs / pitfalls (noise floor, CIs, contamination, MC-insensitivity, chat-template drift,
  determinism via VLLM_BATCH_INVARIANT=1 + concurrency 1). Tool choices validated via `codex exec`.
- Added **docs/storage.md** (8TB SSD policy — README referenced it but it was missing).
- First campaign target: Qwen3-14B across {bf16, fp8, w8a8, w4a8} + create the MISSING ones (w4a16, w8a16)
  with our pipeline (14B is standard dense -> SmoothQuant should work, unlike the 27B). Goal: per-quant
  quality to steer kernel-optimization focus (hypothesis: W8A8-INT8 is the path to lean on B70 INT fastpaths).

### 2026-06-19 — [INFRA] eval session pickup: dev-box venv + the BF16-on-one-card wall
- Dev box (ms-r1) had a broken venv: Debian lacks `ensurepip`/python3.11-venv -> `python3 -m venv` makes a
  pip-less venv. Fix: `venv --without-pip` + get-pip.py bootstrap. Light deps (openai/pyyaml/numpy) for Tier 0.
- **BF16 14B won't serve on one B70 under the v1 engine:** ~29.6 GB weights on a 32 GB card; at util 0.90 the
  28.8 GB budget doesn't even cover weights -> `ValueError: No available memory for the cache blocks` (both
  :v0230 and :tf hit it; both ship the v1 EngineCore). BF16 "barely fit" only on the older v0 engine.
  Implication: the bf16 reference needs `--cpu-offload-gb` (host RAM) or we anchor on FP8 (near-lossless,
  fits). Ironing out the orchestrator on W8A8 first (fits easily, our flagship). Added a one-card path to
  Tier 0: dump per-token argmax/logprob to tier0_tokens.json, compare offline (`tier0_divergence.py compare`).

### 2026-06-19 — [RESULT] Quant-quality campaign: int8 ACTIVATION quant is the cost, not int8 weights (6-quant matrix)
- Ran the full evals/ harness over Qwen3-14B × 6 quants on one B70 (served one-at-a-time, vLLM 0.23.0,
  greedy/eager, concurrency 1, thinking-off). Tier 0 = ppl + top-1 token-agreement + nll-gap vs a
  CPU-scored bf16 anchor (scripts/55; tokenization verified identical to vLLM /tokenize, 0/10 misaligned).
  Tier 2 = self-contained gsm8k (n=150, paired). Full table + analysis in evals/results/SUMMARY.md.
  | quant | w/a | ppl | top1-agree vs bf16 | gsm8k | serves |
  | bf16  | 16/16   | 12.70 | (anchor) | —      | no (29.6GB) |
  | fp8   | 8/8     | 12.70 | 0.968    | 96.0%  | yes |
  | w8a16 | int8/16 | 12.76 | 0.981    | —      | NO KERNEL |
  | w8a8  | int8/int8 | 13.08 | 0.881  | 95.3%  | yes (ours) |
  | w4a16 | int4/16 | 13.55 | 0.841    | 94.7%  | yes (XPUwNa16) |
  | w4a8  | int4/int8 | 14.19 | 0.822  | 92.7%  | yes |
- **HEADLINE: the int8 ACTIVATION quant costs the fidelity, not the int8 weights.** W8A16 (int8 w, fp16 a)
  is near-lossless (0.981 agree, even > fp8's 0.968). Going W8A16→W8A8 (quantize acts to int8) drops
  agreement 0.981→0.881 (−10 pts) — but gsm8k barely moves (95.3 vs fp8 96.0): it flips low-confidence
  tokens, not answers. So Tier-0 agreement is MORE sensitive than gsm8k (the point of the canary).
- **Weight bits dominate:** W8A8 (int8 w) beats W4A16 (int4 w) on every metric despite W4A16's fp16 acts.
- **KERNEL-COVERAGE FINDING:** B70/vLLM-XPU serves fp8, W8A8-int8 (ours), W4A8, **W4A16 (int4 weight-only,
  XPUwNa16)** — but **NOT W8A16 (int8 weight-only)**: `XPUwNa16` only accepts uint4/uint4b8, and our int8
  GEMM needs int8 acts. W8A16 is the one gap; the eval says it'd be near-lossless, but it keeps fp16 acts
  so it does NOT light the INT8 systolic path (a fidelity/memory play, not speed). Priority stays: optimize
  **W8A8** (only int8-systolic path, ≈fp8 task quality). W4A8 only when memory-bound.
- Created the two missing formats with the pipeline (scripts/54 DATAFREE RTN, CPU, ~5 min each):
  Qwen3-14B-W4A16 (9.3 GB), Qwen3-14B-W8A16 (16 GB). 14B is standard dense -> SmoothQuant-free RTN is clean.
- Orchestrator de-warts: gsm8k thinking-off (bounded + more quant-sensitive), openai/gsm8k id (datasets>=5),
  line-buffered stdout, one-card tier0 dump+offline-compare, tier0_matrix anchor table. All committed.
- CAVEATS: all non-fp8 quants are RTN here (GPTQ would lift them); gsm8k n=150 (~±2.5%) so fine gsm8k gaps
  are noise — trust ppl/agreement; no formal bf16-vs-bf16 noise floor yet; 14B + thinking-off may not
  transfer to 27B. NEXT ideas: noise floor run, GPTQ-vs-RTN quality delta, Tier-1 code (EvalPlus) + Tier-3.

### 2026-06-19 — [RESULT] Calibration study: RTN vs GPTQ, and 128 vs 512 samples (closes the campaign)
- Created calibrated checkpoints with scripts/54 (GPU-GPTQ on the B70, ~6 s/module for the 14B — the 27B's
  ~5 min/module was pathological, NOT typical): Qwen3-14B-W8A8-gptq (SmoothQuant+GPTQ@128, ~30 min),
  Qwen3-14B-W4A16-gptq (GPTQ@128, weight-only -> no SmoothQuant), Qwen3-14B-W8A8-gptq512 (@512, ~99 min).
- **Calibration lift scales with quantization error:**
  | scheme | RTN agree | GPTQ@128 agree | RTN->GPTQ ppl | gsm8k |
  | W8A8  | 0.881 | 0.908 (+2.7) | 13.08->13.05 | 95.3->94.7 (noise) |
  | W4A16 | 0.841 | 0.883 (+4.2) | 13.55->13.34 | 94.7->96.7 (+2.0) |
  int8 weights (W8A8) already near-lossless -> small lift, and it's **SmoothQuant** (not GPTQ) doing it
  (sharpens the int8 ACTIVATION quant, the W8A8 bottleneck). int4 weights (W4A16) have real error -> bigger
  lift; GPTQ-W4A16 (0.883) reaches RTN-W8A8 (0.881) fidelity (good int4 calib ~= an activation bit).
- **128 vs 512 samples: NO difference, use 128.** W8A8 GPTQ@512 (ppl 13.15, agree 0.900, gsm8k 95.3%) is
  within noise of @128 (13.05/0.908/94.7%) — marginally worse, pure variance. And @512 took **~99 min vs
  ~30 min (~3x)**: I was wrong that more samples is ~free — the Cholesky inverse is sample-independent but
  the calibration FORWARDS and Hessian accumulation (Sxx^T) scale with samples. 128 is the default.
- Also benched single-stream PERF (perf_probe.py): fp8 decode 30 t/s / TTFT 82 ms / prefill 3531; w8a8 21.9 /
  121 / **5787 (1.64x fp8 prefill, INT8 systolic)**; w4a16 26.4 / 89 / 2939 (out-decodes w8a8 — int4 weights
  stream less — but worst prefill). Tier-3 creative run on all 3 (gallery.py side-by-side). All committed/pushed.

### 2026-06-19 — [MILESTONE] Qwen3.6-27B (Gated-DeltaNet) RUNS + evaluated on a SINGLE B70 (the #1 goal, re-confirmed)
- Question: does B70 have int4 fastpaths, and would W4A16 27B fit+run on one card? **int4: yes the XMX has
  int4/int8 systolic, BUT W4A16 (fp16 acts) does NOT use it — it dequants int4->fp16 and does fp16 GEMM, so
  W4A16 is a MEMORY play (fits + faster decode), not a compute-fastpath play.** The int systolic only fires
  when BOTH operands are int (W4A8/W8A8).
- **W4A16 27B FITS one card: 25 GB** (scripts/49 now SCHEME-parametrized; RTN/data-free, 69 s). vs W8A8 33 GB.
  The memory wall that blocked the 27B is broken with int4 weights (DeltaNet/lm_head kept BF16).
- **BUT our compressed-tensors W4A16 won't serve** — two layers of issues:
  1. config: AutoModelForCausalLM saved a text-only `Qwen3_5TextConfig`; vLLM wants the full `Qwen3_5Config`.
     Fixed by wrapping (model_type qwen3_5 + nested text_config + language_model_only) + copying the
     vision preprocessor_config.json (vLLM loads the mm processor even for the text-only ForCausalLM).
  2. **kernel: `XPUwNa16` requires input sizes that are multiples of 32, but the 27B has a 4304-dim** (from its
     gated attention: 24 heads x 256, attn_output_gate, + hybrid DeltaNet 16/48 heads x 128). The 14B never
     hit this (clean dims). Real kernel-coverage gap -> our W4A16 27B is blocked there.
- **WORKING PATH: AutoRound int4 (`quantization=inc`, Intel's format) serves the 27B on ONE B70.** Different
  kernel path (not XPUwNa16) so it dodges the 4304 issue. Model loads 17.6 GB, KV 7.5 GB.
- **Critical gotcha: our `:int8` image lacks the GDN kernel.** First token crashed with
  `AttributeError: _xpu_C object has no attribute 'gdn_attention'` — because :int8 was built minimal
  (GDN_ENABLED=OFF, per the HANDOFF). **Serve the 27B on `:v0230` (full upstream build) — it has gdn_attention.**
- **On :v0230 it GENERATES coherently** ("The ocean is a vast and mysterious place... the octopus"). So
  **DeltaNet-on-XPU is PROVEN** (updates HANDOFF "UNPROVEN"). Also needed: copy the base 27B chat_template into
  the AutoRound tokenizer_config (it lacked one -> chat endpoint 400'd; raw /completions worked without it).
- **27B eval (AutoRound int4, single B70) — the real headline:** gsm8k **100% (50/50)** (vs 14B ~95% — much
  stronger model), Tier-0 ppl **6.60** (~half the 14B's ~13). PERF (perf_probe): **decode 7.59 t/s, TTFT 305 ms,
  prefill 1369 t/s** (matches the old ~7.9 t/s). Decode is slow (big dense + DeltaNet, unoptimized int4 decode).

### 2026-06-19 — [NEG/GAP] Qwen3.6-35B-A3B int4 (MoE) does NOT serve on a single B70 — the MoE-on-XPU gap, confirmed
- Goal: real-world eval + speed of 27B AND 35B-A3B quants (the 14B was just harness/quant-delta verification).
- Surveyed HF: many Qwen3.6-35B-A3B int4 quants exist. Downloaded **Intel/Qwen3.6-35B-A3B-int4-mixed-AutoRound**
  (21.5 GB, `quant_method=auto-round`, **256 experts**, arch Qwen3_5MoeForConditionalGeneration). 21.5 GB on
  disk would naively fit one 32 GB card.
- **It OOMs at WEIGHT LOAD** (`UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY` inside DeviceMemoryProfiler, before any KV;
  retry at maxlen 2048/1 seq OOMs identically -> NOT a KV/activation spike). Root cause: **vLLM-XPU has no fused
  int4 MoE kernel**, so the 256 experts dequantize toward bf16 (~70 GB) -> exceeds 32 GB. This CONFIRMS the
  HANDOFF's "MoE int8 W8A8 kernel" gap, now for int4 AutoRound too.
- **Implication:** 35B-A3B on a single B70 needs either (a) a **fused int4 MoE XPU kernel** (the real kernel-dev
  target — also the "Quark 99 t/s" path, which was 4xB70), or (b) **multiple cards** (card #2+). int4 weights
  alone don't fit one card if the runtime can't keep them packed through the MoE compute.
- NET for the two REAL targets: **27B = works on 1 card (great quality, slow decode); 35B-A3B = blocked on 1 card
  (MoE int4 kernel gap).** Perf/quality of more 27B quant formats (FP8 doesn't fit @ 54GB... actually 27B FP8 is
  ~28GB, retest; our W4A16 needs the 32-pad fix) is the next axis. Another agent is reviewing the harness + a container.

### 2026-06-19 — [RESEARCH] W8A8 INT8 accuracy-recovery literature survey → docs/literature/07
- Spawned 5 parallel research agents (rotation methods, SmoothQuant family + weight-side, Intel B70 toolchain,
  bleeding-edge 2025-26 papers, DeltaNet/SSM angle) + 1 codebase-grounding agent. Full writeup:
  **`docs/literature/07_w8a8_int8_recovery.md`**. Key conclusions:
- **Confirmed our fast path = INT8 W8A8** (sym per-channel w × per-token dynamic int8 a, oneDNN s8s8s32 on XMX).
  **FP8 is a W8A8 scheme (8-bit w AND a), NOT a W8A16** — just float vs int. **Xe2/Battlemage has NO native FP8
  matrix unit** (oneDNN data-types table: fp8 emulated `.`, int8 native `+`; SYCL joint_matrix spec has no fp8;
  367 INT8 TOPS / no FP8 TOPS). So FP8 on B70 upconverts to bf16 for compute → memory-only, no speedup. **INT8
  W8A8 is the ONLY low-precision compute fast path here** — matches our prefill 1.64x>FP8.
- **Root cause confirmed by 4 papers AND our evals: it's int8 ACTIVATION quant that costs fidelity, not int8
  weights.** W8A16 0.981 agree (near-lossless) → W8A8 0.881 (−10 pts), but answers barely move (flips
  low-confidence tokens). Field considers W8A8 "solved" (~99% BF16) and moved to W4A4 — no new W8A8 algorithm to
  chase; leverage is applying the known recipe well. (BF16-or-Death 2411.02355; Qwen3 reasoning 2504.04823;
  char. 2508.16712; long-ctx 2505.20276.)
- **Ranked recovery for our path:** (1) **fix SmoothQuant selectively** — apply to the 16 full-attn layers +
  MLPs, skip DeltaNet linear_attn (we run SMOOTHQUANT=0 today because pairing breaks on the 16/64 split); NNCF
  per-node alpha is the model. (2) **OS+ (2304.09145)** per-channel *shifting* as the SmoothQuant alt. (3)
  **down_proj@W8A16** for early+late layers (GLU/SwiGLU spike site, "Super Weight"); automate via NNCF
  accuracy-aware layer reversion. (4) **RTN is enough for weights at W8** (RTN≈GPTQ≈AWQ; matches our "lift scales
  with quant error").
- **DeltaNet frontier: NO DeltaNet/Gated-DeltaNet INT8 quant paper exists — we're at the edge.** Transfer recipe
  = Mamba/SSM lit (**Quamba2 2503.22879**: offline sort+cluster recurrence input + per-state-group scales;
  **Q-Mamba**: decoupled state-dim/channel-dim scales). Linear/delta attn is INHERENTLY easier to quantize than
  softmax → **the interleaved full-attn layers, not DeltaNet, are where int8 bites hardest** (validates keeping
  linear_attn BF16). ⚠️ **vLLM #40252 gotcha:** Gated-DeltaNet combined tensor names are `in_proj_qkvz`/
  `in_proj_ba`; stale names → silent layer-zeroing → `!!!!!` garbage. Verify our 27B ignore-list (a4190ba).
- **Skip rotation at W8A8** (QuaRot int8 5.47→5.50, RTN==GPTQ; papers call it marginal at 8-bit) — and no
  Hadamard kernel exists for Intel SYCL/XMX anyway. **Skip QAT** (gap too small). **Eval upgrades:** EAR/dist-
  lossless (SLQ 2605.02404) + forward-only KL layer-sensitivity for hybrids (KL-Lens 2604.13440) to pick
  ignore-list layers by measurement.
- **Next experiments (queued):** selective SmoothQuant on full-attn layers; down_proj@W8A16 sweep; KL-sensitivity
  ranking for the ignore-list; add EAR/KL to the harness; verify in_proj_qkvz/in_proj_ba names.

### 2026-06-20 — [HARNESS] Tier-1 (EvalPlus code-exec) WIRED + Docker-sandboxed; validated on 14B-fp8
- **The last un-run tier is live.** Tier 1 = execution-graded code (HumanEval+/MBPP+ pass@1 via EvalPlus).
  It's the only tier that *runs model-generated code* to grade it, so it needs isolation — built it as a
  3-step split where only the dangerous step is jailed:
  1. **GENERATE** (host, safe) — our own OpenAI-client loop (not `evalplus.codegen`), so Tier 1 inherits the
     harness discipline: greedy/temp0, fixed seed, concurrency 1, and **`enable_thinking` off by default**
     (flag `--tier1-think`). Replicates EvalPlus's exact chat prompt so samples drop into its grader.
  2. **SANITIZE** (host, safe) — `evalplus.sanitize` (tree-sitter; no code runs).
  3. **EVALUATE** (Docker jail) — `evalplus.evaluate` executes the code vs +tests. Sandbox: **`--network none`,
     non-root `--user`, a per-run THROWAWAY cache copy, `--memory 8g --pids-limit 512`.**
- **Why our own generator, not `evalplus.codegen`:** codegen's OpenAI backend can't pass
  `chat_template_kwargs.enable_thinking` → would silently run **thinking-ON** and diverge from tiers 2/3.
- **Why the throwaway cache:** `evalplus.evaluate` *writes* a ground-truth `.pkl` into its cache dir (and
  can't re-download under `--network none`). So we copy `~/.cache/evalplus` into the run dir, mount THAT
  writable, and leave the real cache untouched — verified: post-run, real cache had no stray/root-owned files;
  the `.pkl` + trimmed dataset landed only in the staged copy. Files owned by us (non-root `--user` works).
- **`--limit` smoke support:** EvalPlus asserts FULL dataset coverage (`len(completion_id)==len(problems)`).
  For smokes we trim the *staged throwaway* dataset jsonl to the sampled task_ids so coverage matches; full
  runs stay unfiltered so the assertion still catches a silently-dropped problem.
- **VALIDATED on Qwen3-14B-fp8, HumanEval+ (all 164, thinking-off, greedy):
  pass@1 = 0.915 base / 0.890 plus.** Generate 22.6 min (~8.3 s/problem @ fp8), sandbox eval 39 s. Smoke
  (5 problems) was 1.0/1.0. Credible/leaderboard-plausible → pipeline produces real numbers.
- **Files:** `evals/sandbox/{Dockerfile,build.sh}` (→ `evalplus-sandbox:0.3.1`, pinned to host evalplus);
  rewrote `orchestrator/tier1_code.py`; `run_evals.py` gains `--tier1-dataset|--tier1-think|--tier1-image`;
  `report.py` already surfaces the `tier1 pass@1(+)` column; README §11 documents the sandbox. `--allow-code-exec`
  repurposed as the UNSANDBOXED host escape-hatch (Docker is the default, no longer a refuse-to-run gate).
- **Caveats / next:** (1) HumanEval is contamination-prone — a single anchor, not yet a quant-delta; lean on
  Tier 0 for the precise ranking. (2) Per-quant Tier-1 sweep (w8a8/w4a16/w4a8 vs fp8) is the obvious next run
  now that the GPU is free. (3) thinking-off is the *sensitive* setting for quant damage but suppresses
  absolute pass@1 vs real coding-server use — revisit if we want a thinking-on "feel" pass. (4) roadmap still
  wants LiveCodeBench (contamination-resistant) for the headline code claim.

### 2026-06-20 — [HARNESS] Tier-1 per-quant sweep — 14B {fp8,w8a8,w4a16,w4a8} HumanEval+ + fresh perf
- Ran the full 14B code-quality sweep one model at a time on the single B70 (driver: serve → wait healthy →
  chat-smoke → HumanEval+ 164 thinking-off greedy → `perf_probe`). **HumanEval+ pass@1 base / plus:**
  | quant | pass@1 base | pass@1 plus | decode t/s | TTFT ms | prefill t/s |
  |---|---|---|---|---|---|
  | fp8 (anchor) | 0.915 | 0.890 | 32.1 | 85 | 3525 |
  | w8a8 | 0.902 | 0.860 | 23.8 | 101 | 5780 |
  | w4a16 | 0.866 | 0.829 | 29.1 | 79 | 2921 |
  | w4a8 | 0.860 | 0.817 | 16.5 | 139 | 4403 |
- **Code is a sharper quant-delta than gsm8k.** Plus-test spread fp8→w4a8 = **0.890→0.817 (−7.3 pts)** vs
  gsm8k 0.960→0.927 (−3.3). Ordering **fp8 > w8a8 > w4a16 > w4a8** matches ppl/agreement exactly → the
  long-generation tier behaves as designed (per-token quant error compounds over a whole function).
- **Speed finding: w4a8 decodes SLOWEST (16.5 t/s) despite 9.3 GiB VRAM.** [RE-CORRECTED 2026-06-20
  per commit 0f4e7ee (serve-log verified): 9.3 GiB IS correct -- it is the VRAM-resident size; vLLM
  repacks the int4 weights to 4-bit ON LOAD. The 16 GB is DISK ONLY (unpacked I8). So fit is fine;
  decode is KERNEL-bound, not memory-bound. (My intermediate "9.3 was a mislabel" note was wrong --
  disregard it.)] int8-activation
  dynamic per-token quant overhead outweighs the int4-weight bandwidth win; w4a16 (int4 w / fp16 a) decodes
  29.1. So **w4a8 = pure memory play, NOT speed**; w4a16 is the better small-footprint pick on both quality
  (0.829 vs 0.817 plus) AND decode (29 vs 16). w8a8 keeps its prefill crown (5780 t/s, 1.64× fp8).
- **Mechanics validated at scale:** chat-smoke `OK` (thinking-off honored) on all 4 serves; sandbox Docker
  grading clean each time; serve scripts cross-evict cleanly (`:int8`↔`:v0230` image swaps). Driver in
  `/tmp/tier1_sweep.sh`, perf rows `/tmp/tier1_sweep_perf.jsonl`.
- **27B Qwen3.6 int4-AutoRound: IN PROGRESS** (serves on v0230, 17.6 GB weights + 6.9 GB KV, chat-smoke OK,
  generating at ~7.6 t/s → ~110 min for 164). Row to be appended. NOTE HumanEval near-saturates at 14B
  (~0.89) so the 27B "jump" may be muted on this bench specifically — the real 27B story is gsm8k 100% +
  whether decode speed is tolerable; that's the higher-density tradeoff we're measuring.

### 2026-06-20 — [HYGIENE/BUG] Archived redundant RTN quants; found the coding eval ran RTN not GPTQ
- **Cleanup:** moved 2 redundant RTN 14B quants to `/mnt/vm_8tb/b70/models/archive/`:
  `Qwen3-14B-W4A16` and `Qwen3-14B-W8A8-INT8` (both have winning SmoothQuant+GPTQ twins — W8A8
  0.881→0.908, W4A16 0.841→0.883; comparison already logged in evals/SUMMARY.md). KEPT: 27B RTN
  (`...-W8A8-INT8-RTNtest`, canonical since its GPTQ run was incomplete), 27B W4A16, 14B W8A16 / W4A8-INT
  (RTN-only, no twin), all base/external models. `mv` within the same FS → reversible.
- **BUG (sad days):** the Tier-1 HumanEval+ `w8a8` coding number was served from the **RTN** checkpoint,
  NOT SmoothQuant+GPTQ. Chain: models.yaml `w8a8` → served_model_id `qwen3-14b-w8a8` → scripts/51
  `MODEL=Qwen3-14B-W8A8-INT8` (RTN). Result `...__qwen3-14b-w8a8__w8a8/tier1_code.json` = RTN. The gptq
  W8A8 only ever ran Tier-0. **TODO: re-run Tier-1 (HumanEval+) on `qwen3-14b-w8a8-gptq`** — the published
  w8a8 code pass@1 is an underestimate (GPTQ is +2.7 agreement).
- **Fix (repoint to GPTQ winner):** scripts/{41,45,46,47,48,51} now point at `Qwen3-14B-W8A8-gptq` AND
  serve it under name `qwen3-14b-w8a8-gptq` (path AND served-name — repointing only the path would
  re-mislabel results). models.yaml `w8a8` served_model_id → `qwen3-14b-w8a8-gptq` (+ calibration field).
- **Root-cause fix:** scripts/{40,49,54} OUTNAME now method-tagged (`...-${SCHEME}-${rtn|gptq}`) so RTN and
  GPTQ outputs can NEVER collide / silently overwrite again (old default `...-${SCHEME}` overwrote across
  methods unless a custom OUTNAME was passed — that's how the dups + mix-up happened).
- **Policy guard:** added "ALWAYS verify the served model (RTN vs GPTQ)" to evals/README.md, the
  models.yaml header, and a new top-level `CLAUDE.md`. Verify via `/v1/models` + cross-check models.yaml.

### 2026-06-20 -- [RESULT] Qwen3.6-27B int4 HumanEval+ = 0.963/0.927 -- the higher-density jump (+ its speed cost)
- **27B int4-AutoRound, HumanEval+ (164, thinking-off, greedy): pass@1 0.963 base / 0.927 plus.** Served on
  `:v0230` (gdn_attention), chat-smoke OK, gen 90.4 min (the slow part), sandbox eval 39 s. Fresh perf:
  **decode 7.94 t/s, TTFT 283 ms, prefill 1376 t/s** (confirms the prior 7.6).
- **The tradeoff the campaign was built to show:** 27B int4 beats the best 14B (fp8 0.915/0.890) by
  **+4.8 base / +3.7 plus** at **~4x slower decode** (7.9 vs 32 t/s). Going to the bigger model on one B70
  buys ~+4 pts pass@1 for 4x the per-token latency -- fine for async/agentic, painful for interactive.
- **HumanEval understates it.** The bench near-saturates (~0.89 at 14B), so 0.927 plus is a floor on the
  27B's real edge -- its gsm8k is a clean 100% (50/50) vs the 14B's ~95%. Harder/contamination-resistant code
  (LiveCodeBench, roadmap) would widen the gap. HumanEval = directional; Tier 0 = the precise rank.
- Next (auto-queued, GPU now free): **w8a8-gptq + w4a16-gptq** code numbers -- GPTQ-vs-RTN calibration delta,
  each served identically to its RTN twin (note: the served `w8a8` row in the 14B table is the **RTN**
  checkpoint, correctly labeled; the GPTQ run uses served-id `qwen3-14b-w8a8-gptq`). Expect the int4 lift larger.

### 2026-06-20 -- [OK] Cataloged community W8A8/W4A8 INT8 quants for newer models -> docs/COMMUNITY_QUANTS.md
- **Goal:** which off-the-shelf HF checkpoints our INT8 paths could serve -- `XPUInt8ScaledMMLinearKernel`
  (W8A8, our custom oneDNN kernel, image `vllm-xpu-env:int8`) and `XPUW4A8IntLinearKernel` (W4A8: int4 wt +
  per-token int8 act, oneDNN) -- for newer families (~Oct 2025 -> mid 2026), bucketed by 1x/2x/4x B70 fit.
  4-way HF sweep; every repo verified via HF API / `config.json` (activation dtype read directly: INT8 =
  `I8`/`"type":"int"`, NOT FP8). Only INT8 activations are usable (Xe2 has no native FP8).
- **Try-first (2x B70):** the three nameistoken Quark `ptpc_int8` models -- Qwen3.6-27B (30 GB),
  Qwen3.6-35B-A3B (37 GB), Gemma-4-31B-it (33 GB). Recipe == our W8A8 kernel (W INT8 per-ch static + A INT8
  per-token dynamic); GSM8K ~0.00pp vs BF16. All need 2x (none clears one card's 30.3 GiB + KV).
- **1x picks today:** Qwen3.5-9B w8a8, gemma-4-E4B w8a8 (~8B), GLM-4.6V-Flash w8a8 (10B VL), phi-4 w8a8,
  Granite-4.0-h-tiny, Olmo-3-7B. Giant MoEs exceed 4x (GLM-4.6 363 GB, MiniMax-M2.5/2.7 ~230, Step-3.5 ~191,
  Qwen3-235B 236).
- **Traps recorded:** `amd/Kimi-K2.5-W4A8` is W4-INT4/**A-FP8** (naming trap); cpatonn `*-INT8-INT4` AWQ +
  ModelCloud + `*-Int8Mix` GPTQ are weight-only A16; `inference-optimization/Qwen3-Next-80B...w8a8` are EMPTY
  stub repos. **W4A8 bloat:** Avesed/lokeshe09 "W4A8" 27-31B repos are ~33-36 GB (==W8A8 size), stored
  unpacked -> NO fit win.
- **Gaps (no INT8-act anywhere):** Llama 4, DeepSeek, Kimi, Hunyuan, Nemotron-3/Nano-2, Cohere Command-A, Phi-5.
- **Verdict / next:** highest-value own-quant = real (packed) **W4A8-INT8 of Qwen3.6-27B + Gemma-4-31B** to
  drop them onto 1x B70 via `XPUW4A8IntLinearKernel` (win is fit, not decode speed). Then Nemotron-Nano-9B-v2
  w8a8 (1x, currently FP8/NVFP4-only). Reuse the scripts/{40,49,54} GPTQ pipeline.

### 2026-06-20 -- [WORKSTREAM] Opened w4a8/ -- single-card W4A8-INT8 (3 wins: packing, accuracy, kernel)
- **Trigger:** chasing packed W4A8-INT8 of Qwen3.6-27B + Gemma-4-31B for single-B70 users.
  Investigated our existing `Qwen3-14B-W4A8-INT` (scripts/43, data-free RTN).
- **[KEY -- AMENDED] 16 GB on DISK, but 9.3 GiB in VRAM (fit is fine).** safetensors header: weights
  are `dtype I8` at FULL shape (`gate_proj.weight [17408,5120]`) -> serialized `int-quantized`
  (unpacked int8), NOT `pack-quantized` -> 16 GB on disk. BUT vLLM repacks to 4-bit on load ->
  **9.3 GiB VRAM** (verified, commit 0f4e7ee). A 27B W4A8 = ~18 GiB VRAM -> FITS 1x (cf. 27B
  int4-AutoRound 17.6 GiB). Packing only cuts disk + load time (39s->~23s), NOT VRAM.
- **Decode is the SLOWEST of all quants** (16.5 t/s vs fp8 32.1, w4a16 29.1, w8a8 23.8): unoptimized
  `int4_gemm_w4a8` decode kernel + per-token int8 act-quant overhead -- it is KERNEL-bound, not
  memory-bound (VRAM is only 9.3 GiB). So packed W4A16-gptq strictly dominates w4a8 today on decode
  (29 vs 16.5), accuracy (0.848 vs 0.817), and disk (9.3 vs 16) at the SAME VRAM. w4a8's only
  possible niche = int8-XMX throughput under concurrency -- UNMEASURED; gate the workstream on it.
- **Plan (docs in `w4a8/README.md`): 3 wins, all required.** (1) PACKING [gate]: force pack-quantized
  export -> ~9 GB, verify XPUW4A8IntLinearKernel still loads it + measure resident GiB.
  (2) ACCURACY: replace RTN with **AutoRound** (chosen) -> int4 weights + int8 dyn act; target 14B
  plus 0.817 -> >=0.84 (beat w4a16 0.829). (3) KERNEL: optimize int4_gemm_w4a8 decode (backlog).
- **Scripts written (NOT run -- queued):** `w4a8/10_quant_autoround_w4a8.sh` (B70-accelerated,
  DEVICE=xpu) + `w4a8/11_test_packed_export.sh` (CPU packing probe). Per user: **wait for GPU free,
  use the B70 to accelerate quant** (retest the old "XPU calibration unreliable" caveat for AutoRound).

### 2026-06-20 -- [RESULT] GPTQ-vs-RTN on code (HumanEval+) + the full single-B70 leaderboard
- Ran Tier-1 on the GPTQ-calibrated 14B checkpoints, each served IDENTICALLY to its RTN twin (w8a8-gptq:
  int8 image + fp8 KV; w4a16-gptq: v0230 + default KV) so the only variable is calibration:
  | scheme | RTN base/plus | GPTQ base/plus | code lift | decode (RTN/GPTQ) |
  |---|---|---|---|---|
  | w8a8  | 0.902 / 0.860 | **0.921 / 0.890** | +1.9 / **+3.0** | 23.8 / 23.5 |
  | w4a16 | 0.866 / 0.829 | 0.872 / 0.848 | +0.6 / +1.9 | 29.1 / 29.0 |
- **GPTQ-W8A8 fully recovers the int8 coding loss:** its plus (0.890) MATCHES fp8 and base (0.921) BEATS fp8
  (0.915) -- at identical decode (calibration is free at inference). The lift is real on code where it was
  invisible on saturated gsm8k (~0 move) -- validates weighting long-generation tiers.
- **Surprise: int8 GPTQ helped code MORE than int4 GPTQ (+3.0 vs +1.9 plus) -- opposite of the agreement
  metric** (int4 agreement lift +4.2 > int8 +2.7). Likely HumanEval-saturation + 164-item CI (both code
  deltas are a few problems wide). Tier 0 remains the tight rank; treat code lift as direction-not-magnitude.
- **Final single-B70 leaderboard (HumanEval+ plus, decode):** 27B-int4 0.927 @ 7.9 t/s > {w8a8-gptq, fp8}
  0.890 @ 23.5 / 32.1 > w8a8-rtn 0.860 > w4a16-gptq 0.848 @ 29 > w4a16-rtn 0.829 > w4a8 0.817 @ 16.5.
  Full table (quality + TTFT + prefill + VRAM) in results/SUMMARY.md. **Picks:** chat -> fp8 or w8a8-gptq;
  VRAM-tight -> w4a16-gptq; max quality -> 27B if 7.9 t/s ok; w4a8 dominated (skip for coding).
- Campaign DONE; GPU now free (w4a16-gptq left served -- tear down or repoint to the queued AutoRound W4A8
  quant). 9 served configs x HumanEval+ 164, all via the sandboxed Tier-1 path, zero grading incidents.
- **VRAM verify (w4a8 flagged 16 GB):** confirmed from serve logs -- w4a8 served VRAM = **9.3 GiB** (vLLM
  "Model loading took 9.3 GiB"; Available KV 15.6 GiB corroborates), but the checkpoint is **16 GB on disk**
  because the int4 weights are stored UNPACKED (single 16 GB safetensors, ~1 byte/int4-weight; config =
  w:num_bits4 / a:num_bits8). vLLM packs to 4-bit on load (hence the slowest load in the sweep, 39 s vs ~23).
  So leaderboard VRAM (9.3) was right for GPU memory; added "9.3 VRAM / 16 disk*" + footnote so the unpacked
  disk size (the repack target) is recorded. **Repack-to-4bit cuts disk + load time, NOT VRAM.** (W8A8-gptq is
  also 16 GB on disk but that is int8 ~= 1 byte/weight, so disk ~= its 15.3 GiB VRAM -- nothing to repack.)

### 2026-06-20 -- [PLAN] Consolidated a strategy info-dump into RESEARCH_TODO.md (deduped + AutoRound + Quark)
- **Input:** a two-part strategy dump (W8A8-primary / FP8-control / W4A16-capacity, plus accuracy + MoE + MTP
  thoughts). Deduped against existing docs and laid out as `RESEARCH_TODO.md` (sibling to MTP_TODO.md).
- **Dedup results recorded in the doc's ledger:** most accuracy levers already live in doc 07 + MTP_TODO Playbook B
  (referenced, not repeated); MTP planning stays in MTP_TODO.md (pointer only). **Two dump items were already DONE
  06-20:** the "rerun W8A8 GPTQ Tier-1" (GPTQ-W8A8 0.890+/0.921 base = ties/beats FP8 -- the dump's prediction
  confirmed) and "GPTQ@128 W4A16 HumanEval+" (0.848+). Flagged the dump's stale "W8A8 = 0.860" as the RTN number.
- **New tracks added:** (3) **AutoRound / "autoint"** as a cross-scheme weight lever -- already our int4 leader
  (27B 0.963/0.927), so test AutoRound-W4A16 vs GPTQ-W4A16 first; expect ~tie at int8 weights. (4) **Quark loader
  compatibility** -- a one-shot importer test (serve `--quantization quark`, grep whether it dispatches to our int8
  kernel), NOT a runtime migration. (5) **fused packed-MoE expert kernel** for 35B-A3B, elevated from the SUMMARY gap.
- **Boundary respected:** W4A8 + AutoRound is the other agent's branch (`w4a8/`); this doc cross-links and does not
  touch it. No GPU touched (planning only).

### 2026-06-20 -- [POC] w4a8 int8-XMX fastpath is LIVE; w4a8 vs w4a16 head-to-head on one B70
- **Q (user):** for 27B+ single-card it's w4a8 OR w4a16 (w8a8 ~15-33 GB too big) -- is there ANY advantage to
  w4a8? Hypothesis: int8 fastpath -> better prefill/TTFT. Served on `vllm-xpu-env:int8` via 36_serve.sh,
  MAXLEN=4096, MAXSEQS=32, eager. (gpu-run note: long detached servers self-serialize via 36_serve's container-kill.)
- **[CONFIRMED] int8-XMX engaged on w4a8:** serve log `Using XPUW4A8IntLinearKernel for CompressedTensorsW4A8Int`;
  `Model loading took 9.3 GiB` (VRAM 9.3, not 16) in 44 s (int4 repack tax; w4a16 loads 9.32 GiB in 22 s).
  w8a8 cross-check: `Selected XPUInt8ScaledMMLinearKernel`, 15.34 GiB (confirms w8a8 OUT for 27B single-card).
- **[DECISIVE] TTFT vs input length (out=1, C=1, same harness):**
  | input | w4a16-gptq | w4a8 | w4a8 |
  |---|---|---|---|
  | 512  | 259.9 ms | 177.6 ms | -32% |
  | 2048 | 664.7 ms | 448.6 ms | -33% |
  | 3968 | 1325.8 ms | 908.3 ms | -31% |
  => **w4a8 wins TTFT ~32% across realistic prompt lengths** (int8-XMX prefill; leaderboard prefill 4403 vs 2920,
  +51%). The old "w4a8 TTFT worse (139 vs 84)" was perf_probe's ~128-tok prompt -- below the crossover (<512 tok).
- **w4a8 concurrency (512/128):** agg 15.0 -> 357.5 tok/s (C1->C32), per-stream decode holds 12.4-15.4 (no collapse).
- **Decode (single-stream, leaderboard):** w4a8 16.5 vs w4a16 29.0 -- **w4a16 wins**, BUT w4a8 is KERNEL-limited
  (9.3 GiB => ~65 t/s ceiling => ~75% of the gap is `int4_gemm_w4a8` GEMV/unpack overhead, not physics). w4a16
  (29 t/s, fp16 act) is near its ceiling; its prefill is structurally capped (no int8-XMX).
- **VERDICT:** w4a8's advantage over w4a16 = **prefill/TTFT (~32%) + headroom** (wins long-context/agentic/
  prefill-heavy 27B). w4a16 wins decode (interactive/long-gen). Only w4a8 has the optimization headroom to win BOTH
  (crack int4_gemm_w4a8 decode -- our wheelhouse). Pursue w4a8 if prefill/TTFT/throughput-heavy OR we commit to the
  int4-decode kernel; else w4a16 for decode-heavy. Next: AutoRound (0.817 -> >=0.848) + profile int4 decode. GPU freed.

### 2026-06-20 -- [PLAN] Added docs/quant_methods.md -- quant-method registry (algorithm x scheme x model)
- **Why:** consolidate "which quant algorithm per precision scheme" + "what we've tried on what model" into one
  place (was scattered across SUMMARY + journal + doc 07). 4 tables: (A) method->scheme plan, (B) glossary,
  (C) coverage/evidence ledger (the "GPTQ beat RTN" table), (D) the XPU kernel gate.
- **Folded in two strategy dumps:** rotation method picks by scheme -- **QServe/SpinQuant @ W4A8**,
  **QuaRot/FlatQuant @ W4A4** -- vs SmoothQuant/AutoSmoothQuant/GPTQ/AutoRound. W4A4-INT4 section: FlatQuant first
  (SOTA acc, has Qwen2.5 model_tools; fused Kronecker-affine -> kernel applies a pre-transform), QuaRot fallback
  (parameter-free Hadamard -> cleaner int4xint4 target), PrefixQuant for static acts. QServe/SpinQuant ruled out
  for the INT4 fastpath (they deploy A8).
- **Kernel gate recorded (Table D):** W4A4 is DOUBLY gated -- needs a new `s4 x s4 -> s32` GEMM (we only have
  int4xint8) AND a transform kernel (FWHT or fused Kronecker-affine); neither exists for SYCL/XMX. Offline rotations
  fuse into weights = free to serve; online Hadamard = kernel-gated. Qwen3 QK-norm shifts rotation insertion points
  vs Qwen2.5 -> diff FlatQuant model_tools first.
- **Wiring:** RESEARCH_TODO.md points to the registry (siblings header, dedup ledger, Track 2, Track 8). Numbers stay
  authoritative in SUMMARY.md. Planning only; no GPU; `w4a8/` untouched (other agent's branch).

### 2026-06-20 -- [POC] int4_gemm_w4a8 KERNEL microbench: the GEMM is fine; decode loss is act-quant/eager, not the GEMM
- **Setup:** `w4a8/20_microbench_w4a8_decode.sh` -- times `torch.ops._xpu_C.int4_gemm_w4a8` in isolation (symmetric,
  100 iters) at decode (m=1) + prefill (m=512) shapes on `vllm-xpu-env:int8`. (Bug fixed: `docker run` needs `-i`
  or the python heredoc never reaches stdin -> silent exit 0.)
- **[KEY] decode (m=1) is ~52-64% of peak BW at the kernel level:**
  | shape (k,n) | ms/call | GB/s | % of 608 |
  |---|---|---|---|
  | 5120,17408 (MLP up/gate) | 0.142 | 314 | 52% |
  | 17408,5120 (MLP down)    | 0.115 | 387 | 64% |
  | 4096,11008               | 0.072 | 314 | 52% |
  prefill m=512: 171-182 TFLOP/s (int8-XMX humming).
- **Reframe:** the GEMM kernel is NOT the disaster (~1.6x headroom to ~85% peak). But full-model w4a8 decode is 16.5 t/s
  (~25% effective) -> **~half the decode time is NON-GEMM**: the unfused per-token int8 **activation-quant** (which
  w4a16 doesn't pay -> the 16.5-vs-29 gap) + eager-mode per-op launch overhead (graph capture OFF; compile broke on
  v0230/torch2.11). So the biggest decode lever is **fusing act-quant / enabling graph capture**, THEN the GEMM
  (52%->85%). [task #5 > task #3].
- **Quant:** GPTQ-W4A8 on CPU projected ~5-7 h (subgraph 1/41 took 11 min) -> per user, MOVED to GPU (`scripts/54
  SCHEME=W4A8 METHOD=gptq DEVICE=xpu`, container quant14b). Frees CPU for kernel builds; accuracy number in ~30 min.

### 2026-06-20 -- [RESULT] GPTQ-W4A8 = 0.872/0.835 HumanEval+ -- GPTQ closes most of the int8-act gap to w4a16
- **GPTQ-W4A8 quant done** (`scripts/54 SCHEME=W4A8 METHOD=gptq DEVICE=xpu`, 6724s / ~112 min on GPU; GPTQ is
  Cholesky-bound so GPU ran ~15% util -- slow regardless of device). Saved `models/Qwen3-14B-W4A8-gptq` (16 GB
  disk, unpacked I8). Verified: recipe = SmoothQuant+GPTQ (NOT RTN); served `qwen3-14b-w4a8-gptq` ->
  `Using XPUW4A8IntLinearKernel for CompressedTensorsW4A8Int`, 9.3 GiB VRAM.
- **HumanEval+ (164, thinking-off, greedy, sandboxed): base 0.872 / plus 0.835.** vs the bar:
  | quant (calib) | base / plus |
  |---|---|
  | w4a16 (gptq) -- bar | 0.872 / 0.848 |
  | **w4a8 (gptq) -- new** | **0.872 / 0.835** |
  | w4a8 (rtn) -- archived | 0.860 / 0.817 |
  - **GPTQ +1.8 plus over RTN** (0.817->0.835), consistent with the w8a8/w4a16 GPTQ lifts. **Base TIES w4a16-gptq
    (0.872); plus within ~1 CI (0.835 vs 0.848).** So the int8-activation accuracy penalty is small and largely
    recovered -- **w4a8 is accuracy-viable, NOT "dominated."**
- **Verdict / strategy:** accuracy no longer blocks w4a8. Its case over w4a16 = prefill/TTFT (int8-XMX: +51%
  prefill, ~-32% TTFT); its weak spot = decode (16.5 t/s, KERNEL-limited per the microbench, not physics).
  **AutoRound is now LOW priority** (would chase the last ~1.3 plus, within noise; w4a8's edge isn't accuracy).
  The decisive work is the **decode kernel** (docs/kernel/04 ladder: ONEDNN_VERBOSE -> drop symmetric zp ->
  PIECEWISE capture for w4a8 -> ...). Leaderboard + models.yaml updated; RTN w4a8 archived; w4a8-gptq canonical.

