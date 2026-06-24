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

### 2026-06-20 -- [DIAGNOSTIC] ladder step 1: ONEDNN_VERBOSE=2 on int4_gemm_w4a8 m=1 (FREE) -> B1 is ~2x, not minor
- Ran `w4a8/21_onednn_verbose_w4a8.sh` via the gpu-run flock lease (12 s, exit 0). Decode shapes m=1
  (5120,17408) + (17408,5120) and a m=512 prefill, ONEDNN_VERBOSE=2. Three decisive reads:
  1. **oneDNN bundled = v3.12.0** (commit 80afa71), Level Zero, device [0xe223] (B70). Far newer than the v3.8
     in our notes -> **Lever B5 (upgrade oneDNN) is moot, we are already ahead.**
  2. **impl = `jit:gemm:any`** for every matmul (decode AND prefill), NOT `ref` (good -- not the slow fallback)
     and NOT `grouped_micro_gemm` (the doc's good-vs-bad mental model was wrong). It is the GENERAL JIT GEMM
     emulating a GEMV at m=1 -> confirms Lever C's premise that there is no purpose-built m=1 GEMV path.
  3. **[KEY] the symmetric src zero-point IS applied and IS pure overhead.** Verbose attrs:
     `attr-scales:src0:3:f16:1x5120+wei:3:f16:128x1  attr-zero-points:src0:3:s32:1x5120+wei:0:s8`.
     The `src0:3:s32:1x5120` is a per-token s32 activation zero-point -- but our per-token int8 acts are
     SYMMETRIC (zp all-zero). A per-token src-zp forces oneDNN to carry a `zp_src * reduce_k(weight)`
     compensation, an O(k*n) read of the int4 weights at m=1 -- i.e. it can DOUBLE the weight bytes moved,
     which is exactly the dominant decode cost. So **dropping it (Lever B1) is plausibly a ~2x decode win,
     not the "cheapest minor win" the ladder framed.** The weight zp is only a scalar `wei:0:s8` (cheap).
- Steady-state verbose exec_time ~0.11 ms/call at decode (matches the 52-64%-of-peak microbench); first call
  ~0.33 ms (JIT warm). **Verdict: B1 (drop symmetric src-zp) is now the highest-EV cheap kernel lever.** Next:
  apply the B1 patch (agent drafting the exact diff + cache-key safety), rebuild (scripts/44), re-microbench.
- Method note: gpu-run shipped to the Unraid host (`/mnt/vm_8tb/b70/gpu-run`); all GPU touches this campaign go
  through its flock lease. Launched a 4-agent army (B1 patch, w4a8 register_fake for PIECEWISE, oneDNN/IPEX/vLLM
  survey, custom SYCL int4 GEMV design) -- all CPU/web only; the lead serializes every GPU run.

### 2026-06-20 -- [RESULT] B1 (drop symmetric src-zp) is CORRECT + CLEAN but does NOT speed up decode (~2x hypothesis WRONG)
- Applied the agent-A diff to `int4_gemm_w4a8.h` (drop both `set_zero_points(DNNL_ARG_SRC)` + the runtime
  zp arg + pin `zp_group_size=m2_zp.numel()` so the primitive cache can't alias sym/asym), rebuilt via
  scripts/44 (CPU, ccache -> ~1 min), validated with `w4a8/22_validate_b1.sh` (baseline baked .so vs rebuilt
  patched .so mounted, identical deterministic SYMMETRIC inputs).
- **Correctness PASS:** output fingerprints (sum/absmax/mean) match baseline to ~6 sig figs on all 4 decode
  shapes -- dropping the all-zero src-zp is numerically a no-op (as designed).
- **Verbose PASS:** patched attrs = `attr-scales:src0:3:f16+wei:3:f16  attr-zero-points:wei:0:s8` -- the
  `zero-points:src0` term is GONE, impl still `jit:gemm:any` (NOT `ref`). Patch took effect, no slow fallback.
- **[KEY] Timing: NO speedup.** Patched is a wash-to-slightly-slower vs baseline. **My "src-zp doubles the
  weight read at m=1 -> ~2x" hypothesis was WRONG:** oneDNN v3.12 precomputes the `zp_src*reduce_k(weight)`
  compensation at primitive-CREATE and caches it, so the per-decode cost of a zero src-zp was already ~nil.
  Agent A's conservative "low-single-digit %, no regression" was the right call. B1 is a correctness/cleanliness
  win (removes a logically-dead attr, matches IPEX QMatmul.h + our clean w8a8 kernel), not a perf win.
- **[CONFOUND -> controlled rerun launched]** baseline used the baked `0.1.9` .so, patched the rebuilt
  `0.1.11` -- so the small regression on 2 shapes conflates patch-effect with build-toolchain diff. Rebuilding
  the UNPATCHED control (same scripts/44 pipeline) to A/B `rebuilt-unpatched vs rebuilt-patched` cleanly.
- **[SURPRISE -> needs confirm] baseline decode measured 72-93% of peak BW, not the doc's 52-64%.** The new
  harness uses 200 iters / 30 warmup + CPU-genned inputs; the original `20_microbench` used 100/20 + on-xpu
  input gen. If the higher number is the true steady-state (likely a warmup/clock effect), then **Lever C (a
  custom m=1 GEMV) has far less headroom than the design assumed** (the GEMM is already near BW-bound on the
  best shapes). Re-measure `20_microbench` with matched warmup to settle it before investing in Lever C.
- Banked agent deliverables: `docs/kernel/05_int8_int4_optimization_survey.md`, `docs/kernel/06_sycl_int4_gemv_
  design.md`, `docs/kernel/patches/{B1_analysis.md,A1_graph_capture_w4a8.md,int4_gemm_w4a8_drop_src_zp.diff}`,
  `contrib/vllm_int8_xpu/xpu_int4*.{py,diff}`. Survey flagged doc fixes (literature/06 graph section stale;
  issue #3323 is a closed memory-leak not a perf item; B3 is net-new not an IPEX port; split B4 into int8/int4).

### 2026-06-20 -- [CONFIRM] controlled B1 A/B: perf-NEUTRAL; microbench noise is +/-30% (-> measure at decode-t/s)
- Removed the build confound: rebuilt the UNPATCHED control on the SAME scripts/44 toolchain, then A/B'd
  `b1_unpatched.so` vs `b1_patched.so` (both mounted, identical sym inputs, 2 interleaved rounds) via
  `w4a8/23_ab_b1.sh`. (Gotcha fixed: dropped `docker run -i` -- with the python mounted as a file, `-i` made
  docker slurp the piped script's stdin and only the first round ran.)
- **Result -- B1 is performance-NEUTRAL:** unpatched ~= patched within noise on all 4 shapes (per-shape ms,
  unpatched vs patched avg): 4096x11008 .048/.058, 5120x17408 .108/.102, 17408x5120 .090/.088, 5120x5120
  .040/.041. Correctness fingerprints identical every run. **The decode "regression" in the prior entry was the
  baked-0.1.9-vs-rebuilt confound + noise, NOT the patch.**
- **[META-FINDING] microbench run-to-run noise ~ +/-30%** (k=4096 patched swung 0.072 -> 0.045 ms between
  rounds = 59% spread; % of peak ranged 50-86% for the same shape). This noise DWARFS small kernel deltas ->
  (a) the "52-64%" (doc) vs "72-93%" (prior entry) discrepancy was just sampling; true decode BW is shape-
  dependent ~50-86% and NOISY; (b) **kernel changes must be validated at the full-model decode-t/s level**
  (which averages out the noise over hundreds of tokens), not by these microbenches. Square/small shapes
  (5120x5120 ~52-56%) are consistently the worst -> the real GEMV headroom (Lever C) lives there, not on the
  already-near-BW tall-skinny shapes.
- **Verdict + decision:** B1 kept in the source-of-truth (`int4_gemm_w4a8.h`, matches IPEX QMatmul.h + our clean
  w8a8 kernel) as a correctness/cleanliness improvement, but **prod `:int8` NOT rebaked** (no perf benefit, and a
  flag-mismatched rebuild risks a noise-band regression). Pivot: the dominant decode cost is dispatch/non-GEMM
  (full-model w4a8 = 16.5 t/s ~= 25% effective while the GEMM alone is 50-86% of peak) -> **next = A1 graph
  capture for w4a8** (attacks dispatch; measured at decode-t/s, the stable metric; w8a8 already banked +16.7%).

### 2026-06-20 -- [BREAKTHROUGH] A1: PIECEWISE graph capture lifts w4a8 decode 16.79 -> 48.18 t/s (+187%, 2.87x)
- **THE headline result of the kernel campaign.** Served Qwen3-14B-W4A8-gptq on `:int8g` via
  `w4a8/30_serve_w4a8_graph.sh`, EAGER (GRAPH=0) vs PIECEWISE XPU graph capture (GRAPH=1), same decode-t/s
  probe (`w4a8/31_decode_probe.sh`, 256 tok ignore_eos, temp 0, single-stream):
  | mode | decode t/s | spread | % of ~65 t/s BW ceiling |
  |---|---|---|---|
  | eager     | 16.79 | 16.72-16.93 | ~26% |
  | PIECEWISE | **48.18** | 48.16-48.19 | ~74% |
  => **+187% (2.87x)**, and the PIECEWISE timing is dead-flat (5.312 s +/- 0.001 over 5 trials -- the
  determinism signature of a captured graph). Output verified COHERENT (correct Rayleigh-scattering answer),
  so the captured graph is fast AND correct.
- **Why so much bigger than w8a8's +16.7%?** w4a8's activation quant is the UNFUSED pure-PyTorch
  `dynamic_per_token_int8_quant_ref` (min/max/round/clamp = hundreds of tiny ops/token in eager), so eager
  w4a8 is severely dispatch-bound. Graph capture fuses the whole decode step -> jumps from ~26% to ~74% of the
  BW ceiling (9.3 GiB @ 608 GB/s ~= 65 t/s). w8a8 uses a FUSED custom quant op, so its eager was already lean
  -> far less capture headroom. The int4 unpack adds even more eager ops. So **w4a8 benefits
  disproportionately from capture** -- it was the most dispatch-bound config, now near BW-bound.
- **Leaderboard upheaval:** old story was "w4a8 decode 16.5 t/s, DOMINATED by w4a16 @ 29." NEW: **w4a8
  PIECEWISE = 48 t/s beats EVERYTHING measured eager** (fp8 ~32, w4a16 29, w8a8-piecewise 27, w8a8 23.5). w4a8
  already won prefill/TTFT (int8-XMX, ~-32%) + smallest VRAM (9.3 GiB); now it leads decode too (pending the
  apples-to-apples step: capture the OTHER configs too). w4a8 is no longer "dominated" -- it may be the pick.
- **Two bugs found + fixed en route (both reusable):**
  1. vLLM XPU+compile crash `NameError: MLARoPEKVCacheCatFusionPass` -- vLLM auto-enables CUDA-only inductor
     fusion passes under torch.compile, but XPU never imports those classes (pass_manager gates on
     is_cuda_alike()); the flags default None then resolve True unguarded. FIX: disable the CUDA/ROCm-only
     fusion flags in the serve `pass_config` (committed in 30_serve_w4a8_graph.sh). These fusions can't run on
     XPU anyway; capture is independent of them. **This likely means the banked w8a8 +16.7% needs re-confirming
     on the current image -- next.**
  2. The int4 register_fake (A1's original premise) was REDUNDANT: vLLM already ships a native fake at
     `vllm/_xpu_ops.py:60` for int4_gemm_w4a8, so our fake is skipped (harmless). Capture engaged regardless.
     -> the A1 "register_fake" work was belt-and-suspenders; the real blockers were the compile-pass bug + env.
- **Capture confirmed engaged:** log shows `mode: VLLM_COMPILE`, `enforce_eager=False`, `saved AOT compiled
  function`, `Capturing CUDA graphs (PIECEWISE): 4/4 ... Graph capturing finished in 40 secs, took 0.93 GiB`
  (sizes [1,2,4,8]). First AOT compile ~7.5 min (cached to /vllm_cache for next serve). Served-id verified
  `qwen3-14b-w4a8-gptq` (CLAUDE.md model-check). Next: (a) re-confirm w8a8 PIECEWISE on the current image,
  (b) capture fp8/w4a16 for the apples-to-apples decode leaderboard, (c) try FULL capture (TRITON_ATTN).

### 2026-06-20 -- [CONFIRM] w8a8 PIECEWISE reconciled (+13%) -> validates the "w4a8 unfused-quant" hypothesis
- Re-measured w8a8 on the CURRENT `:int8g` (with the pass_config compile-fix), same probe, eager vs PIECEWISE:
  | config | eager | PIECEWISE | capture gain |
  |---|---|---|---|
  | w8a8 | 23.62 | 26.68 | **+13%** (reproduces the banked +16.7% within noise) |
  | w4a8 | 16.79 | 48.18 | **+187%** |
  Both PIECEWISE timings dead-flat (w8a8 9.593 +/- 0.002 s, w4a8 5.312 +/- 0.001 s = captured-graph determinism).
- **Hypothesis CONFIRMED:** the 14x difference in capture gain (+187% vs +13%) is the activation-quant fusion.
  w4a8's per-token int8 act-quant is the UNFUSED pure-PyTorch `dynamic_per_token_int8_quant_ref` (hundreds of
  eager ops/token) -> eager w4a8 is dispatch-bound -> capture wins huge. w8a8 uses the FUSED custom
  `dynamic_per_token_int8_quant` op -> eager already lean -> small capture headroom. The banked w8a8 +16.7% is
  reproducible (so the compile-pass bug must post-date it, or its run predated the regression -- either way the
  number stands). **Decode leaderboard now (PIECEWISE where measured): w4a8 48.18 >> {fp8 ~32, w4a16 29 (both
  still EAGER), w8a8-piecewise 26.68}.** To finish the ranking, capture fp8/w4a16 too (they use fp16 acts =
  lean eager, so expect ~w8a8-like modest gains -> w4a8 should still lead). Next: A2 FULL capture (push higher
  + flip spec-decode) and the fp8/w4a16 capture sweep.

### 2026-06-20 -- [RESULT] A2 FULL capture is BLOCKED on the current image (confirms the SYCL-Graph premise on B70)
- Tried FULL capture (`30_serve_w4a8_graph.sh GRAPH=1 CGMODE=FULL_AND_PIECEWISE ATTN=TRITON_ATTN`). Two real
  blockers, both now CONFIRMED on our actual B70 (not just literature):
  1. **flash-attn FULL capture hits the SYCL-Graph restriction:** engine init died with
     `RuntimeError: The sycl_ext_oneapi_work_group_scratch_memory feature is not yet available for use with the
     SYCL Graph extension.` FULL_AND_PIECEWISE tries to put attention in a captured graph; flash-attn's
     work-group scratch memory is incompatible with SYCL-Graph capture. Exactly the doc-04 / literature/06
     premise -- now verified live, not predicted.
  2. **TRITON_ATTN could not engage:** the `VLLM_ATTENTION_BACKEND=TRITON_ATTN` env was ignored -> log showed
     `Using Flash Attention backend`. Root cause: the `:int8g` image logs `Triton is installed but 0 active
     driver(s) found (expected 1). Disabling Triton` -> Triton-XPU is auto-disabled (driver detection), so the
     Triton attention path is unavailable and vLLM falls back to flash-attn.
- **Verdict:** A2 (FULL capture, which would also flip spec-decode positive) needs EITHER a working Triton-XPU
  driver in the image OR the oneAPI 2026.0 toolchain (its release notes lift the work_group_scratch_memory
  restriction). Both are larger efforts, and the **decode upside is now small** -- PIECEWISE already put w4a8 at
  ~74% of the BW ceiling (48 t/s); FULL would only recover the residual attention-launch overhead, a few %.
  So A2 is **parked** (documented, not worth the toolchain/Triton yak-shave for a few % decode). The PIECEWISE
  win (+187%) is the headline; FULL is a future toolchain freebie. Pivot to the decisive w4a8-vs-w4a16 decode
  comparison under capture (same 9.3 GiB VRAM/ceiling; w4a16 has no act-quant tax so it is the real rival; fp8
  is capped at ~40 t/s by its 15.3 GiB and cannot beat 48).

### 2026-06-20 -- [RESULT + CORRECTION] apples-to-apples under capture: w4a16 LEADS decode (54.57), not w4a8 (48.18)
- Captured the real rival. w4a16 (int4 weight, fp16 act, XPUwNa16) eager vs PIECEWISE, same probe; output verified
  coherent. Full decode-t/s table, eager -> PIECEWISE:
  | config | VRAM | eager | PIECEWISE | capture gain | % of BW ceiling |
  |---|---|---|---|---|---|
  | **w4a16** | 9.3 GiB | 28.04 | **54.57** | +95% (1.95x) | ~84% of 65 |
  | w4a8  | 9.3 GiB | 16.79 | 48.18 | +187% (2.87x) | ~74% of 65 |
  | w8a8  | 15.3 GiB | 23.62 | 26.68 | +13% | -- |
  | fp8   | 15.3 GiB | ~32 (eager) | not captured | -- | capped ~40 by 15.3 GiB |
- **[CORRECTION of my earlier over-claim]** the prior entries said "w4a8 PIECEWISE 48 beats EVERYTHING" -- that
  was only true while the others were EAGER. Apples-to-apples (both int4 configs captured), **w4a16 wins decode
  54.57 > w4a8 48.18** (+13%). Same 9.3 GiB weight read, but w4a16 skips the int8 activation quant entirely
  (fp16 acts straight into the GEMM), so its int4xfp16 GEMM is ~13% more BW-efficient at m=1 than w4a8's
  int4xint8 path (84% vs 74% of ceiling). w4a8 still wins PREFILL/TTFT (int8-XMX: +51% prefill, ~-32% TTFT).
- **THE real headline (corrected + bigger):** *graph capture roughly DOUBLES int4 decode on the B70* -- w4a16
  1.95x, w4a8 2.87x. Both int4 paths were heavily eager-dispatch-bound (w4a8 most, due to the unfused
  act-quant); w8a8 was already lean (fused quant, +13%). **Capture is the dominant decode lever for int4, full
  stop.** This was the doc-04 top prediction; now quantified on real hardware.
- **Picks (single B70, captured):** decode-heavy/interactive -> **w4a16 (54.57 t/s, near-lossless, 9.3 GiB)**;
  prefill-heavy/long-context/agentic -> **w4a8** (decode 48 + best prefill/TTFT, same 9.3 GiB); chat-quality at
  larger VRAM -> fp8/w8a8-gptq. w4a8 is NOT dominated (it was, pre-capture) -- it is now co-leader with w4a16,
  split by the decode-vs-prefill axis. Next: final synthesis -> update results/SUMMARY.md leaderboard.

### 2026-06-20 -- [WRAP] capture campaign conclusion + 27B deferred (shared GPU) + SUMMARY leaderboard updated
- **27B flagship capture DEFERRED.** Qwen3.6-27B-int4-AutoRound LOADS fine on `:int8g` (17.56 GiB, 41 s, bf16,
  quant=inc) but my decode probe hit a runtime `EngineCore ... InternalServerError` during generation (likely
  the `ignore_eos` param or a GDN-int4 decode-path quirk on `:int8g` vs the `:v0230` it was first benched on).
  Another agent now shares the B70 (Qwen3.6-35B MoE loading, intermittent) -> per the user I tore the 27B
  server down immediately to free the card and did NOT re-occupy it to debug a CONFIRMATORY run. The 27B eager
  baseline already exists (7.59 t/s, perf_probe); only its PIECEWISE number is open. Parked task #8 (attempt
  opportunistically when the card is uncontended; a plain non-ignore_eos request likely sidesteps the error).
- **Campaign scoreboard (docs/kernel/04 ladder):** step 1 ONEDNN_VERBOSE diag DONE; B1 drop-src-zp DONE
  (perf-neutral, kept for cleanliness); **A1 PIECEWISE capture DONE = the breakthrough (int4 decode ~2x:
  w4a16 28->55, w4a8 17->48, w8a8 +13%)**; A2 FULL capture BLOCKED (SYCL-Graph work_group_scratch + Triton
  auto-disabled, both confirmed live); Lever C custom SYCL GEMV NOW LOW-EV (post-capture decode is already
  74-84% of the BW ceiling -> a hand GEMV would chase only the residual ~15-25%, 1-2 wk for <=1.3x).
- **Single biggest takeaway:** the decode bottleneck on the B70 for int4 was NEVER the GEMM -- it was eager
  per-op DISPATCH (worst for w4a8 because its act-quant is an unfused pure-PyTorch ref). PIECEWISE XPU graph
  capture (torch 2.11+xpu, `:int8g`, `30_serve_w4a8_graph.sh GRAPH=1` + the `pass_config` compile-fix) removes
  it and ~doubles int4 decode. Use capture, not eager, for decode-bound int8-activation/int4 serving on B70.
- Banked across JOURNAL + FINDINGS + evals/results/SUMMARY.md (authoritative leaderboard banner) + docs/kernel/04
  ladder + docs/literature/06 (stale "no XPU capture" verdict corrected). GPU left FREE for the other agent.

### 2026-06-20 — [BREAKTHROUGH] Qwen3.6-35B-A3B int4 (256-expert MoE) NOW LOADS + GENERATES on ONE B70
- Reopened the 2026-06-19 "MoE-on-XPU gap" (35B-A3B OOM'd at load). The gap was NOT a missing kernel -- it was a
  one-branch routing bug in vLLM's INC quant integration. **Root cause** (read from the live `:v0230` image,
  vllm 0.23.0+xpu): `auto-round` maps to `INCConfig` (inc.py); `INCConfig.get_quant_method` sends ALL XPU layers
  to `apply_xpu_w4a16_quant_layer`, which only handles `LinearBase`/`ParallelLMHead` and **returns `None` for
  `RoutedExperts`**. A `None` MoE method -> vLLM falls back to `UnquantizedFusedMoEMethod` (bf16 experts) -> the
  256 int4 experts dequantize toward ~70 GB -> `OUT_OF_DEVICE_MEMORY` at load. On CUDA/CPU the same INC routes
  `RoutedExperts` -> `MoeWNA16Config` (int4-preserving); the XPU branch simply never implemented it. (llm-scaler
  0.14.x is worse: its INC returns `UnquantizedFusedMoEMethod` for ANY MoE -> same OOM. So no stock Intel image
  serves this model; community runs must patch, exactly as hinted.)
- **Fix (the missing piece):** add the `RoutedExperts` branch to `apply_xpu_w4a16_quant_layer`, mirroring the
  proven gptq path -- present the experts as a gptq config and return
  `MoeWNA16Config.from_config({...}).get_quant_method(layer, prefix)`. ~16 lines.
  `contrib/vllm_moe_xpu/inc.py` (full patched file) + `contrib/vllm_moe_xpu/README.md`. On XPU
  `should_moe_wna16_use_cuda()` is False (it is `is_cuda`-gated), so `fused_experts` dispatches to the pure-Triton
  `invoke_fused_moe_wna16_triton_kernel` -- NO CUDA-only op needed. Triton is live on XPU here (the GDN linear-attn
  kernel already uses Triton/FLA).
- **No image rebuild:** bind-mount the patched inc.py over the image file
  (`-v .../inc_xpu_moe.py:/opt/venv/.../quantization/inc.py:ro`). Iterate in seconds, not a 1-2 h build.
- Config -> command:
  `scripts/runremote.sh scripts/53_loadtest_35b_moe_xpu.sh`  (offline LLM load + 24-tok greedy generate, eager,
  maxlen 2048, util 0.95, dtype bf16 native; wrapped in host `gpu-run` so the lease is held only for the test).
- **Result (exit 0, 162 s end-to-end):**
  - `quantization=inc`, arch `Qwen3_5MoeForConditionalGeneration` resolved.
  - **`Model loading took 19.6 GiB`** (int4 experts stay PACKED) -- vs the prior ~70 GB bf16 dequant that OOM'd.
  - Memory profiling PASSED (the step that OOM'd before): `Available KV cache 6.24 GiB`, `GPU KV cache 122,880
    tokens`, `Maximum concurrency 60.00x @ 2048`.
  - MoE confirmed int4: `fused_moe.py: Using default MoE config ... E=256,N=512,device_name=Intel(R)_Graphics_
    [0xe223],dtype=int4_w4a16` + Triton JIT of `fused_moe_kernel_gptq_awq` (the wna16 int4 MoE GEMM).
  - **`GENERATION OK` -> "The capital of France is" -> " Paris, a city renowned for its rich history, culture, and
    iconic landmarks. Situated in the north-central part of"** -- coherent. ~6 t/s decode (single-stream, eager).
- **Verdict:** the MoE-on-XPU gap for int4 weight-only is CLOSED for load+inference via a 16-line INC routing fix;
  the wna16 Triton MoE kernel runs fine on Battlemage. Updates FINDINGS / SUMMARY / quant_methods (was "OOM -- no
  fused int4 MoE kernel"). NOT yet done: perf tuning (no `E=256,N=512,int4_w4a16` config -> "sub-optimal" warning;
  ~6 t/s eager is a load proof, not an optimized number), graph capture, accuracy eval, MTP/shared-expert checks.
  GPU lease released immediately after the test (other agent unblocked).

### 2026-06-20 -- [ROOT-CAUSE + DEFER] 27B capture probe error = WRONG IMAGE (:int8g lacks GDN); run-ready on :v0230
- My earlier 27B-on-`:int8g` probe error (`EngineCore ... InternalServerError` on first token) is now ROOT-CAUSED
  by the other agent's note: **`:int8` / `:int8g` were built minimal (`GDN_ENABLED=OFF`) so they LACK
  `gdn_attention` and the Gated-DeltaNet 27B crashes on the first token.** Not a capture problem -- wrong image.
  The 27B must serve on the **full `:v0230` build** (which has GDN). (Confirmed the 27B uses GDN: vLLM splits
  `vllm::qwen_gdn_attention_core` out as a piecewise op.)
- **27B PIECEWISE is RUN-READY (deferred for GPU courtesy).** Exact command for a free-GPU window (the 27B is
  AutoRound int4 = w4a16-like, which captured +95% on the 14B, so expect a big lift on the 7.59 t/s eager):
  `IMG=vllm-xpu-env:v0230 MODEL=/models/Lorbus_Qwen3.6-27B-int4-AutoRound SERVED=qwen36-27b-int4 GRAPH=1
  DTYPE=auto UTIL=0.92 bash /mnt/vm_8tb/b70/30_serve_w4a8_graph.sh` then `SERVED=qwen36-27b-int4 N=256 TRIALS=4
  bash .../31_decode_probe.sh` (use a plain request, not ignore_eos). 30_serve's pass_config fix is
  image-agnostic so it dodges the MLARoPE NameError on :v0230 too. **NOT run now:** it needs a ~10-12 min AOT
  compile = too long a hold on the shared B70 while the other agent iterates on the 35B. Parked until the card
  is free for a longer window (task #8).
- **[POINTER for the MoE workstream, not my branch] graph capture is very likely the 35B-A3B's biggest decode
  lever too.** The other agent flagged "graph capture: not yet done" at ~6 t/s eager. Given capture ~doubled
  every DENSE int4 path here by killing eager dispatch overhead, the 256-expert MoE (even more per-token
  Python/dispatch) is a prime candidate -- try `30_serve ... GRAPH=1` on the MoE image once it is stable
  (watch: the Triton `fused_moe_kernel_gptq_awq` must be in `splitting_ops` / capturable; PIECEWISE keeps
  uncaptured ops eager so it should at least not regress). Left to the MoE owner; flagged here so it is not lost.
- No GPU touched this iteration (campaign core is complete; remaining items are GPU-courtesy-gated or the other
  agent's domain). Kernel campaign verdict stands: PIECEWISE capture ~2x's int4 decode; w4a16 leads, w4a8 prefill.

### 2026-06-20 -- [BREAKTHROUGH] 27B flagship PIECEWISE capture: 7.84 -> 30.84 t/s (+293%, 3.93x!) -- erases the density tax
- User: "work on 27b and 35b MoE decode speedups." The other agent freed the B70. First the 27B (the flagship
  the leaderboard calls the real target). Served Qwen3.6-27B-int4-AutoRound on the GDN-capable **`:v0230`**
  (NOT `:int8g` -- that lacks GDN, the cause of the earlier crash), eager vs PIECEWISE, same probe (192 tok):
  | mode | decode t/s | spread | % of ~34.6 t/s BW ceiling |
  |---|---|---|---|
  | eager     | 7.84  | 7.82-7.88 | ~23% |
  | PIECEWISE | **30.84** | 30.82-30.90 | **~89%** |
  => **+293% (3.93x)** -- the BIGGEST capture gain yet, even past w4a8's 2.87x. Dead-flat timing (6.229 s +/-
  0.001 = captured-graph determinism). Output verified coherent (Qwen3 `<think>` reasoning). Capture engaged:
  `Capturing CUDA graphs (PIECEWISE) 4/4 ... finished in 56 s, took 1.03 GiB`; GDN attention stays piecewise
  (split op) and the rest is captured.
- **Why 3.93x (even bigger than the 14B):** the 27B's Gated-DeltaNet linear attention runs a long chain of tiny
  Triton/FLA ops per token -> eager was EXTREMELY dispatch-bound (only ~23% of BW). Capture fuses the step ->
  ~89% of ceiling. Same mechanism as the 14B int4 (kill eager dispatch), just more headroom because GDN is even
  more op-heavy in eager.
- **[STRATEGIC] the "higher-density tax" is largely ERASED.** Old SUMMARY framing: "27B int4 = +4 quality pts but
  ~4x slower decode (7.9 vs 32 fp8)." With capture the 27B decodes at **30.84 t/s ~= the 14B fp8's 32 t/s** while
  keeping +4.8 base / +3.7 plus HumanEval. So on ONE B70 the 27B is now the **best quality AND ~competitive
  decode** -> arguably the new default single-card pick (if 17.6 GiB fits the KV need). Capture changed the
  whole single-card calculus. Next: the 35B-A3B MoE (eager serve loading on `:v0230moe` now).

### 2026-06-20 -- [BREAKTHROUGH] 35B-A3B MoE PIECEWISE capture: 7.93 -> 56.84 t/s (+617%, 7.17x!!) -- now the FASTEST config
- The second half of the user's ask. Baked **`:v0230moe`** = `:v0230` + the other agent's INC MoE routing patch
  (`inc.py`, RoutedExperts -> MoeWNA16Config), then served Qwen3.6-35B-A3B-int4-AutoRound (256-expert MoE) via
  `vllm serve` (NEW -- the MoE was only ever offline-`LLM()`-tested before; the served API path works), eager
  vs PIECEWISE, same probe (160 tok):
  | mode | decode t/s | spread |
  |---|---|---|
  | eager     | 7.93  | 7.91-7.95 |
  | PIECEWISE | **56.84** | 56.82-56.87 |
  => **+617% (7.17x)** -- the BIGGEST capture gain measured, and at 56.84 t/s the 35B MoE is now the **fastest-
  decoding config on the whole single-B70 leaderboard** (past the 14B w4a16's 54.6). Dead-flat (2.815 s +/-
  0.001). Capture engaged: `Capturing CUDA graphs (PIECEWISE) 4/4 ... finished in 72 s, took 0.08 GiB`.
- **[CORRECTNESS verified -- the MoE+cudagraph trap did NOT bite]** data-dependent expert routing could in
  principle be frozen by a captured graph (all tokens -> the captured experts = garbage). It is NOT: vLLM's
  masked `fused_moe_kernel_gptq_awq` processes a routing-agnostic gather/mask, so the captured graph is correct
  for any routing. Output coherent across 3 prompts ("...is Paris", proper `<think>`, "2+2 equals 4"). The MoE
  `fused_experts` op is NOT in `splitting_ops` (only the attention ops are) -> the expert GEMM IS captured.
- **Why 7.17x (the biggest):** an A3B MoE activates only ~3B params/token, so the per-token weight read is tiny
  (high BW ceiling) while eager pays ENORMOUS dispatch overhead (256-expert routing + gather/scatter + GDN
  attention = the most small-ops-per-token of any config). Capture collapses all that -> 7x. **The more
  dispatch-bound the eager path, the bigger the capture win:** MoE(7.17x) > 27B-GDN(3.93x) > w4a8(2.87x) >
  w4a16(1.95x) > w8a8(1.13x). A clean monotonic trend in eager-op-count.
- **STRATEGIC:** the 35B-A3B (the highest-capacity single-card model) now decodes at 56.84 t/s -- it went from a
  "~6 t/s load proof" to the **fastest single-card decode, period**, on a 35B-class model. Both flagships
  (27B 30.8, 35B-A3B 56.8) are transformed by capture. GPU freed for the other agent. Banked to FINDINGS/SUMMARY.

### 2026-06-20 -- [BASELINE] pp/TTFT/decode of the captured W4A16 flagships at medium ctx (the W4A8 target to beat)
- User plan: convert the 27B + 35B from W4A16 (int4 w, fp16 act) to W4A8 (int4 w + int8 act) via AutoRound, to
  KEEP int4 decode + GAIN prefill via the int8-XMX systolic fastpath (the 14B w4a8 had +51% prefill / -32%
  TTFT vs w4a16). First, the baseline -- `perf_probe.py` (streaming, single-stream, ~1800-tok medium prompt)
  on the CAPTURED (GRAPH=1) servers:
  | model (W4A16, captured) | pp (prefill t/s) | TTFT ms (short) | prefill-TTFT ms (1800 tok) | decode t/s |
  |---|---|---|---|---|
  | 27B dense GDN | **1521.7** | 92.3 | 1183.6 | 31.36 |
  | 35B-A3B MoE   | **2142.5** | 108.6 | 840.6 | 73.43 |
  (perf_probe decode excludes TTFT -> 31.36/73.43 are pure inter-token rates; my earlier round-trip probe gave
  30.84/56.84 incl TTFT. The MoE prefill 2142 > dense 27B 1521 because A3B activates only ~3B for prefill too.)
- Capture also nudged PREFILL up vs the eager leaderboard (27B 1376 -> 1521 captured, +10%) -- prefill is mostly
  compute-bound so the gain is small, but the prefill graph does capture. **This pp is the W4A8 target:** if
  W4A8 lights int8-XMX like the 14B (+51%), expect 27B prefill ~2300 t/s. Gating the quant on the feasibility
  agent (AutoRound-W4A8 recipe + does an int8-ACT MoE path even exist on XPU + which image has W4A8-kernel AND
  GDN). User pref: target Intel AutoRound W4A8-int8; SmoothQuant+GPTQ (scripts/54) is the proven fallback.

### 2026-06-20 -- [BLOCKED->PIVOT] AutoRound CANNOT export W4A8 (W8A8-only); fall back to GPTQ/RTN via llmcompressor
- Built the corrected AutoRound-W4A8 recipe (auto_round 0.13.1, hand-built QuantizationScheme, device_map=xpu,
  quantize_and_save format=llm_compressor) -> `w4a8/{_autoround_w4a8.py,12_*.sh}`. Pre-check passed; smoke run
  loaded the 72 GB bf16 27B and **quantized fine on the 32 GB card (device_map=xpu block-streamed, no OOM)** --
  but FAILED at export with `AssertionError: only support to export llm_compressor ... W4A8 ... group_size=-1`.
- **Root cause (read auto_round/formats.py::check_and_reset_format):** the llm_compressor exporter, for ANY
  int8-dynamic-act scheme, hard-asserts `bits==8 and group_size==-1 and sym and act_bits==8` -> it is **W8A8-
  only**. The error message renders "W4A8" from the actual bits but the REQUIRED config is W8A8. Re-checked with
  group_size=-1 (still fails: bits=4 != 8) and confirmed **0.13.1 is the LATEST auto_round** (no newer version
  fixes it). The native `auto_round` format would serve as W4A16 on XPU (INC path ignores int8 acts). So
  **Intel AutoRound cannot produce a W4A8-int8 checkpoint for the XPU int8-XMX kernel, full stop.**
- **PIVOT (user pre-approved):** SmoothQuant+GPTQ via llmcompressor (`scripts/54`, the PROVEN 14B W4A8 path ->
  CompressedTensorsW4A8Int -> XPUW4A8IntLinearKernel). 27B-specific risk: scripts/54's header warns SmoothQuant's
  mapping resolver fails on the hybrid Qwen3_5 GDN arch (unlike the dense 14B) -> use RTN/GPTQ without SmoothQuant
  if needed. **Now running RTN-W4A8 27B (DATAFREE=1, CPU, no calibration -> fast, lowest arch risk)** to get the
  SPEED answer first (prefill is scheme-dependent, not quant-quality-dependent): serve -> perf_probe -> does the
  27B W4A8 prefill beat the W4A16 baseline (1521.7 t/s)? Then the quality GPTQ-W4A8 run overnight.
- Lesson banked: the speed hypothesis (W4A8 int8-XMX prefill) is testable with a CRUDE RTN checkpoint; only the
  ACCURACY question needs the good quantizer. Decouple them -> answer speed tonight regardless of quant method.

### 2026-06-20 -- [DEFER] 27B-W4A8 serve blocked by VLM odd-dims (4304); the speed hypothesis is already proven on 14B
- Got a CORRECT W4A8 27B checkpoint built (scripts/49 RTN, 42s; ignore linear_attn/visual/mtp/lm_head) +
  solved the repo's standing "27B compressed-tensors won't serve" config issue: the 27B is a Qwen3_5 **VLM**
  (vision + DeltaNet + MTP), and llmcompressor saves the collapsed text config (`qwen3_5_text`) that vLLM
  rejects. Fix (`w4a8/fix_27b_vlm_config.py`): graft the original VLM wrapper config + copy the processor
  files (preprocessor_config/video_preprocessor/merges/vocab/chat_template). That got past config + processor.
- **But the serve then hit the REAL wall:** `compressed_tensors_w4a8_int.py: AssertionError: input_size_per_
  partition 4304 not divisible by group_size 128` -- the **vision MLP** (Qwen3_VisionMLP.linear_fc2, dim 4304)
  and the DeltaNet projections have dims not divisible by 128, so the W4A8 group-128 kernel cannot take them.
  This is exactly the repo's known "27B /32-dim" blocker, now pinned: the 27B's VLM/DeltaNet odd dims are
  fundamentally incompatible with our group-128 W4A8/W4A16 kernel. Serving needs those layers kept BF16 AND a
  loader that builds them BF16 -- a non-trivial VLM-quant exercise (also: AutoModelForCausalLM vs the VLM
  loader changes weight-name prefixes/ignore-matching). Multiple interacting issues = a real rabbit hole.
- **DECISION: defer the 27B-W4A8** (the user redirected to deep w8a8 work). The W4A8 SPEED hypothesis (int8-XMX
  prefill) is ALREADY proven on the dense 14B (w4a8 prefill +51% / TTFT -32% vs w4a16) -- the 27B was only a
  confirmation. **Learning banked:** (1) Qwen3.6-27B is a VLM, not a plain text model -> quant needs the
  27B-aware path + BF16 vision/DeltaNet/MTP + the wrapper-config graft; (2) group-128 int4 kernels reject the
  27B's 4304-dim layers -> a per-channel (group=-1) or padded-group W4A8 kernel would be needed to serve them,
  OR keep them BF16 via a correct VLM-aware quant+load. Filed for the dual-card phase. Pivoting FULLY to the
  deep w8a8-14B GEMM + fused-quant + MTP hand-tuning (3 codex agents mapping the space; GPU now free).


---

## 2026-06-20 -- INT8 W8A8 GEMM hand-tuning plan (docs/kernel/10) -- pp+tg roofline

CONFIG: deep research pass (3 web-research agents + 2 codex SYCL drafts + 1 codex critique) to produce a
ranked, executable hand-tuning plan to push the int8 W8A8 GEMM to the B70 roofline for BOTH prefill
(compute-bound, 367 TOPS) and decode (BW-bound, 608 GB/s). Read ALL live kernel source via ssh.

COMMAND: read int8_gemm_w8a8.h / onednn_ext.h / onednn_matmul.cpp / dynamic_per_token_int8_quant.cpp /
xpu_int8.py on the box; codex exec drafted draft_int8_gemm_prefill (DPAS/joint_matrix) +
draft_int8_gemv_decode (dp4a); agents fetched CUTLASS/Marlin/Machete, oneDNN-JIT/IPEX-XeTLA, ggml-sycl/
sycl-tla sources. -> docs/kernel/10_int8_gemm_handtune_plan.md.

RESULT (key VERIFIED findings):
- **THE PREFILL GAP (VERIFIED from our source):** `onednn_ext.h:795` builds the weight md with EXPLICIT
  STRIDES (`wei_strides={1,ldb}`), NEVER `format_tag::any`. This pins oneDNN to a plain s8 weight and
  forbids its internal blocked/VNNI-crosspacked DPAS layout. oneDNN's inference guide + IPEX QMatmul.h
  (L222-335) both do format_tag::any + a one-time cached weights_desc() reorder -- we don't. Top pp lever.
- oneDNN bundled = **v3.9.1** (not the v3.8 docs assumed) -> lever B5 (bump oneDNN) is MOOT.
- The fused act-quant uses sub_group_size=32; Xe2 native SG is 16 (minor note).
- **THE DECODE GAP:** at m=1 oneDNN runs the general JIT GEMV (jit:gemm:any, ~1/16 systolic util) on the
  plain row-major weight (`[k*N+n]` strides by N across cooperating lanes -> uncoalesced). ~67% of the
  40 t/s ceiling. Fix = offline column-contiguous reorder + a hand dp4a GEMV.
- **VERIFIED atom (2 sources):** int8 DPAS = `XE_8x16x32_S32S8S8S32_TT` (M<=8,N=16,K=32, s8s8->s32), B
  must be `layout::ext_intel_packed` (VNNI). M=1 variant `XE_1x16x32_...` exists but ggml uses dp4a
  (BW-bound -> joint_matrix buys nothing at m=1). codex critique pinned the packed-B SLM stride rule +
  the robust store-then-dequant epilogue.

VERDICT: plan delivered. Top-3 pp levers: (1) format_tag::any weights + cached reorder [~1day, highest EV];
(2) hand DPAS GEMM if oneDNN underperforms; (3) fold scales at quant + fused int32 epilogue. Top-3 tg:
(1) column-contiguous weight reorder; (2) hand dp4a GEMV (ggml mul_mat_vec_q idiom + SLM act reuse);
(3) FULL graph capture. FIRST EXPERIMENT (lead, ~1 day): ONEDNN_VERBOSE=2 microbench to MEASURE XMX util
+ which impl fires, then prototype format_tag::any on the W8A8 prefill GEMM (IPEX QMatmul recipe) -- a
near-zero-risk library win gated by the measurement that decides if a hand kernel is even needed. Both SYCL
skeletons are DRAFTS (not compiled); CAVEATS + a host-reorder/numpy-ref TODO ladder are in doc 10.

### 2026-06-20 -- [SYNTHESIS+EXEC] w8a8 deep-work: baseline MEASURED + 3 agent plans -> prioritized execution
- **Lead ran the int8_gemm_w8a8 baseline** (`w8a8/20_microbench_int8_gemm.sh`, w8a8/PROFILE_BASELINE.md):
  PREFILL **66-81% of 367 TOPS**; DECODE **50-93% of 608 GB/s** (wide-n 4096x11008 worst @ 50.5%, tall
  17408x5120 near-peak @ 92.9%). impl = `jit:gemm:any`, weight = plain row-major `s8::blocked:ab` (NOT
  VNNI/XMX-packed) -- the smoking gun confirming agent A. The gap is REAL + measured, not assumed.
- **3 deep agents (docs/kernel/10,11,12) delivered. Prioritized execution roadmap (by EV/effort):**
  1. **MTP head [TESTING NOW]** -- agent C: the gdn_attention MTP blocker is GONE in vLLM 0.23.0 (PR #43565
     `Qwen3_5MultiTokenPredictor`); the native head is a single-pass drafter that SURVIVES PIECEWISE (launch-
     mult ~1, unlike ngram) and accepts 75-88% vs ngram 16%. Serving Lorbus-27B-int4 `:v0230` GRAPH=1
     `--speculative-config {method:qwen3_5_mtp,num_speculative_tokens:3}` vs the 30.84 t/s MTP-off baseline.
     Go/no-go: accept-len >=3 + decode +>=20%. Potentially the biggest single decode win.
  2. **PP-1 format_tag::any [next deep kernel hand-tune]** -- onednn_ext.h:795 builds the weight md with
     explicit strides, never `format_tag::any`. Codex gave the recipe (wei_md any -> create pd -> query
     pd.weights_desc() -> reorder once if != plain, cache by weight ptr -> execute). Mirror IPEX QMatmul.h.
     Expect ~1.1-1.5x prefill + decode coalescing. ~1 day C++.
  3. **L1 wire the EXISTING fused rmsnorm+int8-quant [free-ish]** -- agent B: `layernorm_quant.cpp::
     rms_norm_dynamic_per_token_quant` ALREADY exists + registered (torch.ops._C) + int8 path, just NOT WIRED.
     Pure Python model-patch -> saves ~2.4 MiB + 80 launches/token. (Fix [-128,127]->[-127,127] sym clamp +
     1e-5 scale floor.) Build gotcha: flip BASIC_KERNELS_ENABLED ON in scripts/44.
  4. **L2 write `silu_and_mul_quant_int8`** (down_proj) + **PP-2 hand DPAS GEMM** (joint_matrix,
     XE_8x16x32_S32S8S8S32_TT, VNNI-B) only IF format_tag::any leaves XMX <60%.
- Principle banked (the *learning*): **hand the library `any`, not a fixed stride, so it picks the operand
  layout the systolic array wants; fuse the quant into the producing op to kill the f16 round-trip; and a
  single-pass drafter (MTP) is the only spec-decode that survives a partially-captured graph.** MTP now, PP-1 next.

### 2026-06-20 -- [RESULT] MTP head WORKS (accept 2.86) but net-NEGATIVE at N=3 under PIECEWISE (-37%); testing N=1
- First real MTP run on the B70. Served Lorbus-27B-int4 `:v0230` GRAPH=1 `--speculative-config
  {method:qwen3_5_mtp,num_speculative_tokens:3}`. **MTP head ENGAGED:** log shows `SpeculativeConfig(method=
  'mtp', num_spec_tokens=3)`, `Detected MTP model. Sharing target embedding + lm_head with the draft`,
  PIECEWISE captured 7 graphs (sizes incl the verify batch 1-32). Healthy.
- **[GOOD] the drafter is ACCURATE -- the "legs" exist:** `Mean acceptance length: 2.86`, per-position accept
  `0.837, 0.569, 0.455`, avg draft acceptance 62.1%. The Qwen3.6 MTP head produces high-quality drafts.
- **[BAD] net decode = 19.65 t/s vs 31.36 MTP-off = -37%.** Confirms agent C's blocker: under PIECEWISE the
  spec step pays (a) eager-attention verify over N+1 tokens, (b) **3 SEQUENTIAL MTP draft forwards** for N=3,
  (c) the Triton-disabled rejection sampler fallback -- together >> the 2.86-token accept saves. The
  acceptance DECAYS fast (0.84 -> 0.57 -> 0.46), so the 3rd spec token barely pays its draft+verify cost.
- **Hypothesis -> N=1:** 1 draft forward (vs 3) + 84% first-token accept + smaller verify (2 tokens) should
  have a far better cost/benefit. Testing num_speculative_tokens=1 now. If N=1 is also negative, MTP-positive
  needs the deeper fix: the Triton "0 active drivers" import-order fix (agent C doc 12) -> TRITON_ATTN (FULL
  capture, no eager-attn verify tax) + a fast Triton rejection sampler. **Net: MTP legs PROVEN (good accept);
  net-positive is gated on N-tuning and/or the Triton/FULL-capture fix -- a clean, diagnosed path for card #2.**
- **[N=1 result] 25.47 t/s (better than N=3's 19.65 but STILL -19% vs 31.36), accept 1.85 / 85% first-token.**
  So even the BEST case (1 draft, 85% accept) is net-negative -> the bottleneck is the per-step MACHINERY
  overhead (eager-attn verify + Triton-disabled reject-sampler fallback), NOT the draft count. MTP verdict
  COMPLETE: **the head is an excellent drafter (legs PROVEN: 85% @ N=1, 2.86 @ N=3), but net-positive decode on
  the B70 is gated on the Triton fix -> FULL capture (no eager-attn verify tax) + a fast Triton rejection
  sampler.** That is the deep MTP work for the card-#2 phase (or a focused single-card Triton "0 active
  drivers" import-order fix per agent C doc 12). GPU freed. Pivoting to the PP-1 int8-GEMM kernel hand-tune.

### 2026-06-20 -- [RESULT] PP-1 (format_tag::any weight reorder) IMPLEMENTED -> CORRECT but NO prefill win + a crash -> REVERTED
- Hand-implemented the headline int8-GEMM lever: `onednn_ext.h` wei_md -> `format_tag::any` (gated on the
  s8s8 dtypes) + `int8_gemm_w8a8.h` reorders the user weight ONCE into the pd-chosen blocked layout, cached by
  ptr (IPEX QMatmul.h pattern; codex-recipe). **Compiled clean, rebuilt (scripts/44), A/B-validated**
  (`w8a8/22_validate_pp1.sh`).
- **[CORRECT] the change works + is numerically exact:** ONEDNN_VERBOSE confirms the weight md is now
  `s8:a:blocked:ab:17472x1` (padded/blocked, NOT plain `ab`) with a ONE-TIME `reorder`; output fingerprints
  are BIT-IDENTICAL to baseline (k=4096 sum +1.665340e+04 both; k=5120 -7.676465e+02 both). The reorder is
  layout-only as designed.
- **[NEGATIVE] but it does NOT improve PREFILL** (the headline target): baseline 67.3/75.0% vs PP-1 68.5/74.9%
  of 367 TOPS = flat (within noise). The verbose shows oneDNN picked a blocked weight BUT kept the SAME
  `jit:gemm:any` impl -> format_tag::any did not unlock a faster kernel. **The decode "gain" in the first run
  (60->70%) was MICROBENCH NOISE** -- the +/-30% run-to-run noise (our own B1 meta-finding) reappeared: the
  "baseline" decode swung 60% -> 91% between runs. So no measurable win at either regime.
- **[BLOCKER] PP-1 also DEVICE_LOSTs on the k=17408 (MLP-down) shape** (the baseline handles it fine) -> a
  blocked-path bug for wide-k -> UNSERVABLE (a real model hits 17408x5120 every layer). REVERTED to pristine
  (prod `:int8` baked .so was never changed; only the host /src dev copy).
- **[LEARNING -- now 2 data points: B1 + PP-1] the "obvious" oneDNN library tweaks do NOT help int8 on the B70,
  because oneDNN v3.9.1's jit GEMM already handles the weight layout + zp internally.** B1 (drop src-zp) was
  perf-neutral (v3.12 caches the zp comp); PP-1 (format_tag::any) is perf-neutral + crashes. The REAL int8 win
  was graph capture (A1). **Verdict on the int8 GEMM: oneDNN is already near the practical ceiling (prefill
  67-80% of TOPS, decode 85-95%+ of BW in the good runs).** The only remaining KERNEL lever is a hand DPAS GEMM
  (PP-2, joint_matrix, 1-2 wk) for maybe ~1.2x -- low ROI vs effort. Better next levers: L1 (wire the EXISTING
  fused rmsnorm+int8-quant -- a different axis, the dispatch/BW round-trip) and, for decode, the MTP/Triton path.
  And: measure library tweaks at SERVE decode-t/s, never the noisy microbench.

### 2026-06-20 -- [FIX] Triton-XPU ENABLED via a 1-file sitecustomize shim (a documented blocker, now solved)
- Agent (doc 13) root-caused the "Triton is installed but 0 active drivers -> Disabling Triton" to triton's
  `is_active()` == `torch.xpu.is_available()`, gated per spawned process with an lru_cache. Fix: a
  `sitecustomize.py` that warms `torch.xpu.device_count()` at interpreter start in EVERY process
  (`/mnt/vm_8tb/b70/triton_shim`, injected via `PYTHONPATH`; `TRITONSHIM=1` knob in 30_serve).
- **[WORKS] Triton is now ENABLED:** served w8a8-14b on `:int8g` with TRITONSHIM=1 -> the "Disabling Triton"
  line is GONE (present in every prior run). So the shim resolves the S2 lru_cache timing-poison: Triton is
  live in the engine worker. A clean, low-risk, reusable fix to a standing blocker.
- **[STILL BLOCKED] FULL capture, for TWO independent reasons:** (1) `VLLM_ATTENTION_BACKEND=TRITON_ATTN` does
  NOT engage -- vLLM-XPU used flash-attn anyway (the attn-backend selector ignores it / TRITON_ATTN unwired on
  XPU); (2) even so, FULL capture dies on the same `sycl_ext_oneapi_work_group_scratch_memory ... not yet
  available with the SYCL Graph extension` (the toolchain limit -> needs oneAPI DPC++ 2026.0). So A2 (FULL
  capture) remains a TOOLCHAIN-gated future item, NOT a Triton-only fix.
- **[VALUE -> testing] the shim's payoff is the MTP rejection SAMPLER** (Triton-jit): with Triton live it should
  now be fast, removing ONE of the two MTP overheads (the other -- eager-attn verify under PIECEWISE -- remains
  until FULL capture). Re-testing 27B MTP N=1 + TRITONSHIM=1 vs the earlier -19% (25.47 t/s). If the fast
  sampler moves it toward/over breakeven, that is MTP partly unblocked on ONE card NOW.
- **[RESULT] MTP+shim = 25.53 t/s -- UNCHANGED from no-shim N=1 (25.47), still -19% vs MTP-off (31.36)**, same
  86.9% accept. So the now-fast Triton SAMPLER is NOT the MTP bottleneck. **CONFIRMED: the MTP cost is the
  EAGER-ATTENTION VERIFY** (N+1 tokens through eager attention under PIECEWISE) -- the head drafts accurately
  but the verify costs more than the 0.87-token accept saves. MTP-positive needs FULL capture (captures
  attention) -> blocked by the work_group_scratch SYCL-Graph toolchain limit (oneAPI 2026.0) + TRITON_ATTN
  unwired on XPU. **MTP verdict FINAL: head proven (legs); net-positive is TOOLCHAIN-gated, not Triton-gated.**
  The Triton shim is still a clean reusable fix (enables Triton for any future Triton op / sampler), just not
  the MTP unblock. Filed for the oneAPI-2026.0 / card-#2 phase.

### 2026-06-20 -- [SESSION WRAP] deep w8a8 + MTP night: what's the best single-card W8A8, honestly
- **The headline of the whole night: graph capture, not kernel hand-tuning, is the B70 win.** PIECEWISE capture
  ~2x'd int4 decode (w4a16 28->55, w4a8 17->48) and 4-7x'd the flagships (27B 7.8->30.8, 35B-MoE 7.9->56.8).
  The int8 W8A8 GEMM is ALREADY near the practical ceiling via oneDNN (prefill 67-80% of 367 TOPS, decode
  85-95% of 608 GB/s) -- two hands-on kernel tweaks (B1 drop-zp, PP-1 format_tag::any) were CORRECT but
  perf-neutral, proving oneDNN v3.9 already handles weight-layout + zp internally. So "best single-card W8A8" =
  oneDNN GEMM + PIECEWISE capture + fp8-KV; the kernel is not the bottleneck.
- **MTP:** head is an accurate drafter (proven legs); net-positive is toolchain-gated (eager-attn verify ->
  FULL capture -> oneAPI 2026.0). Triton-enable solved (shim).
- **Remaining levers, all high-effort or toolchain/card-#2-gated:** PP-2 hand-DPAS GEMM (1-2 wk, ~1.2x);
  FULL capture (oneAPI 2026.0) -> MTP-positive + spec-decode; L1 wire the existing fused rmsnorm (small);
  W8A8/27B dual-card quant (card #2). The high-EV single-card wins are BANKED; the rest needs the toolchain or
  the 2nd card. Honest, thorough, and every negative result documented as a learning.

### 2026-06-20 -- [C1 / LEVER C CLOSED] custom SYCL int4 GEMV is FUTILE -- oneDNN already meets/beats llama.cpp
- Ran the ladder's last open step (doc 04 step 4): bench llama.cpp's purpose-built int4 GEMV on the B70 as a
  BW-CEILING reference, to decide Lever C (the "biggest kernel win, 1-2 wk" custom SYCL GEMV).
- **llama-bench in `ghcr.io/ggml-org/llama.cpp:full-intel` SEGFAULTS on Battlemage** (exit 139 in 6s, right
  after the SYCL backend loads -- a device-init crash, not our config; -fa 0/1 both). So no FRESH direct number.
- **But the existing evidence settles it definitively:** (a) our prior working run (FINDINGS) = Qwen2.5-7B-Q4_K_M
  ~90 t/s decode = ~67% of 608 GB/s; (b) literature/01: llama.cpp's int8 (Q8_0) decode is a REGRESSION vs int4
  on Xe2 (ggml #21517) -- so no int8 GEMV reference to chase either; (c) OUR oneDNN+capture int4 decode = 48 t/s
  on Qwen3-14B-W4A8 = ~73% BW, int8 = 85-95% BW. **=> oneDNN+capture (73-95%) MEETS OR BEATS the only
  purpose-built Xe2 int4 GEMV (llama.cpp, ~67%).** A hand SYCL GEMV (Lever C) has NO meaningful headroom ->
  FUTILE. **Lever C CLOSED.**
- **=> ALL THREE of doc 04's levers are now resolved: A (capture) = THE WIN (done); B (oneDNN tweaks) =
  near-ceiling (B1+PP-1 perf-neutral); C (custom GEMV) = futile (oneDNN >= llama.cpp).** The single-card
  int4/int8 decode kernel campaign is COMPLETE -- the GEMM/GEMV is at the practical BW ceiling. The only real
  remaining upside is FULL capture (toolchain, oneAPI 2026.0) + the dual-card phase (card #2). Doc 04 ladder
  steps 4 + 5 done.

### 2026-06-20 -- [VALIDATED] fp8 KV cache works on the B70 (e4m3) -- a real serve win (capacity + small decode)
- The capstone (doc 14) CLAIMED "oneDNN + capture + fp8-KV" as the best stack -- but I had NOT actually tested
  fp8-KV. Closed the gap (validate-your-own-claims). Added a `KVDTYPE` knob to 30_serve.
- **`fp8_e5m2` is REJECTED** for quantized checkpoints: vLLM `attention.py:168` guards
  `if should_load_quant_weights(...) and kv_cache_dtype=="fp8_e5m2": raise` (e5m2 is unscaled). Use the SCALED
  **`fp8_e4m3` (alias `fp8`)** instead.
- **`fp8` (e4m3) WORKS:** served Qwen3-14B-W4A8 `:int8g` GRAPH=1 KVDTYPE=fp8 maxlen=8192 -> HEALTHY, PIECEWISE
  captured (0.93 GiB), `kv_cache_dtype=fp8` accepted. **Output COHERENT** (correct reasoning on a BW-decode
  prompt -- quality preserved). **decode = 49.8 t/s vs 48.2 fp16-KV at ~1801 ctx (+3.3%)** -- a small win at
  medium ctx (KV is ~10% of decode BW there), GROWS with context as the KV read becomes a larger BW fraction;
  plus **2x KV capacity** (halved KV -> ~2x max context / batch). NB cross-run delta is near the noise floor;
  the durable wins are the 2x capacity + the long-ctx scaling.
- **=> the capstone's fp8-KV claim is now VALIDATED (with the e4m3 caveat). 30_serve KVDTYPE=fp8 is the
  recommended long-ctx / high-batch serve knob.** Doc 14 + serve script updated.

### 2026-06-20 -- [DIAGNOSIS] w8a8 decode 26 t/s is REAL (61% BW) -- the int8 m=1 GEMM, not under-tuning; w8a8 = PREFILL champ
- Re-measured the HEADLINE model with the CURRENT best recipe (Qwen3-14B-W8A8 `:int8g` GRAPH=1 PIECEWISE +
  KVDTYPE=fp8): **decode = 26.09 / 26.11 t/s (2 runs) -- CONSISTENT with the old 26.7.** So the w8a8 decode is
  GENUINELY ~26 t/s; the old number was NOT under-tuned. Coherent (verified). prefill = 5508-5699 t/s.
- **The w8a8-vs-w4a8 decode gap is REAL and DIAGNOSED:** w8a8 26 t/s = 374 GB/s effective = **61.5% of 608**
  (14 GiB int8 weights); w4a8 48 t/s = 446 GB/s = **73%** (9.3 GiB int4). The int4 grouped weight-decompression
  GEMV achieves HIGHER effective BW at m=1 than the general int8 jit:gemm:any -- whose decode BW% is
  shape-sensitive (int8 microbench: wide-n MLP up/gate 5120->17408 = ~50%, tall down 17408->5120 = ~93%), so
  the w8a8 decode is dragged to ~61% by the wide-n MLP GEMMs. **PP-1 (format_tag::any) tried exactly this fix
  and was perf-neutral -> the wide-n int8 m=1 gap is a known oneDNN limit, not easily closed.** So w8a8 decode
  trails w4a8 on BOTH axes: more weight bytes AND lower m=1 GEMM efficiency.
- **But w8a8's STRENGTH is PREFILL: 5508 t/s > w4a8's 4953** (direct int8xint8 XMX, no int4 unpack). =>
  **the clean quant Pareto (Qwen3-14B, B70, capture+fp8-KV): W8A8 = PREFILL champion (5.5k t/s) but decode floor
  (26); W4A8 = balanced (4.9k pp / 48 tg); W4A16 = DECODE champion (55 tg) but no int8 prefill fast-path.**
  Pick by workload: prefill/batch-heavy -> W8A8; decode-heavy -> W4A16/W4A8; balanced -> W4A8. Doc 14 updated.

### 2026-06-20 -- [HEADLINE] one B70 serves Qwen3-14B-W4A8 at ~412 t/s @ 8 users (8.5x single-stream); 658 @ 16
- The single-stream 48 t/s BADLY understates serving capacity. Measured AGGREGATE decode throughput
  (`evals/orchestrator/concurrent_probe.py`, N parallel streams, w4a8 GRAPH=1 PIECEWISE + fp8-KV, maxseqs=16):
  | N (concurrent) | aggregate t/s | per-stream t/s | scaling |
  |----------------|---------------|----------------|---------|
  | 1  | 48.4  | 48.4 | 1.0x |
  | 2  | 109.8 | 54.9 | 2.3x |
  | 4  | 215.0 | 54.0 | 4.4x |
  | 8  | **411.8** | 52.0 | **8.5x** |
  | 16 | **658.3** | 42.0 | 13.6x |
- **WHY it scales: at decode (m=B) the GEMM reads the 9.3 GiB weights ONCE for all B sequences** -> the
  BW-bound weight read amortizes -> aggregate ~= B x single-stream until compute-bound. Near-linear to N=8
  (8.5x). **Per-stream latency even IMPROVES (48 -> 52-55 t/s) up to N=8** as the fixed per-token overhead
  (sampling/scheduling) amortizes; only at N=16 does per-stream drop to 42 (GEMM compute + capture captured
  only [1,2,4,8] so N=16 partially falls back).
- **=> the real "best w4a8 serve" headline: ONE B70 = ~412 t/s aggregate at 8 concurrent users, each still
  getting 52 t/s (excellent UX), scaling to 658 t/s at 16.** A strong serving card, not a single-stream toy.
  (w8a8 would scale the same way from its 26 t/s base -> ~200 t/s @ 8, still BW-amortized on 14 GiB.) Doc 14
  updated. NB to capture N>8 cleanly, raise cudagraph_capture_sizes past 8 (a free serving-capacity bump).

### 2026-06-20 -- [REFINED + HYPOTHESIS DISPROVED] B70 serving ceiling = ~1286 t/s @ 32 users; the N>8 dip is attention-KV, not capture
- Tested the "capture batch>8 is a free bump" hypothesis: added a `CAPSIZES` knob, served w4a8 capturing
  [1,2,4,8,16,32] (6 graphs, +1.0 GiB -> 1.96 GiB), re-swept. **HYPOTHESIS DISPROVED:** N=16 per-stream stayed
  42.2 t/s (NOT recovered to 52) -> the N>8 dip is NOT eager-fallback; capturing 16/32 changed nothing.
  | N | aggregate t/s | per-stream | note |
  |---|---------------|------------|------|
  | 8  | 411.5  | 51.7 | |
  | 16 | 658.3  | 42.2 | per-stream plateaus here (unchanged by capturing 16) |
  | 32 | **1286.2** | 42.6 | **26.6x single-stream**; aggregate still linear |
- **The real cause of the 52->42 per-stream step: the attention KV read SCALES with batch** (each seq reads its
  own KV; unlike the weight read which is amortized once). Past N~8 the KV read becomes a fixed per-stream tax
  -> per-stream plateaus ~42, but AGGREGATE keeps scaling linearly (42 x N). **=> serving ceiling ~1286 t/s at
  N=32**, which is also ~the KV-bound capacity max (32 x 0.5 GiB fp8-KV + 9.3 weights ~ 25 GiB of 32).
- **USEFUL NEGATIVE: capturing batch>8 costs ~1 GiB VRAM for ZERO throughput gain -> keep capture at default
  [1,2,4,8].** Corrected the doc-14 "free bump" note. The genuine serving headline: **one B70 = ~412 t/s @ 8
  users (52/stream, best UX) up to ~1286 t/s @ 32 users (42/stream) -- KV-bound, fp8-KV is what enables N=32.**

### 2026-06-20 -- [HEADLINE MODEL] w8a8 batched serving capacity measured: ~208 t/s @ 8 users, 364 @ 16
- Measured the W8A8 (the headline model) batched throughput (was only ESTIMATED before). Both models now done:
  | model | single | @4 | @8 | @16 | @32 |
  |-------|--------|----|----|-----|-----|
  | w4a8  | 48 | 215 | **412** (52/s) | 658 (42/s) | **1286** (42/s) |
  | w8a8  | 26 | 106 | **208** (26/s) | **364** (23/s) | KV-bound (~N=22 max) |
- **w8a8 scales even MORE linearly than w4a8**: per-stream FLAT at 26 t/s through N=8 (4.0x/7.9x ~perfect),
  only dipping to 23 at N=16. Why flatter than w4a8 (which improved 48->52)? At N=1 w8a8 is already more
  BW-bound (14 GiB weights dominate -> little fixed overhead to amortize), so batching just adds streams at
  the same rate. **w4a8 carries ~2x the aggregate at every batch** (smaller 9.3 GiB weights + higher base).
- **=> serving characterization COMPLETE for both headline models.** Best w8a8 serve = ~208 t/s @ 8 users
  (26/stream), 364 @ 16; best w4a8 serve = ~412 @ 8 (52/stream), 1286 @ 32. Pick w4a8 for throughput+decode,
  w8a8 for prefill+accuracy. Doc 14 updated with the dual-model table.

### 2026-06-20 -- [LONG-CTX] w4a8 decode degrades GRACEFULLY: 46 t/s @ 200 ctx -> 41 @ 15K (-12%); fp8-KV holds it
- Charted long-context decode (`evals/orchestrator/longctx_probe.py`, w4a8 GRAPH=1 PIECEWISE + fp8-KV,
  maxlen=16384). Decode reads the weights (fixed) + the WHOLE KV each step, so it degrades with context:
  | ctx | decode t/s | TTFT |
  |-----|-----------|------|
  | ~200   | 46.3 | 0.8s |
  | ~4000  | 44.1 | 1.1s |
  | ~8000  | 42.7 | 2.2s |
  | ~15000 | 40.6 | 4.7s |
- **Only -12% decode from 200 -> 15K tokens.** Why so graceful: the 9.3 GiB weight read dominates the
  per-step BW (fixed), and fp8-KV keeps the growing KV small (~1 GiB even at 15K single-seq, ~10% of budget).
  **=> B70 holds 40+ t/s decode at 15K context (w4a8)** -- strong long-context serving; fp8-KV is what keeps it
  graceful. TTFT scales with prefill (0.8 -> 4.7s; ~3180 t/s prefill at 15K, down from 4953 @ 1800 = attention
  O(n^2)). **Serving characterization now COMPLETE across all dimensions: latency, batch (both models), context.**

### 2026-06-20 -- [NEW DIRECTIVE] get 27B/35B W4A8 single-card loaded + optimize + MTP (in progress)
User: "fix 27B-W4A8 serve (4304 blocker)... apply 14b lessons" + "work on MTP, qwen3.6 takes to it better" +
"do what you can to get w4a8 loaded, optimize and mtp all 27b and 35b we can single-card load."

**MTP verdict (agent, doc 12 section G) -- DEFINITIVE, net-positive is FULL-capture-gated:** the 27B MTP -19%
@ N=1 is TWO causes, neither fixable on PIECEWISE: (a) under PIECEWISE all attention runs eager AND
`gdn_attention_core_xpu` is in the splitting-op list -> the ~30 majority GDN layers ALSO run eager in the
verify (my "GDN is cheaper" hope was WRONG); (b) a fixed per-spec-step machinery tax (prepare_inputs +
per-layer attn-metadata rebuild + dispatch + reject bookkeeping) that doesn't amortize -- proof: N=1->N=3 =
2.3x T for two CHEAP 1-layer drafter forwards. Net-positive needs FULL capture (FULL_DECODE_ONLY,
uniform_decode_query_len=1+N) -> same work_group_scratch/oneAPI-2026.0 + TRITON_ATTN-unwired wall (vLLM
#33341). **=> MTP filed for the FULL-capture phase; the head is a proven drafter but NOT shippable for
interactive decode now. Will still enable+measure per model (ready to flip when FULL capture lands).**

**27B-W4A8 serve fix -- 4 blockers solved + size fight (in progress):**
1. 4304 group-128: ONLY `model.visual.blocks.N.mlp.linear_fc2` (input 4304, /128=33.6) -> ignore it.
2. Config collapse (TypeError Qwen3_5Config vs Qwen3_5TextConfig): `fix_27b_vlm_config.py` grafts the base VLM
   wrapper config + the quantization_config -> arch Qwen3_5ForConditionalGeneration, model_type qwen3_5.
3. Ignore prefix: llmcompressor saved `model.layers.N.linear_attn` but VLM modules are
   `model.language_model.layers.N.linear_attn` -> patch ignore to regex (`re:.*linear_attn.*` etc.).
4. Load-time DEVICE_LOST (memory): byte-map showed the bulk is text-quant 17.5 GiB I8 (packs to ~8.75 int4 on
   load) + **GDN linear_attn 10.36 GiB BF16 (ignored = the killer)** + lm_head 2.37 + other 2.37 = 33G disk ->
   ~24-33 GiB GPU (incl int8->int4 load transient) -> OOM. Visual was a RED HERRING (negligible bytes).
   Fix: quantize the GDN too (re-quant3, ignore only lm_head+mtp+visual.fc2) -> ~14 GiB GPU. Coherence check
   PENDING (RTN on the linear-attn is the quality risk; if degraded, the bf16-GDN W4A8 is a 2-card model).

### 2026-06-21 -- [ROOT CAUSE + FIX] W4A8 "int-quantized" stores weights UNPACKED (2x); OFFLINE PRE-PACK = true 1-card
- **Why W4A8 disk >> W4A16 (user's question):** SAME 4-bit weights, different storage FORMAT. W4A16 =
  `pack-quantized` (int4 packed in int32, weight_packed I32 [out,in/8], ~0.5 byte/wt). W4A8 = `int-quantized`
  (4-bit stored UNPACKED as int8 [out,in], 1 byte/wt) -> ~2x disk. (compressed-tensors: activation-quantized
  schemes use int-quantized; weight-only uses pack-quantized.) The 14B is the same format (loads fine, 14G I8).
- **This unpacked storage is the WHOLE 27B single-card fit problem:** vLLM loads ALL the unpacked I8 to GPU
  (27B: 28 GiB), then XPUW4A8IntLinearKernel._pack_int4_weight packs to int4 ON LOAD -> the 28 GiB TRANSIENT
  hangs/OOMs the 32 GiB card. The RESIDENT packed size is only ~16 GiB (fits). So it is NOT inherently 2-card.
- **Fix = OFFLINE PRE-PACK** (`w4a8/offline_prepack_w4a8.py`): pack the I8 weights -> int32 [out,in/8] offline
  (byte-matching the kernel: +8, reshape [N,K/8,8], <<arange(0,32,4), sum), save a `-prepacked` model
  (is_prepacked_w4a8=true). Measured: 497 weights, 26.6 -> 14.7 GiB, model 27G -> 15G. Two env-gated
  (VLLM_W4A8_PREPACKED=1) patches mounted at serve (no rebuild): `w4a8/patches/compressed_tensors_w4a8_int.py`
  (create_weights makes the param int32 [out,in/8]) + `w4a8/patches/xpu.py` (kernel skips _pack_int4_weight).
  30_serve PREPACK=1 knob mounts them.
- **GOTCHA found:** do NOT pack vocab layers (lm_head/embed) -- they use VocabParallelEmbedding's own loader
  (expects unpacked) -> "size 5120 must match 640" crash. Script now skips them; and the QUALITY config keeps
  lm_head + GDN bf16 anyway (re-quant Qwen3.6-27B-W4A8-q: ignore lm_head+linear_attn+mtp+visual.fc2). Prepacked
  quality version targets ~24 GiB (text packed 8.75 + GDN bf16 10.36 + lm_head/other bf16) -> true 1-card.
  Next: prepack the quality re-quant -> graft -> serve PREPACK=1 -> coherence. Then GPTQ-W4A8 for best text.
- Recipe so far (for any quantized Qwen3_5 VLM serve): scripts/49 (SCHEME=W4A8 DATAFREE=1, ignore tuned for
  fit) -> fix_27b_vlm_config.py graft -> regex-ignore patch -> copy preprocessor_config.json -> 30_serve
  GRAPH=1 PIECEWISE KVDTYPE=fp8. Single-card-loadable targets: 27B (W4A8 if GDN-quant coheres) + 35B-A3B MoE.

### 2026-06-21 -- [WIN] PRE-PACK makes 27B-W4A8 a TRUE 1-card load (24.35 GiB, captured, healthy) -- + GDN-op gap
- Prepacked QUALITY 27B-W4A8 (GDN+lm_head bf16, Qwen3.6-27B-W4A8-q -> -prepacked, 33G->25G, 256 wts packed
  32.9->24.1 GiB). Served :int8g GRAPH=1 PIECEWISE KVDTYPE=fp8 PREPACK=1 NOMM=1 UTIL=0.90: **Model loading took
  24.35 GiB + 13.7s, Graph capturing finished (67s), Application startup complete, HEALTHY.** => THE PRE-PACK
  FIX WORKS -- no 28 GiB unpacked-I8 transient, no DEVICE_LOST/OOM/hang. The 1-card-load problem is SOLVED.
  (PREPACK=1 mounts the patched loader+kernel; NOMM=1 skips the VLM vision-encoder dummy-profiling crash.)
- **BUT crashes at DECODE: AttributeError _xpu_C has no attribute gdn_attention** -- the 27B gated-delta-net
  decode op is MISSING from our :int8g _xpu_C. Root cause: scripts/44 builds GDN_KERNELS_ENABLED=OFF (fast
  int8-only); the GDN source IS present (csrc/xpu/gdn_attn/, registered torch_bindings.cpp); CMake defaults
  GDN=ON. 14B (std attn) never needed it; 27B (GDN) does. FIX: rebuild _xpu_C with GDN_KERNELS_ENABLED=ON +
  XPU_SPECIFIC=ON, mount the .so, re-serve. In flight.
- **[DONE] GDN rebuild OK (gdn_attention registered True), .so mounted -> 27B-W4A8 FULLY SERVES 1-CARD:**
  rebuilt `_xpu_C.abi3.so` (84 MB) + `libgdn_attn_kernels_xe_2.so` (6 MB, sibling dep -- MUST mount both, via
  30_serve KERNEL_SO knob which mounts the .so + any sibling lib*.so). Served the prepacked quality model
  (Qwen3.6-27B-W4A8-q-prepacked) GRAPH=1 PIECEWISE KVDTYPE=fp8 PREPACK=1 NOMM=1 KERNEL_SO=<rebuilt>:
  **HEALTHY, 24.35 GiB 1-card, coherent, decode 20.9 t/s, TTFT 85.8 ms, prefill 2377 t/s @ 1801 ctx.**
  => TRUE 1-card quality (GDN+lm_head bf16) prepacked 27B-W4A8 ACHIEVED. (Decode 20.9 < W4A16-27B's 30.8 --
  likely the int8 act-quant + GDN decode overhead and/or the 2s capture being cache-reused; perf-tune later.)
  Full serve recipe: prepack (offline) -> graft -> regex-ignore -> processor -> 30_serve GRAPH=1 PIECEWISE
  KVDTYPE=fp8 PREPACK=1 NOMM=1 KERNEL_SO=<gdn .so>. NEXT: MTP test on this serve; GPTQ-W4A8 for best text.
- **[MTP on 27B-W4A8] doubly-blocked.** Served +MTP (qwen3_5_mtp N=1; needed UTIL=0.95 MAXLEN=2048 -- at 0.90
  it OOM'd: base 24.35 GiB + MTP head left no KV). Result: decode 17.99 t/s (-14% vs the 21 baseline),
  **Mean acceptance 1.00 / per-position accept 0.0% (177 drafted, 0 accepted).** vs the earlier Lorbus 27B
  (86.9% accept): the Lorbus uses an AutoRound CO-PACKAGED MTP head matched to its quant; OURS pairs the BASE
  bf16 MTP head with an aggressively RTN-W4A8 main model -> the main model's hidden states diverged enough that
  the head's drafts NEVER match the verify -> 0% accept. **=> two independent blocks: (1) 0% accept (head/quant
  mismatch -- needs GPTQ-W4A8 (less lossy, preserves hidden states) OR a co-calibrated MTP head); (2) even with
  high accept, net-negative on PIECEWISE (verify eager -- needs FULL capture, toolchain-gated).** So MTP value
  on the B70 needs BOTH GPTQ-quant AND FULL capture. The 1-card W4A8 SERVE itself stands (21 t/s, coherent).

### 2026-06-21 -- [*** MAJOR LEAD ***] FULL graph capture is UNBLOCKED on vLLM-XPU (docs 04/14 "blocked" is STALE)
Two read-only research agents (web+GitHub, no GPU) while GPTQ grinds. Headline: **the #1 remaining lever is
achievable TODAY, no toolchain wait.**
- **FULL capture via TRITON_ATTN.** vLLM PR #34482 (merged 2026-02-25, validated torch 2.11+xpu): TRITON_ATTN
  supports ALL cudagraph modes incl FULL; FLASH_ATTN is PIECEWISE-only AND is the silent default -> our serves
  never got FULL because we never flipped the backend. FIX (single-rank): `--attention-backend TRITON_ATTN`
  + `VLLM_XPU_ENABLE_XPU_GRAPH=1` + `-O.cudagraph_mode=FULL` (or FULL_AND_PIECEWISE), torch 2.11+ (PR #37947).
  PR #38193 made xpu-graph opt-in (needs a recent Intel driver). The oneAPI-2026.0 / intel-llvm PR #21029
  work_group_scratch lift (verified merged 2026-01-15) only matters for FULL-capturing the *flash-attn* SYCL
  FMHA kernels -- which vLLM hard-blocks anyway; the Triton path sidesteps it. (vLLM #26970 CLOSED -> PyTorch
  #162143 / PR #166285, torch 2.11 XPUGraph, DONE. IPEX archived 2026-03-30; graph capture now in torch core.)
- **EAGLE3/MTP (2nd agent):** EAGLE3 does NOT sidestep the verify penalty (same single-forward verify over the
  hybrid GDN target); the penalty is the eager verify, which FULL capture is exactly what fixes. `Ex0bit/
  Qwen3.6-27B-PRISM-EAGLE3` is REAL (1-layer Llama drafter, fc fuses target layers [1,31,60], full/+compressed/,
  validated on stock Qwen3.6-27B, chain accept tau~2.2; ships SGLang tooling, vLLM "compatible via full variant"
  but XPU-unproven). No net-positive EAGLE3/MTP on Battlemage exists publicly yet.
- **=> THE experiment when GPU frees:** serve W4A8 `--attention-backend TRITON_ATTN VLLM_XPU_ENABLE_XPU_GRAPH=1
  cudagraph_mode=FULL` (Triton shim from doc 13), single-rank -> measure (a) plain decode vs PIECEWISE (21 / 48),
  (b) MTP net (does captured verify flip it positive?). RISKS to validate on-GPU: (1) does TRITON_ATTN interop
  with our custom oneDNN INT8 W8A8/W4A8 linear path, (2) the driver-version requirement. Corrects docs 04(A2)/14.
- **[VERIFIED on :int8g, no GPU] lead is actionable on our existing image -- AND found the old-attempt bug:**
  vLLM 0.23.0 in :int8g HAS the TRITON_ATTN branch in platforms/xpu.py get_attn_backend_cls (lines 77-79, PR
  #34482 present) AND the `--attention-backend` CLI flag (arg_utils.py:905). BUT **VLLM_ATTENTION_BACKEND is
  ABSENT from envs.py** -> our 30_serve line 28 (`-e VLLM_ATTENTION_BACKEND=$ATTN`) was SILENTLY IGNORED, so the
  serve always fell back to flash-attn = PIECEWISE-only. THAT is why TRITON_ATTN "never engaged" in the earlier
  A2 attempt -- not a missing feature, a dead env var. FIX shipped: 30_serve now also emits `--attention-backend
  $ATTN` (CLI). FULL capture is thus testable on :int8g with NO rebuild: GRAPH=1 CGMODE=FULL ATTN=TRITON_ATTN
  TRITONSHIM=1, single-rank. Fires when GPTQ frees the GPU.

### 2026-06-21 -- [RESULT] FULL capture WORKS on :int8g (TRITON_ATTN) -- mechanism validated on 14B-W4A8
GPTQ deferred to dual-card (user). Killed it, tested FULL capture on 14B-W4A8-gptq (clean, no GDN). The
`--attention-backend TRITON_ATTN` CLI fix WORKED: log shows **"Capturing CUDA graphs (mixed prefill-decode,
FULL): 4/4"** (FULL, not PIECEWISE) -> the dead-env-var really was the only blocker. Loaded 9.3 GiB, captured,
healthy, COHERENT. **TRITON_ATTN interops with our oneDNN int4_gemm_w4a8 -- no errors.** Numbers (3 runs, vs
PIECEWISE/flash-attn baseline): decode **52.3 t/s (+8.5% vs 48.2)**, prefill **~2480 (-50% vs 4953)**, ttft 52ms.
- VERDICT: FULL capture's PLAIN-decode gain is modest (+8.5%) -- PIECEWISE already captured the GEMM (the +187%
  win); FULL only adds the attention dispatch, which is small. AND TRITON_ATTN halves prefill (its attention is
  slower than flash-attn's sycl-tla FMHA). So for plain serving FULL/TRITON_ATTN is ~a wash-to-loss. **The real
  value is the MTP unlock: FULL captures the spec-decode VERIFY, which was the eager cost that made MTP net-neg.**
- NEXT (testable NOW, no dual-card needed): serve the LORBUS 27B (AutoRound int4, the model with the PROVEN 86.9%
  MTP accept) on :int8g FULL+TRITON_ATTN+MTP -> does capturing the verify flip the PIECEWISE -19% to net-positive?
  This is the definitive MTP-on-B70 test (our W4A8 RTN has 0% accept so it can't answer it; the Lorbus can).

### 2026-06-21 -- [BLOCKER] FULL capture + MTP on GDN (Qwen3.6) hits a vLLM-XPU spec-capture bug
Served Lorbus 27B (proven 86.9% MTP accept) on :int8g FULL_AND_PIECEWISE + TRITON_ATTN + MTP + GDN(.so).
First error was trivial (compile_sizes=[1] padded to 2 under spec -> added COMPILESZ knob, set =2). Then the
REAL blocker: **`RuntimeError: spec_query_start_loc must have size [num_spec_decodes + 1]`** at engine init
(the cudagraph FULL-capture dummy-run). It's in vLLM's GDN backend spec metadata (vllm/v1/attention/backends/
gdn_attn.py: spec_query_start_loc shape [num_spec_decodes+1]) -- the FULL-capture dummy run builds the GDN
spec metadata with a mismatched size. So **FULL capture + spec-decode (MTP) on the GDN architecture is blocked
by a vLLM-XPU bug** (the "GDN only supports decode-only full cudagraph" caveat manifesting as a hard error for
the spec path). PIECEWISE+MTP works (the earlier 86.9%/-19% on :v0230); it's specifically FULL+spec+GDN that
breaks. => MTP-net-positive-via-FULL on Qwen3.6 needs a vLLM GDN-spec-capture fix (deep, not a config). The
FULL-capture MECHANISM itself is fine (14B proved it). PIVOT (per user "other single-card optims"): test FULL
capture on 27B-W4A8 PLAIN decode (no MTP, no spec path -> avoids the bug) -> does it beat PIECEWISE's 21 t/s?

### 2026-06-21 -- [VERDICT] FULL capture characterized + CLOSED: not worth it for plain serving; MTP value bug-gated
27B-W4A8-q-prepacked FULL (CGMODE=FULL, TRITON_ATTN, no MTP). First OOM'd KV at UTIL=0.90 MAXLEN=4096 (FULL
graphs + 24 GiB model left 0.25 GiB KV < 0.41 needed) -> memory-tight. Retried UTIL=0.93 MAXLEN=2048: HEALTHY.
Capture log: "(mixed prefill-decode, PIECEWISE): 4/4" + **"(decode, FULL): 3/3"** -> on GDN, FULL is DECODE-ONLY
(confirms the GDN cudagraph constraint; prefill stays PIECEWISE). Result: **decode 20.2 t/s (vs PIECEWISE 21 =
NO gain), prefill 901 (vs flash-attn 2377 = -62%).** => FULL is a CLEAR LOSS on the 27B.
- **FULL-capture FINAL VERDICT (across 14B + 27B):** the mechanism works (TRITON_ATTN + cudagraph_mode=FULL,
  the dead-env-var fix), but it is NOT worth it for plain serving: 14B standard-attn = +8.5% decode / -50%
  prefill (wash); 27B GDN = ~0% decode (decode-only FULL doesn't help the recurrent GDN decode) / -62% prefill
  / memory-tight. The killer is TRITON_ATTN's attention being much slower at PREFILL than flash-attn's sycl-tla
  FMHA. **=> KEEP PIECEWISE + flash-attn + fp8-KV as the best single-card serving stack** (unchanged from doc 14).
  FULL's only real prize was the MTP/spec verify-capture, and that is blocked by the GDN spec_query_start_loc
  bug. Net remaining spec-decode upside is UPSTREAM-gated (vLLM GDN-spec-capture fix) or watch-list (DFlash
  block-drafter). Corrects docs 04(A2)/14 "FULL blocked" -> "FULL reachable but not worth it for plain serving".

### 2026-06-21 -- [LEAD] Community got MTP working on B70 (4-card BF16); single-card port = real eng, uncertain payoff
User surfaced a public B70 run: Qwen3.6-27B BF16, TP=4 on 4x B70, image intel/llm-scaler-vllm:0.14.0-b8.3,
decode 54.2 / prefill 2100, MTP num_spec=5 mean-accept 4.04 (88.9% @ spec=3). "Unblocked from userspace:
vllm_xpu_kernels v0.1.9 + qwen3_5.py spec-wiring (vLLM #43565) + Half-KV." Agent verified the recipe:
- **PR #43565** ("[XPU] support MTP of gdn attention", MERGED 2026-05-29) -- patches _xpu_ops.py/qwen3_5.py to
  FORWARD spec metadata (num_spec_decodes, spec_query_start_loc, spec_token_indx, spec_state_indices, num_accepted)
  into gdn_attention. THIS IS THE FIX for our exact spec_query_start_loc bug.
- **HARD BLOCKER:** needs vllm_xpu_kernels >= v0.1.9 (#368 Xe2-MTP-for-QWEN + #344 GDN padded-dim; v0.1.10 adds
  #411 >=32K NaN fix). Our b8.3.1 image ships **0.1.8.dev0** -- its baked gdn_attention op has NO spec args
  (ABI-verified), and Intel's qwen3_5.py still has `raise NotImplementedError(...spec_sequence_masks...)`. So
  b8.3.1 CANNOT do MTP as-shipped. Half-KV = `--kv-cache-dtype fp8` (not a special flag).
- **W4A8 custom kernel is NOT in llm-scaler** -> the single-card MTP vehicle is the Lorbus W4A16 (standard kernel,
  native MTP head, 86.9% accept). Reproduce needs a DERIVED image: b8.3.1 + pip install v0.1.10 wheel + apply the
  #43565-equiv patch to Intel's qwen3_5.py + debug the Intel-ESIMD-vs-upstream-wheel ABI coexistence (top risk).
- **PAYOFF CAVEAT (key):** their 54.2 was a 4-CARD TP=4 aggregate (4x bandwidth); the recipe runs --enforce-eager
  (no capture). On 1 card, eager+MTP would very likely decode SLOWER than our existing PIECEWISE-no-MTP (Lorbus
  30.8, W4A8 21). And spec-decode is a single-user-latency feature (doc 17), not throughput. => single-card MTP
  is real integration eng for an uncertain/likely-net-negative-vs-PIECEWISE result. DECISION PENDING (user):
  attempt the MTP derived-image engineering, OR pivot to deeply focus on 35B-MoE W4A16 (user's stated fallback).

### 2026-06-21 -- [BAILED, precise blocker] single-card MTP on b8.3.1 = torch 2.10-vs-2.11 ABI split
User chose the time-boxed MTP attempt. Built artifacts (agent): /mnt/vm_8tb/b70/mtp_patch/ (v0.1.10 wheel,
patched qwen3_5.py forwarding #43565 spec args, Dockerfile). Build + empirical probes revealed a hard split
(b8.3.1 runtime torch = **2.10.0+xpu**):
- **vllm_xpu_kernels v0.1.10** HAS the gdn_attention spec args (num_spec_decodes, spec_query_start_loc,
  spec_token_indx, spec_state_indices_tensor, num_accepted_tokens) BUT is built against torch 2.11 ->
  `ImportError: _xpu_C.abi3.so undefined symbol _ZNR5torch7Library4_def...RegisterOrVerify` on torch 2.10.
- **vllm_xpu_kernels v0.1.9** LOADS on torch 2.10 BUT its gdn_attention schema has **NO spec args** (HAS_SPEC
  False, identical to the baked 0.1.8) -> the #43565 patch has nothing to forward into.
- => NO wheel is both torch-2.10-compatible AND spec-capable. The community b8.3 + v0.1.9 run must have used a
  torch-2.11 base (or a custom-built wheel). Intel's ESIMD/int4 ops are a separate package (NOT the blocker --
  the agent's ABI-coexistence call was right; the blocker is torch core ABI, not Intel ops).
- **CLEAN PATHS (future, not time-boxed):** (a) a torch-2.11 llm-scaler image (b8.4+/newer) + v0.1.10 baked
  -- watch Intel releases; (b) build v0.1.10's spec gdn_attention from source against torch 2.10 (~1-2h, like
  our earlier GDN rebuild; uncertain the spec code compiles on 2.10). Payoff still uncertain (single-card
  eager+MTP likely < PIECEWISE 30.8; the community 54.2 was a 4-card aggregate; spec is a latency niche).
- **DECISION: bail per user's condition -> pivot to deeply focus on 35B-MoE W4A16.** Artifacts kept in
  mtp_patch/ for the future torch-2.11 path.

### 2026-06-21 -- [CHARACTERIZE] 35B-A3B MoE W4A16 full single-card profile (best stack + fp8-KV)
Served Intel_Qwen3.6-35B-A3B-int4-AutoRound on :v0230moe GRAPH=1 PIECEWISE KVDTYPE=fp8 DTYPE=float16 UTIL=0.90
MAXLEN=8192. HEALTHY (19.6 GiB load, captured 57s). Coherent ("Mercury, Venus, Earth, Mars"). Numbers:
- **Decode 65.25 t/s single-stream** (vs the 56.8 PIECEWISE/fp16-KV baseline = **+15% from fp8-KV**). The MoE
  gets a BIGGER fp8-KV decode win than dense models: A3B activates only ~3B/token so the active-weight read is
  tiny -> the KV read is a relatively larger fraction -> halving it (fp8) helps more. 35B-A3B is now the fastest
  single-card decode, period. Prefill 1623 t/s, TTFT 121 ms @ 1801 ctx.
- **Batch throughput PLATEAUS ~206 t/s @ N>=8** (N=8: 206, N=16: 207, N=32: 203). The MoE does NOT scale linearly
  like the dense 14B-W4A8 (1286 @ N=32): at batch N the UNION of routed experts grows -> per-step expert-weight
  read rises -> aggregate saturates once the union ~= all 256 experts (doc-17 MoE behavior, confirmed). **=> the
  35B-A3B is a SINGLE-USER-LATENCY champion (65 t/s), NOT a high-concurrency throughput one (use the dense 14B
  W4A8 for aggregate: 1286 vs 206).** Clean serving-profile split.
- Long-ctx not yet measured: MAXLEN=8192 too low, and the raw-completions longctx_probe returned 0 tok on this
  reasoning model (needs higher MAXLEN + chat endpoint). TODO: re-serve MAXLEN>=32K for the long-ctx curve.
- NEXT: the sub-optimal Triton MoE-GEMM config (E=256,N=512,int4_w4a16) -- agent assessing whether tuning it
  helps (likely prefill, not the BW-bound A3B decode).

### 2026-06-21 -- [ASSESS] 35B MoE Triton-config tuning = PREFILL-only, GPU-hours, capped -> low-ROI (like GPTQ)
Agent verified (live image + GitHub): the missing config `E=256,N=512,device_name=Intel(R)_Graphics_[0xe223],
dtype=int4_w4a16.json` is real (only NVIDIA/AMD configs exist) -> default fallback = the "sub-optimal" warning.
- **DECODE (65 t/s) is BW-MAXED, tuning won't move it:** at bs=1 the Triton `fused_moe_kernel_gptq_awq` uses a
  FIXED {BLOCK_SIZE_N:32,K:64} tile (get_moe_wna16_block_config, num_valid_tokens//top_k==1) + is weight-DRAM-
  bound; block/warp/stage choices change launch overhead, not byte movement. Confirmed Triton is the hot path
  on XPU (should_moe_wna16_use_cuda gates on is_cuda()=False -> always Triton).
- **Tuning helps PREFILL/large-batch (M>=128):** the compute-bound regime where BLOCK_M/num_warps/GROUP_SIZE_M
  matter -- could lift the 1623 prefill + the ~206 batch plateau.
- **But low-ROI:** (1) benchmark_moe.py is CUDA-hardcoded (device="cuda", torch.cuda.CUDAGraph, Ray num_gpus)
  -> needs an XPU patch before it runs; the image DOES carry an int4 patch (--dtype int4_w4a16 + a
  Qwen3_5Moe branch). (2) the sweep is GPU-hours (18 buckets x 256 experts). (3) Intel Triton's int4-dequant-
  in-tl.dot is a known ~2x-slow spot (intel-xpu-backend-for-triton #4327) -> even a tuned Triton config leaves
  perf on the table vs a native SYCL grouped-GEMM. RECIPE (future GPU job): patch cuda->xpu + drop CUDAGraph,
  then `benchmark_moe.py --model <35B> --dtype int4_w4a16 --tp-size 1 --tune --save-dir DIR` (tp=1 REQUIRED ->
  N=512); pickup via `VLLM_TUNED_CONFIG_FOLDER=DIR` at serve (checked before built-in configs/).
- **DEEPER lever (bigger potential):** the native SYCL grouped-GEMM MoE path (xpu_moe.py / vllm-xpu-kernels)
  could beat Triton's slow int4-dequant -- but our INC-patched :v0230moe forces the Triton path. Switching MoE
  paths is a bigger change; watch-list.
- **VERDICT: 35B deep-focus largely DONE.** Decode-maxed (65, single-card champ), batch/prefill characterized;
  the only optimization (config tune) is a prefill-only GPU-hours job capped by the Triton int4 slow path ->
  same low-ROI profile as the deferred GPTQ. Remaining minor gap: long-ctx curve (re-serve MAXLEN>=32K).

### 2026-06-21 -- [CHARACTERIZE done] 35B-A3B MoE long-ctx decode is DEAD-FLAT (~64.5 t/s, 2K-7K) -- best long-ctx profile
Chat-based long-ctx probe (evals/orchestrator/longctx_chat_probe.py -- the raw-completions longctx_probe returns
0 tok on reasoning models, so wrote a chat-endpoint variant). On the current MAXLEN=8192 fp8-KV serve:
- ctx~2000 decode 64.7 t/s (ttft 732 ms), ctx~4000 64.6 (1109 ms), ctx~7000 64.3 (1744 ms).
- **Decode degrades only -0.6% over 2K->7K = essentially FLAT.** The A3B activates only ~3B/token (tiny weight
  read) + fp8-KV halves the KV read -> decode stays WEIGHT-bound, not KV-bound, even at long ctx. This is the
  best long-context decode of any single-card config. TTFT grows linearly with prefill (expected). Trend predicts
  it stays ~flat well past 8K (the 256K-native model should sustain 60+ t/s decode at large ctx; >8K needs a
  MAXLEN>=32K re-serve to confirm -- predicted-flat, deprioritized).
- **35B-A3B MoE deep-focus COMPLETE.** Full profile: decode 65 (flat to 7K), batch ~206 plateau (MoE union),
  prefill 1623, TTFT scales, coherent, fp8-KV is a +15% decode win. THE single-card latency champion. Only
  optimization left = the prefill-only Triton config tune (low-ROI, GPU-hours) + the deferred dual-card W8A8/GPTQ.
- Re-verified the AutoRound W4A8-export block on BOTH auto_round 0.13.1 (latest pip) AND `main` (0.14.0-dev,
  unreleased). STILL BLOCKED: `formats.py::LLMCompressorFormat.check_and_reset_format` keeps the same hard
  assertion `bits==8 and group_size==-1 and sym and act_bits==8` for any int8-dynamic-act scheme; W4 (bits=4)
  -> AssertionError. On main, `support_schemes` has no W4A8 entry and `is_wint8aint8` requires weight bits==8,
  so the flexible `construct_ct_scheme()` (which COULD emit W4A8) is never reached. No pip release fixes it.
- Graft option (b) is NOT viable: compressed-tensors W4A8 expects `weight_packed` = UNPACKED int8 [out,in]
  (the XPU kernel re-packs to int32 itself); auto_round/gptq packs int32 `qweight` with different tensor
  names -> vLLM weight-loader mismatch. => W4A8: STILL BLOCKED, ship GPTQ-W4A8 (14B already 0.872/0.835).
- Confirmed our XPU path is NOT hit by upstream vLLM #38064 (W4A8-INT silently runs W4A16 -- Marlin/CUDA only):
  XPUW4A8IntLinearKernel.apply_weights explicitly calls dynamic_per_token_int8_quant_ref + int4_gemm_w4a8 ->
  real int8 acts (matches the 14B +51% prefill). So our compressed-tensors W4A8 GPTQ checkpoints are genuine.
- AutoRound W8A8 export IS supported (the one int8-dyn-act scheme the exporter allows: INT8_W8A8 = per-channel
  int8 w + dynamic per-token int8 act). Wrote the exact 27B command (hand-built QuantizationScheme bits=8
  group_size=-1, layer_config forcing visual/mtp/linear_attn/lm_head to 16-bit, device_map=xpu,
  quantize_and_save format=llm_compressor) -> serve on :int8g after fix_27b_vlm_config.py graft. GPTQ-W8A8
  (scripts/49 default) is the proven low-risk fallback. NOTE: no image bakes auto-round/llmcompressor ->
  recipe pip-installs at runtime; compressed-tensors 0.17.0 + vllm 0.23.0+xpu present on all images.
- 35B-A3B int8-act MoE: NO-GO confirmed. W4A8-int8 MoE oracle raises NotImplementedError on XPU (CPU_INT4
  only). NEW detail: W8A8-int8 MoE oracle -> ONLY `TritonExperts` (no platform gate but no XPU oneDNN int8
  expert); `experts/xpu_moe.py` XPUExperts subclasses cover fp8/mxfp8/blockfp8/int4-WNA16/mxfp4 -- NO int8.
  So W8A8 MoE on XPU would run on (flaky) Triton-XPU, not the XMX systolic path -> no speed win. Plus no bf16
  35B source on host. Keep 35B as W4A16-int4 (56.8 t/s captured). A real fix needs a new XPUExpertsInt8 +
  is_int8 SYCL fused-expert kernel (large, out of scope). DELIVERABLE: docs/kernel/15_autoround_w4a8_w8a8_recipes.md.

### 2026-06-21 -- [RESULT] 27B w4a16 CAPTURED concurrency sweep (the missing concurrent curve) + 256k-ctx fit limit
The 30.84 t/s 27B figure was single-stream only; no captured concurrency data existed (RESULTS.md only had an
old EAGER C4 datapoint). Filled it via `scripts/56_27b_conc_campaign.sh` (one gpu-run lease, ~47 min):
Qwen3.6-27B int4 AutoRound (= w4a16), `:v0230` GRAPH=1 PIECEWISE CAPSIZES=1,2,4,8,16,32,64 NOMM=1 UTIL=0.92
MAXSEQS=64, fp16 KV; `vllm bench serve` random 512/128 --ignore-eos. Served id verified qwen36-27b-int4; 7/7
graphs captured; model load 16.69 GiB. Two configs:
- Normal ctx (8k): C1 28.1 agg / 30.9 per-stream, C2 52.0/29.3, C4 87.8/26.7, C8 134.3/21.7, C16 178.3/14.5,
  C32 216.7/8.4, C64 234.7/6.7 (TTFT 0.44s -> 15.1s). Aggregate max ~235 @ C64; practical knee ~217 @ C32.
  C1 bench (30.9) == perf_probe 30.84 -> capture validated end-to-end.
- Big ctx: 262144 (256k) REJECTED at engine init -- "16.2 GiB KV needed for one max-len seq > 8.31 GiB
  available; estimated max model length 133120". Auto-fell-back to 131072 (128k), served fine; sweep
  near-IDENTICAL to 8k (C2 51.3, C4 87.6, C32 215.6, C64 232.9).
- VERDICT: (1) 27B w4a16 concurrent aggregate tops ~235 t/s but GDN/linear-attn batches sublinearly --
  per-stream drops below single-stream past C8; for latency-sensitive serving stay <=C4. (2) Context window is
  throughput-NEUTRAL (KV pool = f(util), not max-len). (3) Max single-card context at fp16 KV is ~133k; >128k
  needs KVDTYPE=fp8_e5m2. Banked: FINDINGS concurrency section + docs/SERVING.md. CSVs: results/sweep_27b-w4a16-cap-*.csv.
- Scope note: swept w4a16 only (user priority). 27B w4a8 (prepacked, 20.9 t/s captured, needs PREPACK +
  rebuilt GDN .so + fp8 KV) left un-swept; its serve recipe is now documented in docs/SERVING.md (secondary).

### 2026-06-21 -- [DOWNLOAD] DJLougen/Qwable-5-27B-Coder BF16 source -> 8TB SSD (quant target for 2-card)
Fetching the Qwable-5-27B-Coder finetune as a future quant source (w4a16 / w4a8 / w8a8 once the 2nd B70 lands).
- Config -> public Apache-2.0 repo, 28B, BF16, ~55.6GB / 15 safetensors shards. `config.json` verified:
  `model_type: qwen3_5` (qwen3_5_text: hidden 5120, 64 layers, max_pos 262144) -- SAME arch family as our
  existing Qwen3.6-27B quants, so the 27B W4A16/W4A8/W8A8 recipes carry over. `chat_template.jinja` shipped
  (grabbed via `*.jinja` pattern -- matters for the coder/tool-calling path).
- Command -> `scripts/57_download_qwable27b.sh`: DETACHED named container `qwable27b_dl` (python:3.11 +
  huggingface_hub), `snapshot_download` to `/mnt/vm_8tb/b70/models/DJLougen_Qwable-5-27B-Coder`, resumes from
  `hf_cache` on relaunch. Pure disk I/O -> NO gpu-run lease (no GPU touch). Disk: 6.4T free, ample.
- Result -> launched, config + 15 shard locks present, Xet chunked pull in progress. Check:
  `ssh root@192.168.10.5 'docker logs -f qwable27b_dl; du -sh /mnt/vm_8tb/b70/models/DJLougen_Qwable-5-27B-Coder'`
- Verdict -> download underway; quant deferred until 2nd card. Next: quant to W4A16 (priority), then W4A8 / W8A8
  reusing scripts 40/43/49/54 recipes against this source dir.

### 2026-06-21 -- [OK] SECOND B70 INSTALLED -- dual-card bring-up: both cards compute-usable, TP=2 serves
THE 2ND CARD LANDED. Host rebooted (uptime ~7 min when checked; the three Unraid app containers were "Up 5 min").
Bring-up checks, in order:
- **PCI enumeration:** both Battlemage G31 [Arc Pro B70] present -- `0a:00.0` and `44:00.0`. /dev/dri has card0+card1,
  renderD128+renderD129 (4 nodes). Qwable download container died on the reboot (Exited 137, 38G of ~56G on disk)
  -> `docker start qwable27b_dl` resumed it (pure disk, no GPU lease).
- **Compute-usable (the real test, NOT just PCI):** inside `vllm-xpu-env:v0230`, `sycl-ls` shows TWO Level-Zero
  GPUs `[level_zero:0]` + `[level_zero:1]` (both Intel Graphics [0xe223]) AND two OpenCL GPUs. So the oneAPI runtime
  vLLM actually uses sees both cards. (`xpu-smi` not in the image.) => TP=2 viable. [Phase C of MTP_TODO unblocked.]
- **Single-card smoke post-reboot:** Qwen3-0.6B TP=1 via 43_serve_multi.sh -> HEALTHY, model load 1.12 GiB/9s,
  KV 26.66 GiB, coherent. Stack survived the reboot + card add.
- **TP=2 plumbing (first dual-card serve on this machine):** Qwen3-0.6B TP=2 -> world_size=2, rank0+rank1 BOTH
  assigned (TP rank 0/1), backend=xccl, oneCCL came up with the Battlemage stability env. Model SHARDED:
  each card loaded **0.57 GiB** (= half the 1.12 single-card) -> real tensor split, not a replica. KV pool
  **26.77 GiB**, max concurrency **122x** (2x the single-card 60x -> both cards' KV pooled). Generated correct
  primes "2,3,5,7,11" -> coherent across cards. `system_fingerprint: vllm-0.23.0-tp2`.
- 43_serve_multi.sh was staged locally for "when card #2 arrives" but never synced to the host -> copied it to
  the host root. It carries the #41663 Battlemage env (CCL_ENABLE_SYCL_KERNELS=0, CCL_TOPO_FABRIC_VERTEX_
  CONNECTION_CHECK=0, SYCL_UR_USE_LEVEL_ZERO_V2=0, CCL_ATL_TRANSPORT=ofi, spawn workers, --distributed-executor mp).
- **Added a TP knob to 30_serve_w4a8_graph.sh** (the captured-serve path): `TP=` (default 1, backward-compatible).
  TP>1 -> `--tensor-parallel-size $TP --distributed-executor-backend mp`, the multi-GPU CCL env, both cards
  exposed (no ZE_AFFINITY_MASK pin), shm 32g; +CCL_TOPO_P2P_ACCESS=0 +CCL_ZE_IPC_EXCHANGE=pidfd. Original backed
  up to `.bak`. So we can now do CAPTURED TP=2 (the #41663 stable stack keeps XPU graph ON, so capture should work).

### 2026-06-21 -- [WIP] Dual-B70 PCIe/NUMA topology -- the multi-GPU comms bottleneck (idle reads x1, confirming under load)
For TP=2 the all-reduce path is what matters. B70 has NO usable GPU P2P (confirmed via research, below) -> every
TP all-reduce round-trips **GPU -> host RAM -> GPU over PCIe**. So PCIe link state + NUMA placement IS the bottleneck.
- **NUMA:** both cards `numa_node=-1`; host is Threadripper 1950X reporting a SINGLE NUMA node (node0 CPU 0-31,
  UMA/Distributed BIOS mode). So host-staging does NOT cross a NUMA boundary -- one less penalty. (The 1950X is a
  2-die MCM; its inter-die Infinity Fabric still bounds host-mem BW, but no explicit NUMA hop.)
- **PCIe tree:** each card sits behind its own switch: CPU root `00:03.1` (Gen3 x16) -> switch upstream
  `08:00.0`/`42:00.0` (cap 32GT/s=Gen5 x16, but TRAINED to 8GT/s=Gen3 x16 to the CPU) -> downstream port
  `09:01.0`/`43:01.0` -> card `0a`/`44`.
- **[!!] CONFIRMED: links are STUCK at Gen1 x1 (2.5 GT/s x1, ~250 MB/s) even under sustained load.** Rapid
  sysfs polling (200 samples / ~80s) during the eager TP=2 warmup+sweep -> EVERY sample 2.5 x1 on BOTH cards,
  never ramped. NOT idle ASPM (that ramps under load): even BULK weight-load ran at ~220 MB/s/card (8.42 GiB in
  38s = Gen1-x1 rate). max_link_speed AND current both read 2.5 x1 at the card AND its switch downstream port
  (`09:01.0`/`43:01.0`), while the switch->CPU upstream is Gen3 x16. => the switch DOWNSTREAM port to each card
  trained to Gen1 x1. Signature of **x1 PCIe risers / a mining-style PCIe-switch board** (each card behind a
  switch with a single downstream port; matches Puget's "riser was the culprit, direct slots fixed it").
- **THIS is the dominant multi-GPU bottleneck on our rig.** No GPU P2P -> TP all-reduce round-trips through host
  RAM over PCIe, and that PCIe is Gen1 x1 (~250 MB/s) = ~256x slower than a PCIe5 x16 link. Single-card decode is
  unaffected (weights already resident; ~30.8 t/s captured stands). But ANY TP=N pays per-layer all-reduce at
  250 MB/s. **ACTIONABLE: move the cards to direct Gen3+ x8/x16 slots or swap the x1 risers/switch -> would
  transform TP=2 from a curiosity into a real speedup.** Until then, TP=2 is a CAPACITY tool (fit bigger models /
  bigger KV), not a throughput tool. [Verify: is this a mining riser/switch board? check the physical PCIe wiring.]

### 2026-06-21 -- [REF] Community research synthesis: multi-B70 is a CAPACITY play, not a SPEED play
Two web-research passes (P2P/oneCCL + community multi-B70 runs). Findings that shape the campaign:
- **No usable GPU P2P on B70.** vLLM #41663's host check shows `p2p_access:0`; direct peer copies trigger PCIe
  RxErr -> engine reset -> deadlock. TP all-reduce goes via Unified Shared Memory through HOST RAM (set
  `CCL_TOPO_P2P_ACCESS=0` for the USM path). No XeLink/NVLink equivalent. Comms are PCIe-bound.
- **vLLM #41663 is EXACTLY our hardware** (2x B70, 8086:e223, PCIe-switch, x8 each, no XeLink): TP=2 GP-fault +
  Xe BCS engine reset during ProcessGroupXCCL init. The ONE load-bearing fix is **`CCL_ENABLE_SYCL_KERNELS=0`**
  (default 1 -> crash). The stable stack also keeps `VLLM_XPU_ENABLE_XPU_GRAPH=1` (graph ON) -> ~362 tok/s @ C50.
  Rejected non-fixes: `CCL_ALLREDUCE=ring` (~0.5 tok/s!), eager+graph-off (~0.5 tok/s). We already set the fix.
- **TP on Arc HURTS single-stream decode, helps aggregate/prefill.** Measured comms-bound evidence: StorageReview
  8x B60 GPT-OSS-20B batch=1 TP=1 49.2 -> TP=8 22.8 tok/s (2.2x SLOWER going wide); TP=4 626 > TP=8 512 aggregate.
  Puget 4x B70: Qwen3.6-27B 13.1(1u)->50.4(4u)->95.9(8u); 35B-A3B 16.3->63.7->122. Level1Techs dual-B70 TP=2 FP8:
  Qwen3.5-27B 13.25 single / 97.84 @ C8; Qwen3-30B-A3B MoE FP8 TP=2 = 912 tok/s aggregate. => expect our TP=2
  single-stream 27B to be <= the TP=1 30.84, with aggregate competing only at concurrency (where pooled KV helps).
- **NO public clean TP=1-vs-TP=2 same-model decode comparison on 2-card Arc exists** -> our campaign 58 generates a
  novel datapoint.
- **[!] The "4xB70 Qwen3 MTP, accept 4.04, decode 54.2" benchmark that MTP_TODO is premised on was NOT found in any
  public source.** The agent flags the loose "54 tok/s" as likely a single-card 35B-A3B llama.cpp number, conflated.
  -> Treat the MTP external reference as UNVERIFIED; if we pursue MTP, our own numbers are the only ground truth.
- **Multi-XPU quantization (AutoRound/GPTQ) has ZERO public precedent** -- every multi-card example is CUDA
  (`CUDA_VISIBLE_DEVICES`, no ZE_AFFINITY_MASK/xpu:N equivalent). AutoRound uses its OWN device_map ("0,1,..");
  an `xpu:0`/`xpu:1` per-layer dict is plausible-but-unverified. So the "try 2x-XPU AutoRound" task is genuinely
  experimental. Standard practice = quantize on CUDA/CPU, serve on Arc (output is HW-independent).
- llm-scaler (`intel/llm-scaler-vllm:0.14.0-b8.x`) is the de-facto multi-GPU path and recommends `mp` executor for
  Battlemage (we use it). Extra env worth trying: `CCL_ZE_IPC_EXCHANGE=pidfd`, `ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE`,
  `ZE_AFFINITY_MASK=0,1`. Sources banked in FINDINGS.

### 2026-06-21 -- [WIP] TP=2 27B captured concurrency campaign LAUNCHED (vs banked TP=1 curve)
scripts/58_tp2_campaign.sh under one gpu-run lease: Qwen3.6-27B int4 AutoRound (=w4a16), :v0230, GRAPH=1 PIECEWISE
TP=2 CAPSIZES=1,2,4,8,16,32,64 NOMM=1 UTIL=0.92, fp16 KV; sweep CONC 1..64; logs PCIe link state before/under/after.
Comparing against the BANKED single-card TP=1 captured curve (C1 30.9 per-stream / 28.1 agg ... C64 234.7 agg, 6.7
per-stream).
- **[!] CAPTURED TP=2 IS BLOCKED -- precise novel blocker.** Engine init reached graph capture then the worker
  died: `oneCCL: ccl_allreduce_impl: |CCL_SYCL| sched algorithms do NOT support sycl_graph recording, please use
  sycl_algorithms`. The contradiction: vLLM #41663 requires `CCL_ENABLE_SYCL_KERNELS=0` (the L0 "sched" path) for
  STABLE init -- but recording an all-reduce INTO a SYCL/XPU graph requires the "sycl_algorithms" path
  (`CCL_ENABLE_SYCL_KERNELS=1`). So on this oneCCL you cannot both (a) init stably AND (b) graph-capture the
  collective. PIECEWISE capture with the TP all-reduce inside the captured region is therefore a no-go on the
  stable config. (Weights sharded fine first: 8.42 GiB/card load = half the 16.7 single-card -> real split, more
  KV headroom.) => **captured TP=2 needs either SYCL-kernels=1 (risk the #41663 GP-fault) or a vLLM that excludes
  the collective from the captured graph.** A targeted follow-up: try captured TP=2 with CCL_ENABLE_SYCL_KERNELS=1
  to see if it both captures AND survives init on our exact cards.
- Campaign auto-fell-back to **EAGER TP=2** (GRAPH=0, `--enforce-eager`; no graph -> no collective recording ->
  serves). Eager TP=2 27B (load 8.42 GiB/card, healthy):
  - **C1: per-stream decode 4.18 t/s** (TTFT 1650 ms, TPOT 239 ms). **C2: 4.02 t/s/stream, 7.48 agg** (TPOT 249 ms).
  - **vs eager TP=1 single-card 27B = 7.84 t/s -> TP=2 is 0.53x = HALF the single-stream decode.** TP=2 does NOT
    help decode here; it HURTS (comms tax). TPOT ~240 ms is flat C1->C2. The per-token cost = eager dispatch
    overhead (single-card eager is already 4x slower than captured 30.84) PLUS 128 all-reduces/token (2/layer x 64
    layers) each crossing the crippled x1 link with cross-worker sync. Decode is collective-LATENCY-bound, not BW.
  - **Stopped the sweep at C2** (cut C4..C64 -> NA in the CSV) to reclaim the lease: the single-stream-loss story
    is proven, and the slow eager high-conc tail is low-information vs the queued AutoRound/bf16/microbench jobs.
    (Honest cap note: eager TP=2 aggregate at high concurrency NOT measured; re-run 58 with CONC capped if wanted.)
  - CSV: results/sweep_27b-w4a16-TP2-eager_20260620_220141.csv. Link AFTER sweep still 2.5 x1 (idle, post-stop).

### 2026-06-21 -- [RESULT] Cross-card all-reduce microbench -- EXACT comms ceiling (novel; no public B70 numbers)
scripts/60_allreduce_bench.sh: torch.distributed xccl, 2 ranks (1/xpu), fp32 all-reduce 4 KB..256 MB, mp.spawn,
under gpu-run. torch 2.11.0+xpu, xpu count 2. Direct quantification of the TP comms bottleneck:
- **Small-message latency floor ~0.28-0.30 ms** (4-16 KB). This is the per-all-reduce fixed cost.
- **Large-message bandwidth ceiling ~0.67-0.72 GB/s busbw** (>= 2 MB; algbw==busbw for 2 ranks). Crossover from
  latency- to bandwidth-bound around ~1-2 MB.
- **Context:** a healthy PCIe link would give ~12 GB/s (Gen3 x16) to ~50 GB/s (Gen5 x16) busbw. We measure
  ~0.7 GB/s = **~17-70x below a healthy link** -> confirms the Gen1-x1 + host-staging diagnosis from the comms
  side, independent of vLLM. NVLink-class (~600 GB/s) is ~850x our number; this is why TP on this rig is a
  capacity tool only.
- For DECODE (per-layer all-reduce ~10 KB): ~0.29 ms x 128 ops/token = ~37 ms/token of pure all-reduce latency
  on top of compute -> matches the observed TP=2 TPOT inflation. For PREFILL (big activations, MB-scale) the
  0.7 GB/s BW ceiling bounds throughput. Either way the x1 link dominates.
- **Bottom line: the multi-GPU bottleneck is quantified end-to-end** -- x1 link (sysfs) -> 0.7 GB/s all-reduce
  (microbench) -> TP=2 decode 0.53x single-card (serve). Fix the physical PCIe (direct Gen3+ x8/x16 slots / non-x1
  risers) and all three improve together.

### 2026-06-21 -- [RESULT][NOVEL] AutoRound quantization runs ACROSS BOTH XPUs (no public multi-XPU precedent)
The user asked to "just try to get it loaded in two gpus." Did better -- ran a full AutoRound quant across both.
scripts/59_autoround_2xpu.sh -> aq2x.py in :v0230 (pip install --no-deps auto-round at runtime; torch-xpu intact
after). Test B (AutoRound on Qwen3-0.6B, bits=4 g128, iters=1 nsamples=8):
- **`AutoRound(device_map="0,1")` WORKS on XPU** -> maps to xpu:0 + xpu:1. auto_round's own device.py logs
  **`peak_vram {'0': 0.62GB, '1': 0.51GB}`** every block -> BOTH cards hold layers, genuinely split, not a replica.
- **Full quant ran: 196/197 layers in 22 s** (lm_head left bf16), `RESULT_B: device_map='0,1' QUANTIZE_RAN`.
  (Block losses explode in late layers -- expected with 1 iter / 8 samples; the point was multi-XPU LOADING, not
  quality. peak_ram ~10 GB CPU.)
- Significance: research found ZERO public multi-XPU AutoRound/GPTQ examples (all multi-card docs are CUDA,
  `CUDA_VISIBLE_DEVICES`, no ZE_AFFINITY/xpu:N path). This rig demonstrates `device_map="0,1"` Just Works on dual
  B70 -- so a 2-card AutoRound of a model too big to calibrate on one card (e.g. the bf16 Qwable-27B / Qwen3.6-27B)
  is viable HERE, not only on CUDA. (Quant tuning is compute/host-bound, NOT all-reduce-bound, so the x1 link does
  NOT hurt it -- unlike TP serving.) Note Test-A transcript (transformers device_map placement) was clipped by a
  `tail -55` on capture; Test B is the conclusive proof. NEXT (optional): real 2-card AutoRound of a >32GB source.

### 2026-06-21 -- [RESULT][KEY] PP=2 BEATS TP=2 on the x1-link rig (pipeline dodges the all-reduce tax)
Hypothesis: on a comms-crippled rig, pipeline-parallel (ONE hidden-state handoff/token at the stage boundary)
should beat tensor-parallel (~128 all-reduces/token). scripts/62_pp2_27b.sh: 27B int4 AutoRound, TP=1 PP=2, eager,
:v0230, single-stream decode probe. CONFIRMED:

| config (eager, 27B int4, single-stream) | C1 decode t/s | TPOT | KV pool | vs TP=2 | vs 1-card |
|---|---|---|---|---|---|
| single-card TP=1 (eager)                | 7.84 | ~128 ms | (1 card) | --     | 1.00x |
| **PP=2**                                | **6.11** | 163 ms | 19.44 GiB/stage | **+46%** | 0.78x |
| TP=2                                    | 4.18 | 239 ms | tight    | --     | 0.53x |

- **PP=2 (6.11) is +46% over TP=2 (4.18)** and reaches 78% of single-card, because PP's per-token comms is a single
  ~10 KB hidden-state send (TTFT 641 ms, TPOT 163 ms) vs TP's 128 all-reduces (TPOT 239 ms). The x1 link punishes
  TP's per-layer collective; PP barely touches the link.
- **PP=2 also wins on KV capacity** (19.44 GiB/stage pool, max-conc 50.7x) -- each card holds half the LAYERS so
  per-card KV is generous; TP=2 splits each layer (tight KV). Weights still 8.41 GiB/card (= half, like TP).
- Residual PP penalty vs single-card (0.78x): at batch-1 PP can't overlap stages (card1 waits on card0 each token)
  so the handoff is pure added latency, no compute-parallel win. At concurrency PP CAN pipeline microbatches
  (overlap stages) -- not swept here (deprioritized; single-stream is the headline).
- **VERDICT: on THIS x1-link rig, PP=2 strictly dominates TP=2 for serving** (better decode AND bigger KV). The only
  thing TP buys that PP doesn't is splitting a single layer's weights (irrelevant -- our layers fit one card). So:
  **use PP=2, not TP=2, for the dual-card capacity play**. When the PCIe link is fixed to Gen3+ x8/x16, re-evaluate
  -- TP may become competitive (its all-reduce stops being the bottleneck) and could win at high concurrency.
- Correctness across multi-GPU: BOTH TP=2 (bf16 27B) AND PP=2 (int4 27B) solved gsm8k #1 correct (got=72) before
  teardown (later problems only errored on serve teardown / x1-link timeout, not wrong answers). Plus coherent
  multi-card generations throughout (0.6B primes, 27B parallelism explainer). No TP/PP output corruption observed.

### 2026-06-21 -- [DIAGNOSIS] The x1 link is PHYSICAL (switch/riser), not ASPM -- and BCS engine resets seen
`lspci -vv` + dmesg settle the cause of the Gen1-x1 bottleneck:
- **LnkCap (CAPABILITY, not just LnkSta) = `Speed 2.5GT/s, Width x1`** on BOTH B70s (0a:00.0, 44:00.0) AND on the
  switch DOWNSTREAM ports feeding them (09:01.0, 43:01.0). The link *capability* -- not merely the negotiated
  state -- is x1. A PCIe5 x16 card advertising x1 cap = the card trained down to the lowest common denominator
  with a link partner that is itself x1. The switch downstream ports being x1-capable => **the cards hang off a
  PCIe switch that provides only x1 per port = a mining-style PCIe expander / x1 risers.** This is PHYSICAL wiring.
- **NOT an ASPM artifact:** `/sys/module/pcie_aspm/parameters/policy = [default]` (not powersave). The x1 is the
  trained link capability, independent of power policy.
- **=> THE fix is physical:** move both B70s to real CPU/chipset x8/x16 slots (off the x1 switch/risers). That
  single change lifts the ~0.7 GB/s all-reduce ceiling and should make TP=2 viable and PP=2 faster. Nothing in
  software (vLLM/CCL/driver) can work around a x1-capable physical link.
- **[!] Stability caveat -- Xe BCS engine resets during multi-GPU runs:** dmesg shows
  `xe 0000:0a:00.0: Engine reset: engine_class=bcs ... Check job timeout ... Kernel-submitted job timed out ...
  reset done` (+ a device coredump) at timestamps coinciding with the TP=2/bf16 runs, on BOTH cards. BCS = the
  blitter/copy engine doing the host-staged all-reduce copies; the slow x1 link + #41663 makes those copy jobs
  time out. They SELF-RECOVERED (reset done -> serves kept running) thanks to CCL_ENABLE_SYCL_KERNELS=0; without
  the stable env these are the fatal GP-faults of #41663. So multi-GPU works here but is NOT bulletproof -- expect
  occasional copy-engine resets under sustained TP load until the PCIe link is fixed.

### 2026-06-21 -- [CORRECTION][!!] The "Gen1 x1 link" is an Intel Arc lspci ARTIFACT -- real link is Gen3 x16
SUPERSEDES the two entries above ("x1 link is PHYSICAL (switch/riser)" and the WIP topology entry) and the
FINDINGS "move to direct slots" recommendation. User flagged the cards are in real Gen3 x16 slots; re-investigated
read-only (no gpu-run lease; another agent was serving on both cards -- a free under-load datapoint).

**Root cause of the misdiagnosis: we read the WRONG PCIe node.** Arc/Battlemage cards interpose an ON-CARD Intel
PCIe switch between the slot and the GPU die. Per **Intel KB 000094587**, the GPU-adjacent nodes (the GPU endpoint
and the switch's downstream port) *always* report a bogus **2.5 GT/s x1** -- in BOTH LnkSta (current) AND LnkCap
(capability). The prior [DIAGNOSIS] entry saw `LnkCap = 2.5GT/s x1` on `0a/44:00.0` + `09/43:01.0` and concluded
"the capability is x1 => physical x1 wiring." That conclusion is wrong: that low LnkCap is the documented artifact,
not the silicon limit.

**The card's on-card switch (decoded):**
```
  CPU root 00:03.1 === 08:00.0 (switch UP, e2ff) ==+== 09:01.0 (DOWN, e2f0) === 0a:00.0  GPU  e223
  (Gen3 x16, AMD)      Gen5-cap, trained 8GT/s x16 |   2.5GT/s x1 (ARTIFACT)     2.5GT/s x1 (ARTIFACT)
                       << REAL host link, healthy  +== 09:02.0 (DOWN, e2f1) === 0b:00.0  DP-audio e2f7
```
The switch is Intel silicon `8086:e2ff` with two Intel downstream devices (GPU `e223` + DP/HDMI-audio `e2f7`) --
i.e. on the card, NOT a third-party riser/mining board (those use ASMedia/PLX). Both B70s show identical structure.

**Evidence the real link is Gen3 x16 (the 1950X platform max), not Gen1 x1:**
- Read at the switch UPSTREAM bridge (`08:00.0` / `42:00.0`) per Intel KB: `LnkSta: 8GT/s x16` on BOTH cards.
  `LnkCap: 32GT/s x16` (silicon is Gen5-capable; "downgraded" to 8GT/s only because the 1950X host is Gen3).
- Power state ruled out: both GPUs `power_state=D0`, `runtime_status=active` (the other agent's `vllm_multi` was
  serving on them) and STILL read 2.5 x1 at the endpoint -> that's the artifact, not idle ASPM/D3.
- The earlier "corroborations" were misattributed: "200/200 samples x1 under load" polled the artifact node
  (`0a`/`44`); "~220-234 MB/s weight-load = Gen1 rate" is a SATA-SSD/loader bottleneck -- models live on
  `/dev/sdd1` (SATA SSD, btrfs) + safetensors deserialization, comfortably SATA-class, NOT PCIe.
- Community confirmation on this exact card: a B70 clpeak run measured ~55 GB/s H2D (Gen5 host) while lspci showed
  the same fake 2.5 x1 -- "purely cosmetic reporting." (Intel KB 000094587; gist mploschiavo/9968c883...)

**What is STILL true (the real bottleneck, just re-attributed):** TP=2 all-reduce ~0.7 GB/s busbw, TP=2 decode
0.53x single-card, PP=2 0.78x -- all REAL and stand. But the cause is **no GPU P2P -> host-staged oneCCL/xccl
collectives** (GPU->host RAM->GPU) + collective overhead, NOT a crippled PCIe gen. A clean H2D copy should show
~10-12 GB/s (Gen3 x16); queued `scripts/63` will put a positive number on the record to nail it.

**Actionable changes:**
- DELETE "move cards to direct Gen3+ slots" from the plan -- they are ALREADY at full Gen3 x16; reseating gains 0.
- The real levers to "optimize the dual GPUs": (1) GPU P2P if achievable (research in flight -- but cross-die on
  the 2-die 1950X + no Xe-Link makes this unlikely), (2) **data-parallel = 2 independent single-card replicas**
  (zero inter-GPU comms, ~2x aggregate throughput) when the model fits one card, (3) PP=2 for models too big for
  one card (already shown to beat TP=2 here), (4) MTP/spec-decode as a per-replica single-stream multiplier.
- To read the TRUE link on Arc, always read the on-card switch UPSTREAM bridge (`08/42:00.0`), never the GPU
  endpoint. Helper: `for b in 08:00.0 42:00.0; do lspci -vvv -s $b | grep LnkSta; done`.

### 2026-06-21 -- [RESULT] Interconnect TRUTH probe -- Gen1x1 DISPROVEN by measurement; no P2P confirmed
scripts/63_interconnect_probe.sh + bw_p2p_probe.py (torch 2.11+xpu, single proc, gpu-run, 17s). The positive
numbers that close out the misdiagnosis AND scope the P2P question:
- **H2D = 12.82 GB/s** (256 MiB payload). A real Gen1 x1 link would be ~0.20-0.25 GB/s. This IS Gen3 x16
  (~10-12 GB/s expected). **The "Gen1 x1" is conclusively a reporting artifact.** D2H = 3.51 GB/s (the read-from-
  GPU direction is asymmetrically slower -- no pinned host mem; known XPU behavior).
- **`torch.xpu.can_device_access_peer(0,1) = False`** -> Level-Zero `zeDeviceCanAccessPeer` is False on these
  Battlemage cards. So oneCCL's direct-P2P "topo" path will NOT auto-engage; forcing `CCL_TOPO_P2P_ACCESS=1`
  overrides the probe but is exactly what GP-faulted + BCS-reset in vLLM #41663. Matches that report's `p2p_access:0`.
- **D2D xpu0->xpu1 = 1.68 GB/s** = host-staged (GPU->host->GPU), bounded by the slow D2H leg. This is the real
  TP all-reduce ceiling, and it is a no-P2P/host-staging limit -- NOT the PCIe gen (H2D alone is 12.82).
- **ReBAR ON:** BAR2 = 32G full-VRAM aperture on both cards (P2P prerequisite already met). **IOMMU ON:** 57 groups,
  AMD-Vi active (Unraid VFIO passthrough uses it -- `vfio-pci.ids=10ec:8168`). **Kernel 6.18.33-Unraid** (no
  Linux-7.0 xe multi-device-SVM P2P path).

### 2026-06-21 -- [ASSESSMENT] The P2P moonshot is LOW ROI (~15% ceiling); data-parallel + MTP are the real wins
Two deep research passes (kernel/driver + frameworks/forks), cross-corroborated. Verdict on "create our own P2P
lever":
- **B-series has NO Xe-Link** -- multi-GPU is PCIe-only by product design (Xe-Link is Data-Center-Max-only). Intel
  markets "PCIe P2P" (Project Battlematrix) but **IPEX docs explicitly state oneCCL allreduce "does not support
  PCIe for cross-cards communication"** (`TORCH_LLM_ALLREDUCE=1` requires Xe-Link). The optimized path is fabric-only.
- **No geohot-style unlock exists for Intel.** NVIDIA's P2P hack works because NVIDIA *software-locks* a silicon
  capability; Intel's P2P is not locked -- it is just unstable (PCIe RxErr -> BCS reset) and, when it works, only
  **~15% faster at large batch** (Puget). There is nothing to "unlock" for a 10x.
- **Paths that COULD enable it, ranked by cost:** (a) cheap/risky: force `CCL_TOPO_P2P_ACCESS=1` on the current
  stack + run the allreduce microbench watching dmesg -- but `can_device_access_peer=False` means this likely
  crashes (#41663). (b) expensive: Linux 7.0+ xe multi-device SVM P2P (Hellstrom series) -- needs an Unraid kernel
  rebuild AND `iommu=pt`/`amd_iommu=off` (which would BREAK the existing VFIO NIC passthrough), and BMG peer access
  is still unproven even on 7.0. Upside remains ~15%, and it does NOT help batch-1 decode (the headline metric).
- **CONCLUSION: deprioritize P2P.** The high-ROI dual-GPU levers, in order: (1) **data-parallel = 2 independent
  single-card replicas** (zero comms, ~2x aggregate throughput, no infra risk) whenever the model fits one card;
  (2) **MTP/spec-decode** (~3-4x single-stream decode, per replica); (3) **PP=2** for models too big for one card.
  Keep `CCL_TOPO_P2P_ACCESS=0` (host USM) as the stable TP/PP default. [P2P remains an optional novel-data
  experiment -- our clean direct-slot rig is a good test case and no public PCIe-B70 P2P number exists -- but it is
  not the path to a faster dual setup.]

### 2026-06-21 -- [RESULT][KEY WIN] Data-parallel 2 replicas = ~2.1x aggregate, ZERO contention (the dual-GPU answer)
scripts/64_dataparallel_2rep.sh: one captured 27B-int4 (=w4a16) replica per B70 (card0 ZE_AFFINITY_MASK=0 :18080,
card1 :18081), independent, no inter-GPU traffic. Added backward-compat `PORT`/`DEVICE` knobs to 30_serve so two
replicas coexist on one host. Phase 1 = dp0 solo (fresh single-card baseline); Phase 2 = dp0+dp1 swept CONCURRENTLY.

| C  | dp0 solo agg t/s | dp0 conc | dp1 conc | **DP sum** | **scaling vs solo** |
|---:|---:|---:|---:|---:|---:|
| 1  | 26.93 | 28.85 | 27.44 | **56.3**  | **2.09x** |
| 8  | 132.10 | 145.12 | 133.58 | **278.7** | **2.11x** |
| 32 | 213.84 | 240.78 | 216.08 | **456.9** | **2.14x** |
| 64 | 258.81 | 262.21 | 262.62 | **524.8** | **2.03x** |

- **~Linear (2.03-2.14x). NO contention:** each replica under concurrent load matched (or slightly beat, run-to-run
  variance) its solo number -- host CPU/PCIe/mem is NOT a bottleneck for two independent 27B-int4 serves. Solo
  baseline reproduced the banked single-card curve (C32 213.8 ~ banked 217; C64 258.8 ~ banked 235, +variance).
- **DP dominates model-parallel on BOTH axes** for a model that fits one card: throughput ~525 t/s @C64 (2x) AND
  single-stream latency stays full single-card 30.8 t/s/replica -- vs TP=2 4.18 (0.53x) / PP=2 6.11 (0.78x).
  The all-reduce/handoff tax of TP/PP simply doesn't exist when there are no collectives.
- **=> Serving doctrine:** model fits one card -> **2x DP replicas behind a round-robin proxy** (+ MTP per replica).
  Model too big for one card -> PP=2 (beats TP=2 here). TP=2 only if a single layer can't fit one card (never, for us).
- Both replicas served clean (PIECEWISE capture, 16.69 GiB load + 2.35 GiB graphs each, served-id verified
  qwen36-27b-int4-dp0/dp1). CSVs: results/sweep_dp-solo|dp-conc0|dp-conc1_*.csv. (Note: another agent's grouped-int8
  MoE test held the lease first; gpu-run queued us correctly. Minor: it also numbered a script "64" -- filename
  differs, no clash.)

### 2026-06-21 -- [RESEARCH] 4xB70 TP=4 MTP is REAL (owner's primary source); our single-card MTP is -19% -- get the recipe
Deep MTP pass (3 web sub-agents + cross-check vs our own runs), then RECONCILED with the user's attestation.
- **The 4xB70 TP=4 MTP result (Qwen3.6-27B BF16, dec ~54.2 t/s, accept ~4.04 @ spec=5) is REAL** -- the project
  owner knows the author personally (primary source). It is simply NOT publicly documented, which is why the web
  pass couldn't find it (do NOT read "not public" as "not real"). The public numbers that superficially resemble
  fragments of it (PMZFX single-B70 llama.cpp 35B-A3B 54.7; an NVIDIA 5090 MTP ~2.9x; Puget 4xB70 TP=4 27B-dense
  13.1 no-MTP) are coincidence, not the source. **ACTION: get the exact repro (image/kernels/speculative-config/
  whether FULL graph capture/the 4-card interconnect) -- it's the unlock, because it beats our single-card result.**
- **Why "TP=4 enabled MTP" = capacity confound.** BF16 27B ~= 54 GB -> needs ~4 cards just to FIT. TP=4 was the
  entry ticket to run BF16 at all; MTP's benefit is independent of TP. MTP does NOT need TP; TP>1 HURTS it here.
- **MTP on B70 is currently NET-NEGATIVE (our only real B70 MTP data): 25.5 t/s = -19%** vs 31.4 off (single-card
  Lorbus int4, PIECEWISE, 86.9% accept). Cause is structural: PIECEWISE verify runs attention EAGER x(K+1). The
  fix is FULL graph capture, blocked on XPU (needs TRITON_ATTN+single-GPU+non-GDN, or #43565 + vllm_xpu_kernels
  v0.1.10 in a torch-2.11 image -- an ABI split blocks one wheel being both torch-2.10-safe AND spec-capable).
- **Actionable repro ladder (cheapest first):** (1) [done] single-card MTP int4 = -19% baseline; (2) draft-model
  spec on the DENSE 14B-W4A8 with FULL+TRITON_ATTN (14B has no GDN -> dodges the GDN-spec-capture bug; the only
  arch where FULL+spec is unblocked on B70 today) -> first real net-positive spec number; (3) FULL+MTP on int4 27B
  once a torch-2.11 image with #43565 + kernels v0.1.10 exists; (4) then MTP per data-parallel replica (NOT TP=2).
- MTP_TODO.md updated: the 4xB70 figure is REAL (owner's primary source), reframed from "unverified" to
  "get the exact repro". Plan mechanics (Phase A/B quant-recovery) still valid.

### 2026-06-21 -- [RESEARCH] 35B-A3B MoE on dual B70: int4 + 2x DATA-PARALLEL wins; EP is crippled on XPU
Deep MoE pass (comms arithmetic + vLLM-XPU EP/PP support state + corrected external numbers). Arch (served
config): hidden=2048, L=40, 256 experts, top_k=8 + a shared (always-on) expert -> tiny per-token cross-card payloads.
- **Per-token comms (this rig, ~0.29 ms/collective latency floor, latency-bound not byte-bound):** TP=2 ~80
  all-reduce/token (~23 ms -> ~25 t/s ceiling); EP=2 ~80 all-to-all/token (>= TP latency; the shared expert can't
  be localized); **PP=2 = 1 handoff/token (~0.3 ms, negligible).** Matches our measured TP=2 0.53x / PP=2 0.78x.
- **vLLM-XPU EP is present but CRIPPLED + crashing:** `--enable-expert-parallel` exists only via the slow
  allgather+reduce-scatter path (no DeepEP/nvshmem on XPU); open llm-scaler bugs on B70 MoE: #477 (TP+EP
  `moe_topk_softmax unsupported E=512`), #479 (MoE int4 expert-GEMM OOM), #382 (35B-A3B FP8 TP=2 warmup OOM),
  #489 (PP=2 device-lost after 1 req). So EP is NOT a perf contender on our stack today.
- **Capacity:** 35B-A3B int4 (W4A16) = 19.6 GiB -> **FITS ONE CARD** -> data-parallel applies. FP8 (~35 GB) /
  INT8 need 2 cards (and INT8 has no working XPU fused-MoE kernel yet).
- **VERDICT: int4 + 2x DATA-PARALLEL is the answer** -- same as the dense 27B. Zero comms, project ~2x aggregate
  (toward ~400 t/s vs single-card ~206 plateau) AND full single-stream ~56.8 t/s/replica preserved (MoE decode is
  BW-bound on ~3B active params). Bench it via `scripts/64_dataparallel_2rep.sh` (swap MODEL + image :v0230moe).
  Reserve PP=2 for the FP8/capacity case (bigger KV / 8-bit quality); TP=2 only at high concurrency; EP last (probe-only).
- **Corrected stale community numbers:** the banked "dual-B70 TP=2 27B FP8 13.25/97.84" was actually Puget's
  4xB70 TP=4 27B-DENSE (13.1/95.9); "30B-A3B MoE FP8 912 t/s" is UNVERIFIED (no public source). Real: Puget
  4xB70 TP=4 35B-A3B MoE = 16.3(C1)/63.7(C4)/122(C8). FINDINGS.md corrected.

### 2026-06-21 -- [QUANT QUEUE] 48h INT8 fast-path campaign kicked off (QUANTS_TODO Q0-Q7)
Autonomous /loop run: drive every qwen3.6-family model into W8A8 (AutoRound) + W4A8 (selective-SQ+GPTQ),
eval + perf-bench (pp/ttft/tg @ ctx2048, c=1/2/4/8) vs the w4a16 analogues, test MTP receptivity, optimize.
State at start: both B70s free, lock free, all bf16 sources present incl. the 35B-A3B (67 GB/26 shards, download
now COMPLETE -- resolves the "no bf16 35B source" blocker kernel/15 flagged). v0230/int8/int8g/v0230moe images all present.
- **Q0 DONE (code):** added `SMOOTHQUANT=selective` to scripts/49 -- Playbook-B explicit per-layer SmoothQuant
  mappings built by INSPECTING the loaded model (full-attn q/k/v<-input_layernorm + o<-v, MLP gate/up<-post_attn_ln,
  MoE experts.* router-aware), skipping DeltaNet linear_attn/vision/MTP. Fixes the auto-resolver ValueError on the
  hybrid. Embedded-python compiles clean (113 lines), apostrophe-free (bash -c heredoc safe). Gates Q3/Q5/Q7.
- **New reusable tooling:** `scripts/65_autoround_w8a8.sh` + `scripts/_autoround_w8a8.py` (generalizes kernel/15
  sec-2 inline recipe: loader fallback chain, layer_config ignore, DEVMAP xpu|0,1, LOWMEM streaming, IGN_MOE for 35B).
  `scripts/qrun.sh` = detached gpu-run launcher (survives ssh close via setsid; writes results/NAME.log + QRUN_EXIT sentinel).
- **Q1 SMOKE launched:** 14B W8A8 AutoRound, iters=50 nsamples=16 seqlen=512, DEVMAP=xpu, detached, holding the lock.
  Toolchain validation before the full run. Poll results/Q1_smoke.log for QRUN_EXIT + DONE_AUTOROUND_W8A8.

### 2026-06-21 -- [Q1] AutoRound-W8A8 toolchain VALIDATED end-to-end on XPU (2 bugs found+fixed)
Smoke-gated the 14B W8A8 AutoRound path. Two cheap fast-fails caught real bugs before the full run burned hours:
- **Bug 1 (mount):** 14B bf16 lives at /mnt/vm_8tb/specula-build (OUTSIDE $ROOT) -> 65_autoround_w8a8.sh never
  bind-mounted it -> HF treated the invisible path as a repo id (HFValidationError, 37s). Fix: conditional
  `-v $SRC:$SRC:ro` when SRC is not under $ROOT.
- **Bug 2 (export):** AutoRound tuning RAN FINE on XPU (40/40 layers, loss decreasing, peak_vram 1.7GB streaming)
  but the llm_compressor exporter crashed at export.py:152 `layer.scale.to()` (NoneType). ROOT CAUSE (read from
  installed auto_round 0.13.1): `check_to_quantized()` gate = `bits<=8 OR act_bits<=8`. Excluding a layer with only
  `{"bits":16}` left act_bits=8 -> gate True -> exporter tries to pack a never-quantized weight -> scale is None.
  **Fix: ignored layers MUST be `{"bits":16,"act_bits":16}`** (both). Applies to lm_head + all VLM/MoE ignores.
- **Validated:** iters=0 RTN export produced a valid 9-shard compressed-tensors W8A8 checkpoint (QRUN_EXIT 0).
  AutoRound-on-XPU + W8A8 llm_compressor export both WORK -> de-risks Q2/Q4/Q6 (the 27B/Qwable/35B W8A8 items).
- **Quirk:** AutoRound nests output under OUT/<modelname>-w8a8/ -> flatten post-run before serving.
- **Q1 FULL launched:** iters=200 nsamples=128 seqlen=2048, DEVMAP=xpu, ETA ~90min. Then flatten -> serve :int8g
  GRAPH=1 -> eval (accuracy vs bf16 + perf sweep) vs the existing Qwen3-14B-W8A8-gptq.

### 2026-06-21 -- [Q1] full-run OOM fixed (batch_size cap) + optimization-frontier research landed
- **Q1 full OOM:** iters=200 nsamples=128 seqlen=2048 died at layer 0 in 80s -- UR_RESULT_ERROR_OUT_OF_RESOURCES.
  Per-block AutoRound tuning activation ~ batch_size x seqlen; the default batch at seqlen=2048 overflows one 32GB
  card (the smoke survived only because it was 16x512). Fix: added BATCHSIZE + GRADACC knobs to the driver/65
  (gradient_accumulate_steps keeps effective batch up at low peak mem). Relaunched DEVMAP=xpu BATCHSIZE=2 GRADACC=4.
- **Research (2 parallel agents while the GPU quantized):**
  - docs/literature/09_mtp_receptivity_vs_quant.md: literature (arXiv:2505.22179) says with a BF16 draft head,
    body W8A8 vs W4A16 cause only SECOND-ORDER acceptance differences -> ordering BF16 >= W8A8 >= W4A16 >= W4A8.
    Our -19% MTP is a graph-capture problem, NOT acceptance. Reframes the MTP question: int8 is not a big MTP unlock.
  - docs/literature/08_int8_gemm_gemv_xe2_frontier.md: top decode lever = column-reorder + dp4a int8 GEMV
    (W8A8 decode ~26 -> 35-40 t/s; fixes uncoalesced int8 loads, 61% vs W4A16 73% BW). int8 MoE grouped GEMM is a
    PREFILL-only win (1.43-2.01x bf16, needs fused act-quant); 35B decode stays W4A16. 100-GEMM + 100-GEMV shape
    lists specified (17 (K,N) shapes x M sweep) -> the microbench plan for task #11.

### 2026-06-21 -- [Q1 DONE + 14B ladder] W4A8 BEATS W4A16; W8A8 is decode-BW-bound (the headline result)
Q1 (14B W8A8 AutoRound) full run DONE (3146s, iters=200 nsamples=128 seqlen=1024 batch=4, DEVMAP=xpu). Flattened
nested OUT/Qwen3-14B-w8a8 -> OUT. Served :int8g GRAPH=1 (id qwen3-14b-w8a8-autoround) + 3 existing 14B quants;
ctx-2048 sweep at c=1/2/4/8 (NEW config -- prior CSVs were ctx-512). CSVs in evals/results/ctx2048_14b/.

14B ctx-2048 (per_stream_decode t/s | mean_ttft ms | c8 aggregate out_tok/s):
  scheme            c1dec  c2    c4    c8     c1_TTFT  c8_agg
  W8A8-autoround    25.1   24.5  22.8  18.0    347ms    125.0
  W8A8-gptq         24.8   24.6  23.0  18.1    351ms    125.6
  W4A16-gptq        52.5   45.1  37.9  22.2    571ms    132.9
  W4A8-gptq         49.3   45.1  39.3  25.5    405ms    161.7   <- best all-rounder

FINDINGS (answers "where does int8-act beat w4a16"):
- **W4A8 (int4 w + int8 a) BEATS W4A16**: ties int4-weight decode BW (~49 vs 52 t/s c1) but the int8-act prefill
  cuts TTFT -29% (405 vs 571ms) and lifts c8 aggregate +21.7% (161.7 vs 132.9). THIS is the int8-XMX prefill win.
- **W8A8 (int8 w) is decode-BW-bound** -- ~25 t/s decode = HALF the int4-weight schemes (8-bit weights = 2x the
  per-token weight read). BUT lowest single-stream prefill TTFT (347ms). So W8A8 = prefill-latency/accuracy play,
  NOT a decode play. Matches docs/literature/08 (W8A8 decode BW-starved; int4-weight wins decode bandwidth).
- AutoRound-W8A8 == GPTQ-W8A8 on SPEED (identical kernel path); the AutoRound edge is ACCURACY (eval TBD).
- IMPLICATION for the 27B target: expect W4A8 to be the winner (prefill + concurrency over the W4A16 daily driver),
  W8A8 a prefill-latency/quality option. Decode t/s ceiling is set by weight bit-width (BW-bound), not act bits.

### 2026-06-21 -- [Q2] 27B VLM W8A8 AutoRound path VALIDATED (smoke), full launched
27B (qwen3_5 VLM+GDN+MTP) W8A8 AutoRound smoke (iters=0, DEVMAP=0,1) DONE clean: VLM loaded via the fallback
chain, quantized 64 blocks across BOTH B70s, exported valid compressed-tensors W8A8 (13 shards). The previously
UNVERIFIED "AutoRound-on-XPU for the VLM" production path (kernel/15 FIND/COMMUNITY) is now PROVEN -- no need for
the GPTQ-W8A8 fallback. Q2 full launched: iters=200 nsamples=128 seqlen=1024 BATCHSIZE=2 GRADACC=2 (effective
batch 4 at ~half per-block memory -- 27B blocks bigger than 14B; safety-first vs the Q1 OOM). DEVMAP=0,1, ~2-3h ETA.

### 2026-06-21 -- [Q2] GPTQ-on-VLM is O(depth^2); fixed via SAMPLES=128 SEQLEN=1024
Pivoted Q2 (27B W8A8) AutoRound->GPTQ (AutoRound's data-driven calib auto-routes VLMs to its MLLM path -> needs
a processor, breaks for text-only). GPTQ selective-SQ ran: Q0 mapping VALIDATED on the real 27B hybrid
([selective-sq] attn=16 mlp=64 mappings=80, SmoothQuant applied clean after the GQA o<-v fix). BUT the run crawled:
- **llmcompressor's sequential pipeline does NOT cache per-layer inputs on the qwen3_5 VLM** -- it re-runs the full
  layer prefix every calibration step. Per-layer cost is O(depth): measured 0.045 s/it at layer 0 -> 2.0 s/it at
  layer 52 (~45x). Host RAM fine (83G free, no swap), load ~1 -> not thrashing; it is genuine prefix re-compute.
  At SAMPLES=512 SEQLEN=2048 the 64-layer 27B GPTQ projects to ~6 h. (Likely the pip-pulled compressed-tensors
  0.17.1 / newer llmcompressor changed pipeline tracing for the VLM.)
- **Fix: SAMPLES=128 SEQLEN=1024** (~8x faster, ~45-60 min, standard GPTQ quality). Killed the 512/2048 run at 52/65
  and relaunched. Applies to ALL 27B+ GPTQ (Q2/Q3/Q5/Q7) -- QUANTS_TODO recipe 4B updated.
- Also confirmed: AutoRound W8A8 works for the DENSE 14B but its calib breaks on qwen3_5 VLMs (MLLM-processor
  assert) -> VLM W8A8 uses GPTQ-selective-SQ (same int8 kernel/perf; the 14B showed AR==GPTQ on speed).

### 2026-06-21 -- [Q2/Q3 serve] 4304 vision-fc2 fix: graft must add visual/mtp to the ignore list
Both 27B int8 quants PRODUCED (Q2 W8A8-sqgptq 35GB, Q3 W4A8-sqgptq 33GB; W4A8 config confirms W4 g128 + A8).
First 27B ladder bench FAILED to serve all 3 -- root causes:
- **W8A8:** my bench passed KVDTYPE=fp8_e5m2 -> vLLM "fp8_e5m2 kv-cache not supported with fp8 checkpoints"
  (spurious for an int8 ckpt, but it rejects). Fix: serve W8A8 with default fp16 KV (per-channel int8, no 4304 issue).
- **W4A8:** `input_size_per_partition 4304 not divisible by group_size 128` -- the documented vision-fc2 odd-dim.
  ROOT CAUSE: llmcompressor loads the qwen3_5 VLM via AutoModelForCausalLM (text-only fallback) so it NEVER sees
  model.visual.* -> the saved quantization_config.ignore (337 entries: linear_attn + lm_head) has NO visual entries.
  After grafting the VLM wrapper back, vLLM matches the vision Linears to config_group targets=["Linear"] and tries
  int4-g128 on the 4304 vision fc2 -> assert. **FIX: fix_27b_vlm_config.py now appends re:.*visual.* + re:.*mtp.* to
  the grafted ignore** (W4A8 ignore 337 -> 339). Re-grafted Q2+Q3; re-benching.
- **W4A16 (Lorbus daily driver):** failed differently (AttributeError NoneType.size in the capture path) under the
  generic bench flags -- it is proven via daily_driver_serve.sh; use its known numbers (30.8 t/s captured C1) for the
  W4A16 baseline rather than rabbit-holing its serve here.

### 2026-06-21 -- [Q2/Q3 27B] checkpoints PRODUCED; single-card int8 SERVE blocked by stacked XPU-serve bugs
Both 27B int8 checkpoints exist + grafted (Q2 W8A8-sqgptq 35GB, Q3 W4A8-sqgptq 33GB + prepacked 25GB). Tried hard
to serve+bench on one card; hit FIVE distinct XPU-serve issues in sequence (each fixed, next appeared):
  1. fp8_e5m2 KV -> "not supported with fp8 checkpoints" (this vLLM build rejects fp8 KV even for int ckpts). Fix: fp16 KV.
  2. W4A8 int4-g128 on the 4304 vision fc2 -> assert. Fix: graft appends re:.*visual./re:.*mtp. to ignore (committed).
  3. W4A8 raw load -> "No available memory for cache blocks" (28GB unpacked-int8 GPU transient). Fix: offline prepack -> 25GB.
  4. prepacked W4A8 + fp8 KV -> same fp8 reject. Fix: fp16 KV.
  5. prepacked W4A8 fp16 KV -> AssertionError param_data.shape==loaded_weight.shape in load_merged_column_weight
     (prepacked int32 [N,K//8] shape vs vLLM merged qkv/gate_up loader expectation). <- the current blocker.
Also W8A8 27B (35GB) does NOT fit one card regardless (needs TP=2/offload); W8A8 is decode-BW-bound anyway (14B data).
DECISION: stop chasing the single-card 27B int8 serve (documented-fragile path; 5 stacked bugs). The thesis is ALREADY
proven on the 14B ladder (W4A8 beats W4A16: -29% TTFT, +22% c8 agg). 27B int8 perf inferred from (a) the 14B pattern
+ (b) the existing Qwen3.6-27B-W4A8-q-prepacked which DOES serve (20.9 t/s captured decode, SERVING.md). Checkpoints
are the deliverable; pivot to producing Q5/Q4/Q6/Q7 + the GEMM/GEMV microbench + final writeup. (#5 is a good
next-agent task: align the prepack tensor layout with vLLM's merged-column loader, or serve the raw W4A8 with cpu-offload.)

### 2026-06-21 -- [#11 microbench] int8 vs bf16 GEMM/GEMV measured (341-row sweep, real int8_gemm_w8a8 op)
docs/kernel/19. GEMM(prefill): int8 1.06-2.13x bf16 (median 1.68x), grows with M (1.59x@64 -> 1.97x@4096), peak
250.9 INT8 TFLOP/s. GEMV(decode): 1.12-2.12x, BW-bound -- ~2x on large-N dense (14B/27B attn+mlp, up to 433 GB/s)
but only ~1.1x on small-N (35B MoE experts N<=2048, KV-proj) which are overhead-bound. Reconciles the served bench:
decode is bytes-bound so int4-wt > int8-wt > bf16; prefill is compute-bound so int8-XMX ~1.6-2x. Confirms W4A8 = best
all-rounder + 35B MoE stays W4A16-int4 for decode. Next lever (doc 08 P4/P5): col-reorder dp4a GEMV for the small-N shapes.

### 2026-06-21 -- [Q4/Q5 Qwable] both int8 checkpoints PRODUCED + grafted (all four 27B-class quants done)
Qwable-5-27B-Coder W8A8-sqgptq (33G) + W4A8-sqgptq (33G), both vision-grafted. Completes the 27B-class matrix:
{27B-base, Qwable} x {W8A8, W4A8} = 4 int8 checkpoints (+ 14B W8A8). Serve same as the 27B-base equivalents
(single-card fragile/too-big; checkpoints are the deliverable, perf inferred from the 14B ctx-2048 ladder).
Launched the 35B-A3B MoE W8A8 GPTQ SMOKE (samples=8 seqlen=512) to gauge the 256-expert MoE GPTQ path feasibility
+ per-layer cost before committing the full produce-only run (serve gated on the int8 MoE kernel regardless, docs/kernel/18).

### 2026-06-22 -- [P2P research] kernel 7.0 xe P2P + Seguin lab + ZML; new doc P2P_GPU.md
Question: should we leave Unraid (kernel 6.18) for Ubuntu 26.04 (kernel 7.0/7.1) to get B70 GPU-to-GPU P2P?
Recon of the box: two B70s are on SEPARATE 1950X dies (cross-die = worst case for PCIe P2P); real per-card link
is Gen3 x16 (~15.8 GB/s) -- the Gen1 x1 sysfs reading is an SR-IOV VF artifact, not a downtrain. ReBAR 32GB,
ACS override, iommu=pt all set. Deep-research (98 agents, 16 sources, 25 claims 3-vote-verified): the new xe P2P
in Linux 7.0 is REAL but is SVM device-private page migration (xe_svm.c, dma_map_resource, XE_INTERCONNECT_P2P,
Project Battlematrix 15-patch series) gated on pci_p2pdma_distance()>=0 -- NOT the oneCCL TP collective path.
AMD Zen (family 0x17 = our 1950X) IS whitelisted for cross-RC P2P (kernel 5.9 dea286bb71ba) EXCEPT the whitelist
is disabled "when an IOMMU is present" (6dbbd053e6) and whether iommu=pt counts is the load-bearing unknown.
Puget (4x B70) saw P2P fault (RxErr/engine reset) and ships P2P off; host-staged TP still scales ~2x.
KEY CONTRADICTION: Steve Seguin's b70-optimization-lab runs CCL_TOPO_P2P_ACCESS=1 (P2P ON) and faster -- his
win is allreduce/graph-fusion surgery (clone-safe compiled allreduce custom-op = biggest jump), NOT transport;
he proved the B70 bottleneck is graph breaks around collectives (raw allreduce 15-17us), not the wire. ZML's
answer is collectives-as-compiler-IR (StableHLO/XLA), which makes Seguin's hand-fusion the default -- but ZML's
shipping server is single-GPU only today. VERDICT: don't migrate solely for P2P (payoff small, record empty on
discrete-Battlemage cross-die P2P); highest leverage is SOFTWARE (steal Seguin's 3 env vars, A/B P2P on/off on
our Gen3 fabric) + the pioneering bet of compiler-fused XPU collectives. Also explored composable PCIe-Gen5-switch
/ SR-IOV-fabric relocation to fix the cross-die topology. All written up with refs in docs/P2P_GPU.md.

### 2026-06-22 -- [27B W4A8 serve] ROOT CAUSE of the merged-column assert FOUND + fixed
The W4A8-sqgptq-prepacked serve assert (load_merged_column_weight: param_data.shape != loaded_weight.shape) was
NOT a weight problem -- the q/k/v/o/gate/up tensors are byte-identical to the working `27B-W4A8-q-prepacked`. The ONLY
config diff: ignore list had 339 entries (336 EXPLICIT linear_attn module names + lm_head + visual + mtp) vs the
working 4-regex form. An explicit list MISSES the DeltaNet FUSED projections (in_proj_qkvz/in_proj_ba) which vLLM
loads via load_merged_column_weight; for W4A8 those are bf16 [N,K] on disk but vLLM allocates packed-int4 [N,K/8]
=> shape assert. (W8A8 doesn't hit this: int8 [N,K] matches bf16's shape.) FIX: rewrite ignore ->
["lm_head","re:.*linear_attn.*","re:.*visual.*","re:.*mtp.*"] (the regex catches ALL linear_attn submodules incl the
fused ones). Re-serving the prepacked W4A8 to capture the real 27B int8 ctx-2048 ladder. TODO: make scripts/49 emit
the regex ignore (not explicit names), or have fix_27b_vlm_config collapse explicit linear_attn -> regex.

### 2026-06-22 -- [35B MoE] DEFERRED behind a small-MoE kernel bring-up (QUANTS_TODO sec 7)
35B MoE GPTQ path PROVEN (smoke quantized real MoE layers) but ~25-30 min/LAYER x 41 = multi-day produce, and it is
serve-gated (no int8 MoE kernel). Diagnosed it is NOT RAM spill: host 72/125 GB used, 53 GB free, swap=0; device_map=cpu
+ llmcompressor onloads one layer at a time. The cost is inherent MoE-GPTQ (256 experts/layer x Cholesky + O(depth)
prefix replay). Decision (user): defer 35B; bring up the int8 MoE kernel on a SMALL MoE first (OLMoE-1B-7B W8A8 ->
then Qwen3-30B-A3B -> then 35B). Pivoted the GPU to the 27B W4A8 serve fix (real int8 perf data) instead. QUANTS_TODO sec 7.

### 2026-06-22 -- [27B W4A8 serve] ignore-fix WORKED; now KV-memory bound on one card (-> TP=2 motivation)
After rewriting ignore to the 4-regex form, the merged-column shape assert is GONE -- the prepacked W4A8 loads + captures
the PIECEWISE graph. Next error was benign: KV cache too small (0.34 GiB free vs 0.41 needed @ max_len 4096). The 25GB
prepacked W4A8 on a 32GB card leaves little KV room. Refit: MAXLEN=2560 (covers our 2048+128 bench) UTIL=0.97. NOTE: this
is the single-card concurrency wall -- ~2-2.5GB KV left => c=8 @ ctx-2048 (~4GB KV) likely won't fit; c=1/2/4 will. This is
exactly the case for TP=2 (task #12): split the 27B across both B70s (~12.5GB/card) -> abundant KV -> real concurrency.
So: capture TP=1 c=1/2/4 ctx-2048 now (first real servable 27B int8 datapoint), then the Seguin TP=2 A/B for concurrency.

### 2026-06-22 -- [27B W4A8] FIRST real servable 27B int8 ladder (supersedes the inferred Q3 row)
After 6 stacked serve bugs all fixed (fp8-KV, 4304 vision, OOM->prepack, fp8-KV-again, ignore-339->4-regex, KV-too-small),
the 27B W4A8-sqgptq-prepacked SERVES on one B70 @ ctx2048 (MAXLEN=2560 UTIL=0.97):
  c=1: 18.3 agg out t/s, TTFT 876ms, per-stream decode 20.73 t/s, tpot 48.2ms
  c=2: 32.1 agg, TTFT 1039ms, 18.30 decode
  c=4: 51.6 agg, TTFT 2209ms, 16.51 decode
  c=8: 67.8 agg, TTFT 4039ms, 12.24 decode
c1 decode 20.7 confirms the existing w4a8-q-prepacked ~20.9. Decode SAGS 20.7->12.2 across concurrency = single-card
KV pressure (25GB model leaves ~2GB KV on a 32GB card). This is the TP=2 case: split across both B70s -> ~12.5GB/card
-> abundant KV + ~2x weight BW/card (decode could approach the 14B rate). Next: Seguin TP=2 A/B (task #12) -- also the
only way to serve the 27B W8A8 (35GB).

### 2026-06-22 -- [TP=2 / task #12] TP=2+graph-capture FAILS with the oneCCL sycl_graph error (= Seguin's collective wall)
First TP=2 serve (P2P off, GRAPH=1) died: "oneCCL |CCL_SYCL| sched algorithms do not support sycl_graph recording".
The PIECEWISE graph capture can't record the allreduce on our vLLM 0.23 (lacks Seguin's clone-safe allreduce custom-op).
This is EXACTLY his first-order "graph break around the collective" finding -- not an xe P2P fault (dmesg clean). Workaround:
TP=2 eager (GRAPH=0) -> no capture -> no conflict. Relaunched TP=2 eager to get the W4A8 TP=2 ladder + unlock the W8A8.
Documented P2P_GPU.md sec H. Real fix (later): cherry-pick Seguin's allreduce patch (F.5). P2P on/off A/B pending eager-works.

### 2026-06-22 -- [TP=2 / task #12] BIG: SYCLKERNELS=1 unlocks graph-captured TP=2 allreduce (no vLLM patch) + TP1-vs-TP2
CCL_ENABLE_SYCL_KERNELS=1 + GRAPH=1 + TP=2 -> PIECEWISE capture SUCCEEDS, HEALTHY, dmesg clean. The sycl-kernel oneCCL
allreduce IS graph-recordable (the default sched algo is not -- that was the H.1 failure). A no-source-patch route to
graph-captured TP=2 on B70 (Seguin used a vLLM patch). 27B W4A8 @ctx2048 TP=2-graph: c1 dec 22.08 (vs TP=1 20.73, +6.5%
from 2x weight-BW), but TTFT 2858ms (vs 876, 3.3x worse) and c8 agg 34.3 (vs 67.8, 2x worse) -- the Gen3 cross-die
allreduce tax outweighs the BW edge for a model that fits one card. VERDICT: TP=1 wins for fit-one-card models; TP=2 is
the ENABLER for the >32GB W8A8 (launching that now). P2P-on A/B next to shrink the allreduce tax. Doc: P2P_GPU.md H.5/H.6.

### 2026-06-22 -- [research] Mined steveseguin/b70-optimization-lab (patches + MTP + qwen3.6-int8) -> docs/literature/10
Cloned repo, reviewed main + 5 codex/* branches. KEY finds for us:
(1) MTP (asked): his qwen3_5 MTP loader patch (VLLM_QWEN35_MTP_FORCE_FP8_BLOCK -> block-FP8 mtp head) + serve flag
    --speculative-config '{"method":"mtp","num_speculative_tokens":3}'. BUT MTP perf catastrophic (2.36 t/s) and the
    int8 branch names the ROOT CAUSE: a spec-decode VERIFIER / KV / input-position BOUNDARY bug ("target rejects the
    suppressed-bonus draft"), NOT draft quality -> gated on "oracle k=1 parity". Deepens our doc-09 MTP finding.
(2) branch codex/qwen36-quark-int8-tracking: he HAS the 35B-A3B int8 MoE SERVING via AMD **Quark** W8A8 (not GPTQ),
    ~99 t/s on TP4 + a persistent-W8A8-MoE-layerlet kernel -> proof docs/kernel/18 is real; Quark may be the cheap
    production path for our deferred Q6/Q7 (our GPTQ was multi-day). Also: W8A8 offsets regress (use symmetric, we do);
    TP2<TP4 (matches our TP=capacity finding).
(3) Applicable knobs: n-gram speculative (helped his Qwen FP8 +1.5 t/s -> try on our 27B W4A8), async+static-compile
    (1.7x), unset CUDAGRAPH_PARTITION_COLLECTIVES. Skip all VLLM_MINIMAX_*/GGML_* (not our arch). Corroborates our
    fp8-KV-broken + MTP-broken + TP2-capacity findings. Full catalog + try-list: docs/literature/10.

### 2026-06-22 -- [TP=2] 27B W8A8 now SERVED via TP=2 (graph+SYCLKERNELS); full 27B int8 served-ladder logged
W8A8 27B @ctx2048 TP=2: c1 dec 17.5 / c8 dec 6.1, TTFT 2728ms, agg 12.8->34.0. The 35GB W8A8 (single-card N/A) is now
servable thanks to the SYCLKERNELS=1 graph-capture unlock. Full picture (P2P_GPU H.7): W4A8-TP1 best for fit-1-card;
W4A8-TP2 +6.5% c1 dec but worse TTFT/conc; W8A8-TP2 17.5 (int8-wt < int4-wt decode, bytes-bound). Next: P2P-on A/B.

### 2026-06-22 -- [P2P probe] torch d2d xpu0<->xpu1 = ~1.35 GB/s, 452us latency (NOT peer-direct); oneCCL A/B running
Built 70_xpu_p2p_probe.py (direct d2d bandwidth + ping-pong). Result: torch .copy_ cross-device ~1.35 GB/s (vs ~13-15
peer-direct, ~7-8 host-bounce) + 452us/copy latency -> unpipelined host bounce, NOT peer DMA. P2PACCESS=1+drmfd didn't
help torch (torch.copy_ != oneCCL). dmesg clean (no fault). No ze_peer in image (need level-zero-tests for the
authoritative matrix). The REAL P2P-on test is the oneCCL serve A/B (W4A8 TP=2 P2PACCESS=1 vs off c1 22.1) -- launched. P2P_GPU H.8.

### 2026-06-22 -- [optimize] n-gram speculative decode VALIDATED on 27B W4A8: ~1.8x c1 decode (20.7->37.8 t/s)
Served 27B W4A8 (TP=1, graph) + --speculative-config '{"method":"ngram","num_speculative_tokens":3,"prompt_lookup_max":4,
"prompt_lookup_min":2}' (scripts/72_ngram_bench.sh). c1 per-stream decode 37.81 t/s vs no-spec 20.73 (~1.8x); tpot
26.45ms vs 48.2. HUGE decode win on repetitive output. CAVEATS: (1) c2/c4/c8 returned NA -- concurrent spec requests
failed (spec-decode+batch/KV issue, TODO); (2) ~1.8x is workload-inflated (the 35_sweep prompt has high n-gram
repetition -> high draft acceptance; Seguin saw only +1.5 t/s on diverse Qwen FP8); (3) aggregate out_tok_s at 128-tok
gens is TTFT-dominated (2541ms) so the decode win shows on LONG generations. NET: n-gram is a real free decode lever for
decode-bound/long-output workloads on the int8 path; concurrency path needs a fix. The first lever that doubled a number.

### 2026-06-22 -- [CORRECTION] n-gram speculative is NOT a reliable win -- earlier 1.8x was a short-output artifact
The 37.8 t/s (k=3, OUT=128) does NOT generalize. Retry at k=2, OUT=256: c1 decode 17.87 t/s -- SLOWER than no-spec
20.73. And c2/c4 still NA (spec-decode + concurrency broken on our XPU build, both runs). So the OUT=128 number was an
artifact of short output + high draft acceptance on the synthetic 35_sweep prompt; at realistic generation length the
draft acceptance falls and the failed-draft forward passes make n-gram NET NEGATIVE on diverse output. CORRECTED VERDICT:
n-gram speculative is a NICHE lever (helps ONLY highly-repetitive output like code/structured), NOT a general decode win
on the int8 path, and its concurrency path is unusable (NA at c>=2). De-prioritized. Robust wins stand (served ladders,
SYCLKERNELS TP=2 unlock, microbench, P2P verdict); n-gram is a logged NEGATIVE result. Retracting the FINDINGS 1.8x claim.

### 2026-06-22 -- [BREAKTHROUGH intel from user] int8 MoE serving + MTP BOTH already solved via intel/llm-scaler-vllm
Two community results that SUPERSEDE our "no int8 MoE kernel" + "MTP not viable" conclusions:
(1) int8 MoE SERVES: steveseguin runs Qwen3.6-35B-A3B **Quark W8A8 INT8** at 99.77 tok/s on 4xB70 via
    `intel/llm-scaler-vllm` + `--quantization quark --tensor-parallel-size 4 --language-model-only
    --compilation-config '{"cudagraph_mode":"PIECEWISE"}'`. The HF ckpt: nameistoken/Qwen3.6-35B-A3B-Quark-W8A8-INT8.
    => the int8 MoE kernel EXISTS in llm-scaler-vllm (Intel's XPU vLLM). Our docs/kernel/18 "build it" is mostly moot;
    the path is Quark-quantize + serve on llm-scaler. Image **intel/llm-scaler-vllm:0.14.0-b8.3.1 is ALREADY on host.**
(2) MTP WORKS on 4xB70 (user ytnszmy): Qwen3.6-27B BF16 TP=4, MTP unblocked from USERSPACE via:
    vllm_xpu_kernels v0.1.9 wheel + qwen3_5.py spec-wiring patch (vLLM #43565) + Half-KV; num_speculative_tokens=5,
    mean accept length 4.04 (88.9% accept @ spec=3); prefill ~2100 tok/s; image intel/llm-scaler-vllm:0.14.0-b8.3.
    => our doc-09 "MTP not viable" was a STACK gap (missing #43565 spec-wiring + the kernel wheel), NOT a B70 limit.
PLAN UPDATE: (a) OLMoE-1B-7B -> W8A8 -> serve on llm-scaler (validate small int8 MoE) -> bench; (b) then 35B via Quark
W8A8 + llm-scaler TP=2/4 -> unblocks Q6/Q7 SERVING; (c) study llm-scaler's int8 MoE kernel for our contrib port; (d)
MTP: re-test on llm-scaler:0.14.0-b8.3.1 + the #43565 patch (Half-KV, spec=5). docs/kernel/18 + docs/literature/09 to update.

### 2026-06-22 -- [doc] Captured the llm-scaler int8-MoE + MTP unlock -> docs/kernel/20 (supersedes kernel/18 + lit/09)
On-host probe of intel/llm-scaler-vllm:0.14.0-b8.3.1 (vLLM 0.14.1.dev): supports OlmoeForCausalLM +
Qwen3_5MoeForConditionalGeneration + Qwen3_5MoeMTP; quant methods incl quark, compressed-tensors, experts_int8, rtn,
moe_wna16. Quark QUANTIZER not in image (serves only). Documented Steve's 35B Quark-W8A8 99.77 t/s recipe + ytnszmy's
MTP recipe (vllm_xpu_kernels 0.1.9 + #43565 + Half-KV, 88.9% accept@3) + our ordered plan (OLMoE experts_int8 validate
-> 35B TP=2 serve -> MTP re-test -> port kernel to contrib). QUANTS_TODO sec 7 updated to point at doc 20.

### 2026-06-22 -- [hygiene] Archived deprecated models + bannered superseded docs
Per request, cleaned up (good hygiene):
- models/archive/ <- Qwen3.6-27B-W8A8-INT8-RTNtest (bad RTN, superseded by Q2 sqgptq), Qwen3.6-27B-W4A8-q-prepacked
  (no-SmoothQuant, superseded by Q3 sqgptq-prepacked which now serves), OLMoE-1B-7B-0924-Instruct (canned test vehicle).
  (Left other non-campaign models untouched: gemma-4-12B, Qwen2.5-GGUF, Qwen3-0.6B, 27B-FP8, etc. -- not mine.)
- Bannered SUPERSEDED: docs/kernel/18 (int8 MoE kernel "build it" -> llm-scaler has it, doc 20; now a port goal) +
  docs/literature/09 (MTP "not viable" -> works via doc-20 recipe). Kept both (cross-linked + reference value).
- Removed host scratch (tmp_cmp*.py, tmp_ign/cfgdiff.py, dl_olmoe.sh). Kept dl_q35.* (35B Quark download in flight, 25GB).

### 2026-06-22 -- [tooling] localmaxxing.com API puller for B70 community benchmarks -> scripts/75
Set up a read-only puller for the crowd-sourced localmaxxing.com inference leaderboard
(https://www.localmaxxing.com/en/api-docs). GET endpoints are public (no key); Cloudflare 1010-bans the
default Python-urllib UA so the script sends a browser UA. `scripts/75_localmaxxing.py` pages through
`GET /api/benchmarks?gpuName=Intel Arc Pro B70` (105 records as of today) and exposes:
summary (best out-tok/s per model/engine/quant/gpus) | leaderboard | configs (rows WITH the full
engineFlags.commandSnippet) | raw | save (data/localmaxxing/{*_raw.json gitignored, b70_summary.md tracked}).
Payoff: the `configs` view dumps the EXACT reproducible serve commands behind COMMUNITY_CONFIGS rows --
- steveseguin 35B-A3B Quark-W8A8-INT8 99.77 t/s: `vllm serve .../nameistoken--Qwen3.6-35B-A3B-Quark-W8A8-INT8...
  --quantization quark --tensor-parallel-size 4 --language-model-only --compilation-config '{cudagraph_mode:PIECEWISE}'`
  (served-name was `qwen36-35b-a3b-fp8` despite int8 ckpt -- exactly the [!] verify-the-checkpoint trap; this is the
  same ckpt our 74/doc20 work serves).
- RagingNoper 35B-A3B BF16 102.5 t/s graph-mode: docker run with `VLLM_XPU_ENABLE_XPU_GRAPH=1 VLLM_XPU_CUSTOM_AR=1
  CCL_ENABLE_SYCL_KERNELS=1` + DISABLE_ESIMD_* knobs + cudagraph FULL_DECODE_ONLY w/ explicit splitting_ops -- the
  un-gated XPU-graph + capture-safe all-reduce frontier (COMMUNITY_CONFIGS section B).
Documented in docs/COMMUNITY_CONFIGS.md (new "Live feed" section). Writes (submitting OUR numbers) would need
LOCALMAXXING_API_KEY=bhk_... from the dashboard; not needed for these read-only pulls.

### 2026-06-22 -- [progress+BLOCKED] Quark-W8A8-INT8 35B on 2x B70 TP=2: 5 blockers cleared, 1 real kernel gap
Ran steveseguin's exact Quark W8A8 INT8 recipe (ckpt nameistoken/Qwen3.6-35B-A3B-Quark-W8A8-INT8) at TP=2 on
intel/llm-scaler-vllm:0.14.0-b8.3.1 (scripts/74 rewritten + contrib/llm_scaler_quark_int8_moe). The earlier
JOURNAL "TP=2 collective init or OOM" guess was WRONG. The ckpt is GLOBAL int8 (W int8 per-channel symmetric,
IN int8 per-channel DYNAMIC; only vision excluded). Fixed FIVE blockers in sequence (full chain: kernel/20 sec 8
+ contrib README):
1. SYCL "No device of requested type available" in the model-inspect subprocess <- steve's env double-pins
   ONEAPI_DEVICE_SELECTOR + ZE_AFFINITY_MASK (his 4-card vals). Fix: expose both cards, no pin.
2. oneCCL zeMemOpenIpcHandle ZE_RESULT_ERROR_INVALID_ARGUMENT (TP=2 collective) <- steve's box CCL env wrong for
   our cards. Fix: our #41663 Battlemage env (CCL_TOPO_P2P_ACCESS=0, CCL_ZE_IPC_EXCHANGE=pidfd,
   CCL_ENABLE_SYCL_KERNELS=0, SYCL_UR_USE_LEVEL_ZERO_V2=0).
3. "Unsupported FusedMoe scheme" <- image quark_moe.py only wires fp8 MoE. Fix: contrib quark_moe.py adds
   QuarkW8A8Int8MoEMethod (mirrors the image's CompressedTensorsW8A8Int8MoEMethod) + int8 dispatch branch.
4. "No quark compatible scheme was found" (LINEAR) <- image has NO XPU int8 scaled-mm kernel (_POSSIBLE_KERNELS
   = CPU/CUDA/ROCM only). Fix: contrib quark.py adds QuarkW8A8Int8DequantXPU (int8->bf16 dequant GEMM; W8A16-eq)
   for the minority linear layers (linear_attn.*, mlp.shared_expert.*); experts stay true int8.
5. "Inference tensors do not track version counter" (torch.compile) <- dequant weight was an inference tensor.
   Fix: dequant under inference_mode(False) + --enforce-eager (B70 TP=2 capture blocked anyway).
=> Model now FULLY CONSTRUCTS, TP=2 collective up (backend=xccl world_size=2), all 7 shards load. Both patches valid.
BLOCKED at #6 (real image gap, NOT patchable on 0.14.1): first eager MoE forward ->
"AttributeError: '_OpNamespace' '_moe_C' object has no attribute 'topk_softmax'". vllm._moe_C DOES NOT EXIST in
this image (the compiled MoE op suite -- routing topk_softmax AND int8 fused-expert GEMMs -- was not built),
vllm_topk_softmax has no fallback, VLLM_XPU_USE_LLM_SCALER_MOE not honored. steve's 99.77 t/s used vLLM
0.20.2rc1.dev2 (newer build WITH XPU MoE kernels). FINISH PATH: pull a newer intel/llm-scaler-vllm tag (~0.20.x)
with _moe_C, then re-run scripts/74 (IMG=). Patches + recipe captured for that. Lease released; 0.14.1 verdict: int8
MoE EXECUTE is unsupported on this image (only the dispatch gaps were ours to fix).

### 2026-06-22 -- [SOLVED] int8 W8A8 Quark MoE 35B SERVES + GENERATES on 2x B70 (TP=2) -- via vLLM 0.23, no source build
Followup to the llm-scaler 0.14.1 dead-end (that image has no XPU MoE op suite: vllm._moe_C unbuilt ->
topk_softmax missing). User asked to source-build steve's 0.20.2rc1 stack; probing our EXISTING images found
we don't need to: **vllm-xpu-env:v0230 = vLLM 0.23.0** (NEWER than steve's 0.20.2) already ships
QuarkW8A8Int8MoEMethod + a _is_dynamic_per_token_w8a8 int8 LINEAR dispatch, AND routes the 256 int8 experts
through the Triton fused_moe_kernel on XPU (same path our int4 MoE uses -- contrib/vllm_moe_xpu). The
vLLM-version jump (0.14.1 _moe_C-only -> 0.23 Triton MoE) IS the unlock. Only gap on v0230: no XPU int8
scaled-mm LINEAR kernel (_POSSIBLE_KERNELS KeyError) -> ONE bind-mounted quark.py
(contrib/llm_scaler_quark_int8_moe/v0230/quark.py) reroutes the int8 linear layers (linear_attn.*,
mlp.shared_expert.*) to a weight-only int8->bf16 dequant GEMM (QuarkW8A8Int8DequantXPU); experts stay TRUE int8.
scripts/76_quark35b_v0230.sh (TP=2 #41663 env, enforce-eager). RESULT (served id qwen36-35b-a3b-quark-w8a8-int8,
fingerprint vllm-0.23.0-tp2): Model loading 17.54 GiB/card, KV 10.2 GiB, concurrency 89x@8192, backend=xccl
world_size=2, Triton fused_moe_kernel (E=256,N=256, int8), gen "The capital of France is" ->
" Paris, a city renowned for its rich history, culture, and iconic landmarks." Served EAGER; graph capture
(+617% on int4 MoE) + tuned E=256,N=256-int8 MoE config are the open perf levers. The Q6/Q7 int8-MoE SERVE
datapoint is finally REAL. (Also: Unraid host has no python3 -> serve scripts must not parse JSON with it;
use --served-model-name. Pruned ~22GB docker build cache earlier; 122G free.)

### 2026-06-22 -- [MTP M0 PASS] MTP wiring sanity gate GREEN on v0230 (no patch, no 0.14.x)
Ran MTP_TODO M0: serve Qwen3.6-27B (Lorbus W4A16 int4-AutoRound) on `vllm-xpu-env:v0230` + PIECEWISE graph +
`--speculative-config '{"method":"mtp","num_speculative_tokens":3}'` (script `m0_mtp_gate.sh`, wrapped in ONE gpu-run
lease serve+probe+stop). Chose W4A16-Lorbus as the gate vehicle (the most-proven v0230+GDN serve AND the exact Lorbus
45.2 t/s precedent) to isolate MTP-wiring from the W4A8 :int8g/KERNEL_SO/prepack confounders.
RESULT = **PASS**. The central question ("does v0230 load the MTP head, draft, and NOT crash on the GDN + spec-mask
path?") is YES, on STOCK v0230 with NO mtp_patch and NO 0.14.x kernels wheel:
- `Resolved architecture: Qwen3_5MTP`; `SpeculativeConfig(method='mtp', num_spec_tokens=3)` (plain `mtp` resolves fine --
  no need for `qwen3_5_mtp`). `quantization=inc`.
- `Loading drafter model...` -> `Detected MTP model. Sharing target model embedding + lm_head weights with the draft model.`
- PIECEWISE capture (sizes [1,2,4,8]) finished in 5s (2.37 GiB); `Application startup complete`; HEALTHY :18080.
- Coherent greedy gen (Fibonacci w/ docstring). `rejection_greedy_sample_kernel` Triton-JIT'd => the spec verify path ran.
- Model load 16.97 GiB / 58s. Lease held 293s total (~5 min real GPU), released clean.
KEY confirmation for M1: `splitting_ops` INCLUDES `vllm::gdn_attention_core_xpu` -> under PIECEWISE the GDN op is SPLIT
OUT of the captured graph (runs EAGER in the verify pass). This is exactly the codex/repo diagnosis of why PIECEWISE+MTP
is net-negative (-19%): the verify pass runs attention + GDN eager x(K+1). So M0 unblocks M1-M5, and the M1 frontier is
NOT "re-measure PIECEWISE" (known -19%) but "get attention+GDN INTO the captured graph" via `--attention-backend
TRITON_ATTN` -> FULL capture (host serve script's ATTN knob; vLLM PR #34482). Warnings seen (for M2): spec>1 "runs MTP
layer multiple times -> lower acceptance"; `max_num_scheduled_tokens=2048` suboptimal (raise max_num_batched_tokens).

### 2026-06-22 -- [MTP M1 HEADLINE WIN] single-card MTP is +72% (1.72x) on PIECEWISE -- REFUTES the old -19%
Ran MTP_TODO M1 (`m1_mtp_bench.sh`, one gpu-run lease, 3 configs, TTFT-cancelled decode = 256/(t_long-t_short),
greedy + ignore_eos so A/B emit identical token streams). 27B Lorbus W4A16 int4-AutoRound on `vllm-xpu-env:v0230`:
- **A MTP-OFF PIECEWISE:        30.84 t/s** (320 tok 10.57s)
- **B MTP-ON PIECEWISE spec=5:  52.95 t/s** (320 tok  6.00s)  -> **MTP x = 1.72 (+72%), NET-POSITIVE**
- C MTP-ON FULL via TRITON_ATTN: **CRASHED** at cudagraph capture -- `RuntimeError: spec_query_start_loc must have
  size [num_spec_decodes + 1]` in `_xpu_ops.py::_gdn_attention_core_xpu_impl` -> the v0230 baked gdn_attention spec op
  is NOT shape-compatible with FULL/FULL-decode cudagraph capture (a real op bug, not our config). Fallback ladder
  noted; **but PIECEWISE already wins, so FULL is upside, not a blocker.**
THE BIG REVERSAL: the long-standing "single-card MTP = -19% (25.5 vs 31.4 PIECEWISE)" is STALE. Our MTP-OFF (30.84)
matches their 31.4 baseline, but MTP-ON is 52.95, NOT 25.5. What changed: the **warmup-spoof PIECEWISE fix (commit
910182c)** -- the spec-decode decode batch (1+num_spec tokens) now CAPTURES under PIECEWISE instead of falling back to
eager, so the verify pass is no longer eager-attention-bound. So on the CURRENT stack, single-card MTP is already
strongly positive WITHOUT needing FULL capture. 52.95 t/s also BEATS the Lorbus 45.2 t/s single-card precedent (the
"number to beat"). Caveats: single measurement (M2 will take the median over the spec sweep + pull accept length from
/metrics -- my docker-logs accept grep returned empty, accept lives in /metrics per codex). Lease 739s, released clean.
Next: M2 spec sweep {2,3,4,5,6} (confirm spec=5, find max tok/s, log accept-vs-position) then M3 Half-KV.

### 2026-06-22 -- [MTP M2 DONE] spec sweep -> WINNER spec=4 (1.79x); accept_len rises but tok/s peaks at 3-4
`m2_spec_sweep.sh` (one lease, 5 serves, TTFT-cancelled decode + /metrics accept). 27B Lorbus W4A16 v0230 PIECEWISE:
| spec | decode_tps | MTPx | accept_len | accept_rate | per-DRAFT-position accept |
|------|-----------|------|-----------|-------------|---------------------------|
| 2    | 53.16     | 1.72 | 2.48      | 0.740       | 194/154 |
| 3    | 55.15     | 1.79 | 2.91      | 0.635       | 163/125/99 |
| 4    | **55.28** | **1.79** | 3.25  | 0.561       | 143/108/84/67 |
| 5    | 52.60     | 1.71 | 3.46      | 0.491       | 132/103/74/63/43 |
| 6    | 50.71     | 1.64 | 3.74      | 0.456       | 123/91/73/61/44/38 |
WINNER = **spec=4: 55.28 t/s (1.79x)** -- beats spec=5 (1.71x). Classic spec tradeoff: accept_len rises monotonically
(2.48->3.74) but decode_tps peaks at spec=3-4 then declines (verify cost grows faster than acceptance). Per-DRAFT-position
acceptance decays steeply within each verify step (pos0 ~80% -> last pos ~37%) -- that decay is WHY higher spec saturates.
(Note: this is per-DRAFT-position decay from /metrics; the Lorbus 86->65% "accept-vs-generation-position" decay is a
different axis -- would need short-vs-long-gen accept comparison; deferred, the per-draft decay already explains the knee.)
So the production single-card MTP pick is **spec=4 (or 3), ~55 t/s, 1.79x** -- NOT the spec=5 the localmaxxing rows used.
Next: FULL_DECODE_ONLY frontier retry at spec=4 (caps incl 5) + M3 Half-KV (fp8 KV accept vs full-KV 3.25).

### 2026-06-22 -- [MTP FULL retry FAIL + M3 DONE] FULL blocked (kernel bug); Half-KV is FREE for acceptance
Combined lease (`m3_and_full.sh`).
- **FULL_DECODE_ONLY retry (spec=4, caps incl 5): CRASHED, same `spec_query_start_loc must have size [num_spec_decodes+1]`**
  as plain FULL -- now inside the inductor-compiled `gdn_attention_core_xpu` for `layers.0.linear_attn`. Mode-independent
  (FULL == FULL_DECODE_ONLY) and capture-size-independent. The v0230 baked gdn_attention (xpu_kernels 0.1.9) spec op cannot
  run inside ANY captured graph. **CONCLUSION: FULL capture is BLOCKED on stock v0230; single-card MTP ceiling = PIECEWISE
  1.79x.** (Stretch to unblock: mount xpu_kernels 0.1.10 _xpu_C.so via KERNEL_SO -- deferred, PIECEWISE already wins.)
- **M3 Half-KV: PIECEWISE spec=4 + fp8_e4m3 KV -> decode 53.73 t/s, accept_len 3.29 vs full-KV 3.25 (delta +0.04) ->
  Half-KV OK, KEEP IT.** Half-KV does NOT depress MTP acceptance (Playbook A #5 confirmed), so the 2x-context trick is free.
SINGLE-CARD MTP CAMPAIGN COMPLETE: 27B W4A16 + MTP spec=4 + PIECEWISE + Half-KV = ~54-55 t/s, 1.79x, accept ~3.25-3.29,
beats Lorbus 45.2. Serving-ready. Next: M5 35B-A3B int4 MoE + MTP captured (single-card on :v0230moe; the int4 MoE already
captures at 56.8 t/s, avoiding the int8-MoE dequant-linear capture blocker; checkpoint HAS the mtp head -- verified config).

### 2026-06-22 -- [MTP M5 DONE] KEY FINDING: MTP is a DENSE lever (+79%), NOT a sparse-MoE lever (+3%)
`m5_moe_mtp.sh`: 35B-A3B int4-AutoRound MoE on :v0230moe, single-card, PIECEWISE capture + fp8 KV. NOVEL combo (no
community row has MoE + capture + MTP). Works, no crash, MTP head drafts.
- **A MoE MTP-OFF PIECEWISE: 66.82 t/s** (512 tok 7.76s)  [the MoE capture headline; > the 56.8 in SERVING.md]
- **B MoE MTP-ON spec=4:    68.83 t/s, accept_len 2.68** (512 tok 7.37s)  -> **MTP x = 1.03 (+3%), essentially FLAT**
THE FINDING: MTP gives +79% on the DENSE 27B but only +3% on the 35B-A3B MoE. Mechanism: the MoE activates only ~3B of
35B params per token, so per-token decode is already cheap (little weight-bandwidth to amortize), AND the spec-verify pass
runs the MoE forward x(1+spec) with a WIDER union of activated experts -> the verify overhead nearly cancels the draft
savings. So spec-decode ROI is architecture-dependent: BIG on dense (bandwidth-bound, 27B params/token), SMALL on
sparse-MoE (compute-light, 3B params/token). **Production implication: the 35B MoE headline is graph CAPTURE (66.8 t/s),
not MTP; the dense 27B headline is MTP (1.79x). Don't waste MTP plumbing on the MoE.** MTP campaign M0-M3+M5 COMPLETE
(M4 TP=2 = quick confirm remaining). Lease 740s, clean.

### 2026-06-22 -- [MTP M4 DONE -> MTP CAMPAIGN COMPLETE] TP=2 MTP is dead (spec-allgather not graph-capturable)
`m4_tp2_mtp.sh`: 27B int4 W4A16 TP=2 on v0230 + CCL_ENABLE_SYCL_KERNELS=1 + PIECEWISE.
- **A TP=2 MTP-OFF PIECEWISE: 26.96 t/s** (512 tok 18.74s) -- 0.87x of single-card 30.84 (the allreduce tax; TP=2 is a
  CAPACITY play, not speed, for a model that fits one card -- confirmed).
- **B TP=2 MTP-ON spec=4: CRASH** -> `oneCCL ccl_allgather_impl: |CCL_SYCL| sched algorithms do not support sycl_graph
  recording, please use sycl_algorithms`. So CCL_ENABLE_SYCL_KERNELS=1 makes ALL-REDUCE graph-safe (MTP-off captures), but
  the spec-decode path adds an ALLGATHER collective that is NOT graph-capturable under our CCL config. This REFRESHES the
  old Lorbus TP2-MTP [NEG] with a precise root cause (theirs was "slower"; ours CRASHES on the spec-allgather under capture).
  Unblocking needs RagingNoper's capture-safe collectives (custom xpu_communicator) OR TP=2 MTP eager (slow). Not worth it.
VERDICT: **single-card DP-replica MTP is decisively the path; TP=2 MTP is dead on stock v0230.**

### 2026-06-22 -- [MTP CAMPAIGN SUMMARY] M0-M5 COMPLETE. Headline: single-card dense 27B MTP = 1.79x (55 t/s)
The whole MTP_TODO queue is closed. Net production guidance:
| dimension | result | takeaway |
|---|---|---|
| single-card dense 27B W4A16 + MTP spec=4 PIECEWISE | **55.28 t/s vs 30.84 = 1.79x** | THE WIN; beats Lorbus 45.2; serving-ready |
| ctx=2048 random 2048/128, C1 | `tg` **46.69 vs 29.78**; agg out 30.99 vs 23.10; TTFT 1.410s vs 1.275s | MTP wins single-stream decode, costs a little TTFT |
| ctx=2048 random 2048/128, C4 | `tg` **16.09 vs 19.54**; agg out 40.56 vs 51.69; TTFT 4.444s vs 3.398s | MTP loses under concurrent ctx=2048 fan-out |
| spec-token | spec=4 best (3-4 plateau); accept rises 2.48->3.74 but tok/s peaks at 3-4 | use spec=4 |
| Half-KV (fp8 KV) | accept 3.29 vs 3.25 full = FREE | keep Half-KV for 2x context |
| FULL capture | BLOCKED (gdn_attention spec op can't run in ANY captured graph, v0230 0.1.9) | PIECEWISE is the ceiling |
| TP=2 MTP | DEAD (spec-allgather not graph-capturable; MTP-off TP2 already 0.87x) | single-card DP per replica |
| 35B-A3B MoE + MTP | +3% FLAT (sparse 3B-active) | MoE headline is CAPTURE not MTP |
The old "single-card MTP = -19%" was STALE (pre warmup-spoof-PIECEWISE-fix 910182c). The campaign refuted it and delivered
the first real single-card MTP multiplier on this stack. PRODUCTION ACTION: enable MTP spec=4 on the daily driver (the
27B int4 W4A16 DP replicas) for +79% interactive single-stream. Scripts: 77-83. Logging table: MTP_TODO.

### 2026-06-22 -- [QUANTS Q8 BREAKTHROUGH] AutoRound int4 on the qwen3_5 VLM -- the MLLM-calib block is BEATEN
The Q2/Q4 "AutoRound MLLM-calib blocked on VLM -> fell back to GPTQ" wall is NOT a real blocker -- it was a missing-API
problem. SOLVED (scripts/84_q8_qwable_int4.py, smoke fully validated, full run + Q5 prepack now launched):
- **Root cause:** AutoRound 0.13.1 auto-detects the qwen3_5 VLM and forces MLLM mode (`entry.py L587: Using MLLM mode`),
  then `quantize()` asserts `processor should not be None`. `quant_nontext_module=False` ALONE does NOT dodge it.
- **The dodge (works):** load `AutoProcessor.from_pretrained(SRC)`, construct via `AutoRoundMLLM(model, tokenizer,
  processor=proc, quant_nontext_module=False, dataset=<text list>, scheme="W4A16", layer_config={241 modules -> bits:16})`,
  then `quantize()` + `save_quantized(format="auto_round")` (the inc-servable path, NOT llm_compressor). `AutoRoundMLLM`
  is a deprecated alias forwarding to AutoRound; MLLM kwargs (processor, quant_nontext_module) pass via **kwargs (codex).
- **Smoke (iters=2) PROVED end-to-end:** quantizes all 64 layers (DeltaNet linear_attn correctly SKIPPED -> bf16; small
  int4 losses 0.0003->0.007), `save_quantized format=auto_round` -> `RESULT_Q8: DONE`, 6 int4 shards + quantization_config
  + the 348 vision/mtp tensors copied verbatim (bf16). Minor warning: Qwable tokenizer wants `fix_mistral_regex=True`
  (calib-tokenization nicety, non-fatal).
- **LAUNCHED:** full Q8 run (iters=200 nsamples=128, ~4-8h, GPU lease) + Q5 prepack (Qwable W4A8 sqgptq 33G->~25G int4-pack,
  CPU container NO lease, parallel) -- scripts/84,84b + q5_prepack.sh.
**Bigger implication:** this MLLM-dodge means AutoRound CAN quantize the qwen3_5 VLMs (27B base + Qwable), so RESEARCH_TODO
Track 3 (AutoRound vs GPTQ on the 27B/Qwable) and the QUANTS Q2/Q4 W8A8-AutoRound (which fell back to GPTQ) are now
UNBLOCKED -- re-runnable with the same processor+AutoRoundMLLM recipe. Documented in QUANTS_TODO + docs/kernel/15.

### 2026-06-22 -- [QUANTS Q5 prepack DONE + Q8 full calib-batching fix]
- **Q5 prepack DONE:** `q5_prepack.sh` (CPU, no lease, scripts/85) packed the Qwable W4A8 sqgptq -> int4-packed:
  1107 tensors, 256 weights packed, **32.9 -> 24.1 GiB** -> `Qwable-5-27B-Coder-W4A8-sqgptq-prepacked` (25G,
  is_prepacked_w4a8=True). The "repack" item is closed; serve-ready.
- **Q8 full crashed (then fixed):** the iters=200/nsamples=128 full run died at layer 0 calib with
  `RuntimeError: Sizes of tensors must match ... Expected 23 but got 24` -- my in-script text-list calib has VARIABLE
  tokenized lengths, and AutoRound's batched calib cat (nsamples>=batch_size=8) can't stack 23-tok vs 24-tok samples.
  The smoke (nsamples=8) dodged it by luck. FIX (codex's documented path): `CALIB_DS=NeelNanda/pile-10k` -- AutoRound
  tokenizes + chunks pile to UNIFORM seqlen. scripts/84 now defaults to pile-10k (CALIB_DS=list for the old behavior).
  Full RELAUNCHED; the calib phase (~5 min) validates the fix before the 4-8h iters phase.
LESSON for AutoRound: pass a HF text dataset name (pile-10k), NOT a variable-length in-script list, for nsamples>=batch_size.

== 2026-06-22 :: QUANTS Q8 -- Qwable-5-27B-Coder int4-AutoRound PRODUCED + VALIDATED (the QUANTS queue is CLOSED) ==
config -> the full AutoRound int4 run completed after a 3rd fix: low_gpu_mem_usage=True (the iters=200 gradient loop
  exhausted Level-Zero resource HANDLES -> UR_RESULT_ERROR_OUT_OF_RESOURCES error 40 at ~layer 3, peak_vram only 23GB).
  So the full fix chain was MLLM-dodge (AutoProcessor+AutoRoundMLLM) -> pile-10k calib -> low_gpu_mem_usage. RESULT_Q8 DONE,
  25G out (6 shards), mtp.fc + visual + mtp norms copied bf16; int4 = language_model MLP+full-attn + visual.blocks + mtp.layers.
result (serve) -> FIRST serve attempt crashed: `AttributeError: 'Qwen3_5TextConfig' has no attribute 'vision_config'`.
  ROOT CAUSE (diagnosed from the weights, ground truth via the .qweight/.weight tensors): AutoRound MLLM-save writes a
  checkpoint with MULTIMODAL weight naming (model.language_model.layers/visual/mtp) but a FLAT qwen3_5_text config.json
  (architectures=Qwen3_5ForCausalLM, NO vision_config), AND extra_config bf16-overrides mis-named model.layers.* not
  model.language_model.layers.* . FIX (no re-quant): served config.json = base Qwable MULTIMODAL config
  (Qwen3_5ForConditionalGeneration + vision_config + text_config) + the quant's quantization_config with extra_config keys
  renamed model.layers.->model.language_model.layers. -- matches the PROVEN Lorbus 27B int4-AutoRound structure.
  Tool: scripts/87_fix_autoround_vlm_config.py. Installed on host (orig -> config.json.textonly.bak).
result (validated) -> 2nd serve: HEALTHY, served id qwable-27b-int4, quantization=inc auto-detected, weights 10.29s,
  PIECEWISE graph capture 3.55 GiB. **decode = 29.13 t/s** (TTFT-cancelled, GRAPH=1, single-card) == Lorbus 27B int4 ~30.8 ref.
  Generation verified by the bench (352 real tokens). 1-time inductor compile ~303s (cached after). Lease freed clean.
verdict -> Q8 DONE/VALIDATED. The QUANTS_TODO queue is CLOSED (Q0-Q5,Q8 done; Q6/Q7 35B correctly deferred). Qwable now has
  a one-card quality serve (the only int4-AR for this coder model; none on HF). NEW REUSABLE FINDING = the serve-side
  counterpart to the MLLM-calib dodge: any AutoRound-on-qwen3_5-VLM checkpoint needs the scripts/87 config repair to serve.
  This also de-risks the now-unblocked Q2/Q4 W8A8-AutoRound re-runs + RESEARCH_TODO Track 3.

== 2026-06-22 :: POST-Q8 FRONTIER -- FULL-capture MTP chased to ground: it is KERNEL-gated (RESEARCH_TODO Track 1d CLOSED) ==
config -> Lorbus 27B int4, cudagraph_mode=FULL_DECODE_ONLY + --attention-backend TRITON_ATTN + MTP spec=5 + caps
  [1,2,4,6,8,16,32] (incl 1+spec=6), WITH a port of vllm-ascend PR #7148 (scripts/88) appended to triton_shim/sitecustomize.py
  -- gates the dispatcher assert `num_tokens_padded % uniform_decode_query_len == 0` instead of crashing on it.
command -> mtp_full_retry.sh under gpu-run.
result -> the #7148 dispatcher patch LOADED in all procs and WORKED: capture got PAST the dispatcher and reached
  `Capturing CUDA graphs (decode, FULL): 0/3`, THEN crashed in the BAKED KERNEL:
  torch.ops.vllm.gdn_attention_core_xpu -> vllm/_xpu_ops.py:151 -> torch.ops._xpu_C.gdn_attention ->
  RuntimeError: spec_query_start_loc must have size [num_spec_decodes + 1].
verdict -> DEFINITIVE BISECTION: FULL-capture MTP on B70 is **kernel-gated** (the _xpu_C.gdn_attention op in
  vllm_xpu_kernels 0.1.9), NOT dispatcher-gated and NOT fixable from the vLLM Python layer. TRITON_ATTN does not help --
  the GDN decode core always routes through the baked gdn_attention_core_xpu op. So **PIECEWISE 1.79x is the CONFIRMED
  single-card ceiling on stock v0230.** The fix requires Intel (vllm_xpu_kernels). Filed-ready issue: docs/kernel/21.
  This closes the long-standing FULL/TRITON_ATTN open lever (RESEARCH_TODO Track 1d) with a concrete, evidenced answer.
  Cleanup: restored triton_shim/sitecustomize.py from .bak (patch preserved in scripts/88); lease freed.

== 2026-06-22 :: CORRECTION -- Q8 Qwable int4 is BROKEN (XPU-calib corruption); "VALIDATED 29.13 t/s" was a FALSE POSITIVE ==
config -> ran the Q8 HumanEval+ accuracy eval (evals/ harness, sandboxed via evalplus-sandbox:0.3.1) against the served
  qwable-27b-int4 (served-id verified). SMOKE (5 problems) -> pass@1 = {base:0.0, plus:0.0}, gen 361s (~72s/problem).
result -> the generated "solutions" are 2048 chars of pure `!!!!!`. A direct greedy completion ("def add(a,b):") ALSO
  returns pure `!` -> immediate forward-pass degeneration -> the MODEL IS BROKEN, not a harness/format issue.
bisection -> the served checkpoint's config + structure MATCH the working Lorbus 27B int4 EXACTLY: quant_method=auto-round,
  bits=4, group_size=128, sym, data_type=int, packing_format=auto_round:auto_gptq, tie_word_embeddings=False, lm_head.weight
  present, scripts/87 config-repair applied. Same inc/AutoRound serve path that serves Lorbus coherently (daily driver).
  So it's NOT the config/repair/serve -> it's the WEIGHTS. The only thing unique to Q8 vs Lorbus: I ran AutoRound's
  gradient calibration ON THE XPU (device_map=auto over both B70s) + low_gpu_mem_usage offloading. Lorbus was quantized
  on CUDA/CPU (community).
verdict -> **Q8 is INVALID (broken weights). My earlier "VALIDATED 29.13 t/s" was WRONG** -- the decode bench used
  ignore_eos + token-COUNT, which masks garbage tokens; the coherence text-print had failed and I wrongly waved it off.
  Root cause = AutoRound XPU-calibration corruption (CONFIRMS RESEARCH_TODO Track 3e). FIX = re-quant on CPU/CUDA, serve
  on B70. Working 27B int4 remains the community Lorbus. **LESSONS: (1) token-throughput != coherence -- always read text /
  run a real eval (the eval CAUGHT what the bench missed); (2) do NOT run AutoRound calibration on the B70 -- quantize on
  CPU/CUDA.** Corrected: QUANTS_TODO Q8 ([!] BROKEN), RESEARCH_TODO Track 3e ([x] confirmed). Serve stopped, lease freed.
  UNAFFECTED: the MTP campaign (1.79x) + FULL-capture verdict used the WORKING Lorbus int4 -> those results STAND.

== 2026-06-23 :: W8A8 AutoRound vs GPTQ (Track 3b) -- GPTQ slightly WINS; "autoround supersedes" REFUTED ==
config -> served Qwen3-14B-W8A8-autoround (compressed-tensors int-quantized) on vllm-xpu-env:int8 (TRUE W8A8:
  "Selected XPUInt8ScaledMMLinearKernel for CompressedTensorsW8A8Int8"), enforce-eager, served-id verified, coherence
  pre-checked (clean code). run_evals.py Tier-1 HumanEval+ 164, sandboxed (evalplus-sandbox:0.3.1).
result -> **w8a8-autoround pass@1 = 0.909 base / 0.872 plus** vs **w8a8-gptq 0.921 / 0.890** (SUMMARY.md, the 06-20
  measured winner). GPTQ ahead by +1.2 base / +1.8 plus (near HumanEval CI for n=164, but consistent direction).
verdict -> GPTQ >= AutoRound at W8A8; AutoRound does NOT supersede GPTQ here. Matches the field (Track 3 / doc07 S3.4:
  weight-rounding barely matters at int8; the gap is activations, not rounding -- so Hessian-OBQ GPTQ marginally edges
  optimization-based AutoRound). AutoRound W8A8 is SOUND + coherent (int8 weights survive even XPU calib, unlike the int4
  Q8 which broke -> int4 is the fragile case, int8 is forgiving). NOTE: we'd archived+deleted the gptq W8A8 checkpoint
  (keeping autoround on the "supersedes" hypothesis); this eval shows gptq was marginally BETTER -- but the difference is
  ~CI-noise, the measured gptq numbers persist in SUMMARY.md, and gptq is re-quantizable. Eval harness + sandbox: WORK.
  GPU0 freed, lease released. (The v0230 dense-int8 capture test ran in parallel on GPU1.)

== 2026-06-23 :: REORG -- rdy_to_serve golden shelf + _common engine + bin/ tools (anti-clobbering contract) ==
motivation -> we kept clobbering working serve recipes/patches/images while testing new stuff. Established a 4-tier
  layout by MUTABILITY CONTRACT (ORGANIZATION.md): scripts/ = append-only lab notebook; bin/ = stable shared tools;
  images/ = immutable digest-pinned recipes (future); rdy_to_serve/<m>/ = VERIFIED self-contained golden serves.
config -> rdy_to_serve/_common/lib.sh = shared model-agnostic serve engine (docker-run builder + graph-capture flags +
  #41663 multi-GPU env + health wait + gen probe + bench), ported from the proven host 30_serve_w4a8_graph.sh. Each
  model serve.sh = thin (env + local patches/ + `source ../_common/lib.sh; b70_dispatch`). Built 3 shelf dirs:
  qwen36-27b-int4 (:v0230, 1 card, PRIMARY), qwen36-35b-a3b-int4 (:v0230moe, 1 card, FASTEST), and refactored
  qwen36-35b-a3b-quark-w8a8-int8 (:v0230, TP=2) onto _common. bin/: moved gpu-run/35_sweep/64_dp/dp_nginx + pulled the
  CANONICAL host 30_serve (the repo w4a8/ copy had DRIFTED) + serve-sweep gate harness. SERVING.md -> golden-path index;
  CLAUDE.md -> the contract; contrib quark README -> blessed-copy banner. Host is NOT a git repo (hand-synced via tar-pipe).
result -> tested on the 2 free B70s (gpu-run lease): (1) eager single-card 27b-int4 (card0:8001) + 35b-a3b-int4
  (card1:8002) SIMULTANEOUSLY -> both HEALTHY, ids ok, coherent "Paris" gens. (2) eager TP=2 quark W8A8 -> HEALTHY,
  quark.py per-container mount ok, gen "Paris, a city renowned for its rich history...", 233s. (3) 27b GRAPH=1 PIECEWISE
  capture (full capsizes) -> HEALTHY, coherent gen.
  BUG CAUGHT BY THE SMOKE GATE: b70_serve built the EAGER/CC arrays but never appended them to the vllm args, so
  GRAPH=1 served WITHOUT --compilation-config (capture was a silent no-op -> the default `bash serve.sh` would ship
  ~7.8 t/s eager, not ~30.8 captured) and GRAPH=0 lacked --enforce-eager. Fixed: ARGS+=(EAGER CC). Also fixed an empty
  MOUNTS element that docker read as the image ("invalid reference format"). Both re-verified GREEN.
gotchas -> (a) bash CANNOT export arrays -> serve.sh sets MOUNTS as a plain array (sourced, so visible). (b) release a
  STUCK serve by killing its `gpu-run` PID, NOT `docker rm -f` the container (removing it makes serve.sh's wait loop
  poll a missing container and keep holding the flock). (c) gpu-run --status can show a STALE owner if a gpu-run dies
  without its EXIT trap (ssh drop) -- flock is actually free; `: > gpu.lock.owner` to clear. (d) NEVER `pkill -f <str>`
  where <str> is in your own ssh command line (kills your session).
verdict -> the golden shelf works; the _common engine is validated across eager-single / eager-TP2-patched /
  graph-capture. :int8g is GONE from the host -> the int8-kernel family (27B/14B W4A8/W8A8) is BLOCKED pending an
  image rebuild (documented in rdy_to_serve/README status table, not enshrined as fake-verified). Commits: a8671be
  (ORGANIZATION.md), b113350 (_common+dirs), 0620757 (bin/+contract), 8e81bfa (capture-flags fix). Lease freed, host clean.
  DEFERRED: images/ Dockerfiles + :int8g rebuild; full SERVING trim + daily_driver calling the golden path (it still
  uses the host flat 30_serve engine); dropping the NN_ numbers off bin/ tools + scripting the host sync.

== 2026-06-23 :: int8 family UNBLOCKED (:int8g rebuilt) + daily_driver picker refactor ==
motivation -> :int8g (our INT8 W8A8 kernel image) had been clobbered off the host -> the int8 quant family
  (27B/14B W4A8/W8A8) was unservable. Rebuild it as the working int8 baseline (for the planned int8
  GEMM/GEMV optimization research), shelve the family, and make the daily driver model-pickable.
config -> (1) rebuilt :int8g via the bake recipe (FROM :int8 [v0230+oneDNN INT8 W8A8 GEMM .so +
  XPUInt8ScaledMMLinearKernel] + swap in the register_fake-enabled xpu_int8.py so XPU graph capture can
  trace the custom int8 ops). Op check int8_gemm=True fused_quant=True. New image 8e25c758...; codified in
  images/int8g/{build.sh,README} (dated-immutable-tag + digests; supersedes scripts/52). (2) Built 3 golden
  dirs on :int8g: qwen3-14b-w8a8 (compressed-tensors W8A8, no prepack), qwen3-14b-w4a8 (prepack: mounted
  loader+scheme patches + VLLM_W4A8_PREPACKED=1 via new _common DOCKER_ENV passthrough), qwen36-27b-w4a8
  (prepack + GDN: mount the GDN-enabled _xpu_C.abi3.so + libgdn over the baked GDN-OFF build). (3) Refactored
  daily_driver_serve.sh into a thin picker: DD_MODEL=<rdy_to_serve dir> served via the model's own serve.sh
  (zero recipe duplication), 2x data-parallel (or DD_REPLICAS=1 for TP=2), DD_MTP/DD_MAXLEN/DD_ENV knobs.
result -> all 3 int8 models smoked GREEN on :int8g (eager, single card): 14b-w8a8 62s, 14b-w4a8 83s,
  27b-w4a8 120s -- HEALTHY, correct ids, coherent "Paris" gens. The 27B exercised prepack AND the GDN .so
  mount (loaded past the gated-delta-net decode op). daily_driver picker validated end-to-end: brought up
  qwen36-27b-int4 as 2x DP + nginx proxy on :18080 (eager test), proxy HEALTHY, served id correct, both
  replicas 200-OK on /health,/v1/models,/v1/completions.
gotchas -> (a) vLLM 0.23 REJECTS fp8 KV on the 27B-W4A8 ckpt ("fp8_e5m2 kv-cache is not supported with
  fp8 checkpoints") -> 27B-w4a8 serves fp16 KV (SERVING's old fp8-KV recipe was a different image/vLLM).
  (b) "Unknown vLLM env var VLLM_W4A8_PREPACKED" is COSMETIC -- the prepack patch reads os.environ directly;
  the 14B W4A8 served fine with it. (c) the daily_driver poller exit-1 was a teardown RACE (I ran stop while
  it polled), not a bug -- logs show clean 200-OK serving until SIGTERM.
verdict -> int8 baseline WORKS and the family is shelved (rdy_to_serve now has 6 verified models). Solid
  foundation for the int8 GEMM/GEMV optimization research. Commits: 304ac3c (:int8g + 14b-w8a8), 77f2cac
  (full int8 family + images/int8g), this (daily_driver picker + SERVING + journal). NOTE: 27B-W4A8 is
  SECONDARY (w4a16 int4 decodes faster, ~30.8 vs ~20.9); smokes were EAGER (the GRAPH=1 capture default is
  per-model-proven on 27b-int4 but not re-bench'd for the int8 family this pass). Lease freed, host clean.

== 2026-06-23 :: per-card gpu-run lease + daily_driver 3 modes + captured int8 baseline ==
config -> (1) gpu-run rewritten for PER-CARD leasing: default locks BOTH cards (backward compatible: TP=2/
  DP/PP), `--card N` locks ONLY card N. Two flocks gpu.lock.0/.1 (fd 8/9); --status is per-card. Synced to
  BOTH the flat host path AND bin/ (kept identical -- else two lock schemes would not see each other).
  (2) daily_driver three modes: DP=2 (default, replicate a fits-one-card model, ~2.1x), TP=2
  (DD_REPLICAS=1 + a TP=2 model, both cards), and ONE-CARD (DD_CARD=N -> pin + lease only that card, leaving
  the other free for `gpu-run --card <other>` experiments). (3) GRAPH=1 capture bench on qwen3-14b-w8a8.
result -> per-card lease VERIFIED on host: --card 0 + --card 1 ran CONCURRENTLY (3s), two --card 0
  SERIALIZED (6s), default WAITS for a held card. Owner record fixed to show the real cmd.
  CAPTURED INT8 W8A8 BASELINE (qwen3-14b-w8a8, :int8g, PIECEWISE, 512/128, fp16 KV, card0):
    c1 per-stream decode 25.54 t/s (agg 25.13, ttft 121ms) ; c2 26.22 (agg 51.12) ; c4 25.52 (agg 97.83).
  Clean linear aggregate scaling, NO recompile stall (capture sizes covered the conc levels). This is the
  number to beat for the int8 GEMM/GEMV optimization research.
verdict -> all 3 daily-driver use cases supported with one-line invocations (see daily_driver_serve.sh
  header + docs/SERVING.md). The captured int8 baseline is confirmed. Lease freed, host clean.

== 2026-06-23 :: Qwen3.6-27B int4 MTP ctx=2048 C1/C4 follow-up ==
motivation -> the MTP campaign proved a strong TTFT-cancelled single-stream decode win (55.28 vs 30.84 t/s), but
  docs did not have the requested visual table for ctx=2048 with pp, TTFT, tg, and C4 concurrency.
config -> used the free B70 only via `gpu-run --card 1`; did NOT edit `rdy_to_serve/`. Served the golden
  `rdy_to_serve/qwen36-27b-int4/serve.sh` recipe on card1, port 18081, `GRAPH=1`, `MAXLEN=8192`, `MAXSEQS=8`,
  `CAPSIZES=1,2,4,8`, fp16 KV. Bench was `vllm bench serve` random 2048 input / 128 output, `--ignore-eos`,
  C=1 and C=4. MTP row used `MTPTOK=4 COMPILESZ=`. Note: the first wrapper used `docker exec -i`, which consumed
  the SSH heredoc after one row; reran the remaining rows with plain `docker exec`. Another agent may also have
  killed an earlier card1 container; final rows below are from healthy serves.
result ->
  config       C  pp tok/s  TTFT ms  TPOT ms  tg tok/s  agg out tok/s  total tok/s  accept_len
  no-MTP       1  1605.8    1275.36  33.58    29.78     23.10          392.71       -
  MTP spec=4   1  1453.0    1409.52  21.42    46.69     30.99          526.84       2.92
  no-MTP       4  2410.9    3397.89  51.17    19.54     51.69          878.79       -
  MTP spec=4   4  1843.5    4443.65  62.15    16.09     40.56          689.56       2.41
verdict -> MTP spec=4 is a C1 interactive decode win at ctx=2048 (`tg` +57%, aggregate out +34%), despite a small
  TTFT/pp cost. It is NOT a C4 throughput win for random ctx=2048 prompts: aggregate out -22%, `tg` -18%, TTFT +31%,
  and accept_len falls to 2.41. Keep `DD_MTP=1` for one/few interactive coding streams; leave it OFF for C4+ batch
  or fan-out unless the exact workload re-benches positive. Host result CSVs:
  `results/mtp_table_qwen36-27b-int4_ctx2048_20260622_144303.csv`,
  `results/mtp_table_qwen36-27b-int4_ctx2048_continue_20260622_144611.csv`,
  `results/mtp_table_qwen36-27b-int4_ctx2048_mtp_20260622_144813.csv`. Container stopped; card1 lease freed.

== 2026-06-23 :: 27B W4A16 compressed-tensors FIXED (text-only-checkpoint load bug; int4 kernel exonerated) ==
goal -> serve `Qwen3.6-27B-W4A16` (compressed-tensors pack-quantized) on the B70 for format parity (and as
  the W4A4-research substrate). It had been written off as "won't serve (4304 dim)". Card 0 only (another
  agent on card 1 -- per-card gpu-run `--card 0`). Full log: docs/kernel/22_compressed_tensors_w4a16_xpu.md.
config -> the checkpoint is a LANGUAGE-MODEL-ONLY quant (architectures Qwen3_5ForCausalLM, all 1363 tensors
  model.language_model.* + lm_head, ZERO vision). vLLM resolved it to the VL class and built a weightless
  vision tower. Fix = a sitecustomize+module shim (rdy_to_serve/qwen36-27b-w4a16/patches/) doing 5 things:
  (1) register the real text arch -> no vision tower; (2) is_hybrid=True marker -> GDN/mamba KV-cache setup;
  (3) graft get_mamba_state_{shape,dtype,copy}_from_config from the VL class; (4) supports_mrope + a
  text-only get_mrope_input_positions (== the VL text path); (5) load_weights remap model.language_model. ->
  model. . Image :v0230 (GDN). UTIL 0.95 (24.35 GiB model is tight; GRAPH=1 OOMs at 0.90).
result -> after (1)-(4) it SERVED HEALTHY but generated "!!!!" garbage. Two numerical unit tests of
  torch.ops._xpu_C.int4_gemm_w4a16 on card 0 EXONERATED the int4 kernel: it needs the weight in NT format
  (weight_packed.t() as a NON-contiguous view -- `.contiguous()` raises "Int4 weight must be in NT format!"
  -- + scale.t().contiguous()), and then matches a reference dequant matmul on synthetic AND a real layer
  (maxerr 0.0156). The REAL bug: every weight skipped ("language_model.<x> not found in params_dict") ->
  random init -> garbage. The (5) name remap fixed it: skipped-weight warnings 0, coherent gens (Paris /
  the ocean / RGB), EAGER and GRAPH=1 PIECEWISE captured.
verdict -> 27B W4A16 compressed-tensors SHELVED + verified (rdy_to_serve/qwen36-27b-w4a16). The "4304 dim"
  story was a red herring (that assert was the weightless vision tower, also fixed by loading text-only).
  LESSONS (research): (a) "serves HEALTHY + coherent infra logs" != "weights loaded" -- always grep the load
  for `not found in params_dict`/`skip loading` when re-homing a checkpoint; all-same-token output is the
  random-weights signature (cf. the Q8 false positive). (b) the XPU compressed-tensors int4 W4A16 GEMM
  (int4_gemm_w4a16) is CORRECT -- usable for W4A16 research; weight is NT-format (transposed view).
  (c) Lorbus int4 (works) uses INC auto_round_kernel + Triton/FLA GDN, a DIFFERENT path from XPUwNa16.
  Commits: e53f6f8 (fix+findings) + this (GRAPH=1 default UTIL=0.95 + journal). Card 0 freed; card 1 untouched.

== 2026-06-23 :: Qwen3.6-27B int4 MTP spec sweep at ctx=2048 (full KV vs Half-KV) ==
motivation -> fill the requested table for PIECEWISE + MTP spec=3/4/5 and PIECEWISE + MTP spec=3/4/5 + Half-KV,
  compared against the best no-MTP baseline at real ctx=2048. Question: does MTP only help decode, or is there a
  prefill/TTFT tax that changes actual usability? Also reconcile the external Lorbus single-card "45.2 tok/s" row.
config -> used only the free GPU1 via `gpu-run --card 1`; did NOT edit `rdy_to_serve/`. Served the golden
  `rdy_to_serve/qwen36-27b-int4/serve.sh` recipe on card1, port 18081, image `vllm-xpu-env:v0230`, `GRAPH=1`,
  `MAXLEN=8192`, `MAXSEQS=8`, `CAPSIZES=1,2,4,8`, NOMM=1, tool/reason parsers on. Bench was `vllm bench serve`
  random 2048 input / 128 output, `--ignore-eos`, C=1, N=8. MTP rows used `MTPTOK=3/4/5 COMPILESZ=`. Half-KV rows
  additionally used `KVDTYPE=fp8_e4m3`. Host CSV:
  `/mnt/vm_8tb/b70/results/mtp_spec_sweep_qwen36-27b-int4_ctx2048_20260622_150846.csv`; repo copy:
  `results/mtp_spec_sweep_qwen36-27b-int4_ctx2048_20260623.csv`.
result ->
  config              KV        pp tok/s  TTFT ms  TPOT ms  tg tok/s  agg out tok/s  total tok/s  accept%  accept_len
  no-MTP              fp16      1608.2    1273.49  32.81    30.48     23.53          399.93       -        -
  MTP spec=3          fp16      1499.0    1366.26  17.47    57.24     35.70          606.95       71.59    3.15
  MTP spec=4          fp16      1472.4    1390.96  17.35    57.64     35.60          605.28       63.36    3.53
  MTP spec=5          fp16      1392.4    1470.86  17.35    57.64     34.83          592.19       57.16    3.86
  MTP spec=3 Half-KV  fp8_e4m3  1483.6    1380.42  18.87    52.99     33.89          576.15       70.52    3.12
  MTP spec=4 Half-KV  fp8_e4m3  1470.3    1392.88  17.94    55.74     34.87          592.71       65.03    3.60
  MTP spec=5 Half-KV  fp8_e4m3  1456.6    1406.04  18.08    55.31     34.57          587.69       57.67    3.88
verdict -> C1 ctx=2048 winner depends on the metric. Spec=4 fp16 KV is best for pure `tg` (57.64 tok/s, tied with
  spec=5 but with lower TTFT), while spec=3 fp16 KV is best for aggregate output throughput (35.70 tok/s) and has
  lower TTFT. MTP adds a real prefill/TTFT tax: no-MTP pp 1608 tok/s -> spec=3/4/5 pp 1499/1472/1392, and TTFT
  1.273s -> 1.366/1.391/1.471s. Half-KV is NOT a 2K-context speed lever on this stack: every Half-KV row is slower
  than the matching fp16-KV row on `tg` and aggregate output; keep it for capacity/context headroom. The external
  Lorbus 45.2 tok/s row is MTP-on (not no-MTP), no graph capture, no off-baseline, short-run accept 86%; our v0230
  PIECEWISE C1 `tg` now exceeds it, so there is no hidden no-MTP lever implied by that row. Container stopped; card1
  lease freed.

== 2026-06-23 :: compressed-tensors W4A16 MTP probe -- loads, but 0% accept ==
motivation -> answer whether MTP can work on our own `Qwen3.6-27B-W4A16` compressed-tensors artifact, rather than
  the Lorbus AutoRound int4 checkpoint used for the successful MTP campaign.
config -> first inspected checkpoint indexes on the host: Lorbus AutoRound has 29 `mtp` key hits and 333 `visual`
  key hits; our compressed-tensors W4A16 has 0 `mtp` and 0 `visual` key hits. Then used GPU1 only via
  `gpu-run --card 1`; did NOT edit `rdy_to_serve/`. Served `rdy_to_serve/qwen36-27b-w4a16/serve.sh` with env-only
  overrides: `NAME=w4a16_mtp_probe PORT=18081 DEVICE=1 SERVED=qwen36-27b-w4a16-mtp GRAPH=1 MAXLEN=2048
  MAXSEQS=8 CAPSIZES=1,2,4,8 UTIL=0.95 MTPTOK=4 COMPILESZ=`. Bench was `vllm bench serve` random 1024 input /
  64 output, C=1, N=4, `--ignore-eos`.
result -> the server reached health and generated coherent text. Logs confirmed the spec path was active:
  `Resolved architecture: Qwen3_5ForCausalLM`, then `Resolved architecture: Qwen3_5MTP`,
  `SpeculativeConfig(method='mtp', num_spec_tokens=4)`, `Loading drafter model`, and
  `Detected MTP model`. But the bench showed the missing trained MTP tensors are fatal for acceptance:
  output 12.08 tok/s, total 205.38 tok/s, TTFT 835.69 ms, TPOT 70.82 ms (`tg` 14.12 tok/s),
  acceptance rate 0.00%, accept_len 1.00, drafts 252, draft_tokens 1008, accepted_tokens 0.
verdict -> our compressed-tensors W4A16 can load the MTP code path, but it does NOT have a usable MTP head. This is
  not a serve-flag problem and not an XPU spec-decode crash; the checkpoint was exported without trained `mtp.*`.
  Re-quantize/export a full Qwen3.6 artifact that preserves `mtp.*` in BF16 (like Lorbus AutoRound did) before
  expecting W4A16+MTP to work. Container stopped; card1 lease freed.

== 2026-06-23 :: W4A16 compressed-tensors BF16-MTP graft works (with unquantized drafter shim) ==
motivation -> try the requested graft path now, starting with `Qwen3.6-27B-W4A16`, and run a ctx2048 perf table.
config -> created host model dir `/mnt/vm_8tb/b70/models/Qwen3.6-27B-W4A16-mtp-graft` as `cp -al` hardlink copy of
  `Qwen3.6-27B-W4A16`, then added `model-mtp-graft.safetensors` containing the 15 BF16 `mtp.*` tensors from
  `Qwen_Qwen3.6-27B` (811 MiB). Original W4A16 dir untouched; no `rdy_to_serve/` edits. First raw graft attempt
  loaded both shards but skipped MTP linears: the compressed-tensors quant config made vLLM instantiate the MTP
  drafter as quantized/fused (`fc.weight`, `qkv_proj`, `gate_up_proj` missing). Then used a temporary combined
  sitecustomize: existing text-only W4A16 shim + monkeypatch that sets `vllm_config.quant_config=None` only during
  `Qwen3_5MultiTokenPredictor.__init__`, leaving the target model W4A16 while making the MTP drafter BF16.
  Validation was GPU1 via `gpu-run --card 1`, image `vllm-xpu-env:v0230`.
result -> quick eager 1024/64 C1 validation with the BF16-MTP shim: acceptance 68.75%, accept_len 3.75, accepted
  tokens 187/272. Captured PIECEWISE ctx2048 table used `MAXLEN=3072`, `MAXSEQS=4`, `UTIL=0.97`, caps 1,2,4.
  Host CSV `/mnt/vm_8tb/b70/results/w4a16_mtp_graft_ctx2048_20260622_155426.csv`; repo copy
  `results/w4a16_mtp_graft_ctx2048_20260623.csv`.
  config       C  pp tok/s  TTFT ms  TPOT ms  tg tok/s  agg out tok/s  total tok/s  accept%  accept_len
  no-MTP       1  1713.0    1195.56  46.01    21.73     18.18          309.10       -        -
  MTP spec=4   1  1465.9    1397.06  23.27    42.97     29.41          499.91       62.24    3.49
  no-MTP       4  2409.4    3400.06  59.99    16.67     46.29          786.97       -        -
  MTP spec=4   4  1108.9    7387.70  34.51    28.98     38.58          655.88       51.88    3.08
verdict -> grafting is viable and gives real acceptance. C1 is a strong W4A16 CT win (`tg` 21.73 -> 42.97,
  aggregate output 18.18 -> 29.41) despite TTFT/pp tax. C4 is the same MTP story as Lorbus but harsher: per-stream
  decode improves, but aggregate output and TTFT regress because the spec verify gets compute/KV constrained. Memory
  is tight: real BF16 MTP + PIECEWISE capture left only 6,283 KV tokens at MAXLEN=3072, so production needs either
  Half-KV, lower max concurrency, less capture memory, or a quantized/co-packed MTP head. Next step: turn the temporary
  BF16-MTP shim into a proper shelf-local patch before repeating on W8A8-sqgptq and W4A8-sqgptq-prepacked.

== 2026-06-23 :: W8A8/W4A8 BF16-MTP graft dirs created (not yet perf-tested) ==
motivation -> extend the validated W4A16 graft method to the other two Qwen3.6-27B compressed-tensors quants:
  `W8A8-sqgptq` and `W4A8-sqgptq-prepacked`.
config -> created host sibling dirs only; originals untouched. Used `cp -al` hardlink copies, then added the same
  15 BF16 `mtp.*` tensors from `Qwen_Qwen3.6-27B` as `model-mtp-graft.safetensors` (849,400,464 bytes / 811 MiB).
  Also wrote `mtp_bf16_patch/sitecustomize.py` in each dir. That patch forces only `Qwen3_5MultiTokenPredictor` to
  instantiate unquantized/BF16 so vLLM does not treat the grafted MTP drafter as compressed-tensors quantized/fused
  and skip the BF16 linears.
result -> created and verified by safetensors key count:
  `Qwen3.6-27B-W8A8-sqgptq-mtp-graft`: 2 safetensors files, 1122 total keys, 15 `mtp.*` keys.
  `Qwen3.6-27B-W4A8-sqgptq-prepacked-mtp-graft`: 2 safetensors files, 1122 total keys, 15 `mtp.*` keys.
verdict -> graft artifacts exist and are structurally ready for serve tests. No GPU perf/acceptance run yet. When
  serving, mount the model dir's `mtp_bf16_patch` on `PYTHONPATH` (plus any model-specific text/VLM shim already
  required by that quant) before enabling `--speculative-config '{"method":"mtp","num_speculative_tokens":4}'`.

== 2026-06-23 :: act-quant fusion LIVE A/B -- both eager-microbench projections fail under capture ==
motivation -> the kernel/23 eager microbench said two things would be cheap wins: (a) W4A8's slow pure-torch
  `dynamic_per_token_int8_quant_ref` (~210us, ~5 launches) should be replaceable by the native
  `_xpu_C.dynamic_per_token_int8_quant` kernel for a "free win"; (b) the RMSNorm+quant fusion would add ~1.14x
  (eager linear-path sim). Both were projected EAGER. This session measured both LIVE in the real GRAPH=1
  PIECEWISE captured serve. Two cards in parallel via the per-card lease: card 0 = W4A8 swap A/B, card 1 = W8A8
  fusion A/B. Each: reuse the already-booted baseline container, then serve the variant from an ISOLATED recipe
  copy (so the shared `_common/lib.sh` and the host `patches/xpu.py` are never mutated -> safe alongside the
  other card). 14B, ctx in=2048/out=128, C=1, GRAPH=1, dtype float16, 2 measured repeats each, coherence probe
  both sides.
config -> W4A8: `Qwen3-14B-W4A8-gptq-prepacked` on `vllm-xpu-env:int8g`, served `qwen3-14b-w4a8-gptq`, card 0:8101.
  baseline = committed `patches/xpu.py` (ref quant); swap = the one-spot change in `XPUW4A8IntLinearKernel.apply_weights`
  guarded by `hasattr(torch.ops._xpu_C,"dynamic_per_token_int8_quant")` (op confirmed built: registered-fake count=2,
  so the native path was taken). W8A8: `Qwen3-14B-W8A8-autoround` on `:int8g`, served `qwen3-14b-w8a8-autoround`,
  card 1:8112. baseline = shipped serve (`pass_config.fuse_norm_quant=false`); fused = same recipe with
  `sed s/"fuse_norm_quant":false/:true/` in the isolated lib.sh copy (pass_config confirmed `True` in the engine log).
command -> per-card driver scripts `/tmp/drive_w4a8.sh` (card 0) and `/tmp/drive_w8a8.sh` (card 1), each launched
  `setsid gpu-run --card N bash drive_*.sh`; both leases held independently; host-side waiter for completion.
result -> columns = concurrency,req_s,out_tok_s,mean_ttft_ms,mean_tpot_ms,per_stream_decode_tok_s
  W4A8 baseline (ref quant)        1,0.31,40.20,364.84,22.19,45.07  /  1,0.31,40.19,365.86,22.20,45.05
  W4A8 swap   (native quant op)    1,0.26,33.10,398.88,27.30,36.63  /  1,0.26,33.08,399.54,27.32,36.60
  W8A8 baseline (fuse_norm=false)  1,0.19,23.88,305.74,39.79,25.13  /  1,0.19,23.89,304.99,39.79,25.13
  W8A8 fused   (fuse_norm=true)    1,0.19,24.31,306.70,39.04,25.61  /  1,0.19,24.30,306.29,39.06,25.60
  All four coherent ("The capital of France is Paris ..."). Repo CSV: `results/actquant_fusion_ab_14b_ctx2048_20260623.csv`.
verdict -> BOTH eager projections FAIL in the captured serve.
  (1) W4A8 native-quant swap is a 19% REGRESSION (45.07 -> 36.63 decode t/s; TPOT 22.19 -> 27.30 ms), reproducible
      across both repeats and both ran captured (swap booted ~4.5 min: load+dynamo+PIECEWISE capture, GRAPH=1).
      Mechanism (consistent with kernel/23): under GRAPH=1 PIECEWISE + inductor graph-partition, the pure-torch
      `_ref` DECOMPOSES into elementwise+reduction ops that inductor FUSES into the surrounding captured graph
      (its eager weakness -- ~5 launches -- vanishes once captured), whereas the native custom op is an OPAQUE
      captured node whose serial per-row K-reduction "persists under capture" (~101us for the K=17408 down_proj).
      So the eager microbench (where launches dominate and `_ref` looks 0.17-0.50x) inverts under capture. KEEP
      THE REF; the working-tree swap patch was reverted (shelf unchanged, still committed `_ref`).
  (2) W8A8 `fuse_norm_quant=true` does NOT crash on `:int8g` (the lib.sh "NameError under torch.compile on XPU"
      fear did not materialize for this image/model): served healthy + coherent, +1.9% decode (25.13 -> 25.61 t/s,
      TPOT 39.79 -> 39.04 ms). BUT no INFO-level "fused N patterns" evidence was emitted, so the +1.9% is "flag on,
      clean, tiny gain" -- not an attributed fusion win, and it sits far below the eager 1.14x sim and far below
      the real decode levers (MTP ~1.79x PIECEWISE, graph capture). Cross-check: W8A8 14B decode ~25 t/s vs W4A8
      ~45 t/s at M=1 matches the roofline (int8 reads 2x the weight bytes/token of int4). Net: neither activation-
      quant lever is worth shipping on the captured 14B path; the int8-decode headroom is in FUSING quant INTO the
      GEMM prologue (one opaque node that also does the GEMM), not swapping the standalone quant op or toggling the
      inductor norm-quant pass. Host clean: both containers removed, both leases freed.

== 2026-06-23 :: W4A8 27B compressed-tensors BF16-MTP graft -- single-card sweep, ~2.0x FEASIBLE ==
motivation -> run the deferred serve/acceptance bench on the W4A8 BF16-MTP graft dir (created 06-23 but never
  perf-tested), i.e. MTP_TODO Phase B2 + the user "MTP sweep, what's feasible" ask. Single-card is the path:
  TP=2 MTP is dead (M4), so the W4A8 graft (fits one card) is the only int8-activation 27B that can do MTP today.
config -> new self-contained host script `scripts/90_mtp_graft_sweep.sh` (append-only; does NOT touch bin/ or
  _common). Reuses the proven 30_serve engine wiring (PREPACK loader xpu.py+compressed_tensors_w4a8_int.py,
  GDN-enabled _xpu_C.abi3.so + libgdn_attn_kernels_xe_2.so, NOMM, VLLM_W4A8_PREPACKED=1) and adds the ONE missing
  piece the engine lacks: the graft dir's `mtp_bf16_patch` on PYTHONPATH (forces only Qwen3_5MultiTokenPredictor
  to instantiate unquantized/BF16). IMG=vllm-xpu-env:int8g, model
  /models/Qwen3.6-27B-W4A8-sqgptq-prepacked-mtp-graft, served qwen36-27b-w4a8-sqgptq-mtp, GRAPH=1 PIECEWISE,
  dtype auto(bf16), UTIL=0.97 MAXLEN=2048 MAXSEQS=4 CAPS=1,2,4. Bench = the M2/M4 TTFT-cancelled decode (gen 64
  vs 512, subtract) + /metrics spec_decode counters (accept_len = accepted/drafts + 1). spec in {off,3,4,5}.
  Ran on card 0 via `gpu-run --card 0`.
BUG CAUGHT + FIXED -> the committed `mtp_bf16_patch/sitecustomize.py` in BOTH the W4A8 and W8A8 graft dirs was
  syntactically BROKEN (string quotes stripped: `prefix=)`, bare `quant_config`, unquoted print) -> Python logged
  "Error in sitecustomize" and never applied the patch. Harmless for the off baseline (no MTP) but it would have
  given every CT-graft MTP run a QUANTIZED drafter -> 0% accept (the exact failure the shim exists to prevent,
  same signature as the 06-23 W4A16 compressed-tensors 0%-accept probe). Rewrote a correct ASCII sitecustomize.py
  and pushed it byte-exact (scp) to both host graft dirs; backed up the broken originals as *.broken.bak; cleared
  stale __pycache__. Fixed in time (caught during the off-baseline load, before any spec run started).
result -> host CSV /mnt/vm_8tb/b70/results/mtp90_w4a8_20260622_183543.csv; repo copy
  results/mtp90_w4a8_27b_single_20260623.csv. Logs confirmed the path: speculative_config=SpeculativeConfig(
  method='mtp', num_spec_tokens=N), "Detected MTP model. Sharing target model embedding/lm_head weights with the
  draft model", and NONZERO acceptance (the corrected shim is live).
  spec  decode_tps  MTPx   accept_len  accept_rate  (accepted/drafts/draft_tok, gen512 s)
  off   20.74       -      -           -            (gen512 24.74s)
  3     41.99       2.02   2.98        0.660        (388/196/588, 12.11s)
  4     40.44       1.95   3.32        0.580        (408/176/704, 12.69s)
  5     42.03       2.03   3.79        0.559        (433/155/775, 12.10s)
verdict -> W4A8 27B single-card MTP is FEASIBLE and ~2.0x. WINNER spec=5 (42.03 t/s, 2.03x) -- spec=3 ties it
  (41.99) at lower accept_len; spec=4 dips to 40.44. The decisive Phase-B2 finding: accept_len HOLDS on int4
  weights -- 3.79 at spec=5 (vs the BF16 4.04 reference and the W4A16 graft's 3.49 at spec=4). The plan's
  "int4 drift may drop accept to 3.0-3.5" worry is REFUTED for the 27B. So both single-card int4-weight schemes
  land the same MTP'd decode: W4A8 ~42 t/s @2.0x vs W4A16 graft ~43 t/s @~2.0x -- W4A8 buys the int8-activation
  (systolic) path at no acceptance cost. Caveats: (a) accept measured over a 512-token gen only -- watch for the
  Lorbus 86->65% long-decode decay; (b) bf16 model dtype triggers a "int4_gemm_w4a8 produces float16 output,
  recommend --dtype float16" perf warning (both off and MTP share it, so the multiplier is fair; a float16 re-run
  could lift absolute t/s); (c) VRAM tight (UTIL 0.97, MAXLEN capped 2048) -- production wants Half-KV or a
  co-packed/quantized MTP head. Container stopped; card0 lease freed (gpu-run done in 1251s).

== 2026-06-23 :: W8A8 27B BF16-MTP graft TP=2 retry -- MTP head FINE, but TP=2 MTP not viable (exact oneCCL cause) ==
motivation -> user asked to "try tp=2 again for w8a8". The W8A8-sqgptq graft is 34GB -> does NOT fit one 32GB card,
  so TP=2 is the ONLY option; M4 already found TP=2 MTP DEAD (spec-allgather not graph-capturable) but on the W4A16
  int4 model, not a real grafted W8A8 head. This re-tests with the actual graft + captures the precise root cause.
config -> scripts/90 MODE=w8a8tp2 under the full `gpu-run` lease (both cards). IMG=vllm-xpu-env:int8g, model
  /models/Qwen3.6-27B-W8A8-sqgptq-mtp-graft (CT W8A8-int8 -> XPUInt8ScaledMMLinearKernel auto-selected), GDN .so
  mounted, NOMM, the corrected mtp_bf16 shim on PYTHONPATH, TP=2 + CCL_ENABLE_SYCL_KERNELS=1 + the #41663 Battlemage
  CCL env, GRAPH=1 PIECEWISE, UTIL=0.90 MAXLEN=4096 MAXSEQS=8 CAPS=1,2,4,8. Three configs: off (PIECEWISE baseline),
  spec=4 (PIECEWISE), spec=4e (EAGER fallback = same MTP, --enforce-eager, no capture). Same TTFT-cancelled bench.
result -> host CSV /mnt/vm_8tb/b70/results/mtp90_w8a8tp2_20260622_185750.csv; repo results/mtp90_w8a8tp2_27b_20260623.csv.
  spec  decode_tps  MTPx   accept_len  accept_rate   outcome
  off   18.74       -      -           -             PIECEWISE TP=2 healthy (matches QUANTS_TODO Q2 ~17.5 c1)
  4     CRASH       -      -           -             PIECEWISE: serve-fail at engine init
  4e    14.27       0.76   3.63        0.658         EAGER: runs, accept great, but 0.76x = NET LOSS vs off
  - spec=4 PIECEWISE crash root cause (now EXACT, sharper than M4's "spec-allgather not graph-capturable"):
    `oneCCL: coll.cpp:1204 ccl_allgather_impl: EXCEPTION: |CCL_SYCL| sched algorithms do not support sycl_graph
    recording, please use sycl_algorithms`. So allreduce records under CCL_ENABLE_SYCL_KERNELS=1 (off baseline +
    the Q2 W8A8 TP=2 serve both work), but the MTP spec-verify introduces an ALLGATHER that oneCCL routes to its
    scheduler ("sched") algorithm, which has NO sycl_graph-recordable implementation -> capture aborts.
  - The MTP HEAD itself is correct at TP=2: the eager run got accept_len 3.63 / 65.8% accept (better than W4A8's
    3.32 @spec4 -- consistent with W8A8 being near-lossless so the BF16 head drafts it better), and the shim
    printed `[mtp-bf16-shim] Qwen3_5MultiTokenPredictor forced unquantized for grafted BF16 mtp.*` on all 4 worker
    procs + "Detected MTP model. Sharing target embedding/lm_head". So this is NOT a head/graft problem.
verdict -> W8A8 27B TP=2 MTP is NOT VIABLE on stock int8g/v0230, CONFIRMING + sharpening M4. Two dead ends: MTP-on
  PIECEWISE crashes (oneCCL allgather can't record in a SYCL graph), and MTP-on eager runs but is 0.76x (uncaptured
  TP collectives + 5x verify body cost outweigh the accept-3.63 draft savings). The acceptance is great -- the
  blocker is purely the TP collective capture, exactly the gap RagingNoper's custom capture-safe collectives
  (local-write/remote-read xpu_communicator.py) close. The oneCCL error names the cure ("use sycl_algorithms") ->
  next probe is whether a oneCCL env knob can force the allgather onto a graph-recordable sycl/topo algorithm
  (codex research underway). Even if it could, the eager 0.76x + M4's "TP=2 is 0.87x single-card even MTP-off" mean
  the single-card W4A8 MTP (2.03x, 42 t/s) decisively beats any TP=2 W8A8 path for single interactive streams;
  TP=2 W8A8 stays a CAPACITY/VRAM play. Both cards' leases freed (gpu-run done in 1047s); host clean.

== 2026-06-23 :: BREAKTHROUGH -- splitting_ops REVIVES TP=2 MTP (overturns M4 "TP=2 MTP DEAD") ==
motivation -> user: "while codex works, do deep research + lots of small proof/spoof experiments to fix PIECEWISE
  or other levers". The W8A8 TP=2 MTP crash (90) was `oneCCL ccl_allgather_impl: CCL_SYCL sched algorithms do not
  support sycl_graph recording`. Source dig (int8g vLLM): the spec verify adds a model-forward `vllm::all_gather`
  (a REAL registered custom op, parallel_state.py:160 + fake:170) that gets recorded into the SYCL graph; oneCCL
  (2021.17 in the image) routes allgather to its scheduler algorithm which has no graph-recordable impl. allreduce
  DOES record (the MTP-off TP=2 baseline captures fine). Two orthogonal fix families tested + a spoof staged.
config -> scripts/91_tp2_mtp_capture_fix.sh, W8A8 graft TP=2 MTP spec=4 PIECEWISE, 4 named variants under one
  `gpu-run` (both cards). A=splitting_ops adds the 3 collectives (`vllm::all_gather/all_reduce/reduce_scatter`) so
  inductor graph-partition runs them EAGER as partition boundaries (never recorded), decode GEMMs/attn stay CAPTURED
  (the RagingNoper recipe insight). B=codex oneCCL env `CCL_SYCL_ALLGATHERV_SCALEOUT=ring`+thresh+TMP_BUF+COMM_SIZE.
  C=codex env `CCL_ALLGATHER(V)=topo`+monolithic. D=A+B. WIN = reaches /health + bench vs the 90 off=18.74.
result -> repo results/mtp91_tp2fix_20260622_192450.txt.
  variant  fix                                   outcome        decode_tps  vs-off-x  accept_len
  A        splitting_ops eject collectives       CRASH CLEARED  55.32       2.95      5.00
  B        CCL allgatherv ring env               STILL CRASHES  -           -         -
  C        CCL allgather/allgatherv topo env     STILL CRASHES  -           -         -
  D        A + B                                 CRASH CLEARED  56.02       2.99      5.00
  - The oneCCL env knobs (B,C) do NOTHING on this 2021.17 build -- same exact `sched algorithms do not support
    sycl_graph recording` crash. The fix is entirely splitting_ops; D == A within noise (CCL env adds nothing).
  - accept_len 5.00 = 100% on this prompt (acc 472 = drafts 118 x spec 4 exactly): near-lossless W8A8 body -> the
    BF16 MTP head drafts it almost perfectly. Prompt is repetitive code (easy) -> real-workload accept will be lower;
    the SPEEDUP MECHANISM is what's proven here.
verdict -> **M4 "TP=2 MTP DEAD" is OVERTURNED.** Ejecting the TP collectives from the captured graph via
  splitting_ops clears the oneCCL-allgather-can't-record crash, and TP=2 W8A8 27B MTP spec=4 runs at **55-56 t/s =
  ~2.95-2.99x vs the 18.74 MTP-off TP=2 baseline** -- FASTER than single-card W4A8 (42) or W4A16-graft (~43) MTP,
  because (a) near-lossless W8A8 -> high accept and (b) MTP amortizes the TP collective tax (which is exactly why
  MTP-off TP=2 was only 0.87x single-card -- the collective overhead that hurt off-baseline is what MTP hides).
  So the heavy near-lossless W8A8 (needs 2 cards) is now the FASTEST single-stream 27B MTP config we have, not a
  dead end. The earlier M4 result was an artifact of NOT having the collectives in splitting_ops (default
  splitting_ops has only attn/GDN ops). No custom capture-safe communicator (RagingNoper's xpu_communicator.py)
  is needed for the SPEC allgather -- splitting_ops alone suffices because allreduce already records under
  CCL_ENABLE_SYCL_KERNELS=1. Codex's oneCCL-env lead was a dead end but cheap to rule out. (Spoof shim
  all_gather->allreduce-of-padded staged as a fallback, NOT NEEDED.) Next: spec sweep + harder prompt for honest
  accept. Both leases freed; host clean.

== 2026-06-23 :: W8A8 TP=2 MTP spec sweep (splitting_ops fix) -- spec=5 = 63.11 t/s, ~3.4x production ==
motivation -> nail down the overturned TP=2 verdict: full off+spec{3,4,5} curve with the splitting_ops fix, and a
  HARDER natural-language prompt (Roman-Empire reasoning, not the trivially-predictable LRU code) for honest accept.
config -> scripts/93, W8A8 graft TP=2 PIECEWISE, splitting_ops includes the 3 collectives (the 91-A fix), CCL
  sycl-kernels on, harder prompt, gpu-run both cards.
result -> repo results/mtp93_w8a8tp2_20260622_193943.csv.
  spec  decode_tps  MTPx(same-cfg)  accept_len  accept_rate  gen512_s
  off   14.46       -               -           -            34.14
  3     50.37       3.48            3.96         0.986        10.47
  4     57.24       3.96            4.93         0.983         9.39
  5     63.11       4.37            5.90         0.980         8.41
verdict -> WINNER spec=5 = 63.11 t/s, monotonically climbing (50->57->63) -- accept ~98% greedy hasn't saturated,
  so spec=6+ may go higher (untested). TWO baselines matter:
  - This config's off = 14.46 (splitting_ops ejects ALL collectives to EAGER, slowing even no-MTP) -> 4.37x.
  - Best MTP-off TP=2 = 18.74 (default splitting_ops, collectives CAPTURED, scripts/91) -> **honest production
    multiplier 63.11/18.74 = 3.37x**. Lead with 3.37x; the 4.37x is vs the same eager-collective config.
  ACCEPT CAVEAT: ~98% is a temp=0 greedy single-prompt best case (near-lossless W8A8 body -> the BF16 MTP head
  predicts almost perfectly, even on reasoning text); real serving (temp>0, varied/short prompts) will be lower --
  report greedy headline, expect production accept to fall (Playbook A item 3). Net: **W8A8 27B TP=2 MTP = ~63 t/s
  is the FASTEST single-stream 27B config on the rig** (single-card W4A8 42, W4A16-graft 43), now that splitting_ops
  unblocks TP=2 capture. The 2-card W8A8 quality scheme went from "MTP-dead capacity play" to "MTP speed king".
  Leases freed; host clean.

== 2026-06-23 :: W4A8 single-card MTP levers -- capture-the-verify-batch = +14% (45.70 t/s); fp16 neutral ==
motivation -> push the proven single-card W4A8 MTP (90 winner spec5 = 42.03, caps 1,2,4) higher. Two hypotheses:
  (a) --dtype float16 (the int4_gemm_w4a8 fp16-output warning), (b) capture the spec-verify batch 1+spec=6 (the
  winner's caps 1,2,4 miss it -> verify falls back to eager).
config -> scripts/92, single card (card 0), MTP spec=5 PIECEWISE, per-dtype off baseline. variants base(bf16,
  caps1,2,4) / fp16(float16,1,2,4) / capspec(bf16,1,2,4,6,8) / combo(float16,1,2,4,6,8).
result -> repo results/mtp92_w4a8levers_*.csv.
  variant  dtype    caps        decode_tps  MTPx  accept_len
  off      auto     1,2,4       21.13       -     -
  base     auto     1,2,4       40.13       1.90  3.60
  fp16     float16  1,2,4       41.29       2.02  3.64
  capspec  auto     1,2,4,6,8   45.70       2.16  3.79   <- WINNER
  combo    float16  1,2,4,6,8   41.83       2.05  3.55
verdict -> CAPTURING THE SPEC-VERIFY BATCH (caps include 1+spec) is a real +14% (base 40.13 -> capspec 45.70 t/s):
  the winner's caps 1,2,4 left the verify decode at batch 6 running EAGER; adding 6,8 captures it. New single-card
  W4A8 MTP best = 45.70 t/s (2.16x). --dtype float16 is NEUTRAL (41.29 vs 40.13, noise) -- the fp16-output warning
  did not translate to a real win; combo(fp16+capspec)=41.83 is actually below capspec(bf16)=45.70, so fp16 slightly
  HURTS the captured path. KEEP bf16/auto. CROSS-CUTTING: the W8A8 TP=2 headline (93, spec=5, caps 1,2,4,8 -- MISSES
  6) likely left the same ~14% on the table -> worth re-running W8A8 TP=2 spec5/6 with caps including the verify
  batch. accept_len varies 3.55-3.79 across identical temp=0 runs -> the bench's accept metric has ~+-0.2 noise
  (warmup+64+512 accumulation); treat accept_len as approximate, decode_tps as solid. Card0 lease freed.

== 2026-06-23 :: W8A8 TP=2 capspec stacking + spec=6 ceiling -- spec=5 ~64 t/s is the peak ==
motivation -> two questions on the revived TP=2 MTP: (1) does the 92 "capture the verify batch" lever stack on the
  91 splitting_ops fix? (2) is spec=5 still climbing -> try spec=6?
config -> scripts/94, W8A8 graft TP=2 PIECEWISE + splitting_ops fix, caps INCLUDE the 1+spec verify batch
  (spec5->caps 1,2,4,6,8; spec6->1,2,4,7,8), harder prompt. gpu-run both cards.
result -> repo results/mtp94_w8a8tp2caps_*.csv.
  spec  caps         decode_tps  x-vs-bestoff(18.74)  accept_len  accept_rate
  5     1,2,4,6,8    64.05       3.42                 5.90        0.980
  6     1,2,4,7,8    28.67       1.53                 3.00        0.333
verdict -> (1) capspec does NOT stack on TP=2: spec5 63.11 (93, no batch-6 cap) -> 64.05 (batch-6 capped) = +1.5%,
  noise. Unlike single-card W4A8 (+14%), the W8A8 TP=2 verify is COLLECTIVE-bound (the eager allgather dominates),
  not GEMM-batch-bound, so capturing the verify GEMM batch is marginal. (2) spec=6 COLLAPSES: accept_rate 0.98 ->
  0.333, accept_len 5.90 -> 3.00, decode 64 -> 28.67. The MTP module has mtp_num_hidden_layers=1 -> its useful draft
  horizon is ~5 tokens; spec=6 drafts past it -> most drafts rejected -> net slowdown. **spec=5 is the ceiling.**
  FINAL W8A8 TP=2 MTP headline: ~63-64 t/s, 3.4x vs best MTP-off (18.74) -- MEETS the MTP_TODO primary success
  criterion (>=3x). Both leases freed; host clean. Campaign of scripts/90-94 complete.

== 2026-06-23 :: W8A8 TP=2 prefill/TTFT @ 2048 + a long-context MTP HANG (recipe blocker) ==
motivation -> user wants prefill+TTFT at 2048 ctx (the int8-prefill optimization baseline), and a rdy_to_serve
  recipe for the TP=2 MTP win.
result (prefill, MTP-OFF, scripts/96 section A -- VALID, vllm bench serve random 2048/128) ->
  conc  req/s  out_tok/s  TTFT_ms   TPOT_ms  per-stream_decode
  1     0.10   12.85      2747.84   56.79    17.61      -> prefill ~= 2048/2.748s ~= 745 tok/s
  4     0.21   27.18      7085.86   92.44    10.82
  So int8 W8A8 TP=2 prefill @2048 ~745 tok/s / TTFT 2.75s -- ~10x below the int8 XMX compute ceiling -> big
  prefill headroom (the user's int8-prefill optimization target). Decode 17.6 matches the ~18.74 off baseline.
LONG-CONTEXT MTP HANG (the real find) -> scripts/95 (MTP-on, splitting_ops, spec5) @ 2048 ctx PREFILLED
  (204.8 prompt tok/s) then DECODE STALLED at ~0 t/s for 30 min: 1 req "Running", engine threads asleep
  (epoll/nanosleep, NOT D-state) -> a TP-collective / spec-decode long-context deadlock. Short prompts (80 tok,
  scripts 93/94) never hit it. Container was hard to remove (needed explicit docker kill -SIGKILL). scripts/96
  section B FALSELY showed instant 0-tok "hangs" -- that was a SCRIPT BUG (host has NO python3 -> empty request
  bodies); ignore it. scripts/97 (pure-bash JSON) is characterizing the real break point by ctx length + testing
  --no-enable-chunked-prefill as the candidate fix (chunked-prefill x spec-decode x TP is the prime suspect).
verdict (so far) -> the TP=2 MTP recipe is a DECODE win but has a LONG-CONTEXT serving HAZARD; do NOT shelf it
  as production-ready until 97 pins the break point + a fix. The MTP-off prefill number stands. Host cleaned,
  leases freed after each run.

== 2026-06-23 :: hang isolated (--random only) + W8A8 TP=2 MTP recipe SHELVED (smoke GREEN) ==
hang isolation (scripts/98) -> the scripts/95 MTP-on stall is ONLY the `vllm bench serve --random` gibberish-token
  path. Real prompts are FINE: 2048-ctx/128-out OK (8.85s), 2048-ctx/256-out OK (7.84s), 1024-ctx back-to-back x3
  OK; only --random 2048/128 HUNG (>200s, same 0-gen signature). So it is a benchmark artifact (OOD random token
  IDs x MTP rejection sampler), NOT a production hazard. scripts/97 separately proved ctx length alone is not the
  trigger (64..2048 all OK) and --no-enable-chunked-prefill was not needed.
recipe -> shelved rdy_to_serve/qwen36-27b-w8a8-sqgptq-mtp (serve.sh + README + corrected MTP-BF16 shim). Added an
  additive SPLITOPS knob to _common/lib.sh (empty-default = byte-identical CC; offline dry-run reproduces the
  scripts/93 config exactly). `bin/serve-sweep --smoke` = ALL GREEN across all 8 shelf models (new recipe HEALTHY
  TP=2 eager; no regression). Commit e2c8c76. README documents the --random benchmark hazard.
verdict -> the TP=2 MTP win is now a self-contained, smoke-gated shelf recipe. Next: 262K KV-capacity + long-prompt
  prefill (scripts/99), and an honest real-dataset bench before any localmaxxing submission.

== 2026-06-23 :: W8A8 TP=2 262K KV capacity + prefill-vs-length (the int8-prefill baseline) ==
config -> scripts/99: W8A8 graft TP=2, MTP-OFF, --enforce-eager, MAXLEN=262144, UTIL=0.95, fp16 KV, both cards.
result -> repo results/mtp99_262k_*.txt.
  KV FIT: "Available KV cache memory: 14.81 GiB" (per card), "GPU KV cache size: 479,090 tokens",
          "Maximum concurrency for 262,144 tokens per request: 1.83x" -> a SINGLE 262K session FITS at fp16 with
          83% headroom. Confirms the hybrid math (16 full-attn + 48 GDN layers, kv_heads=4, head_dim=256 ->
          64 KB/token fp16 -> 262K ~= 16 GiB total KV; pool 479K tok >> 262K).
  PREFILL scaling (vllm bench serve --random, MTP-off so no hang, OUT=8, C=1):
    ctx      TTFT_ms     prefill_tok/s
    2048     5394.93     380
    8192     20689.56    396
    32768    83566.13    392
    131072   352346.91   372
verdict -> prefill throughput is FLAT ~380-396 tok/s from 2K to 131K -> NOT attention-bound (the hybrid keeps the
  16 full-attn layers' O(n^2) cheap); the limiter is a CONSTANT per-token cost = eager kernel launches + per-layer
  TP all-reduce over PCIe + un-fused int8 GEMM. So (a) long context fits AND scales linearly (131K TTFT ~6 min --
  usable but slow), (b) the int8-prefill optimization target is the per-token rate, not the algorithm. Cross-check:
  captured prefill (scripts/96, MAXLEN=4096) = 745 tok/s @2048 vs eager 380 here -> capture alone ~2x on prefill.
  Next (scripts/100): MTP-ON KV capacity @ 262K (does the recipe's drafter+capture still leave room for 262K?).

== 2026-06-23 :: MTP-ON 262K KV capacity -- full model max FITS at fp16 (1.42x) ==
config -> scripts/100: the RECIPE config (W8A8 TP=2, MTP spec=5, splitting_ops, GRAPH=1 PIECEWISE), MAXLEN=262144,
  UTIL=0.95, fp16 KV.
result -> HEALTHY. "Available KV cache memory: 12.84 GiB" (per card), "GPU KV cache size: 372,809 tokens",
  "Maximum concurrency for 262,144 tokens per request: 1.42x". (Shim + Detected-MTP confirmed on both workers.)
verdict -> With MTP ON, maxlen = the FULL 262,144 model max, fp16 KV, 42% headroom -- NOT VRAM-limited at TP=2.
  The drafter + PIECEWISE capture cost ~2 GiB/card vs MTP-off (KV pool 479,090 -> 372,809 tokens; 1.83x -> 1.42x),
  but 373K >> 262K. Half-KV (fp8) would ~2x the pool (~745K tok) for multi-session long-context. The constraint
  on long context is SPEED (flat ~390 t/s eager prefill -> 262K TTFT ~12 min), not memory. Recipe defaults
  MAXLEN=4096 for interactive; raise to 262144 per-session for long docs. Host clean; leases freed.

== 2026-06-23 :: [!!! MAJOR CORRECTION] W8A8 27B TP=2 MTP "63 t/s / 3.4x headline" was GARBAGE -- root-caused + fixed (eager) ==
motivation -> user: "our headline w8a8 MTP is broken when we serve." Served the shelved recipe
  (rdy_to_serve/qwen36-27b-w8a8-sqgptq-mtp) AS THE HEADLINE config (TP=2, MTP spec=5, PIECEWISE capture,
  splitting_ops) and READ THE TEXT (the shelf "smoke GREEN" only served TP=2 EAGER + never coherence-checked).
repro -> serves HEALTHY, MTP loads ("Detected MTP model"), shim applies, capture finishes -> generation is
  PURE GARBAGE ("!!!!", "is is is"). The "63 t/s, accept 98%, accept_len 5.90" was a FALSE POSITIVE: a degenerate
  body makes the BF16 draft head (drafting the same garbage) and the target AGREE on "!" -> trivial ~98% accept,
  accept_len saturates at spec+1. The bench used ignore_eos + token-count and NEVER read the text -- the exact
  QUANTS_TODO Q8 trap, repeated.
bisection (config->command->result->verdict, scripts/101-106) ->
  14B W8A8 single-card eager   COHERENT  -> int8 W8A8 kernel + image OK
  14B W8A8 TP=2 eager          COHERENT  -> TP=2 int8 path OK
  27B W4A8 single-card eager   COHERENT  -> GDN mount + VLM text-only config + 27B family OK
  27B W8A8 TP=2 eager (body)   GARBAGE   -> isolated to the 27B W8A8 CHECKPOINT (not kernel/GDN/image/TP)
  weight audit (scripts/101): dequant int8*scale vs bf16 base = per-channel cosine 0.97-0.9999, 0% int8
    saturation, no NaN scales -> WEIGHTS ARE GOOD. Not corrupt, not a requant problem.
ROOT CAUSE (scripts/102,103) -> the checkpoint config.json `ignore` list had 336 ENUMERATED leaf names with the
  WRONG FLAT PREFIX `model.layers.N.linear_attn.*`. The actual keys are VLM-NESTED `model.language_model.layers.N.*`
  -> the ignore entries matched NOTHING (verified: ignore-match for 'model.language_model.layers.0.linear_attn...'
  = []). So the 48 GDN linear_attn layers (stored BF16, no scale) were NOT exempted -> vLLM built them as W8A8 int8
  -> a BF16 [out,in] weight SILENTLY shape-matches the int8 param buffer, weight_scale missing -> 48 recurrent GDN
  layers of garbage -> degenerate output. (codex insight: W4A8 would ASSERT on the int4-packed shape mismatch --
  which is exactly why the W4A8 hit this same bug earlier, was caught, and fixed to a 4-regex ignore [its
  config.json.ignore339.bak]; W8A8 silently misloads because int8 == bf16 shape.) llmcompressor quantized
  CORRECTLY (scripts/49 passes re:.*linear_attn.*; GDN stored BF16) -- the bug is ONLY the SAVED config.json ignore
  serialization (wrong prefix), surviving the VLM config graft. -> CONFIG-ONLY fix, NO requant.
FIX (scripts/104) -> replace ignore with the regex form ["lm_head","re:.*linear_attn.*","re:.*visual.*","re:.*mtp.*"]
  (backs up to config.json.ignore339.bak; the W8A8 base + graft configs are hardlinked so one edit fixes both).
  Verified: W8A8 pure body (TP=2 eager) now COHERENT ("Paris... Seine... art, fashion", correct Fibonacci, correct
  hash-table). Applied the same fix to the 27B W4A16 + W4A16-graft (config.json.ignore337.bak) -- same flat-prefix
  bug there, but HARMLESS on int4 (the packed-int4 loader falls back to unquantized on a BF16 weight, doesn't
  silently corrupt -> W4A16 already served coherently; fixed for cleanliness).
CAPTURED PATH STILL BROKEN (Bug B) -> with the GDN now correctly BF16, PIECEWISE capture is numerically broken on
  this TP=2 hybrid: use_inductor_graph_partition=true -> KeyError weight_scale at capture (partitioner packs a
  weight_scale placeholder for a region MIXING W8A8(has scale)+BF16 GDN(no scale)); IGP=false (legacy piecewise,
  new lib.sh knob) -> capture succeeds but decode is GARBAGE even WITHOUT MTP (clean-cache confirmed). 14B W8A8
  captures coherently single-card -> it's a TP=2 + BF16-GDN-in-captured-pieces + custom-int8 capture-numerics bug.
honest numbers (eager, the only coherent path; temp=0 greedy 256-tok) ->
  eager MTP-off (pure body) ~4.1 t/s ; eager MTP spec=5 ~9.0-9.6 t/s, accept ~48% (accept_len ~3.9) = 2.3x, COHERENT.
  So the coherent W8A8 27B TP=2 serve is CORRECT-BUT-SLOW (~9 t/s). The "fast" captured number was always garbage.
verdict -> Bug A (the user's "broken when we serve") is FIXED: ignore-list regex, no requant, eager coherent 2.3x
  MTP. Recipe now DEFAULTS GRAPH=0 (eager). README rewritten honestly. Added a COHERENCE GATE to b70_gen_probe
  (lib.sh): a degenerate repeated-token reply now FAILS smoke instead of passing on /health alone -- this class of
  bug would have been caught. Bug B (captured numerics) is characterized + documented as open. The repo's
  "fastest 27B = W8A8 TP=2 MTP 63 t/s" claim is RETRACTED (it was garbage on two counts: degenerate output, and the
  coherent version is ~9 t/s). For a fast COHERENT 27B int8-act serve today, single-card W4A8 (captures coherently)
  is the pick. scripts/101-106 + 104 fixer committed; host configs fixed in place with backups; leases freed.

== 2026-06-23 :: W4A8 single-card MTP graft VERIFIED COHERENT (2.03x is real) -> Bug B is TP=2-specific ==
check -> the user also asked "check the w4a8 if it has problems too." Served the W4A8-sqgptq MTP graft single-card,
  PIECEWISE captured, MTP spec=5 + BF16-MTP shim (scripts/107, mirrors scripts/90 w4a8) and READ THE TEXT.
result -> COHERENT ("Paris... thinking process", correct Fibonacci code), capture finished (2.88 GiB), accept
  63/80 = 79% on a single prompt (accept_len ~4.9, REAL not saturated). So the "W4A8 single-card MTP 2.03x"
  headline (scripts/90) is GENUINE, unlike the W8A8 TP=2 one. The W4A8 body was already proven coherent + its
  ignore list is already the regex form, so no garbage there.
refined Bug B -> captured corruption is TP=2-SPECIFIC, not int8/GDN/capture per se:
  14B W8A8 captured single-card = COHERENT ; W4A8 27B captured single-card = COHERENT ; W8A8 27B captured TP=2 =
  GARBAGE (even no-MTP, clean-cache). So the W8A8 capture break is a TP=2 + piecewise-split-at-collectives +
  custom-int8/BF16-GDN numerics bug. Practical consequence: the WORKING fast coherent int8-activation 27B MTP path
  TODAY is single-card W4A8 (~42 t/s captured, accept holds at int4) -- NOT the 2-card W8A8 (eager-only ~9 t/s, or
  captured garbage). W8A8's only edge is near-lossless accuracy; if its speed matters, the captured-TP=2 numerics
  must be fixed first (separate focused session -- leads: per-token int8 dynamic-quant buffer aliasing across the
  collective split boundaries; or BF16 GDN linears inside captured pieces under TP). Recipe README already points
  users to W4A8 for a fast coherent serve.
verdict -> task done: W4A8 is HEALTHY (no garbage bug). W8A8 fix shipped (eager). Bug B scoped to captured-TP=2.

== 2026-06-23 :: Bug B capture-recovery RULED OUT (both partition modes garbage) -- it's the TP=2 captured numerics ==
attempt -> tried to recover the fast captured W8A8 TP=2 path (the user's "build whatever's needed"). codex #3:
  the IGP=true crash is `KeyError weight_scale` (partitioner packs a weight_scale placeholder for a region MIXING
  W8A8(has scale)+BF16 GDN(none)). Wrote scripts/108 dummy-wscale shim: patch UnquantizedLinearMethod.
  process_weights_after_loading to register a dummy weight_scale(ones) on every unquantized linear -- the BF16
  forward ignores it, so numerics unchanged; it only satisfies the partition input-collection.
result -> the shim CLEARED the KeyError (capture finished, 2.93 GiB) -- but IGP=true captured no-MTP TP=2 output is
  STILL pure "!!!!" garbage, identical to IGP=false. So **both inductor-graph-partition modes produce garbage** ->
  the partition MECHANISM is not the corruptor; the actual CAPTURED COMPUTATION at TP=2 is numerically wrong.
verdict -> Bug B is conclusively a **captured-TP=2 numerics** bug, NOT a crash/partition-packing bug. The
  dummy-scale shim is a valid crash-unblock but does not fix numerics, so it is NOT shipped (would just let the
  broken captured path serve garbage without the KeyError). Capture recovery for W8A8 27B TP=2 needs real
  kernel/capture-level debugging (leading hypotheses, for a focused session): (1) the per-token dynamic int8
  activation quant (dynamic_per_token_int8_quant -> int8_gemm_w8a8) intermediate buffers (x_q/x_s) being clobbered
  or frozen across the piecewise-split-at-collective boundaries during graph replay; (2) BF16 GDN linears executing
  inside captured pieces interleaved with the custom int8 op under TP=2. Repro is single-command (scripts/106
  IGP=false, or scripts/108 IGP=true). Until fixed: W8A8 27B serves coherent EAGER (~9 t/s MTP), and the fast
  coherent int8-activation 27B path is single-card W4A8 (captured, 2.03x, verified). Capture-recovery night CLOSED.

== 2026-06-24 :: [!!! BUG B ROOT-CAUSED + DISPROVEN] it is NOT "captured TP=2 numerics" -- it is COLLECTIVE EJECTION ==
motivation -> user: dig to the tiniest detail, fix the W8A8 27B TP=2 captured/MTP bug. The prior verdict ("captured
  TP=2 numerics are wrong, both partition modes garbage") had a HOLE: every garbage repro (scripts/106,108) ejects
  the TP collectives to eager via splitting_ops. Nobody had run 27B W8A8 TP=2 CAPTURED with the collectives LEFT
  INSIDE the captured graph after the ignore-list fix (the old 18.74 "baseline" was measured pre-fix, never read).
source dig (int8g vLLM, extracted to host:/mnt/vm_8tb/b70/bugb_src) ->
  - compilation/cuda_graph.py CUDAGraphWrapper (L161-167, L346-360): the wrapper does PURE .replay() and does NOT
    copy any runtime input into a static buffer. Contract: every captured-piece input MUST be at the SAME device
    address on replay as at capture (debug build asserts new_input_addresses == entry.input_addresses).
  - distributed/parallel_state.py + device_communicators/xpu_communicator.py: the TP collectives are ALL
    OUT-OF-PLACE. all_reduce = `output = input_.clone(); dist.all_reduce(output); return output` (fresh tensor).
    reduce_scatter / all_gather = `torch.empty(...)` + `.movedim().contiguous()` (fresh tensor(s)). Fakes return
    torch.empty_like / torch.empty (fresh).
  => MECHANISM: when a collective is EJECTED to eager (added to splitting_ops -> runs as a piecewise boundary), its
     FRESH output tensor must land at the exact capture-time address for the next captured piece's replay to read
     it. On XPU/oneCCL that address is not reproducible across forwards -> the next piece reads STALE capture-time
     data -> garbage. Captured single-card = no collectives -> no ejected boundary. Eager = no graph -> no address
     contract. That is the entire matrix.
config -> scripts/109_bugb_matrix.sh (parameterized launcher + COHERENCE READ-OUT via the container's python, since
  the host has no python3 -- the scripts/96 gotcha). Cell C: CKPT=27B-W8A8-graft, TP=2, GRAPH=1 PIECEWISE,
  EJECT=0 (collectives LEFT IN the captured graph -- splitting_ops = attn/GDN only, NO all_reduce/all_gather/
  reduce_scatter), no speculative-config. gpu-run both cards.
command -> SERVED=bugb_c TP=2 GRAPH=1 EJECT=0 KEEP=1 ./gpu-run bash scripts/109_bugb_matrix.sh
result -> HEALTHY, captured, COHERENT. temp=0 reads:
  Q1 "capital of France" -> "...The capital of France is Paris." (correct, with a clean think trace)
  Q2 "nth Fibonacci"     -> correct reasoning + standard F(0)=0,F(1)=1 definition.
  Load 112s (warm). NO crash, NO garbage.
verdict -> **PRIOR "captured-TP=2 numerics are broken" IS WRONG.** The captured W8A8 27B TP=2 compute is NUMERICALLY
  FINE. Bug B = COLLECTIVE EJECTION ONLY. scripts/106,108 produced garbage purely because they ejected the
  collectives (splitting_ops includes all_reduce/all_gather/reduce_scatter), which breaks the captured-piece input-
  address contract on XPU. Consequence: for NO-MTP, the W8A8 27B TP=2 captured serve just needs splitting_ops WITHOUT
  the collectives -> coherent AND captured (fast). The reason collectives were ejected at all was the MTP spec-verify
  all_gather (oneCCL can't SYCL-graph-record allgather). all_reduce/reduce_scatter DO record. So the MTP fix narrows
  to: handle the ONE all_gather (make it capturable, or eject only it with a stable-buffer landing). codex consult
  (gpt-5.5, read-only) independently ranked the collective boundary as #1. NEXT: (E) MTP spec=5 collectives-captured
  -> confirm it crashes only on all_gather; (F) eject ONLY all_gather; (G) capture-safe all_gather if F corrupts.

== 2026-06-24 :: [!!! MTP FIXED] eject ONLY all_gather (not all 3 collectives) -> captured TP=2 MTP COHERENT ==
motivation -> Bug B root cause = collective ejection breaks the captured-piece input-address contract. The old
  broken recipe ejected ALL THREE collectives (all_reduce + reduce_scatter + all_gather). Hypothesis: only the
  all_gather genuinely MUST be ejected (oneCCL 2021.17 cannot SYCL-graph-record allgather; the scripts/90 crash).
  all_reduce + reduce_scatter DO record -> they should stay CAPTURED. Ejecting the per-layer all_reduce (on every
  captured boundary) is what corrupted decode. So: eject ONLY all_gather, keep the other two captured.
config -> scripts/109 cell F: CKPT=27B-W8A8-graft, TP=2, GRAPH=1 PIECEWISE, IGP=false, MTP spec=5 + BF16-MTP shim,
  EJECT=ag (splitting_ops = attn/GDN ops + ONLY "vllm::all_gather"; all_reduce/reduce_scatter NOT listed -> captured).
  caps=[1,2,4,6,8] (incl. the 1+spec verify batch). gpu-run both cards.
command -> SERVED=bugb_f TP=2 GRAPH=1 EJECT=ag MTP=5 ./gpu-run bash scripts/109_bugb_matrix.sh
result -> capture SUCCEEDED (no oneCCL allgather crash -- all_gather is the ejected partition boundary, never
  recorded), HEALTHY, and COHERENT. temp=0 reads:
  Q1 "capital of France" -> "...The capital of France is Paris." (clean think trace)
  Q2 "nth Fibonacci"     -> correct reasoning + F(0)=0,F(1)=1.
  MTP shim confirmed ("[mtp-bf16-shim] Qwen3_5MultiTokenPredictor forced unquantized for grafted BF16 mtp.*").
verdict -> **MTP IS FIXED on captured W8A8 27B TP=2.** The minimal, correct config: PIECEWISE capture, IGP=false,
  eject ONLY all_gather, keep all_reduce + reduce_scatter CAPTURED. The old "all 3 ejected" recipe corrupted decode
  via the ejected all_reduce boundary; the new config ejects only the one collective that cannot be recorded, whose
  boundary turns out to be benign (its output is consumed in/next-to an eager region, not read by a captured piece
  across a graph boundary). NOTE: scripts/90-94 "63 t/s" headline numbers were benched on the all-3-ejected GARBAGE
  (token-count bench, never read text); the captured DECODE is identical here, so the real speed should be similar
  but now COHERENT. scripts/111 benches off vs spec=5 in this fixed config WITH a coherence read-out (running now).
  Plan B (scripts/110_csag_shim: capture-safe all_gather via all-reduce-of-padded so even all_gather stays captured,
  EJECT=none) is staged as a more-robust alternative if the eager all_gather proves a bottleneck.

== 2026-06-24 :: [!!! MTP REALLY WORKS] plan-B capture-safe all_gather -> 26.10 t/s COHERENT w/ REAL 26% accept ==
motivation -> eject-ONLY-all_gather (cell F) gave COHERENT output but the spec bench revealed a SECOND facet of
  Bug B: accept_rate ~0.001 (zero) -> MTP was pure overhead (9.63 t/s, SLOWER than no-MTP captured 18.10). The
  body stayed coherent only because rejected drafts fall back to the target's own token. So ejecting all_gather
  ALSO corrupts the multi-token spec-VERIFY path (where the all_gather lives), killing acceptance.
discriminator (scripts/111 + coherence read-out, hard Roman-Empire prompt, temp=0) ->
  config                                          decode_tps  accept_rate  accept_len  coherence
  EAGER, no-MTP                                   ~4.1        -            -           OK    (prior floor)
  CAPTURED, no-MTP (collectives captured)         18.10       -            -           OK    (cell C fix)
  EAGER, MTP spec5                                10.43       0.361        2.80        OK    (accept WORKS eager)
  CAPTURED, MTP spec5, eject-only-all_gather (F)   9.63       0.001        1.00        OK    (verify DEAD: 0% accept)
  CAPTURED, MTP spec5, PLAN-B all_gather (G)      26.10       0.258        2.29        OK    <- WIN
  - Eager MTP accepts 36% -> the BF16 drafter graft is GOOD; the 0% on F is capture+all_gather-eject, not the head.
  - PLAN B = scripts/110_csag_shim: monkeypatch XpuCommunicator.all_gather to an ALL-REDUCE-OF-PADDED
    (buf=zeros[world,*x]; buf[rank]=x; dist.all_reduce(buf); concat along dim) -- semantically identical to the
    base concat all_gather but built from all_reduce, which DOES SYCL-graph-record. So all_gather stays CAPTURED
    (EJECT=none, splitting_ops = attn/GDN only, NO collectives ejected) -> no ejected boundary anywhere -> verify
    is numerically correct -> drafts accepted (26%) AND the whole decode is captured (fast).
verdict -> **MTP IS NOW A REAL WIN ON CAPTURED W8A8 27B TP=2: 26.10 t/s, 26% accept, COHERENT.** Multipliers (all
  coherent): 26.10/18.10 = 1.44x vs captured-no-MTP; 26.10/10.43 = 2.50x vs eager-MTP; 26.10/4.1 = 6.4x vs
  eager-no-MTP. The config: PIECEWISE capture, IGP=false, splitting_ops = attn/GDN ONLY (eject NOTHING), + the
  capture-safe all_gather shim + the BF16-MTP graft shim (combined in scripts/110_csag_shim). The old "63 t/s"
  was garbage; the honest captured MTP headline is ~26 t/s (accept-limited by the 1-layer drafter on hard prompts;
  easier/code prompts accept higher -> faster). Captured-verify accept (26%) is a touch below eager (36%) -- a
  minor residual (captured int8 verify batch and/or all-reduce-of-padded numerics, or single-bench noise), net
  still +44%. NEXT: ship this as the recipe (csag shim, GRAPH=1, IGP=false, no-eject), smoke-gate, commit.

== 2026-06-24 :: captured-MTP spec sweep on the FIXED path -- spec=3 is the WINNER (34.82 t/s, 51% accept) ==
motivation -> the old "spec=5 winner, climbing 50/57/63" was on garbage. Re-sweep spec {3,4,5} on the FIXED captured
  config (plan-B capture-safe all_gather, eject nothing) with a coherence read-out + REAL accept (scripts/111).
result -> repo results/mtp111_*.csv (hard Roman-Empire prompt, temp=0, coherence-gated COHERENT all rows):
  spec  decode_tps  accept_rate  accept_len  vs captured-no-MTP(18.10)
  3     34.82       0.512        2.53        1.92x   <- WINNER
  4     30.56       0.368        2.47        1.69x
  5     26.10       0.258        2.29        1.44x
verdict -> captured MTP decode MONOTONICALLY DECREASES with spec (accept 51 -> 37 -> 26%): the 1-layer MTP head's
  useful draft horizon is ~3 tokens; drafting further just wastes verify compute. **spec=3 = 34.82 t/s @51% accept
  is the winner** = 1.92x vs captured-no-MTP (18.10), 3.3x vs eager-MTP (10.43), 8.5x vs eager-no-MTP (~4.1) -- ALL
  COHERENT. This is the EXACT OPPOSITE of the old garbage sweep (which "climbed" because degenerate draft==target
  gave fake ~98% accept at any spec). Recipe default flipped MTPTOK 5 -> 3. (spec=2 untested -- possible further
  small gain.) codex (gpt-5.5, read-only) REVIEWED the capture-safe all_gather shim: semantically identical to base
  vLLM concat all_gather for dim=0 and dim=-1, capture-safe, covers the real path (vllm::all_gather ->
  _all_gather_out_place -> device_communicator.all_gather + seq-parallel); NOT covering all_gatherv/gather (MoE/gather
  paths, not on this dense-hybrid's captured path -- noted in the shim). codex agrees the 51->26% accept-vs-spec
  trend and the eager(36%)-vs-captured(spec5 26%) gap are NOT shim numerics (sum-with-zeros is exact) but the
  captured int8 verify batch / drafter horizon / bench noise. FINAL HEADLINE for the W8A8 27B TP=2 MTP recipe:
  **~35 t/s coherent (spec=3)**, captured, real 51% accept. Bug B fully resolved end-to-end. Host clean; leases freed.

== 2026-06-24 :: PER-STEP DISASSEMBLY of TP=2 W8A8 27B (PP + TG) -> new doc 27b_w8a8_research.md ==
motivation -> Isaac wants every prefill/decode step disassembled into time/cycles/latency/bandwidth/compute ahead
of the Linux-7.0 (drm/xe pcie-p2p) migration, to know exactly what TP P2P buys. Method: component microbench at the
EXACT per-card TP=2 shapes (scripts/113, torch.xpu.Event back-to-back = in-graph per-op time) + allreduce bench
(both cards) + E2E [ref]. (vLLM torch-profiler endpoint is ABSENT on :int8g -- VLLM_TORCH_PROFILER_DIR is an unknown
env, /start_profile=404 -- so scripts/112's in-situ Kineto trace was a dead end; pivoted to the microbench.)
hardware [measured] -> B70 = Battlemage G31, 256 EU (32 Xe-cores), gt0 clock **2.8 GHz**, ~581 GB/s read, int8 GEMM
  ~290-305 TOPS, bf16 ~139-145 TFLOPs. Model: 64 layers = 48 GDN (BF16 in/out proj, in ignore-list!) + 16 full-attn
  (int8) + 64 MLP (int8). Per layer = 2 allreduces -> **128 allreduce / forward**.
per-op device times (scripts/113, card 0, per-card TP=2) -> DECODE M=1: MLP gate_up int8 157us @567GB/s(90%roof),
  down 77us @577(99%), GDN in_qkvz bf16 144us @581(roofline), out_proj 57us, qkvg 62us, act_quant K5120=41us /
  K8704=51us (serial reduce, persists under capture), lm_head 2149us (1.27GB bf16 read). PREFILL M=2048: gate_up
  1236us @295 TOPS, down 599 @304, GDN qkvz 1239 @139 TFLOPs.
allreduce [measured, scripts/allreduce_bench.py] -> **1.16 GB/s** host-staged (SYCL_KERNELS=1) / 0.68 eager; decode
  10KB = 88us (latency-bound); prefill 21MB = ~18ms each.
RESULT (apportionment) ->
  * DECODE 55ms/token (captured no-MTP 18.1 t/s): GEMM+quant ~40ms (73%, weight-BW-bound; bf16 GDN proj alone ~10ms
    /18% since GDN is unquantized), act-quant ~6.5ms (12%), 128 allreduce ~7-11ms (13-20%, latency-bound). Weight-
    read floor = 28ms (50.9 t/s). -> DECODE IS WEIGHT-BANDWIDTH-BOUND.
  * PREFILL 2748ms TTFT @2048 (745 tok/s): **128 allreduce x 21MB / 1.16 GB/s = ~2304ms = 84%**; int8 GEMM compute
    only 298ms (11%); other ~150ms. -> PREFILL IS COLLECTIVE-BOUND. The "~10x below compute ceiling" gap is the wire.
KERNEL-7.0 P2P (quantified) -> if allreduce 1.16 -> ~10-13 GB/s (Gen3 wire): PREFILL TTFT 2.75s -> ~0.7s = **~4x**;
  DECODE single-stream 18.1 -> ~21.7 t/s = **~1.2x**; big c>1 concurrency gain. So P2P is a PREFILL/TTFT + multi-user
  win, NOT a single-stream decode win (decode needs int4 + quant-fusion + MTP, orthogonal & stacks with P2P).
cross-check -> codex (gpt-5.5, analytical, no measurement) independently derived the 128-allreduce count and matched
  the per-op roofline within a few % (gate_up 153us vs 157, GDN qkvz 144 match, lm_head 2.19 vs 2.15). It differed
  ONLY on the interconnect (assumed 8-16 GB/s prefill allreduce / Seguin's 15-17us decode) -- exactly where our
  MEASURED 1.16 GB/s / 88us flips prefill to 84% collective-bound. Measurement was the decisive contribution.
verdict -> full per-step diagnosis in 27b_w8a8_research.md (ASCII data-flow diagrams for one decode token + one
  prefill pass, per-op roofline tables, optimization board). docs/P2P_GPU.md H.10 = the allreduce datapoint. Both
  cards used (microbench card 0; allreduce + the earlier eager serve both cards). Host clean; leases freed.

## 2026-06-23 -- MIGRATION to Ubuntu 26.04 / kernel 7.0 + B70<->B70 P2P UNLOCKED [HEADLINE]
host -> b70s4dayz, Ubuntu 26.04 LTS, kernel 7.0.0-22-generic, booted off the 500G NVMe (Samsung 970 EVO,
  nvme0n1). Both B70s under xe (0b:00.0, 44:00.0; renderD128/129). Data drives carried over by UUID (Phase 3,
  all serials/UUIDs match MIGRATION.md sec 0; PARITY JEH9VZHN reshuffled sdb->sda as warned). BIOS: IOMMU OFF
  (iommu_groups=0), ACS off, mem-interleave off (kernel still shows 1 NUMA node). No iommu= kernel param.
config -> kernel 7.0 + IOMMU off (BIOS) + DEFAULT L0 env (stock 26.04 archive: intel-opencl-icd / libze1 /
  libze-intel-gpu1, all 26.05.37020.3 NEO).
command -> cd /mnt/vm_8tb/b70 && ./gpu-run python3 71_ze_p2p_ctypes.py
result ->
  zeDeviceCanAccessPeer dev0<->dev1 = True (BOTH directions)      [6.18: False on all 12 variants, H.9]
  zeDeviceGetP2PProperties flags = 0x1  ACCESS=Y ATOMICS=N        [6.18: 0x0]
  IPC path: zeContextCreate / zeMemAllocDevice / zeMemGetIpcHandle = 0x0; zeMemOpenIpcHandle(peer) = 0x0 PEER MAP OK
    [6.18: zeMemOpenIpcHandle = 0x78000004 FAILED -- the exact call that blocked oneCCL drmfd TP]
  True with DEFAULT env -- no EnableCrossDeviceAccess / EnableP2P / affinity-mask hacking needed.
verdict -> B70<->B70 P2P is AVAILABLE on kernel 7.0 with IOMMU disabled. Both H.9 levers necessary, both now hold:
  (A.1) drm/xe pcie-p2p interconnect path shipped in 7.0, AND (A.2) IOMMU-off restores the AMD-Zen (fam 0x17)
  pci_p2pdma allow-list. The unpublished F.3/F.4 datapoint, now measured. Closes I.1/I.2 reboot-gated TODOs.
  NEXT (the BW prize, H.10): re-run scripts/allreduce_bench.py on 7.0 -- does 1.16 GB/s climb toward the ~15 GB/s
  Gen3 wire (=> ~4x prefill TTFT)? Needs Docker + int8g image (not yet installed); also build ze_peer
  (level-zero-tests, not in 26.04 apt) for the authoritative peer BW/latency matrix. See P2P_GPU.md H.11.

## 2026-06-23 -- [HEADLINE, MEASURED] P2P-ON allreduce = 9.7 GB/s = 8.4x over host-staged (the H.10 prize, CONFIRMED)
Same-day follow-up to canAccessPeer=True (H.11): MEASURED the bandwidth the entire P2P case rests on. Recovered the
exact bench image vllm-xpu-env:v0230 from the old Unraid docker.img (200G btrfs loopback on cache -- images were NOT
lost) via a throwaway dockerd save/load; int8g recovered too. Docker installed with data-root on /mnt/vm_8tb/docker.
config -> kernel 7.0, IOMMU off, vllm-xpu-env:v0230 (torch 2.11.0+xpu, native xccl; no ipex/oneccl_bindings), 2x B70
  cross-die. A/B = CCL_TOPO_P2P_ACCESS 0 vs 1  x  CCL_ENABLE_SYCL_KERNELS 0 vs 1, IPC=pidfd.
  GOTCHA: this oneCCL accepts CCL_ZE_IPC_EXCHANGE=sockets|pidfd ONLY (the old 6.18 drmfd is rejected -> ValueError).
command -> cd /mnt/vm_8tb/b70 && ./gpu-run bash 61_allreduce_p2p_ab.sh   (script also in repo scripts/)
result -> allreduce algbw=busbw (GB/s), 2 GPUs:
  msg      A p2pOFF eager   B p2pOFF sycl(=H.10)   C p2pON eager   D p2pON sycl
  1 MB     0.67             1.22                   1.16            9.77
  16 MB    0.66             1.18                   3.35            9.43
  256 MB   0.67             1.14                   3.43            9.70    (D peak 10.22 @ 8MB)
  small-msg (decode ~10KB) latency ~0.085-0.09 ms in BOTH B and D (latency-bound -> P2P ~1.2x decode, as predicted).
verdict -> P2P-on + SYCL kernels (D) = ~9.7 GB/s plateau vs 1.16 host-staged (B) = **8.4x**, ~61% of the 15.8 GB/s
  Gen3 x16 wire. PREFILL recompute with the measured 9.43 GB/s @16-21MB: 128 allreduce x 21MB / 9.43 = ~283 ms (was
  ~2304 ms @1.16) -> TTFT 2748 ms -> ~727 ms = **~3.8x faster prefill**. H.10's estimate is now MEASURED. The
  kernel-7.0 migration delivered its headline payoff. P2P_GPU.md H.12. NEXT (end-to-end confirm): TP=2 P2P-on serve
  A/B with int8g (recovered) -> real-world TTFT. Both cards (lease held whole run). Host clean; lease freed.

## 2026-06-24 -- shelf bench on the new install: all 6 Qwen3.6 models serve COHERENTLY (install validated)
config -> new Ubuntu/kernel-7.0 box, images recovered from old Unraid docker.img (v0230/int8g/v0230moe). Each model
  via its own rdy_to_serve serve.sh (real GRAPH/TP/MTP defaults). vllm bench serve, random, IN=2048 OUT=128, captured.
  TP=1 models swept TWO-UP (one per card, ZE_AFFINITY_MASK) via new 68_shelf_bench_par.sh; TP=2 solo.
command -> ./gpu-run/sg-docker: bash 68_shelf_bench_par.sh qwen36  (+ solo re-bench of w4a8 and moe)
result -> PP = 2048/TTFT (prefill tok/s); TG = decode tok/s:
  model                          TP  TTFT_c1  PP_c1  TG_c1   TTFT_c4  agg_c4
  qwen36-35b-a3b-int4 (MoE)      1   441ms    4641   68.5    1239ms   123.7   <- fastest (A3B ~3B active)
  qwen36-27b-w4a8                1   853ms    2400   20.7    2201ms   51.2    <- fastest 27B prefill
  qwen36-27b-w4a16               1   1224ms   1673   21.2    3213ms   45.5
  qwen36-27b-int4                1   1326ms   1545   30.5    3438ms   51.3
  qwen36-27b-w8a8-sqgptq-mtp     2   2961ms   692    25.2*   6837ms   19.0    *MTP spec=3, RANDOM data (NL ceiling ~35)
  qwen36-35b-a3b-quark-w8a8      2   2241ms   914    4.6     5899ms   13.8    <- slowest (35B dense int8 TP=2, EAGER)
  NB: both TP=2 rows ran HOST-STAGED (serve recipes default CCL_TOPO_P2P_ACCESS=0) -- P2P NOT exercised here.
  Random-data bench depresses MTP accept (undraftable tokens); the 25.2 understates the ~35 t/s NL ceiling.
verdict -> NEW INSTALL VALIDATED: every shelf model serves + passes the coherence-gated gen probe with the recovered
  images on kernel 7.0 + xe. Parallel TP=1 benching is clean (27b-int4 co-resident decode 30.48 == solo 30.32 t/s).
  One transient: w4a8 OOM'd engine-init in parallel wave-2 (card VRAM not fully released from wave-1 teardown before
  realloc @UTIL=0.90) -> re-benched solo OK; 68 now settles 15s between waves to prevent it. Table -> README.
  CSVs in results/sweep_*.csv. Both cards used; host clean; leases freed.

## 2026-06-24 -- [LEVER A] P2P in vLLM serve is BLOCKED (DEVICE_LOST at worker init) -- microbench-only for now
motivation -> H.12 showed P2P allreduce = 8.4x; test the END-TO-END serve win on 27B W8A8 TP=2 (P2PACCESS 0 vs 1).
config -> qwen36-27b-w8a8-sqgptq-mtp serve.sh run, GRAPH=1 (->SYCLKERNELS=1), IN=2048 OUT=128 c=1/4, int8g.
command -> ./gpu-run bash 69_lever_tests.sh A   (+ IPCX=sockets isolation re-run)
result ->
  P2P OFF (P2PACCESS=0, host-staged): HEALTHY + coherent. c1 TTFT 2901ms decode 26.2 t/s; c4 TTFT 4576 agg 20.5.
  P2P ON  (P2PACCESS=1, pidfd):  CRASH at worker init_device warmup `all_reduce(torch.zeros(1).xpu())`
    (xpu_worker.py:105) -> RuntimeError: level_zero backend failed with error 20 (UR_RESULT_ERROR_DEVICE_LOST).
    CCL logs confirm CCL_TOPO_P2P_ACCESS=1 took effect; engine core init fails -> container exits.
  P2P ON  (P2PACCESS=1, sockets): SAME crash -> NOT an IPC-exchange-mechanism issue.
verdict -> the crash is in init_device BEFORE any graph capture -> CAPTURE RULED OUT; it is oneCCL's P2P all_reduce
  inside vLLM's multiproc-executor worker topology that loses the device. The RAW 2-rank mp.spawn allreduce
  microbench (61 / H.12) hits 9.7 GB/s with the SAME CCL_TOPO_P2P_ACCESS=1 -> the P2P FABRIC WORKS; the gap is the
  oneCCL <-> vLLM-multiproc-worker P2P path (separately spawned workers + IPC handle exchange), NOT hardware.
  => the ~3.8x P2P prefill prize (H.12) is REAL at the collective layer but NOT YET accessible through vLLM serve.
  Both GPUs recovered (xpu count 2) after the loss. FOLLOW-UPS (dedicated session): VLLM_WORKER_MULTIPROC_METHOD=fork,
  a newer oneCCL, NEO EnableP2P/EnableCrossDeviceAccess debug keys, or a custom P2P all-reduce bypassing the warmup.
  P2P_GPU.md H.13. P2P-OFF remains the only working serve path today (TTFT 2901ms is host-staged, as expected).
SEVERITY ESCALATION -> the P2PACCESS=1 DEVICE_LOST does NOT clean up: it WEDGES the cross-GPU oneCCL/Level-Zero
  state. After the two P2P-on attempts, a fresh container running the KNOWN-GOOD P2P-OFF 27B W8A8 TP=2 (the exact
  config that served minutes earlier in Phase A) ALSO failed with the identical UR_RESULT_ERROR_DEVICE_LOST at
  xpu_worker init_device all_reduce -- so EVERY TP=2 serve is broken until the GPU state is reset. Single-GPU is
  unaffected (xpu count 2; all TP=1 serves fine). Recovery = reload xe (`modprobe -r xe; modprobe xe`, needs no
  /dev/dri in use) or reboot. LESSON: do NOT retry P2PACCESS=1 in serve without a GPU reset between attempts --
  it corrupts the multi-GPU collective state for all subsequent TP>1 runs.

## 2026-06-24 -- [LEVER C] MTP on the 35B-A3B int4 MoE: works no-graft, ~1.1x single-stream, HURTS at concurrency
config -> qwen36-35b-a3b-int4 (Intel AutoRound, 2335 mtp tensors PRESENT), TP=1 single-card, GRAPH=1, fp8_e5m2 KV,
  IN=2048 OUT=128 c=1/4, RANDOM data (underrates MTP accept), v0230moe. 69_lever_tests.sh C.
result ->
  arm        c1_TTFT   c1_decode   c4_TTFT   c4_agg   c4_perstream
  MTP off    440.7ms   66.05 t/s   1235ms    123.40   43.65
  MTP spec3  506.8ms   73.58 t/s   1361ms    59.93    18.29
verdict -> MTP is COHERENT on the MoE with NO graft (the int4 ckpt's mtp head works as-is, unlike the quark W8A8
  which dropped its mtp tensors). c=1 decode 66 -> 74 t/s = ~1.11x (a FLOOR; random data suppresses accept, NL
  higher), at a small TTFT cost (440 -> 507ms). But c=4 MTP HURTS HARD: agg 123 -> 60, per-stream 43.6 -> 18.3 --
  spec verify competes with concurrent decode. ROOT: the MoE activates only ~3B params/token -> decode already
  fast (NOT weight-BW-bound like the 27B dense), so MTP's verify overhead nearly cancels the draft gain. NET: MTP
  is a marginal single-stream lever for this MoE, net-negative under load -- UNLIKE the 27B dense (1.9x). Also
  CONFIRMS single-card serves are UNAFFECTED by the Lever-A multi-GPU wedge. Host clean; lease freed.

## 2026-06-24 -- [LEVER A RECOVERY] reboot clears the P2PACCESS=1 DEVICE_LOST wedge
context -> after the two P2PACCESS=1 serve attempts wedged the cross-GPU oneCCL/Level-Zero state (every TP=2
  serve, even known-good P2P-off, failing at xpu_worker init_device all_reduce with UR_RESULT_ERROR_DEVICE_LOST).
action -> rebooted the box.
result -> wedge CLEARED; TP=2 serve path restored. xe driver reload (modprobe -r xe; modprobe xe, no /dev/dri in
  use) is the lighter alternative but the reboot is the confirmed-working recovery.
verdict -> Lever B (35B quark-W8A8 eager vs captured) is now UNBLOCKED. Recorded the do-not-chain-P2PACCESS=1 rule
  in AGENTS.md (GPU Discipline danger note) and README levers so this is not re-discovered the hard way.

## 2026-06-24 -- [LEVER B] 35B quark-W8A8 TP=2: PIECEWISE graph capture = 8.7x decode over eager
context -> Lever-A wedge cleared by reboot (prev entry) -> the pending eager-vs-captured comparison is unblocked.
  Ran arms 1+2 ONLY (eager P2P0, captured P2P0); arm 3 (P2PACCESS=1) DELIBERATELY SKIPPED -- it re-wedges the box
  (see AGENTS.md GPU Discipline). Driver = scratchpad one-off replicating arm() from 69; numbered script untouched.
config -> qwen36-35b-a3b-quark-w8a8-int8 (TRUE int8 MoE, 256 routed experts via Triton fused_moe), TP=2, v0230,
  P2PACCESS=0 (host-staged), IN=2048 OUT=128, c=1/4, RANDOM data. GRAPH=0 (eager) vs GRAPH=1 (PIECEWISE capture).
command -> ./gpu-run bash <scratchpad>/lever_b_12.sh  (gpu-run held both cards 1249s; lease freed clean)
result ->
  arm                  c1_TTFT  c1_TPOT   c1_decode   c4_TTFT  c4_agg   c4_perstream
  eager   (GRAPH=0)    2172ms   201.7ms   4.96 t/s    5763ms   14.37    4.26
  captured(GRAPH=1)    1512ms    23.2ms   43.05 t/s   3866ms   53.20    22.08
  speedup              1.44x    8.7x      8.7x        1.49x    3.7x     5.2x
  CSVs: results/sweep_quark-eager-p2p0-tp2_*.csv , results/sweep_quark-graph-p2p0-tp2-graph_*.csv
verdict -> HUGE win. Eager's 201ms TPOT is pathological: this MoE activates only ~3B params/token, so per-token
  Python op-launch overhead (256-expert Triton fused_moe dispatched eagerly) DOMINATES the tiny compute. PIECEWISE
  capture removes the launch overhead -> TPOT 201 -> 23ms -> decode 5.0 -> 43.0 t/s (8.7x). Both arms coherent
  (gen-probe OK). This means the shelf table's 4.6 t/s for this model (served EAGER, serve.sh GRAPH default 0)
  drastically understated it: CAPTURED it is the FASTEST single-stream 35B we serve (43 t/s, beating the 27B dense
  at 20-30 t/s). RECOMMENDATION: flip the quark-w8a8 serve.sh default to GRAPH=1 (pending a coherence smoke at all
  conc; capture worked clean here at c=1 and c=4). NOTE: the c4 agg gain (3.7x) < c1 (8.7x) because at c>1 the
  per-op overhead amortizes across the batch even eager, so capture's marginal benefit shrinks -- but it is still
  strongly net-positive at concurrency (UNLIKE Lever C's MTP-on-MoE, which went net-negative at c>1). Host clean.

== 2026-06-24 P2P campaign J.8-J.14: hand-rolled PUSH all-reduce beats oneCCL, LIVE in 27B-W8A8 TP=2 serve ==
config -> kernel 7.0 + new BIOS (IOMMU off), 2x B70 cross-die. Built a custom posted-write all-reduce below
  oneCCL and wired it into vLLM. Full detail in docs/P2P_GPU.md section J (J.8-J.14). Scripts 103-108 + contrib/
  vllm_push_allreduce/. Target model: qwen36-27b-w8a8-sqgptq-mtp (rdy_to_serve UNEDITED; 108 is a standalone wrapper).
command -> ./bin/gpu-run bash scripts/{103,104,105,106,107}_run_*.sh ; ./bin/gpu-run bash scripts/108_serve_push_ar_ab.sh {smoke,start,bench,stop}
result ->
  J.8  2-proc IPC peer-write          11.08 GB/s (= single-dir ceiling; IPC boundary free)
  J.9  decode latency: events 1.36x (44us); device-flag fusion HANGS (Xe: mid-kernel peer write invisible)
  J.10 2-proc custom allreduce        34.5us decode (2.5x oneCCL), 10.64 GB/s prefill (1.13x); shm spin barrier; P2PACCESS=0
  J.11 bind to TORCH's L0 ctx         runs on real torch.xpu tensors via sycl_queue addr (no pybind); 10.6 GB/s, verify OK
  J.12 bf16 drop-in                   9.9 GB/s prefill (4-byte-word push; 2-byte bf16 push was 12x slower)
  J.14 LIVE 27B-W8A8 TP=2 serve A/B (eager, MTP spec=3, push-ar vs oneCCL):
       c1 TTFT 1613->481ms (3.35x), c1 decode 8.40->12.41 t/s (+48%), c1 out 7.65->11.95 (+56%), c4 out 19.5->31.9 (+64%)
       [push_ar] ENGAGED confirmed in logs; gen probe COHERENT; sitecustomize chained the MTP shim cleanly.
verdict -> WIN. A hand-rolled posted-write collective beats the vendor lib (oneCCL) at the allreduce layer AND
  end-to-end in a coherent production-shaped serve, using its own L0-IPC P2P (independent of CCL_TOPO_P2P_ACCESS, so
  P2PACCESS=0 -> oneCCL warmup stays host-staged -> dodges the H.13 DEVICE_LOST wedge). CAVEAT: this is EAGER-vs-EAGER
  (isolates the collective). Production runs CAPTURED (GRAPH=1) which push-ar can't enter (host barrier not graph-
  recordable); prefill is NOT captured so the 3.35x TTFT win should survive in production. NEXT: size/capture-gated
  engagement (push-ar for prefill only, oneCCL-captured decode); PP=2 prototype (1 push/microbatch vs 128 allreduces).
  All committed+pushed. Host clean, GPU lease freed.

---

## 2026-06-24 [BLOCKED] P2P J.16 -- capture-gated A/B re-wedged the box (worse than H.13)
config -> post-reboot resumption; box rebooted ~1h40m prior, both cards free, clean tree. Goal: bank the
  J.14 3.35x TTFT win inside a GRAPH=1 production serve via the capture-gated path (push-ar prefill-only,
  oneCCL-captured decode). Image int8g, TP=2, GRAPH=1, P2PACCESS=0, PUSH_AR_MIN_NUMEL=65536.
command -> GRAPH=1 PUSH_AR_MIN_NUMEL=65536 ./bin/gpu-run bash scripts/108_serve_push_ar_ab.sh smoke
result ->
  J.15 deferred-dlopen FIX HELD: load got PAST the rotary-emb crash point, XPUInt8 kernel selected, both
       TP0/TP1 workers alive, safetensors 50% loaded. But never reached /health in the 15-min wait window.
  On script-stop, worker shutdown threw _cleanup_profiling_kv_cache -> synchronize() -> DEVICE_LOST (err 20).
  WORSE THAN H.13: post-teardown a TP=1 SINGLE-CARD probe also failed -- 2048x2048 matmul OOM'd after 87s
       (UR_RESULT_ERROR_OUT_OF_RESOURCES, err 40); a 16x16 retry HUNG >13 min. No userspace cause (no stray
       procs / containers; lease free) -> xe/driver-level degradation on BOTH cards.
verdict -> BLOCKED on a GPU reset (root; not available this session). Trigger model now 3 datapoints (H.13
  P2PACCESS=1 / J.15 chained worker-init crashes / J.16 workers killed mid-GRAPH-capture by the health
  timeout); common factor = TP>1 teardown while the L0/oneCCL collective ctx is mid-op. Suspicion: the
  15-min timeout may have SIGKILLed a merely-SLOW capture (partly self-inflicted). Full reasoning + the open
  decisions (scoped passwordless sudo for modprobe xe; pre-flight env guard + auto-reset-on-DEVICE_LOST;
  kernel-7.1 purgeable-BO assessment) in docs/20260624_devicelost_thoughts.md; factual log in P2P_GPU.md J.16.
  The J.14 EAGER win is UNAFFECTED -- only the capture-gated production variant is still unmeasured.

P2P J.17 [GUARD] -- wedge guard shipped (detect+recover, not prohibit) ---------------------------------
config -> clean box (rebooted, both cards free). Principle: heal aggressively, prohibit almost nothing,
  so TP>1 / P2P pioneering is NOT hampered. Root cause = TP>1 worker SIGKILLed mid-collective; our own
  flat-900s health timeout -> docker rm -f supplied the SIGKILL.
command -> new bin/xpu-health (per-card matmul probe, timeout-wrapped, exit 0/1/2) + bin/xe-reset
  (stop containers -> modprobe -r/xe reload -> re-probe, scoped sudoers) + lib.sh guard (TP>1 only):
  L1 pre-flight probe, L2 graceful docker stop -t teardown, L3 stall-aware health wait (raised 1800s
  ceiling for TP>1+GRAPH=1), L4 post-teardown verdict + B70_AUTO_RESET, L5 refuse P2PACCESS=1 in TP>1
  serve unless I_KNOW_P2P_WEDGES=1. TP=1 path byte-for-byte unchanged (sweep gate unaffected).
result -> xpu-health VALIDATED live: both cards OK, exit 0, 16s. lib.sh + tools syntax-clean; xe-reset
  --dry-run clean. PENDING: install bin/xe-reset.sudoers (1 root cmd); run bin/serve-sweep --smoke before
  committing the lib.sh change (shared-infra gate).
verdict -> guard live; NEXT = re-attempt the capture-gated A/B with B70_AUTO_RESET=1 + the longer capture
  budget, and confirm whether GRAPH=1 TP=2 capture genuinely needs >15min. Detail in P2P_GPU.md J.17.

P2P J.17 RE-ATTEMPT [WIN] -- guard passes the exact J.16 wedge command, no wedge ---------------------
config -> clean box, guard live. command ->
  B70_AUTO_RESET=1 HEALTH_TIMEOUT=2400 HEALTH_STALL=600 GRAPH=1 PUSH_AR_MIN_NUMEL=65536 \
    ./bin/gpu-run bash scripts/108_serve_push_ar_ab.sh smoke
result -> L1 pre-flight both cards OK -> GRAPH=1 TP=2 push-ar HEALTHY in 147s -> coherent gen ("Paris.")
  -> L2 graceful docker stop -t30 -> L4 post-probe both cards OK. exit 0, 188s. BOX DID NOT WEDGE.
verdict -> guard validated end-to-end on the real wedge-prone path. The J.16 ">15min capture" suspicion
  is FALSE (capture = 147s) -> J.16 was a PRE-EXISTING degraded box (no pre-flight then) whose 900s-stuck
  serve got force-killed into the full wedge; L1 catches that class, L2 removes the force-kill. BONUS: the
  capture-gated GRAPH=1 prefill-only push-ar PRODUCTION variant is now MEASURED healthy+coherent (was the
  J.14/J.16 blocked item); full A/B bench (push vs oneCCL, GRAPH=1) still TODO. P2P_GPU.md J.17.

P2P J.17 FULL A/B BENCH [WIN] -- capture-gated push-ar beats oneCCL, GRAPH=1 TP=2 27B-W8A8 ----------
config -> both GRAPH=1 TP=2, IN=512/OUT=128, CONC 1/2/4/8. A=push-ar prefill-gated, B=oneCCL baseline.
  Ran A then B chained (two TP=2 GRAPH=1 starts) -- guard carried both, graceful teardown + post-probe
  between, NO wedge, box ended healthy.
result -> out_tok/s A vs B: c1 29.70/25.44 (+16.7%), c2 45.40/39.64 (+14.5%), c4 59.86/47.52 (+26.0%),
  c8 107.74/69.52 (+55.0%). mean TTFT A vs B (ms): c1 321/811, c2 455/1101, c4 561/1296, c8 724/1659
  = 2.3-2.5x faster. CSVs in $ROOT/results (sweep_*-pushar-tp2-graph_170832, *-oneccl-tp2-graph_171516).
verdict -> production GRAPH=1 capture-gated push-ar is a clear win (+15-55% thruput, 2.3-2.5x TTFT);
  J.14/J.16 blocked variant now MEASURED. Decode-side capturable push is the next lever. Also wrote
  docs/literature/p2p_access_devicelost.md (why CCL_TOPO_P2P_ACCESS=1 wedges: oneCCL peer copy across
  cross-die boundary -> xe copy-engine reset -> L0 DEVICE_LOST; xe-level corruption, not recoverable).
  P2P_GPU.md J.17.

P2P J.18 [MEASURED] PP=2 first run on current Gen3 box -- prefill/TTFT + scaling win, decode eager-gated --
config -> 27B-W8A8 PP=2/TP=1 EAGER no-MTP, IN=512/OUT=128, guard-wrapped (scripts/109_serve_pp2.sh, run
  via scratchpad pp2_inner: pre-flight probe -> serve -> 35_sweep_bench -> graceful stop -> post-probe).
result -> coherent ("Paris."), NO wedge. PP=2 out_tok/s|ttft|ps_decode: c1 6.03|345|6.08, c2 11.82|637|6.05,
  c4 23.05|938|5.98, c8 44.21|1394|5.84. vs eager push-ar TP=2 WITH MTP (J.14): c1 7.64|1603|8.38 ...
  c4 18.44|2538|5.55. PP=2 TTFT ~4.7x lower; per-stream decode FLAT ~6 across c1-c8 while eager TP=2 decode
  degrades under load; PP aggregate scales to 44 t/s @c8.
verdict -> J.13 PP bet CONFIRMED on prefill/TTFT + concurrency scaling at matched eager config (1 handoff vs
  ~128 allreduces). BUT PP=2 decode 6 t/s is ~5x below production GRAPH=1+MTP TP=2 push-ar (30 t/s, J.17) --
  that gap is capture+MTP, not topology. PP=2 PROMISING, not yet production. NEXT for PP: GRAPH=1 capture
  then +MTP (both unproven w/ PP send/recv). TP=2+push-ar stays the production path. P2P_GPU.md J.18.

P2P J.19 [BLOCKED] PP=2 production config (GRAPH=1+MTP) -- two upstream blockers ---------------------
config -> built lib.sh PP support (b70_multicard) + scripts/110_serve_pp2_graph_mtp.sh (27B shelf
  GRAPH=1+MTP env, TP=2 -> PP=2/TP=1). command -> 110 smoke (MTP), then MTPTOK= (no-MTP).
result -> (1) MTP+PP UNSUPPORTED: config-time NotImplementedError, MTP drafter lacks SupportsPP interface.
  (2) PP=2+GRAPH=1 no-MTP: HEALTHY (capture records the PP handoff) but /v1/completions EMPTY on all 3
  probes -- captured PP numerically broken on the W8A8+GDN hybrid (same class as captured-TP=2 bug B).
  Eager PP=2 was coherent (J.18). Box stayed clean (guard handled all; no wedge).
verdict -> PP=2 has NO working production path today: only coherent form is eager no-MTP ~6 t/s, 5x below
  production TP=2+push-ar+MTP (30 t/s). TP=2+push-ar stays production; PP=2 PARKED pending upstream
  (SupportsPP on MTP drafter + captured-PP-hybrid numerics fix). Tooling (lib.sh PP, scripts/110) committed
  for a one-line retry. P2P_GPU.md J.19.

P2P J.20 [INFRA] captured-PP corrupts collective state; xe-reset CANNOT recover this box ------------
result -> the string of captured-PP serves corrupted the multi-GPU collective state: the next production
  TP=2 serve produced empty output + DEVICE_LOST, post-probe found card 1 HUNG. Single-card pre-flight
  passed (doesn't exercise the collective -> guard gap; add a 2-proc allreduce probe). xe-reset FAILED:
  `modprobe -r xe` -> FATAL "Module xe is in use" even with 0 containers -- xe drives the console/display
  (baseline refcount ~5), so it is NEVER removable while up. REBOOT is the only recovery on this box.
verdict -> do NOT run captured PP=2. Corrected AGENTS.md + bin/xe-reset (escalate to reboot). P2P_GPU.md J.20.

P2P J.21 [WIN] post-reboot clean production A/B (IN=2048) -> README updated to push-ar best ----------
config -> 27B-W8A8 TP=2 GRAPH=1 MTP, IN=2048/OUT=128, clean box. A=oneCCL default, B=PUSH_AR=1 overlay.
result -> both coherent, chained cleanly (guard graceful teardown + post-probe healthy between, NO wedge;
  also validated the lib.sh PP/b70_multicard refactor on the live TP=2 path). push-ar vs oneCCL: c1 TTFT
  762 vs 2916 ms (3.83x), prefill 2688 vs 702 tok/s (3.83x); c4 agg 48.2 vs 26.9 (+80%); c8 68.5 vs 32.7
  (+109%). decode unchanged ~25 t/s (prefill-only push). CSVs: *-tp2-graph_192053, *-pushar-tp2-graph_193418.
verdict -> README 27B-W8A8 TP=2 row updated to push-ar best (762/2688/25.3/1354/48.2, labelled push-ar);
  other 5 rows audited (agents) vs all CSVs/JOURNAL = already best, unchanged. P2P_GPU.md J.21.

P2P K.1-K.2 [RECON+FINDING] decode-capturable push all-reduce -- toolchain has primitives; SYCL graph is 1-device
config -> new session (handoff_decode_push_ar.md): make push-ar graph-recordable so DECODE all-reduces use the
  11 GB/s posted-write path instead of falling back to oneCCL inside the captured graph.
command -> header/version recon in :int8g; scripts/114_graph_allreduce.cpp (SYCL command_graph, 2-device record).
result -> (K.1) toolchain HAS it all: SYCL_EXT_ONEAPI_GRAPH=1, L0 IPC events (zeEventPool{Get,Open}IpcHandle,
  AppendWaitOnEvents/SignalEvent), SYCL external_semaphore (opaque_fd/timeline_fd). (K.2) a SYCL command_graph
  is SINGLE-DEVICE: begin_recording rejects a 2nd device -> cannot put a cross-device edge in one graph.
verdict -> the decode sync must be an EXTERNAL primitive recorded into each rank's OWN single-device graph
  (L0 IPC event command-streamer wait, or SYCL external_semaphore) -- which matches vLLM (one captured graph
  per TP worker). NEXT: prove the raw-L0 cross-device command-streamer event wait is replayable (the oneCCL
  mechanism, the keystone). No GPU wedge (single-ctx test). docs/P2P_GPU.md K.1-K.2.

P2P K.3 [KEYSTONE WIN] cross-device command-streamer L0-event wait is correct + replayable on B70
config -> scripts/115_ze_event_sync.c: 2 closed L0 command lists (1/card), push(peer memcpy)+signal-event,
  AppendWaitOnEvents(peer), proxy read; re-executed 200x with per-iter sentinel + poisoned scratch.
command -> ./bin/gpu-run bash scripts/115_run_event_sync.sh
result -> verifyA/verifyB OK at 10KB/64KB/1MB across all 200 replays; decode-sized sync ~17-21us.
verdict -> the J.9-C EU-spin dead end is BYPASSED: a command-streamer/HW-semaphore wait (zeCommandListAppend
  WaitOnEvents) signaled by the peer card IS correct AND replayable (closed lists = graph replay). The push
  all-reduce rank-sync can be made graph-recordable -> the decode-capture path is open. ~17-21us already beats
  J.9-B (44us) and oneCCL (~85us). No wedge (single-ctx). docs/P2P_GPU.md K.3. NEXT: full allreduce w/ real
  reduce kernel + this sync (K.4), then the SYCL-graph/external-semaphore form (K.5).

P2P K.4 [WIN] full push all-reduce records into a SYCL command_graph + replays correctly -> decode IS capturable
config -> scripts/116_graph_native_ar.cpp: per-rank command_graph [push kernel]->[native cmd: signal/wait/reset
  L0 events via ext_codeplay_enqueue_native_command + ext_codeplay_get_native_graph]->[reduce kernel]. torch-xpu
  XPUGraph == sycl command_graph (ATen/xpu/XPUGraph.h), so this is the exact capture mechanism vLLM uses.
command -> ./bin/gpu-run bash scripts/116_run_graph_native_ar.sh
result -> verifyA/verifyB OK(sum) across 200 replays at 10KB/64KB/1MB; perLaunch 64.5us @decode standalone.
verdict -> the DECODE all-reduce IS graph-capturable on B70 (handoff central question = YES). push+cross-device
  L0-event sync+reduce all record into a SYCL command_graph and replay correctly. get_native_queue<level_zero>
  is broken in DPC++ 2025.3 -> use ext_codeplay_get_native_graph. No wedge (single-ctx). docs/P2P_GPU.md K.4.
  NEXT: cross-process IPC event pools (K.5), then wire into push-ar .so + GRAPH=1 PUSH_AR_MIN_NUMEL=0 serve A/B.
