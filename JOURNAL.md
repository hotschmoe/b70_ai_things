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

<!-- New entries below -->
