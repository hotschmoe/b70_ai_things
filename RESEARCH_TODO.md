# RESEARCH_TODO.md -- W8A8-INT8 is the main path; next research tracks

**Created:** 2026-06-20
**Status:** PLAN -- consolidates a strategy info-dump (deduped) + adds AutoRound (autoint) + Quark.
**Authoritative siblings (do NOT duplicate their content here):**
- [`MTP_TODO.md`](MTP_TODO.md) -- owns ALL MTP / speculative-decoding planning (the ~3-4x decode lever).
- [`docs/literature/07_w8a8_int8_recovery.md`](docs/literature/07_w8a8_int8_recovery.md) -- owns the W8A8 accuracy-recovery survey (SmoothQuant, down_proj, rotation-skip, DeltaNet, citations).
- [`docs/kernel/02_int8_w8a8_status.md`](docs/kernel/02_int8_w8a8_status.md) -- owns the kernel status + how-to.
- [`docs/quant_methods.md`](docs/quant_methods.md) -- owns the quant-method **registry** (algorithm x scheme x model matrix; W4A8/W4A4 rotation method picks; the XPU kernel gate). The "which algorithm + what have we tried" tables live there, not here.
- [`evals/results/SUMMARY.md`](evals/results/SUMMARY.md) -- owns the measured leaderboard.
- [`w4a8/README.md`](w4a8/README.md) -- **another agent owns the W4A8-INT8 + AutoRound-W4A8 branch. Do not edit `w4a8/`.**

This doc owns: the **W8A8 kernel sprint**, **W8A8 accuracy beyond what's done**, **AutoRound (autoint)**,
**Quark loader compatibility**, and the **fused-MoE kernel** track. Everything else is a pointer.

---

## 0. The settled strategy (one screen)

Format roles -- decided, not re-litigated here:

```
W8A8  INT8  -> PRIMARY. Only 14B-class path that lights the B70 INT8 systolic (XMX DPAS) datapath.
                Prefill champion (1.6x FP8); decode ~matches FP8 after PIECEWISE; code quality = FP8 (GPTQ).
FP8         -> CONTROL / interactive fallback. The "does it still feel pristine?" baseline + best single-stream decode.
                Note: FP8 on Xe2 is EMULATED (upconverts to bf16) -- memory play, NOT a compute fast path.
W4A16 INT4  -> CAPACITY fallback. "Run the bigger model today" / VRAM-tight. No int systolic.
W8A16 INT8  -> QUALITY REFERENCE only (near-lossless) -- but NO XPU kernel (XPUwNa16 is int4-only). Not a speed path.
W4A8  INT4  -> OTHER AGENT'S BRANCH (w4a8/). Memory-emergency format; dominated on decode+accuracy today. Don't touch.
35B-A3B MoE -> needs a fused packed-expert kernel before it fits (Track 5). Separate frontier.
```

Rough effort split (info-dump's recommendation, **reconciled** below): 60% W8A8 kernels / 25% W8A8 accuracy /
10% MTP-plumbing / 5% W4A16+rotation side-quests.

> [!] **Reconciliation with MTP_TODO.md.** The info-dump pre-dates the MTP reframe. `MTP_TODO.md` (06-20) shows
> MTP is a ~3-4x multiplier vs ~1.2-1.5x for format choice, so MTP is NOT a 10% side-task -- it is the top decode
> prize and has its own plan. Treat **MTP_TODO.md as authoritative for the MTP allocation.** This doc keeps MTP as
> a pointer only (Track 6) and concentrates on the kernel + accuracy + packaging work that MTP stacks on top of.

---

## 1. Dedup ledger -- where each info-dump item already lives (so we don't re-run it)

| Info-dump item | Status | Lives in |
|---|---|---|
| "Focus W8A8, FP8 control, W4A16 capacity fallback" | SETTLED | this doc S0 + FINDINGS.md |
| "Rerun 14B W8A8 SmoothQuant+GPTQ Tier-1 (RTN underestimates it)" | **DONE 06-20** | JOURNAL.md -- see S1.1 below |
| "GPTQ@128 W4A16 HumanEval+" | **DONE 06-20** | SUMMARY.md (0.872/0.848) |
| SmoothQuant alpha sweep / per-layer groups | QUEUED, specced | doc 07 S3.1 + MTP_TODO Playbook B |
| Mixed-precision exceptions (lm_head, down_proj, first/last) | QUEUED, specced | doc 07 S3.3 + MTP_TODO Playbook B |
| Activation quant granularity (per-token vs per-tensor/static) | QUEUED, specced | doc 07 S3 (Tier 2: PrefixQuant/TWEO) |
| Rotation / SpinQuant / QuaRot at W8A8 | DEFER (marginal at 8-bit + no SYCL Hadamard kernel) | doc 07 S6 |
| KV-cache -> INT8/FP8 | KNOWN lever | doc 07 S3 Tier 2 (already compose FP8 KV) |
| MTP after W8A8 stable; spec_toks sweep; keep MTP head BF16 | OWNED ELSEWHERE | MTP_TODO.md (whole doc) |
| MoE fused expert kernel for 35B-A3B | ELEVATED to a track here | Track 5 + SUMMARY "MoE-on-XPU gap" |
| W8A8 kernel work (GEMM/GEMV/PIECEWISE/quant K-loop) | OPEN -- the sprint | Track 1 + kernel doc 02 "make-it-faster" |
| Quark = checkpoint/loader ecosystem, importer test | **NEW** | Track 4 |
| AutoRound / "autoint" (Intel's quantization) | **NEW (user add)** | Track 3 |
| Rotation method picks per scheme (QServe/SpinQuant @ W4A8; QuaRot/FlatQuant @ W4A4) | **NEW (user add)** -> registry | docs/quant_methods.md + Track 8 |
| "method x model" coverage table (show GPTQ beat RTN) | **NEW (user add)** | docs/quant_methods.md Table C |

**The stale number to stop citing:** the info-dump repeats "W8A8 HumanEval+ = 0.860". That is the **RTN** checkpoint.
The **GPTQ** W8A8 is **0.890 plus / 0.921 base** (06-20) -- ties FP8 on plus, beats it on base. Use the GPTQ number.

---

## 1.1 What 06-20 already proved (close these, don't re-open)

- **GPTQ fully recovers the W8A8 coding loss.** Tier-1 HumanEval+ on GPTQ-calibrated 14B (served identically to its
  RTN twin, only calibration changed): **w8a8-gptq 0.921 / 0.890** vs w8a8-rtn 0.902/0.860 (**+3.0 plus**) and FP8
  0.915/0.890. Decode unchanged (23.5, calibration is free at inference). The info-dump's "RTN underestimates W8A8"
  prediction is **confirmed** -- W8A8 is now a genuine FP8-quality coding path at 1.6x FP8 prefill.
- **GPTQ beats RTN at W4A16 too:** 0.872/0.848 vs 0.866/0.829 (+1.9 plus, within HumanEval CI).
- Single-B70 leaderboard (HumanEval+ plus @ decode t/s): 27B-int4 0.927@7.9 > {w8a8-gptq, fp8} 0.890@23.5/32.1 >
  w8a8-rtn 0.860 > w4a16-gptq 0.848@29 > w4a16-rtn 0.829 > w4a8 0.817@16.5.

---

## Track 1 -- W8A8 kernel sprint (HIGHEST LEVERAGE, ~60%)

Goal: keep HumanEval+ at the current 0.890, push **decode ~23 -> 27-30 t/s**, preserve the prefill advantage.
The decode gap to FP8 is the only weak spot and it is **kernel-bound** (M=1 oneDNN int8 GEMM single-row path),
not accuracy-bound. All of this is GPU-touch -> **gate behind `scripts/gpu-run`** (other agent is on the card now).

- [ ] **1a. M=1 / decode GEMV fast path** for `int8_gemm_w8a8` -- the single-row path is the remaining 22.6-vs-29
      gap (kernel doc 02 "make-it-faster"). Highest single item.
- [ ] **1b. Vectorize the quant K-loop** in `dynamic_per_token_int8_quant` -- cut per-token activation-quant overhead.
- [ ] **1c. Fuse scale application** where possible (per-token act scale x per-channel weight scale into the epilogue).
- [ ] **1d. PIECEWISE -> FULL graph capture.** PIECEWISE already gave +16.7% (~23 -> 27.2 t/s, image `:int8g`).
      FULL is blocked by Intel SYCL-Graph `work_group_scratch_memory` (via flash-attn); `torch.xpu.XPUGraph` now
      exists upstream (PyTorch 2.11) but vLLM-XPU hasn't wired it. FULL is also the unblock for spec-decode/MTP.
- [ ] **1e. Layer-timing traces** -- identify any op NOT landing on the int8 fast path (dequant-to-bf16 leaks).
- [ ] **1f. Upstream (parked):** PR1 -> vllm-xpu-kernels (int8_gemm_w8a8 + s8_s8 joint dtype + fused quant);
      PR2 -> vllm (`XPUInt8ScaledMMLinearKernel` + registry + `.get()` hardening). RFC #37979 omits INT8 W8A8 for XPU.

Target: `W8A8 decode 27-30 t/s, HumanEval+ >= 0.890, prefill advantage over FP8 intact`. If hit, W8A8 becomes the
obvious 27B-on-2xB70 pick.

---

## Track 2 -- W8A8 accuracy beyond GPTQ (~25%)

The remaining gap is the int8 **activation** quant (W8A16 0.981 agree -> W8A8 0.881), not int8 weights. GPTQ already
banked +2.7 agreement (0.881 -> 0.908). The full recipe + citations live in **doc 07 S3**; **MTP_TODO Playbook B**
has the exact `scripts/49` knobs, and the method-coverage matrix is in [`docs/quant_methods.md`](docs/quant_methods.md)
Table C. Do NOT re-derive them here -- just execute, newest-result-to-JOURNAL:

- [ ] **2a. Selective SmoothQuant** on the 16 full-attn layers + MLPs, skip DeltaNet `linear_attn` (doc 07 S3.1).
      Measure agreement lift over the GPTQ-only 0.908. This is the activation-fidelity lever -- top accuracy item.
- [ ] **2b. `down_proj`-at-W8A16 carve-out** (early+late layers) via an ignore-list knob (doc 07 S3.3, the GLU/Super-Weight site).
- [ ] **2c. KL-sensitivity layer ranking** (arXiv 2604.13440, forward-only) to replace the hand-curated ignore-list (doc 07 S5).
- [ ] **2d. Add EAR / KL-divergence to the eval harness** (doc 07 S5) -- catches what top-1 agreement misses.
- [ ] **2e. SmoothQuant alpha sweep** {0.40,0.50,0.60,0.70,0.80,0.85} x layer-group (early/mid/late, attn-proj vs MLP).
      Calibration locked at **128 samples x 2048 tok, chat-domain data** unless a layer-specific test proves otherwise.

Out of scope (decided): rotation at W8A8 (doc 07 S6), QAT (overkill at 8-bit), runtime migration to OpenVINO/llm-compressor.

---

## Track 3 -- AutoRound ("autoint", Intel's quantization)  [NEW]

**What it is.** AutoRound (`github.com/intel/auto-round`, ex-"SignRound", arXiv 2309.05516) is Intel's PTQ method:
it *learns* the optimal rounding (up/down) per weight via signed-gradient descent over a few hundred calibration
steps. Same **role** as GPTQ -- a weight-error recovery method -- but optimization-based rather than
Hessian/OBQ-based. Exports to compressed-tensors / AutoGPTQ / AWQ / GGUF; runs on CPU / CUDA / **XPU**.

**Where it already won in this repo.** AutoRound int4 produced our **best single-card quality**: Qwen3.6-27B
**0.963 / 0.927** (the leaderboard #1) and the 35B-A3B int4. So at **int4 weights it already beats our GPTQ**
(GPTQ-W4A16 14B is 0.848). AutoRound's edge is largest where weight error is largest = low-bit weights.

**Where it likely does NOT help.** At **int8 weights** the field (and doc 07 S3.4) finds RTN ~= GPTQ ~= AWQ -- the
gap there is activations, not weight rounding. So expect AutoRound-W8A8 ~= GPTQ-W8A8 (0.890); confirm cheaply, don't over-invest.

Experiments (weight-lever comparison; all CPU/XPU quant, then serve behind `scripts/gpu-run`):

- [ ] **3a. AutoRound-W4A16 vs GPTQ-W4A16 (14B + 27B).** Does the int4-leader recipe beat our GPTQ fallback
      (0.848)? Highest-value AutoRound test -- W4A16 is exactly the capacity fallback (Track 7) and int4 is AutoRound's home turf.
- [ ] **3b. AutoRound-W8A8 (14B) vs GPTQ-W8A8 0.890.** Low priority -- expected ~tie (int8 weights). One run to confirm, then move on.
- [ ] **3c. AutoRound + selective-SmoothQuant interplay.** AutoRound is weight-side; SmoothQuant is activation-side
      -> orthogonal, should compose. Test AutoRound weights + SmoothQuant acts for W8A8/W4A8.
- [ ] **3d. Export-format check:** confirm AutoRound's compressed-tensors W8A8/W4A8 export loads into OUR
      `XPUInt8ScaledMMLinearKernel` / `XPUW4A8IntLinearKernel` (same check as Quark, Track 4). If it dispatches, AutoRound is a drop-in producer.
- [ ] **3e. Retest the "XPU calibration unreliable" caveat for AutoRound** specifically (the w4a8/ agent is testing
      this for W4A8 -- coordinate, don't duplicate; reuse their finding for the W8A8/W4A16 cases).

> Boundary with the other agent: **W4A8 + AutoRound is owned by `w4a8/`** (their README: AutoRound chosen for the
> accuracy win, target plus 0.817 -> >=0.84). This track covers AutoRound for **W4A16 and W8A8** only. Cross-link, don't fork.

---

## Track 4 -- Quark loader / format compatibility  [NEW]

**Framing (do not over-scope):** Quark is **not an algorithm we are missing** -- it is AMD's quant *toolkit +
export format* (its algorithms are AWQ/GPTQ/SmoothQuant/Rotation, which we already have or have ranked). vLLM loads
Quark checkpoints via `--quantization quark`. For us the only open question is **loader compatibility**: does a
Quark-exported W8A8-INT8 checkpoint dispatch into OUR oneDNN int8 kernel, or fall back to bf16? Doc 07 S6 already
decided we do NOT migrate runtimes; this is a one-shot importer test, not a pipeline change.

- [ ] **4a. Importer test.** Serve a community Quark W8A8-INT8 checkpoint and grep the dispatch:
      ```
      scripts/gpu-run vllm serve /path/to/quark-w8a8-int8 \
        --quantization quark --dtype auto --trust-remote-code --max-model-len 32768
      # then:
      grep -E "XPUInt8ScaledMMLinearKernel|Quark|fallback|dequant|fp16|bf16" serve.log
      ```
      - Dispatches to our kernel -> Quark is a valid **packaging route** (load community quants; reproduce the
        35B-A3B community result more faithfully; standardize exchange).
      - Falls back to bf16 -> we need a small **loader/metadata adapter** (map Quark's config to the compressed-tensors
        fields our kernel reads). Scope it then; don't pre-build.
- [ ] **4b.** Do the same dispatch check for **AutoRound's** compressed-tensors export (folds into Track 3d).

**Do NOT make Quark a dependency:** Quark *quantization* (producing checkpoints) is documented for CUDA/ROCm, not
Intel XPU. If we ever need a Quark checkpoint, quantize on RunPod-NVIDIA/CPU and **serve** on B70. B70 time is for
serving-kernel + evals, not fighting Quark's quant environment.

---

## Track 5 -- Fused packed-MoE expert kernel (35B-A3B frontier)  [ELEVATED]

The 35B-A3B int4 OOMs at weight-load (21.5 GB on disk) because **vLLM-XPU has no fused int4 MoE kernel** -> the 256
experts dequantize toward bf16 (~70 GB) -> `OUT_OF_DEVICE_MEMORY`. Dense W8A8 robustness (Tracks 1-2) does not fix
this; the MoE path needs its own kernel. This is the real unlock for Qwen3.6-35B-A3B on 2-4 B70s.

- [ ] **5a. Fused packed-expert GEMM, W8A8 experts first** -- keep experts packed through the MoE path (no per-expert dequant).
- [ ] **5b. int4 experts second** (the harder, smaller-footprint case).
- [ ] **5c. Keep router/gate high-precision** if needed for routing stability.
- [ ] **5d. Optimize top-k expert dispatch layout** (gather/scatter is the XPU pain point).

This is a Phase-after-dense item -- do not start until Track 1's dense W8A8 GEMM/GEMV is robust. Captured here so the
"MoE-on-XPU gap" from SUMMARY.md has an owner.

---

## Track 6 -- MTP  [POINTER ONLY]

Owned entirely by **[`MTP_TODO.md`](MTP_TODO.md)**. Do not plan MTP here. One-line summary for context: MTP is the
~3-4x decode multiplier, it stacks orthogonally on whichever format wins, and it is gated on FULL graph capture
(Track 1d) + the image-integration work (our W8A8 kernel + `gdn_attention` + #43565 spec-wiring + `vllm_xpu_kernels`).

---

## Track 7 -- W4A16 capacity fallback  [LOW, ~5%]

Keep as the "fits on one card" path for the 27B; it is NOT the throughput story (no int systolic).

- [x] GPTQ@128 W4A16 14B HumanEval+ -- DONE (0.872/0.848, 06-20).
- [ ] **7a. Fix the 27B W4A16 serving wall:** `XPUwNa16` needs input dims divisible by 32; the 27B gated-attention
      4304 dim breaks it. Options: 32-pad / ignore-list the layer / kernel fix. Verify dims serve before counting on this scheme.
- [ ] **7b. AutoRound-W4A16 vs GPTQ-W4A16 on the 27B** (folds into Track 3a) -- pick the capacity fallback on measured code, not gsm8k.

---

## Track 8 -- Rotation + sub-8-bit frontier (W4A8 / W4A4)  [DEFERRED]

Method picks + kernel-gating are owned by **[`docs/quant_methods.md`](docs/quant_methods.md)** (Tables A/D + the
W4A4 section). Summary so the order is clear:

- **W8A8: skip rotation** -- marginal at 8-bit + no SYCL/XMX Hadamard kernel (doc 07 S6).
- **W4A8: rotation enters** -- **QServe-QoQ or SpinQuant** to rescue int8 acts at int4 weights. This is the `w4a8/`
  agent's branch; SpinQuant R1/R2 fuse offline (servable on our `int4_gemm_w4a8`), R3/R4 need an online Hadamard kernel.
- **W4A4 (true int4 acts): the real frontier** -- **FlatQuant** (SOTA acc, has Qwen2.5 model_tools) first, **QuaRot**
  (parameter-free Hadamard = cleaner int4xint4 kernel target) fallback; **PrefixQuant** for static acts. Doubly
  kernel-gated: needs a new `s4 x s4 -> s32` GEMM **and** a transform kernel (FWHT or fused Kronecker-affine).

Do not start the rotation kernels until dense W8A8 (Track 1) is robust. **First step when W4A4 opens:** diff Qwen3 vs
Qwen2.5 in FlatQuant `model_tools` to size the port (Qwen3's QK-norm shifts the rotation insertion points). Validate
in fake-quant (perplexity) before writing any XMX kernel.

---

## Track 9 -- Port `benchmark_moe.py` MoE-config tuner to XPU  [TODO, deferred from 2026-06-22 perf chase]

**Why:** the int8 MoE 35B serve (`rdy_to_serve/qwen36-35b-a3b-quark-w8a8-int8/`) logs "Using default MoE config.
Performance might be sub-optimal!" (no `E=256,N=256,device_name=Intel(R)_Graphics_[0xe223].json`). A tuned Triton
fused-MoE config is a real lever -- in CAPTURED mode the MoE GEMM is a meaningful slice, so a good config could add
~10-30% on top of the 41 t/s single-stream capture. (See FINDINGS "GRAPH CAPTURE perf" + JOURNAL/commits 1cc7900.)

**The blocker (why it's deferred, not done):** `vllm/benchmarks/kernels/benchmark_moe.py --tune` does not run on XPU:
1. `ray.init()` -> `ray.available_resources()["GPU"]` KeyError (Ray doesn't see XPU as a "GPU"). **Bypass already
   found + proven:** a `sitecustomize.py` that forces `ray.init(num_gpus=1)` (synced to host `patches/sitecustomize.py`).
   That gets past Ray (worker spawns, enumerates ~1920 configs).
2. Then the benchmark worker is CUDA-centric and dies with `AssertionError: Torch not compiled with CUDA enabled`:
   `device="cuda"` (line ~260) + CUDA-graph timing `torch.cuda.CUDAGraph()` / `torch.cuda.graph()` (lines ~307-308).
   `torch.accelerator.synchronize()` is already device-agnostic (fine); only the device string + the graph-timer need porting.

**The port (a later session):** bind-mount a patched `benchmark_moe.py` that (a) `device="cuda"` -> `"xpu"`, (b) replaces
the CUDA-graph capture/replay timing block with a plain timed loop: warmup N iters, `torch.accelerator.synchronize()`,
time `for _ in range(iters): run_kernel()`, `synchronize()`, divide -- less precise than graph-replay but correct on XPU.
Then run `--dtype auto --tp-size 2 --batch-size 1 2 4 8` (auto -> the no-dtype-suffix filename int8 reads; ~1.5 h/batch,
so consider `--batch-size 1` first). Note it tunes the **fp16 shape as a proxy** for int8 (tuner has no `int8_w8a8` dtype);
the tiling mostly transfers. Alternative if the port is painful: hand-write the config from a known-good XPU MoE tiling.
Deploy: drop the resulting JSON into the image's `model_executor/layers/fused_moe/configs/` (bind-mount) and re-serve.

---

## Execution order (the 3-5 items to actually run, deduped)

1. **W8A8 kernel sprint** (Track 1) -- M=1 decode GEMV, quant K-loop, PIECEWISE->FULL. Behind `scripts/gpu-run`.
2. **W8A8 accuracy sprint** (Track 2) -- selective SmoothQuant + down_proj carve-out + KL ranking; HumanEval+ every run.
3. **AutoRound compare** (Track 3) -- AutoRound-W4A16 vs GPTQ-W4A16 (3a) first; the int8 confirm (3b) is one run.
4. **Quark + AutoRound importer test** (Track 4) -- one serve + grep; decides if they're drop-in producers.
5. **MoE fused-expert kernel** (Track 5) -- only after dense W8A8 (Track 1) is robust.
6. **Port the MoE-config tuner to XPU** (Track 9) -- deferred; Ray bypass proven, CUDA-graph timing still needs porting.

MTP runs on its own plan (MTP_TODO.md). W4A8 runs on the other agent's plan (`w4a8/`). Rotation stays parked.

**GPU discipline:** every serve/bench/perf_probe/on-GPU-quant in these tracks goes through `scripts/gpu-run`
(one B70, the W4A8 agent is currently on it). Editing/compiling stay parallel; only the GPU touch is serialized.
