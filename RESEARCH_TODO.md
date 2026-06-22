# RESEARCH_TODO.md -- W8A8-INT8 is the main path; next research tracks

**Created:** 2026-06-20 - **Status-synced:** 2026-06-22 (two-agent codebase+git audit; see banner)
**Status:** PLAN -- consolidates a strategy info-dump (deduped) + adds AutoRound (autoint) + Quark.

> ### [STATUS SYNC 2026-06-22] -- audited every track vs code/git/JOURNAL. Net picture:
> - **Track 4a DONE** -- Quark W8A8-INT8 35B-A3B SERVES on v0230 TP=2 (commit dc740cc). Quark is a valid producer; dispatches fine.
> - **Track 5 REFRAMED (serving SOLVED, not a build blocker)** -- the int8 MoE serves TODAY via vLLM 0.23 Triton `fused_moe` (256 int8 experts) on v0230; building our own fused-expert kernel is now a *research/port* goal, low priority. AND commit 15918cc proved a TRUE int8 LINEAR kernel is **NO speed win on the MoE** (experts already int8; linear is the minority path) -- correctness+memory only. So Track 1's "int8 GEMM = decode speedup" premise is NEGATIVE for MoE.
> - **Track 2a DONE** -- selective SmoothQuant shipped in `scripts/49` (QUANTS Q0); accuracy delta still pending (Q3/Q5 gsm8k TBD).
> - **Track 1d PARTIAL** -- PIECEWISE capture DONE (+16.7%, commit 910182c); **FULL blocked** on stock v0230 (SYCL-Graph `work_group_scratch_memory`, commit a5645b2). FULL/`TRITON_ATTN` is the open MTP-positive lever.
> - **Track 1a/1b/1c/1e/1f, 2b-2e, 3c/3e NOT-STARTED**; **3a/3b/3d, 4b, 7b PARTIAL** (14B W8A8-AutoRound validated, accuracy numbers pending); **7a SIDESTEPPED** (27B int4-AutoRound serves, dodges the 4304-dim XPUwNa16 wall); **Track 8 correctly DEFERRED**; **Track 9 DEFERRED** (Ray bypass proven, CUDA-graph timing port pending).
> - **Section 1.1 numbers VERIFIED** vs SUMMARY.md (w8a8-gptq 0.921/0.890, w4a16-gptq 0.872/0.848) -- no change.
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
      gap (kernel doc 02 "make-it-faster"). NOT-STARTED. **Re-prioritized DOWN:** W8A8 decode is BW-bound at large-N
      and overhead-bound at small-N; and commit 15918cc showed int8 LINEAR gives no MoE speedup. Only a dense-W8A8 lever.
- [ ] **1b. Vectorize the quant K-loop** in `dynamic_per_token_int8_quant` -- cut per-token activation-quant overhead.
- [ ] **1c. Fuse scale application** where possible (per-token act scale x per-channel weight scale into the epilogue).
- [~] **1d. PIECEWISE -> FULL graph capture. RESOLVED 2026-06-22: PIECEWISE is the ceiling; FULL is KERNEL-gated.**
      PIECEWISE DONE -- and the warmup-spoof fix made PIECEWISE+MTP **+1.79x** (NOT -19%; MTP_TODO M0-M5). The FULL-capture
      lever was chased to ground: ported vllm-ascend #7148's dispatcher fix (scripts/88) AND tried `--attention-backend
      TRITON_ATTN` + `FULL_DECODE_ONLY` + spec-aligned caps. Result: the dispatcher patch works (capture proceeds), but it
      then crashes in the BAKED XPU KERNEL -- `torch.ops._xpu_C.gdn_attention -> RuntimeError: spec_query_start_loc must
      have size [num_spec_decodes + 1]`. TRITON_ATTN does NOT dodge it (the GDN *decode* core always routes through the
      baked `gdn_attention_core_xpu` op). So FULL-capture MTP needs an Intel `vllm_xpu_kernels` fix -- not a vLLM-Python
      fix. Issue draft: docs/kernel/21. The old "work_group_scratch_memory / flash-attn" framing is superseded -- the
      real wall is the kernel's spec-metadata shape check. **Lever CLOSED on stock v0230; PIECEWISE 1.79x stands.**
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

- [x] **2a. Selective SmoothQuant** -- SHIPPED in `scripts/49` (`SMOOTHQUANT=selective`, QUANTS Q0; builds per-layer
      Playbook-B maps: 16 full-attn + 64 MLP, skip DeltaNet `linear_attn`). Used in Q3/Q5 27B/Qwable W4A8. **Open: the
      accuracy lift measurement** -- run agreement/gsm8k for the selective-SQ quants vs GPTQ-only 0.908 (Q3/Q5 gsm8k TBD).
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

- [x] **4a. Importer test -- DONE 2026-06-22 (commit dc740cc/76).** Qwen3.6-35B-A3B Quark W8A8-INT8 SERVES on v0230
      TP=2 (load 17.54 GiB/card, coherent gen). Dispatch: 256 experts -> Triton `fused_moe` (TRUE int8); int8 LINEAR
      layers (`linear_attn.*`, `mlp.shared_expert.*`) -> a one-file `quark.py` dequant-to-bf16 graft (XPU has no int8
      scaled-mm linear kernel). **Verdict: Quark is a valid drop-in PRODUCER for serving on B70.** Original recipe below.
- [ ] **(orig 4a importer recipe, kept for reference):** Serve a community Quark W8A8-INT8 checkpoint and grep the dispatch:
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

## Track 5 -- Fused packed-MoE expert kernel (35B-A3B frontier)  [REFRAMED 2026-06-22: SERVING SOLVED]

> **The "MoE-on-XPU gap" is CLOSED for SERVING.** As of 2026-06-22 the 35B-A3B int8 MoE SERVES on `vllm-xpu-env:v0230`
> (vLLM 0.23.0): the 256 int8 experts route through the native Triton `fused_moe_kernel` on XPU (same path our int4 35B
> MoE uses), via `QuarkW8A8Int8MoEMethod`. Both Quark W8A8 (commit dc740cc) and int4-AutoRound (56.8 t/s captured) work.
> So a hand-written packed-expert kernel is **no longer a serving blocker** -- it's a *research/port* goal (study/port the
> int8-MoE GEMM into `contrib/vllm_int8_xpu` for ownership + perf tuning). **AND** commit 15918cc measured that swapping
> the minority int8 LINEAR layers to a true XMX int8 kernel gave NO MoE speedup (experts already int8) -- so the only real
> MoE perf levers are (a) FULL/PIECEWISE graph capture and (b) a tuned `E=256,N=256` Triton MoE config (RESEARCH_TODO Track 9).

The original (now-stale) framing: the 35B-A3B int4 OOMs at weight-load because vLLM-XPU lacked a fused int4 MoE kernel ->
256 experts dequantize toward bf16 -> OOM. **That premise is obsolete** -- the Triton fused-MoE path handles both int4 and int8.

- [ ] **5a. (RESEARCH, low pri) Port the int8 fused-expert GEMM to `contrib/vllm_int8_xpu`** -- ownership + a tuning surface; NOT needed to serve.
- [ ] **5b. int4 experts** -- already served via Triton fused_moe (int4-AutoRound 56.8 t/s); port only for ownership.
- [x] **5c. Router/gate high-precision** -- already done (router/gate kept bf16 in the ignore-list; quants serve coherently).
- [ ] **5d. Optimize top-k expert dispatch layout** -- fold into Track 9 (tuned Triton MoE config), not a from-scratch kernel.

---

## Track 6 -- MTP  [POINTER ONLY]

Owned entirely by **[`MTP_TODO.md`](MTP_TODO.md)**. Do not plan MTP here. One-line summary for context: MTP is the
~3-4x decode multiplier, it stacks orthogonally on whichever format wins, and it is gated on FULL graph capture
(Track 1d) + the image-integration work (our W8A8 kernel + `gdn_attention` + #43565 spec-wiring + `vllm_xpu_kernels`).

---

## Track 7 -- W4A16 capacity fallback  [LOW, ~5%]

Keep as the "fits on one card" path for the 27B; it is NOT the throughput story (no int systolic).

- [x] GPTQ@128 W4A16 14B HumanEval+ -- DONE (0.872/0.848, 06-20).
- [~] **7a. 27B W4A16 serving wall -- SIDESTEPPED, not fixed.** `XPUwNa16` needs input dims /32; the 27B gated-attn 4304
      dim breaks the *compressed-tensors* W4A16. **Operationally dodged:** the Lorbus int4-AutoRound 27B serves via
      `quantization=inc` (30.8 t/s captured, the daily driver), avoiding the broken path. The compressed-tensors-W4A16
      kernel fix (32-pad / ignore-list / kernel) is still open but **low value** since AutoRound int4 is the better serve anyway.
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

1. **MTP FULL-capture frontier** (Track 1d / MTP_TODO) -- the highest decode lever now. PIECEWISE+MTP = -19% (attn+GDN
   stay eager during verify); the unlock is `--attention-backend TRITON_ATTN` -> FULL capture so the verify pass is captured.
   This is MTP_TODO M1; everything else is second-order until it lands.
2. **W8A8 accuracy sprint** (Track 2) -- 2a selective-SQ is SHIPPED; what's open is the *measurement* (gsm8k/agreement for
   Q3/Q5 vs GPTQ-only). down_proj carve-out + KL ranking are lower-pri follow-ons. HumanEval+ every run.
3. **AutoRound accuracy confirms** (Track 3) -- 14B W8A8-AutoRound is validated/serving; run its HumanEval+ vs GPTQ (3b, one run);
   AutoRound-W4A16-vs-GPTQ on 27B (3a/7b) only if the AutoRound 0.927 isn't already decisive.
4. ~~Quark + AutoRound importer test (Track 4)~~ -- **DONE** (4a Quark serves; 4b AutoRound dispatch validated on 14B).
5. **MoE: tuned Triton `E=256,N=256` config + graph capture** (Track 9 + capture), NOT a from-scratch kernel (Track 5 serving is solved).
6. **Port the MoE-config tuner to XPU** (Track 9) -- deferred; Ray bypass proven, CUDA-graph timing still needs porting.

MTP runs on its own plan (MTP_TODO.md) -- but item 1 above IS the current MTP bottleneck, so it leads here too.
W4A8 runs on the other agent's plan (`w4a8/`). Rotation stays parked.

**GPU discipline:** every serve/bench/perf_probe/on-GPU-quant in these tracks goes through `scripts/gpu-run`
(one B70, the W4A8 agent is currently on it). Editing/compiling stay parallel; only the GPU touch is serialized.

---

## POST-Q8 FRONTIER STATUS (2026-06-22) -- the explicit mandate is DONE; remaining items ranked by value

The explicit queue is complete: MTP M0-M5 (1.79x shipped), QUANTS Q0-Q5+Q8 (queue closed; Q8 Qwable int4 validated
29.13 t/s), docs hygiene, frontier research. Headline post-Q8 result: **FULL-capture MTP is KERNEL-gated** (Track 1d
CLOSED -- ported #7148, bisected the crash to `_xpu_C.gdn_attention`; PIECEWISE 1.79x is the ceiling; issue draft
docs/kernel/21). Remaining open frontier items, **honestly ranked** (all are lower-value than what's done):

1. **[MED] Accuracy evals (Track 2 + 3 + the Q8 close).** The genuinely-open *measurements*: (a) Q8 Qwable int4
   HumanEval+ (it's a CODER model -> the right eval; closes the Q8 validation's "accuracy TBD"); (b) Track 2 -- does
   selective-SmoothQuant (Q3/Q5) actually beat GPTQ-only 0.908 on gsm8k/agreement? (c) Track 3 -- AutoRound-W4A16 vs
   GPTQ confirm. These need the evals/ harness + a GPU serve (~30-60 min each). **Highest-value remaining work.**
2. **[LOW-MED] Mount vllm_xpu_kernels 0.1.10 (v0230 = 0.1.9, confirmed).** Will NOT fix FULL (0.1.10 has no spec-path
   changes -- research + the kernel bisection above). Only a MoE-prefill speedup (#378/#379) + a >=32K NaN-race fix
   (#411). Requires a FULL-dir `.so` overlay (matched set incl. a ~1.5GB libattn_kernels), with ABI risk (0.1.10 may
   want compute-runtime 26.18). Reversible overlay. Narrow win; do only if long-context NaN or MoE prefill bites.
3. **[LOW] Q2/Q4 W8A8-AutoRound re-run** (now unblocked by the MLLM-dodge + scripts/87 config repair, scheme=W8A8).
   Expected ~TIE with GPTQ-W8A8 0.890 (int8 weights -> weight-rounding method barely matters; Track 3). One run to confirm.
4. **[LOW] Track 9** MoE-config tuner XPU port (deferred; Ray bypass proven, CUDA-graph timing still needs porting).
