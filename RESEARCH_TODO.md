# RESEARCH_TODO.md -- compressed-tensors-first quant research

**Created:** 2026-06-20 - **Status-synced:** 2026-06-23 (compressed-tensors-first focus update; see banner)
**Status:** PLAN -- consolidates a strategy info-dump (deduped) + adds AutoRound (autoint) + Quark.

> ### [FOCUS UPDATE 2026-06-23] -- research format policy
> - **Use compressed-tensors for research artifacts across schemes and models.** W8A8, W4A8, W4A16, TP=2, PP=2,
>   and custom B70 kernels should share one comparable packaging/metadata path wherever possible.
> - **Keep AutoRound/INC int4 as the proven W4A16 serve baseline**, not the default research format. The 27B
>   compressed-tensors W4A16 `/32` shape wall is deferred to a focused padding/ignore-list/kernel session.
> - **GPTQ is the current default producer for compressed-tensors quants.** It slightly beat AutoRound on 14B W8A8
>   HumanEval+; treat that as a working prior, not a final claim. Re-check on harder evals, especially before
>   drawing W4A8 conclusions.
> - **W4A4 remains later frontier research.** Keep method notes alive, but do not start W4A4 kernels until W8A8/W4A8
>   are robust.

> ### [STATUS SYNC 2026-06-22] -- audited every track vs code/git/JOURNAL. Net picture:
> - **Track 4a DONE** -- Quark W8A8-INT8 35B-A3B SERVES on v0230 TP=2 (commit dc740cc). Quark is a valid producer; dispatches fine.
> - **Track 5 REFRAMED (serving SOLVED, not a build blocker)** -- the int8 MoE serves TODAY via vLLM 0.23 Triton `fused_moe` (256 int8 experts) on v0230; building our own fused-expert kernel is now a *research/port* goal, low priority. AND commit 15918cc proved a TRUE int8 LINEAR kernel is **NO speed win on the MoE** (experts already int8; linear is the minority path) -- correctness+memory only. So Track 1's "int8 GEMM = decode speedup" premise is NEGATIVE for MoE.
> - **Track 2a DONE** -- selective SmoothQuant shipped in `scripts/49` (QUANTS Q0); accuracy delta still pending (Q3/Q5 gsm8k TBD).
> - **Track 1d PARTIAL** -- PIECEWISE capture DONE (+16.7%, commit 910182c); **FULL blocked** on stock v0230 (SYCL-Graph `work_group_scratch_memory`, commit a5645b2). FULL/`TRITON_ATTN` is the open MTP-positive lever.
> - **Track 1a/1b/1c/1e/1f, 2b-2e, 3c/3d/3f NOT-STARTED**; **3a, 4b, 7b PARTIAL**; **3b DONE**
>   (GPTQ slightly beat AutoRound on 14B W8A8); **3e DONE** (XPU AutoRound int4 calib is unsafe); **7a SIDESTEPPED**
>   (AutoRound/INC serves while compressed-tensors W4A16 waits on the 4304-dim XPUwNa16 wall); **Track 8 W4A8 NEXT /
>   W4A4 DEFERRED**; **Track 9 DEFERRED** (Ray bypass proven, CUDA-graph timing port pending).
> - **Section 1.1 numbers VERIFIED** vs SUMMARY.md (w8a8-gptq 0.921/0.890, w4a16-gptq 0.872/0.848) -- no change.
**Authoritative siblings (do NOT duplicate their content here):**
- [`MTP_TODO.md`](MTP_TODO.md) -- owns ALL MTP / speculative-decoding planning (the ~3-4x decode lever).
- [`docs/literature/07_w8a8_int8_recovery.md`](docs/literature/07_w8a8_int8_recovery.md) -- owns the W8A8 accuracy-recovery survey (SmoothQuant, down_proj, rotation-skip, DeltaNet, citations).
- [`docs/kernel/02_int8_w8a8_status.md`](docs/kernel/02_int8_w8a8_status.md) -- owns the kernel status + how-to.
- [`docs/quant_methods.md`](docs/quant_methods.md) -- owns the quant-method **registry** (algorithm x scheme x model matrix; W4A8/W4A4 rotation method picks; the XPU kernel gate). The "which algorithm + what have we tried" tables live there, not here.
- [`evals/results/SUMMARY.md`](evals/results/SUMMARY.md) -- owns the measured leaderboard.
- [`docs/intel_support_per_backend.md`](docs/intel_support_per_backend.md) -- **NEW 2026-06-30:** owns the
  per-backend Intel-Arc-B70 support comparison (vllm/sglang/llamacpp/zml): stock support, what's patched,
  quant->config mapping, TP/DP, and qwen3.6 arch-support. llamacpp (SYCL/GGML, weight-only GGUF) + zml
  (oneAPI PJRT, bf16) are the two NEW backend roots; their bring-up lives in `llamacpp/` and `zml/`
  (+ `docs/patch_applicability_matrix.md` for which of our patches transfer). GPU serve tests pending idle.
- [`w4a8/README.md`](w4a8/README.md) -- the W4A8-INT8 + AutoRound-W4A8 branch. **W4A8 is our next targeted research path.**
- [`docs/literature/11_int4_fp4_landscape_w4a8_roadmap.md`](docs/literature/11_int4_fp4_landscape_w4a8_roadmap.md)
  -- INT4/FP4 format landscape (NVFP4/MXFP4/ROCmFP4/Intel ceiling) + W4A8 SOTA (QServe/QoQ W4A8KV4, QQQ) +
  W4A4 SOTA + the recommended W4A8/W4A4 next steps (re-quant the too-big 27B W4A8; nail a GPTQ/SmoothQuant recipe).
- [`research_moe_optimizations.md`](research_moe_optimizations.md) -- MoE multi-GPU optimization IDEAS (PP-for-MoE
  hypothesis: PP keeps each layer's experts whole on one card -> removes the cross-card routing all-to-all; plus
  push-based reduce_scatter/all_gather as a fallback). Scratch/idea doc, nothing measured yet (P2P_GPU.md J.18-J.21).
- [`docs/ornith_1.0_35b.md`](docs/ornith_1.0_35b.md) -- **NEW 2026-06-25:** Ornith-1.0-35B (qwen3_5_moe coder,
  claims it beats Qwen3.5-35B on SWE-bench/Terminal-Bench). Arch notes + maker's suggested serve params (+ our
  TP=2 adaptation) to test later. Downloading via `scripts/121`; quant plan = QUANTS_TODO "Ornith" block (O1-O3).

This doc owns: the **W8A8 kernel sprint**, **W8A8 accuracy beyond what's done**, **AutoRound (autoint)**,
**Quark loader compatibility**, and the **fused-MoE kernel** track. Everything else is a pointer.

---

## 0. The settled strategy (one screen)

Format roles -- current working policy:

```
W8A8  INT8  -> PRIMARY compressed-tensors path. Lights the B70 INT8 systolic (XMX DPAS) datapath.
                14B single-card baseline works; 27B W8A8 serves via TP=2. Default producer: GPTQ.
FP8         -> CONTROL / interactive fallback. The "does it still feel pristine?" baseline + best single-stream decode.
                Note: FP8 on Xe2 is EMULATED (upconverts to bf16) -- memory play, NOT a compute fast path.
W4A16 INT4  -> CAPACITY fallback. AutoRound/INC is the proven serve baseline; compressed-tensors W4A16 is the
                research parity target after we fix the 27B `/32` shape wall.
W8A16 INT8  -> QUALITY REFERENCE only (near-lossless) -- but NO XPU kernel (XPUwNa16 is int4-only). Not a speed path.
W4A8  INT4  -> NEXT compressed-tensors research path. Needs int4-weight x int8-act kernels/GEMV work and harder evals.
35B-A3B MoE -> serving works; tuned MoE config / packed-expert ownership is separate frontier work.
W4A4  INT4  -> LATER frontier. Interesting, but after W8A8/W4A8 are stable.
```

Effort priority: W8A8 kernels first, W8A8 accuracy next, then W4A8/W4A16 compressed-tensors parity and TP/PP
plumbing. W4A4 stays note-taking only for now. MTP remains owned by `MTP_TODO.md`.

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

## Track 3 -- GPTQ vs AutoRound producer checks  [COMPRESSED-TENSORS OUTPUT]

**What it is.** AutoRound (`github.com/intel/auto-round`, ex-"SignRound", arXiv 2309.05516) is Intel's PTQ method:
it *learns* the optimal rounding (up/down) per weight via signed-gradient descent over a few hundred calibration
steps. Same **role** as GPTQ -- a weight-error recovery method -- but optimization-based rather than
Hessian/OBQ-based. Exports to compressed-tensors / AutoGPTQ / AWQ / GGUF; runs on CPU / CUDA / **XPU**.

**Current policy.** Research outputs should be compressed-tensors when the exporter supports it. AutoRound/INC int4
remains the proven W4A16 serving baseline, but it is a different loader/kernel lineage from the compressed-tensors
artifacts we want for cross-model kernel work.

**Where AutoRound already won operationally.** AutoRound int4 produced our **best single-card serving baseline**:
Qwen3.6-27B **0.963 / 0.927** (the leaderboard #1) and the 35B-A3B int4. That result is important, but it is
confounded by model/format/kernel differences and should not be read as "AutoRound always beats GPTQ."

**Where it likely does NOT help.** At **int8 weights** the field (and doc 07 S3.4) finds RTN ~= GPTQ ~= AWQ -- the
gap there is activations, not weight rounding. So expect AutoRound-W8A8 ~= GPTQ-W8A8 (0.890); confirm cheaply, don't over-invest.

Experiments (weight-lever comparison; all CPU/XPU quant, then serve behind `scripts/gpu-run`):

- [ ] **3a. Same-format W4A16 producer check (compressed-tensors).** Compare GPTQ vs AutoRound-exported
      compressed-tensors W4A16 where export and serving are clean. Do not compare AutoRound/INC against
      compressed-tensors and call it a pure quantizer result.
- [x] **3b. AutoRound-W8A8 (14B) vs GPTQ-W8A8 -- DONE 2026-06-23. GPTQ slightly WINS; AutoRound does NOT supersede.**
      HumanEval+ (164, sandboxed, true W8A8 via :int8): **w8a8-autoround 0.909/0.872 vs w8a8-gptq 0.921/0.890** -> GPTQ
      +1.2 base / +1.8 plus (near CI, but consistent direction; matches the field: weight-rounding barely matters at int8,
      Hessian-OBQ GPTQ edges optimization-AutoRound). AutoRound W8A8 is SOUND + coherent (int8 weights survive even XPU
      calib, unlike the int4 Q8). So "autoround supersedes gptq at W8A8" is REFUTED -- they ~tie, GPTQ marginally ahead.
- [ ] **3c. Harder eval confirmation.** HumanEval+ says GPTQ slightly wins at 14B W8A8, but that is not enough
      to settle the producer choice. Re-check on harder/agentic code, long-context, and reasoning evals before
      making a broad GPTQ-vs-AutoRound claim.
- [ ] **3d. AutoRound + selective-SmoothQuant interplay.** AutoRound is weight-side; SmoothQuant is activation-side
      -> orthogonal, should compose. Test only when the output is compressed-tensors and dispatches to the same kernel.
- [ ] **3f. Export-format check:** confirm AutoRound's compressed-tensors W8A8/W4A8 export loads into OUR
      `XPUInt8ScaledMMLinearKernel` / `XPUW4A8IntLinearKernel` (same check as Quark, Track 4). If it dispatches, AutoRound is a drop-in producer.
- [x] **3e. "XPU calibration unreliable" for AutoRound -- CONFIRMED 2026-06-22 (Q8 repro).** Quantizing Qwable-5-27B int4
      via AutoRound with `device_map=auto` on the two B70s (gradient rounding on XPU + low_gpu_mem_usage offloading)
      produced **broken weights -> pure `!` garbage** (HumanEval+ 0.0/0.0), even though the checkpoint LOADS + serves
      cleanly and its config matches the working Lorbus EXACTLY. So the corruption is in the XPU CALIBRATION, not packaging.
      **RULE: quantize AutoRound on CPU/CUDA (RunPod-NVIDIA), serve on B70** -- same as Quark (Track 4). This also
      retro-flags any other XPU-calibrated AutoRound quant. (See QUANTS_TODO Q8 correction + JOURNAL 2026-06-22.)

> W4A8 producer policy: default to compressed-tensors GPTQ/SQ first. Re-open AutoRound for W4A8 only after exporter
> and same-kernel dispatch are verified, then judge on harder evals rather than producer reputation.

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
- [ ] **4b.** Do the same dispatch check for **AutoRound's** compressed-tensors export (folds into Track 3f).

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
- [ ] **5e. PP=2 for the MoE (multi-GPU parallelism, not a kernel)** -- hypothesis: PP keeps each layer's experts whole
      on one card, removing the cross-card routing all-to-all that TP-MoE pays; only a single activation push/microbatch
      crosses. MoE also dodges the dense-PP blockers (does not want MTP; captured-PP bug was a GDN-hybrid artifact).
      Full reasoning + test order + cautions in [`research_moe_optimizations.md`](research_moe_optimizations.md). UNTESTED.

---

## Track 6 -- MTP  [POINTER ONLY]

Owned entirely by **[`MTP_TODO.md`](MTP_TODO.md)**. Do not plan MTP here. One-line summary for context: MTP is the
~3-4x decode multiplier, it stacks orthogonally on whichever format wins, and it is gated on FULL graph capture
(Track 1d) + the image-integration work (our W8A8 kernel + `gdn_attention` + #43565 spec-wiring + `vllm_xpu_kernels`).

---

## Track 7 -- W4A16 compressed-tensors parity  [DEFERRED FIX]

Keep W4A16 as the "fits on one card" capacity path. It is not the int8 throughput story, but we still want
compressed-tensors parity eventually so research artifacts stay comparable across W8A8/W4A8/W4A16.

- [x] GPTQ@128 W4A16 14B HumanEval+ -- DONE (0.872/0.848, 06-20).
- [~] **7a. 27B W4A16 serving wall -- SIDESTEPPED, not fixed.** `XPUwNa16` needs input dims /32; the 27B gated-attn 4304
      dim breaks the *compressed-tensors* W4A16. **Operationally dodged:** the Lorbus int4-AutoRound 27B serves via
      `quantization=inc` (30.8 t/s captured, the daily driver), avoiding the broken path. Keep using that for serving.
      **Research fix later:** add padding-aware loader/kernel support or precise BF16 ignores so compressed-tensors
      W4A16 works on the 27B too.
- [ ] **7b. Same-format W4A16 eval.** Once 7a is fixed, compare GPTQ vs AutoRound-exported compressed-tensors W4A16
      on the same model and kernel. Include harder evals, not just gsm8k/HumanEval+.
- [ ] **7c. SLIM COMPRESSED-TENSORS MODELS -- chase AutoRound-size CT artifacts.** The current 27B CT W4A16 is
      much larger than the Lorbus AutoRound int4 daily driver for a concrete, measured reason: the CT quant kept
      almost the whole Gated-DeltaNet `linear_attn` stack BF16. Host tensor-byte audit, 2026-06-23:
      ```
      Lorbus_Qwen3.6-27B-int4-AutoRound: 17.7 GiB tensor bytes
        int32 packed 11.591 GiB, bf16/fp16 6.100 GiB
        linear_attn total 2.726 GiB

      Qwen3.6-27B-W4A16 compressed-tensors: 24.1 GiB tensor bytes
        int32 packed 8.750 GiB, bf16 15.371 GiB
        linear_attn total 10.360 GiB
      ```
      The delta is almost entirely GDN projection storage. Lorbus packs
      `linear_attn.in_proj_qkv`, `linear_attn.in_proj_z`, and `linear_attn.out_proj` as int4
      (`qweight` + scales), while keeping only the small recurrent/projection-support pieces BF16
      (`in_proj_a`, `in_proj_b`, norm/state tensors). Our CT W4A16 left the large projections BF16:
      `in_proj_qkv` 4.688 GiB, `in_proj_z` 2.812 GiB, `out_proj` 2.812 GiB.

      **What we are chasing:** a "CT W4A16 slim" artifact whose ignore list matches the Lorbus shape:
      keep BF16 only for the small or fragile GDN pieces, MTP head, lm_head/embeddings if needed, and vision
      if the artifact is VL; quantize the large GDN projections plus normal MLP/self-attn linears. Target loaded
      footprint is **~17-18 GiB instead of 24.35 GiB**, restoring AutoRound-like KV room while keeping the
      compressed-tensors packaging path. Expected practical impact: fp16-KV context headroom should move from the
      current cramped CT W4A16 regime (~4 GiB KV at UTIL=0.95, graph capture OOM at UTIL=0.90) back toward the
      AutoRound regime (8.31 GiB KV at UTIL=0.92, ~133k max context). This may also improve performance by avoiding
      BF16 GDN projection GEMMs and by reducing memory pressure / graph-capture pressure, though the first target is
      capacity and parity.

      **Why it matters beyond W4A16:** the same "over-broad BF16 ignore list" can silently poison future CT W4A8 and
      W8A8 artifacts. W4A8/W8A8 are supposed to exercise int8-activation / int8-XMX paths; leaving the large GDN
      projections BF16 both bloats VRAM and hides the kernels we are trying to evaluate. For every future 27B CT
      quant, add a tensor-byte audit by category (`linear_attn`, MLP, self_attn, lm_head, MTP, visual) and compare to
      the Lorbus AutoRound int4 layout before trusting VRAM, KV, or perf conclusions.

      **Concrete next experiment:** produce a new CT W4A16 slim checkpoint with narrow GDN ignores, then serve it
      through the existing `rdy_to_serve/qwen36-27b-w4a16` text shim. Verify: no skipped weights, coherent gen,
      model load near 17-18 GiB, available KV near AutoRound, ctx2048 perf vs current CT W4A16 and Lorbus AutoRound,
      then repeat MTP graft acceptance/perf only after the non-MTP slim artifact is stable.

---

## Track 8 -- W4A8 then W4A4 frontier  [W4A8 NEXT, W4A4 LATER]

Method picks + kernel-gating are owned by **[`docs/quant_methods.md`](docs/quant_methods.md)** (Tables A/D + the
W4A4 section). Summary so the order is clear:

- **W8A8: skip rotation** -- marginal at 8-bit + no SYCL/XMX Hadamard kernel (doc 07 S6).
- **W4A8: next compressed-tensors research target** -- start with GPTQ/SmoothQuant and same-kernel evals. Then test
  whether AutoRound or rotation methods actually help on harder evals. SpinQuant R1/R2 can fuse offline; R3/R4 need
  an online Hadamard kernel.
- **W4A4 (true int4 acts): the real frontier** -- **FlatQuant** (SOTA acc, has Qwen2.5 model_tools) first, **QuaRot**
  (parameter-free Hadamard = cleaner int4xint4 kernel target) fallback; **PrefixQuant** for static acts. Doubly
  kernel-gated: needs a new `s4 x s4 -> s32` GEMM **and** a transform kernel (FWHT or fused Kronecker-affine).

Do not start W4A4 kernels until dense W8A8 and W4A8 are robust. **First step when W4A4 opens:** diff Qwen3 vs Qwen2.5
in FlatQuant `model_tools` to size the port (Qwen3's QK-norm shifts the rotation insertion points). Validate in
fake-quant/perplexity before writing any XMX kernel.

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

## Track 10 -- XPU serving-feature ports (prefix/radix cache, fp8 KV, HiCache)  [NEW 2026-06-30]

**Why this is here:** these are not quant/kernel *scheme* work -- they are sglang serving-infra features that are
CUDA-only upstream and need an XPU port. The daily driver (`rdy_to_serve/sglang/qwen36-27b-w8a8/`) serves the HYBRID
Qwen3.6 (attention + GDN/mamba) and currently runs `--disable-radix-cache`, so EVERY turn re-prefills the full prompt
(observed 2026-06-30: a ~88K-token conversation re-prefilled ~85 s, 2400 -> 675 tok/s as KV filled). For a long
multi-turn agentic/coding daily driver this repeated prefill is the single biggest avoidable TTFT cost. Prefix caching
-- NOT NVMe offload -- is the win.

**10a -- Prefix / radix cache for the hybrid model on XPU.  [HIGHEST VALUE in this track]**
- Blocker (CONFIRMED): `--enable-radix-cache` (RADIX=1) CRASHES at arg-parse in `server_args._handle_mamba_radix_cache`.
  sglang's mamba radix path needs to checkpoint/restore the SSM recurrent state at arbitrary prefix boundaries via
  either `extra_buffer` (CUDA/MUSA/NPU-only -> AssertionError on XPU) or `no_buffer` (forces `page_size=1`, untested
  with NEXTN + our fused int8 kernels).
- Two candidate paths: (1) port the `extra_buffer` mamba state-copy to XPU (keeps page_size=64); (2) validate the
  `no_buffer` + `page_size=1` path with MTP + fused W8A8 kernels (simpler, but page_size=1 may cost throughput and
  collide with the spec mamba cache cap `--max-running-requests 4`). Path (1) preferred if the state-copy is small.
- Gate: must stay coherent under concurrent prefill+decode AND keep MTP accept-len; prove a cache HIT skips re-prefill
  on a repeated long prefix (watch `cache_hit_rate` once Track 10d metrics are live). Pure-attention models could use
  radix on XPU already; the blocker is specifically the mamba state.

**10b -- fp8 KV cache on XPU.  [MED]** Unsupported today (`xpu_backend.py:505`). Would ~halve KV memory -> bigger
context / batch / `--max-running-requests`. Compose with (not before) 10a. Research item, not a prod flag.

**10c -- Hierarchical cache / HiCache (KV -> CPU RAM -> NVMe/SSD).  [LOW, BLOCKED on 10a]** This is the "cache to
NVMe/SSD" ask. It is a layer ON TOP of radix, so it is unreachable until 10a works. Also scope expectations: HiCache
helps cross-request reuse / effective KV capacity -- it is NOT a single-stream speedup and NOT a free context extender.
Do not pursue before 10a lands.

**10d -- Observability: Prometheus metrics + Grafana.  [DONE 2026-06-30]** (1) Added the `METRICS` knob (default on
-> `--enable-metrics`) to the w8a8 shelf `serve.sh`; exposes `/metrics` on the serve port (input/output token
counters, TTFT, gen throughput, cache_hit_rate, queue depth). (2) Vendored sglang's `examples/monitoring/` stack to
`bin/monitoring/` (prometheus.yaml retargeted 30000->18080; datasource uid pinned to the dashboard's hard-coded
`ddyfngn31dg5cf`) and wired Prometheus+Grafana into `vllm/daily_driver_serve.sh` as `monitor_up`/`monitor_down`,
lifecycle-tied exactly like Open WebUI -- Grafana on :3001 (WebUI owns :3000), anon Viewer, sglang dashboard as home.
Validated: containers up, dashboard+datasource provisioned, Grafana :3001 healthy. REMAINING: the live Prometheus
target reads 404/down until the daily driver is RESTARTED to pick up `--enable-metrics`. `cache_hit_rate` will read
~0 until 10a lands.

See also: memory `xpu-serve-limits-fp8kv-and-radix`, JOURNAL 2026-06-29 (RADIX knob), and the serve.sh `RADIX=` comment.

---

## Track 11 -- NVFP4 follow-ups  [NEW 2026-07-04, after the M6-M9 champion session]

The NVFP4 27B (`rdy_to_serve/vllm/qwen36-27b-nvfp4/`) is now the box quality #1 (HumanEval+
0.988/0.945) AND fastest single-card serve (40.7-44.1 t/s, 67 on code). Open items, ranked:

- [ ] **11a. int4-on-v0.24.0 port + the same-stack A/B.** The honest NVFP4-vs-int4 comparison is
      blocked: v0230 int4 on card 1 soft-poisons ("!!!!") in BOTH captured and eager modes
      (JOURNAL 2026-07-04 NEG x2). Port the dense int4 entry to v0.24.0 (in-tree INC int4, reuse
      the MoE-port lessons: ARK gate + fusion knobs), re-eval HumanEval+ on the same stack.
- [ ] **11b. v0230-on-kernel-7.1 revalidation.** All v0230 images predate the 2026-07-02 kernel
      7.1/ICR 26.22 upgrade; the card-1 poisoning may be a v0230-on-7.1 incompatibility. Cheap
      test: the same int4 eval on CARD 0 v0230 eager; if it also poisons, v0230 rows are stale.
- [x] **11c. NVFP4 KV headroom -- SOLVED via TP=2 (2026-07-04 evening).** TP=2 MAXLEN=131072
      serves: 902,083 KV tokens (6.88x @ 128K), needle PASS at 115K, decode 21.4 t/s @ 60K depth,
      c1 24.07, gate 18/18. Root cause of low single-card KV: 24.1 GiB weights + the hybrid
      unified-page allocator (1664-token pages charged across all 64 layers = ~4x naive attn
      math); single-card ceiling ~19,968 tokens. fp4 KV researched: Blackwell-only kernels, and a
      small lever on this hybrid anyway (4 GiB per 128K seq at fp8).
      **MTP-on-TP2 DONE (2026-07-04, commit b8fc79f):** ported the capture-safe all_gather
      (all-reduce-of-padded) + splitting_ops to the nvfp4 TP>1 path; fixed an XPU vLLM bug
      (fuse_rope_kvcache_cat_mla NameError under TP>1 fusion) and a MAXSEQS>=8 NaN-garbage
      correctness floor. Result at MAXLEN=131072: 705,356 KV tokens (5.38x @128K), single-stream
      decode 52-58 t/s (was 24 no-MTP; BEATS single-card 40-44), needle PASS @88k, gate 18/18.
      This is the recommended long-context serving config. REMAINING: a soak before shelf
      promotion; push-AR overlay (perf only, NOT needed for coherence -- MAXSEQS>=8 alone is clean);
      the one transient sample_tokens RPC hang (not reproduced this session).
- [~] **11f. Official 35B MoE NVFP4 bring-up -- FEASIBILITY PROVEN (2026-07-04, commit pending).**
      nvidia/Qwen3.6-35B-A3B-NVFP4 (256 experts W4A16_NVFP4 g16 + FP8 attn/GDN + FP8 KV + bf16
      vision/router/mtp) LOADS + generates COHERENTLY on XPU via the EMULATION MoE backend
      (--moe-backend emulation, dequant-on-the-fly -> stock TritonExperts). Blocker cleared: shim
      block (5, env NVFP4_MOE_W4A16_EMUL=1) relaxes Nvfp4QuantizationEmulationTritonExperts.
      _supports_quant_scheme, which hard-gated on W4A4 only, to also accept the W4A16 (kNvfp4Static,
      None) scheme. Single-card OOMs (21.8 GiB resident + 2 GiB fp32 dequant transient/forward);
      TP=2 fits + generates but at 0.37 t/s (emulation dequants ALL 256 experts to fp32 every
      forward). serve = vllm/nvfp4/serve_nvfp4_moe_35b.sh.
      FUSED per-expert path DONE (shim block 7, NVFP4_MOE_FUSED=1, serve MODE=fused): reuses the dense
      27B nvfp4_gemm_w4a16 op per active expert (no new .so) -> 1.91 t/s = 5.2x the emulation baseline,
      coherent, lower memory. REMAINING (the real serve gate): 1.91 is launch-bound (per-expert Python
      loop + M=1 GEMV x 40 layers) -> needs (a) a grouped/batched-expert nvfp4 gemm (one kernel over all
      active experts), (b) MoE graph capture, (c) MTP-on-TP2 amortization.
- [~] **11g. NVFP4 TP=2 prefill optimization (user-requested 2026-07-04).** The TP=2 long-context DD's
      one weakness was cold prefill: PP 666 t/s vs single-card 1702 (TP=2 collective cost), TTFT 3076 ms
      @ IN=2048.
      - [x] **push-AR PREFILL overlay = SOLVED the headline gap (2026-07-04 evening, JOURNAL).** Ported
            the proven W8A8 push all-reduce onto the NVFP4 v0.24.0 TP=2 path (sitecustomize block 8 +
            serve_nvfp4_27b.sh PUSH_AR=1, MIN_NUMEL=65536 prefill-only gate). MEASURED 2.4-3.3x prefill
            (TTFT + PP, win grows with length), push-AR PP 1730-2185 now MATCHES/BEATS single-card 1702,
            decode NEUTRAL (15.87 vs 15.88 c1), gate_concurrent 18/18 on both configs, wedge-free.
            Recommend PUSH_AR=1 for the TP=2 DD. Levers still open below.
      - [x] chunked-prefill tuning (--max-num-batched-tokens) STACKS on push-AR for LONG prefills
            (2026-07-05, JOURNAL): sweet spot MAXBATCH=16384 + PUSH_AR_MAXB=256 MiB = +11.5% @ 32K cold
            (PP 1772 -> 1976). Single-chunk 32768 ties it at more memory (no gain past 2 chunks). Default
            8192 optimal for IN<=8192. GUARDRAIL: raising MAXBATCH past ~13k WITHOUT raising PUSH_AR_MAXB
            silently defeats push-AR (chunk bytes > 128 MiB scratch -> oneCCL, PP 734). Env-only knobs.
      - [x] **PUSH_AR_GRAPH=1 on NVFP4 (DECODE push) -- DONE 2026-07-05 (JOURNAL). IT HOLDS + is a strict
            faster-or-equal, but the win is SMALL.** Ported the W8A8 capturable decode push (graph .so +
            MIN_NUMEL=0, ar_allreduce_graph recorded into the XPU graph) onto NVFP4 TP=2 -- serve-script
            wiring only (shared .so + patch already graph-aware; NO rebuild). serve_nvfp4_27b.sh now honors
            PUSH_AR_GRAPH (default 0 = byte-identical prefill-only; guarded to 0 unless GRAPH=1 && CGMODE!=NONE).
            A/B (fused GRAPH=1 MTP5, 18-stream gate + bench): COHERENT 18/18 both, WEDGE-FREE. Decode delta
            = +5-6% single-stream AND agg on the LOW-accept random/chat workload (16.75 vs 15.91 c1; 30.60
            vs 28.89 agg), NEUTRAL on high-accept CODING (45.9 vs 46.8, noise -- MTP5 accept amortizes the
            per-token all-reduce away, so little AR left to push). Prefill PP unchanged. Smaller than W8A8's
            K.8 +8-10% precisely because NVFP4 MTP5 accepts more. Recommend PUSH_AR_GRAPH=1 for the mixed
            chat+coding DD; NOT yet flipped into the unattended shelf default -- DD restored to known-good
            prefill-only; gate the shelf flip on a longer unattended soak (decode-capture push = higher-wedge
            path per docs/handoff_decode_push_ar.md). One-env-var change once soaked.
      - [ ] PP=2 prefill-biased variant: structurally eliminates the per-layer all-reduce (128 AR -> 1
            P2P/stage) but adds a single-stream decode bubble -> separate entry, not a DD swap. Medium
            risk (untested PP oneCCL/L0 path, re-gate wedge).
      - DEAD on XPU (levers-agent 2026-07-04): sequence-parallel enable_sp (no volume cut + adds an
        uncapturable all_gather), AsyncTP fuse_gemm_comms / fuse_allreduce_rms (CUDA-symmetric-memory /
        Hopper+FlashInfer only), AR-coalescing (no structural lever). FRONTIER reserve: TP=1 prefiller +
        TP=1 decoder P/D-disaggregation with L0-IPC KV+GDN-state transfer (highest ceiling, highest cost).
- [x] **11d-prereq DONE: NVFP4 TP=2 daily-driver config** (256K ctx + prefix cache + MTP, commit pending):
      764k KV @ 2.92x, prefix 3.98x, gate 18/18, decode 48-50 warm. shim block (6) mamba ptr fix +
      PREFIXCACHE toggle. Remaining for full promotion: swap live DD + tool-call/reasoning parsers + API key.
- [ ] **11d. Upstream the register_fake pattern.** One `torch.library.register_fake` per custom
      _xpu_C op is the whole distance between eager-only and PIECEWISE-capturable on XPU. Fold
      into the parked PR set (Track 1f) -- applies to nvfp4_gemm_w4a16, int8 ops, w4a8 ops alike.
- [ ] **11e. Harder evals on the champion** (agentic code, long-context) before promoting it past
      "quality pick" toward daily-driver conversations (it lacks KV for that today; see 11c).
- [ ] **11h. [HIGH -- ACTIVE, unblocks MTP+graph on BOTH quants] Fix the MTP-verify-in-piecewise-cudagraph
      NEO command-stream leak.** [REFRAMED 2026-07-07 -- see docs/20260707_dd_mtp_piecewise_neo_abort.md.]
      NOT NVFP4-specific: the W8A8 DD hit the IDENTICAL abort 2026-07-07 (~3h real use; 6-way concurrent
      soak reproduces at ~96k decode tokens). ROOT CAUSE (binary disasm of libtorch_xpu):
      `at::xpu::XPUGraphImpl::replay` submits the executable SYCL command_graph via `submit_with_event` onto
      the in-order queue and NEVER synchronizes -> each replay leaves a graph-exec command + un-waited L0
      event in the NEO immediate command list, reclaimed only on a full queue sync. The MTP propose loop
      (llm_base_proposer.py:613-687) fires (spec-1) x pieces replays/step with no host sync between draft
      steps -> the command list grows until LinearStream::getSpace overflows (linear_stream.h:84). Needs BOTH
      MTP and capture. NVFP4 crashes ~50-100x SOONER (~1-2k tok vs W8A8 ~96k) because nvfp4_gemm_w4a16 encodes
      far more commands/replay -> **NVFP4 is the fast-iteration repro vehicle AND the biggest beneficiary**
      (higher 4-bit decode ceiling + top quality, currently fully blocked).
      RULED OUT (2026-07-07, GPU-tested): (a) per-step torch.xpu.synchronize (sitecustomize block 5,
      B70_XPU_CG_SYNC_STEPS) -- NO effect, sync does not reclaim the command-list growth; (b) recapture-every-N
      (block 3, B70_XPU_CG_RECYCLE_STEPS) -- crashes when it fires (clears the in-flight graph mid-step, racy
      under concurrent load); (c) event-cleanup env / buffer-enlarge -- sync-ineffective implies events aren't
      the reclaimable unit, and OverrideCmdListCmdBufferSizeInKb only multiplies the threshold.
      NEXT (on NVFP4, ~1min crash cycles): (1) DRAFTER-EAGER -- force the MTP drafter's cudagraph mode to NONE
      while keeping the TARGET decode captured (kills the sync-free drafter replay loop = the leak engine per
      the disasm; keeps most decode-capture speed); (2) FULL_DECODE_ONLY / fewer pieces (fewer replays/step);
      (3) torch-level root fix (rebuild libtorch_xpu: submit_without_event, or reset/update the command list
      per replay -- the upstream-worthy fix, novel: no public issue exists). Repro harnesses:
      vllm/nvfp4/bisect_probe.py (single-stream NVFP4 fast crash), vllm/soak_concurrent.py + vllm/fix_test.sh.
- [ ] **11i. [MED] Get fp8 KV working for NVFP4 (would restore smaller/faster KV without the repetition).**
      The ModelOpt NVFP4 checkpoint declares config.json kv_cache_scheme fp8 but ships NO calibrated k/v
      scales -> vLLM "scaling factor 1.0" -> precision loss ACCUMULATES over gen length -> repetition
      collapse (JOURNAL 2026-07-06; current fix = KV_FP8=0 -> bf16 KV, which halves KV capacity + ~2x decode
      KV BW). Path: produce calibrated k/v scales (modelopt calib pass, or compute from a calibration set and
      inject into the checkpoint) so fp8 KV is accurate -> reclaim the 757k-vs-461k KV @256K and the decode BW.

## Execution order (the 3-5 items to actually run, deduped)

1. **Compressed-tensors W8A8/W4A8 kernel path** (Tracks 1/2/8) -- keep the 14B W8A8 baseline green, then use the same
   format path for 27B TP=2/PP=2 and W4A8.
2. **W8A8 accuracy sprint** (Track 2) -- 2a selective-SQ is SHIPPED; what's open is the *measurement* (gsm8k/agreement for
   Q3/Q5 vs GPTQ-only). down_proj carve-out + KL ranking are lower-pri follow-ons. HumanEval+ every run.
3. **Harder producer evals** (Track 3) -- GPTQ is the default because it currently edges AutoRound on W8A8, but verify
   harder code/reasoning/long-context before locking the choice, especially for W4A8.
4. **27B compressed-tensors W4A16 slim fix** (Track 7c first, then 7a/7b) -- narrow the over-broad GDN BF16 ignore
   list so CT W4A16 loads near AutoRound size (~17-18 GiB instead of 24.35 GiB), restores KV/context headroom, and
   prevents the same bloat from corrupting future W4A8/W8A8 conclusions. Keep AutoRound/INC serving in the meantime.
5. **MoE: tuned Triton `E=256,N=256` config + graph capture** (Track 9 + capture), NOT a from-scratch kernel (Track 5 serving is solved).
6. **MTP work** stays in `MTP_TODO.md` and should use whichever compressed-tensors research artifact is under test.

MTP runs on its own plan (`MTP_TODO.md`). W4A8 implementation details may live under `w4a8/`, but the research
policy here is compressed-tensors first. Rotation stays parked until W4A8 needs it.

**GPU discipline:** every serve/bench/perf_probe/on-GPU-quant in these tracks goes through `scripts/gpu-run`
(dual B70 host; use `--card N` for true one-card work and the default both-card lock for TP=2/PP=2). Editing/compiling
stay parallel; only the GPU touch is serialized.

---

## POST-Q8 FRONTIER STATUS (2026-06-22) -- the explicit mandate is DONE; remaining items ranked by value

The explicit queue is complete: MTP M0-M5 (1.79x shipped), QUANTS Q0-Q5+Q8 (queue closed; Q8 Qwable int4 validated
29.13 t/s), docs hygiene, frontier research. Headline post-Q8 result: **FULL-capture MTP is KERNEL-gated** (Track 1d
CLOSED -- ported #7148, bisected the crash to `_xpu_C.gdn_attention`; PIECEWISE 1.79x is the ceiling; issue draft
docs/kernel/21). Remaining open frontier items, **honestly ranked** (all are lower-value than what's done):

0. **[HIGH for the daily driver] Prefix/radix cache on XPU hybrid (Track 10a).** The daily driver re-prefills the
   FULL prompt every turn (`--disable-radix-cache`; ~85 s for an ~88K-token conversation, observed 2026-06-30). Porting
   the mamba radix state-copy to XPU (or validating `no_buffer`+`page_size=1` with MTP) would skip that re-prefill on
   repeated long prefixes -- the biggest avoidable TTFT cost for long agentic sessions. Higher day-to-day value than the
   accuracy/MoE items below; ranked 0 because it's a serving-infra port, not part of the original quant mandate.
1. **[MED] Accuracy evals (Track 2 + 3).** (a) ~~Q8 Qwable int4 HumanEval+~~ -- **DONE 2026-06-22: scored 0.0/0.0
   (GARBAGE) -> the XPU-calibrated quant is BROKEN (Track 3e CONFIRMED); needs a CPU/CUDA re-quant. The eval harness +
   sandbox WORK (they caught the garbage the perf-bench missed).** (b) Track 2 -- does selective-SmoothQuant (Q3/Q5)
   beat GPTQ-only 0.908 on gsm8k/agreement? (c) Track 3 -- harder GPTQ-vs-AutoRound producer evals, same output
   format and same kernel only. Need evals/harness/serve.
2. **[LOW-MED] Mount vllm_xpu_kernels 0.1.10 (v0230 = 0.1.9, confirmed).** Will NOT fix FULL (0.1.10 has no spec-path
   changes -- research + the kernel bisection above). Only a MoE-prefill speedup (#378/#379) + a >=32K NaN-race fix
   (#411). Requires a FULL-dir `.so` overlay (matched set incl. a ~1.5GB libattn_kernels), with ABI risk (0.1.10 may
   want compute-runtime 26.18). Reversible overlay. Narrow win; do only if long-context NaN or MoE prefill bites.
3. **[LOW] Q2/Q4 W8A8-AutoRound re-run** (now unblocked by the MLLM-dodge + scripts/87 config repair, scheme=W8A8).
   Expected ~TIE with GPTQ-W8A8 0.890 (int8 weights -> weight-rounding method barely matters; Track 3). One run to confirm.
4. **[LOW] Track 9** MoE-config tuner XPU port (deferred; Ray bypass proven, CUDA-graph timing still needs porting).
