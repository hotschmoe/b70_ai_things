# 12 - MTP / Speculative-Decoding Plan for the B70 (path to NET-POSITIVE decode)

Goal: make speculative decoding / Multi-Token-Prediction NET-POSITIVE on the Arc Pro B70 so our
w8a8 Qwen3-14B (and the headline Qwen3.6-27B when card #2 lands) decodes faster. Today spec-decode is
net-NEGATIVE (FINDINGS.md): ngram on 14B w8a8 = eager 23.33->21.51 (-7.8%), PIECEWISE 27.23->25.28
(-7%), ~16% accept.

Synthesis of: repo docs (FINDINGS, MTP_TODO, docs/kernel/04, docs/literature/04+06) + a live grep of
the actual `vllm-xpu-env:v0230` image + codex reasoning. Verified-vs-inferred split is in section E.
ASCII only.

================================================================================
TL;DR (read this first)
================================================================================
1. The -7% is TWO independent failures, not one: (i) eager-attention verify overhead (PIECEWISE
   leaves attention eager; verify N+1 tokens pays eager-attn launch x(N+1)), AND (ii) ngram's
   acceptance is just too low (~16%). FULL capture only fixes (i). At 16% accept / N=4 the Amdahl
   ceiling is ~1.64x BEFORE overhead -- ngram on this workload cannot deliver a big win even with
   perfect capture. **Do not bet the roadmap on flipping ngram; raise acceptance.**
2. The single most promising path is the NATIVE Qwen3.6 MTP head -- and it is testable on ONE card
   NOW. The checkpoint already on the host (`Lorbus_Qwen3.6-27B-int4-AutoRound`) ships the `mtp.*`
   head and fits a single 32 GB B70 (~18 GB int4). The old `gdn_attention ... spec_sequence_masks`
   blocker is FIXED in vLLM 0.23.0 (verified: no such NotImplementedError in the v0230 image; PR
   #43565 landed). MTP is a SINGLE-PASS drafter (launch-multiplier ~1) so it survives eager/PIECEWISE
   far better than ngram, and accepts ~75-88% (vs ngram's 16%).
3. FULL capture (TRITON_ATTN or oneAPI 2026.0) is now a SECOND-ORDER lever: it would help, but
   PIECEWISE already puts decode at ~74-84% of the BW ceiling, and the Triton-XPU "0 active drivers"
   disable is a real yak-shave. Pursue MTP first; FULL capture is a later compounding win.
4. The 2nd card does NOT help spec-decode. PR #34482: XPU graph capture matrix = distributed: NO.
   TP2 => no capture => spec-decode's launch+collective overhead is unhidden. Spec-decode's only
   viable home is a SINGLE card with capture. Use TP2 for VRAM/throughput, not for MTP latency.

================================================================================
A. THE EXACT BLOCKER CHAIN (verified)
================================================================================
Why spec-decode is net-negative on B70 today, decomposed:

  [1] PIECEWISE graph capture leaves ATTENTION eager.
        - vLLM-XPU captures linear/MLP but NOT attention. flash-attn FULL capture dies on
          `RuntimeError: The sycl_ext_oneapi_work_group_scratch_memory feature is not yet available
          for use with the SYCL Graph extension.` (verified live, JOURNAL 2026-06-20).
        - So multi-token verify of N+1 positions pays full EAGER attention launch overhead x(N+1)
          every decode step. This is failure mode (i). [VERIFIED]

  [2] The alternative that WOULD allow FULL capture -- TRITON_ATTN -- cannot engage.
        - `VLLM_ATTENTION_BACKEND=TRITON_ATTN` is ignored; log shows `Using Flash Attention backend`.
        - Root cause: the image logs `Triton is installed but 0 active driver(s) found (expected 1).
          Disabling Triton`. Once HAS_TRITON=False, the Triton attention path is unavailable and vLLM
          silently falls back to flash-attn. [VERIFIED]
        - The exact gate is vLLM's own `vllm/triton_utils/importing.py`: it does
          `active_drivers = [x.driver for x in triton.backends.values()
                             if x.driver and x.driver.is_active()]`
          and in a non-distributed env sets `HAS_TRITON=False` when `len(active_drivers) != 1`.
          On the B70 the intel-xpu-backend-for-triton driver's `is_active()` returns False ->
          0 drivers -> Triton disabled. [VERIFIED - source read from the v0230 image]

  [3] The rejection sampler itself is Triton-jit based.
        - `vllm/v1/sample/rejection_sampler.py` imports `from vllm.triton_utils import tl, triton`
          and uses `@triton.jit` kernels (lines 14, 714+). So the SAME Triton disable that blocks
          TRITON_ATTN ALSO degrades the spec-decode verify/reject kernel to a fallback path. Fixing
          Triton-XPU therefore unblocks BOTH FULL capture AND the native reject sampler. [VERIFIED]

  [4] Independently, ngram acceptance is too low (~16%) for the workload.
        - 39/244 draft tokens accepted; per-position 17/8/7/7% (JOURNAL 2026-06-19). At N=4,
          expected accepted = 4*0.16 = 0.64 -> ~1.64 tokens/verify ceiling BEFORE any overhead.
          This is failure mode (ii) and is ORTHOGONAL to capture. [VERIFIED measurement; Amdahl INFERRED]

  Chain summary:
     low-acceptance drafter (ngram, mode ii)  +  eager-attention verify (mode i, from blockers 1+2)
        -> verify costs more than the ~0.64 extra tokens save -> NET NEGATIVE.
     Minimal flip path = attack mode (ii) FIRST (higher-acceptance drafter), capture (mode i) SECOND.

================================================================================
B. RANKED PATH-TO-POSITIVE (experiment + expected accept/speedup + WHY)
================================================================================
Ranked by expected EV on ONE B70 today. Each row: effort / expected accept / expected net speedup /
the experiment / the WHY / the key risk.

--- RANK 1: Native Qwen3.6 MTP head (single-pass drafter) -- on ONE card NOW ---------------------
  Effort:   MEDIUM (no kernel work; checkpoint + serve flags exist on host).
  Accept:   ~75-88% (CUDA reference, Lorbus card / public 4xB70 bench accept-len 4.04 @ spec=5).
  Speedup:  target 1.5-3x decode (MTP_TODO success gate >=3x; realistic first number 1.5-2x on 1 card).
  Experiment:
      Serve `Lorbus_Qwen3.6-27B-int4-AutoRound` on ONE B70 (fits, ~18 GB int4),
      `--speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":3}'` (sweep 2..5),
      PIECEWISE capture ON. Measure decode MTP-on vs MTP-off + accept length. (See section F cmd.)
  WHY it beats ngram: MTP is a SINGLE forward/step (launch-multiplier ~1) so it does NOT multiply the
      eager-attention overhead the way an autoregressive K-step drafter (or N+1 ngram verify) does --
      its advantage SURVIVES the no-FULL-capture penalty (JOURNAL 2026-06-18 DFlash insight). And the
      head was TRAINED against this verifier -> high acceptance attacks failure mode (ii) directly.
  WHY now: the `spec_sequence_masks`/`gdn_attention` NotImplementedError that blocked this on the old
      llm-scaler b8.1 image is GONE in vLLM 0.23.0 (verified: not present in v0230; `qwen3_5_mtp.py`
      ships `Qwen3_5MultiTokenPredictor`, forward is device-agnostic, no `forward_xpu` raise).
  Risk:   (a) the Lorbus checkpoint quantizes `mtp.layers.0.*` to int4 (qweight/qzeros/scales) --
              `mtp.fc` stays BF16 but the MTP body is int4; this MAY depress accept length vs a
              BF16-head checkpoint. Fallback: serve full-BF16 `Qwen_Qwen3.6-27B` (also on host) which
              does NOT fit decode-comfortably (54 GB) -> use it only as an accept-length REFERENCE on
              CPU/short ctx, or produce a W4A16 quant that keeps the whole `mtp.*` BF16 (ignore-list
              `re:.*mtp.*`, per MTP_TODO).
          (b) gdn_attention decode kernel maturity on Xe2 (the 27B is Gated-DeltaNet); verify it
              serves coherently first (it does -- int4 27B already runs at 30.84 t/s w/ PIECEWISE).
          (c) the MTP method string: use `qwen3_5_mtp` (the registered MTP type) or `mtp`; confirm
              from the served log which proposer loads.

--- RANK 2: Real draft model (Qwen3-0.6B) for the dense 14B -- on ONE card NOW -------------------
  Effort:   LOW (drafter `Qwen_Qwen3-0.6B` already on host; vocab-matched 151936; no training).
  Accept:   ~40-70% (workload-dependent; far above ngram's 16%).
  Speedup:  0-35% decode (single-card; drafter forward overhead is the tax).
  Experiment:
      Serve `Qwen3-14B-W8A8-gptq`, `--speculative-config '{"model":".../Qwen_Qwen3-0.6B",
      "num_speculative_tokens":3}'`, PIECEWISE ON, vs the no-spec PIECEWISE baseline (26.68 t/s).
  WHY: attacks failure mode (ii) -- a real model drafts far better than ngram -- and gives the dense
      14B (which has NO MTP head) a spec path. Caveat: draft-model is K sequential drafter forwards
      -> launch-multiplier ~K -> MORE exposed to the no-FULL-capture penalty than MTP. So expect a
      smaller win than MTP, but it is the only spec path for the dense 14B.
  Risk: drafter forward + verify overhead on one card may still eat the gain at low acceptance; keep
      num_speculative_tokens LOW (2-3) to bound waste. draft-model on XPU is plausible-but-not-source-
      confirmed (doc 06 sec 3a) -> this experiment IS the confirmation.

--- RANK 3: Fix Triton-XPU -> TRITON_ATTN -> FULL capture ----------------------------------------
  Effort:   LOW-MEDIUM if it is import-order/runtime-visibility; HIGH if a rebuild is needed.
  Accept:   unchanged (capture does not change the drafter).
  Speedup:  for ngram: -3% to +10% (flips mode i only; mode ii still caps it). For MTP/draft: COMPOUNDS
            on top of rank 1/2 -- this is where it earns its keep.
  Experiment: see section D (the "0 active drivers" diagnosis) -> get `triton.runtime.driver.active`
            to resolve a device, then `VLLM_ATTENTION_BACKEND=TRITON_ATTN` + `cudagraph_mode=FULL`.
  WHY second-order: PIECEWISE already hits ~74-84% of the BW ceiling; FULL only recovers the residual
      attention-launch overhead (a few % standalone). Its real value is REMOVING the eager-attn
      verify tax so MTP/draft spec-decode keeps more of its theoretical win. Also re-enables the
      native Triton reject sampler (blocker [3]).
  Risk: the #34482 "TRITON_ATTN -> FULL" claim is NOT visible in the v0230 `platforms/xpu.py` mode
      logic (no TRITON_ATTN<->FULL linkage found) -> must be tested live; FULL may still need the
      driver-version gate from PR #38193 ("not stable yet").

--- RANK 4: oneAPI 2026.0 toolchain -> flash-attn FULL capture -----------------------------------
  Effort:   MEDIUM-HIGH (rebuild the image on the 2026.0 DPC++ toolchain).
  Accept:   unchanged. Speedup: same as rank 3 for the capture half, no Triton dependency.
  WHY: oneAPI DPC++ 2026.0 release notes lift exactly the `work_group_scratch_memory` SYCL-Graph
      restriction that blocks flash-attn FULL -> FULL capture with NO vLLM/Triton change. A clean
      "toolchain freebie" but gated on doing the image rebuild + revalidating the whole stack.
  Risk: torch/vLLM/flash-attn/oneAPI compatibility churn; new regressions; bigger blast radius.

--- RANK 5 (parked): capture-free ngram tuning --------------------------------------------------
  Lower num_speculative_tokens to 2-3, ngram ONLY for code/RAG/repetitive. Expected: flat-to-small-
  positive on favorable prompts only. Keep as a cheap A/B for the coding server, not a headline lever.

================================================================================
C. ONE CARD NOW  vs  CARD #2 (what waits)
================================================================================
DO ON ONE CARD NOW (capture works, spec-decode's only viable home):
  - [RANK 1] MTP on `Lorbus_Qwen3.6-27B-int4-AutoRound` (fits 1 card) -- THE experiment.
  - [RANK 2] Qwen3-0.6B draft-model on the dense 14B w8a8 (the dense-model spec path + plumbing proof).
  - [RANK 3/4] Unblock FULL capture (Triton-XPU fix or oneAPI 2026.0) to compound the above.
  - Build/prove the MTP serve+bench plumbing on 1 card so it is READY for the 27B-W8A8 on card #2.

WAITS FOR CARD #2 (and even then, NOT for spec-decode latency):
  - 27B-W8A8 (~33 GB) needs TP2 just to FIT. But TP2 DISABLES graph capture (#34482 distributed: NO)
    and adds per-step cross-card collectives (no P2P on B70, host-staged) -> spec-decode at TP2 is
    expected NET-NEGATIVE for interactive decode (matches the prior TP2+MTP negative result).
  - Verdict: on card #2, run 27B-W8A8 TP2 for CAPACITY/throughput WITHOUT spec-decode; keep the MTP
    win on the SINGLE-card configs that fit (27B int4/W4A8 ~17-18 GB, where capture stays on).
  - Pipeline-parallel does NOT rescue this (still distributed, still uncaptured, batch-1 bubbles).
  - The headline single-card MTP'd 27B is therefore W4A8 or W4A16 (fits 1 card, capture ON, MTP ON),
    NOT the 2-card W8A8 -- a notable steer for the MTP_TODO Phase B/C ordering.

================================================================================
D. THE TRITON-XPU "0 active drivers" DIAGNOSIS (can we get TRITON_ATTN -> FULL capture?)
================================================================================
WHAT happens (verified): vLLM's `triton_utils/importing.py` counts Triton backend drivers whose
`is_active()` is True. In a non-distributed env it requires EXACTLY 1; the B70 yields 0 -> it logs
`Triton is installed but 0 active driver(s) found (expected 1). Disabling Triton` and sets
HAS_TRITON=False. That kills both TRITON_ATTN and the Triton reject sampler.

WHAT it is NOT (ruled out): a wrong/stock-triton package. Verified on `:int8g` AND `:int8` AND
`:v0230`: all carry `triton-xpu 3.7.0` correctly matched to `torch 2.11.0+xpu`. So codex's "stock
CUDA triton got pulled in" hypothesis is FALSE for our images. [VERIFIED]

WHAT it most likely IS: the intel-xpu-backend-for-triton driver's `is_active()` returns False at
vLLM-import time even though `torch.xpu` sees the GPU (flash-attn/PIECEWISE both work on the same
image WITH the GPU passed). The Intel Triton "xpu" driver probes the SYCL/Level-Zero runtime (and
PTI) directly; it returns inactive when:
  (a) the SYCL/L0/PTI runtime libs it probes are not visible at import (LD_LIBRARY_PATH / a missing
      libze/pti), OR
  (b) XPU is not yet initialized when Triton imports (import-order: torch.xpu device not realized
      before the backend probe runs), OR
  (c) the device-detection in the bundled triton-xpu 3.7.0 build does not match the image's L0/driver.

DIAGNOSTIC LADDER (run on the GPU host, GPU passed, behind `scripts/gpu-run`):
  1. `python -c "import torch; print(torch.xpu.is_available(), torch.xpu.device_count())"`  (expect True,>=1)
  2. `python -c "import torch; import triton; print(triton.__version__);
        from triton.backends import backends;
        print([(k, b.driver.is_active() if b.driver else None) for k,b in backends.items()])"`
     -> this prints PER-BACKEND is_active(). Find which backend is the xpu one and why it is False.
  3. Force torch.xpu init BEFORE the triton probe: `import torch; torch.xpu.init();
        x=torch.ones(1,device="xpu"); import triton; ...is_active()...` -> if THIS flips it True, the
     fix is import-order (vLLM imports triton before realizing an XPU device) -> a tiny patch /
     a `torch.xpu.init()` shim, LOW effort.
  4. Check the SYCL/PTI libs the Intel triton driver needs: `python -c "import triton.backends.intel"`
     (or the xpu backend module) and inspect its ImportError if any; `ldd` the driver .so for missing
     libze_intel_gpu / libpti.
Ranked fixes by effort: (3) import-order shim [lowest] -> (a) LD_LIBRARY_PATH / add missing runtime
lib [low] -> rebuild image with aligned triton-xpu/L0 [medium] -> build Intel Triton from source [high].

IF FIXED -> TRITON_ATTN should engage -> test `cudagraph_mode=FULL`/`FULL_AND_PIECEWISE`. NOTE the
v0230 `platforms/xpu.py` mode logic does NOT show an explicit TRITON_ATTN->FULL unlock, so confirm
FULL actually captures attention (vs silently staying PIECEWISE) from the capture logs. Expect FULL
to also require the PR #38193 driver gate ("not stable yet" -> off by default).

================================================================================
E. VERIFIED  vs  INFERRED (with source pointers)
================================================================================
VERIFIED (live grep of `vllm-xpu-env:v0230` / `:int8g`, or measured in JOURNAL/FINDINGS):
  - vLLM 0.23.0, torch 2.11.0+xpu, triton-xpu 3.7.0 in all our images. [docker pip list]
  - The `gdn_attention ... spec_sequence_masks` NotImplementedError is NOT present in v0230; MTP
    forward is device-agnostic; `qwen3_5_mtp.py` ships `Qwen3_5MultiTokenPredictor`,
    `mtp_num_hidden_layers` default 1. [grep v0230 model files]
  - Spec-decode methods present & ungated on XPU: ngram, draft_model, eagle/eagle3, dflash, medusa,
    and MTP variants incl. `qwen3_5_mtp`/`qwen3_next_mtp`/`mtp`. [vllm/config/speculative.py,
    vllm/v1/spec_decode/*]
  - Rejection sampler is Triton-jit (`vllm/v1/sample/rejection_sampler.py:14,714+`). [grep]
  - Triton disable logic + the exact `len(active_drivers)!=1` gate. [vllm/triton_utils/importing.py]
  - `triton_attn.py` exists; TRITON_ATTN is a valid XPU backend choice. [vllm/v1/attention/backends/]
  - XPU graph gates on `supports_xpu_graph()` (torch>=2.11) + `VLLM_XPU_ENABLE_XPU_GRAPH`.
    [vllm/platforms/xpu.py:193-204, utils/torch_utils.py]
  - PIECEWISE works; FULL via flash-attn dies on work_group_scratch_memory; TRITON_ATTN ignored due
    to the Triton disable. [JOURNAL 2026-06-20 live runs]
  - Spec-decode measured net-negative: ngram 14B w8a8 -7.8% eager / -7% PIECEWISE, ~16% accept.
    [FINDINGS, JOURNAL 2026-06-19]
  - Host has: `Lorbus_Qwen3.6-27B-int4-AutoRound` (ships `mtp.fc` + `mtp.layers.0.*`, int4 body),
    `Qwen_Qwen3.6-27B` (full BF16, mtp.* present), `Qwen_Qwen3-0.6B` (draft), the 14B quants.
    [ssh ls + safetensors index grep]

INFERRED / CROSS-ARCH (benchmark on B70 before trusting):
  - MTP accept ~75-88% and the 1.5-3x speedup -- CUDA-derived (Lorbus card, 4xB70 BF16 bench);
    XMX/accept on B70 single-card UNMEASURED. THIS is the rank-1 experiment's job.
  - The Lorbus int4-quantized mtp body lowering accept length -- plausible, unmeasured.
  - "TRITON_ATTN unlocks FULL on XPU" (#34482) -- claimed in PR notes; NOT visible in v0230 xpu.py
    mode logic -> live-test required.
  - Amdahl 1.64x ceiling for ngram@16%/N=4 -- arithmetic from the measured accept, not a run.
  - The Triton `is_active()` root cause (import-order vs missing-lib vs build-skew) -- diagnosed by
    elimination (package is correct, GPU is visible); the exact sub-cause needs the section-D ladder.

PRs / issues: #43565 (XPU GDN-attention MTP, in 0.23.0), #34482 (XPU graph matrix: distributed NO,
TRITON_ATTN->all modes), #38193 (FULL off-by-default, driver gate), #43092 (torch.cuda->xpu shim),
intel/llm-scaler #386 (the OLD spec_sequence_masks crash, pre-0.23.0), intel-xpu-backend-for-triton
#6658 (gdn recurrent DEVICE_LOST).

================================================================================
F. FIRST EXPERIMENT FOR THE GPU LEAD (one card, behind scripts/gpu-run)
================================================================================
RANK-1 (highest EV): native MTP on the int4 27B (fits one card, capture ON).
  Adapt scripts/51 (it already supports --speculative-config + GRAPH + CGMODE). Shape:

    MODEL=/mnt/vm_8tb/b70/models/Lorbus_Qwen3.6-27B-int4-AutoRound
    served-model-name qwen3.6-27b-int4-autoround-mtp        # method-tagged (CLAUDE.md guard)
    image vllm-xpu-env:v0230    # has GDN; NOT :int8g (that's the 14B int8 path)
    --dtype float16 --tensor-parallel-size 1 --trust-remote-code
    VLLM_XPU_ENABLE_XPU_GRAPH=1   cudagraph_mode=PIECEWISE   (drop --enforce-eager)
    --speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":3}'   # sweep 2,3,4,5

  CONTROLS (same harness, scripts/38): MTP-OFF PIECEWISE baseline (no --speculative-config).
  MEASURE: decode t/s MTP-on vs MTP-off, mean accept length, accept%@3, prefill/TTFT, VRAM, coherence.
  VERIFY served id encodes the method; cross-check vs models.yaml (CLAUDE.md model-check).
  GO/NO-GO: accept length >= 3 AND decode MTP-on > MTP-off by >=20% -> MTP is the lever, proceed to
            sweep + the W8A8/W4A8 format question (MTP_TODO Phase A/B). If accept < 3 or decode not
            positive -> suspect the int4 mtp body; re-run on a W4A16 quant that keeps mtp.* BF16, or
            fall back to RANK 2 (Qwen3-0.6B draft on the 14B) to prove the spec loop, then build the
            BF16-mtp 27B quant.

  Cheaper alternative first run (if you want the spec LOOP proven before touching the 27B/GDN):
    RANK-2 draft-model on the dense 14B w8a8, PIECEWISE ON:
      --speculative-config '{"model":"/mnt/vm_8tb/b70/models/Qwen_Qwen3-0.6B",
                             "num_speculative_tokens":3}'
    vs the 26.68 t/s PIECEWISE no-spec baseline. Proves draft acceptance >> ngram's 16% and that the
    spec plumbing + reject path work on our stack -- low risk, fast, no GDN.
