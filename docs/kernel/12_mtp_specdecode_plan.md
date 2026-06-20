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

================================================================================
## Making MTP net-positive under PIECEWISE (no toolchain)
================================================================================
Written AFTER the measured MTP runs (JOURNAL 2026-06-20: N=1 = 25.5 t/s = -19% vs 31.4 MTP-off,
N=3 = 19.65 = -37%; 86.9% first-token accept; Triton-sampler enable moved decode ~0). This section
answers: WHAT actually costs the -19%, is the spec-verify shape capturable under PIECEWISE, and a
RANKED list of serve configs the lead can test. All vLLM mechanics below were read live from the
v0230 image (paths cited); the cost split is codex arithmetic over our measured numbers.

--------------------------------------------------------------------------------
G.1  CONFIRMED COST DECOMPOSITION (what runs each spec step, what is eager)
--------------------------------------------------------------------------------
Per decode step the spec path runs THREE things (vllm/v1/spec_decode/llm_base_proposer.py +
gpu_model_runner.py execute_model):
  (a) DRAFTER forward(s). For qwen3_5_mtp the drafter is normalized to method="mtp" and routed to
      EagleProposer (speculative.py:1071 use_eagle includes "mtp"; gpu_model_runner.py:599). The MTP
      head is ONE decoder layer (qwen3_5_mtp.py:73 mtp_num_hidden_layers default 1, layer_type=
      "full_attention") + shared embed + a small fc + shared lm_head -- it is CHEAP, ~1/40 of a main
      forward. For N>1 it is run AUTOREGRESSIVELY: 1 forward then a sequential loop of (N-1) more
      (llm_base_proposer.py:588 `for token_index in range(num_speculative_tokens-1)`), each its OWN
      set_forward_context + attn-metadata rebuild (build_per_group_and_layer_attn_metadata) + dispatch.
      vLLM even WARNS for mtp+N>1: "run multiple times of forward on same MTP layer ... may result in
      lower acceptance" (speculative.py:716). Matches our measured accept decay 0.84/0.57/0.46.
  (b) VERIFY forward: the FULL ~40-layer main model over (1+N) query tokens, 1 seq.
  (c) reject-sample + spec bookkeeping (rejection_sampler, _calc_spec_decode_metadata, prepare_inputs).

WHAT IS EAGER UNDER PIECEWISE (the crux):
  - PIECEWISE captures GEMM/MLP and SPLITS OUT every op in compilation.py `_attention_ops`
    (lines 756-769). That list includes BOTH `vllm::unified_attention_with_output` (full flash-attn)
    AND `vllm::gdn_attention_core_xpu` (the GDN linear-attn core). So ALL attention runs EAGER --
    not just the ~1-in-4 full-attn layers; the ~30 GDN layers' core runs eager too.
  - There is NO "capture attention but stay PIECEWISE" mode: setting splitting_ops=[] under PIECEWISE
    falls back to cudagraph_mode=NONE (compilation.py:1151-1176). Only FULL captures attention.
  - The verify GEMM IS captured/replayed. The dispatcher (cudagraph_dispatcher.py) stores PIECEWISE
    keys with num_reqs=None,uniform=False and matches purely on num_tokens; the 2-/4-token verify
    shape is in cudagraph_capture_sizes [1,2,4,8,..], so the verify's GEMM/MLP replays at the right
    shape (our capture log "verify batch 1-32" confirms). The eager part is ONLY the attention split.
  - The DRAFTER is hardcoded PIECEWISE-or-NONE, NEVER FULL (llm_base_proposer.py:386-401
    initialize_cudagraph_keys: "Only supports PIECEWISE"). So the drafter's own full-attn layer is
    also eager. (Upstream tracks this: vLLM issue #33341 "Support Full CUDA Graph for the drafter".)

THE NUMBER (codex over measured t/s; T = plain step = 31.8 ms):
  N=1: step = 72.5 ms = 2.28x T, mean accept 1.85  -> pay 2.28x for 1.85x tokens = -19%.
  N=3: step = 145.6 ms = 4.58x T, mean accept 2.86 -> pay 4.58x for 2.86x tokens = -37%.
  The decisive clue is the N=1->N=3 DELTA: +2.30x T (73 ms) for just TWO extra CHEAP 1-layer drafter
  forwards + verify going 2->4 tokens. Two 1-layer math passes CANNOT cost 2.3x a 40-layer step. So
  the dominant cost is a HEAVY PER-SPEC-STEP / PER-DRAFTER-FORWARD FIXED OVERHEAD that does NOT
  amortize: each (drafter and verify) sub-pass pays its own prepare_inputs + per-layer attn-metadata
  rebuild + set_forward_context/dispatch + (for verify) the rejection sampler + extra lm_head rows,
  ON TOP of eager attention over N+1 tokens. This is consistent with N=1 already being 2.28x T even
  though the drafter is one cheap layer: most of the 1.28x extra is FIXED spec machinery + the eager
  m=2 verify attention, NOT drafter math.

  RULED OUT as the bottleneck: (1) the Triton rejection SAMPLER -- enabling it moved decode ~0
  (JOURNAL); (2) the verify GEMM "running eager" -- it replays at the captured 2-/4-token shape;
  (3) num_speculative_tokens too low -- N=1 (1 draft, 85% accept) is the BEST case and is still -19%.

--------------------------------------------------------------------------------
G.2  IS THE VERIFY SHAPE CAPTURABLE? -- yes for GEMM, NO for attention; the exact knob
--------------------------------------------------------------------------------
  - The verify N+1 shape's GEMM/MLP is ALREADY captured under PIECEWISE (G.1). cudagraph_capture_sizes
    already covers 1,2,4,8 so the 2-/4-token verify replays. Adding sizes does NOTHING for the -19%.
  - The verify ATTENTION (the actual tax) is capturable ONLY under cudagraph_mode=FULL /
    FULL_DECODE_ONLY, via the dispatcher's uniform-decode path: uniform_decode_query_len =
    1 + num_speculative_tokens (cudagraph_dispatcher.py:37-40) and the FULL keys are populated for the
    decode shape only when `cudagraph_mode.decode_mode()==FULL` (lines 207-231). The exact field is
    CompilationConfig.cudagraph_mode = FULL_DECODE_ONLY (or FULL); the verify shape it would capture
    is num_tokens = (1+N)*num_reqs.
  - BUT on the B70 this is the SAME wall we already hit: FULL routes attention into the SYCL graph and
    flash-attn dies on `sycl_ext_oneapi_work_group_scratch_memory ... not yet available with the SYCL
    Graph extension` (toolchain, oneAPI DPC++ 2026.0), and TRITON_ATTN (the other FULL-capable backend)
    is unwired on XPU (JOURNAL 2026-06-20). So the capturable verify shape EXISTS and the knob is
    cudagraph_mode=FULL_DECODE_ONLY, but it is blocked by the exact toolchain limit doc 13 documents,
    NOT by a config we are missing. There is no PIECEWISE knob that captures attention.

--------------------------------------------------------------------------------
G.3  RANKED CONFIGS TO TEST ON THE B70 NOW (no toolchain change)
--------------------------------------------------------------------------------
Honest expectation: NONE of these is likely to flip MTP net-positive on this hybrid-GDN model under
PIECEWISE -- the -19% is structural (fixed spec machinery + eager-attn verify, neither removable
without FULL). They are ranked as the cheap A/Bs worth burning a GPU slot on BEFORE conceding, plus
the one diagnostic that would change the verdict. Each: exact flag + expected effect.

  RANK 1  num_speculative_tokens=1 is the floor; CONFIRM the verdict, do not expect a flip.
     --speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":1}'  (GRAPH=1 PIECEWISE)
     Expected: ~25.5 t/s (-19%), our measured best case. This is the CEILING of PIECEWISE MTP here.

  RANK 2  disable_padded_drafter_batch -- cheapest knob that touches per-step waste.
     --speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":1,
                            "disable_padded_drafter_batch":true}'
     Expected: small/uncertain (a few %). Removes padded-batch waste in the drafter prepare path but
     does NOT remove eager attention or the fixed metadata/dispatch overhead. Low cost to A/B; if it
     does not move N=1 toward breakeven, the fixed overhead is confirmed and we stop here.

  RANK 3  PROFILE the spec step to PROVE the split (the diagnostic, higher value than any knob).
     Serve N=1 with profiling on and read where the 1.28x-extra goes: eager-attn wall vs prepare_inputs/
     metadata/dispatch fixed cost. ONEDNN_VERBOSE=2 + the vLLM forward timers, or a torch profiler trace
     of one spec step. This decides between "needs FULL capture" (eager-attn dominant) vs "needs upstream
     spec-path overhead reduction" (fixed machinery dominant) -- both point off-PIECEWISE, but it tells
     the lead which upstream lever (oneAPI 2026.0 FULL vs a vLLM spec-path patch / issue #33341) to back.

  RANK 4  num_speculative_tokens=2 -- map the curve (do NOT expect positive).
     --speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":2}'
     Expected: ~21-23 t/s, between N=1 and N=3, still negative. The accept-decay (0.84/0.57) means the
     2nd token rarely pays its draft+verify; only completes the N=1/2/3 curve for the writeup.

  NOT WORTH TESTING (reasoned out, no expected effect on the root cause):
   - raising cudagraph_capture_sizes to include verify shapes: the verify GEMM already replays at
     2/4/8; this captures GEMM not attention -> ~0 effect, +VRAM (matches the earlier "capture >8 =
     0 gain" negative).
   - parallel_drafting=true: collapses the SEQUENTIAL drafter loop (helps N>1) BUT (i) at N=1 there is
     only one drafter forward so it is a no-op, and (ii) for mtp it is not auto-enabled (only dflash
     sets it, speculative.py:756) and the qwen3_5_mtp head is single-layer autoregressive, not a
     parallel multi-head predictor -- forcing it is unsupported/likely-incoherent. Skip on this model.
   - draft_tensor_parallel: attacks compute, not the launch/fixed-overhead bottleneck; single card =
     no TP anyway. Neutral-to-worse.
   - VLLM_XPU_* envs: do not change launch count or the eager-attn split; at most shave constants.

--------------------------------------------------------------------------------
G.4  VERDICT: net-positive on PIECEWISE = NO for this model; it needs FULL capture
--------------------------------------------------------------------------------
The MTP head is an EXCELLENT drafter (legs proven: 85% @ N=1, 2.86 @ N=3). But net-positive decode on
the hybrid-GDN Qwen3.6-27B is NOT achievable by any PIECEWISE serve config, because the -19% is
structural: (1) ALL attention (incl. the majority GDN layers) runs EAGER under PIECEWISE and the
verify pays it over N+1 tokens; (2) a heavy FIXED per-spec-step / per-drafter-forward overhead
(prepare_inputs + per-layer attn-metadata rebuild + dispatch + reject bookkeeping) that does not
amortize at our accept rate. Neither is removable without FULL capture (captures the verify+drafter
attention into one graph, killing both the eager-attn tax AND most of the per-pass dispatch overhead).
FULL is gated on the work_group_scratch SYCL-Graph toolchain limit (oneAPI DPC++ 2026.0) + TRITON_ATTN
being unwired on XPU -- exactly doc 13's blockers. So MTP-positive is a TOOLCHAIN item (oneAPI 2026.0
-> FULL_DECODE_ONLY, which captures uniform_decode_query_len = 1+N), not a config we can set today.
Upstream corroboration: vLLM issue #33341 ("Support Full CUDA Graph for the drafter") -- the drafter
being PIECEWISE-only is a known, tracked limitation, not a B70 misconfig. Keep N=1 MTP filed for the
oneAPI-2026.0 / FULL-capture phase; do NOT ship MTP-on for interactive decode on the current stack.

================================================================================
## Community Qwen3.6 MTP recipes + B70 applicability
================================================================================
Written AFTER a web survey of how the community actually gets NET-POSITIVE Qwen3.6 /
Qwen3-Next MTP decode (sources cited inline). The short answer CONFIRMS our section-G FULL-capture
conclusion: every community single-stream MTP win runs on CUDA with the DEFAULT cudagraph_mode
FULL_AND_PIECEWISE, where the spec-decode VERIFY batch (a uniform-decode batch, query_len = 1+N)
is captured into a FULL cuda graph -- the exact capture the B70 cannot do today. No community
recipe nets positive WITHOUT that full verify capture; the published negative single-stream results
fail for the SAME structural reasons we measured.

--------------------------------------------------------------------------------
H.1  THE COMMUNITY'S PROVEN CONFIGS + HARDWARE + NUMBERS (all CUDA / NVIDIA)
--------------------------------------------------------------------------------
  [POS] vLLM canonical recipe -- the ONE that nets positive, single GPU:
        `--speculative-config '{"method":"mtp","num_speculative_tokens":1}'` (sweep up to 5),
        cudagraph_mode = DEFAULT (FULL_AND_PIECEWISE), attention backend flashinfer / FA3.
        HW: single H100/H200/L40S (40 GB), Qwen3.6-27B-FP8. "MTP supported out of the box for
        low-latency decoding"; num_speculative_tokens=1 is the recommended latency default; disable
        prefix caching for the latency path.
        [recipes.vllm.ai/Qwen/Qwen3.6-27B ; docs.vllm.ai/.../recipes/.../Qwen/Qwen3.5.html ;
         docs.vllm.ai/en/latest/features/speculative_decoding/mtp/]
  [POS] zolotukhin.ai (2026-05-08), Qwen3.6-27B dense Q8, SINGLE device (DGX Spark / GB10):
        method=qwen3_next_mtp gamma=3 -> 7.0 -> 16.8 t/s = 2.40x decode, accept 0.72; gamma=2 = 2.4x
        @ 0.83 accept. A single-device CUDA win (default full capture available).
        [zolotukhin.ai/blog/2026-05-08-why-mtp-heads-are-the-speculative-decode-draft-qwen3-a3b-deserves/]
  [POS] dasroot.net (2026-05), Qwen3.6-35B-A3B (MoE), single 12 GB consumer GPU: MTP ~80 t/s vs
        ~30-35 standard spec-decode (~2.3-2.6x), single-stream. [dasroot.net/posts/2026/05/...]
  [POS] DFlash (LMSYS 2026-06-15), Qwen3.5-397B-A17B BF16, 8x B200 (Modal), SGLang, HumanEval c1:
        DFlash >4.3x baseline and 1.5x NATIVE MTP. So even on the fastest HW, native MTP is the
        BASELINE that DFlash beats -- MTP itself is solidly net-positive there.
        [lmsys.org/blog/2026-06-15-next-generation-speculative-decoding-dflash-v2/]
  [NEG] dredyson.com "I tested every solution", Qwen3.6-27B-FP8, 2x GPU TP2 + Ray, MULTI-NODE:
        method=mtp N=2 -> decode 14.4 -> 7.2 t/s (-50%), accept 88-94%, mean-accept 1.7-1.9. Root
        cause: each spec token = an extra cross-node allgather; "enable MTP only on single-node
        NVLink GPUs." == our section-C TP2 verdict (#34482). [dredyson.com/i-tested-every-solution-...]
  [NEG] vLLM #35387, Qwen3-Next-80B-A3B-FP8, 4x H100 TP4 single-node, N=2: 0.894 -> 1.578 s
        (+76.5% latency). Reporter root-cause: a per-step CPU sync ("copying num_accepted_tokens back
        to CPU before mamba_postprocess"). This is a HYBRID-MAMBA/GDN-specific fixed per-spec-step
        overhead == our section-G.1 "heavy FIXED per-spec-step overhead" thesis, upstream-confirmed on
        the SAME GDN-class arch. [github.com/vllm-project/vllm/issues/35387]
  [NEG] thc1006 RTX 3090, Qwen3.6-35B-A3B, draft/ngram (NOT MTP): 135.7 -> 121.1 t/s best (-11%)
        even at 100% accept; root cause = MoE expert-load union on verify. Single-GPU spec-decode
        CAN go negative even on CUDA with a bad drafter/model -- but this is ngram/draft, not MTP.
        [github.com/thc1006/qwen3.6-speculative-decoding-rtx3090]

--------------------------------------------------------------------------------
H.2  THE DECISIVE QUESTION: does net-positive need FULL capture? -- YES (confirmed)
--------------------------------------------------------------------------------
Mechanism, from the vLLM CUDA-graphs design doc (verbatim): the default cudagraph_mode is
FULL_AND_PIECEWISE on CUDA; it "uses full CUDA Graph for uniform decode and piecewise CUDA Graphs for
others." A spec-decode VERIFY batch IS a uniform-decode batch with "query length 1+num_spec_tokens"
(max_query_len = 1+N), so on CUDA the verify's ATTENTION is captured into a FULL graph by default.
Under PIECEWISE-only "attention or other CUDA-Graphs-incompatible operations stay eager" -- exactly
the B70 state. So the community wins ride on the DEFAULT full verify capture; the B70 negative is the
PIECEWISE case the community never ships into. [docs.vllm.ai/en/stable/design/cuda_graphs/]

Backend support for the verify capture (why XPU is stuck): the FULL/uniform-decode verify needs an
attention backend that supports full cuda graph for the 1+N batch -- per the design doc's table that
is Triton Attention (ALWAYS) and FlashAttention v3 (ALWAYS); FlashInfer/FlashMLA only do narrower
uniform-decode. On XPU our two FULL-capable options are both unavailable: flash-attn FULL dies on the
work_group_scratch SYCL-Graph limit, and TRITON_ATTN is unwired (doc 13). [same doc + our JOURNAL]

Decomposing the B70 -19% against the four candidate causes the task posed:
  (a) MISSING FULL CAPTURE of the VERIFY -- THE dominant cause. Confirmed primary: the community's
      net-positive recipes all have it by default; we don't. This is the single biggest delta.
  (b) Missing fast (Triton) sampler -- RULED OUT as material (our JOURNAL: enabling it moved decode
      ~0). Not the cause.
  (c) Eager GDN/attention in the DRAFTER -- SECOND-ORDER, quantified by upstream. vLLM #33341
      measured FULL cuda graph for the DRAFTER at only ~5% TPOT, and the drafter is PIECEWISE-only
      even on CUDA (eagle.py). And PR #25847 (merged 2025-09-29) makes the EagleProposer EXPLICITLY
      "avoid using cuda graph for xpu platform" -> on XPU the drafter is hard-eager by design. So the
      drafter being eager hurts but is NOT the ~19%; the VERIFY capture is.
  (d) A recipe we haven't tried -- NO net-positive PIECEWISE recipe exists in the community corpus.
      Plus a GDN/Mamba-specific FIXED per-step CPU-sync overhead (#35387) that hits this exact arch
      regardless of capture -- so even WITH full capture, Qwen3.6's hybrid-GDN MTP may net less than a
      pure-attention model would. [#33341, #25847, #35387]

External confirmation of the platform gap: the Intel Arc Pro B-series vLLM launch blog lists
spec-decode methods "n-gram, EAGLE and EAGLE3" as supported on Arc but its own serve example runs
`--enforce-eager`; and the Intel-GPU vLLM guidance states cuda graph "is not supported on XPU, with a
fallback to eager mode." [vllm.ai/blog/2025-11-11-intel-arc-pro-b ; community.intel.com vLLM-on-Intel-GPUs]

--------------------------------------------------------------------------------
H.3  RANKED NEW serve configs worth testing on the B70 (NOT already tried)
--------------------------------------------------------------------------------
Filtered against section-G (already-tried: N=1/2/3 PIECEWISE, disable_padded_drafter_batch, Triton
sampler, capture-size raises). These are genuinely NEW from the community survey. Honest prior: none
flips PIECEWISE net-positive (H.2), but two are cheap and one is the real unlock.

  RANK 1 (the only real unlock, but it is the TOOLCHAIN/Triton item, not a "new flag"):
     Get the VERIFY into a FULL graph via cudagraph_mode=FULL_DECODE_ONLY -- the community default in
     all but name. The community confirms this captures uniform_decode_query_len=1+N. On the B70 this
     needs EITHER doc-13 RANK-4 (TRITON_ATTN engaged: it is the "ALWAYS" full-cudagraph verify backend
     in the design-doc table -> the highest-value reason to finish the doc-13 Triton-enable) OR oneAPI
     2026.0 (flash-attn FULL). Exact, once one is unblocked:
       --speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":1}'
       -O '{"cudagraph_mode":"FULL_DECODE_ONLY"}'  + VLLM_ATTENTION_BACKEND=TRITON_ATTN  GRAPH=1
     This is the community recipe transplanted; it is the config that, IF capture engages, should flip
     the sign. (Still subject to the #35387 GDN CPU-sync tax -- see RANK 3.)

  RANK 2 (NEW, cheap, single card): try standard EAGLE/EAGLE3 spec-decode, which the Intel blog
     lists as SUPPORTED on Arc (unlike a fresh claim about MTP). A community Qwen3.6-27B EAGLE3 head
     exists (Ex0bit/Qwen3.6-27B-PRISM-EAGLE3 on HF). EAGLE3 is the method Intel themselves validated
     on B-series, so it is the most likely-to-be-wired spec path on our stack:
       --speculative-config '{"method":"eagle3","model":"<EAGLE3-head>","num_speculative_tokens":2}'
     Expected: still PIECEWISE-eager-attention-limited (same wall), but it EXERCISES the path Intel
     claims works and may have less GDN-specific overhead than the qwen3_5_mtp head. Low EV on net
     speedup, but it is the one spec method with a vendor "supported" claim we have not tried.

  RANK 3 (NEW diagnostic, highest information value short of FULL capture): confirm the #35387
     GDN/Mamba CPU-sync tax is present in OUR run. Profile one N=1 spec step (section-G.3 RANK-3
     profiler) and look specifically for a device->host copy of num_accepted_tokens before the GDN/
     mamba postprocess. If present, it is a per-step fixed cost that FULL capture alone will NOT
     remove (it is a host sync, not an attention launch) -> tells us Qwen3.6-GDN MTP may stay short of
     the pure-attention community 2.4x even after FULL, and that the real fix is an upstream spec-path
     patch (track #35387), not just our toolchain. Decides whether to keep betting on MTP for THIS
     model vs pivoting to EAGLE3 / a non-GDN target.

  NOT WORTH IT (community corpus adds no new lever here): any PIECEWISE-only MTP flag (corpus has
     zero PIECEWISE net-positive); qwen3_next_mtp method string variants (same EagleProposer path,
     same XPU no-graph PR #25847); raising num_speculative_tokens (community latency default is N=1,
     and our accept-decay already says N>1 loses) -- all already covered by section G.

--------------------------------------------------------------------------------
H.4  RECONCILE: can the B70 get Qwen3.6 MTP net-positive NOW? -- NO (toolchain-gated)
--------------------------------------------------------------------------------
On the CURRENT oneAPI/vLLM-XPU stack: NO net-positive serve config exists. It is genuinely
CUDA-FULL-capture-only until we either (1) finish doc-13's Triton-enable so VLLM_ATTENTION_BACKEND=
TRITON_ATTN engages and cudagraph_mode=FULL_DECODE_ONLY can capture the 1+N verify (TRITON_ATTN is the
"ALWAYS" full-cudagraph verify backend), or (2) move to oneAPI DPC++ 2026.0 so flash-attn FULL works.
Either unblocks the community's exact recipe. The community corpus CONFIRMS our section-G verdict and
adds one caveat: even after FULL capture, Qwen3.6's hybrid-GDN arch carries a per-step CPU-sync tax
(#35387) that a pure-attention model would not -- so the realistic post-FULL B70 number may land below
the cross-arch 2.4x. THE single most promising untried serve config is RANK-1 above
(TRITON_ATTN + FULL_DECODE_ONLY + qwen3_5_mtp N=1), but it is GATED on the doc-13 Triton-enable
landing first -- it is not settable on today's stack. If a sign-flip is wanted with the LEAST new
work, RANK-2 (EAGLE3, the one method Intel claims is supported on Arc) is the cheapest NEW thing to A/B.
