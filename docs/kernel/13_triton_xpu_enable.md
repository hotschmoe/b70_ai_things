# 13 - Enabling Triton-XPU on the B70 (make is_active() return True -> TRITON_ATTN + Triton sampler)

Goal: make vLLM stop logging `Triton is installed but 0 active driver(s) found (expected 1).
Disabling Triton` inside image `vllm-xpu-env:v0230` on the Arc Pro B70, so that
(1) `VLLM_ATTENTION_BACKEND=TRITON_ATTN` actually engages (the FULL-graph-capture lever, A2), and
(2) the Triton-jit rejection sampler runs natively instead of a slow fallback (the MTP net-positive lever).

This doc supersedes the "section D" diagnosis in `12_mtp_specdecode_plan.md` with the EXACT verified
code path. All sources were read live from the installed packages in `vllm-xpu-env:v0230`.

================================================================================
TL;DR (read this first)
================================================================================
1. The gate is NOT a missing oneAPI lib, NOT a missing setvars, NOT a Triton-internal L0/SYCL/PTI
   probe. In THIS triton-xpu 3.7.0 build the Intel driver's `is_active()` is LITERALLY:
       @staticmethod
       def is_active():
           try:
               import torch
               return torch.xpu.is_available()
           except ImportError:
               return False
   (`triton/backends/intel/driver.py` lines 974-978, class `XPUDriver`). So the ENTIRE Triton
   enable/disable decision reduces to one boolean: `torch.xpu.is_available()` in whichever process
   first imports `vllm.triton_utils`. [VERIFIED - source read]

2. `torch.xpu.is_available()` == `torch.xpu.device_count() > 0`, and `device_count()` is decorated
   `@lru_cache(maxsize=1)` and calls `torch._C._xpu_getDeviceCount()` (pure Level-Zero enumeration,
   never throws, NO `_lazy_init`, NO tensor op needed). [VERIFIED - torch/xpu/__init__.py l.64-74]
   Consequence: the FIRST call to `device_count()` in a process is cached forever. If it ever runs
   when the L0 device is not enumerable, that process is poisoned to "no Triton" permanently.

3. ROOT CAUSE of the inspection-container log: NO `--device /dev/dri` -> `_xpu_getDeviceCount()`
   returns 0 -> `is_available()` False -> 0 active drivers -> disable. CONFIRMED by running the
   image with no device: `is_available False count 0`. This is an INSPECTION ARTIFACT and expected.
   [VERIFIED - ran the image without the device]

4. ROOT CAUSE in the real serve (HAS /dev/dri) is ONE of exactly two, and they have OPPOSITE
   implications -- the lead must run ONE 30-second command (section C) to decide which:
     (S1) the disable line is logged by the FRONT/API process and the EngineCore `spawn` child
          INDEPENDENTLY recomputes HAS_TRITON=True -> Triton is ALREADY fine in the engine and the
          log is a harmless red herring. TRITON_ATTN/sampler may already work; just set the flag.
     (S2) `torch.xpu.is_available()` is genuinely False at import time IN the engine process
          (lru_cache poisoned by an early enumeration before the device/affinity was ready) ->
          a real blocker -> fix = warm the cache / init XPU before the gate (section B, RANK 1).

5. Why S1 is plausible and must be checked FIRST: vLLM forces `spawn` on XPU
   (`platforms/xpu.py` l.242-244: "spawn is the only supported multiprocessing method on XPU"), so
   each worker is a fresh interpreter with its OWN module globals and its OWN lru_cache -- a
   front-process False CANNOT poison the child. The "Disabling Triton" you saw may simply be the
   front process, while the engine that runs attention computed True. [VERIFIED - spawn forced]

================================================================================
A. THE EXACT VERIFIED CHAIN (every link read from the v0230 image)
================================================================================
  [1] vllm/triton_utils/importing.py builds:
        active_drivers = [x.driver for x in triton.backends.values()
                          if x.driver and x.driver.is_active()]
      and (non-distributed) sets HAS_TRITON=False unless len(active_drivers)==1. The three backends
      registered via entry-points are: amd, intel, nvidia. [VERIFIED]

  [2] Per-backend is_active() (all three read live):
        - nvidia: torch.cuda.is_available() and torch.version.hip is None   -> False (no CUDA)
        - amd:    torch.cuda.is_available() and torch.version.hip is not None -> False
        - intel:  torch.xpu.is_available()                                   -> the ONLY swing vote
      So active_drivers count == (1 if torch.xpu.is_available() else 0). EXACTLY one driver when
      XPU is up; the "expected 1" in the message is the intel one. [VERIFIED]

  [3] torch.xpu.is_available() -> device_count()>0 ; device_count() is @lru_cache(maxsize=1) over
      torch._C._xpu_getDeviceCount() (L0 enumeration). [VERIFIED]

  [4] Image env is ALREADY oneAPI-sourced: LD_LIBRARY_PATH has compiler/dnnl/mkl/pti/tbb;
      OCL_ICD_FILENAMES set; ldconfig resolves libze_loader.so.1 AND libze_intel_gpu.so.1 (in
      /lib/x86_64-linux-gnu, the default loader path -- not only oneAPI). torch._C._has_xpu=True.
      intel-sycl-rt 2025.3.2, triton-xpu 3.7.0, torch 2.11.0+xpu. [VERIFIED]
      => the "needs setvars / missing libze" hypotheses from doc 12 sec D(a) are RULED OUT for the
         serve. The serve container has the libs and env; only the DEVICE visibility / cache timing
         is in question.

  [5] vllm/platforms/xpu.py get_attn_backend does NOT gate TRITON_ATTN on HAS_TRITON: if
      selected_backend==TRITON_ATTN it returns the TRITON_ATTN path directly (l.77-79). And the
      v1 attention selector has NO HAS_TRITON filter. So selection is not blocked by the disable;
      what breaks under HAS_TRITON=False is that `vllm.triton_utils.triton` becomes a
      TritonPlaceholder (no-op @jit dummy) -> the @triton.jit kernels in triton_attn.py and in
      rejection_sampler.py degrade. Fixing is_active() restores the real triton module. [VERIFIED]

WHAT THIS OVERTURNS from doc 12 section D: the inferred sub-causes there (b) "import-order before
torch.xpu device realized" and (c) "triton-xpu build skew vs L0" are the WRONG framing -- the Intel
is_active() does not probe L0 itself, it just calls torch. The only real sub-cause is "torch.xpu
device_count() == 0 in the gating process", whose two flavors are S1 (wrong process) and S2
(lru_cache poison). Sub-cause (a) "missing lib/env" is disproven for the serve.

================================================================================
B. THE RANKED FIX LADDER (each: what / exact patch+test / risk)
================================================================================
Run everything on the GPU host behind `scripts/gpu-run` (CLAUDE.md). The lead runs all GPU tests.

--- RANK 0 (DO FIRST, NO FIX YET): decide S1 vs S2 -- is there anything to fix at all? -----------
  See section C. If the engine process already has is_available()==True (S1), SKIP the fixes and
  just test TRITON_ATTN (RANK 4). Only if the engine process is genuinely False (S2) do RANK 1.

--- RANK 1: warm torch.xpu in EVERY spawned process before the Triton gate (the real S2 fix) -----
  Idea: force a correct device_count() enumeration to populate the lru_cache BEFORE
  vllm.triton_utils imports, in EVERY process (front AND each spawn child). Because workers are
  spawned, an entrypoint one-shot is NOT enough; use a sitecustomize that python auto-imports at
  interpreter startup in every process. ZERO code change to vLLM.
  Exact test (mount a sitecustomize into the serve; no image rebuild):
    1) On the host, create the shim:
         mkdir -p /mnt/vm_8tb/b70/triton_shim
         cat > /mnt/vm_8tb/b70/triton_shim/sitecustomize.py <<'PY'
         import os
         try:
             import torch
             # populate the @lru_cache(maxsize=1) device_count() while the device is visible
             _ = torch.xpu.device_count()
         except Exception:
             pass
         PY
    2) In scripts/51 docker run, add to the env/mounts:
         -v /mnt/vm_8tb/b70/triton_shim:/opt/triton_shim
         -e PYTHONPATH=/opt/triton_shim
       (PYTHONPATH makes python import sitecustomize.py at startup in front + every spawn child.)
    3) Serve with GRAPH=1 CGMODE=PIECEWISE and grep the log:
         scripts/gpu-run bash -lc 'GRAPH=1 CGMODE=PIECEWISE SPEC=0 scripts/51_serve_int8_specdecode.sh'
         docker logs vllm_int8 2>&1 | grep -iE "active driver|Disabling Triton|Using .* backend"
       PASS = the "Disabling Triton" line is GONE (or count is 1).
  Risk: LOW. sitecustomize runs before vLLM in each process; only effect is one early L0
  enumeration. If device_count() was ALREADY correct (S1), this is a harmless no-op. The one failure
  mode: if device_count() is 0 even here (device truly not enumerable in that process), the warm
  caches 0 and nothing improves -> that proves the device is the problem, not ordering (go RANK 2).

--- RANK 2: make enumeration deterministic via env (covers a flaky L0/affinity enumeration) ------
  If RANK 1 shows device_count()==0 even when warmed early, the L0 enumeration itself is returning 0
  in that process. Pin the selector/hierarchy so enumeration is unambiguous. The serve already sets
  ZE_AFFINITY_MASK=0; ADD (in scripts/51 docker run -e list), one at a time, re-test the grep:
       -e ONEAPI_DEVICE_SELECTOR=level_zero:gpu
       -e ZE_FLAT_DEVICE_HIERARCHY=FLAT
  Exact test (combine with the RANK-1 grep):
       scripts/gpu-run bash -lc 'ONEAPI_DEVICE_SELECTOR=level_zero:gpu ZE_FLAT_DEVICE_HIERARCHY=FLAT \
         GRAPH=1 CGMODE=PIECEWISE SPEC=0 scripts/51_serve_int8_specdecode.sh'
       docker logs vllm_int8 2>&1 | grep -iE "active driver|Disabling Triton|Using .* backend"
  Risk: LOW-MEDIUM. ZE_AFFINITY_MASK=0 + ONEAPI_DEVICE_SELECTOR can interact; if the model stops
  seeing the GPU, drop ZE_AFFINITY_MASK and keep only ONEAPI_DEVICE_SELECTOR. Validate the model
  still serves coherently (it must still load + answer), not just that the log changed.

--- RANK 3: one-line monkeypatch of importing.py (last resort; can MASK a real failure) ----------
  Only if RANK 1/2 cannot flip enumeration but you have INDEPENDENTLY proven (RANK 0 / a tensor op)
  that torch.xpu DOES work in the engine -- i.e. the gate is wrong, not the device. Force the cache
  refresh right before the count, inside the gate, in-place (mounted overlay, no rebuild):
    Patch (prepend inside the `if HAS_TRITON:` try-block, before `from triton.backends import backends`):
        import torch
        torch.xpu.device_count.cache_clear()  # drop any poisoned 0
        torch.xpu.init()                       # realize XPU in THIS process
        torch.xpu.device_count.cache_clear()
    Mount it over the installed file:
        -v /mnt/vm_8tb/b70/patches/importing.py:/opt/venv/lib/python3.12/site-packages/vllm/triton_utils/importing.py
    (copy the real file out first: docker run --rm ... cat .../importing.py > patches/importing.py, edit, mount)
  Test: same grep as RANK 1.
  Risk: MEDIUM. If the device is genuinely absent in that process this FORCES HAS_TRITON=True over a
  dead device -> the @triton.jit kernels then fail at first call (worse than the clean fallback).
  Never ship this without RANK-0 proof that XPU works in the gating process. Brittle to vLLM bumps.

--- RANK 4: (after Triton is enabled) actually engage TRITON_ATTN + FULL capture -----------------
  Independent of the enable fix; this is the PAYOFF test. Once the disable line is gone:
       VLLM_ATTENTION_BACKEND=TRITON_ATTN  (env in docker run -e), GRAPH=1, CGMODE=FULL (or
       FULL_AND_PIECEWISE). scripts/51 already plumbs CGMODE.
    scripts/gpu-run bash -lc 'GRAPH=1 CGMODE=FULL SPEC=0 \
       DOCKER_EXTRA="-e VLLM_ATTENTION_BACKEND=TRITON_ATTN" scripts/51_serve_int8_specdecode.sh'
       (add `-e VLLM_ATTENTION_BACKEND=TRITON_ATTN` to the docker run line if scripts/51 has no
        passthrough; one-line edit.)
    Confirm from the log: "Using Triton backend." (NOT "Using Flash Attention backend.") AND that
    FULL capture proceeds without the `work_group_scratch_memory` SYCL-Graph RuntimeError that kills
    flash-attn FULL.
  Risk: MEDIUM. v0230 xpu.py mode logic shows NO explicit TRITON_ATTN->FULL unlock (doc 12 E), so
  FULL may still silently stay PIECEWISE or need the PR #38193 driver gate; verify from capture logs.

--- RANK 5: rebuild image with aligned triton-xpu/L0 / build Intel Triton from source ------------
  NOT indicated. The package is correct (triton-xpu 3.7.0 matched to torch 2.11.0+xpu) and is_active()
  is a pure torch call -- a rebuild changes nothing about torch.xpu.is_available(). HIGH effort, ~0 EV.
  Skip unless RANK 1-3 all fail AND torch.xpu itself is found broken (which would break the model too).

================================================================================
C. RANK 0 - THE ONE COMMAND THAT DECIDES S1 vs S2 (run this FIRST)
================================================================================
Add PID + process-name + the actual booleans to the gate, in-place, then read which process logs
False. Copy the installed importing.py out, change ONLY the logger.info disable line to include
identity + counts, mount it, serve, and read the log. Minimal probe edit:

  In the non-distributed branch, replace the disable log with:
     import os, multiprocessing as _mp, torch as _t
     logger.info("TRITON-GATE pid=%s proc=%s xpu_avail=%s count=%s active=%d -> %s",
                 os.getpid(), _mp.current_process().name,
                 _t.xpu.is_available(), _t.xpu.device_count(),
                 len(active_drivers),
                 "KEEP" if len(active_drivers)==1 else "DISABLE")
     if len(active_drivers) != 1:
         HAS_TRITON = False

Then:
  scripts/gpu-run bash -lc 'GRAPH=1 CGMODE=PIECEWISE SPEC=0 scripts/51_serve_int8_specdecode.sh'
  docker logs vllm_int8 2>&1 | grep "TRITON-GATE"

READ THE OUTPUT:
  - If the line for proc=MainProcess (or the API front) says DISABLE but there is ALSO a line for
    the EngineCore/Worker process saying xpu_avail=True KEEP -> this is S1. Triton is fine in the
    engine; the front-process log is cosmetic. Go straight to RANK 4 (TRITON_ATTN), no enable fix.
  - If the EngineCore/Worker line ITSELF says xpu_avail=False count=0 DISABLE -> this is S2 (real
    blocker). Apply RANK 1 (sitecustomize warm); if count stays 0, RANK 2 (env); RANK 3 last.
  - If there is only ONE process line and it is in the engine and False -> S2, RANK 1.

================================================================================
D. MOST-LIKELY SINGLE FIX (the bet)
================================================================================
MOST LIKELY OUTCOME: S2 with an lru_cache timing poison, fixed by RANK 1 (the sitecustomize warm via
PYTHONPATH), because: the libs+env are present (rules out lib/setvars), spawn isolates processes
(so if the engine logged it, its OWN cache was poisoned by an early enumeration during torch/vLLM
import before ZE_AFFINITY_MASK/L0 settled), and warming device_count() in-process before the gate is
the cleanest deterministic counter. RANK 1 is also a SAFE no-op if it turns out to be S1.

SINGLE MOST-LIKELY FIX + EXACT TEST (RANK 1):
  # host:
  mkdir -p /mnt/vm_8tb/b70/triton_shim
  printf 'import torch\ntry:\n    torch.xpu.device_count()\nexcept Exception:\n    pass\n' \
      > /mnt/vm_8tb/b70/triton_shim/sitecustomize.py
  # add to scripts/51 docker run:  -v /mnt/vm_8tb/b70/triton_shim:/opt/triton_shim -e PYTHONPATH=/opt/triton_shim
  scripts/gpu-run bash -lc 'GRAPH=1 CGMODE=PIECEWISE SPEC=0 scripts/51_serve_int8_specdecode.sh'
  docker logs vllm_int8 2>&1 | grep -iE "active driver|Disabling Triton|Using .* backend"
  # PASS = no "Disabling Triton"; then re-serve with -e VLLM_ATTENTION_BACKEND=TRITON_ATTN CGMODE=FULL.

================================================================================
E. VERIFIED vs INFERRED
================================================================================
VERIFIED (read live from vllm-xpu-env:v0230, or run):
  - intel XPUDriver.is_active() == torch.xpu.is_available() (driver.py l.974-978).
  - importing.py gate = exactly-1 active driver; only intel can be active (cuda/hip False). 
  - device_count() is @lru_cache(maxsize=1) over _xpu_getDeviceCount, no _lazy_init.
  - oneAPI env + libze_loader/libze_intel_gpu present in the image (ldconfig + env dump).
  - no-device run: is_available False count 0 (inspection artifact reproduced).
  - vLLM forces spawn on XPU (xpu.py l.242-244); spawn child re-runs the gate independently.
  - get_attn_backend returns TRITON_ATTN without a HAS_TRITON check (xpu.py l.77); selector has no
    HAS_TRITON filter -> selection not blocked; only the @jit kernels degrade under the placeholder.
  - torch 2.11.0+xpu, triton-xpu 3.7.0, intel-sycl-rt 2025.3.2, torch._C._has_xpu True.

INFERRED (needs the lead's GPU run to confirm):
  - Whether the serve's disable line is S1 (front-process, harmless) or S2 (engine, real). RANK 0
    settles it. This is THE open question; everything else is mechanically determined.
  - That RANK 1 (warm via sitecustomize) flips S2 -> depends on whether enumeration in-process is
    timing-poisoned (fixable) vs structurally 0 (then RANK 2 env).
  - That TRITON_ATTN actually yields FULL capture on XPU (#34482 claim; not visible in xpu.py mode
    logic; possibly gated by #38193). RANK 4 confirms.
