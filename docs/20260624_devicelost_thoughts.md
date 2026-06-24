# 2026-06-24 -- DEVICE_LOST / GPU-wedge investigation and recovery thoughts

Context: post-reboot resumption of the P2P push-allreduce campaign (docs/P2P_GPU.md J.8-J.15).
This note captures the J.16 capture-gated A/B attempt, the re-wedge it caused, a diagnosis of
the wedge, and the open decisions (passwordless sudo, a pre-flight guard, kernel 7.1). Factual
log is in P2P_GPU.md J.16; this file is the reasoning/decision companion.

## What happened today (J.16 capture-gated A/B attempt)

Plan (queued from J.15): re-run the capture-gated A/B on a freshly-rebooted box to bank the J.14
3.35x TTFT win inside a GRAPH=1 production serve.

    GRAPH=1 PUSH_AR_MIN_NUMEL=65536 ./bin/gpu-run bash scripts/108_serve_push_ar_ab.sh smoke

Pre-flight was clean: box rebooted ~1h40m prior, both cards enumerated and free, no containers,
clean git tree. The deferred-dlopen fix (J.15) HELD -- this run got materially further than any
J.15 attempt:

- push_ar sitecustomize loaded, chained the MTP shim, patched `XpuCommunicator.all_reduce`.
- GRAPH=1 model load progressed PAST the rotary-emb crash point that failed pre-J.15-fix.
- `XPUInt8ScaledMMLinearKernel` selected, FlashAttention up, both workers (TP0+TP1) alive,
  safetensors shards loading (saw 50%).

But it never reached `/health` within the 15-min `b70_wait_healthy` window. The script then
stopped the container, and the worker shutdown path threw:

    gpu_model_runner.py _cleanup_profiling_kv_cache -> torch.accelerator.synchronize()
    -> RuntimeError: level_zero backend failed with error: 20 (UR_RESULT_ERROR_DEVICE_LOST)

gpu-run exited 1 after 916s (the 15-min health timeout).

### The wedge this time is WORSE than classic H.13

After teardown, a single-card sanity probe (TP=1, explicitly "wedge-immune" per the old H.13
note) FAILED:

- First probe: a trivial 2048x2048 fp32 matmul (~16 MB) failed after **87s** with
  `UR_RESULT_ERROR_OUT_OF_RESOURCES` (error **40**, OOM-class -- NOT DEVICE_LOST/20).
  `torch.xpu.is_available()` still True, `device_count()` 2.
- Retry probe: a tiny **16x16** alloc+matmul hung **>13 minutes** with no return, on a single
  pinned card, until killed.

No userspace cause: zero stray python/vllm/EngineCore procs, no `fuser` holders that we could
confirm, no docker remnants, lease free. So the degradation is at the **kernel/driver (xe)
level**, on BOTH cards, and it is NOT spared on single-GPU. This contradicts the prior
"Single-GPU (TP=1) serves are unaffected" guidance from H.13 -- today's wedge took out single-card
work too.

## Diagnosis: what actually triggers these wedges

Three datapoints now:

1. **H.13** -- `CCL_TOPO_P2P_ACCESS=1` in a TP>1 serve -> DEVICE_LOST at worker-init allreduce
   warmup; corrupts oneCCL/L0 collective state.
2. **J.15** -- a *string* of TP>1 worker-init crashes (repeated GRAPH=1 model-load failures, serves
   killed mid-init) corrupts the same state, even at `P2PACCESS=0`.
3. **J.16 (today)** -- TP>1 workers **killed mid-GRAPH-capture** (health-wait timeout SIGKILLed
   them); shutdown `synchronize()` -> DEVICE_LOST; aftermath presents as `OUT_OF_RESOURCES` and a
   hard hang even on single-card.

The common factor is NOT a single offending instruction. It is: **TP>1 worker processes torn down
while the Level-Zero / oneCCL collective context or device queues are mid-operation.** The
`DEVICE_LOST` during the shutdown `synchronize()` is the smoking gun -- killing workers during
capture/collective leaves the driver's device context unrecoverable without an xe reload.

Corollary suspicion for today specifically: the health-wait 15-min timeout may have killed a serve
that was merely SLOW to capture (not truly failed). i.e. we may have self-inflicted the wedge by
SIGKILLing mid-capture. Worth confirming the GRAPH=1 capture actually needs > 15 min before
treating the timeout as a real failure.

## Open decisions / recommendations

### 1. Reset (do now)
Containers are clear, so try the lighter reset first; reboot if it hangs/refuses:

    sudo modprobe -r xe && sudo modprobe xe      # needs no /dev/dri in use
    # fallback:
    sudo reboot

### 2. Scoped passwordless sudo (recommended)
Every recovery currently blocks on the human. Recommend a SCOPED NOPASSWD (not blanket) so the
agent can self-recover the loop and auto-reset between attempts:

    # /etc/sudoers.d/xe-reset   (edit via: sudo visudo -f /etc/sudoers.d/xe-reset)
    hotschmoe ALL=(root) NOPASSWD: /usr/sbin/modprobe -r xe, /usr/sbin/modprobe xe

Deliberately scoped to the xe reset only -- not blanket NOPASSWD, not `reboot`.

### 3. Pre-flight guard + auto-reset hook (highest leverage)
A pure "block the bad instruction" linter can't catch the whole class (the trigger is a teardown
condition, not one flag), but a layered guard catches most:

- **Env linter:** refuse to launch any TP>1 serve with `CCL_TOPO_P2P_ACCESS=1` (kills the one
  deterministic trigger).
- **Serialize TP>1 starts:** never chain two TP>1 launches without a health-confirmed teardown (or
  reset) in between; encode in the lease.
- **Auto-reset-on-DEVICE_LOST:** grep serve logs for `DEVICE_LOST`/`OUT_OF_RESOURCES`/hang and run
  the xe reset (needs #2) before the next TP>1 start.
- **Graceful drain, longer capture budget:** stop SIGKILLing TP>1 workers mid-capture; give capture
  a longer budget so a slow-but-fine capture is not mistaken for a failure and force-killed.

### 4. Kernel 7.1 -- partially relevant, NOT a proven fix
The Linux 7.1 Xe pull request adds purgeable buffer objects (madvise `DONTNEED`/`PURGED`) for
coping with vRAM pressure / OOM. That maps onto ONE symptom we saw -- today's single-card retry
failed with `OUT_OF_RESOURCES` (err 40), consistent with the crashed serve's `UTIL=0.90`
allocation never being reclaimed. So 7.1 could plausibly help the OOM-flavored failure mode and
general memory-pressure robustness.

But the CORE wedge is `DEVICE_LOST` (err 20) -- a oneCCL/L0 collective-context corruption, not a
vRAM-pressure problem -- and the purgeable-BO change does not obviously touch that path. Verdict:
worth trying as a cheap experiment (we are on 7.0.0-22; 7.1 is the next merge window), but do NOT
bet the campaign on it fixing the collective-state corruption.
Ref: https://www.phoronix.com/news/Intel-Xe-Purgeable-BO

## Status of the actual research result (unchanged)
The J.14 EAGER win stands: push-ar +48-64% throughput, 3.35x TTFT vs oneCCL, live and coherent on
a 27B-W8A8 TP=2 serve. Only the capture-gated PRODUCTION variant (GRAPH=1, prefill-only push) is
still unmeasured -- blocked on a clean box, not on a code defect.

## Recommended order after reboot
1. Reset (above) -> confirm health with a single-card matmul probe BEFORE any TP>1 start.
2. Decide on #2 (scoped sudo) and #3 (guard) -- both unblock the loop.
3. Re-attempt the capture-gated A/B WITH the guard + a longer capture budget, so a slow capture is
   not force-killed and any crash self-recovers.
