# TP=2 hardware wedge ROOT CAUSE: BCS (copy engine) kernel-job timeout on xe/GuC (2026-06-25)

> **RESOLVED 2026-07-02 by the host upgrade to kernel 7.1 + ICR 26.22.38646.4 (docs/20260702_kernel71_upgrade_plan.md).**
> On 7.1 both cards use GuC **70.58.0 with NO "(wanted X)"** skew (7.1's KMD wants 70.58.0), so the 70.54.0 pin
> below is RETIRED (removed; do NOT re-add on 7.1). 5/5 back-to-back TP=2 W8A8 fused+MTP serve cycles ran clean
> with ZERO Engine reset/DEVICE_LOST/Timedout/bcs. This is the real cure vs the 70.54 pin workaround (which was
> "rare, not impossible"). One SUSTAINED concurrent-decode soak still owed before formally lifting the
> "w8a8 TP=2 = attended-only" rule. History below is the 7.0-era diagnosis; kept for the record.

Status: ROOT-CAUSED **and FIX CONFIRMED** (2026-06-25). The wedge was the "device_lost / cumulative-TP2 wedge".
This is a DRIVER/FIRMWARE bug in the Intel xe + GuC stack on Battlemage B70 -- NOT a vLLM bug.

## FIX CONFIRMED: downgrade GuC firmware to 70.54.0 (match the KMD)

Operator placed `xe/bmg_guc_70.54.0.bin` and rebooted; dmesg now: `GT0/GT1 (both cards): Using GuC firmware
from xe/bmg_guc_70.54.0.bin version 70.54.0`. Result: a single TP=2 batch then ran **5 serve cycles back to back
over ~1 hour with ZERO BCS Timedout-job / device_lost / card hang** (campaign 120 run 3: B_none, E, F2_recycle,
F1_imm0, F1_cleanup). Before the downgrade (GuC 70.58.0) the box wedged within ~3 serves and even on serve 1.
=> The GuC firmware<->KMD skew (70.58.0 running vs 70.54.0 wanted) WAS the BCS-timeout cause. **Recommended box
config: pin GuC 70.54.0** until a kernel/xe that wants 70.58.0 is validated. (The remaining MTP+PIECEWISE-graph
slowdown/crash is the SEPARATE software issue; cudagraph=NONE avoids it.)

Forensics captured live (before reboot): `/mnt/vm_8tb/b70/wedge-capture/20260625_214449_grab/`
(devcoredump.*, dmesg.full.txt, dmesg.important.txt, journal-kernel.txt, debugfs-snapshot.txt, lspci-b70.txt,
xe-params.txt). Capture helper: `/mnt/vm_8tb/b70/wedge-capture/grab.sh` (run with sudo before reboot).

## The smoking gun

xe devcoredump (`/sys/class/drm/card1/device/devcoredump/data`):
```
**** Xe Device Coredump ****
Reason: Timedout job - seqno=3738, lrc_seqno=3738, guc_id=0, flags=0x73
GuC firmware: xe/bmg_guc_70.bin   GuC version: 70.58.0 (wanted 70.54.0)
**** Contexts ****  GuC ID: 0  Name: bcs0
**** HW Engines ****  bcs8 (physical), logical instance=1
    RING_HEAD 0x03401b0c  RING_TAIL 0x00001b68  ACTHD 0x...03401b0c  (ring stuck: HEAD != TAIL)
```

dmesg (`Tile0: GT0`, PCI `0000:0b:00.0`):
```
xe: Tile0: GT0: Timedout job: seqno=5784/5854/5924/6790... guc_id=0 flags=0x73 in no process [-1]
xe: Tile0: GT0: Kernel-submitted job timed out
WARNING: drivers/gpu/drm/xe/xe_guc_submit.c:1594 guc_exec_queue_timedout_job
xe: Tile0: GT0: trying reset ... reset queued/started/done
xe: [drm] exec queue reset detected   (repeats -> reset does NOT clear -> card wedged)
```

## What it means

- The hung engine is **bcs0 / bcs8 = the Blitter / COPY engine (BCS)** on GT0.
- The job is **kernel-submitted** (`in no process [-1]`, "Kernel-submitted job timed out") -> it is **xe's own
  VRAM clear-on-alloc / migration / eviction copy**, NOT a userspace SYCL kernel or a oneCCL collective. It fires
  during the VRAM-heavy phase of a TP=2 serve (model load of the 33 GiB checkpoint into VRAM at UTIL=0.90, KV
  cache alloc, warmup) -- which is why the wedge hits at model-load `mrope _compute_cos_sin_cache` (first GPU op)
  or the first decode step.
- A **storm** of these timeouts cascades; each GT0 reset "completes" but does not actually recover the engine
  ("exec queue reset detected" loops), so the card stays hung until a full **reboot**.
- It is STOCHASTIC per serve (hit the 1st serve of a fresh boot and the 3rd of a session) because it depends on
  the exact migration/clear copy pattern, not a deterministic op count.
- This is the SAME class as vLLM #41663 (dual Arc Pro B70, TP=2, xe BCS engine reset). It is DISTINCT from the
  software MTP+PIECEWISE-graph NEO crash (that leaves GPUs healthy; this hangs a card).

## Leading hypothesis: GuC firmware <-> KMD skew

The coredump says GuC **70.58.0 (wanted 70.54.0)**: Ubuntu linux-firmware (dated Apr 15) ships
`/lib/firmware/xe/bmg_guc_70.bin.zst` = 70.58.0, but the kernel `7.0.0-22-generic` xe driver was validated for
70.54.0. A newer GuC than the KMD expects can carry BCS-scheduling behavior the KMD mishandles. This is the
prime suspect for the BCS timeout/reset-recovery failure.

## Candidate fixes / mitigations (ranked; firmware/kernel = operator's device_lost workstream)

1. **Align GuC firmware <-> xe KMD** (most likely real fix):
   - Pin/downgrade GuC to the KMD's wanted 70.54.0: place `bmg_guc_70.bin` (70.54.0) and set
     `xe.guc_firmware_path=xe/<file>` (the `guc_firmware_path` module param EXISTS, currently empty), or pin the
     linux-firmware package version, then `update-initramfs` + reboot.
   - OR move to a kernel/xe KMD that wants 70.58.0 (newer mainline kernel, or Intel out-of-tree xe), so firmware
     and driver match.
2. **Reduce BCS migration/clear pressure (vLLM-side mitigation, test first -- cheap):** lower
   `gpu_memory_utilization` (UTIL=0.80 / 0.70) so the model-load + KV alloc do not push VRAM into
   eviction/migration. Reduces the number of kernel BCS clear/migrate copies -> may cut wedge probability (NOT a
   guaranteed fix; the load copies still happen). Also try fewer concurrent allocations (serialize load).
3. **xe diagnostic knobs** (cleaner first-failure evidence, not fixes): boot with
   `drm.debug=0x1ff log_buf_len=128M xe.guc_log_level=5 xe.wedged_mode=2`.
4. **PCIe stability:** test `pcie_aspm=off` (lspci AER deltas in the capture will say if PCIe is involved).

## Next-step bisect to confirm (codex plan; non-vLLM minimal repro)

Confirm it is the kernel BCS copy path, independent of vLLM:
1. TP=1 matmul loop on card0, then card1 (should be clean -> it is a TP=2/migration thing).
2. A VRAM-pressure repro WITHOUT vLLM: allocate near-full VRAM + force clears/migrations in a loop on one card,
   watch for the bcs0 timeout. If it wedges without vLLM/oneCCL, it is purely xe/GuC.
3. Pure mp.spawn(2) oneCCL all_reduce loop (no model) -- separates collective copies from migration copies.
Full armed-instrumentation + bisect + L0 LD_PRELOAD shim + Sysman watcher plan: see the codex consult archived in
JOURNAL (2026-06-25) and `/mnt/vm_8tb/b70/wedge-capture/` (codex_wedge_trace).

## Impact on the W8A8-27B MTP campaign

Item 1 (cudagraph=NONE promotion, 2x stable) is SHIPPED and unaffected. Items 2-4 (coherent NONE number, E
re-soak, Tier F) are GATED by this wedge: every TP=2 serve now risks an immediate BCS-timeout wedge + reboot, so
they cannot complete reliably until the firmware/kernel is fixed. All Items 2-4 tooling is staged (one command
away once TP=2 is stable). Recommend pausing the Items 2-4 GPU grind until the GuC firmware/KMD skew is resolved.
