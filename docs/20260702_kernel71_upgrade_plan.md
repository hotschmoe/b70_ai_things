# HOST-UPGRADE PLAN -- cure the dual-B70 TP=2 wedge (kernel 7.1 + Intel Compute Runtime 26.22.38646.4)

Status: PLAN (nothing installed/executed). Drafted 2026-07-02; pre-flight VERIFIED 2026-07-02 (see section 8).
Decision: install kernel **7.1 GA** (build 202606141628) -- NOT a 7.1.x point release, NOT 7.2 (both reasoned in
section 8). Execute in a GPU-idle maintenance window (this box is the daily driver + the NAS). Cross-ref: vLLM
issue #41663 (our exact hardware), docs/20260625_bcs_wedge_rootcause.md, docs/20260624_devicelost_thoughts.md,
AGENTS.md (GPU Discipline), MIGRATION.md (sections 10/12/13).

## 0. What the box is right now (verified read-only 2026-07-02)

| Fact | Value |
|---|---|
| Host / user | b70s4dayz, hotschmoe (uid 1000) |
| OS | Ubuntu 26.04 LTS |
| Running kernel | 7.0.0-27-generic |
| Kernels installed | 7.0.0-14, 7.0.0-22, 7.0.0-27 (headers 22/27) -- keep as fallback |
| GuC firmware | 70.54.0, PINNED via /etc/modprobe.d/xe-bmg-guc.conf (guc_firmware_path=xe/bmg_guc_70.54.0.bin) |
| linux-firmware | 20260319.git217ca6e4 (ships bmg_guc_70.bin = 70.58.0) |
| Compute Runtime | intel-opencl-icd / libze-intel-gpu1 = 26.05.37020.3 |
| Level-Zero loader | libze1 = 1.28.2-2 |
| Board / CPU | ASRock X399 Professional Gaming, AMD Threadripper 1950X (X399, dual-die MCM), BIOS P4.05 |
| B70 topology | card1 0b:00.0 behind pci0000:00; card0 44:00.0 behind pci0000:40 -- DIFFERENT dies / root complexes |
| Secure Boot | disabled, Platform in Setup Mode (unsigned mainline kernel will boot) |
| Boot | UEFI; /boot on / (nvme0n1p2), ~275 GB free |
| GRUB | GRUB_DEFAULT=0, GRUB_TIMEOUT=0, GRUB_TIMEOUT_STYLE=hidden -- menu invisible at boot (fix in step 4.2) |
| Recent event | Jul 02 00:05:04 44:00.0 GT0 Engine reset engine_class=ccs -- a CCS reset fired TODAY even with GuC 70.54.0 pinned |

## 1. Why kernel 7.1 (upstream #41663, load-bearing quotes)

- F3zz1k, 2026-06-17: "driver fixes were added in 7.1, won't work without them." -> kernel 7.1 mandatory.
- F3zz1k, 2026-05-29 (our EXACT CPU, 1950X): same-die TP=2 crashes on cards sharing one PCIe root complex;
  DIFFERENT root complexes = "101 t/s @ N=32, 0 errors". Working config: "Linux >= 7.1 with xe driver,
  intel-compute-runtime 26.18.38308.1."
- mjsabby, 2026-06-28: "Intel B70 P2P Works! It does require using AM4+ motherboard (could not get it to
  work on the Intel motherboard) and of course, Linux 7.1 and Intel Compute Runtime 26.22.38646.4."

Two DIFFERENT wins -- keep them separate:
1. Wedge cure / stable TP=2 -- proven on 1950X + kernel >= 7.1 with cross-root-complex cards, WITHOUT GPU P2P
   (host-routed collective). This is what our box needs; our cards ARE cross-root-complex (the favorable case).
2. True GPU P2P bandwidth -- mjsabby got it only on AM4+, explicitly NOT on an Intel board. Our board is X399
   (sTR4), untested by anyone in the thread. Expect wedge cured + stable TP=2; do NOT assume P2P uplift on X399.

Repo facts this must respect: reboot is the ONLY recovery on this box (xe drives the display, so
`modprobe -r xe` / xe-reset cannot help); the 70.54.0 pin is a 7.0-era workaround, not a cure (today's CCS
reset proves it); keep P2P=0 (CCL_TOPO_P2P_ACCESS=0) for first validation.

## 2. Exact packages

### 2a. Kernel 7.1 mainline (unsigned; fine with Secure Boot off)
From https://kernel.ubuntu.com/mainline/v7.1/amd64/ (build 202606141628):
- linux-headers-7.1.0-070100_7.1.0-070100.202606141628_all.deb
- linux-headers-7.1.0-070100-generic_7.1.0-070100.202606141628_amd64.deb
- linux-image-unsigned-7.1.0-070100-generic_7.1.0-070100.202606141628_amd64.deb
- linux-modules-7.1.0-070100-generic_7.1.0-070100.202606141628_amd64.deb
- CHECKSUMS (+ CHECKSUMS.gpg)
In-tree xe; no DKMS/out-of-tree modules on this box, so no rebuild concerns.

### 2b. Intel Compute Runtime 26.22.38646.4
From https://github.com/intel/compute-runtime/releases/tag/26.22.38646.4 (ships Level-Zero spec 1.15.31,
loader 1.28.6):
- intel-opencl-icd_26.22.38646.4-0_amd64.deb
- libze-intel-gpu1_26.22.38646.4-0_amd64.deb
- intel-ocloc_26.22.38646.4-0_amd64.deb
- libigdgmm12_22.10.0_amd64.deb

**MANDATORY companion: Intel Graphics Compiler (IGC) 2.x** -- 26.22 DEPENDS on `intel-igc-opencl-2 (>= 2.36.3, <<
2.36.3+~)` and `intel-igc-core-2 (same)`, which are NOT on this box (it has the older IGC: `libigc2`/`libigdfcl2`
2.28.4, plus IGC-1.x `libigc1`/`libigdfcl1`). The new runtime uses the RENAMED `intel-igc-*-2` packages. Get the
matching build from https://github.com/intel/intel-graphics-compiler/releases/tag/v2.36.3:
- intel-igc-core-2_2.36.3+21719_amd64.deb    (filename says +21719; the .deb's internal Version is `2.36.3`,
- intel-igc-opencl-2_2.36.3+21719_amd64.deb   which is what satisfies the `>= 2.36.3, << 2.36.3+~` pin)

Verified 2026-07-02 via `dpkg-deb -f`: internal Version = 2.36.3 (NOT 2.36.3+21719; the +build is filename-only,
and `dpkg --compare-versions 2.36.3+21719 lt 2.36.3+~` is FALSE, so do not go by the filename). Upgrades
intel-opencl-icd + libze-intel-gpu1 from 26.05.37020.3 -> 26.22.38646.4. Does not bundle the L0 loader (libze1);
box has 1.28.2, release wants 1.28.6 (optional bump, see Unknowns).

GOTCHA (hit 2026-07-02): installing the 4 runtime debs WITHOUT the 2 IGC debs fails the IGC dependency, and the
`|| sudo apt -f install` fallback then REMOVES intel-opencl-icd + libze-intel-gpu1 to satisfy apt (leaving the box
with no Intel OpenCL/L0 userspace). NEVER run `apt -f install` here -- put all 6 debs in one dir and
`sudo dpkg -i ./*.deb` so they resolve among themselves.

### 2c. Firmware
Remove the 70.54.0 pin, refresh linux-firmware (apt), let 7.1 load its wanted GuC, then verify the
"using vs wanted" dmesg line matches. Do NOT carry the 70.54.0 pin onto 7.1.

## 3. Pre-upgrade: bullet-proof rollback (HEADLESS box; decision 2026-07-02)

The box runs HEADLESS (no monitor normally attached). Decision: 7.1 is the PERMANENT default; rollback is manual
(attach a monitor + keyboard, pick 7.0 at the GRUB menu). Two facts make this safe:
- `GRUB_DEFAULT=0` boots the top-level "Ubuntu" entry, which Ubuntu always aims at the NEWEST installed kernel.
  Once 7.1 is installed it becomes the default automatically -- NO `grub-reboot` / `grub-set-default` needed.
- GRUB draws on the UEFI framebuffer BEFORE xe loads, so even a broken 7.1 xe still shows the menu when a
  monitor is attached. Keep `GRUB_TIMEOUT_STYLE=menu` + `GRUB_TIMEOUT=15` so the menu is catchable.

HEADLESS HARDENING: also set `GRUB_RECORDFAIL_TIMEOUT=15`. A failed boot sets `recordfail=1` (already set in this
box's grubenv); without a finite recordfail timeout GRUB can wait FOREVER at the menu for a keypress that never
comes on a headless box. A finite value counts down and proceeds. (It will re-attempt the 7.1 default after a
failed boot, which is correct for "default 7.1 + manual rollback"; if you ever want zero-touch auto-fallback to
7.0 instead, use the display-attached one-shot in 4.5-ALT.)

```bash
# 3.1 back up state
sudo cp /etc/default/grub /etc/default/grub.pre-7.1
mkdir -p ~/b70-upgrade-2026-07-02 && cd ~/b70-upgrade-2026-07-02
uname -r > running-kernel.txt
dpkg -l | grep -E 'linux-image|linux-modules|intel-opencl|libze|libigdgmm|linux-firmware' > pkgs-before.txt
lspci -nnk > lspci-before.txt; ls /sys/class/drm > drm-before.txt
sudo dmesg | grep -iE 'guc|huc|xe ' > dmesg-xe-before.txt
cp /etc/modprobe.d/xe-bmg-guc.conf ./xe-bmg-guc.conf.bak

# 3.2 UNHIDE the GRUB menu + headless hardening (edit /etc/default/grub):
#   GRUB_TIMEOUT_STYLE=menu
#   GRUB_TIMEOUT=15
#   GRUB_RECORDFAIL_TIMEOUT=15   # headless: don't hang forever at the menu after a failed boot
#   GRUB_DEFAULT=0               # keep 0 -> boots NEWEST kernel = 7.1 once installed (no grub-set-default)
sudoedit /etc/default/grub
sudo update-grub

# 3.3 record the exact known-good 7.0.0-27 menu entry id
grep -E "submenu|menuentry '" /boot/grub/grub.cfg | grep -n "7.0.0-27"
```
Keep 7.0.0-27 and 7.0.0-22 installed. Do NOT remove them.

## 4. Install (all sudo; nothing here touches the GPU -- but STOP the sglang container first)

```bash
cd ~/b70-upgrade-2026-07-02
# 4.1 kernel 7.1
BASE=https://kernel.ubuntu.com/mainline/v7.1/amd64
for f in \
  linux-headers-7.1.0-070100_7.1.0-070100.202606141628_all.deb \
  linux-headers-7.1.0-070100-generic_7.1.0-070100.202606141628_amd64.deb \
  linux-image-unsigned-7.1.0-070100-generic_7.1.0-070100.202606141628_amd64.deb \
  linux-modules-7.1.0-070100-generic_7.1.0-070100.202606141628_amd64.deb \
  CHECKSUMS ; do wget -q "$BASE/$f"; done
sha256sum -c CHECKSUMS 2>/dev/null | grep -E '7.1.0-070100'   # all OK
sudo apt install ./linux-*7.1.0-070100*.deb                   # runs update-grub

# 4.2 refresh firmware
sudo apt update && sudo apt install --only-upgrade linux-firmware

# 4.3 Intel Compute Runtime 26.22.38646.4 + its MANDATORY IGC 2.36.3 companion
ICR=https://github.com/intel/compute-runtime/releases/download/26.22.38646.4
IGC=https://github.com/intel/intel-graphics-compiler/releases/download/v2.36.3
mkdir icr && cd icr
for f in intel-opencl-icd_26.22.38646.4-0_amd64.deb \
         libze-intel-gpu1_26.22.38646.4-0_amd64.deb \
         intel-ocloc_26.22.38646.4-0_amd64.deb \
         libigdgmm12_22.10.0_amd64.deb ; do wget -q "$ICR/$f"; done
for f in "intel-igc-core-2_2.36.3+21719_amd64.deb" \
         "intel-igc-opencl-2_2.36.3+21719_amd64.deb" ; do wget -q "$IGC/$f" -O "$f"; done
# install ALL 6 together so the IGC<->runtime deps resolve among themselves.
# DO NOT append "|| sudo apt -f install" -- on any dep miss it REMOVES the runtime (see 2b GOTCHA).
sudo dpkg -i ./*.deb
sudo apt-mark hold intel-opencl-icd libze-intel-gpu1 intel-ocloc libigdgmm12 intel-igc-core-2 intel-igc-opencl-2
cd ..

# 4.4 remove the 7.0-era GuC pin, rebuild initramfs
sudo rm /etc/modprobe.d/xe-bmg-guc.conf
sudo update-initramfs -u -k all

# 4.5 boot into 7.1 as the PERMANENT default (HEADLESS: GRUB_DEFAULT=0 already boots newest = 7.1)
dpkg -l | grep 7.1.0-070100          # confirm image + modules + headers all installed
sudo update-grub                     # regenerate; 7.1 is now newest => top-level "Ubuntu" boots it
sync
sudo reboot
```

### 4.5-ALT. Display-attached ONLY (zero-touch auto-fallback; NOT used for this headless box)
If a monitor were permanently attached and you wanted the box to auto-return to 7.0 on a failed 7.1 boot without
any manual step, boot 7.1 ONCE and leave the default at 7.0:
```bash
sudo grub-reboot "Advanced options for Ubuntu>Ubuntu, with Linux 7.1.0-070100-generic"
sync && sudo reboot
```
We are NOT doing this -- the box is headless and 7.1 is the chosen permanent default.

## 5. Post-upgrade verification (in order; stop at first failure -> roll back)

Every GPU touch under the lease. Keep a second terminal on `sudo dmesg -w` watching for
Engine reset / bcs / ccs / DEVICE_LOST.

```bash
uname -r                                  # 7.1.0-070100-generic
ls /sys/class/drm | grep card             # card0 card1
lspci -nnk | grep -A3 e223                # both B70, driver: xe
clinfo -l                                 # two B70 devices; clinfo shows Driver Version 26.22.38646.4
# NOTE: sycl-ls is NOT on the bare host (ships with oneAPI, which lives inside the serve container).
# Host enumeration = clinfo (OpenCL). Level-Zero gets verified in-container by bin/xpu-health below.
sudo dmesg | grep -iE 'guc|huc'           # "Using GuC firmware ... version X" NO "(wanted Y)"; no taint
cd /mnt/vm_8tb/github/b70_ai_things
IMG=sglang-xpu:mtp bin/xpu-health         # HEALTHY (cards 0 1)
./bin/gpu-run bash scripts/70_run_p2p_probe.sh      # informational (cure does NOT need P2P on X399)
# THE TEST -- TP=2 W8A8 smoke (P2P OFF, default):
./bin/gpu-run bash rdy_to_serve/sglang/qwen36-27b-w8a8/serve.sh smoke
# SOAK -- old wedge hit within ~3 serves; fix bar = 5 clean cycles, ZERO BCS reset:
for i in 1 2 3 4 5; do echo "=== cycle $i ==="; \
  ./bin/gpu-run bash rdy_to_serve/sglang/qwen36-27b-w8a8/serve.sh smoke || { echo "FAIL $i"; break; }; done
sudo dmesg | grep -iE 'bcs|ccs|Engine reset|DEVICE_LOST|Timedout' | tail    # expect EMPTY
bin/serve-sweep --smoke                   # shelf-wide gate
```
Only after 5+ clean cycles, optionally A/B GPU P2P as a SEPARATE experiment (may not help on X399), with a
REBOOT between every attempt: `I_KNOW_P2P_WEDGES=1 CCL_TOPO_P2P_ACCESS=1 ./bin/gpu-run bash .../serve.sh smoke`.

## 6. Recovery

- 7.1 boots but SSH is alive (e.g. only the GPU is broken): roll back over the network, NO monitor needed --
  `sudo apt remove 'linux-*7.1.0-070100*' && sudo update-grub && sudo reboot` (newest is 7.0.0-27 again), or
  revert ICR/firmware per below and stay on 7.1. This is the most likely failure mode.
- 7.1 hangs / no network (headless, can't SSH in): attach a monitor + keyboard, power-cycle, and at the GRUB
  menu (visible 15s) pick Advanced options for Ubuntu > Ubuntu, with Linux 7.0.0-27-generic. Then purge 7.1 as
  above so the next unattended boot defaults to 7.0.
- Boots 7.1 but GPUs misbehave: revert ICR + firmware:
  `sudo apt-mark unhold intel-opencl-icd libze-intel-gpu1 intel-ocloc libigdgmm12;`
  `sudo apt install --reinstall --allow-downgrades intel-opencl-icd=26.05.37020.3-1 libze-intel-gpu1=26.05.37020.3-1`
  Restore the 7.0 GuC pin only if returning to a 7.0 kernel.
- Permanent 7.0: `sudo grub-set-default "...7.0.0-27-generic"` (needs GRUB_DEFAULT=saved) + update-grub.
- Purge 7.1: `sudo apt remove 'linux-*7.1.0-070100*' && sudo update-grub`.
- Mid-serve TP=2 wedge still = reboot; do not chain crash-prone TP=2 starts.

## 7. Unknowns / assumptions

1. Board mismatch is the biggest caveat: "P2P Works!" is AM4+, explicitly failed on an Intel board; our X399
   is untested for FULL P2P. But the WEDGE CURE (stable TP=2) is separately confirmed on a 1950X with
   cross-root-complex cards on kernel >= 7.1. Success criterion = 5 clean TP=2 cycles, NOT a P2P bandwidth number.
2. Our cards are on different dies/root complexes (pci0000:00 vs pci0000:40) = F3zz1k's WORKING case. Re-confirm
   with lspci after upgrade.
3. GuC version 7.1 wants is unknown yet; plan handles it by removing the pin + refreshing linux-firmware, then
   verifying the dmesg using/wanted line. If 7.1 wants newer than shipped, drop that bmg_guc_XX.bin in and
   rebuild initramfs. Do NOT re-pin 70.54.0 on 7.1.
4. ICR-on-26.04 regression risk: intel/compute-runtime #922 (Xe2/BMG multi-rank abort after CR upgrade on
   Ubuntu 26.04). mjsabby's success pairs 26.22.38646.4 WITH kernel 7.1, so 7.1 is believed to resolve it, but
   the debs are validated on 24.04. If clinfo/sycl-ls/xpu-health fail after the ICR bump, revert ICR and retest
   on 7.1 with stock 26.05 to isolate kernel-vs-runtime.
5. L0 loader: box has libze1 1.28.2; release targets 1.28.6. Likely compatible; bump from intel/level-zero
   releases only if L0 init errors appear.
6. Version drift: re-list both directories at execution time in case a 7.1.x point build or newer ICR appears.
7. Serve env (out of scope): F3zz1k's working vLLM env (SYCL_UR_USE_LEVEL_ZERO_V2=0, CCL_ALLREDUCE=ring,
   CCL_ENABLE_SYCL_KERNELS=1, UR_L0_USE_IMMEDIATE_COMMANDLISTS=0) is llm-scaler-specific; our sglang serve.sh
   does not set these. If TP=2 is stable but flaky/slow after the host upgrade, that env stack is the next lever.

Sources: vLLM #41663; kernel.ubuntu.com/mainline/v7.1/amd64; github.com/intel/compute-runtime releases
26.22.38646.4 and issue #922; repo AGENTS.md, MIGRATION.md, docs/20260625_bcs_wedge_rootcause.md,
docs/20260624_devicelost_thoughts.md.

## 8. Pre-flight verification + kernel-version decision (2026-07-02, read-only)

All checks GO. The command list in sections 3-5 is unchanged; run as written with 7.1 GA.

### 8a. Downloads live + no drift
- Kernel 7.1 GA debs (build 202606141628): all 4 + CHECKSUMS return live (wget --spider OK).
- ICR 26.22.38646.4 debs: all HTTP 200; `intel/compute-runtime` latest release tag IS still 26.22.38646.4 (no newer runtime to chase).
- Disk: 275 GB free on / (holds /boot; no separate /boot partition). Ample.

### 8b. Topology re-confirmed = F3zz1k's WORKING case
Both B70 bound to `xe`. Cards are on DIFFERENT root complexes / dies:
- card1 `0000:0b:00.0` -> root `pci0000:00` (via 00:03.1)
- card0 `0000:44:00.0` -> root `pci0000:40` (via 40:03.1)
`numa_node=-1` on both (board/kernel does not expose PCI NUMA; the root-complex split is the signal, and it holds).
This is the cross-root-complex case F3zz1k confirmed stable for TP=2 on a 1950X + kernel >= 7.1.

### 8c. Why 7.1 GA and NOT 7.1.1/7.1.2 (verified against the actual CHANGES)
Point releases are pure stable backports; read in full:
- 7.1.1: only drm change is `drm/amdgpu` (AMD). Rest = arm64 errata, driver-core, HID, fs. ZERO Intel xe.
- 7.1.2: io_uring, ksmbd, fuse, virtiofs, agp, media, iio, serial, nfsd. ZERO Intel xe, ZERO drm/i915.
=> A 7.1.x point release buys the B70 nothing over 7.1 GA (wedge fix already in 7.1 GA; no GPU perf content).
Install 7.1 GA -- the exact build the upstream reporters validated.

### 8d. 7.2 is where B70 work lands -- but only rc1 exists; defer to 7.2 GA
7.2-rc1 is out (238 drm/xe commits). Battlemage-relevant items to re-evaluate AFTER 7.2 GA:
- `drm/xe: Set GT rp min frequency as 1.2GHz default for BMG/CRI` -- the one direct BMG perf change; raises the
  min GPU clock floor, targets the decode/light-load downclock (and possibly the display-attached card1 downclock).
- BCS copy-engine changes (`Stop programming BLIT_CCTL on Xe2`, Xe2-blitter-instructions feature flag,
  `Mark BCS engines as belonging to the GT forcewake domain`) -- touch the exact engine that wedges us; possible
  extra wedge insurance.
- `Restore IDLEDLY register on engine reset` (our CCS-reset path); L3-bank MCR steering on Xe2; OA/eustall profiling.
Do NOT run an rc kernel on this daily-driver + NAS box; the wedge-cure evidence base is all on 7.1.

### 8e. Reframe -- the real compute-perf lever is the runtime, not the kernel
Kernel does scheduling/power/memory, not matmul codegen. INT8-XMX throughput lives in the compute-runtime/oneDNN/
Level-Zero layer, which this plan already bumps (ICR 26.05.37020.3 -> 26.22.38646.4, Level-Zero spec 1.15.31).
That runtime bump, not any 7.1->7.2 kernel delta, is what can move compute perf here.
