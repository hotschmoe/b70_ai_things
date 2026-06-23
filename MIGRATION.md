# MIGRATION.md -- Unraid -> Ubuntu 26.04 on the B70 host (192.168.10.5)

Status: plan (2026-06-22). Goal: move the GPU host off Unraid (kernel 6.18) to **Ubuntu 26.04 LTS
(kernel 7.0)** to get the `xe` B70 stack on a current kernel (and the option to chase GPU P2P, see
[docs/P2P_GPU.md](docs/P2P_GPU.md) / [MOONSHOT_RESEARCH.md](MOONSHOT_RESEARCH.md)), while carrying the
existing data drives over intact. The whole plan is built around ONE principle:

  **The Ubuntu installer must never see the data drives, and we read them RO before we ever write.**

Pre-installing Ubuntu on a separate machine (the user's plan) satisfies the first half for free.

Keep/drop decisions (from Isaac):
- KEEP: OpenWebUI; NAS file sharing (Samba + NFS, LAN-only); the data on disk1/disk2/cache/vm_8tb.
- REBUILD: Nextcloud (just serving mother's data) -- remake fresh or pick a lighter method.
- DROP: all VMs (libvirt/domains), Syncthing. Generic fresh install, no special users/permissions.
- REDUNDANCY: SnapRAID (parity disk reused) -- accepted in place of Unraid's real-time parity.


## 0. Disk inventory / snapshot (captured 2026-06-22, BEFORE any change)

All three 10TB are the SAME model (WDC WD100EMAZ) -- **identify them by SERIAL, not by /dev/sdX**
(letters reshuffle when the USB is removed and the new NVMe is added).

| role        | dev now   | size  | model               | serial               | fs    | UUID                                   |
|-------------|-----------|-------|---------------------|----------------------|-------|----------------------------------------|
| Unraid boot | sda1      | 28.6G | SanDisk Ultra Fit   | 4C531001410813122563 | vfat  | 2736-60C3 (LABEL UNRAID)               |
| disk1 DATA  | sdc1      | 9.1T  | WDC WD100EMAZ       | JEHA1MZN             | xfs   | 5fee9294-79d7-4b69-90fc-9f89f4f58077   |
| disk2 DATA  | sde1      | 9.1T  | WDC WD100EMAZ       | JEHA1LNN             | xfs   | 03cfc09b-d410-40ce-986f-bee573ed4628   |
| PARITY      | sdb1      | 9.1T  | WDC WD100EMAZ       | JEH9VZHN             | none  | (Unraid XOR parity - NOT a filesystem) |
| vm_8tb      | sdd1      | 7.3T  | Samsung 870 QVO 8TB | S5SSNF0WA05872J      | btrfs | 65f898c0-c921-4719-80b8-ee3df6654239   |
| cache       | nvme0n1p1 | 954G  | SPCC M.2 PCIe SSD   | A0D6079502A700037914 | btrfs | 914dca8a-4357-4727-abba-fca7df597347   |
| NEW boot    | (incoming)| 500G  | (the spare NVMe)    | -                    | ext4  | (assigned at install)                  |

Notes:
- **PARITY (sdb, serial JEH9VZHN) has NO filesystem** -- it is pure Unraid XOR parity, useless off
  Unraid. This is the disk we reformat and reuse as the SnapRAID parity drive. Everything else is
  carried over untouched.
- The Unraid array disks store standard XFS on the raw partition (sdc1 / sde1). On Ubuntu we mount the
  partition DIRECTLY -- there is no Unraid `md`/parity layer to peel off, the filesystem is right there.
- After adding the new NVMe there will be TWO nvme devices; enumeration order is NOT stable. The 500G
  boot vs 954G cache are different sizes, but **fstab MUST use UUID**, never nvme0/nvme1.


## 1. Does "install Ubuntu elsewhere, then move the disk" work? (the user's question)

Yes -- Linux is not Windows; a root filesystem moved between machines boots fine because it mounts root
by UUID. The ONE catch is the **UEFI boot entry**, which lives in the motherboard's NVRAM, not on the
disk:

- The Ubuntu installer on machine A writes a boot entry to **machine A's** firmware and installs GRUB to
  the disk's EFI System Partition (ESP) at `\EFI\ubuntu\shimx64.efi`. The B70 box's firmware has no such
  entry, so it may not auto-find the bootloader.
- Three ways to fix, easiest first:
  1. **Populate the UEFI fallback path** (do this on machine A right after install, before unplugging):
     copy the ESP's `\EFI\ubuntu\shimx64.efi` -> `\EFI\BOOT\BOOTX64.EFI` (and `grubx64.efi` alongside).
     Any UEFI auto-boots `\EFI\BOOT\BOOTX64.EFI` from a disk with no NVRAM entry. Keeping `shimx64.efi`
     preserves Secure Boot; otherwise just disable Secure Boot on the B70 box.
  2. **Pick it manually** in the B70 box's one-time boot menu (F8/F11/F12 on the X399 board): select the
     NVMe / "ubuntu" EFI file once, then run `sudo grub-install && sudo update-grub` to register it.
  3. **Boot a live USB** on the B70 box and `grub-install` to the NVMe's ESP from there.
- Make sure both machines are in **UEFI** mode (not legacy/CSM) so the ESP approach is consistent.

If the move-the-disk dance is annoying, the alternative is to **install directly on the B70 box** with
the Ubuntu USB -- but then you MUST protect the data drives (physically unplug them, or very carefully
select only the 500G NVMe as the target). Pre-installing elsewhere trades a small boot-entry chore for
zero risk of the installer wiping the array. Recommended: pre-install elsewhere.


## 2. Safety model / reversibility

- **KEEP THE UNRAID USB STICK UNTOUCHED.** Rollback = unplug the Ubuntu NVMe (or just pick the USB in the
  boot menu) and you are back on Unraid. The migration is fully reversible until we reformat the parity
  disk or write to the data disks from Ubuntu.
- **Do not reformat ANY data drive** (sdc1/sde1/sdd1/nvme cache). Only the new 500G NVMe (install) and
  later the parity disk sdb (SnapRAID) get formatted.
- **Mount data RO first**, verify contents, only then switch to RW.
- Caveat once you go RW on the array from Ubuntu: Unraid's parity (sdb) goes stale, so a rollback to
  Unraid would trigger a parity rebuild (recoverable -- parity is rebuilt from data, no data loss).


## 3. Phase 0 -- capture EVERYTHING before touching hardware (do now, on Unraid)

- This inventory (section 0) -- done. Re-verify with `lsblk -f` and `blkid` on the day.
- Samba shares to recreate: currently flash/StrongSync/appdata/domains/isos/proxmox-vms (drop domains +
  proxmox-vms VM shares; keep what mother/clients use). NFS exports: StrongSync, isos, proxmox-vms.
- Docker: capture the `docker run`/compose for each container we keep (OpenWebUI; the vLLM-xpu serve
  command lives in `docs/SERVING.md`). Save Nextcloud's data path + DB so mother's data survives the
  rebuild (her files live under the array share; the Nextcloud *app+DB* is what we recreate).
- Note free M.2 slot on the X399 board for the new NVMe (the 2x B70 occupy PCIe x16 slots; M.2 is
  separate, should be free). The cache SPCC NVMe is in one M.2 slot already.

Capture from the LIVE Unraid box (read-only) into vm_8tb so it carries over by UUID:
```
D=/mnt/vm_8tb/migration_capture/2026-06-24; mkdir -p "$D"
lsblk -f > "$D/lsblk-f.txt"; blkid > "$D/blkid.txt"
ls -la /dev/disk/by-id > "$D/by-id.txt"; ls -la /dev/disk/by-uuid > "$D/by-uuid.txt"
for d in /dev/sd?; do smartctl -a "$d" > "$D/smart-$(basename "$d").txt" 2>&1; done   # drive health baseline
ip a > "$D/ip-a.txt"; ip r > "$D/ip-route.txt"; cp /boot/config/network.cfg "$D/" 2>/dev/null
cp /etc/samba/smb.conf "$D/" 2>/dev/null; cp /boot/config/smb-extra.conf "$D/" 2>/dev/null
cp -r /boot/config/shares "$D/unraid-shares" 2>/dev/null; cat /etc/exports > "$D/exports.txt" 2>/dev/null
docker ps -a > "$D/docker-ps.txt"
for c in $(docker ps -a --format '{{.Names}}'); do docker inspect "$c" > "$D/inspect-$c.json"; done
cp -r /boot/config/plugins/dockerMan/templates-user "$D/docker-templates" 2>/dev/null
crontab -l > "$D/crontab.txt" 2>/dev/null; cp /boot/config/go "$D/go" 2>/dev/null
lspci -tv > "$D/lspci-tv.txt"; lspci -nnk > "$D/lspci-nnk.txt"   # P2P topology baseline (which die/root port each B70 sits behind)
```
Optional (only if you want mother's Nextcloud accounts/share-links, not just her files):
`docker exec <db-container> mysqldump ... > "$D/nextcloud.sql"`. Her actual files live on the array
and carry over regardless; the Unraid USB (kept) also still holds every config above as rollback.

Phase 0 capture -- RESULTS (run 2026-06-24, saved to /mnt/vm_8tb/migration_capture/2026-06-24/, small
configs mirrored to the laptop at ~/b70_migration_capture/2026-06-24/). Findings that shape the rebuild:
- NETWORK IS DHCP, NOT static. Host got 192.168.10.5 via DHCP from the router (192.168.10.1).
  DECISION: keep DHCP on Ubuntu, pin 192.168.10.5 via a DHCP RESERVATION on the router keyed to
  eth0 MAC = 70:85:c2:5d:b3:db. (Unraid bridged eth0->br0 sharing that same MAC; Ubuntu requests
  from eth0 directly with the same MAC, so the reservation carries over.) #1 "do not break mother's
  access" item -- serve recipes + docs hardcode .10.5.
- Disk serials + UUIDs re-verified -- ALL MATCH section 0. The section 6 fstab is correct as written.
  Parity = sdb / serial JEH9VZHN, confirmed NO filesystem -> the SnapRAID reformat target.
- open-webui data lived on the Unraid docker.img loopback (/dev/loop2), NOT on any carried-over
  drive -- it would have been lost. Snapshotted to open-webui_data.tgz (1.1G). Restore on Ubuntu:
  `docker volume create open-webui` then extract the tgz into /var/lib/docker/volumes/open-webui/_data.
- Containers: KEEP open-webui, nextcloud (linuxserver:25.0.2) + mariadb. DROP Syncthing.
- Nextcloud: mother's FILES are at /mnt/user/StrongSync/nextcloud (on the array -> carries over).
  DB lives in mariadb appdata, dumped to all-databases.sql (insurance only). 25.0.2 is old; on a
  fresh rebuild prefer pointing new Nextcloud at her files + `occ files:scan` over restoring the old
  DB across many major versions.
- Samba: only user share to recreate is StrongSync (appdata/system are Unraid-internal; domains +
  proxmox-vms are VM shares -> drop). virbr0 is libvirt's net -> irrelevant on Ubuntu.


## 4. Phase 1 -- prepare the boot NVMe (on a separate machine)

1. Install **Ubuntu 26.04 LTS** to the 500G NVMe. Use the whole 500G NVMe; default ext4 root + ESP is
   fine. Root by UUID (installer default).
2. Right after install, mount the ESP and populate the fallback boot path (section 1, fix #1):
   `sudo cp /boot/efi/EFI/ubuntu/shimx64.efi /boot/efi/EFI/BOOT/BOOTX64.EFI` (mkdir BOOT if needed; copy
   grubx64.efi too). This makes it boot on the B70 box with no NVRAM entry.
3. (Optional, saves a step) pre-install packages while you have internet: `mergerfs snapraid samba
   nfs-kernel-server xfsprogs btrfs-progs docker.io` (or Docker CE), and the Intel GPU userspace (Phase 5).
4. Power off, pull the NVMe.


## 5. Phase 2 -- swap + first boot on the B70 box

1. `ssh root@192.168.10.5` -> stop Docker containers, `docker stop $(docker ps -q)`; stop the Unraid
   array (web UI) so nothing is mid-write; power down.
2. Install the 500G NVMe in the free M.2 slot. Leave ALL data drives connected (the installer is already
   done; nothing will format them). Leave the Unraid USB in (rollback).
3. Power on -> firmware boot menu -> boot the 500G NVMe (or its EFI entry per section 1).
4. You are in Ubuntu. `lsblk -f` should show every drive: the two XFS data partitions, the two btrfs
   volumes, the empty parity disk, the Unraid USB -- all UNMOUNTED. Nothing auto-mounts; nothing is at
   risk yet.


## 6. Phase 3 -- mount the data drives (RO verify -> RW by UUID)

Read-only sanity check first (copy-paste, UUIDs from section 0):
```
sudo mkdir -p /mnt/disk1 /mnt/disk2 /mnt/vm_8tb /mnt/cache
sudo mount -o ro UUID=5fee9294-79d7-4b69-90fc-9f89f4f58077 /mnt/disk1     # xfs, disk1
sudo mount -o ro UUID=03cfc09b-d410-40ce-986f-bee573ed4628 /mnt/disk2     # xfs, disk2
sudo mount -o ro UUID=65f898c0-c921-4719-80b8-ee3df6654239 /mnt/vm_8tb    # btrfs, vm_8tb (b70 repo+models)
sudo mount -o ro UUID=914dca8a-4357-4727-abba-fca7df597347 /mnt/cache     # btrfs, cache
ls /mnt/disk1 /mnt/disk2 /mnt/vm_8tb /mnt/cache    # verify your data is all there
```
When satisfied, remount RW and persist in /etc/fstab BY UUID:
```
# /etc/fstab  (rw, nofail so a missing disk never blocks boot)
UUID=5fee9294-79d7-4b69-90fc-9f89f4f58077  /mnt/disk1   xfs    defaults,noatime,nofail  0 2
UUID=03cfc09b-d410-40ce-986f-bee573ed4628  /mnt/disk2   xfs    defaults,noatime,nofail  0 2
UUID=65f898c0-c921-4719-80b8-ee3df6654239  /mnt/vm_8tb  btrfs  defaults,noatime,nofail  0 0
UUID=914dca8a-4357-4727-abba-fca7df597347  /mnt/cache   btrfs  defaults,noatime,nofail  0 0
```
`sudo systemctl daemon-reload && sudo mount -a`. The b70 repo is now back at `/mnt/vm_8tb/b70`.
(Do NOT add the parity disk sdb yet -- it has no filesystem.)


## 7. Phase 4 -- redundancy (mergerfs pool + SnapRAID), reuse the parity disk

This reproduces the Unraid model (independent disks, one dedicated parity disk, no striping, each disk
independently readable).

1. **Pool** disk1+disk2 into one namespace (replaces Unraid's `/mnt/user` user-share):
   ```
   sudo mkdir -p /mnt/storage
   # /etc/fstab
   /mnt/disk1:/mnt/disk2  /mnt/storage  fuse.mergerfs  defaults,allow_other,use_ino,category.create=mfs,nofail  0 0
   ```
2. **Parity**: format the freed parity disk (sdb, serial JEH9VZHN -- CONFIRM by serial:
   `lsblk -o NAME,SERIAL`) and mount it:
   ```
   sudo wipefs -a /dev/sdbX        # ONLY after confirming serial JEH9VZHN
   sudo mkfs.xfs /dev/sdbX
   sudo mkdir -p /mnt/parity1 && mount it by its new UUID, add to fstab
   ```
3. **SnapRAID** `/etc/snapraid.conf`:
   ```
   parity /mnt/parity1/snapraid.parity
   content /mnt/disk1/snapraid.content
   content /mnt/disk2/snapraid.content
   data d1 /mnt/disk1
   data d2 /mnt/disk2
   ```
   `sudo snapraid sync` (first sync is long). Schedule `snapraid sync` + `scrub` via cron/systemd-timer.
   NOTE: SnapRAID parity is point-in-time (computed on sync), not real-time like Unraid -- a disk lost
   between syncs loses changes since the last sync. Fine for a media/model store; choose sync cadence
   accordingly. vm_8tb + cache can stay outside SnapRAID (cache = scratch; vm_8tb = mostly reproducible
   models + git-synced repo) or be added as another data disk if you want them protected.


## 8. Phase 5 -- services

**Intel B70 GPU stack (the point of the migration):**
- Ubuntu 26.04 ships kernel 7.0 with the `xe` driver -> both B70s should come up (`ls /sys/class/drm`,
  `lspci -k -s 0a:00.0`). Add yourself to `render` + `video` groups.
- Install Intel userspace: `intel-opencl-icd` / `intel-level-zero-gpu` / `level-zero` (and `clinfo`,
  `intel-gpu-tools` for `intel_gpu_top`). Verify `clinfo` / `sycl-ls` see both cards.
- vLLM-xpu runs in Docker as before -- pull `vllm-xpu-env:*`, expose `/dev/dri`, follow the serve
  recipe in **`docs/SERVING.md`** (do not reconstruct it). The repo at `/mnt/vm_8tb/b70` carries over,
  so `contrib/vllm_int8_xpu` and the scripts are right there.

**File sharing (Samba + NFS, LAN-only):**
- `sudo apt install samba nfs-kernel-server`. Recreate the few shares pointing at `/mnt/storage`
  (pooled) and/or specific disks. LAN-only, no internet exposure -> simple `smb.conf` (guest ok or one
  user), `/etc/exports` for the Linux clients. (I can generate both from the captured Unraid config.)

**Docker apps:** OpenWebUI (keep -- restore its compose/volume). Nextcloud (rebuild: nextcloud +
mariadb/postgres containers; point its data at mother's existing files on the array share so nothing of
hers is lost). Drop Syncthing + the VM stack.


## 9. Phase 6 -- validate

- Both B70s enumerate under `xe` on kernel 7.0; `clinfo`/`sycl-ls` list two devices.
- Serve a known-good model per `docs/SERVING.md`; sanity bench vs the Unraid numbers in FINDINGS.md.
- **Tie-in to the P2P research:** install `ze_peer`, check `pci_p2pdma` permits the pair under whatever
  IOMMU mode we boot, run a peer-bandwidth probe between the two B70s. This is the experiment that was
  impossible on 6.18 -- and the data point nobody has published (docs/P2P_GPU.md F.2/F.3).
- File shares reachable from the Windows + Linux LAN clients; mother's Nextcloud data intact.


## 10. Rollback

Any time before committing: power off, boot the **Unraid USB** (boot menu), optionally pull the Ubuntu
NVMe. Back on Unraid. If Ubuntu had written to the array RW, let Unraid rebuild parity (no data loss).


## 11. Open decisions / checklist
- [x] Confirm Ubuntu 26.04 LTS (kernel 7.0) is the target (vs 25.10 / a 7.1 HWE stack).
      DONE -- running 26.04 LTS, kernel 7.0.0-22-generic (section 13).
- [x] Confirm a free M.2 slot for the 500G NVMe (else use a PCIe M.2 adapter).
      DONE -- 500G NVMe installed and booting (nvme0n1, Samsung 970 EVO).
- [~] Decide SnapRAID coverage: disk1+disk2 only, or include vm_8tb. DEFERRED (Phase 4 deferred by Isaac).
- [x] Capture OpenWebUI + Nextcloud current configs/volumes before powerdown (Phase 0) -- DONE
      2026-06-24, see "Phase 0 capture -- RESULTS" in section 3.
- [ ] Generate Ubuntu `smb.conf` + `/etc/exports` from the captured Unraid share config.
- [ ] Pick Secure Boot on (keep shim) vs off (simpler) on the B70 box.
- [ ] Maintenance window: this is the box running quant27b + the NAS for mother -- schedule downtime.
- [x] P2P/IOMMU boot profile decided 2026-06-24 -- see section 12 (IOMMU off, NOT iommu=pt; ReBAR + Above 4G on).
- [x] Run the Phase 0 capture block (section 3) on the live Unraid box BEFORE powerdown -- DONE 2026-06-24.
- [x] On the router (192.168.10.1): set a DHCP reservation for MAC 70:85:c2:5d:b3:db -> 192.168.10.5.
      DONE -- host came up at 192.168.10.5 on enp3s0 (section 13).


## 12. P2P probe boot profile (BIOS + kernel) -- decided 2026-06-24

We no longer care about VM/VFIO passthrough, so bias the box toward GPU P2P. On this 1950X (X399, a
2-die MCM), the two B70 x16 slots are cross-root-complex / cross-die. Linux blocks P2P across host
bridges unless the platform is allow-listed; AMD Zen IS allow-listed, but the kernel ignores that
allow-list when an IOMMU is present. So the move is to remove the IOMMU from the path entirely.

BIOS:
- KEEP ON:  Above 4G Decoding; Resizable BAR / Smart Access Memory; UEFI (no CSM). (Large BAR is what
  makes P2P actually useful -- the full VRAM aperture becomes visible to the peer.)
- TURN OFF: IOMMU / AMD-Vi; ACS (if the board exposes it); CSM / legacy boot.
- SVM (CPU virtualization) can stay ON -- it is NOT the I/O MMU. Disable specifically the AMD-Vi path.
- Secure Boot: OFF if you will test mainline 7.1 (unsigned). For Ubuntu's signed 7.0 kernel, it can
  stay ON.

Kernel cmdline (only if BIOS lacks a clean IOMMU toggle):
- Add:    amd_iommu=off iommu=off
- Do NOT use iommu=pt -- it may still count as "IOMMU present" for the kernel's P2PDMA policy gate.

Kernel sequence:
1. Boot Ubuntu's supported 7.0 kernel FIRST -- the B70 xe P2P foothold is already in the 7.0-era xe
   work; 7.1 looks incremental (driver fixes, SR-IOV, Xe3/Nova Lake, TLB/context), not a known B70
   P2P unlock.
2. Probes: zeDeviceCanAccessPeer, ze_peer, torch.xpu.can_device_access_peer, and a P2P on/off
   vLLM/oneCCL A/B (see docs/P2P_GPU.md F.2/F.3).
3. If 7.0 says no P2P or misbehaves, install mainline 7.1.x as an A/B kernel and rerun the EXACT same
   probes. Keep 7.0 in GRUB as rollback.

Caveat: pcie_acs_override=downstream,multifunction is NOT in stock mainline (it needs the out-of-tree
VFIO ACS-override patch). On stock Ubuntu the real lever is the BIOS ACS toggle, not a kernel param --
so confirm the BIOS exposes ACS, or accept whatever the root ports advertise.


## 13. First boot on Ubuntu -- progress log (2026-06-23, run on the box itself, not over SSH)

Host = b70s4dayz. Acting user = hotschmoe (uid 1000), NOT root (Unraid ran everything as root -- expect
ownership friction on carried-over files; see NOT DONE).

DONE:
- Phases 1-2: Ubuntu 26.04 LTS / kernel 7.0.0-22-generic booted off the 500G NVMe (Samsung 970 EVO = nvme0n1).
  Network up at 192.168.10.5 on enp3s0 -- the DHCP reservation (MAC 70:85:c2:5d:b3:db) carried over. UEFI fallback
  boot path worked (box boots the NVMe with no manual menu pick).
- Disk identity RE-VERIFIED by SERIAL -- letters reshuffled exactly as warned: PARITY JEH9VZHN is now /dev/sda
  (was sdb); disk1 JEHA1MZN -> sdb1; disk2 JEHA1LNN -> sdd1; vm_8tb (S5SSNF...) -> sdc1; cache (A0D6...) -> nvme1n1p1.
  ALL UUIDs match section 0. ** Phase 4 wipefs target is /dev/sda (NOT the doc's literal /dev/sdb). **
- Phase 3: disk1/disk2/vm_8tb/cache mounted RW by UUID, persisted in /etc/fstab (nofail). RO-verified first; data all
  present (StrongSync + isos on both array disks, b70 repo at /mnt/vm_8tb/b70, cache appdata). fstab backup at
  /etc/fstab.pre-b70. Script: /home/hotschmoe/phase3_mount_data.sh.
- Phase 5 (GPU userspace): both B70s under xe (0b:00.0 / 44:00.0). Installed stock-archive intel-opencl-icd + libze1
  + libze-intel-gpu1 (all 26.05.37020.3) + clinfo + intel-gpu-tools -- NO external Intel apt repo needed on 26.04.
  `clinfo -l` = 2 Intel Graphics [0xe223] devices. hotschmoe added to render+video. Script: phase5_gpu.sh.
- BIOS P2P profile (section 12) APPLIED by Isaac: IOMMU/AMD-Vi OFF (iommu_groups=0), ACS off, memory-interleave off.
  No iommu= kernel param (BIOS toggle was enough).
- Phase 6 P2P HEADLINE: `71_ze_p2p_ctypes.py` -> zeDeviceCanAccessPeer = True (both dirs), P2PProperties ACCESS=Y,
  IPC zeMemOpenIpcHandle(peer) = PEER MAP OK. **B70<->B70 P2P UNLOCKED on kernel 7.0** (was False on all 12 variants
  on 6.18). The migration's central thesis, confirmed. See JOURNAL 2026-06-23 + docs/P2P_GPU.md H.11.

NOT DONE YET:
- Phase 6 BW measurement (the actual prize, H.10): allreduce_bench.py / oneCCL TP=2 P2P-on serve A/B -- BLOCKED on
  Docker + the int8g image (not installed). ze_peer peer-BW matrix -- needs level-zero-tests BUILT (not in 26.04 apt).
- Phase 5 Docker: NOT installed. OpenWebUI restore (open-webui_data.tgz on vm_8tb migration_capture) pending.
- Phase 5 file sharing: samba + nfs-kernel-server NOT installed. StrongSync share + Nextcloud rebuild pending ->
  mother's NAS access NOT yet restored (the #1 "do not break" item -- prioritize when ready).
- Phase 4 redundancy (mergerfs + SnapRAID): DEFERRED by Isaac. Unraid USB rollback still valid until parity (sda)
  is reformatted.
- Memory-interleave-off did NOT change the kernel NUMA view (still 1 node). Revisit only if NUMA-local host staging
  is wanted; irrelevant to P2P.
- Repo split: git repo at ~/github/b70_ai_things (canonical -- has GitHub remote + all docs) vs runtime scripts at
  /mnt/vm_8tb/b70 (scripts only, NO .git, NO docs/). Reconcile (make the host copy a clone) so "commit/push often
  when working on the host" actually works end to end.
- Ownership friction: vm_8tb files are root-owned from the Unraid era; gpu.lock* had to be removed so uid 1000 could
  recreate them. Sweep ownership (chown -R to hotschmoe where appropriate) when convenient.
