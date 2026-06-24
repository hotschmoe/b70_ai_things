# migration/ -- one-time build scripts for the 2026-06-23 Ubuntu 26.04 box

These are the actual scripts that built the current GPU host (`b70s4dayz`, kernel 7.0)
during the 2026-06-23 migration off Unraid. They ran ONCE from `/home/hotschmoe/`
and are kept here as the reproducible build record paired with
[`../MIGRATION.md`](../MIGRATION.md) (which references them by name). They are
DONE -- do not re-run on the live box without reading MIGRATION.md first.

## Host setup (run as root via `sudo bash <script>`)

- `phase3_mount_data.sh` -- mount data drives RO-verify then RW + fstab by UUID.
- `phase4_raid.sh` -- mergerfs pool (disk1+disk2 -> /mnt/storage) + SnapRAID parity
  on the freed Unraid parity disk (serial-guarded wipefs target /dev/sda).
- `phase5_gpu.sh` -- Intel B70 GPU userspace (opencl-icd, libze) + render/video group.
- `phase5_shares.sh` -- Samba + NFS for the media pool, LAN-only, guest access.
- `phase5b_winshare.sh` -- make the no-password SMB shares Windows-discoverable.
- `phase5c_wsdd.sh` -- hand-create the missing wsdd systemd unit; prove guest SMB.

## Docker image recovery (from the old Unraid docker.img)

- `install_docker.sh` -- install Docker with data-root on the 8TB SSD.
- `inspect_dockerimg.sh` -- RO loop-mount the old Unraid docker store to confirm images.
- `extract_images.sh` -- recover vllm-xpu-env:v0230 / :int8g via a throwaway dockerd.
- `extract_moe_and_cleanup.sh` -- recover vllm-xpu-env:v0230moe, then clean leftovers.
- `prep_bench.sh` -- make results/ writable + recover the last image before the shelf bench.

Raw run logs from the migration + initial lever/shelf testing are parked locally
(uncommitted) in `../archive/migration_logs/`; their findings are distilled into
`../JOURNAL.md` and `../README.md`.
