# Repeated cached-source bit and direct-reader discriminator

CONFIG -> Exact selected shard3 and preserved first-event original page. New
corrected engine had passed source uploads/control capture; model use remains
subject to buffered identity and known-page guards.

COMMAND -> New corrected C1 preparation/health/launch, independent read-only
C/Python/GNU matrices, whole prefaulted direct hash, targeted page reload and
separate health exposure with page checks.

RESULT -> Prepared full hashes and sentinel pass by09:14:39UTC. Only per-card
and collective health containers run in the following interval. At09:16:22 the
launch refuses the sentinel before creating a model container. The same byte
changes again: page3857879040, offset2796, original0x45 versus cached0x65,
xor0x20. Inode/size/mtime/ctime remain unchanged. CPU two-card preparation is
interrupted and all model work stops; owned parent completes post-health.
The failed launch is preserved and is not a new inference result.

Initial untouched-destination O_DIRECT reads also return the bad page, including
GNUdd and Python4KB/64KB/1MB buffers. That does not establish changed disk data.
Independent initialized C direct reads return the preserved original. Controlled
C heap/private/shared allocations and pread/read/readv/preadv, alignment and
O_CLOEXEC comparisons isolate destination initialization: initialized buffers
return0x45, untouched buffers return0x65. Python reproduces the same pair.
All descriptors/paths/offsets/lengths are checked by strace. FIEMAP affected
extent has no encoded/shared/delalloc/unwritten flags on this Btrfs filesystem.
Upstream nofault/direct-read to cached-read fallback is a supporting mechanism;
the exact running7.1 branch remains uninstrumented and is not claimed proven.

Whole49,376,141,504B shard3 direct hashing with an initialized reusable16MiB
buffer matches publisher56758f40..., stat including ctime unchanged. Both bad
and original4KB views and independent reader results are preserved. Only then
POSIX_FADV_DONTNEED targets that4KB page; reload restores original2780fef9...
without a file write/replacement/redownload/global cache drop. Corruption origin
is still unknown. No relation to earlier model history drift is established.

A controlled CPU-idle20-second baseline and separate card0/card1/compiledP2P0
health exposures all pass, retaining original buffered and initialized-direct
sentinels before/after each stage. This one controlled sequence does not
reproduce the recurrence or identify its cause. Source audits find intended
health commands expose no model path, but inaccessible/historical process
mappings and driver/DMA behavior are not ruled out by that source inspection.
Package metadata also records a process-stack difference: health image d556
contains PyTorch2.13.0+xpu with NEO26.27, while model image39992 uses the26.22
lane. This is an identity fact, not evidence that the UMD difference caused it.

VERDICT -> Exact buffered model identity was invalidated and blocked before
serving. Initialized-direct whole-shard identity and targeted current recovery
are verified, and the health exposure did not reproduce the bit change. Further
model qualification retains full-hash/page/lifecycle gates; no hardware, storage,
stability or causality claim follows. New preparation performs all four full
buffered hashes before a guarded retry; failed earlier runs stay failed.

Raw evidence is under F06:

- shard3-cache-recurrence-v1/receipt.json and all captured/discriminator files
- shard3-recurrence-independent-readonly-j8v_6ncs/receipt.json and read matrices
- shard3-prefaulted-direct-full-v1/receipt.json
- shard3-cache-recurrence-v1/targeted-cache-reload.json
- page-sentinel-health-discriminator-v1/receipt.json
- c1-onecard-corrected-streams-prepared-v1/parent-qualification.json (failed)

F06 is /mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f06-20261009/.
