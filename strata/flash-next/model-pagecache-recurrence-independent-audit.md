# Source page recurrence: independent read-only audit

CONFIG -> selected shard3, inode15212298, offset3857879040,4KB; no GPU,
source writes, cache invalidation, redownload or recovery. Parent initial
recurrence receipts and original page remain unchanged. Source identity is
blocked for model serving because the buffered/mapped bytes are bad.

COMMAND -> python3 strata/flash-next/audit_source_page_recurrence_readonly.py,
then preserved C allocation/syscall matrix and Python/GNU direct reader checks.
Primary independent receipt and all bounded raw/source/trace captures are in:
/mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f06-20261009/shard3-recurrence-independent-readonly-j8v_6ncs/.
A first CPU compile attempt found no host gcc; the completed reader used gcc
in the pinned compiler container without devices, then executed on the host.
The initial uncompleted audit directory is retained.

RESULT -> buffered pread and PROT_READ/MAP_PRIVATE source mmap consistently
return SHA6128219c4ced7dc7852371b3b0153c34b0df0529e2b1a8d6b07ba08da764fece,
byte2796=0x65. Initialized aligned C O_DIRECT destination buffers return
SHA2780fef9ce50fa1acbd4bdbf6c311b847395571fcc5e6ddb55898841fbcee90e,
byte2796=0x45, matching the preserved original. Inode/size/mtime/ctime remain
unchanged. The affected FIEMAP extent has no encoded flag; compressed-extent
fallback is not supported by that metadata. Readable /proc maps show no
persistent source mappings, but44 process maps were unreadable; this is not
proof that no process/runtime retains an inaccessible mapping or physical alias.

The decisive matrix controls destination initialization independently from
allocation and syscall. C heap, private-anonymous and shared-anonymous buffers
crossed with pread/preadv and O_CLOEXEC all return0x45 after memset0xA5, and
0x65 when left untouched. Earlier C read/readv variants also returned0x45
when initialized. Python mmap preadv/readv reproduces untouched0x65 versus
initialized0x45. GNUdd iflag=direct returns0x65. Strace verifies O_DIRECT,
the same source path, exact offset and read lengths; calltype/alignment/maptype
and O_CLOEXEC do not explain the result. Raw final Python paired snapshots
and direct-cloexec-prefault-matrix.json preserve the discriminator.

Current upstream Linux provides a matching mechanism: btrfs_direct_read
suppresses user-buffer faults during direct reads, and btrfs_file_read_iter
can finish a short direct attempt through cached filemap_read. A zero-progress
read lacks the ret>0 retry condition in that source. This supports a destination
prefault-dependent buffered fallback explanation, rather than treating every
successful O_DIRECT descriptor as proof of page-cache bypass. This is an
inference; the exact running7.1 build and actual branch remain uninstrumented.
Primary source links:
https://raw.githubusercontent.com/torvalds/linux/master/fs/btrfs/direct-io.c
https://raw.githubusercontent.com/torvalds/linux/master/fs/btrfs/file.c

VERDICT -> the original recurrence is real in buffered/mapped source bytes.
The initial bad O_DIRECT samples are reader-dependent and do not establish
bad underlying storage bytes. Independently initialized direct destination
buffers presently match only the preserved4KB page. This does not prove the
whole49GB shard, disk/RAM health, or corruption origin. Preserve both views
and the original failed launch gate. No relation to prior model drift is
established and no recovery was performed in this audit. Future direct-reader
receipts must record destination prefault/initialization alongside accepted
fd flags; full artifact, coherence and lifecycle gates remain mandatory.
