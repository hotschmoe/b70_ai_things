CONFIG -> Named Btrfs backing-file device-domain association. Actual no-DRI
reproduction V3 mapsdev00:32=0:50, inode17686394, file3818408B; path, openedFD,
nativeFD3 and hostpath stat all dev0:45 with the SAME inode/size. Covering /pti
readonly mount is btrfs /dev/sdc1, subvolid5, exact actual install bindroot.
map_files readlink equals /pti/lib/libunitrace_tool.so; stat returns EPERM.
The separate /mnt/storage mergerfs0:43 is not this tool's bind source.

Primary Linux v7.1 confirms different fields: btrfs_getattr in inode.c:7737-7741
sets stat.dev from root.anon_dev; show_map_vma in task_mmu.c:443-448 reads
inode.i_sb.s_dev and inode.i_ino for maps. This is consistent with the reproduced
metadata; it does not prove the lost original model-stderr failure cause.
Sources: https://raw.githubusercontent.com/torvalds/linux/v7.1/fs/btrfs/inode.c
and https://raw.githubusercontent.com/torvalds/linux/v7.1/fs/proc/task_mmu.c

COMMAND -> Root-owned exact no-DRI test first, new standalone helper:
/opt/b70-c1-python/bin/python /pti-census.py --output NEW_RESULT
--expected-sha256 PINNED_LIB_SHA --phase after_READY
--expected-bind-root /b70/build/pti-unitrace-model-finalizer-c388-build-v1/install
--expected-filesystem-source /dev/sdc1
Mount new pti_mapped_injection_census_v2.py at /pti-census.py read-only. It is
stdlib-only; no extra repo modules inside the container. Actor/ownedexec/public
reader successor is separately owned by c112, with exact helper/source/hash and
actual Docker install mount relation. No frozen helper/actor/reader is edited.

RESULT -> New helper holds O_RDONLY|O_NOFOLLOW regular library FD, hashes its
bounded actual bytes and requires original path/openedFD stat5 before/after.
Original inode/path/nondeleted map links, executable/ordered/page-bounded
segments and actual PID1 start/mapping stability remain required. VMA device
must equal exact longest covering readonly btrfs /pti mountinfo device; expected
source/bindroot/subvolid5 and before/after mount row remain exact. Both devices
are preserved as named domains, not silently rewritten. Raw maps/mountinfo
before/after and exported backing ELF are saved. No map_files stat/open,
process-memory read, GPU query, capability change, or syscall privilege change.

VERDICT -> Source/CPU helper proposal, not actual runtime/profiling authority.
Eighteen tiny controls include actual temporary-file FD/hash/census execution,
wrong inode/mount/fs/maplink/SHA, changed map/stat, aliases and exact standalone
AST copies. Temporary fixture metadata is synthetic; no actual tool payload,
model or GPU data/action was read/executed by author. Full source hashes plus
small original no-DRI V3 metadata are pinned. Root must execute new no-DRI test
and peer-reviewed actor successor before a retained model retry. Loaded memory,
relocations, symbol origin and LevelZero handle ownership remain unobserved.
