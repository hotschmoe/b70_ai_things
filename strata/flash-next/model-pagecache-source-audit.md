# Selected GGUF cached-view identity contradiction: source audit

CONFIG -> CPU read-only audit after fresh buffered shard3 SHA mismatch. No GPU,
redownload, cache dropping, source payload write or file replacement. Exact
publisher revision766911a remains selected; the goal is not narrowed or changed.

COMMAND -> Query pinned Hugging Face metadata; inspect lock/original intake,
fullhash implementations, source file stat and actual mmap/upload/read paths.
Parent independently completed buffered and read-only directIO full hashes and
is preserving bounded direct/buffered mismatches before any recovery action.

RESULT -> Publisher LFS metadata, lock and initial full intake agree on shard3:
size49,376,141,504 and SHA56758f40269cad5cd9b0d3d6fbae0f40f6d5be6de49e4ab392dbe83157d9cbd3.
Parent directIO full hash matches it. Buffered Python/GNU hashes instead start
6ea; inode size/mtime/ctime did not change. Direct-vs-buffered evidence therefore
identifies a contradictory cache/RAM view while the disk blob is healthy.
It does not identify which process, driver, CPU/RAM or GPU path changed it.

Source findings:

- fetch_model.py hashes entire files in16MiB blocks; the C1 controller uses1MiB
  blocks. Neither uses a prefix-only shortcut. Initial intake explicitly
  recorded the full matching digest on October8.
- Strata GgufFile opens O_RDONLY and maps PROT_READ/MAP_PRIVATE. tensor_data()
  returns const bytes. NativeDense queue copies use the original pointer as
  source and a separately allocated device buffer as destination, followed by
  completion waits. Original key/value/conv and HC source uploads use this path.
- iq_pack maps source files with numpy.memmap(mode='r'). Our source audits and
  source-roster tools use rb. Pack outputs use separate new files/directories.
  No normal source file write path was found in this review.
- Strata expert streaming uses O_RDONLY and pread into independent host ring
  buffers; mirrors use separately allocated pinned host USM. GPU-consumed
  mirror/address tables and expert caches are writable allocations, not direct
  source file mappings.
- llama.cpp maps PROT_READ/MAP_SHARED. CPU weight buffers can point into this
  read-only mapping, while SYCL host-buffer allocations use separate malloc_host
  storage. SYCL buffer_from_host_ptr returns nullptr and direct host-registration
  hooks are unsupported in this source. Source-host tensor transfers declare
  host_to_device. In-place reorder targets SYCL-backed tensor data and requires
  a SYCL extra descriptor; no intended mmap source permutation was found.
- Both engines can ask GPU DMA/runtime paths to read pageable mapped source
  memory. CPU PROT_READ and const pointers constrain intended CPU writes; they
  do not independently prove every runtime/driver DMA mapping and destination.
  Allocator physical aliases, late writes/lifetime defects, unexpected source
  pointers or ordinary RAM/cache faults remain hypotheses requiring evidence.

Recognizable writer signatures may help parent mismatch localization: Strata
arena_alias_check writes uint64(block+1)*0x9E3779B97F4A7C15 at64KiB boundaries;
upload postfree probes write0xA5; allocator controls use0xA5/0x3C starts and
0x5A/0xC3 endpoints. Reset SCALE0 produces zeros. A matching sequence would
support a writer relationship; a generic differing chunk does not. No such
signature is claimed by this source audit.

VERDICT -> Preserve the bad cached view and healthy direct bytes, offsets,
tensor mapping, repeated read hashes and immutable receipt identities. Do not
redownload the healthy disk blob or discard cache evidence as an unexplained
fix. No causal link to prior history drift is established. Full390 uploads,
model inference and any shelf/speed claims remain blocked by current cached
artifact identity until qualified recovery and fresh full hashing
restore matched buffered/direct publisher identity. The corruption origin may
remain unknown and must not be described as fixed by cache invalidation.

Raw CPU audit and publisher metadata:
/mnt/vm_8tb/b70/results/strata-model-identity-audit-20261009/identity-audit.json
and pinned-publisher-metadata.json. Parent authoritative direct/buffered evidence
is F06/shard3-direct-identity.json and shard3-buffered-identity.json. Historical
source/GPU receipts retain their original limited scopes and are not rewritten.

Parent localization/recovery follow-up: a complete49GB buffered/direct comparison
found exactly one changed4KB cached page, at3857879040, and one byte within it:
offset2796, cached0x65 versus direct0x45 (xor0x20). It maps to
blk.13.ffn_up_exps.weight Q4_K, expert79, withinblock124. The parent preserved
both views, then targeted POSIX_FADV_DONTNEED to that page only. The reloaded
page matches preserved disk bytes; inode size/mtime/ctime remain unchanged.
Fresh full buffered SHA is still pending at this note. No writer signature or
origin is proven and no relationship to historical drift is claimed. Future
GPU/model receipts must watch this original page sentinel before/after in
addition to the full artifact identity gate. This repair is not a disk write,
redownload or global cache drop. Preserve the complete discriminator receipt:
F06/shard3-cache-discriminator-v1.

Current recovery evidence: F06/restored-buffered-identity.json records fresh
full buffered shard3 SHA matching56758f40269cad5cd9b0d3d6fbae0f40f6d5be6de49e4ab392dbe83157d9cbd3.
The subsequent c1-onecard-prepared-v2/prepared.json independently records
matching fresh full hashes for all four selected shards. This closes the
current buffered/direct artifact-identity contradiction after the targeted
page reload; it does not identify the bit corruption's origin. The original
bad cached/disk views and failed preparation remain preserved. Sentinel
pre/post observation and whole-artifact identity gates remain mandatory for
subsequent model experiments. The first research C1 attempt reached source
weights/cache loading but a single48GiB host allocation returnedNULL before
readiness; parent recorded normal teardown/post-health. It is not model serving
or coherence evidence and motivates separately qualified segmented mirrors.
