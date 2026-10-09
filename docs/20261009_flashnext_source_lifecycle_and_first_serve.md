# Native source lifecycle and first full-model launch

CONFIG -> Exact pinned UD-Q4_K_XL, Strata fb58e0d, source engine generation5,
unchanged oneAPI/UMD image39992d70. Derived Python runtime002f80b6 adds only
pinned server dependencies; GPU library hashes before/after agree. No shelf.

COMMAND -> F06/source-upload-v2, all four full shard hashes for
c1-onecard-prepared-v2, qualify_c1_serving.py, and host-alloc-limit-v1.

RESULT -> The separate version2 source owner oracle passes card0/card1/pair:
27HC and3PLE original source images per case, exact GPU readback SHA/bytes,
allocation accounting, owner-specific chronological UR allocation/free ledger,
fresh probes and missing/duplicate/failed-free negative controls. Each case has
62 matched allocations/frees,53 registered owners and3 context bridges, with
zero live logical allocations. Clean removal and strict per-card plus compiled
P2P0 collective pre/post-health pass. This is30-image coverage, not full390 or
model inference. The failed version1 freed-pointer-type gate remains archived.

Fresh full shard3 buffered hashing initially contradicted the pinned publisher
SHA, despite unchanged inode size/mtime/ctime. Independent O_DIRECT full hashing
matched the publisher. A complete49GB direct/buffered comparison localized one
changed cached4KB page and one bit: page3857879040, byte2796,0x65 versus0x45.
It maps to blk.13.ffn_up_exps.weight Q4_K, expert79, withinblock124. Both views
were preserved before targeted POSIX_FADV_DONTNEED for that page. Reload and
fresh full buffered hash then matched disk/publisher; all four fresh full hashes
passed C1 preparation. No source file write, replacement, redownload or global
cache drop. Cause remains unknown; no connection to historical drift established.

The first full-model launch uploaded691 native projection images,3671.29MiB,
and filled8083 expert slots,23.59GiB on card0. It then requested one pinned host
mirror for16493 remaining experts,49293MiB, and malloc_host returned null.
VERIFY_NO_HOST refused uncovered fallback. The process exited before readiness;
no inference/coherence/latency result exists. Owned container removal and strict
per-card/compiled collective post-health pass. The previously changed page
sentinel still matches preserved disk bytes after the launch; this single-page
check is not a new whole-file proof.

The independent card0 host allocation probe reports max_alloc32530182144,
global memory34242297856 and memlock8388608. Untouched malloc_host allocations
of1MiB/1GiB/8GiB succeed;48GiB returns null. All successful allocations are
freed through the owning queue, process exits0 and container removal is verified.
Post-health passes. The probe supports bounded mirror segments, without proving
the exact allocator rejection cause or RAM residency under a full workload.
Initial missing dpct include and incorrectly nested image shell invocations
produced no executable; the corrected compile uses the image's bash-lc entrypoint
with one command string and the tracked Strata SYCL include path.

VERDICT -> Native30-image source/lifecycle gate passes; full390 source coverage
and full-model state/logit/coherence remain open. Segmented stage-owned mirrors
are the next measured response to the allocation failure. Two-card placement,
prefix-cache correctness,1/2/4/6 concurrency and matched latency/shelf gates
remain required. No serving speed or stability claim.

Raw evidence root:
/mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f06-20261009/

- source-upload-v2/receipt.json and per-case raw traces/logical-free reports
- shard3-buffered-identity.json and shard3-direct-identity.json
- shard3-cache-discriminator-v1/receipt.json, preserved pages,
  targeted-cache-reload.json and restored-buffered-identity.json
- c1-python-runtime-v1/receipt.json
- c1-onecard-prepared-v2/prepared.json, parent-qualification.json,
  server/stop/post-health logs and cache-sentinel-before/after.json
- host-alloc-limit-v1/probe.cpp, run.py, probe.log and receipt.json

The previous status-only goal turn is classified no progress. This continuation
revalidated terminal process84809 and actual container absence, executed the
allocation discriminator and post-launch sentinel check, and prepared the next
source generation; it therefore changes evidence and the next action.
