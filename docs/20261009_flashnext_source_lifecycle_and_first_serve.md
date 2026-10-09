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

## Subsequent source generation and identity check

CONFIG -> New immutable engine planv6 includes0011 and0012; original model,
toolchain/runtime and external source pins unchanged.
COMMAND -> build_native_hc_engine.py --plan native-source-engine-build-plan-v6.json
with the previous clean pinned GGML checkout; independent full buffered hash
scan after the failed first serve and allocation probe.
RESULT -> All three required executables build/link, exit0 in277 seconds,
with external source and plan snapshots unchanged. Final segment patch CPU
ASan/UBSan contracts pass, SHAe620578b. All four whole-file hashes match the
publisher lock again, with pre/post stat signatures including ctime unchanged.
VERDICT -> Build/source identity gates pass. Full390 uploads, GPU mirror
consumption and complete model fidelity remain unqualified.

Engine receipt:
/mnt/vm_8tb/b70/build/strata-native-hc-engine-20261009T065210Z-e6m4q9uo/receipt.json.
Fresh whole-model identity receipt:
F06/model-full-identity-after-first-serve-v1/receipt.json.

## Complete HC/PLE source uploads

CONFIG -> Actual linked generation6 NativeDense loader, full390 source roster,
unchanged exact4-GGUF/pack/runtime, logical-free trace version2.
COMMAND -> run_source_upload_oracle_full.py, F06/source-upload-full390-v1.
RESULT -> All five cases pass: fullcard0, fullcard1, same-device24/24,
two-device24/24, and actual static model stage bounds. Every case reads back
387 HC and3 original PLE tensors with exact source SHA/bytes/type/shape/offset.
Ordinary300 matrices are independently enumerated and allocated; their payloads
are not read back by this HC/PLE oracle. Measured owned image total is
3,821,772,800B: HC695,132,160B, PLE34,979,840B, ordinary3,091,660,800B.
The earlier estimate understated the PLE sum by640B; actual accounting agrees.
Single-owner cases have690 image pointers plus scratch,694 matched UR alloc/free
pairs. Split cases have345 image pointers per stage,698 matched pairs total;
source ownership is192HC/3PLE/150ordinary on stage0 and195HC/0PLE/150ordinary
on stage1. Missing/duplicate/failed-free controls reject in every case, logical
live ledgers are empty, all owners return through their owning destructor.
Normal removal, strict per-card and compiled P2P0 pre/post-health pass, with no
reported fault signatures. The original page sentinel remains exact per case.
VERDICT -> Complete HC/PLE source bytes and loader ownership/lifecycle pass.
Ordinary GPU payload fidelity, expert mirrors, graph execution, full-model
state/logits/coherence, prefix caching and concurrency/latency remain open.
