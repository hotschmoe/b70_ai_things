# Source page recurrence and health runtime audit

CONFIG -> Current artifact identity INVALID. The preserved known page at
3857879040+2796 now returns0x65 through both buffered and fresh alignedO_DIRECT
readers, versus preserved original0x45 (xor0x20), with size/mtime/ctime unchanged.
No source/cache invalidation, file write, replacement, driver change or GPU
launch was performed by this audit.

COMMAND -> Read preserved F06 recurrence/launch/health receipts, actual scripts,
Docker event metadata, current process FD metadata, immutable image package
lists, exact allocator headers and binary imports. CPU image inspections used
networknone and exposed no GPU devices; they did not import Torch or execute its
allocator. Record identities in health-dma-source-audit-v1.json.

RESULT -> Preparation had matching full hashes and original page sentinel by
09:14:39UTC. Per-card health containers ran09:15:19..09:15:43; the compiled
collective ran09:15:45..09:16:21. Launch refused at09:16:22 in validation, before
launch.command.json or any model container. Retained Docker events show only
health containers in that interval. This bounds exposure, not corruption time
or cause.

Health commands mount the DRM device paths, the read-only collective script and
its writable compiler cache. They do not mount the model directory; the image
has no declared volumes. The actual scripts have no model open/write path:
per-card2048x2048F32 random matmuls plus16x16 check, and4x5120BF16 two-rank eager/
compiled all-reduces plus object gather. These tests validate their own outputs,
not arbitrary host/file-backed memory. No allocation/DMA trace was recorded in
the failed-launch interval.

The health image d55637b3 contains Torch2.13.0+xpu (gitcf30153c), NEO26.27.39122.11,
IGC2.38.2 and LevelZero loader1.32.0. The model/fixture image39992d70 contains
NEO26.22.38646.4, IGC2.36.3 and loader1.28.2. Exact GPU/Torch/CCL library hashes
are preserved in the JSON audit. This is a process-runtime mismatch and a future
controlled variable, not evidence that26.27 caused the bit change.

Installed headers expose device caching allocation and event-tracked host
caching allocation. Actual binary imports include device aligned allocation,
physical-memory mapping, host allocation and queue memcpy; oneCCL exposes typed/
byte memcpy kernels. These establish allocation/DMA interfaces only. Without
execution traces, pointer/extent records, PFN/IOMMU history or complete source
for the executed binary, they do not establish what memory was touched or rule
out a driver/allocator/runtime defect.

Current accessible process FDs showed no descriptor matching the source inode;
656 entries were inaccessible and the health processes were already removed.
This cannot prove historical descriptor absence. The file is on btrfs, with no
reported file attributes/compression property; the target FIEMAP extent has no
encoded flag. This metadata does not convert O_DIRECT return bytes into proof
of corruption origin or physical-media bypass. Intended pathname isolation rules
out an ordinary intended model-file writer in these scripts; it does not rule
out DMA, memory, filesystem, storage or other faults.

VERDICT -> Preserve the current bad views and prior good direct view. Source
fidelity remains INVALID and model work stays gated. No causal claim, automatic
repair or image/driver substitution is authorized by this audit.

A source-only matched health option is prepared:
matched_sycl_health_control.cpp and build_matched_sycl_health.py use the exact
39992d70 runtime for deterministic16MiB GPU fill/copy/readback and tiledF32 A@A,
with finite outputs and exact dyadic FP64 sample checks. DefaultN256 is bounded;
N2048 matches the per-card test's matrix dimensions. It is a distinct pureSYCL
workload, not equivalent to Torch RNG/GEMM or compiled XCCL, so comparison alone
cannot isolate UMD causation. No compilation or GPU execution pass is claimed.

For a potential same-Torch/CCL comparison, matched_health_runtime/Dockerfile is
only a recipe using checksum-pinned official NEO26.22/IGC2.36.3/GMM packages.
Intel publishes package checksums in the official
[NEO26.22 release](https://github.com/intel/compute-runtime/releases/tag/26.22.38646.4)
and [IGC2.36.3 release](https://github.com/intel/intel-graphics-compiler/releases/tag/v2.36.3).
The manifest records URLs/hashes without downloading/installing. Loader package
provenance and byte match remain unresolved; first candidate would retain the
health loader1.32.0, explicitly unlike model1.28.2. Existing Torch/CCL/SYCL and
non-GPU libraries must remain unchanged. Compatibility and compiled collective
qualification are unproven; existing images/drivers remain unchanged.

Before any parent GPU control: preserve evidence, perform parent-approved source
recovery/reload and fresh prefaulted full direct+buffered publisher verification.
Record independent buffered/direct known-page reads and immutable stat/command/
image/library identities before and after EACH isolated workload (CPU idle,
card0, card1, collective separately). Never infer unchanged pages from healthOK.
health_page_control_contract.py rejects bad/short/wrong-offset/stat/stale boundary
records; eight CPU negative controls passed using preserved pages only. On any
change, stop and preserve evidence, without invalidating or rewriting the views.
The plan matched-health-control-plan-v1.json preserves these scope limits.

## Historical snapshot and later restored status

The CONFIG and VERDICT above describe the preserved 2026-10-09 09:14..09:16 UTC
failed-launch interval. "Current artifact identity INVALID" and "model work
stays gated" are the status at that incident, not the later live status. Prior
incident prose and raw failure evidence are preserved.

Later CPU receipt inspection confirms that
F06/model-full-identity-after-corrected-twocard-v1/receipt.json passes all four
full buffered publisher hashes, with unchanged before/after stat records.
Receipt SHA256: af72dc62daf0c52c7933564d38281c1c2104f6be4723e64a118571acb4a6dd00.
Corrected one-card and two-card C1 qualification receipts both report screen /
lifecycle, teardown and post-health pass:

- F06/c1-onecard-corrected-streams-prepared-v2/qualification.json,
  SHA256 b38817a7040848be55acb879254345728c8f255a72a25892e2e82d04aeb8869a.
- F06/c1-twocard-corrected-streams-prepared-v2/qualification.json,
  SHA256 ec92f3d05a4a7b328ed7d5c8f03624e92b6db5da8a5e7a4a065bc717f9f32304.

These receipts explicitly leave full model numerical / state and concurrent
serving qualification outstanding and deny shelf promotion. Later recovery and
successful controls do not establish or fix the cause of the historical page
change. No matched-health build, GPU run or stack replacement was performed by
this source-preparation review. See health-source-preparation-review-v1.md for
the optional control file list and unresolved pinned identities.
