# Actual-model original-source upload oracle

CONFIG -> Selected UD-Q4_K_XL revision766911a; Strata fb58e0 plus current
source-preserving HC/PLE patches; image39992d70. No inference or GPU execution
in this preparation. Full-engine static libraries are from its isolated source
build and pinned3cf03257 GGML dependency.

COMMAND -> Prepare native_source_upload_gpu_oracle.cpp and immutable
native-source-upload-plan.json. Link actual WeightTable/NativeDense/core/kernel
libraries in a compiler container with no devices. Generate30-image SHA roster
from existing387-HC source receipt plus bounded reads of the three original PLE
source images. Preserve source offsets and publisher-intake provenance.

RESULT -> The first helper generation compiles and links against the actual full
engine. Final compile receipt is separate below/in the parent evidence. The
roster covers27 HC images (layer0,layer1,layer47 and final mixer) and3 PLE images.
No synthetic model tensors, full canonical arena, experts, PLE table, state,
logits or sampling are loaded. GGUF mappings cover virtual files but only the
selected native matrices and headers are read; no103GB payload scan occurs.

VERDICT -> Source upload oracle is ready for parent-owned device qualification.
Compilation does not establish GPU byte identity, ownership or teardown. The
known pre0010 NativeDense PLE accounting omission must FAIL this oracle; do not
waive it to obtain a passing launch.

Build against the final complete engine receipt after0010:

```sh
python3 strata/flash-next/build_source_upload_oracle.py \
  --engine-receipt FINAL_ENGINE_BUILD_RECEIPT
```

The helper clones no engine source and edits no live build. It creates a new
helper source/binary directory, snapshots exact preregistered plan bytes, checks
actual engine executable/source/config hashes, links the actual static libraries,
and records every library hash plus preservation checks. The container has no
DRM devices and network is disabled. Dependencies remain the pinned image and
existing engine/ggml source; libcrypto provides streaming SHA256.

The prepared source roster and receipt are:

- /mnt/vm_8tb/b70/results/flashnext-reset-diag-source-20261009/source-upload-roster.tsv
- /mnt/vm_8tb/b70/results/flashnext-reset-diag-source-20261009/source-upload-roster-receipt.json

The roster producer is prepare_source_upload_roster.py. It binds the existing
actual-HC receipt SHA and selected inventory SHA, reads only original PLE key,
value and convolution payloads, and records the selected model revision. It
explicitly does not refresh full-shard hashes. Parent preflight must bind the
model lock/intake, pack/tokenizer/template and exact helper/build/roster hashes.
The GPU helper must receive that immutable roster through --source-roster.

Example command within the parent's leased, bounded container, with explicit
physical affinity and health monitoring already established:

```sh
/work/source-upload-oracle --pack /pack --source-roster /roster/source-upload-roster.tsv \
  --shard /model/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf \
  --shard /model/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00002-of-00004.gguf \
  --shard /model/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00003-of-00004.gguf \
  --shard /model/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00004-of-00004.gguf \
  --stage 0:1:0 --stage 1:2:0 --stage 47:48:0 --output /results/upload.json
```

Use a new output path; the helper creates it exclusively. Case order is the
three owners on card0, the same three on card1 with logical ordinal0, then
pair-affinity0,1 with stages0:1:0,1:2:1,47:48:1. No stage's owner is freed until
all simultaneously live owners are checked. Runtime env is in the plan: strict
native flag1, legacy HC/QFUSE/GRV3/packed-secondary flags0, persistent cache0.

What the actual helper proves when it passes:

- Actual packed index rows pass the real WeightTable parser. Skipping every
  canonical row preserves all metadata and requires zero canonical pool bytes.
  Its temporary internal staging allocations are still real; they are bounded
  separately and do not justify a full-model memory-fit claim.
- Actual NativeDense load uploads original source bytes with explicit stage
  ranges. Every local HC image and layer1 PLE source is present, foreign layers
  and foreign final mixer/source pointers are absent, and legacy HC copies are
  absent. Standard native dense images are enumerated for allocation accounting
  but their GPU data is not read back in this narrow oracle.
- Device USM type and pointer device match the owning default in-order queue and
  context. Within each default device context, simultaneously live stage images
  and shared scratch cannot alias. Distinct contexts can use identical numerical
  addresses; those alone are not evidence of physical aliasing.
- Every full HC/PLE image is copied back in4MiB chunks on its owning queue. Bytes
  equal original GGUF source slices and streaming SHA256 equals the frozen
  source roster. Type, shape, original shard, absolute offset and loader FNV
  metadata also match. Signed-zero/nonfinite bits are preserved by byte checks.
- The actual shared native_hc_bind helper selects those exact original pointers,
  and rejects an invalid source type. Actual ple_exact_sources_ok rejects an
  invalid byte extent. These are dispatch-descriptor checks, not model forward
  or numerical math. A host-byte corruption negative is detected for each row.
- Independent ordinary-native+HC+PLE image-byte sum equals weight_bytes().
  Legacy tensor_count is recorded separately; source image count is not a
  renamed ordinary-projection count. Packed secondary copies are disabled.
- Foreign-device pointer metadata is rejected before a transfer is submitted.
  No deliberately invalid cross-device dereference or GPU fault is injected.
- Each owner is synchronized and destroyed with its device current; old USM
  pointers lose registration, then a fresh64KiB allocation/write/read/free
  probe succeeds on that queue. Parent normal exit/removal, fault audit and
  strict per-card/compiledP2P0 pre/post-health remain mandatory.

This first path passes explicit load ranges. The actual model's static-bound
call sites, full multi-stage placement/mirrors, consumed model forward, all387
HC images, PLE table/kernel math, recurrent/KV/prefix state, cancellation,
concurrency and latency still need their own gates. Source-upload success must
not be presented as a full-model fidelity or serving qualification.
