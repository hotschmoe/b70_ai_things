# Source-upload lifecycle contract v2

CONFIG -> Same selected model and exact native source paths. The v1 source,
plan, build helper and document are preserved under source-upload-v1-archive/;
archive-receipt.json binds their exact SHA256. The actual source-upload-v1 run
remains failed. No old result is rewritten or relabeled as passing.

COMMAND -> Parent ran default allocator controls and UR trace controls on each
card. Agent inspected actual JSON/logs, installed runtime strings/headers,
contemporary primary source, and the installed UR parameter formatter in a
CPU-only fixture. Prepare a separate v2 helper and strict chronological parser.

RESULT -> Each card's42 controls passed allocations, matched frees and fresh
probes. Every postfree SYCL type remained device, even raw zeMemFree controls.
Raw LevelZero address-range queries failed after free, while properties still
returned the old type/id. Therefore the original unknown-type assertion is not
a reliable logical lifetime test on this stack; retained UMF slabs alone do not
explain all observations. Traced controls have60 successful device allocations
and60 successful frees per card,0 live logical allocations. Missing, duplicate
and failed-free negative controls reject. Installed parameter formatting and
synthetic owner/context/extent/destructor-boundary negatives also reject.

VERDICT -> v2 replaces the unsupported lifetime inference with explicit logical
free evidence, while preserving live USM ownership, exact source SHA/bytes,
foreign-image, allocation accounting, cross-owner isolation and fresh-probe
checks. Cached physical backing release is a separate question. The parent
must combine the trace collector with source/probe receipts and health/lifecycle
before any v2 qualification. This is not a model inference or speed claim.

Actual controls: F05/usm-free-control-v1 and F05/usm-free-trace-v1 under
/mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/. Agent parser receipts are
/mnt/vm_8tb/b70/results/strata-usm-free-audit-20261009/card0-trace-parser.json
and card1-trace-parser.json. No GPU operations were performed by this agent.

The image has UR0.12 adapters v1 db43d6ff and v2 19a8a859, loader07c6310a and
libsycl9 c22729e4. The control records actual loaded library maps. Contemporary
UR source d51668b supports pooled free versus backend allocation-info separation,
but its exact match to the installed binary is not established; local controls
are the deciding evidence. Primary references:

- [SYCL pointer queries](https://github.khronos.org/SYCL_Reference/iface/usm_pointer_queries.html)
- [UR tracing and logging](https://oneapi-src.github.io/unified-runtime/core/INTRO.html)
- [UR LevelZero environment](https://oneapi-src.github.io/unified-runtime/core/LEVEL_ZERO.html)
- [Pinned UR USM implementation](https://github.com/oneapi-src/unified-runtime/blob/d51668b8845600b9d9361f1ba734ee2ba97f313f/source/adapters/level_zero/v2/usm.cpp)

Build v2 against the final engine generation:

```sh
python3 strata/flash-next/build_source_upload_oracle_v2.py --engine-receipt FINAL_ENGINE_RECEIPT
```

Use UR_ENABLE_LAYERS=UR_LAYER_TRACING and
UR_LOG_TRACING=level:info;flush:info;output:stderr in a separate diagnostic
process. Allocator settings stay unchanged. The executable requires tracing,
registers every HC/PLE/ordinary/scratch role, source name, pointer, extent,
nativeZe context and stage while alive, and brackets the real NativeDense
owning-device destructor. Postfree type is recorded, never dereferenced and
never used as proof of liveness. A fresh64KiB probe must still pass.

Raw C++ receipt reports source_and_probe_passed and
all_owners_destructor_returned, while passed=false until independent collection.
This avoids an executable-only success claim without observing logical frees.
Run the parser after normal process exit:

```sh
python3 strata/flash-next/parse_usm_logical_free_trace.py \
  --log CARD_LOG --oracle-json CARD_RAW_JSON --output NEW_TRACE_RECEIPT
```

The parser reads actual successful return records. It keeps a chronological
ledger keyed by UR context and pointer, with allocation generations for reused
addresses. urContextGetNativeHandle joins UR handles to registered nativeZe
contexts. Every owner marker must match exactly one live allocation and its
extent. Its successful free must occur inside that same owner's destructor
bracket. Unknown/missing/double/failed frees, duplicate live allocations,
ambiguous contexts, wrong extents, unmatched destructor markers and an open
final ledger all reject. Parser success proves logical retirement only; parent
normal exit/removal and strict per-card/compiledP2P0 pre/post-health are mandatory.

The new full contract is separate: native_source_upload_gpu_oracle_full.cpp,
native-source-upload-plan-full.json and build_source_upload_oracle_full.py.
Its roster has387 HC plus3 PLE images and rejects subset or duplicate totals.
One-card and24/24 same-card/real-two-card owners are explicit cases. --bounds
static is a later model-call control, mirroring static stage binding then the
four-argument load. Do not silently waive foreign ordinary PLE copies if this
control exposes them.

Header inventory estimates full native images at3.58524GiB:301 ordinaryQ8
matrices plus387 HC and3 PLE source images. Split24/24 is1.81862/1.76662GiB.
These are logical payload counts, excluding scratch, physical USM granules,
allocator retention, runtime/kernel images, loader staging and alias-guard
spacers. Actual per-card allocation peaks/headroom remain required. No experts,
canonical full arena, PLE table, recurrent/KV buffers or inference run are
included. Full390 upload is required before model promotion, but still cannot
replace actual model math/state/coherence/prefix/concurrency qualification.
