# Strict native PLE: remove unused legacy key image

CONFIG -> Current fb58e0 full-source generation through0010; selected GGUF;
CPU-only audit and copied-source object compilation. Existing patches, plans,
builds and source checkouts retained. Current buffered shard3 identity failure
blocks all GPU/model runs even though read-only directIO matches publisher SHA.

COMMAND -> Trace actual PLE setup, ready, scalar, verifier and prefill branches;
prepare new0011 and compile both affected actual SYCL translation units with the
v5 compiler defines/includes, O1 instead of ReleaseO3, no devices exposed.

RESULT -> Patch0011 SHA4219c7fd206bb2667f2ffb0e130c9adcea80c1b01a0bd361c5fc4d6c82a4b06c.
NativeDense ordinary pending-upload loop omits only blk.1.ple_key.weight when
native_hc_requested() is true. eligible()/served_names() remain unchanged: the
native pack's quantized key row must still be skipped by the canonical loader,
retaining its metadata for the separate exact-source owner.

The generate.cpp legacy-key bootstrap guard is also conditional on native HC
being off. Without this companion change, omitting native_data would fail
startup before the strict source fields were installed. The subsequent exact
key/value/convolution type/shape/extent checks are preserved and still reject
missing original images; there is no fallback to a packed BF16 or legacy key.

Actual source_exact consumers do not use key_native_data: PLE ready checks
ple_exact_sources_ok; scalar ple_block dispatches original keyQ8/valueQ8 times
original F32 embeddings before the old branches; verifier and prefill disable
the old batched key/value paths in strict mode, retaining the scalar history
loop. This is why removing the old key image does not substitute model math.

Both affected objects compile, exit0. Receipt:
/mnt/vm_8tb/b70/build/strata-native-ple-legacy-omit-source-20261009/object-compile-receipt.json.
Source patch git apply --check passes against the unchanged isolated v5 tree.
No full new engine build, GPU upload or actual model startup was performed.

VERDICT -> New source increment is ready for a separate full build generation.
The non-native branch retains its existing ordinary key upload and validation.
Full390 original source bytes, exact PLE binding, model-static stage bounds,
logical-free trace and healthy lifecycle must pass before any model promotion.
The shard3 buffered/cache identity contradiction must be resolved first.

Expected strict footprint is300 ordinary native matrices (previously301),
387 HC and3 original PLE source images. Remove27,852,800B from the ordinary
pending image total, leaving3,091,660,800B ordinary and3,821,772,160B all-native
logical image bytes (3.5593GiB), plus independent scratch/runtime/allocator
costs. The full single-owner logical trace expects690 image pointers plus one
shared scratch. Actual receipts must establish these counts; metadata estimates
are not measured VRAM peaks.

The original PLE key source image remains on the layer1 owner. Model static
stage24..48 should have neither the old ordinary key image nor original PLE
source images. New native-source-upload-plan-full-v6.json preregisters this
without modifying the old full-v5 plan, roster or narrowv2 lifecycle evidence.
