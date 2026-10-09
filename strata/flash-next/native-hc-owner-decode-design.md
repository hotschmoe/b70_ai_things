# Exact-source native HC owner and ordinary decode increment

CONFIG -> Pinned Strata fb58e0dbc8399662c0e47c76578c6e878b14f6cf, patch0001
plus0002 overlay, selected UD-Q4_K_XL unchanged. CPU preparation only;
no external checkout edits, GPU discovery, SYCL compilation or model execution.

COMMAND -> Prepare patch0003-sycl-native-hc-source-owner-decode.patch;
sequential apply checks; compile actual runtime-policy header and typed
WeightRef fields in both built/unbuilt configurations with CPU g++17 and
ASan/UBSan in no-device image39992d70. Exercise16 independent-process policy
cases. Reproduce: python3 strata/flash-next/test_native_hc_owner_cpu.py.

RESULT -> Sequential apply passes; metadata/policy CPU compilation and all16
policy cases pass. Default unset/0 accepts both builds. Requested native mode
rejects unbuilt kernels, malformed runtime values, legacy HC_Q8,
HC_Q8_INJECT, QFUSE and GR_V3. Explicit incompatible-flag0 accepts native mode.
Patch SHA256: c03ad7d771f3f6a89cb14f2357a4b653f0e9c8e2aeca53f6d4c4aff5bd4fba79.

VERDICT -> Actual source implements exact-source ownership and ordinary
single-token HC routing, with explicitly rejected prefill/verifier/MTP routes.
This is an intermediate source checkpoint. SYCL translation-unit compilation,
GPU upload identity, model-call execution, arithmetic, full serving and
concurrency remain unqualified. Rejection of unsupported routes cannot satisfy
the campaign's completion criteria. Do not run a full model based on this patch.

## Implemented source behavior

CMake defines STRATA_SYCL_Q8_HC_BUILT through the shared include target only
when primitive compilation is enabled. STRATA_SYCL_NATIVE_HC=1 is a new strict,
default-off option, read at process startup and held fixed for session lifetime.
Unset/0 preserves ordinary old source branches and does not upload native HC.
A changed environment after startup is unsupported; captured sessions cannot
change math contracts by changing an environment variable.

NativeDense owns exact-source images in its existing per-stage allocation
vector. The source walk uploads original Q8_0 down/up bytes and original F32
norm/injection bytes, never the packed BF16 compatibility view. Norm's source
GGUF already folds1+w; no additional fold is made. It checks original types,
selected geometry, source bounds and duplicate publication, then attaches
explicit const-float or const-byte pointers to WeightRef. Shard path, absolute
offset, original shape/type/length and FNV64 source-byte receipts accompany the
pointers. FNV64 is an audit checksum, not a cryptographic model identity;
model/shard identity must still come from the pinned SHA256 campaign lock.
Metadata is published only after upload completion. Allocation teardown stays
owned by NativeDense; tables and captured graphs must not outlive that owner.

The source walk filters active layer ranges before allocating. Only the stage
whose range reaches the table's last block uploads output_hc_* tensors. A
startup audit explicitly requires both four-tensor HC halves for every stage
layer and three output-mixer tensors on the final stage. Missing metadata,
missing source tensors, unsupported shape/type or duplicate source tensors
fails the load. Additional native bytes enter NativeDense::weight_bytes();
actual allocator granularity and placement planner coverage still need auditing
before a full model allocation budget is trusted.

Ordinary block_layer_pre selects native composition for each half. Native mode
forces the unfused execution order: attention HC read, attention compute,
explicit residual write, FFN HC read, router/MoE, explicit FFN residual write.
Both halves use the already-owned F32 xn/lo/gated/rs workspace in BlockBuffers.
Their current injection output is bb.inject as in the old unfused ordering.
Each residual write uses F32 sigmoid and FMA. Native mode never carries a
pending_ffn write into the next layer and never uses inject2 as deferred state.
This initially favors a simple correct ordering over fused launch count.
Ordinary head mixing uses the same native composition with no injection,
after the final FFN write has already completed in queue order.

Native mode dispatches before BF16 kind checks or fused-read selection;
compatibility views cannot serve as fallback. Control vectors remain applied
after completed native FFN writes. PLE retains its placement before layer1
attention; no pending native write remains to interact with that PLE boundary.
No new decode allocations or mutable global scratch are introduced.

Prefill::init, Verifier::init and MtpDrafter::load reject native mode before
registering state, allocating buffers or touching devices. This intentionally
prevents the current separate BF16 prompt and verifier paths from executing.
Those routes must be integrated in the next source increment before ordinary
native serving can be considered. A server that initializes these components
unconditionally will fail startup in native mode; it must not swallow the
error and continue with compatibility HC.

## Required next evidence

Parent coordinator must compile all changed SYCL TUs against the independent
pinned overlay in built and disabled configurations; CPU policy compilation
covers headers only. Validate exact uploaded source bytes and descriptor
ownership on both stages, all single-token block halves, native residual
writes, final mixing, PLE boundary and control-vector ordering. Compare whole
model logits and intermediate HC tensors before treating readable output as
fidelity. Stage handoff must preserve already-written R and avoid an extra
FFN write. Keep source receipt identity and allocator bytes in the test record.

Patch0004 should integrate prefill using1..8-row native chunks and the existing
explicit prefill residual writes, with native HC producing downstream mixed
precision views only after the F32 HC boundary. Verify token-major layout and
rs/xn/lo/gate ownership. Verifier integration should pack each Rt(t) or use
one-row native calls into distinct scratch slices; it must resolve pending
writes explicitly and use the same native final mixer after the last FFN.
Only remove init rejection after every attention/FFN/final route is wired and
negative tests prove incompatible flags cannot bypass it. MTP remains rejected
until its separate tensors/routes are addressed; campaign serving keeps MTP off.

## Exact-artifact source-rank correction

CONFIG -> Fresh F04/current-gguf-inventory.json and native-hc-source-tensors.json
receipts;387 actual HC tensors across48 layers plus head. The initial source
increment incorrectly assumed a rank2 norm from the logical HC layout.

COMMAND -> Cross-check all actual dtype/rank/shape/byte/absolute-offset records
against the pinned model lock and payload receipt. Factor source shape, byte
and bounds checks into CPU-testable native_hc_source.hpp; retain source rank
explicitly and log source shape separately from the normalized row view.

RESULT -> Actual norms are97 F32 rank1[10240] tensors (40960 bytes), with logical
row view[10240,1]. Down/up are97 each Q8_0[10240,320]/[320,10240] matrices;
injection is96 F32[10240,4] matrices. The original norm uploader would reject
the pinned artifact and is superseded. Current0003 SHA256:
2c227ffbf8be91b6061d0aa12487f5ffaee743499675456d09b0e0de67b8b717.
Upload memcpy now uses wait_and_throw() before metadata publication. Source
bounds are checked before forming tensor_data pointers. Original source rank
and shape remain distinct from logical ne0/ne1 view metadata.

VERDICT -> Shape assumption corrected against all actual artifact tensors.
This audit uses the parent's fresh header and payload receipts; it does not
reverify complete shard SHA256. See native-hc-artifact-descriptors.json and
native_hc_artifact.py for exact tensor SHA256 and offset evidence. F32 remains
the native injection source even though actual injections are BF16-exact.
GPU upload identity and model execution remain separate qualification gates.

## Effective stage-bound ownership correction

CONFIG -> Source audit of generate.cpp's NativeDense load calls: the driver
sets static layer bounds for trimmed stages, but omits explicit load bounds.
The previous HC-specific outside()/head test therefore observed defaults and
could upload the complete HC set on both devices.

COMMAND -> Add CPU-testable effective-bound resolution for HC only. Prefer
explicit load bounds; otherwise use the existing configured static bounds.
Keep generic dense and replicated PLE filtering unchanged. Add a native-mode
multi-GPU startup guard requiring an explicit --layer-split and
--trim-stage-weights, so both existing load calls have known configured bounds.
Automatic/untrimmed native multi-GPU loading rejects until another source
increment can defer HC ownership until its split is known.

RESULT -> Current0003 SHA256:
207642b5ed8d7c6a90d4b4783b46aa4f189a98cee84f3454635a3ca2d3d7f57d.
All three successive0003 hashes above are historical snapshots; this is the
current stage-bound snapshot. The shared bounds helper passes7 CPU cases
covering first/final stage defaults, explicit-bound precedence, full-model
loading and invalid bounds. The actual artifact has192 HC source tensors for
layers0..23 and195 for layers24..47 plus head; these are expected allocation
coverage counts, not observed GPU uploads.

VERDICT -> HC source ownership no longer mistakes omitted load arguments for
full-model ownership in the supported explicit trimmed split. Static bounds
remain a startup configuration mechanism; loads must be coordinated serially.
Current generate.cpp and uploader compilation/upload gates remain outstanding.
