#0023 remaining layer0 numerical producer hooks

CONFIG -> NEW draft against frozen20+0021; neither generation, nine0021 files,
reference/packet source nor active V6 source is changed. Source/CPU only. Raw
fused hidden remains UNOBSERVED. Seven reserved fields are now supplied by two
actual GU/HQ producer boundaries plus explicitly DERIVED native-exp evaluation.

COMMAND -> python3 strata/flash-next/test_layer0_producer_capture_cpu_v1.py.
Reconstruct21 then apply23 in a private overlay; compile ACTUAL observer-v2 with
mock SYCL using CPU icpx/no-fsycl/no-device/networknone in image39992d70. No real
SYCL translation unit, library linking, model payload/forward or GPU is executed.

RESULT -> Source reconstruction and actual-helper mock compile pass. Default off
has zero observer allocations/copies.33 fields and actual entry map publish from
one mock frame; both five VRAM/five owning-mirror pointer bindings match. Eleven
shape/mapping/nonce/key controls reject duplicate destination, wrong token,
foreign blob pointer, absent group/coverage, overlap, missing mirror, invalid
expert, out-of-range group count, stale completion, alternate key and wrong
shared shape. Deliberate full scratch overwrite between two producer calls does
not corrupt the first call's captured GU/HQ. This is NOT hardware validation.

## Producer boundary and ABI scope

NEW SYCL-shadow shared_expert.hpp and iq_kernels.hpp retain the shared/CUDA headers
unchanged and add one optional diagnostic callback descriptor to the SYCL host
entrypoints. Core and ALL SYCL callers/kernel libraries must be rebuilt together;
old ABI objects/oracles must not be reused. Explicit arguments avoid TLS across
engine/kernel DSOs. Callbacks are null by default. Host stack descriptors live
through synchronous callback invocation; device lambdas copy only source/snapshot
addresses, scalar extents and binding values, never the host descriptor pointer.

shared_expert_multi observes actual original GU after their native projections,
BEFORE either fused or unfused hidden production can overwrite gate. It captures
HQ after quantization, before down consumes/reuses scratch. Each callback uses
the ACTUAL producer queue. Existing shared fork/join ordering remains unchanged;
callbacks precede the existing join. No new source-mode switch or barrier is
added, and no observer operations occur when the optional descriptor is null.

native_expert_grouped observes actual gate/up after the existing GU kernels and
HQ after actual SwiGLU/quantization, before scratch reuse. It retains default
fused mode and does not read the unwritten2fa raw hidden region. The observed
expert scratch layout is gate0, upfa, hq3fa, with fa=align256(cap_entries*nff*4).
Callback view capacity10/nff640 binds the actual T1 call and its device group
start/count plus ent_tok/ent_dst. Benchmark partial phases are refused only when
diagnostic callbacks are supplied; no benchmark/math flag is changed.

## GPU mapping and coverage draft

A diagnostic-only single work item gathers each call's actual active group
spans. It validates0<=groups<=10 and ordered bounded starts, token0, destination
0..9 and router expert0..511. Output is in ROUTER RANK order, never scratch order.
Entry map columns are [expert_id,ent_tok,ent_dst]. It records actual grouped blob
addresses and VRAM(1)/owning-stage mirror(2) tiers as auxiliary metadata.

Validate group_ptr against the same live residency table: a resident expert's
cache_base+device slot_offset (uniform fallback when offsets absent), otherwise
the owning layer mirror table entry. Missing/foreign pointers cannot publish a
qualified mapping. Do not classify tier from call order: the CPU fixture uses
both tiers in each call. Residency/table ownership guards are preserved.
Legacy GPU staging/rebased pointer routes are not established by this direct
cache/mirror binding; such a mismatch rejects the diagnostic frame rather than
changing model placement or declaring copied staging bytes original.

Per-rank GU and HQ coverage bits detect duplicate/overlapping calls and missing
ranks across actual resident/RAM spans. Shared coverage detects missing/duplicate
producers. Masks/error/auxiliary data clear in arm before every replay; dump checks
all ten masks3/shared3/error0 AFTER current GPU nonce/key. Cached static33 rosters
alone do not prove actual producer coverage. A mapping failure publishes no new
frame. Original four combined0013 graph keys and marker/lifetime guards remain.

## Truthful hidden provenance and bounds

Shared/expert hidden fields are DERIVED: evaluate contract(off)
(g/(1+sycl::native::exp(-g)))*up from actual captured producer values in an observer
kernel. Raw original fused hidden is still UNOBSERVED and metadata explicitly
says so. CPU mock std::exp is only an SDK stand-in, not hardware packet proof.
Actual GPU intrinsic/compiler rounding may affect boundary cases; independent
packet comparison and FP64 SiLU error must qualify it before any conclusion.

All33 reserved fields total7023048 raw bytes/frame; four raw frames28092192 bytes.
One reused snapshot plus256 bytes control/coverage/auxiliary space requests
7023304 bytes, below64MiB. Observer single-task gathers are deliberately bounded
diagnostics, not optimized or clean latency measurements. Model math/RNG/cache
policy and source buffers are unchanged. Readback, graph/USM context retirement,
source hash/sentinel and actual off/on equivalence remain parent gates.

complete_preregistered_layout=true means all33 numerical field observations are
present and their mapping/coverage is valid; it does NOT mean whole first-layer
or model arithmetic is correct. full_model_math_qualified remainsfalse. Original
own-state/router FP64 prefix evaluation, native packet/affine-consumer operator
checks, full native-storage propagation/prefill seams and actual lifecycle must
still pass. Root numerical controller should preserve the26-only21 lane and
recognize33-field23 as a separate immutable generation.

VERDICT -> Source draft and CPU shape/mapping/nonce controls are ready for
independent review. Actual SDK full rebuild, GPU producer coverage, intrinsic /
packet equality,21/23 off/on outputs/logits/state, original reference numerics,
owner/context frees and pre/post health remain unqualified.
