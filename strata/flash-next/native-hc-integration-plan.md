# Native HC composition and model integration gate

CONFIG -> Read-only Strata source fb58e0dbc8399662c0e47c76578c6e878b14f6cf;
selected UD-Q4_K_XL identity; no external source edits, compiler GPU access,
or model execution. Patch0002 depends on patch0001 (SHA2567edd8631).

COMMAND -> Inspect native_dense.cpp, layer.cpp, verify.cpp, prefill.cpp,
gr.dp.cpp and CUDA fused_gr.cu; prepare primitive composition patch; apply
both patches to a temporary source overlay; execute CPU descriptor checks
with ASan/UBSan inside runtime image39992d70, no devices exposed.

RESULT -> Patch0002 provides hc_native_read_f32 and CPU-testable preflight.
Both patch apply checks pass. CPU preflight passes32 valid configurations and
1120 invalid descriptors, covering1..8 tokens, optional pending writes and
final no-injection reads, all required pointer/capacity fields, geometry,
nonpositive/nonfinite epsilon and incomplete optional groups.
Patch SHA256: bde6376706e3cda1cd2c5b74d0e4f18aa7aa948233436b9530294a95bfd5910b.
Reproduce: python3 strata/flash-next/test_native_hc_composition_cpu.py.

VERDICT -> A concrete composed kernel increment exists, default-off under the
existing STRATA_SYCL_Q8_HC_PRIMITIVES CMake option. It has NO model call-site
wiring, original-weight uploader, SYCL compile receipt or GPU numerical
qualification yet. This is progress toward native fidelity, not proof of it.
Model runs must remain gated until all routes and ownership below are wired
and qualified. Existing STRATA_HC_Q8=1 is still unsafe as an activation path.

## Exact composition contract

For tokens t, HC streams c and embedding coordinates d, D=HC*N:

1. When a prior block output is pending, write R'[t,c,d] =
   fma(bo_prev[t,d], 2*sigmoid(inj_prev[t,c]/HC), R[t,c,d]).
   A separate ordered launch finishes the whole write before normalization;
   R_out may equal R exactly. Without a pending write, R'=R.
2. rs[t,c] = rsqrt(sum_d R'[t,c,d]^2 / N + eps).
   xn[t,c,d] = (R'[t,c,d] * original_f32_norm[c,d]) * rs[t,c].
   RMS is per HC stream over N, not over D and not over weighted xn.
3. down[t,k] = sum_i exact_dequantized_Q8_down[k,i] * xn[t,i].
   lo[t,k] = silu(down[t,k]/HC). Retain F32 lo.
4. gate[t,i] = sum_k exact_dequantized_Q8_up[i,k] * lo[t,k].
   mixed[t,d] = sum_c xn[t,c,d]*sigmoid(gate[t,c,d]) / HC.
5. For block halves only, inject_out[t,c] =
   sum_i original_f32_injection[c,i]*xn[t,i]. Final mixer has no injection
   tensor/output. No injection sigmoid or division belongs in this projection;
   those belong in the eventual residual write in step1.

Patch0002 uses fixed32-lane subgroup reductions with F32 FMA for projections
and RMS. The mean is ordered by c. These are declared arithmetic choices,
not bitwise equivalence claims against another reduction tree. Multiplication
ordering follows the CUDA fused reference for normalization. GPU validation
must compare every intermediate and final outputs to exact-source CPU FP64
references and declared F32 tolerances. Existing primitive tolerances are
proposals; composition tolerances need preregistration before measurement.

The complete descriptor and in-order queue are checked before any submission.
Invalid descriptors return false with zero submissions. Submission/execution
exceptions propagate and abort the request; callers must never retry through
BF16 on partly updated residual state. The caller must prove USM accessibility,
queue context/device ownership and pairwise buffer nonoverlap, except exact
R_out==R. Preflight does not establish these caller contracts or byte identity.
The32 CPU valid cases intentionally use dummy pointers and test only preflight.

Workspace per token: xn[D], lo[LR], gate[D], rs[HC];83216 bytes at the
selected geometry. Mixed[N] and injection[HC] outputs are additional. No
allocations, barriers, host waits or global stage/session state are added by
the composition. One in-order stream orders its launches. Runtime exceptions
may leave earlier submitted work; they are fatal for that request.

## Required original-source owner

Add a SYCL-specific NativeHcOwner to each stage, with a map from exact tensor
name to source-type descriptor. Keep device/context, original GGUF shard,
tensor offset, type, dimensions, byte length and source SHA256 receipts.
Upload the original Q8_0 bytes for down/up and original F32 bytes for injection.
Norm may borrow an existing original F32 upload only after proving byte
identity and lifetime. Own every new device allocation through teardown and
account its bytes in placement budgets; tensors beyond the stage layer range
must not be uploaded. Handle final output_hc_* only on the head stage.

Do not infer original F32 injection fidelity from NativeDense's comment that
its values are BF16-exact. Verify actual tensor bytes. Do not use q8_0_of(),
STRATA_HC_Q8_INJECT, packed BF16 data or a hc_q8 void pointer as native F32
ownership. A startup audit must require every attention/FFN norm/down/up/inject
for the active stage and head norm/down/up for its head; reject missing types,
wrong shapes, unsupported tensors and partial upload before serving.

Use one new strict runtime flag, STRATA_SYCL_NATIVE_HC=1, default off. Requesting
it in a binary built without the primitive option must be a startup error.
Reject legacy STRATA_HC_Q8 and STRATA_HC_Q8_INJECT simultaneously enabled.
Reject MTP/speculative execution until its separate routes are integrated.
Select native HC before GR_V3, STRATA_QFUSE, PF_HCDOWN, HCD_EXACT and HC_UPMIX
branches so none can silently route native mode back through BF16. Do not let
pointer presence select a compatibility path. Apply the policy to every stage.

## Route inventory and next source edits

| Route | Current pinned behavior | Required native edit |
| --- | --- | --- |
| native_dense.cpp:318 | STRATA_HC_Q8 optionally uploads Q8 projections; injection may be requantized | Separate typed owner and original F32 injection uploader; atomic startup audit |
| layer.cpp:1608,1634,1657 | Single-token attention/FFN gr_read or fused_gr_read with BF16 weights | Native composition for both halves; preserve inject versus inject2 ownership and pending_ffn |
| layer.cpp:1450 | Ordinary final mixer always gr_read BF16 | Native final read with no injection after the final FFN pending write is committed |
| layer.cpp:1713 | block_layer_post decides residual write using fused capability | Explicit native pending-write policy; exactly one write per half and no stale pending flag |
| prefill.cpp:2772 | Separate RMS and BF16 down/inject/up/mix matrix paths | Chunk native composition1..8 rows; no xn16/lo16 inputs; preserve chunk token layout |
| prefill.cpp:4126 | Prefill writes may fuse with next-half normalization | Disable that fusion in native mode initially; do explicit write then native RMS; reset normed bookkeeping |
| verify.cpp:900 | Attention/FFN FusedGrArgs q8 pointers passed to fused SYCL path | Native composition for each contiguous chunk or pack noncontiguous Rt(t); retain independent token scratch |
| verify.cpp:1474 | Final Q8 branch changes pending-write handling and calls fused_gr_read_multi | Resolve whether prior FFN write is committed, then choose explicit pending native read or ordinary native read |
| fused_gr.dp.cpp | Existing fused entrypoints do not establish native Q8/F32 contract | Typed native dispatch or caller bypass; reject native calls into unqualified entrypoints |

Prefill constructs downstream mixed_h/mixed_bf forms for other model operators.
Convert native mixed only at those existing operator boundaries; native HC's
xn,lo and projection inputs remain F32. Allocate native scratch in stage/session
arenas rather than globals. Batch chunks across independent requests cannot
share live xn/lo/gate/rs or pending injection state. Verify Rt(t) layout before
passing contiguous token-major descriptors; pointer arrays are not contiguous
by implication. Repeated/grouped verifier windows need explicit gathering or
one-token launches and must share the exact same mathematical implementation.

## Remaining qualification gates

First compile patch0002 in an independent pinned overlay without devices, under
parent lifecycle ownership. Then run composition fixtures under bin/gpu-run:
all1..8 rows, two sessions, separate and in-place pending writes, final reads,
finite cancellation/extreme signed Q8 inputs, F32 injection that BF16 changes,
guards/source immutability, exact repeats and chunked-versus-one-row bytes.
Compare R',rs,xn,down,lo,gate,inject and mixed against exact-source CPU math.

After route integration, compare one-row verifier versus ordinary decode,
prefill-to-decode transitions, large-prefill chunk boundaries, both block
halves, final pending FFN/head handling, two independent sequences, repeated
history and return to one stream. Qualify one card before two-stage handoff.
Require original-weight upload receipts and negative tests for missing native
tensors, wrong type/shape, unsupported MTP, incompatible flags and unbuilt
native mode. Lifecycle health/teardown and model full-logit/coherence gates
remain required before performance work or any shelf promotion.

Source SHA256 audit (pinned revision above):

- sycl/src/core/native_dense.cpp: 43168a833d0ce0be9c529f3895ccfd4fdc575dc53f71932bc2a423e7173a152f
- sycl/src/core/layer.cpp: 4c7ffa37a36d2370882cb961b4fd09aeb0f4228c97b6d9d1d9ed2d1856488d94
- sycl/src/core/verify.cpp: fadbf789353c67e9efb18eb7d37940ec12c2a4fd0ea067e290b531f152f6c6b0
- sycl/src/prefill/prefill.cpp: 890e964f5583211fef89a4fabf030f75e230f6785584a1f0d37a3d8b471839f3
- sycl/src/kernels/cuda/fused_gr.dp.cpp: 0b6272f7b9d54dc7889ab162a5d36c4b7fe95b14968d254e6d0567ad540073cf
