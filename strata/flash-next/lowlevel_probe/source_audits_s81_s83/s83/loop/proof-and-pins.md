# S83 full-loop source supplement (2026-10-15)

CONFIG -> SOURCE PROPOSAL ONLY. No compile, payload, GPU, ISA extraction,
finite-domain execution or main-source edit. Supersedes only the recentering
expression proposed in frozen decode-criticalpath-s83-prep/findings.md.
COMMAND -> Copy the exact current complete native Q6 wide function, replacing
only its two recenter-by32 lines with root's smaller unsigned add/XOR identity.
RESULT -> complete-candidate-function.txt is the FULL original function,
including A2 loads, all6 column activation arrays, original DP4A and FP ledger,
SG/lane/store mapping, with ONLY TWO source lines changed. Inverse replacement
reconstructed the original function exactly in this static generation step.
VERDICT -> SOURCE_ALGEBRA_ELIGIBLE; HOLD implementation/build/GPU pending
actual compiled cost in the existing baseline image. No new test queued.

## Exact pins and current-source binding

Current full source: /mnt/vm_8tb/b70/build/strata-ab/src-ep/sycl/src/kernels/cuda/native_mmvq.dp.cpp
SHA256 81558e95f721413a75c3d3ec6ed7cb5eee766e4f662080cf500c7985090f03e2
Original function SHA256 f7f88106ed042051b422f82a9bb326eb89d107e2b3f6a308a05cf5ffe1589e5d, bytes4684.
Candidate function SHA256 ec1b2c6c33faf7902f1a707842257784b647891d96c7caf484a30ac9e47fbcbe, bytes4672.
Helper source: /mnt/vm_8tb/b70/build/strata-ab/src-ep/sycl/include/dpct/math.hpp
SHA256 7c1b7175dbe29a43bff30a94381131096437d88415881a8a8c0704de53394c58
Vector helper: /mnt/vm_8tb/b70/build/strata-ab/src-ep/sycl/include/dpct/detail/math_detail.hpp
SHA256 93b193a0ea4c6c53de6d44e95b60e244b211bf1508a6862c6f79e43ca691fc58
Earlier source+retained159-row CSV pins are in frozen S83 source-pins.json;
manifest416ab692fd75dd2b624534454b9e0d64c5a53671d9177e042761ae6862697f95.
Current kernel source is read only; no stale generic executable used as truth.

## Reconstructed-byte and intrinsic signedness proof

At current source1606..1607 each packed word E is OR of terms masked by
0x0f0f0f0f and0x30303030. For EVERY byte, low4bits<=15 and disjoint high2bits
<=48; bits6/7 arezero, hence unsigned E byte u lies0..63. This includes the
highest byte: arithmetic right-shift fill bits are removed by these masks.
The existing original packed construction expressions remain text-identical
inside uint32_t(...); this proposal does not alter their shift/load semantics.
Original vectorized_binary<sycl::char4> reinterprets packed bits as signed
8-bit elements; sub_sat functor invokes sycl::sub_sat(x,y). Input bytes0..63
are positive under signed8 interpretation, and subtract32 gives[-32,31].
No signed8 saturation activates. detail/math_detail.hpp19..36 dispatches the
vector-call overload when available, otherwise invokes the same functor for
each element; both retain this signed-byte subtraction contract.

For each byte u, u+0x60=u+96 lies96..159, so no byte carry occurs in the
unsigned32 addition E+0x60606060u. Highest-byte result also<=159, so no lost
carry above bit31 matters. XOR0x80 gives u+224 for u<32 (signed u-32), and
u-32 for u>=32. Thus all32 output bits equal four signed-byte u-32 results.
The final sycl::bit_cast<int> preserves those bits for signed DP4A operands;
it adds no numerical conversion or FP instruction in the source contract.
Actual compiler support/lowering and strict byte identity remain UNMEASURED.

The two lines occur under unrolled p=0..3, therefore8 packed recenter calls
per lane/K iteration in current source (32 byte subtractions). Candidate has
8 unsigned packed additions and8 XOR expressions in source. These are NOT
ISA/dynamic instruction counts, register forecasts or a claim that sixteen
scalar operators beat native vector saturation. Compilers may already fold
away the saturation or select an equally cheap native byte sequence.

## Unchanged ledger/data path and minimal discriminator

All RPW/COLS loops, weight loads, Q8 loads/ds, scale products, eight DP4As per
token, source FP multiply/add expression and XOR SG/2..1 tree remain exactly
as current source. This is NOT RPW2, prefetch2, SG16, packed Q6 storage,
expanded IQ3 planes, XMX or a template/math policy change. It differs from
S65's IQ3 table/sign SWAR and addresses historical Q6 dense71calls/3.8256372ms,
not a claimed wall saving or proof of today's dispatch.

Retained S56 small artifacts contain only Q6 resource YAML/index, NO decoded
centering ISA body. Current C6/RPW1 metadataGRF128/SIMD32 and absent spill/
scratch fields do not justify this integer transform. Root may later select
ONE offline existing-kernel decode from its pinned resource index. If actual
centering instructions are already folded or do not show a materially avoidable
sequence, close HOLD; no private adapter/build/GPU solely for operator counts.

Only after actual static support and a separate root GO would a private
source adapter preserve public original anchor, original A2/SG32/RPW1/C6
geometry and exact original Q8/weight operands, with full strict outputs,
immutable bytes/guards, changed-input own graphs and bounded completed-wall
comparison. Source text proof is not an executed device proof. No tolerance,
serving flag, build or GPU authorization follows from this supplement.

Exact unapplied function diff:

```diff
--- current/source-function
+++ UNAPPLIED/s83-source-proposal
@@ -55,8 +55,8 @@
             int vi0[4], vi1[4];
 #pragma unroll
             for (int p = 0; p < 4; ++p) {
-                vi0[p] = dpct::vectorized_binary<sycl::char4>((vl[p] & 0x0f0f0f0f) | ((vh[p] << 4) & 0x30303030), 0x20202020, dpct::sub_sat());
-                vi1[p] = dpct::vectorized_binary<sycl::char4>(((vl[p] >> 4) & 0x0f0f0f0f) | (((vh[p] >> 4) << 4) & 0x30303030), 0x20202020, dpct::sub_sat());
+                vi0[p] = sycl::bit_cast<int>((uint32_t((vl[p] & 0x0f0f0f0f) | ((vh[p] << 4) & 0x30303030)) + 0x60606060u) ^ 0x80808080u);
+                vi1[p] = sycl::bit_cast<int>((uint32_t(((vl[p] >> 4) & 0x0f0f0f0f) | (((vh[p] >> 4) << 4) & 0x30303030)) + 0x60606060u) ^ 0x80808080u);
             }
 #pragma unroll
             for (int j = 0; j < NCOLS; ++j) {
```
