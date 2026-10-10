# Separate source-indexed HC F32 host prototype

CONFIG -> current source35 native HC projection/composition, pinned kernel/source
and current static library hashes in hc-f32-arithmetic35-host-source-plan-v1.json.
Original FP64 Full48 helpers, original evidence and native quantizer are unchanged.

COMMAND -> PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s
strata/flash-next -p 'test_hc_f32_arithmetic35_host_cpu_v1.py'.

RESULT -> 15 CPU tests PASS using existing host libm fmaf and tiny synthetic
operands. C++ helper was NOT compiled or executed. Adapter process invocation
is mocked in its positive unit control; it is not an actual ELF/native result.
Known tests independently establish fused cancellation -2^-46 versus separately
rounded0, RNE midpoint ties, gradual subnormal behavior, onehot original row
algebra, dyadic all-lane sum and lane-strided2^24+1-2^24 cancellation/order negative.
Half-scale signedcode and subnormal-half tests use known exact answers.

VERDICT -> source prototype only; implementation/native/model qualification stays
false. It models only32lane column-strided FMA followed by XOR16/8/4/2/1 F32 sums,
original Q8_0 decode plus F32 operands, and elementwise FMA. It does not implement
HC sigmoid/exp/rsqrt/division, ordinary mix contraction or complete HC/model math.
No tolerance, fitted constants, native values as owned fullmodel inputs, GPU,
Docker, runtime, original payload or git action by the preparing agent.

The C++ process requires IEC559 LE F32, FE_TONEAREST and x86 SSE gradual underflow
(FTZ/DAZ cleared and checked). It refuses nonfinite/truncated/out-of-scope operands
and existing output paths. The Python adapter uses LE bytes, pins explicit host
ELF SHA, checks subprocess scope/rounding metadata and rehashes the helper. It
never discovers model files. Root must compile/link and validate the C++ helper
before any actual-data use; current archives are pinned, without retroactive
original SDK archive-provenance claims.

Future ROOT-only host command, never executed for this preparation:

```sh
c++ -std=c++20 -O2 -fno-fast-math -ffp-contract=off -frounding-math strata/flash-next/hc_f32_arithmetic35_host_v1.cpp -o <NEW-host-helper>
```

Helper ABI: HOST OP N A B C-or-dash OUTPUT source35_hc_f32_v1. OP dot/q8dot
consumes one packed weight row A and LE_F32 input B, returns one F32; C is dash.
OP fma consumes three equal LE_F32 vectors and returns elementwise fmaf(A,B,C).
N is bounded1..10240; projection N must be divisible32. Python run_helper requires
helper SHA and caller-supplied bytes. Original-data admission remains a separate
root responsibility, never inferred from this adapter.

A future separate fresh SYCL fixture can call public hc_q8_0_project_f32 and
hc_f32_project_f32 with synthetic held-out inputs, compare full raw row outputs
against this host schedule and test wrong source/order/weight controls. Public
hc_native_read_f32 exposes rs/xn/postSiLUlo/rawgate/inject/mixed; preSiLUdown needs
a separately labeled projection from the same xn, as the existing composition
fixture does. Do not compare postSiLUlo to the frozen reference's preSiLUdown.
Standalone hc_native_write_f32 combines a still-unqualified sigmoid with FMA;
this prototype verifies the FMA primitive only, not the computed sigmoid weight.

Before claiming intrinsic emulation, root should freeze held-out exp/rsqrt/
division/boundary inputs and independently assess device results, rounding and
subnormals. CPU expf is not automatically equivalent to sycl::exp. Pin source,
image/compiler, binary, queue/device identity and all raw operands. Preserve
complete packet comparisons near quantization boundaries and independently
qualify dtype/contraction before adding a separate composed arithmetic lane.
Any actual SYCL fixture must use GPU leases, strict/compiled pre/post health,
owned allocation/free/teardown and applicable source4 gates. No new SDK/backend
patch is part of this prototype. Keep failed fulloriginal results and preregister
quality gates; passing the implementation-shaped local oracle is not model quality.
