# Source35 native HC arithmetic contract audit

CONFIG -> Read-only source35/C113 SDK source at
`/mnt/vm_8tb/b70/build/strata-native-hc-engine-20261009T233718Z-i6s74tbl/source`,
normal graph, nativeHC1/QFUSE0, original Q8_0 HC projections and F32 activations.
COMMAND -> Read/hash HC dispatch/composition/projection, frozen owned HC and
encoder; inspect consumed build.ninja; run four synthetic CPU projection rows.
RESULT -> Native source specifies F32 lane/FMA/tree arithmetic; owned reference
uses FP64 dots and selected F32 stage stores. Those are distinct arithmetic lanes.
VERDICT -> No new quantization/datatype bug or justified kernel/reference fix.
No GPU, Docker, model payload, runtime, SDK, git or frozen helper modification.
Only this evidence document was written; all math/quality thresholds remain pending.

| Tag | File (SDK-relative unless repository path shown) | SHA256 |
| --- | --- | --- |
| P | `sycl/src/kernels/hc_native_projection.cpp` | `40a14446456199a4532e9e52b78c8131944afbe7c1615e75a6231e850cb2c025` |
| C | `sycl/src/kernels/hc_native_composition.cpp` | `c63ce71dee3572daca974f03587db54ade1d9ec842176bd83bb81dc7d22bb83f` |
| D | `sycl/include/strata/core/native_hc_dispatch.hpp` | `e6ebfc7da2e4e66af349413e50acc134463f02488ed156b1ada9b5ac47d047ef` |
| V | `sycl/src/core/verify.cpp` | `cdd837fd3dd31873094587e549623bae887765c48d8c4de7f542185f52c4d84f` |
| O | `strata/flash-next/full48_owned_composition_storage_v1.py` | `9f98e1b7b7088dc0b12ec0474bcad5660f60878d23ecd24cfabc3032661d1d28` |
| Q | `strata/flash-next/independent_q8_1_activation_v1.py` | `b58d73282051bfa301ebb6f781d5d59c63128d9074a42d53cf4d9ff05d75a714` |
| B | `/mnt/vm_8tb/b70/build/strata-native-hc-engine-20261009T233718Z-i6s74tbl/build/build.ninja` | `717e840d054d5804705dfaa26250bb72a56c972e2cbdb3e845ab9a9ee2137565` |

## Established source contracts and reference differences

Dispatch D:8-25 binds norm/inject originalF32 and down/up originalQ8_0 geometry
10240->320->10240, four streams. V:986-1007 submits nativeHC separately per row,
including T2 windows. No HC Q8_1 activation or BF16/F16 projection substitution
occurs in this lane; gate scratch is reused only after ordered mixing.

| Stage | Native source | Frozen owned HC O:41-70,121-131 |
| --- | --- | --- |
| Norm | C:29-36: each32lane has80 F32 square FMAs; XOR16/8/4/2/1 float reduction, division by2560, epsilon, sycl::rsqrt; xn=(R*norm)*rs storedF32. | F32-rounded squares, NumPy FP64 mean of those squares, F32 mean store, mathematical reciprocal sqrt and selected F32 products. No native reduction/rsqrt emulation. |
| Down/up | P:30-48: half scale widened tofloat, signed integercode, scale*code weight; each lane accumulates columns lane+32j with explicit sycl::fma, XOR16/8/4/2/1 float sums; outputfloat. Down has320 inputs per lane; up has10. | Original exact dequantized weights dot FP64 representation of F32 input via NumPy/BLAS; only final output castF32. No32lane sum/FMA schedule. Q8_0 scale/code algebra is unchanged. |
| Low/SiLU | C:12,41-43: down F32 buffer overwritten with (down/4)*sigmoid(down/4). Sigmoid uses sycl::exp(float), float1+exp and division. | f32(down/4), then f32(low*f32(sigmoid evaluated with NumPyexp/F64))). Device intrinsic intermediate rounding differs. |
| Gate | C:44-45: up projection output is rawF32 gate; sigmoid is applied later. | Raw up FP64 dot/finalF32; not pre-sigmoided either. |
| Injection | C:46-48: F32 originalweights/F32 inputs, same32lane explicitFMA/XOR projection. | OriginalF32 weights exactly widened and FP64 dot/finalF32. |
| Mix | C:49-53: stream order0..3, float sum+=xn*sigmoid(gate), divide4, storeF32. | Explicit F32 sigmoid result, F32 product, F32 sum after each stream. Ordinary multiply/add contraction in the compiled native expression is not established by reading C++ alone. |
| Residual write | C:23-26,64-67: float weight2*sigmoid(inject/4), explicit sycl::fma(block,weight,R), F32 store. | O:131 rounds block*weight separately before addition. This is a concrete fused versus separated operation distinction. |

The Q8_0 half-scale times signedcode has the same real weight meaning in both
lanes. Persistence/output buffers are F32; FP64 reference accumulation is a
mathematical estimate, not evidence that the device uses FP64. HC uses ordinary
sycl::exp, not GDN's native::exp. Exact device exp/rsqrt/division results,
denormals, compiler lowering and ordinary-expression contraction remain
unqualified. B pins -O3/-DNDEBUG/C++20/-fsycl/subgroup32/per_kernel/
-fp-model=precise for both HC objects; flags alone do not establish all resulting
instruction sequences or intrinsic bit patterns. No disassembly claim is made.

Capture semantics matter: V:1009-1015 attn_hc_low is post-SiLU, while O:128
returns down before SiLU. The existing native_hc_composition_gpu_fixture.cpp
labels down_pre_silu and lo_post_silu separately and reconstructs a FP64 oracle;
that historical fixture's thresholds are not a new full-model acceptance rule.
Wrong capture-point comparisons must fail before assigning arithmetic error.

## Incoming mixed activation and Q8_1 discontinuity

Root-reported NUM10/ownedL0 prefix4 observations (not recollected in this audit):
normalized residual equal; mixed NMSE3.4e-15. Block35 amax changes oneF32 ULP,
2.031032085 owned versus2.031031847 native. Index1123 itself is equal, but
x/dF32 is71.499992 versus71.5, so code71 versus72 while storedhalfD andS coincide.
Conditional nativeMixed plus owned incoming GDN state then gives matching full
packets and tiny qkv/z/block-output errors. This localizes that case's incoming
quantization seam; it does not qualify all history/GDN/model mathematics.

Q:19-28 computes dF32=amax/127 and rounds x/dF32 before Q:39 stores halfD/S.
The current native-contract encoder already encodes that rule. Half header
equality is insufficient: the complete codes plus header must be compared.
Do not choose codes using storedhalfD, alter midpoint rounding, or substitute
nativeMixed into the independent owned fullmodel to conceal upstream differences.
Small continuous errors can cross a discrete quantizer boundary; their small
NMSE does not bound the downstream head/logprob/argmax impact.

CPU-only synthetic demonstration: seed3510, four10240-column rows, Q8_0 codes
-127..127, exact half scale0.0625, randomF32 input[-2,2]. CPU libm fmaf +32lane
XOR schedule differs from FP64-dot/finalF32 in all four rows. Case0 gives
784.78564453125 versus784.7859497070312. This establishes that source-declared
schedules can differ without a decoder/model-byte bug. It is not a native GPU
oracle, intrinsic equivalence result, tolerance selection or actual-input fit.

## Defensible next gates

Keep the frozen owned mathematical reference and its failed fullmodel results.
A separate implementation arithmetic oracle can independently decode original
weights and model explicit F32 FMA/lane/tree steps on frozen inputs. It must
first validate exp/rsqrt/division and subnormal handling against the exact compiled
native primitives; CPU expf or a device observation fed back as the expected
value is not an independent proof. Trace every matched intermediate and complete
Q8_1 packet; target actual boundary cases plus held-out rows/contexts, with
wrong weight/order/packet negative controls. Declare any approximation and
unresolved contraction rather than assigning bitwise authority to a CPU port.

For a pure performance change, unchanged compiled-control repeatability and
full-head/phase/packet equality remain separate prerequisites. For model quality,
use the independently owned original lane with preregistered evaluation inputs,
metrics and limits, including probability/PPL/task agreement across histories.
No tolerance is inferred from this one mixed ULP or widened after observing
prefix4 head drift. Passing a source-shaped local oracle cannot replace model
quality, cache/history isolation, concurrency, health, teardown or source4 gates.
