# S83 decode critical-path lead audit (2026-10-15)

CONFIG -> Current source only plus retained S31 mapped159-operation CSV,
final analysis and S56/S61 compact summaries. Primary dense/shared/MTP audit.
Main S79 source is restored; generic03bb executable/objects are stale candidate
artifacts, not current source. No server, build, GPU, payload, raw270MB trace,
model consultation, source patch, repository doc or benchmark execution.
COMMAND -> Direct source reads; crosscheck six Q6 rows in the159-row CSV;
rank the retained dense family sums; inspect existing compiler metadata only.
Exact full-file hashes/paths are source-pins.json. All outputs together <20KB.
RESULT -> ONE source-eligible integer-only lead on default Q6_K C6 decode:
remove unneeded signed-byte saturation from the recenter-by32 operation.
VERDICT -> HOLD implementation/build/GPU. First justify compiled instruction
cost from the EXISTING selected AOT kernel; no new test or compiler authorized.

## Coverage and ranking: historical service, never promised savings

S31 reviewed cycles5..9 are an earlier batch2/depth2/headpairON binary.
Dense excludes vocabulary heads and experts. Separate device pools/clocks and
host spans must not be added into a critical-path saving or idle budget.

| Dense format family | Calls/cycle | GPU0 service ms/cycle |
| --- | ---: | ---: |
| Q6_K |71|3.8256372|
| Q4_K |45|2.3121056|
| IQ4_XS |67|2.0693470|
| IQ3_S |55|1.9335818|
| BF16 |130|1.2114892|
| Q5_K |14|1.0027652|
| Q8_0 name-inferred |38|0.7442326|
| IQ4_NL |40|0.5820252|
| other/unclassified |8|0.2178506|

Total468calls/13.8990344ms. Q6 covers27.5245% of this dense service pool,
not27.5% of wall. All observed Q6 calls are C6 default A2/RPW1. S31 grid640
is29calls/1.750572ms, grid2560 is14/1.1578066; together76.02% of Q6 service.
Other grids3072,1536,128,160 contribute28calls/0.9172586ms. Grid is workgroup
count, not matrix row count. Source default has four rows/workgroup, so aligned
grid640 is M2560; the retained source/pack join identifies K6144->M2560 output
projection. Grid2560 is M10240 (K2560 attention QKV source/pack geometry).
No trace pointer/layer identity is invented from a symbol/shape alone.

S31 HC7.1990958ms/447calls and primary experts7.7380890ms/144calls are larger
other pools, but their tested mechanisms do not justify a repeat. MTP's Q8
name-inferred dense pool is0.7442326ms; weighted RMS/Q8 fusing is much smaller:
S61's shared-name norms8calls/0.0388726ms and grid10 quant14/0.0355352ms
also have non-MTP consumers. It is not a new large150/C2 candidate.

## Exact seam and byte algebra

native_mmvq.dp.cpp1541..1639: native_mmvq_q6k_wide_kernel<NCOLS,RPW,...>.
Default launch_q6k_wide6 selects A2,SG32,RPW1 at1708..1736. One subgroup owns
one output row; four subgroups/workgroup; six token columns share each weight
decode. Each lane g=lane&7,sub=lane>>3 visits kbx=sub,sub+4,... .
At1606..1607, four p values each make TWO unsigned6-bit packed-byte words:
low-nibble|high2bits and upper-nibble|high2bits. Masks ensure every byte b is
in0..63. The code calls dpct::vectorized_binary<sycl::char4>(v,0x20202020,
dpct::sub_sat()) on each, giving8 packed calls=32 byte operations per lane
per K iteration (source count, not dynamic ISA count). dpct/math.hpp604..608
calls sycl::sub_sat;621..632 bitcasts the input/output vector representation.
Each signed input byte0..63 minus signed32 yields[-32,31]; saturation cannot
activate. It is not valid to replace packed subtraction by v-0x20202020:
negative low bytes would borrow from neighbors.

A candidate unsigned32 expression is:
  center(v) = ((v | 0x80808080u) - 0x20202020u) ^ 0x80808080u
Proof per byte: OR gives128+b; subtract32 gives96+b, no byte borrow because
every source byte>=128>32. For b<32, XOR128 gives224+b (signed b-32); for
b>=32 it gives b-32. No carry/borrow crosses a byte. All32 output bits match
the existing signed-byte saturating subtraction for the allowed inputs.
Use uint32 arithmetic and an explicit bit-preserving cast into DP4A input.
This is source algebra, not an executed CPU finite-domain or GPU proof.

Only these integer recenter expressions would change in a future PRIVATE
C6/RPW1/A2 leaf. Keep original210-byte Q6 layout, original aligned ql/qh loads,
original Q8 bytes/scales, lane/k order, eight DP4A calls per token and exact
FP source expression at1617. Same six accumulators and XOR16..1/store-lane0.
No FP contraction policy change, plane allocation, repack, FP16 activation,
SG16/DPAS/SLM, preload distance or persistent engine change is proposed.

## Why this is conditional rather than a new GPU campaign

S56 retained C6/RPW1 image hasGRF128/SIMD32; spill/scratch fields absent
(not measured zero). C6/RPW2 has7936B spill/8192B scratch, a known liability.
These metadata do NOT show the centering instruction lowering. The compiler
could already simplify the builtin, or three packed integer operations could
be worse than native byte operations. No cost or speed follows from source
operator count. This audit deliberately does not extract/disassemble a new
image or compile a private variant.

Cheapest next discriminator IF root selects it: decode ONLY the existing
C6/RPW1/A2 image already indexed by S56, using its pinned historical image
identity/offset and existing ELF/ocloc tools. Confirm actual sign-extension,
clamp/saturate/pack operations at this seam; if saturation has already folded
or no material avoidable instruction sequence is evident, close HOLD without
implementation. S56 image771 starts39553776,43432B in frozen S55 executable
bb066900...; index and current source pins must be revalidated before reuse.
Do not silently use the stale generic03bb binary or rehash all common archives.
This is a proposed offline prerequisite, not execution authorization.

ONLY if that static prerequisite supports a distinct implementation later:
reuse the existing S44 actual Q6 out-projection K6144/M2560 and QKV
K2560/M10240 fixture/adapter patterns; private default-same-math leaf, no main
TU rebuild. Strict full public-anchor/direct/own initial+changed graphs/repeat
outputs, Q8 read-only bytes/weights/guards before any balanced completed-wall
comparison; exact DOT/ISA binding and normal leased health/cleanup. No new
reference tolerance is needed because integer bits/FP sequence must remain
identical, but actual compiled byte identity is still mandatory. No C1/other
formats/knob sweep/serving run or output-ID framework. No test is queued here.

## Distinct from retained negatives; fusion/readiness conclusion

S15 Q6 prefetch2 median1.1281 and RPW2/4 1.1057/1.5087 were rejected;
this changes neither distance nor rows. S44 packed planes changed storage,
parallel reduction and resources and regressed warmed cases; no planes here.
S65 IQ3 SWAR changed a different grid/sign decoder and smaller1.934ms pool;
its mixed results do not establish this expression's instruction cost.
S75 SG16 DPAS changed output/K parallelism and added SLM/barriers; absent here.
S76 expanded IQ3 planes, S73 MORE, S50 HC graph, S55 head RPW2, S79 overlap
and S80 prefill grouping stay closed; none is renamed as this lead.

Shared gate/up already feed native_swiglu_quantize_q8_1 (shared_expert.dp.cpp
313..352); native_mmvq_pair3700..3711 is Q8-only and requires the old exact
multi path, so it cannot simply pair heterogeneous Q6/GDN projections.
verify.cpp1030 and1051 share xq already; moving gate before conv would change
scheduling but not prove true fusion or a readiness gap. QFUSE's dormant
batch-GDN missing-output/finite-helper blockers remain as S61, not runtime
failures or a flag to enable. No small larger fusion/readiness change is
justified from these summaries. The one above is an integer-decode source
prerequisite; implementation and all runtime claims remain HOLD.
