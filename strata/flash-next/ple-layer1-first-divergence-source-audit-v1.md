# Layer1 first major difference: readonly PLE reference audit

CONFIG -> Preserved actual original48 prefix1 exploration report and frozen29
consumed source, strict originalPLE source_exact Q8/F32; no weight payload reads.
COMMAND -> python3 strata/flash-next/test_audit_ple_equations_cpu_v1.py
RESULT -> Six independent source-derived synthetic controls pass. Existing own
layer0FFN and layer1 prePLE input arrays are bitwise identical; ownPLE first8
history rows are zero and its newest row is the normalized value. Original
embedding equals native bitwise; first major observed endpoint difference is
layer1 postFFN. Detailed artifact hashes/metrics are in the separate readonly audit.
VERDICT -> No structural PLE formula/layout bug established. No native attribution
or model qualification follows from this audit.

Actual source evidence (fresh29 source consumed by finalized V9):
- verify.cpp880-969 runs PLE at the beginning of layer1, after layer0FFN and before
  HC attention. Native HC has already materialized the previous FFN write.
- generate.cpp2985-3048 binds original Q8 key/value, original F32 convolution and
  F32 norm pointers; native PLE log records original tensor offsets/byte counts.
- ple.dp.cpp336-345 rejects incomplete exact sources/preprojected routes. Lines
  421-425 and492-496 project original Q8 weights against F32 embedding directly.
- ngram.cpp70-102 implements uint64 wrapped products/XOR, oldest-first previous
  tokens, EOS substitution then per-head modulus and offset. Lines106-117 decode
  IQ4 split-halves. Lines507-550 gather head-major160 values into2560 columns.
- native_ple_postops.dp.cpp51-100 grouped dot uses materialized F32 multiplication,
  invsqrt2560, signed sqrt(abs(score) floored1e-6), then sigmoid.
- Lines103-114 broadcast value[d] by gate[stream]. Lines340-341 and379 apply
  grouped2560 RMS weights to key/query and gated conv input respectively.
- Lines117-156 use history[c*9+3*k], original weight[c*4+k], normalized[c] for
  the current last tap, SiLU of the sum, then hidden+gated+activation.
- Original helper's logical[9,10240] and transposed channel-tap weights match this
  physical indexing. Its selected rows0/3/6/current, normalized new history and
  explicit direct-gated residual contribution are present. No duplicate write,
  missing SiLU/directgate, head interleave, endian or conv transpose was found.

The frozen original scalar helper uses FP64 reductions and fewer materialized
F32 intermediate products. The new audit_ple_equations_cpu_v1 source explicitly
materializes source F32 square/dot/conv products and stage stores, but still
estimates native subgroup/FMA reductions and exp/sqrt with CPU math. Six controls
reject wrong current tap, transposed channels, missing gated residual, adjacent
history/dilation and nonfinite/wrong geometry. At one bounded synthetic fixture,
new vs frozen result NMSE is below1e-12. This is not a bound for the actual weights
or evidence that rounding cannot affect routing; it does not explain or correct
actual layer1 NMSE1.172092. Frozen reference and actual run remain unchanged.

Metadata semantic issue remains separate: selected GGUF EOS248046/BOS248044,
while BOTH this frozen owned lane and native ngram.hpp use248044 as the padding
sentinel. Hence it cannot explain THEIR observed difference on this first token.
It still needs original llama/HF semantic evidence before either can be called an
original tokenizer/model semantic oracle. No sentinel change is proposed here.

Current observability gap: V9 exposes layer0 internals and postFFN48 endpoints.
Layer1 combines PLE, GDN/HC attention and FFN, so the observed endpoint cannot
identify which producer first differs. P30 adds native layer1 prePLE input and
postattention; an additional bounded PLE producer witness should expose actual
hash rowIDs/prev tokens,2560 table embedding, Q8 key/value projections, gate,
normalized conv row, conv activation and postPLE residual. Compare those to own
computed values without feeding observations into the reference. If postattention
agrees but postFFN differs, capture actual layer1 router IDs/weights/GU/HQ/combine.
Original ordinary norm/table payload readback remains unqualified until observed.
No unobserved scope is converted to a zero or numerical pass.
