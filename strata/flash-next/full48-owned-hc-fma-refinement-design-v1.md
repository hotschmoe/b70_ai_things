CONFIG -> NEW HC-only independent original48 candidate. Original route-aware
V2 composition and its GDN/QSA/FFN/PLE objects, zero states, routing, activation
packets and output-head projection remain inherited. Only hc_read/hc_write
are overridden. Public token input accepts IDs only; no native activation,
state, matrix or routing input is admitted. Original decoded row tiles supply
HC down/up/inject weights. All native captures are output targets only.

COMMAND -> Root-only authorized payload computation:

    PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 strata/flash-next/explore_full48_original_hc_fma_v1.py --native-run-root ORIGINAL_P30_FINAL --model-identity ORIGINAL_POST_FULL4_JSON --prefix 4 --bulk-build-root /mnt/vm_8tb/b70/build/hc35-host-bulk-fma-v1-20261010 --output NEW_OUTPUT

The runner retains the original source35 dispatch admission, finalized native
output-target admission, current source/stat/pages before and after, and NEW
post-computation complete four-shard identity scan. It admits the fresh bulk
helper and its qualified scalar equivalence before original payload reads.
Bulk finalized_binding and source closure are rechecked before/after tokens.
No GPU, Docker or native model execution is launched by this CPU runner.

RESULT -> Pending root execution. RMS and SiLU/inject use frozen scalar host
F32 helpers: square FMA lane sums then XOR reduction, mean and epsilon,
(R*norm)*rs, host sqrtf reciprocal and expf candidates. HC projections use
original decoded F32 weights and exact bulk block32 schedule. Four-stream mix
uses ordered FMA followed by F32 divide4. Both HC write paths use fused
fma(block, 2*sigmoid(inject/4), residual), as explicit source35 kernels do.
Seven CPU controls cover fused cancellation, owned shape/nonfinite refusal,
qualification before composition, bounded original weight tiles, token-only
interface and inherited non-HC computation. They run no compiled helper or
model payload. The prior owned reference remains unchanged as its own control.

VERDICT -> Exploratory arithmetic refinement, not device arithmetic proof or
model quality qualification. Host expf/rsqrt, native window/GEMM rounding,
non-HC reductions and native state remain distinct unresolved seams. No
threshold, tolerance or PASS is assigned to native numeric comparisons. The
fresh CPU helper qualification establishes synthetic host scalar equivalence;
it does not qualify native SYCL or the full independent original48 result.
