CONFIG -> NEW ordinary CPU-only bulk helper. Frozen scalar C++ body is included
verbatim. Matrix projections use its block32 FMA lanes and XOR16/8/4/2/1
reduction. Elementwise write uses fmaf. Inputs are bounded finite LE_F32 tiles,
not captured native operands. No model files or SDK mounts are permitted.

COMMAND -> Root owns the fresh compile and all helper execution:

    PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 strata/flash-next/qualify_hc35_host_bulk_runtime_v1.py --build-root NEW_CPU_BUILD --scalar-run-root /mnt/vm_8tb/b70/build/hc35-host-runtime-v1-20261010 --scalar-build-root /mnt/vm_8tb/b70/build/hc-f32-arithmetic35-host-v1-container-20261010

The ordinary Docker compile has no device, network, SDK or model mounts; it
uses the pinned image, two CPUs, 2 GiB RAM/swap ceiling, strict FP flags and
owned normal teardown. Observed inspect metadata must match the exact recipe.
Host execution binds current ELF resolved libraries, Python mapped libm,
CPU, affinity, environment and gradual/RNE probes before and after. The
compile image does not identify the host libraries used at execution.

RESULT -> Pending root execution. Fifteen preserved scalar known-answer cases
and nine bulk fixtures are mandatory. Every bulk row also executes the
previously qualified scalar ELF using the same complete synthetic operands;
its command, stdout, output and exact bitwise match are preserved. Matrix
fixtures include shared/paired inputs and K32/64/320/10240. FMA includes fused
cancellation and gradual input/output. Read-only admission recollects full
operands, outputs, commands, receipts, source, build and current libraries.

VERDICT -> CPU source controls alone qualify no arithmetic implementation.
No tolerance, GPU intrinsic, whole HC or full model qualification is claimed.
A downstream consumer must call finalized_binding before helper use and
recheck afterward; run_bulk alone is a bounded invocation, not qualification.

Next original48 refinement must remain a separate subclass/control. Decode
original owned weights, use qualified scalar normalization/SiLU/inject and
bulk block32 projections; preserve original GDN/QSA/FFN/PLE. Actual source35
HC write is sycl::fma(block, 2*sigmoid(inject/4), residual) for both pending
and standalone paths; implement bulk FMA, not a separately rounded product
and add. Norm order is (R*norm)*rs and source FMA/XOR; host exp/rsqrt remain
candidate arithmetic seams. No captured native activations or states may
enter the independent original48 computation.
