# Independent original HC F32 block35 control

CONFIG -> current finalized NUM10 source35/C113, accepted prefix1/2/4/8,
original publisher all4 identity, original role/type/shape/stat guards and current
qualified host FMA/library binding. Native fields are comparison targets only.
COMMAND -> root-only explore_owned_hc35_f32_block_v1.py, with --prefix4.
RESULT -> source/CPU controls only until root runs actual original payloads.
VERDICT -> no device intrinsic, native bitwise, full HC/model, quality or speed
qualification and no tolerance gate.

The frozen original48 route reference is unchanged. Completed head NMSE is
8.40917e-5, 1.01251e-4, 9.32150e-3 and 5.24039e-2 for prefixes1/2/4/8.
Layer0 attention agrees closely through row2 but row3 changes by NMSE2.62141e-6.
The owned NUM10 prefix4 replay isolates one Q8_1 code, block35 column1123:
own71/native72. Its input at that column is identical; the own block maximum
differs by one F32 ULP. This nonlinear rounding threshold is evidence for a
source arithmetic control, not permission to change the packet or its tolerance.

Every layer0 HC input is the accepted token's independently decoded original
embedding broadcast over four streams. There is no incoming layer0 HC state.
The new control therefore recomputes all four token rows independently from
original embeddings/weights, not from captured normalized/low/mixed arrays.
It computes the 32-lane square FMA accumulation, XOR16/8/4/2/1 reduction,
F32 mean/epsilon, and (R*norm)*rs normalization. It projects all320 original down
rows, original up rows for four streams at columns1120..1151, and all4 injection
rows using the separately qualified host F32 FMA/XOR schedule. Every decoded
weight tile is <=64MiB. Only this complete block can determine its own maximum,
scale and32 Q8 codes; a single selected column cannot.

Unresolved contracts are explicit: ordinary sycl::exp(float) for HC sigmoid,
sycl::rsqrt(float), floating division/add lowering, and whether the compiler
contracts mix sum+=xn*sigmoid(gate) to FMA. The control uses host sqrtf followed
by F32 reciprocal, and host expf followed by F32 add/divide. It preserves both
separate multiply/add and FMA mix candidates, without selecting one from native
agreement. These are unqualified hypotheses. Host FMA/projection arithmetic
qualification and the conditional native8 up/inject matches do not qualify
these device intrinsics or the independent end-to-end operands.

A whole HC device oracle is required to settle those contracts. It must observe
actual source35 norm square sums/rs, normalized rows, pre/postSiLU low, gate,
inject, mixed and explicit pending/standalone FMA write on synthetic T1/T2
inputs, with adversarial cancellation, reciprocal-sqrt rounding and subnormal
cases. It must use the exact current compilation and normal device route.
HC uses ordinary sycl::exp, unlike GDN's native intrinsics. No captured device
value may become an operand of this independent original-weight control.

Root runnable command, with paths resolved from completed NUM10:

    OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python3 strata/flash-next/explore_owned_hc35_f32_block_v1.py --native-run-root /mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f16-source35-20261009/num10-onecard-run --model-identity /mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f16-source35-20261009/num10-onecard-run/post-model-identity.json --prefix 4 --helper /mnt/vm_8tb/b70/build/hc-f32-arithmetic35-host-v1-container-20261010/hc35-host --helper-build-root /mnt/vm_8tb/b70/build/hc-f32-arithmetic35-host-v1-container-20261010 --host-runtime-root /mnt/vm_8tb/b70/build/hc35-host-runtime-v1-20261010 --output NEW_OUTPUT_DIRECTORY

The adapter admits NUM10/current raw proofs, source3/current HC kernels,
pre-original all4 identity/pages, original roles and current host binding before
math. Native arrays never enter tokens(). Comparisons occur after all own rows
finish. It saves every own normalized/low/selected-gate/inject/mixed/packet
array. It rechecks host/source/native proofs and performs a NEW postCPU all4
publisher hash plus known-page brackets. Root alone may run actual payloads.
No GPU, Docker or compiled helper is executed by this CPU control.
