# Independent original-weight HC block35 arithmetic result

CONFIG -> Source35/C113 NUM10 prefix4 observed row3. Recompute all four token
rows from original embeddings and HC norm/down/up/inject weights, exact source
F32 epsilon bd378635, native32lane FMA/XOR RMS and projection schedule.
Host sqrtf/reciprocal and expf remain explicit device-intrinsic hypotheses;
separate multiply/add and fused mix candidates were frozen before runtime.
No captured inputs, state, routes or selected experts enter own computation.

COMMAND -> explore_owned_hc35_f32_block_v2.py with the exact command in its
frozen design, OPENBLAS/MKL/OMP threads1. Root35758 terminal0/errors empty;
current host/source/raw target proof and NEW postCPU completefour publisher
hashes plus known-page brackets passed. Original failed V1 is preserved.

RESULT -> For the last observed row:

| Field | Independent original vs native observation |
| --- | --- |
| Normalized HC input, 10240 floats | BITWISE |
| Injection, 4 floats | BITWISE |
| Post-SiLU low, 320 floats | Not bitwise; NMSE1.5308460281692753e-17 |
| Selected gates, 128 floats | Not bitwise; NMSE5.690358802844491e-16 |
| Separate mix, block35 32 floats | Not bitwise; NMSE4.715157863111812e-15 |
| Separate Q8_1 block packet | One code differs: own71/native72 at1123 |
| Fused mix, block35 32 floats | BITWISE |
| Fused Q8_1 block, all36 bytes | BITWISE, including code72 at1123 |

Report:
/mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f16-source35-20261009/owned-hc35-f32-block-prefix4-v2/report.json
SHA6634cea052a0b5b9942a82a6f1b1dfec788f20d62b42faf5b6c206837790b495.
Sourceplan427a44571c4573253d573be61ce0d9685b4dbd131e8f8df85cf3d6bf36034cdf.

VERDICT -> Concrete independent arithmetic localization. The observed first
Q8_1 boundary can be reproduced using original weights and a predeclared F32
fused-mix candidate. This is stronger than substituting native inputs into a
conditional reference. It does not establish universal device exp/rsqrt/div
behavior, actual compiler contraction, full HC/model fidelity or a tolerance.
Low/gate differences remain real. A source-bound compiled whole-HC T1/T2 oracle
with ordinary intrinsics, direct and graph routes, alias/nonalias writes and
separate shadow observations is the next step before extending the original48
reference. No model/backend source, weight or quantization change is justified
solely by this result. Full cache/API/profiling/latency/shelf remain open.
