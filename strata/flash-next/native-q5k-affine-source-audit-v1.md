# Layer2 native Q5_K affine source audit

CONFIG -> source29 NUM9 actual command metadata, layer2 GUQ5_K/type13 and
DownQ8_0/type8. Frozen actual SDK source and pinned GGML3cf03257 primary source.
COMMAND -> python3 -m unittest discover -s strata/flash-next -p
 'test_native_q5k_affine_source_cpu_v1.py'
RESULT ->7 source/synthetic CPU tests PASS, no actual weights/GPU/Docker.
VERDICT -> stored Q8_1.s hunch is contradicted by the actual consumer source.
No kernel change is justified by that hunch. Layer2 numerical cause unresolved.

Verifier::post invokes native_expert_grouped with layout.gu_type13/d_type8.
The SYCL launcher takes launch_gu<13> -> launch_gu_lanes -> native_gu_kernel
-> row_dot_lanes -> Fmt<13>::dot -> vec_dot_q5_K_q8_1. There is no Split<13>
specialization; AMD/HIP S27/TSUM branches are not current SYCL routes. Actual
command env has no lane/group/old-MMVQ override; compile-time default is8 lanes.
Down uses symmetric Q8_0, with no affine minimum correction.

The Q5_K caller loads bq8i->ds[0] only. The integer dot chain computes dot2
by DP4A with0x01010101, then sumf_m += d8*(dot2*minimum). The final expression
is d_weight*sumf_d - dmin_weight*sumf_m. It never reads ds[1]/stored_s. Pinned
GGML CUDA impl_vmmq has the same code-sum expression. Separate native_mmvq
scalar/WideQ5K consumers also use code sums, but they are not the current
expert-grouped launcher. Other formats/routes are not generalized from this.

In real arithmetic this equals original dequantized Q5_K weights times the
Q8_1 reconstructed activation d_x*codes, including the high fifth bit and
packed six-bit scale/minimum groups. Synthetic code confirms that identity;
changing stored_s cannot change the source-contract result. Replacing code sums
with stored raw sums creates a different result and is not an appropriate fix.

The identity does NOT emulate native F32 partial accumulation, eight-lane XOR
reduction, intrinsic exp/SwiGLU, activation-code tie/rounding or downstream
router/expert selection. Frozen conditional Q4_K/Q5_1 helpers remain unchanged
and do not automatically qualify Q5_K. Actual layer2 GU/HQ projections, packet
codes, selected experts and reducer intermediates remain unobserved. The
remaining reported layer2 difference cannot be assigned to a stored-s defect
or general model bug from this source audit alone.
