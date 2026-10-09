# Owned L0 packet boundary and layer2 Q5_K storage audit

CONFIG -> Preserved finalized V9 prefix1 native33 fields and independently
completed original48/explicit-zero-PLE CPU reports; source29 SYCL grouped kernels.
COMMAND -> python3 strata/flash-next/test_owned_layer0_packet_hashes_cpu_v1.py
COMMAND -> python3 strata/flash-next/test_q5k_q8_8lane_storage_estimate_cpu_v1.py
RESULT -> Five SHA/rank/provenance controls and five synthetic packed-block
storage controls pass. Actual readonly audits match36/36 consumer packet hashes
for EACH original48 and zero-PLE report, while rawF32 input hashes match0/36.
VERDICT -> Actual observed L0 packet bytes unchanged despite small input storage
seams; no layer2/native reducer/fullmodel or tolerance qualification.

The readonly auditor validates the actual complete33 frame through the frozen
source/nonce/PID/packet/live-owner binding reader, then binds own completed report
to the actual request SHA/PID/ordinal. It maps qkv and gate to attn_input_q81,
ssm_out to gdn_output_q81, shared/expert GU to ffn_input_q81, shared down to
shared_hq81, and ten native-ranked expert down rows to expert_hq81. Own and native
expert IDs/ranks must match exactly. Their order is136,317,77,257,175,2,414,373,242,368.
The corresponding actual F32 inputs are attn_mixed, gdn_output_gated, ffn_mixed
and DERIVED shared/expert hidden. The latter are not raw fused hidden witnesses.

Both reports have all36 native packet byte SHAs equal to the independently
computed packet event SHAs, and all36 corresponding F32 input SHAs unequal.
Thus the observed prefix1 L0 small F32 differences did not alter these packets,
including d, stored_s and codes. Own reports store packet hashes rather than
bytes, so byte difference positions are unavailable if a future hash differs;
the auditor never invents them. This does not say later-layer packets are equal.
Actual audit outputs are NEW/tmp files, SHA-bound in the source CPU receipt.
No model weight payload, GPU, Docker or runtime-directory write was performed.

Layer2 GUQ5_K/type13 uses current SYCL native_gu_kernel<13,8>, row_dot_lanes and
vec_dot_q5_K_q8_1. Backend's separate native-q5k-affine-source-audit confirms
only ds[0] is loaded and minimum uses d_x*integer_code_sum, not stored_s. DownQ8_0
is symmetric. This agrees with frozen OwnedFfn's original-real decoded weights
dot independently reconstructed Q8_1 d*codes. No stored_s fix is justified.

The new q5k_q8_8lane_storage_estimate_v1 is NOT wired into frozen full48 math.
It extracts the source call indexing: Q5_K256-value blocks have16 pieces, each
visiting eight columns in each of two32-value groups; iqs=2*part,
bq8_offset=2*(part//4), within=part%4. Q8_0 blocks have four eight-column pieces.
Each column is visited once; each of eight lane partials walks k=sub,k+=8 and
reduces by XOR4/2/1. Synthetic controls check high fifth bits, six-bit scale/min
packing, minimum correction, row2560/640 extents, symmetric dyadic onehot output,
source real algebra against independent decoder and stored_s mutation invariance.

It estimates NONCONTRACTED F32 multiply/add/sub stores. Actual SYCL compiler FMA
contraction, native subgroup/device rounding/exp/SwiGLU, HQ ties and actual layer2
inputs/GU/HQ/router/expert outputs remain unobserved. The helper also returns
unrounded FP64 piece algebra for synthetic mathematical comparison. It assigns
no numerical threshold/pass and does not silently replace the original generic
mathematical lane. Layer2's residual error cannot be assigned to this arithmetic
schedule without actual producer input/projection/hidden/down witnesses.
