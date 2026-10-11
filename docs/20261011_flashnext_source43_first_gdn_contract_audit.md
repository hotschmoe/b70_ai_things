# Source43 first-GDN operation and storage audit

CONFIG -> Actual SDK43 kfiohqmn, four fresh [248045], MAX1, T1/position0,
NUM10ON/QSA3OFF/materializedmode1. Source inspection only; no model bytes,
reference computation, build or GPU execution. Mode1 changes QSA indexer
storage, not this layer0 GDN path. Runtime flags/dispatch must still be read
from the actual case; source defaults are not an observed dispatch witness.

COMMAND -> Read SDK43 source and unchanged original_first_gdn_layer_v1 plus
retained F32/device reference source. Native paths below are relative to
SDK43/source. CPU paths are relative to strata/flash-next. Bare native filenames below mean
sycl/src/kernels/cuda/, except hc_native_composition.cpp under
sycl/src/kernels/.

| Operation / output targets | Native source43 contract and anchor | FP64 NUM10 diagnostic | Existing F32/device whole-model reference |
| --- | --- | --- | --- |
| Original embedding -> HC read / residual_input, HC normalized/low/gate/inject/mixed | sycl/src/core/verify.cpp:993-1025 binds one row; sycl/src/kernels/hc_native_composition.cpp:26-53 uses F32 FMA-square/XOR RMS, sycl::rsqrt, (R*norm)*rs, native Q8/F32 HC projections and sycl::exp sigmoid. Mix source is multiply-add; no new contraction claim. | original_first_gdn_layer_v1.py:105-117 uses F64 mean/sqrt, decoded-weight F64 dots, logaddexp sigmoid and mean. It does not reproduce native reduction, rsqrt, projection or storage order. | full48_owned_hc_device_rs_v1.py:14-30 uses own FMA/XOR arguments, actual device rsqrt, host F32 dot/FMA helpers, own operand publication in v3. Non-HC methods remain inherited; universal rounding/internal native argument unqualified. |
| Mixed -> QKV/Z / attn_input_q81, gdn_qkv, gdn_z | verify.cpp:1077-1093 and1100-1101 packs raw F32 mixed into Q8_1 then native_mmvq Q8_0 projections. native_mmvq.dp.cpp:190-204 / q8_1_finite.hpp:28-42: F32 amax/127 divisor and round-away, stored half d/s. Q8_0 dispatch3628-3632 may packed/wide/small; wide1894-1954 uses integer dots, F32 products/accumulation and XOR sum. | mixer:129 projects unquantized mixed with exact decoded original weights in F64. Q81 packets are separate diagnostic encoding of narrowed outputs, not its projection inputs. | OriginalStorageProjector:44-69 encodes/decodes Q81 activation, then original decoded rows dot in F64 and final F32 store. Does not reproduce native integer/F32 lane accumulation; stored sum unused for Q8_0. |
| Convolution/SiLU/q,k normalization / gdn_normalized_qkv and conv_after | verify_kernels.dp.cpp:44-102 reads [hist0,hist1,hist2,x], four F32 tap terms, native::exp SiLU; q/k use square +32-lane XOR, four warp partials, sycl::rsqrt(sum+eps). T1 self_commit is verify.cpp:830,1093-1094; histories are native[channel,3]. | mixer:133-142 uses F64 sum/SiLU and sqrt; logical history[3,channel]. No native intrinsic/tree/storage emulation. | prefill_first_gdn_owned_storage_v1.py:88-93 stores F32 convolution/SiLU but NumPy reduction and host exp/reciprocal sqrt, not source XOR/intrinsics. |
| Alpha/beta/decay / gdn_decay_beta | verify.cpp:1096-1098 calls gdn_ab_multi; verify_kernels.dp.cpp:128-233 reads BF16 weight mirrors, F32 input, eight sequential FMA terms per lane then XOR; beta=1/(1+native::exp(-a)); softplus branch v>20 otherwise log1p(native::exp(v)), times original F32 ssm_a. | mixer:130-131 uses original F32 weights decoded to F64, F64 dots/sigmoid/logaddexp. Does not emulate BF16 mirrors, FMA grouping or device exp/log1p. | mixer:86-87 explicitly rounds small-projection weights to BF16, F32 stage stores and >20 softplus branch; dots/host exp/log remain estimates. |
| One-step recurrence / state_after | verify.cpp:1115-1116; verify_kernels.dp.cpp:338-490 owns native[i,head,j], head key=head%16. F32 four groups of32 state rows, FMA dot, g=native::exp(decay), delta=(v-g*kv)*beta, FMA(g,old,k*delta), then FMA q-dot and four partials. Commit stores state only on actual keep. | mixer:143-146 keeps logical[head,i,j], uses F64 decayed state and dots/outer product; mathematically equivalent placement of decay, different rounding/order. | recurrent_estimate:71-78 uses F32 stage stores with F64 einsum estimates; update is a host product-plus-add estimate rather than native explicit FMA/reduction. |
| Core RMS/gating / gdn_output_gated | verify_kernels.dp.cpp:434-484 multiplies core by sycl::rsqrt(128), XOR squared sum plus four partials, sycl::rsqrt(mean+eps), then ((oc*scale)*gamma)*sigmoid(z) using native::exp. Ungated core/raw norm argument are NOT raw33 fields. | mixer:147-149 uses F64 sqrt/mean/sigmoid and normalization expression order; native rounding/internal values unobserved. | mixer:96 uses F32 stage stores/mean and host reciprocal sqrt/exp; own device-RS refinement covers HC only, not this GDN normalizer. |
| Gated -> output projection / gdn_output_q81, gdn_block_output | verify.cpp:1118-1130 with QFUSE0 repacks actual gated F32 using ordinary Q81, then native Q8_0 mmvq. QFUSE uses distinct unclamped packer and is excluded from this case. | mixer:149 original unquantized gated F64 projection. Optional native-gated/Q81 projection lane is explicitly conditional and separately reported. | mixer:97-98 uses own verifier gated F32 -> Q81 -> original dot estimate; threshold crossings can amplify tiny earlier differences without proving their cause. |
| HC write / residual_after_attn | verify.cpp:998-1003 applies native write before FFN HCread; hc_native_composition.cpp:60-67 uses explicit sycl::fma(block,2*sigmoid(inject/4),R). | hc_write:116-119 has F64 separate product/add. | full48_owned_hc_fma_refinement_v1.py:43-44 uses qualified host F32 FMA with own block/inject. Native sigmoid/device operand equality still requires comparisons. |

RESULT -> Smallest raw33 probes, after the actual CLI metrics arrive:

1. **No-weight state/commit check.** Every native beforestate and conv is
   bytezero. Native conv_after must be [0,0,captured qkv] per channel for T1
   self-commit. This is an exact layout/owner/keep check, not a tolerance.
   Conditional zero-state recurrence can use captured normalized q/k/v and
   beta to check next-state head%16 mapping and source multiplication stages;
   it never injects native state into the independent reference. For finite decay, initial zero
   removes the decay-dependent old-state term, so do not blame decay alone for
   a first-row discrepancy without evidence.
2. **Existing optional native-mixed lane.** Compare QKV/Z, decay/beta,
   normalized QKV, next state, gated and block targets under the same ownzero
   state. This removes only the HC input seam. It retains BF16/projection/tree/
   intrinsic differences described above; a remaining error is not uniquely GDN.
3. **Existing optional native-gated/Q81 projection lane.** Verify whole packet
   bytes first, then original Q8_0 projection. This isolates upstream gated
   differences from output packing/dot order; it is conditional, never a new
   full-reference operand. Stored sum differences cannot affect this Q8_0 dot.
4. **Only if conv/norm becomes the first useful gap:** root may later replay
   unchanged gdn_conv_l2_multi with captured QKV + ownzero history and original
   four-tap weights, comparing normalized QKV and committed history. It is a
   bounded conditional device probe, not currently implemented/qualified.

VERDICT -> No new math or fitted threshold. Raw33 does not expose raw conv sum,
alpha/beta pre-sigmoid dots, ungated core or GDN RMS argument. The table cannot
identify which unobserved operation causes a discrepancy. Use the closed actor
and forthcoming FP64/conditional scalar metrics to choose one bounded seam;
retain the original F32/device reference and all old evidence unchanged.
