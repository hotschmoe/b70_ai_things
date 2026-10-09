# Owned FFN storage adapter for original48 roles

CONFIG -> Synthetic geometry, all48 original type-role exceptions, explicitly
source-audited SYCL F16/noMMQ/noFUSED prefill or Q8_1 verifier mathematical lane.
COMMAND -> python3 strata/flash-next/test_ffn_full48_owned_storage_v1.py
RESULT -> Six tests pass for role failures, owned router/expert outputs, route
rounding, input perturbation and saturation/NaN handling.
VERDICT -> Source/storage estimate only; no actual-weight or native numeric gate.

Fresh29 prefill.cpp3230-3236 uses F16 native shared GU/down and BF16 router/shared
scalar inputs and original F32 weights roundedBF16. Lines4021-4039 dequantize
expert down toF16, produce F16 expert hidden and consume F16 GEMM operands.
mmq_plan at858 reads STRATA_PREFILL_MMQ and built() controls availability. The
fresh29 build.ninja contains no STRATA_PREFILL_MMQ compile definition. Alternative
STRATA_PREFILL_MMQ/PF_FUSED/BF16_TC/PREFILL_BF16X2 routes fail closed in this lane.
Actual runtime route remains unobserved; source profile does not prove execution.

The adapter owns router probabilities, selected ranks, shared/expert GU, hidden,
down and weighted combine from its mixed input. It uses original Q4_K/Q5_K/Q5_1/
Q8_0/F32 roles, including layer2 Q5_K GU and layers2/4/30/46/47 Q8_0 down. For
verifier boundaries it independently encodes Q8_1 d/codes and reconstructs d*codes;
original affine real decoded weights imply code-sum minimum contributions, never
a generic stored packet-s affine substitution. Prefill instead rounds original
decoded operands toF16. The provider is lazy and bounded to64MiB row tiles.

Router tie choice, native exp, intrinsic/FMA/reduction/GEMM order, hf saturation
and probability/combine storage seams are mathematical estimates, not bitwise
native emulation. No tolerance gate is assigned. CPU synthetic rows do not prove
mixed-format decoder correctness. Decoder/source and supplied conditional FFN
checks remain separately frozen. A standalone supplied HC mixed input remains
conditional; an owned48 composition must produce that input from embeddings and
its own GDN/QSA/PLE/HC state. Fullmodel math remains false.
