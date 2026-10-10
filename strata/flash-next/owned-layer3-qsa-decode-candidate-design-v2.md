# Bounded original layer3 QSA decoder candidate

CONFIG -> The frozen V1 contract is retained. V2 adds a CPU transcription of
the resident one-chunk decoder for 1..4 independently owned cells. Query/gate
shape is 24x256; key/value shape is cells x2x256. Pools round to FP16 and selection
is the complete own cell roster. No captured arrays or selection IDs enter it.

COMMAND ->

```
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=strata/flash-next python3 -m unittest test_owned_layer3_qsa_decode_candidate_cpu_v2 test_owned_layer3_qsa_contract_cpu_v1 -q
```

RESULT -> Twelve synthetic CPU controls pass. Known one-cell and equal-score
two-cell fixtures exercise softmax, value accumulation, the second chunk stage,
gate order, exact own argument records, and scope/error rejection.

VERDICT -> This is source/CPU preparation. No model projection, native target
comparison, device operation, or whole-model qualification has run.

## Arithmetic boundary

The consumed `qsa_decode_attn.dp.cpp` source writes an unannotated left-associated
eight-product expression and then XOR32. The new CPU score function deliberately
declares separate F32 products/adds as one candidate. Compiler contraction is
unobserved. It must not be represented as exact device lowering. A subsequent
device control must execute that same source expression with the admitted HC
object/device link flags and publish own partial scores as outputs.

The value loop explicitly uses FMA in ascending own cell order. Even one chunk
retains the source's exp(0), second FMA stage, and final division. The gate uses
native exp(-gate), reciprocal division, and a final F32 multiplication. Every
nonlinear operation, explicit FMA, and division is tagged with exact own operands
and output bytes. The test provider is synthetic and cannot establish device
equivalence. This bounded candidate rejects nonfinite intermediate results; a
future actual producer must establish that its own inputs fit this scope or add
an explicit preregistered overflow branch, rather than silently replacing values.

## Required successor before actual execution

An original projection producer must obtain layer3 Q/K/V and indexer values
from original weights, independently computed residuals, and zero own history.
It must retain exact source dispatch storage and first prove norm/RoPE/pool
intermediates. Captured native arrays remain output targets only. Original
non-QSA computation stays inherited from the existing owned reference.

The operation helper must batch bounded owned arrays across the process boundary;
the CPU scalar callback is an arithmetic interface, not a proposed per-scalar
RPC implementation. The helper must expose source score expression, two-stage
norm reduction, rsqrt, RoPE pow/sin/cos, native exp, FMA and divide witnesses with
tagged input echo and direct/replay outputs. A matching owned lifecycle successor
must retain current source/library/flag identities, strict lease and health,
actual EOF, teardown, and full-four/page brackets. No helper launch recipe or
runtime-ready status is granted by this source plan.
