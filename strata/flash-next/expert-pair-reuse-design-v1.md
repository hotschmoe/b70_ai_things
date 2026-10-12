# Original-quant E2 expert weight reuse

CONFIG -> Default OFF, exact STRATA_EXPERT_PAIR_REUSE=1 enables this candidate.
The value is resolved once per process on the host and captured as a bool in the
existing GU/down port kernels. Two separate processes are required for OFF/ON.
Only groups of exactly two entries qualify: GU types12/13 and down7/8. Other
sizes/types and the disabled path retain the original row loops. The exported
native_expert_grouped interface, layouts and scratch sizes are unchanged.

COMMAND -> prepare_expert_pair_reuse_v1.py creates a NEW copied iq_kernels source
from the exact actual SDK52 source. Root compiles that object with the actual
SDK52 flags. For the standalone operator, root copies libstrata_kernels.a and
removes exactly its old iq_kernels.dp.cpp.o before linking the new object first.
No old/new duplicate object, saved ELF substitution or engine qualification is
allowed. The current original SDK52 and all old experiment files remain intact.

RESULT -> Each quant weight term is loaded once using exact original disabled
S27 load/apply bodies, renamed ExpertPairV1. Two independent activation terms
use the same original float expression, k/lane stride and lanes_sum XOR order.
GU preserves ent_tok source indices; down preserves per-entry hq and ent_dst
locations. Stack weight packets/accumulators replace duplicate scalar loads; no
USM allocation, queue wait/query, shared-local array or FP16 dequantization is
added. Hardware register allocation/instruction reuse and bitwise behavior are
unqualified until fresh compilation and actual operator comparison.

Prior candidates are kept separate: the failed45 grouped-F16 path used oneMKL
and ordinary-heap pointer arrays, repaired47 used USM ownership, and ESIMD51
changed model IDs. This candidate calls none of those paths and preserves the
original Q4_K/Q5_K/Q5_1/Q8_0 encoded weights and Q8_1 activation representation.
It does not activate the disabled AMD S26/S27 launch machinery or apply a new
numerical tolerance.

Bounded fixture required before model use: exact2560x640 layouts12/7,12/8,13/7,
13/8; both device and sycl::malloc_host synthetic valid original-format blobs;
plans [2] distinct and duplicate activation, [1,1] different experts, [3] fallback
and [1,2] paired group at nonzero offset. Use permuted/noncontiguous ent_tok and
ent_dst, compare all output/scratch/Q8 input/weight bytes across fresh OFF/ON
processes, and changed-input captured graph versus same-mode direct call.
Owned graphs/queues must retire before frees; root owns lease/health/cleanup.
The fixture is owned separately by C112 and remains an execution prerequisite.

VERDICT -> Four focused CPU source controls check actual apply/reverse/full undo,
exact typed helper extraction, row/lane order, default-OFF role and group gates,
original fallback bytes, destination indexing, and unchanged allocation/wait/query
rosters. These are source checks, not native bitwise or performance evidence.
Actual same-expert overlap and the runtime share of these kernels are unknown.
No model fidelity, speed, memory bandwidth or serving authority is transferred.
