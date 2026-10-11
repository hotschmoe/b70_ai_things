# Conditional first-step GDN state update

CONFIG -> Same closed original CLI2 and actual43 NUM10 actor3 as the packet
audit. Own initial state is independently bytezero; actual before-state must
also be bytezero, including sign bits. Native normalized QKV and decay/beta
are captured CONDITIONAL operands only, never primary original-reference inputs.

COMMAND -> audit_source43_zero_gdn_state_v1.py uses the same five explicit
input/output CLI arguments as the packet auditor. No model/public readmission,
weights, device queries, exponentials or GPU. It privately ports only the
selected field/comparison loop while retaining packet-auditor compact/source/
raw33/PID/nonce/T1/artifact and per-consumed-prefix failure-closing guards.

RESULT -> Actual source43 verify_kernels.dp.cpp:370-436 maps qh=head%16 and
state[i,head,j]. For positive-zero initial state and finite nonnegative g,
kv=+0. Require captured log-decay<=0 and beta within[0,1]. Compute binary32
delta=(v-+0)*beta, product=k*delta, then include the explicit FMA's zero-add
semantics. Signed-zero products are counted; before negative zero is refused.
Compare all786432 state words per request, recording first(i,head,j) and hashes.
Query values are not used for state. Tiny test geometry is CPU-only; production
selected fields and dimensions remain fixed128/16/48.

VERDICT -> Exact state isolates this supplied-input recurrence boundary;
upstream projections/norm, ungated core, RMS argument and original full48
math remain unqualified. Subnormal/fastmath equivalence is not presumed.
Audit passed means source/data collection completed; all_four_states_exact
is a separate comparison. No tolerance fitting or old reference replacement.
