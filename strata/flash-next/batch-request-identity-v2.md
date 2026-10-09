# Versioned0022 identity CPU transport follow-up

CONFIG -> Frozen compiled20 base and tracked0022-v2 patch reconstructed with
base/patch/overlay hash validation. Earlier ea36f0f4 review patch is preserved.
COMMAND -> test_batch_request_identity_v2_cpu.py and
 test_batch_identity_transport_v2_cpu.py.
RESULT -> PASS actual API methods over CPU socketpair and producer/control pump;
actual generate_batched loop for BYIELD continuation and slot-to-solo migration;
request-local TLS/segments; detached stale BDONE rejected; restart generation/
queue isolation; stale T/YIELDED cannot emit/attribute another request's tokens.
Exact native capacity/RID/BSTOP blocks extracted from hashed source compile/run
with hostg++ ASan/UBSan. Bad/overflow/stale cancellation never stops peer state.
VERDICT -> Identity-only source and CPU transport fixture ready for complete
source compilation and later actual GPU qualification. No model mathematics,
full-state cache or2/4/6 numerical/fairness claim.

Transport tests exposed two issues in the initial review draft. The actual
control reader now retains its original engine generation and queue across a
restart, and validates tagged T/YIELDED before emitting/attributing them. Native
strict admission T output now carries RID/slotgen; the actual API token parser
reads its token field separately. Correct scheduling, request token IDs, cache
selection and arithmetic are unchanged. These are protocol/ownership guards,
not numerical drift switches.

Both suites reconstruct the tracked patch from verified compiled20 base files
and check every resulting overlay hash. API code is AST-extracted from that
actual source; producer/model calls are explicit CPU mocks. Native parser blocks
are extracted verbatim from the same source, not hand-copied tests. Socket and
thread primitives are real, but no model, GPU or XPU compiler is touched.

Use patches/0022-sycl-strict-batch-request-identity-v2.patch, not the preserved
initial draft, in a new coherent source generation. New API helper must be bound
in artifact identity; frozen C1/V6 sources and plans remain untouched. Final0021
application/source hashes must be checked independently before a combined build.
Per-slot raw heads/48 residuals/logical spans and automatic bounded full-chain
cache continuity remain following source increments. Pin/fresh guards still
reject unsupported concurrent calls; metadata and parallel settings alone do
not qualify the goal.
