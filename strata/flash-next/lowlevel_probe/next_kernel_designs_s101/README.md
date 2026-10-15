# S101 independent next kernel designs

Actual claude-opus-5-5 (104.737s) proposes a Q6 C6 dword-aligned five-word
weight-load reconstruction audit. Actual claude-fable-5-1 (111.676s) proposes
peer wait/uncached-copy/plan fusion, conditional on ownership, visibility
and data-readiness proof. Root accepts source audits only: no candidate
build, GPU timing or fusion qualification is established.

[Recommendations and corrections](../../../../docs/20261015_strata_next_kernel_candidates.md).
[Exact copies](record-copies.json) retain final packets/pins, actual answers,
model IDs, source decision, partial doorbell audit and preflight correction.
Two SHA preflights stopped before model invocation while packets were still
being shortened. Explicit author freeze then allowed one actual invocation
per model. Initial answers remain verbatim; root corrections are separate.
The S102 existing-main Q6 audit is prepared source only, not a result here.
Graph-boundary follow-up is tracked separately from this checkpoint.
No graph implementation or speed result is claimed here.
