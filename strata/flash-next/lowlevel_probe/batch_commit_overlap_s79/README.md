# S79 batch commit/draft overlap, 2026-10-15

HOLD_UNMATCHED_NO_LATENCY_PROMOTION. Both finite server arms closed normally,
all output/functional/health checks passed, but the third timed pair had
one extra ON solo window and two extra offered drafts. No aggregate timing
median or speed claim; no rerun or promotion. First two matched timed pairs
were slightly slower. The prototype is parked and all three current source
readback hashes equal their pre-S79 bases, without a rebuild/server restore.
Generic build artifacts still contain the candidate and do not represent
restored source; the frozen03bb executable is experimental only.

Both actual post reviewers accept this disposition. Their restoration-record
ambiguity is resolved by the separate readback supplement; the original
before/after fields describe historical application, not restoration.
See docs/20261015_strata_decode_overlap_and_prefill_contract.md.

Exact source base/candidate hashes and external raw patch are retained in
source-bindings.json and the S79 source-prep folder. The checked-in
commit-overlap-zero-context.patch is a derived context-free representation
of those same three file changes, avoiding redundant unchanged source.
Apply to the exact pinned base with git apply --unidiff-zero after checking
base hashes. Do not apply to arbitrary upstream versions.

The six harness scripts are exact execution copies; their original parent
paths are in record-copies.json. Their path-relative runtime assumes the
original results folder. Root's actual build recipe emits the harness's
strata-batch-commit-s79 filename. Source reviews are conditional on build,
health, activation, output/work matching and normal teardown.
