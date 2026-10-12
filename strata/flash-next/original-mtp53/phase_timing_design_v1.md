# Existing-boundary host phase timing

CONFIG -> Same original greedy64 program and checked SDK52 shutdown. Timing is
absent unless STRATA_ORIGINAL_MTP53_PHASE_TIMING=1. Enabled mode reserves2048
CPU rows, with no USM/GPU allocations or altered expert-cache budget. Each scope
records an existing synchronous host boundary and refuses nesting. Rows print
only after the original decode interval, preserving token/commit output format.

COMMAND -> Root freshly compiles the V3 native proposal and invokes the V4 runner
with --phase-timing. Actual compile receipt/ELF remain mandatory runtime pins.
The runner rechecks exact phase counts against observed prompt/window/commit
counts, alongside all original coherence, health, identity and teardown gates.

RESULT -> target_run and target_commit_wait use existing verifier operations.
own_head_read_argmax includes the existing full own-head host copy and CPU argmax;
the own-head GPU projection itself is included in draft_record_wait.
draft_prepare_submit is host enqueue time only; following record_wait includes
that queued work, the full original draft computation and its EXISTING wait.
draft_settle_wait includes the original state-copy submission and existing wait.
draft_prior_drain/discard_recovery retain the original verification boundaries.
final_existing_drain records the already-present end drain. No new GPU waits,
queries, per-layer callbacks or mathematical operations are introduced. Existing
Verifier wait/pool/host/commit deltas are separately reported; they may overlap
and are not an additive budget or a GPU-only attribution.

Root's closed64 experiment (producer03cd317e...) matched every output ID. OFF
reported63 windows and6746.735ms decode, ON33 windows (30accepted/2rejected/1tail)
and6732.815ms. This establishes no meaningful speed gain. Resolved placement was
NOT equal: OFF8160resident experts /51446579200B mirror, ON7270 /54231654400B,
despite the same requested6500 argument. New runs retain actual startup residency
lines. Those coupled costs prevent attributing flat decode time solely to draft
compute or inferring target-window scaling without measured phase evidence.

VERDICT -> Phase rows are descriptive host intervals from an instrumented run.
They do not establish clean production speed, physical GPU idle, bandwidth,
population stability, optimized prefill or full512 completion. Frozen V2 native
and earlier controllers remain unchanged. All original numeric comparisons and
actual target verification remain authoritative.
