# Bounded serial native fidelity and prefix fixture

CONFIG -> New combined engine generation 0013..0016, exact original selected
model/tokenizer, fixed segmented mirror/native HC/PLE routes. Separate controller;
frozen C1 serving controller and source-upload evidence remain unchanged.
COMMAND -> CPU source preparation and synthetic collector/transport tests only.
RESULT -> Source-native GEN keys fresh=1 and pin=N are reachable directly. CPU
collector tests pass, including pipe/QUIT transport and rejection controls.
VERDICT -> Prepared for parent-owned leased GPU execution. No GPU experiment,
model fidelity, cache correctness, API/concurrency or clean latency claim yet.

## Preparation and launch

Build the new immutable combined generation first:

```
python3 strata/flash-next/build_native_hc_engine.py \
  --plan strata/flash-next/prefix-full-state-engine-build-plan-v2.json
python3 strata/flash-next/serial_prefix_qualification.py prepare \
  --prepared EXACT_BASELINE_C1_PREPARED_DIRECTORY \
  --engine-root NEW_COMBINED_BUILD_ROOT --model-identity FRESH_COMPLETE_MODEL_HASH_JSON --output NEW_PREPARATION_DIRECTORY
```

Preparation validates the original full390 gated baseline, selected shards and
known page sentinel, source/build/pack/runtime receipts and exact unchanged base
patches, with only 0013..0016 added. It uses the actual exported tokenizer and
source frontend ChatTemplate in a CPU-only pinned runtime container. Token IDs
and tokenizer/template hashes are written to the new plan. This is a provenance
chain, not a claim that the new observer binary ran a new whole390 upload oracle.
Its report keeps new_engine390_source_upload_qualified=false.

The parent supplies fresh complete per-card/pair health receipts and owns leases:

```
bin/gpu-run --card 0 python3 strata/flash-next/serial_prefix_qualification.py run \
  --plan NEW_PREPARATION_DIRECTORY/plan.json --pre-health PRE_HEALTH_JSON \
  --group basic --output NEW_BASIC_RESULT_DIRECTORY
python3 strata/flash-next/serial_prefix_qualification.py finalize \
  --output NEW_BASIC_RESULT_DIRECTORY --post-health POST_HEALTH_JSON \
  --model-identity POST_RUN_COMPLETE_MODEL_HASH_JSON
```

The controller verifies inherited lease descriptors. Pair execution requires a
matched combined one-card receipt with post-health. Each later prefix group
requires the same-card basic receipt (or use group=all to run basic first):

```
bin/gpu-run --card 0 python3 strata/flash-next/serial_prefix_qualification.py run \
  --plan NEW_PREPARATION_DIRECTORY/plan.json --pre-health NEW_PRE_HEALTH_JSON \
  --basic-receipt NEW_BASIC_RESULT_DIRECTORY/report.json \
  --group pin --output NEW_PIN_RESULT_DIRECTORY
```

The run owns only its uniquely labelled containers, requires clean engine QUIT,
terminal exit0/no OOM and confirmed removal, watches the known original page
sentinel before/after every group, and preserves errors/raw captures. Parent
post-health must follow the finished fixture; finalize cannot turn a strict
numerical/teardown failure into a pass. No recovery or broad process kill is
performed here. The parent still owns full model-identity rechecks and recovery.

## Exact bounded comparisons

Only three useful questions are used: arithmetic 17+25/19+23 and the capital of
France. Shared-pin variants use a bounded, useful shelf inventory and ask for
one number. All generated replies are bounded to64 tokens with natural stop
required for non-cancel comparisons. No forced reasoning close or token rewrite
is used. Thinking=false goes through the actual exact chat template.

Every process performs the same three unarmed fresh warmups before ARM is
created. Cache fresh/hit comparisons then remain in that same warmed process,
with identical engine args and normal root/turn/pin cut geometry. A process
restart is never counted as an uncached half of a cache pair. The source observer
has six armed requests per process; every group stays within that bound.

- basic: same binary flag0, flag1/activations0, flag1/activations1 in separate
  warmed processes. Flag0/on compares exact output IDs, LP20 and natural finish.
  Flag1 activations0/1 additionally compares all248320 raw first logits. True
  flag0 has no raw export; flag0_full_logits_observed remains false. Enabled
  instrumentation can change queues and races; passing is bounded equivalence,
  not proof that its copies/barriers are irrelevant to every history.
- root/pin/turn: two repeated triples of fresh target, fresh establishment with
  divergent suffix, cached target. Parking is disabled so a deeper previous
  target snapshot cannot counterfeit the desired root/pin/turn hit. Expected
  actual boundary, complete prompt work and same-process first logits must match.
  The turn variant changes only a trailing assistant-prefix space, preserving
  the preceding completed conversation and latest assistant turn checkpoint.
- parked: two triples of fresh target, fresh independent branch, restored target.
  Requires actual ram_snapshot route and the exact deeper turn boundary, not
  merely a root hit. Complete full-state stage restore is exercised indirectly
  by first-logit and48-layer residual equality with the fresh reference.
- eviction: independent system contexts with the same useful questions, one RAM
  snapshot slot and512MiB. Requires actual eviction count growth after the armed reference, a cold reconstruction
  of the evicted target, and equality with fresh references. If source admission
  skips snapshots or no actual eviction occurs, this fixture fails; a planned
  budget is not evidence of an eviction.
- cancel: parking disabled, fresh long prefill stopped on first PP, then replay
  and fresh replay; decode stopped on first T, then fresh unrelated replay.
  Requires actual STOP/cancel, full fresh/replay equality and healthy teardown.
  Cancellation counters are completed spans only: partial failed device work is
  not claimed zero. This group tests partial active-chain state, not cancellation
  concurrent with another stream or an API session.

Non-cancel ledger acceptance requires exact input IDs, selected versus actual
reuse, every prompt position covered once with no gaps/overlap, correct per-stage
work multiplier, and fresh reused=0. Full logits must contain248320 finite F32
values at final-prompt position with the original last input token. Captured
first-window residuals must cover all48 layers once, each10240 floats. Raw
prefill vectors from the observer's first two chunks are retained for diagnosis;
they are not compared across prompts whose suffix geometry differs.

All finite first logits and matched first-window residuals initially require
bitwise equality. Failures save max absolute error, RMSE, hashes and first
observed divergent layer; no posthoc tolerance widening. Full generated IDs,
LP20 and natural finish must also match. This is internal consistency against
fresh reconstruction, not an independent mathematical reference for the whole
model. Persistent GDN/PLE/QSA raw buffers remain unobserved in0013.

Native branches and changed-system contexts exercise token/state isolation;
there is no API session cookie or /v1/models check in this transport. Reports
retain hotschmoe-dd first as identity metadata and a distinct nativeGEN research
alias, but do not claim the alias was served over HTTP. API/tokenizer served-ID,
stale session, image/control-vector identity, live continuation and concurrent
one/two/four/six-stream fixtures remain subsequent qualification work.
complete_prefix_qualified stays false even after all bounded serial groups pass.
Diagnostic timing is not clean latency evidence.
