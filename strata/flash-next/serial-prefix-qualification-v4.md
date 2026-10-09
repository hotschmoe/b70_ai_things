# Bounded serial native fidelity and prefix fixture

CONFIG -> New combined engine generation 0013..0018, exact original selected
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
  --plan strata/flash-next/verifier-stream-engine-build-plan-v1.json
python3 strata/flash-next/serial_prefix_qualification_v4.py prepare \
  --prepared EXACT_BASELINE_C1_PREPARED_DIRECTORY \
  --engine-root NEW_COMBINED_BUILD_ROOT --model-identity FRESH_COMPLETE_MODEL_HASH_JSON --output NEW_PREPARATION_DIRECTORY
```

Preparation validates the original full390 gated baseline, selected shards and
known page sentinel, source/build/pack/runtime receipts and exact corrected0013..0018 patches, including fixed0017 SHA f92143d9. It uses the actual exported tokenizer and
source frontend ChatTemplate in a CPU-only pinned runtime container. Token IDs
and tokenizer/template hashes are written to the new plan. V4 requires the prepared full390 source-upload gate to match this exact
corrected engine receipt and binary. An oldv6 source gate is rejected.

The parent supplies fresh complete per-card/pair health receipts and owns leases:

```
bin/gpu-run --card 0 python3 strata/flash-next/serial_prefix_qualification_v4.py run \
  --plan NEW_PREPARATION_DIRECTORY/plan.json --pre-health PRE_HEALTH_JSON \
  --group basic --output NEW_BASIC_RESULT_DIRECTORY
python3 strata/flash-next/serial_prefix_qualification_v4.py finalize \
  --output NEW_BASIC_RESULT_DIRECTORY --post-health POST_HEALTH_JSON \
  --model-identity POST_RUN_COMPLETE_MODEL_HASH_JSON
```

The controller verifies inherited lease descriptors. Pair execution requires a
matched combined one-card receipt with post-health. Each later prefix group
requires the same-card basic receipt (or use group=all to run basic first):

```
bin/gpu-run --card 0 python3 strata/flash-next/serial_prefix_qualification_v4.py run \
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

## Matched one-card/two-card pilot geometry

V4 normalizes both prepared engine argument sets to context2048, prefill64 and
PLE row cache65536. FP16 KV, cache3, root1, periodic checkpoint interval1e9,
no MTP/lookup/suffix/adaptation/prefill borrowing, exact token fixtures and all
request bounds/tolerances remain unchanged. V4 requires exact0018 SHA
 a9f382a8861132746475b34b738b723d281f44dc33e83aa8828ae849a87987a3
and the actual matching full390 upload/engine gate. The owned-stream engine
build plan is verifier-stream-engine-build-plan-v1.json, SHA
97243b44806418ce81557923405e8b219b33082ebb4a8f3e7b72a903413df08d.

The older onecard2048/64 versus pair8192/128 receipts cannot establish numerical
parity. They change prefill row geometry and allocation/capacity; no cross-card
match is inferred from readable outputs. The small shared pilot fits every
unchanged fixture (max209 prompt tokens plus64 generated tokens). It qualifies
only this declared geometry. Production8192 capacity, long context/prefill and
one/two/four/six-stream qualification remain required work, not silently removed.

Source memory audit: session_bytes in sycl/src/core/session.cpp allocates owned
GDN recurrence/conv, QSA state/buffers, MoE/block scratch and PLE history. GDN and
PLE history sizes are independent of context. layer.cpp qsa_state_bytes adds
context-sized KV, page/indexer directories and RoPE tables. Selected FP16 QSA
has12 attention layers,2 KV heads and256 head dimension: logical K+V payload is
2048 bytes/cell/layer. Total payload is48MiB at2048 versus192MiB at8192, excluding
alignment, indexer/RoPE, shared buffers, weight/expert caches and diagnostic
scratch. Split ownership partitions the12 layers; a second stage adds its own
first-owner RoPE/scratch. These source arithmetic counts are not measured VRAM
headroom. Smaller matched prefill also reduces row-sized scratch without
changing the declared math. Keep the existing105GiB host cgroup, reserve and
512MiB parked-snapshot group budget; actual admission/allocations still must pass.

Run/finalize one-card basic first, then one-card cache groups. Two-card run
requires the completed matched combined one-card receipt and exactly the same
geometry. The separate numerical parity collector runs only on completed
health/teardown-qualified matching group reports:

```
python3 strata/flash-next/serial_prefix_qualification_v4.py compare-topologies \
  --one-card ONECARD_BASIC_DIRECTORY/report.json \
  --two-card PAIR_BASIC_DIRECTORY/report.json --output NEW_PARITY_REPORT.json
```

It revalidates input plan fingerprints, corrected binary/source generation,
exact source IDs and tokenizer fixture, fixed geometry and native/SYCL/oneAPI
environment. Only layer split/device/trim arguments and physical affinity may
differ. It reparses actual raw files and applies the same exact output/LP20 and
finite bitwise first-logit/residual contracts. Flag0 still has no raw full-logit
export. Both logits-only and all-layer arms are compared explicitly. It does
not compare cancelled work as matched numerical parity; cancellation remains
its own isolation fixture. Missing vectors cannot be called equal.

## Strict observed coverage and immutable predecessors

The coverage checker audit_fidelity_observer_coverage.py is pinned to
c23f73131b4d525c7e322f913ae6f0b600b07073c0125b9d1d66824725dbec47.
Every noncancelled activations1 request requires all48 first-window residuals;
empty or partial logs fail. All raw vectors must match configured stage bounds
(onecard0:48, pair0:32/32:48), original source token/position and request/PID.
The full head must belong to the configured last stage even in logits-only mode.
Each observed prefill point also requires all48 layers; absent prefill points
remain explicitly unobserved. Persistent GDN/PLE/QSA state is still unobserved.

V1 SHA9356586943ee66a1254bee792428bcfcaefea8df1696570ea63ea7fd5e0de2f9
and V2 SHA5a930456b90257274358910eff71c04d3b25a5ea870bebfc2fc533c89c2db21e
remain untouched. V4 consumes the same tracked serial_prefix_prompt_fixture.py,
byte-identical to the preserved ignored token fixture. No prompts, request
roster, math route or numerical tolerances changed. The shared geometry is an
explicit new numerical-control configuration, not a retroactive reinterpretation
of old results. No GPU run accompanies this preparation.

## V4 identity-only increment

V4 preserves frozen V3 SHA8fc7e840b126c13903df6119b9a566cb09082c7e69acd0328b03b854495ed155.
V3 already replaced ctx8192 with ctx2048, but inherited one/pair aliases could
collapse onto the same string. V4 builds an explicit model/method/scheme pilot
alias containing pilotctx2048-pf64-ple65536 and cards/layer-bound labels.
base_prepared_alias retains the original C1 alias separately. hotschmoe-dd stays
the primary identity. Native GEN still makes no HTTP served-ID claim.

Geometry is checked from actual args before generating/accepting the alias.
The cross-topology comparator still permits only split/device/trim differences
and rejects all other normalized argument differences. Model/math/prompt/group/
raw-vector/counter/tolerance/health/teardown gates are unchanged. This does not
qualify production8192 capacity or concurrency.
