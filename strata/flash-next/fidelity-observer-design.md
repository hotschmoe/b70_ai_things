# Bounded native full-model residual and first-logit observer

CONFIG -> Frozen actual v6 source:
/mnt/vm_8tb/b70/build/strata-native-hc-engine-20261009T065210Z-e6m4q9uo/source.
Original fb58e0 plus frozen0011/0012 source generation. Selected UD-Q4_K_XL,
c1 one stream, MTP0, no pipeline windows, unchanged math/cache/scheduler flags.
Source/CPU work only; no model/GPU execution or external checkout edits.

COMMAND -> Audit actual consumed SYCL Verifier header, serial GEN input path,
prefill token layout, verifier row mapping/head sampling and existing debug
copies. Prepare new0013 and independent engine generation; run CPU ASan/UBSan
on real observer helper declarations with explicit mock SYCL allocation/copies.
Reproduce: python3 strata/flash-next/test_fidelity_observer_cpu.py.

RESULT ->0013 SHA256:
5589910ef51d0e665f95e34f8f56ef26b021b0a724c95e0b2d173342d8894116.
Applies to the exact hashed v6 files. CPU tests pass default-off zero mock
allocations/copies, selected model geometry, stage/layer offsets, first T1 row
mapping, retained queue context/device guards, canonical LE signed-zero/NaN
payload bytes and6-request bounds. The mock writes24 complete40960-byte
residuals and one993280-byte full vocabulary vector for the tested stage.
Actual SYCL TUs/graphs and model outputs were not run by these tests.

VERDICT -> Concrete source enables initial localization of a first observed
layer divergence. It does not establish model fidelity, state isolation,
prefix-cache correctness, healthy serving or speed. Off/on output/full-logit
and lifecycle equivalence must pass on the SAME new binary before attribution.

## Existing source and mapping

Verifier::copy_logits(t) already forwards to the last stage and reads a full
head row after run(). CLI STRATA_DUMP_FIRST_LOGITS has that access, but serial
native serving did not expose a bounded raw full-vocabulary capture. Existing
STRATA_VERIFY_DEBUG captures only the first HC stream after each layer and is
not enabled by this increment.

c1's first window runs the prompt's final token alone: T=1, row0, absolute
position prompt_tokens-1.0013 records accepted full GEN IDs/PID/request ordinal,
actual resume count and that input token boundary. The final-stage snapshot
queues a copy of ALL248320 raw F32 logits BEFORE the existing argmax/sample
node. Readout occurs only after the successful first window. It is the raw
model row, not sampled/penalized/top-K probabilities. MTP/batch/pipeline/vision
requests are explicitly unobserved under this first c1 diagnostic.

## Scope and immutable graph variants

STRATA_FIDELITY_DIAG is default off, including snapshot allocations and graph
variants. Enable with a new regular STRATA_FIDELITY_DIAG_ARM after readiness
and a fresh absolute STRATA_FIDELITY_DIAG_DIR. Set
STRATA_FIDELITY_DIAG_ACTIVATIONS=1 separately to capture residuals. Existing API
engine-token-trace.jsonl continues recording renderer/consumed/generated IDs;
the source observer logs actual GEN input IDs and absolute token positions.

Raw vectors are4x2560 F32 residuals after EVERY layer. Prefill captures the last
token of up to2 chunks/request/stage. The first T1 verifier window captures
all layer outputs and the final full-vocabulary row. Each stage owns one
contiguous device snapshot allocation in its exact queue/context, with16-byte
compatible F32 layout. Diagnostic file output preserves LE float bits without
normalizing NaNs or signed zero. Model weights are never dumped or modified.

Verifier normal captured graphs retain their original topology. A distinct
first-window diagnostic graph includes snapshot copies; it is selected only
for a bounded active first request window, then normal graph selection resumes.
There are separate diagnostic variants for resident/doorbell graph selection.
Prefill copies run only for bounded selected chunks. Bounds are6 requests,
2 chunks/stage, one verifier row and64 MiB reserved raw output per process.
Opaque per-graph snapshot buffers cannot retain stale request labels: labels
are attached at successful runtime readout, not at graph capture. Request
correlation is shared across the existing asynchronous prefill stage thread;
this source mode assumes c1 serial GEN execution and does not support concurrent
requests. It adds no inference/math/scheduler option or reset workaround.

Snapshot readback waits and extra copy nodes perturb diagnostics. No timing
from this lane is a clean latency result. Normal EOF explicitly drains and
releases the added verifier graph/snapshot and prefill snapshot owners while
their original contexts are alive. Actual driver teardown remains a GPU gate.

## State and coverage limits

This first increment intentionally uses full residuals as a small first-layer
localizer. Persistent GDN, PLE, QSA/indexer/KV and pending-state internals are
UNOBSERVED. Raw F16/quantized state is explicitly metadata-only skipped under
the canonical F32 contract; it is not converted or declared zero. A matching
residual does not prove complete hidden-state or all-token equality. This
covers only the last token of each selected prefill chunk and first prediction.
Later selected-state snapshots should follow the first observed divergent layer.

Cached-versus-uncached qualification remains independent. The observer records
resume boundaries but does not change cache policy or prove reused/evaluated
counts. The prefix audit/counter increment must cover GDN/QSA/PLE/indexer,
all stages, state ownership and actual reuse. Preserve c1 no-cache baseline
until that later experiment is preregistered and qualified.

## Independent parent build and qualification

Use fidelity-observer-engine-build-plan.json with the existing parent-owned
build_native_hc_engine.py --plan workflow under bin/gpu-run. It reconstructs
v6 from original tracked source and appends0013 in a fresh source/build tree;
source hashes and frozen patch hashes are recorded. Leave v6 source/binaries
unchanged. Full compile targets consume sycl/include/strata/core/verify.hpp;
shared prefill.hpp has no SYCL shadow at this generation.

Parent controller compares the same-version flag0, logits-only and activation
arms for fresh/repeated/history requests with identical GEN IDs, flags, cache
placement and arrival order. Require exact first-vocabulary/output equivalence
and actual per-stage/layer coverage before blaming any intermediate. If only
the observed arm passes, retain a timing-sensitive diagnosis; it is not a fix.
Collect pre/post-health and clean teardown. Never report diagnostic token rates
or readable answers as whole-model speed/coherence evidence.

collect_fidelity_observer.py validates full raw widths, GEN position/ID binding,
explicit skips, signed-zero/nonfinite summaries and matched request comparisons.
A first observed residual difference localizes a layer/token boundary; it does
not prove the earliest unobserved operation. Missing points remain unobserved.

Parent/peer CPU compilation follow-up: backend_audit's independent reconstructed
0013+0014+0015+0016 overlay exactly matches its compiled consumed files,
including SYCL verify.hpp and shared prefill.hpp. generate.cpp object compiles
without devices. Receipts:
/mnt/vm_8tb/b70/build/strata-prefix-combined-source-20261009/compile-receipt.json
/mnt/vm_8tb/b70/build/strata-prefix-combined-source-20261009/reconstruction-receipt.json.
This verifies the combined main translation unit, not every observer TU or
full linking/runtime behavior. Collector one-byte negative control passes.


Full-build correction0017: the combined v2 full build exposed14 prefill.cpp
errors because frozen0013 placed its observer fields in PeerPrefill instead
of Prefill::Impl. Preserve that failed build and frozen0013. New0017 relocates
only those declarations using the explicit Impl/Gemm owner. Actual consumed
prefill.cpp and verify.cpp now compile without devices, including the SYCL
verify.hpp override. See prefill-fidelity-owner-correction.md and its receipt.
Use the new prefix-full-state-engine-build-plan-v3.json for full linking;
previous helper/main-object checks did not cover this owner error.
