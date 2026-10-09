# Strict C1v4 parent qualification

CONFIG -> New immutable qualify_c1_serving_combined_v4.py parent. Old
qualify_c1_serving.py, controllers and previous results remain unchanged.
Source-only preparation; no actual payload/page reads, runtime, GPU, SDK build,
commit or JOURNAL change. Strict C1V4/uploadV2/watchdog sources are pinned.

COMMAND ->

```
python3 strata/flash-next/test_qualify_c1_serving_combined_v4.py
python3 -m py_compile strata/flash-next/qualify_c1_serving_combined_v4.py
```

RESULT -> Eighteen CPU controls PASS using mock metadata and tiny synthetic
files. Legacy controller PASS, missing/failed parent, old preparation/new SDK
association, missing upload-v2 identity, stale/current-stat mismatch, incomplete
four-shard roster, wrong publisher hash, invalid chronology, partial/changed
page views, unbracketed scan and mismatched parent/final proof are rejected.
Same-inspection failed page views are preserved by the frozen uploadV2 helper.
No mock is published as a real model/GPU qualification receipt.

VERDICT -> Source-ready parent workflow; actual C1 qualification pending.
Use only a genuine strict prepared directory and parent-owned GPU leases:

```
python3 strata/flash-next/qualify_c1_serving_combined_v4.py \
  --prepared <genuine-new-onecard-C1V4-prepared-directory>
python3 strata/flash-next/qualify_c1_serving_combined_v4.py \
  --prepared <genuine-new-twocard-C1V4-prepared-directory> \
  --one-card-receipt <matched-onecard-directory/qualification.json>
```

The wrapper preserves inherited lease descriptors, GPU health and compiled-pair
health, source-bound private API configuration, readiness, screen/token traces,
owned stop/container label/cleanup, zero-exit supervisor evidence and matched
onecard-before-pair behavior. It pins the strict C1v4 six-Python-source controller
and its exact combined51/all8 SDK/new390/uploadV2/current source gates. Stable
hotschmoe-dd remains first and exact research alias/context/topology stays in
actual /v1/models and configuration. No model math/scheduler flag is added.

Both known shard3 buffered pages are inspected before prepared validation and
admission, at two-second readiness and screen polls, after owned terminal and
before/after final full hashing. Guard failure preserves both exact observed
buffers and stats without re-reading them as replacement evidence. Polls do not
observe activity between samples or prove direct-I/O/disk health. Current views
may be captured after a later controller error; they do not retroactively prove
the original failure interval. No automatic cache invalidation/repair occurs.

All owned processes/containers must terminate successfully, followed by matched
post-health. A NEW complete four-shard buffered publisher scan then runs with
float timestamp bound max(actual terminal, post-health finished). The independent
c1-post-model-identity-v4.json records every full byte count, publisher SHA and
pre/post stat. It cannot reuse admission or sourceupload hashes. A failed source,
health or incomplete lifecycle does not produce final C1 PASS. Failed scan output
is preserved; skipped dependent scans are explicitly UNOBSERVED.

Only after that proof passes does controller finalization run. Its original
result is preserved as qualification-controller-v4.json. A final qualification.json
is validated before publication and adds c1_parent_generation4,
post_full4_source_qualified=true, c1_parent_controller_sha256 and
c1_source_identity_proof(path,SHA). Separate c1-source-identity-proof-v4.json binds
the actual parent/controller/watchdog source, prepared/new SDK/upload-v2 identity,
terminal/health/scan bounds, health artifacts, independent scan SHA and both
bracketing page views. parent-qualification.json then binds the final result and
proof SHA with its completed terminal/health/source lifecycle status.

New consumers MUST call the default public helper:

```
from qualify_c1_serving_combined_v4 import validate_final_source_proof
proof = validate_final_source_proof(prepared_directory, prepared_metadata)
```

It validates both completed parent and final/controller/proof artifacts, source
hash/roster/chronology/current-stat metadata, actual health artifacts and both
page views. It performs no new model payload read. The parent uses its internal
candidate_final argument solely to validate the pending result before publication;
external consumers must use the default two-argument call after parent completion.
Frozen numericalV4 does not provide this stronger C1 final-proof gate and remains
a source prototype. New numericalV5 and concurrent consumers must pin this
helper/source plan and actual C1 proof artifacts; they may not trust only the
legacy passed/teardown/posthealth core fields.

This bounded six-request API screen does not qualify full model math, own-prefix
state, arbitrary sessions, concurrency2/4/6, latency/stability or shelf promotion.
