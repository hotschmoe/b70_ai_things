# Replay-aware GPU capture

CONFIG -> Fresh pinned PTI unitrace on the unchanged source40 four-row indexer
fixture. GPU0 workload, pair leases for mandatory collective health. No model
inference, quantization change or driver update. The watcher actively sampled
the previously mismatching model page and can affect its cache residency.

COMMAND -> Fresh source build passed; current-container metadata and a minimal
card0 API-query smoke verified dependencies and the writable Chrome JSON path.
Reviewed unitrace V2 uses --output-dir-path alone. Root85357 completed in 147s;
the independent public reader passed. Failed V1 and device-free smoke attempts
remain unchanged.

RESULT -> All 18 direct/two-replay frames and eight fields are bitwise equal
between OFF and ON. Normal owned teardown, pre/post strict and compiled GPU
health, kernel journal and source checks passed. The raw 887734-byte JSON has
6069 rows and 919 unique GPU operation IDs:

- 126 compute operations, including 18 each of the three materializer kernels.
- 631 copies and 162 fills, matching the declared source operation counts.
- 12 actual host replay calls, joined through disjoint H2D flow IDs to their
  31/30/32-operation bodies. Each replay includes 24 state-snapshot copies and
  the expected 7/6/8 compute operations. Root independently reproduced these
 joins from the immutable raw file.

This recovers the replay coverage that the earlier VTune collector missed.
The graph-body D2H completion flows remain absent for 372 instances; these
missing host-flow endpoints must not be treated as complete synchronization
coverage. GPU operation intervals are present, but raw clock pairs, frequency
and mask values were not exported. No exact device-idle or latency claim follows.

Evidence under results/flashnext_udq4xl_20261008/f17-source37-20261010:

- unitrace-indexer40-v2-capability-run-v1/report.json:
  9dd367ba199fafe48437c53da42f5e5ed03f17f9390b6a7817e5253c127ea8e6
- on/unitrace/indexer-materia.30.json under that run:
  44ed516f2c13be8df911c54efafded25bf81d781f006ac039eb2bbcd5f561efa
- b70-unitrace-v2-raw-coverage-audit-v1.json:
  847c87782aa2d3c4812fbf0f478073d513f2ec9d51060102d68daedc2745507f
- unitrace-v2-root-replay-flow-query-v1.json records the independent joins.

VERDICT -> Replay operation coverage is established for this fixture and stack.
Calibration, collector-added waits and normal model critical-path coverage
remain open. The helper's reset/readback waits do not represent serving overlap.
The watched page stayed correct and a fresh post-GPU four-shard scan passed;
this does not establish the cause of the earlier cached-byte mismatch or
qualify host stability. Default-off export of already queried raw clock data
is being prepared. Source41 accepted-row GPU qualification is a separate live
experiment; full-model fidelity, cache, concurrency and serving shelf remain open.
