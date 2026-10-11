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
# Clock-enabled capture on 2026-10-11

CONFIG -> Fresh clock-export PTI build, unchanged source40 indexer fixture,
GPU0 workload with pair exclusion for health. Source plan V3 b593f409; the
previous stock-profiler capture remains intact.

COMMAND -> Root actual run41083 used probe_unitrace_indexer40_v3.py with a new
unitrace-indexer40-v3-clock-capability-run-v1 output directory. Nine CPU controls
and the focused peer passed. Actual no-device inventory resolved all16 helper,
5 tool and9 injection dependencies. The actor acquired bin/gpu-run exclusion.

RESULT -> PASS151s, report SHA256
91b08034d2e7fa1121175c9ce7ebc3542085d2bea083e29f012941dca5b51ad2.
The independent public reader passed. All18 OFF/ON frames and eight fields
match bitwise; exact-owned normal terminal/removal, current source/dependencies,
pre/post strict and compiled health and kernel journals passed. Actual clock
file738304B has921 records: one host origin, one device origin and919 unique
operation records, matching919 original Chrome GPU IDs. Each raw timestamp
normalization recomputes exactly using its retained frequency, mask and scalar.
Clock file SHA256:
5b08ab763f9070f720b265b120f900572cd30b08015856d6b2df1e50e42df610.

VERDICT -> Raw per-operation clock export and fixture correctness now pass.
Independent conversion/serialization precision and replay-flow audit remain
separate. Instrumentation adds observer work and stock collector waits; no
normal-model GPU idle fraction, bandwidth saturation or serving gain follows.
No model file was mounted in these fixture containers.

## Independent clock audit findings

CONFIG -> Read-only root checks of the completed V3 capture, preserving raw
clock SHA5b08ab76 and Chrome SHA6094ed00 before and after each query.

COMMAND -> Reproduce the source's F64 epoch conversion and six-decimal Chrome
serialization for all919 operations, using Decimal for the consumed JSON.
Separately compare operation23 and24 in raw device and normalized host time.

RESULT -> All919 displayed start/duration values match the original source
conversion exactly. Epoch F64 spacing is0.25us; maximum serialization error
against the exact integer epoch is0.125us. Duration serialization is exact for
these records. This verifies serialization, not physical clock calibration.

Operations23/24 share the same device and engine. Their raw GPU timestamps
are ordered with a133-tick gap (6927.083ns at19.2MHz), but separate submit-pair
normalizations reverse their starts and create a200ns host-time overlap.
Between the two submit pairs, host time advances67857ns while device time
advances1614ticks (84062.5ns): a16205.5ns residual. The apparent overlap is
not proof of parallel execution. The profiler can allocate another synthetic
track to represent such overlap; track count is not physical engine count.

VERDICT -> Use original device-clock ordering for descriptive fixture
intervals. Do not treat separately normalized host-track unions as physical
GPU busy time or infer model idle time from their complement. Host-clock pair
variability, boot-versus-MONOTONIC_RAW origin handling and observer overhead
remain explicit limits before host/model critical-path correlation.

The complete independent audit is preserved under f17 as
b70-unitrace-v3-clock-raw-coverage-audit-v1.json (SHA256
c6f3ea2f645606c479ad3f10465e1774e4b158364aedd324d310bc5c83bdc040),
alongside its exact script and analysis plan. It joins all126 compute,631 copy
and162 fill operations, all12 replay flows and all18 source frames. Root
independently confirmed no overlapping raw device intervals and union42150
ticks over all919 operations. This is an instrumented fixture description,
not a model idle or latency measurement.
