# H50 native6 instrumented CPU preparation profile

CONFIG -> Frozen H50 source5f719918, exact source37 SDK/private196/registry165,
native6 diagnosticOFF. Same original source/identity guards; CPU metadata only.
Concurrent H50 native4 parent was a separate leased actor. Profiler results
are instrumented diagnostics, not clean serving latency or matched improvement.

COMMAND -> Actual session5705 terminal0: python3 -m cProfile -o
batch50-native6-off-prepare-profile-v1.pstats batch_numerical_execution_v50.py
prepare, using batch-numerical-case6-source37-v11.json and original prerequisites.

RESULT -> Genuine prepared native6 plan SHA256
a62de9a86715a9dcfbdc3bebac461ad904a20c695ce0acad7bd37c59035ca580.
PStats total instrumented time1455.5672289610027s, total2201896304calls.
Exact original_upload_gate in c137_sdk37_digest_ports_v1 called624 times,
cumulative1139.240974213s. It directly called parse_trace3120 times and
negative_controls3120 times. Each negative_controls performs four parses,
including another positive:12480 additional calls. All15600parse_trace calls
came from that same upload gate. parse_trace self679.015823521s,
cumulative1070.612996959s (73.55 percent of total instrumented time).
Regex search361064003calls/self136.257222873s. Hashlib update self21.518779497s.
Cumulative times include children and must not be summed across overlapping
functions. Profiling adds overhead; the fractions are localization evidence.

VERDICT -> Repeated logical-free parsing is an actual dominant preparation
cost. The reviewed unintegrated immutable parser V2 proposal targets the exact
five-case byte/parser/owner-count work, retaining negative controls and leaving
all live source/model/health/ownership gates outside the memo. Future consumer
integration and a matched new profile plus clean execution are required before
claiming a dev-loop speedup. No GPU inference, serving speed, complete model
math, cache or shelf qualification follows from native6 plan preparation.

Profile SHA256: 910e24a0838954e0948b20a5d5323f1b0a44121de59c7a06d222edc2938057f5

Summary artifact: /mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f17-source37-20261010/batch50-native6-off-prepare-profile-summary-v1.json

Summary SHA256: 0d8f2e008ef7489323c4df5fb4f104fc5d4ead8b1df684a9a5a4b984747c535a
