# Explicit immutable upload logical consumer ports V1

CONFIG -> Source-only successor proposal to frozen immutable parser V2
75f2433b. H50, original C137/prepared/private/serial helpers, raw receipts and
runtime source are unchanged. This proposal creates admission-only namespaces,
not a new SDK or an executable model controller. Source37 identity remains
64/28/37/8/6. No source39 marker or old-proof transfer is introduced.

COMMAND -> PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=strata/flash-next python3 -m unittest test_upload_logical_gate_ports_cpu_v1 test_logical_free_immutable_epoch_cpu_v2 test_logical_free_worker_owned_cpu_v2 test_operation_evidence_snapshot_cpu_v2

RESULT -> 45 CPU controls pass (11 consumer/compact/AST/callgraph plus34 frozen
parser/purity controls). Tiny temporary synthetic upload records use mocked
SDK scope/full-source shape checks explicitly; they are no actual SDK/upload
qualification. All16 generated static admission modules reprepare identically.
Undo restores the complete original AST, including every nonparser predicate
and returned provenance field. Resolved callees require logical_epoch and
logical_roster with no defaults, including aliased origin_arm and dynamic
historical selected_jobs/serial_vectors. Unrelated callees reject those kwargs.

Actual root instrumented bounded6 H50 preparation completed1455.567seconds
and2,201,896,304 calls. c137_sdk37_digest_ports_v1.original_upload_gate line35
had624 calls/cumulative1139.241seconds. Exactly15,600 parse_trace calls were
3120 direct positives plus12,480 via3120 negative_controls calls. parse_trace
self679.016/cumulative1070.613seconds, regex search361,064,003calls/self136.257,
hashlib.update self21.519seconds. This identifies a concrete repeated parsing
hot path. It is instrumented CPU preparation with other activity, not clean
serving latency or a measured improvement from this proposal. The summary
metadata alone was read by the agent; actual upload logs, SDK, pack and model
payloads were not. Summary binding is recorded in the source plan.

VERDICT -> Reviewable unintegrated source/CPU proposal. original_upload_gate
replaces only per-case logical file/read/positive/negative/ledger/count checks
with the exact immutable byte subset. Original SDK/oracle/pack/source receipts,
inventory390/source roster, source shapes, case state/static bounds, health,
model/full4/page/stat/ownership and provenance return values are unchanged.
strict_v2_upload_provenance is still executed fresh before original_upload_gate.
No whole-function result is cached. Current full_source_case_gate executes
again for every case/caller even when its pure logical token is reused.

To avoid copying a large parsed ledger3120times, the compact requirement API
returns immutable True only after complete original positive+negative+saved
ledger/count validation. It exposes no cached mutable ledger. The original
collect() API remains frozen and continues deep-copying full results; tests
mutate those returned full results and prove later values/tokens independent.
Only one initial isolation copy is made for each unique compact case key.
Key includes exact context/runtime/source/raw paths+byte hashes and owner count.
Every call keeps current roster/PID/phase/source checks, and any failed epoch
refuses token reuse. Predevice/post complete byte scans still detect mutations;
between boundaries mutation remains unobserved. No fields are dropped from
the original validation to get the smaller return API.

Required prospective routes
---------------------------
SDK upload -> prepared -> genuine C137 baseline/adjudication -> current H50
manifest and readonly final audit; historical privateV4 -> first49 -> serial
selection/manifest/reader -> private196; all route through separate explicit
ported namespaces with mandatory logical_epoch/logical_roster forwarding.
Original producer file identities remain original because their actual saved
source snapshots are immutable. New consumer source files require separate
future controller/source-plan pins; original plans are not rewritten into new
views. Current H50/private/serial CLI producers do not import these ports.
No runtime CLI/model launch function is copied into the admission-only ports.

Future integration must construct independent process-local compact epochs
for preparation/parent/child over the explicit complete sameSDK upload
log/ledger/source/runtime corpus and rejoin the actual caller expected roster.
It must persist/revalidate original worker attempt/retirement/output records,
entry/predevice/post byte witnesses and exact source/owner/chrono joins before
model launch and after terminal. No saved parser result, SHA monkeypatch,
stat-only evidence, original producer view or health timestamp substitution
is permitted. Actual H50 controls and full goal math/quality/cache4/6/latency
remain separate and unqualified by these source/CPU tests.
