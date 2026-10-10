# Immutable upload logical parser proposal V1

CONFIG -> New source/CPU proposal only. H50 and every original upload receipt,
parser, generic purity prototype and admission helper remain immutable.
No actual upload/SDK/model/pack bytes were read. Tiny synthetic logs were
parsed by actual fresh isolated CPU Python workers. Own worker process maps
and loaded Python/library bytes were observed only to bind those workers.

COMMAND -> PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=strata/flash-next python3 -m unittest test_logical_free_immutable_epoch_cpu_v1 test_operation_evidence_snapshot_cpu_v2

RESULT -> 25 CPU controls pass:15 narrow parser controls plus10 unchanged
purity controls. Five tiny cases called16 times execute5 positive parses and
15 real mutated negative parses. Source/byte/owner/lifetime/context/parameter
mutations fail, returned values are independent copies, and failed attempts
retain command/error/counter evidence. This is not an actual upload profile.

VERDICT -> Reviewable unintegrated proposal. The original parser has json/re
module and compiled regex dependencies; generic PurityV2 correctly refuses
that function. Its contract is unchanged. This narrow API accepts only exact
frozen sources, declared immutable file bytes and explicit scalar parameters,
not arbitrary callbacks, closures or saved results. Parser source is rebuilt
in a fresh -I -B -S interpreter with empty cwd and exact clean environment.
The worker records its actual PID/start and parent PID, loaded module/source,
pyc and mapped-library file hashes/stat5, executable/version/flags, packetSHA
and original terminal stdout. Parent observes actual child termination and
pipe drain. The runtime descriptor is freshly probed each operation and
freshly compared on every semantic miss, never imported as saved trust.
The complete observed worker runtime corpus joins ordinary immutable evidence
entry/predevice/post full-byte FD/stat5 reads. External unobserved process or
physical-memory runtime authority remains unqualified.

The pure subset preserves original chronological owner/free parsing, saved
logical comparison, exact owner count and all three negative mutations.
A positive parse is reused for the duplicate positive validation inside
negative_controls; missing-free, double-free and failed-free each still run
on independently mutated text. UTF8 strict universal-newline decoding matches
original Path.read_text(). Complete source/dependency hashes, runtime closure,
paths, raw byte hashes, require_owners=True and expected count are in the key.
Every caller rejoins its complete declared expected roster and execution
context; only this process/operation can use the memo. Full byte reads at
predevice and postoperation still catch changed evidence/runtime files.
Between those complete byte observations mutation remains unobserved.
No live health, time, lease, model/page, container, source-case, device shape,
source roster, inventory or SDK predicate is cached.

Source-indexed call scope
------------------------
Current c1_serve_controller_combined_v137.original_upload_gate imports
parse_trace/negative_controls atline232. For each case, line246 parses the
positive and line247 calls negative_controls. Its positive validation at
parse_usm_logical_free_trace.py line90 parses again, then line95 performs
three actual negative parses. Five actual upload cases therefore imply25
scans per admission before recursive duplication. Metadata-only earlier
census found14,232,467 bytes in five log/ledger pairs; logs themselves were
not read by the agent. Actual profile counts/cost remain pending root's
bounded6 preparation cProfile, so no causal timing or speed claim is made.

Reachable H50/SDK49/private paths retain the earlier DAG: genuine_baseline
rejoins prepared->original_upload_gate; current private arm/serial joins
recursively rejoin baseline/upload. first49 and serial selection have their
own pre/post artifact gates. H50 uses historical c137_prepared_pack49_v1 and
c137_sdk37_digest_ports_v1 through mandatory explicit pack/SDK epochs. Their
pure parser call subset is the proposed reuse boundary, not whole-function
memoization. The new original_upload_logical_bytes_v1.logical_case adapter
is deliberately unintegrated and runs only after original state,
full_source_case_gate and static-bounds predicates in a future successor.

Future integration must bind the proposal plan in a new consumer generation,
construct one explicit evidence epoch covering all declared upload log/ledger
pairs and complete program/runtime sources, and forward it to every pure
upload parser call. All original receipt/path/hash, source roster/inventory,
SDK magic/source/recipe/model/ownership and freshness gates remain live.
At predevice and postoperation it must publish and rejoin the complete byte
witness and original worker command/stdout/failure roster with parent/child
ownership and chronology. A generic source SHA view, monkeypatch or saved
result cannot substitute for this interface. No current H50 source or plan
is patched and no actual admission, quality, coherence or latency is qualified.
