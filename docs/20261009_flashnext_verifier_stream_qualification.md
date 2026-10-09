# Verifier init ownership and corrected source qualification

CONFIG -> Exact pinned model/runtime/Strata source; new immutable0018 after
0013..17. All four verifier stream fields stay null before initialization.
Explicit streams must match init device/context and ordering before allocation.
Capture, warm and teardown use the owning device. Mirror guard stays mandatory.

COMMAND -> verifier-stream-engine-build-plan-v1 full build; actual Verifier
two-stage control oracle; full390 corrected upload with actual32/16 static bounds.

RESULT -> Full engine and both required test executables build/link, exit0 in
277 seconds, with original source and plan snapshots unchanged. CPU contract
negatives reject foreign device/context/unordered queues and preserve the
construction0/init1 regression. These mocked checks are not GPU evidence.

The actual Verifier control oracle constructs both objects under GPU0, calls the
same stream initializer used by model init under GPU0/GPU1, and captures T2 then
T1 with the caller on the opposite device. Four tagged mirror-pointer graphs
replay eight times correctly; nine ownership negatives reject. Caller device
is restored, compatible explicit streams are borrowed, and the real mirror
guard remains active. Model weights and Verifier::record_window math are not
executed by this control. Actual process exit0 plus14 matched traced USM alloc/
free pairs establish bounded logical cleanup; no per-owner markers or physical
backing-release claim. Missing/duplicate/failed-free controls reject. Strict
per-card and compiled P2P0 collective pre/post-health pass; normal removal and
no reported fault signatures.

Corrected full390 uploads pass card0/card1/same-device24/24/two-device24/24 and
actual static32/16. All387 HC and3 original PLE GPU payloads match source SHA,
bytes/type/shape/offset. Ordinary300 matrices are enumerated/accounted, with
their GPU payload readback still outside this oracle. Actual32/16 ownership is
256HC/3PLE/200ordinary/2,554,839,040B then131HC/0PLE/100ordinary/1,266,933,760B.
Total owned image bytes3,821,772,800; allocation accounting, owner-specific
chronological frees, all free negatives, zero live ledger, normal exit/removal
and strict/compiled pre/post-health pass. All four fresh whole-file hashes after
the previous two-card failure/rebind match the publisher; pre/post stat including
ctime is unchanged. Cached-bit origin remains unknown.

VERDICT -> Corrected constructor/init/capture ownership control and exact model
source uploads pass their declared scopes. Corrected full-model one-card and
two-card serving, full-logit/layer/state/cache correctness, concurrency and
latency remain required; no shelf promotion follows these tests.

Evidence:

- /mnt/vm_8tb/b70/build/strata-native-hc-engine-20261009T085407Z-pzbw4jgp/receipt.json
- /mnt/vm_8tb/b70/build/strata-verifier-capture-oracle-yq0agoqy/receipt.json
- /mnt/vm_8tb/b70/build/strata-source-upload-oracle-full-2ipmlttr/receipt.json
- F06/verifier-two-stage-capture-v1/receipt.json and raw trace/free audit
- F06/source-upload-full390-corrected-v1/receipt.json and all five case reports
- F06/model-full-identity-after-twocard-fail-v1/receipt.json

F06 is /mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f06-20261009/.

## Corrected full-model serving screens

CONFIG -> Same0018 engine and actual32/16 two-card configuration; preserved
cached-source recurrence recovered and all four buffered hashes reverified.
COMMAND -> C1onecard-corrected-streams-prepared-v2 then matched
C1twocard-corrected-streams-prepared-v2.
RESULT -> Both bounded six-request API screens pass exact rendered/submitted
IDs, consumed/generated counts, model aliases, constrained answers and greedy
repeat IDs. Two-card T2 and T1 windows both capture on both stages without the
previous ownership guard failure. Every overflow expert is covered; stage0 has
7535 RAM experts/23,557,120,000B and stage1 none missing, with17041 total resident
experts. Native engine and launch supervisor exit0, owned removal and strict
per-card/compiled P2P0 pre/post-health pass; known source sentinel remains exact.
New post-one-card independent whole4-shard hashes match the publisher.
VERDICT -> Actual corrected full-model two-card serving passes this bounded
coherence/identity/lifecycle scope, beyond control-only capture. Full raw logits,
state isolation/prefix reuse, concurrent fairness and matched latency remain
unqualified; both receipts explicitly prohibit shelf promotion.

Full-model records: F06/c1-onecard-corrected-streams-prepared-v2/qualification.json
and c1-twocard-corrected-streams-prepared-v2/qualification.json.
