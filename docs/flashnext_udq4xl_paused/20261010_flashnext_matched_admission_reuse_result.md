# Matched CPU upload-admission parser reuse

CONFIG -> Frozen benchmarkV2 d321495c/412 and compactconsumerV2 c756/400,
current actual source37 SDK/pack/upload evidence. Two ABBA rounds, eight trials,
first plus four repeated calls per trial. Existing outer upload guards run every
call; fresh logical/SDK/pack entry/predevice/post byte edges for both arms.
Known concurrent test/compiler jobs were held; perfect host isolation and OS
page-cache coldness are not claimed. No GPU/model inference.

COMMAND -> Actual60152 terminal0/PASS. Report
172ff115da00da97fb95aec646eb852684ad1a8e63c696ab4a79b6d0e90ea7f3.
All412 frozen source hashes rechecked after execution; all trial artifact SHAs
checked. Old/new outputs equal, wrong-owner latch and owned-copy mutation
controls pass, original inputs remain unchanged.

RESULT -> Original A repeated-gate median1.255748557s; reused B0.225575324s:
5.566870235x faster. First-call median A1.2387491175s versus B2.6596827925s,
so the first call is slower. Complete operation median (setup, first+four
repeats, generation guards and final byte/provenance closure) A9.9216711521s
versus B7.5933243036s,23.467285percent lower. Four trials/twenty timed calls per
arm; sixteen repeated observations per arm. Independent final reader pending.

VERDICT -> Measured CPU admission reuse benefit on this configuration, with
first-call cost. No complete harness preparation, decode tok/s, TTFT, serving
latency, model correctness/cache/shelf or OS-cold claim. New H54 integration
f7a473a4/480 passed root39CPU2.700s and peer scope review; root group0/1 CPU
preparations66898/73653 are live before any model execution.

Report root: /mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f17-source37-20261010/matched-upload-logical-gate-benchmark-v2-run-v1/
Summary SHA: c59e0a52ef16ea8d7092d3a3f6eb0536f538a8dc61496d3a81b18ba3e293317b

Historical failures preserved: benchmarkV1 da627 failed before trial0 because
its factory expected absent ledger log_sha256. H53 6a74 remains held because
its saved reader accepted mutated packet/argv/start fields. Successors use the
actual ledger schema and exact saved-worker provenance; prior files unchanged.

## Independent full recollection

Root full reader56730 passed, receiptSHA256
67464601f4ccc8897883c262715f2410d0066055091a846fe486f70b48714a54.
It rejoins all original trial/source/result/worker evidence and fresh complete
SDK/pack byte boundaries. Root-added summary is explicitly admitted/recomputed
by the new reader; the earlier exact-nine-file reader refusal is preserved.
Original negative temporary-copy bytes were deleted by the benchmark's own
scope and are explicitly unavailable for current recollection; fresh negatives
are rerun. GPU/serving and complete preparation qualification remain false.
