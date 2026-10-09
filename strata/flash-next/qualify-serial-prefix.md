# Parent lifecycle for frozen V5 serial prefix diagnostics

CONFIG -> Frozen V5 controller46b15abba3fd60d432968420113a6f643a9e29a837f533ba5272d0481c91d253,
completed corrected0018/full390 source gate, actual one/pair CPU preparations,
matched2048/64/65536 pilot. Parent wrapper source is separate from V5.
COMMAND -> CPU mocked lifecycle tests and actual prepared-plan/controller/source
identity checks. No agent GPU launch.
RESULT -> PASS: actual wrapper main with mocked success/failure, fd8/9 forwarding,
stdout capture/forwarding, post-health retained after child failure, all4 tiny
buffered hashes, strict terminal-time/finalization negative controls. Actual latest
one-card source plan and independent full model identity pass CPU validation.
VERDICT -> Ready for parent-owned GPU execution. CPU fixtures are not model or
runtime qualification.

```
python3 strata/flash-next/qualify_serial_prefix.py \
  --plan F06/serial-prefix-onecard-v5-prepared-v1/plan.json \
  --group basic --output NEW_ONECARD_PARENT_DIRECTORY
```

Use absolute plan/output paths. The wrapper re-execs through bin/gpu-run and
holds BOTH leases even for one-card work, because pre/post strict health covers
both cards and the compiled collective checks both ranks. V5 receives fd8/9 and
its physical-card pin from the unchanged prepared plan. Do not nest gpu-run
around the automatic wrapper; --leased is only for a caller already holding
both inherited descriptors.

Later pair/group runs consume finalized child receipts:

```
python3 strata/flash-next/qualify_serial_prefix.py \
  --plan TWO_CARD_PREPARED_PLAN --group basic \
  --one-card-receipt NEW_ONECARD_PARENT_DIRECTORY/child/report.json \
  --output NEW_PAIR_PARENT_DIRECTORY
python3 strata/flash-next/qualify_serial_prefix.py \
  --plan ONE_CARD_PREPARED_PLAN --group pin \
  --basic-receipt NEW_ONECARD_PARENT_DIRECTORY/child/report.json \
  --output NEW_PIN_PARENT_DIRECTORY
```

Use a new output directory every time. Parent receipts are
parent-qualification.json; numerical results/captures live under child/.
The wrapper forwards supervisor stdout and retains complete command/log/SHA
receipts, kernel journal, health source/image identities and expected stage
ranges. It verifies immutable V5 and the model identity receipt before health,
after pre-health, during child work, and after teardown.

Each health phase requires strict per-card finite matmuls followed by the
compiled pair with P2P0. A failed strict per-card phase records failure and skips
the collective rather than claiming compiled health. Health shell cleanup traps
are allowed to finish; wrapper verifies any leftover owned health containers by
PID/name/label/image before releasing the lease. It never removes a foreign
container merely because it shares an experiment plan.

V5 performs normal QUIT and removes its containers. The parent independently
checks exact plan labels and child PID names, preserving logs/inspect state if
extra owned cleanup is required. Interruption/deadline sends SIGTERM; bounded
supervisor escalation preserves failure and still verifies container terminal
state. The leases remain held through this cleanup. There is no automatic reset,
cache invalidation, model mutation or redownload.

After child terminal and post-health, the wrapper reads ALL four original
publisher shards with ordinary buffered reads under the held leases. Every row
records SHA256, expected SHA, actual byte count and pre/post inode/stat signature;
all four rows are retained even when one mismatches. The scan starts strictly
after the child's report.finished_epoch and supervisor terminal timestamp. This
is a fresh complete scan, not a cached prepared hash or sentinel substitute.
The known4KiB source page is also watched during work. A watch failure preserves
a buffered page capture and metadata, stops the dependent child, and remains a
qualification failure; no cache/source repair is attempted.

V5 finalize runs only when child return0, numerical/teardown pass, no interruption
or forced removal, no kernel faults, both health phases and the fresh full-source
scan all pass. Failed child data and post-health/hash evidence remain available.
The parent never turns a strict numerical failure into a pass because health
recovered. Clean latency, independent full-model mathematics, HTTP/API identity,
production8192 capacity and concurrency remain separate qualification work.
