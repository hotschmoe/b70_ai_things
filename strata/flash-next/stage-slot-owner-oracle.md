# No-weight stage slot owner oracle preparation

CONFIG -> Actual future fully linked combined20 engine and consumed owner header,
not the untracked source overlay. Fixed image39992/ggml3cf, cells64, no model files.
COMMAND -> CPU source preparation; mocked source reconstruction/owner tests;
synthetic Device/Host UR trace collector controls. No oracle compile or GPU run.
RESULT -> Reconstructed exact20 owner test PASS; synthetic82-owner collector PASS
and missing/double/failed-free, empty/wrong-roster controls reject.
VERDICT -> Source/recipe ready for parent compilation and leased GPU execution.
Raw GPU ownership/math/runtime remain unqualified until actual receipts exist.

Build after the actual combined20 full engine finishes:

```
python3 strata/flash-next/build_stage_slot_owner_oracle.py \
  --engine-receipt ACTUAL_COMBINED20_ENGINE/receipt.json
```

Parent compiler workflow must be leased under project rules. Builder itself
exposes no devices; it requires passing actual20 source/image/GGML/plan/ABI hashes,
consumed header48e1e2be and exact patch695689, links actual static engine/core/
kernel/CPU/GGML libraries and rehashes them unchanged. Old18/19/reference-only
library/source association is rejected. Full model390 gate is separate: this
oracle loads no weights and cannot replace it.

Runtime cases in stage-slot-owner-oracle-plan.json are one physicalcard0, one
physicalcard1, then pair0:32/32:48. The parent owns bin/gpu-run, strict per-card/
compiled health before/after, deadline, owned container terminal/removal and
kernel faults. Use the plan's masks and UR trace environment. Example container
process arguments are:

```
stage-slot-owner-oracle --cells 64 --stage 0:32:0 --stage 32:48:1 \
  --output NEW_RAW_REPORT.json
```

Three slot owners per stage use real bounded device allocations, session_init,
session_zero, host staging and owning queue. Constructor runs with currentdevice0
while the second-stage owner belongs todevice1. GDN sentinel writes/readback
across all three owners detect shared/overwritten storage. Vector3-to1 resize
invokes actual owning destructors, then a surviving RoPE table is rebound and
the final owner retires. Each stage also gets a fresh allocated/written/read/
freed probe. Aggregate live device arenas never exceed1GiB; cells are restricted
32..256 and qualification requires64. Same-device split is not a runtime case.

Production patch20 is not changed for fault injection. The oracle TU includes
actual session/owner headers and interposes only allocator and initializer call
names while compiling the owner header. Normal paths delegate to real SYCL and
actual20 session_init. Null injection returns nullptr without a driver OOM or
allocation. Partial zero/throw initializes a valid four-layer prefix inside an
adequately sized full arena, then deliberately returns0/throws. The real owner's
rollback releases that prefix and arena. This proves the owning failure branch
against actual driver allocations; it does not reproduce an unpredictable real
allocator failure halfway through its internal call. No malformed dimensions,
out-of-bounds memory or oversized allocation is used.

Every device arena, QSA host_step/host_pos, any hostKV and fresh probe emits
UPLOAD_USM registrations with live type, requested extent and owning native
context. Destroy markers bracket actual owner frees including rollback. There
are82 registered owners per onecard case and92 forpair. Typed roles and exact
normal/partial/probe stage rosters are independently checked. The existing parser
handles UR Device/Host/Shared allocation/free and chronological context bridges:

```
python3 strata/flash-next/collect_stage_slot_owner_oracle.py \
  --raw NEW_RAW_REPORT.json --log COMPLETE_UR_STDERR.log \
  --case owner_pair32_16 --output NEW_LOGICAL_FREE_REPORT.json
```

Raw report passed remains false while awaiting trace. Collector requires raw
owner/probe pass, exact counts/role/stage coverage, all registered allocations
retired between owning begin/end markers, no live trace allocations and mandatory
missing/double/failed-free negative controls. It never treats stale postfree
pointer type as a leak or dereferences freed storage.

This does not test actual inference, source model bytes, native HC/PLE math,
batch graph retirement or session/cache correctness. Those require complete
source390 and full model per-slot diagnostics after this ownership foundation.
The legacy process _Exit endpoint and child container teardown/post-health remain
separate observed facts. Preserve all raw traces and failures.
