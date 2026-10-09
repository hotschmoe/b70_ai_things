# Batch numerical observation source draft 0024

CONFIG -> frozen actual20 source + final21 (3c282ced) +22-v2
(1c7e7352) + new0024. Native HC, batch2/4/6/group1, greedy text,
no-MTP/non-pipelined distinct-device stages. Old serial observer flags and eager
verification are explicitly rejected with this observer. No existing source,
patch, build plan or serial controller is changed.

COMMAND -> test_batch_fidelity_observer_cpu.py reconstructs pinned20 source,
verifies its38-file ledger and tracked patches, then compiles the actual new
headers with ordinary host g++ against mock queues in image39992. It does not
pass devices, SYCL flags or runtime libraries. Separate fresh final21+22 receipt
is named in batch-fidelity-observer-draft-plan.json.

RESULT -> source/helper mock validation only. No full SYCL TU build or GPU
execution. Actual compiler/runtime still need to establish graph recording,
replay, ordering and complete lifecycle coverage. This source remains a draft.

VERDICT -> no concurrent numerical, prefix, speed or fairness qualification.

The opt-in environment is STRATA_BATCH_FIDELITY_DIAG=1, with an existing regular
STRATA_BATCH_FIDELITY_ARM file and writable STRATA_BATCH_FIDELITY_DIR directory.
Without the flag no diagnostic device buffers, copies, graph variants or waits
are added. Before ARM, no snapshot buffers are allocated. Snapshots are allocated
on each verifier's existing owning queue only when a selected request runs.

There are at most six admitted RID records and two raw samples per RID. Each
sample contains all48 immediate FFN residuals (4x2560 F32) and the complete248320
F32 head before argmax, spanning every stage's own layer range. Across all stages,
6-row reusable device snapshots hold17756160 bytes; six complete paired samples
write35512320 bytes. The output cap is64MiB and span event cap8192. Ordinary graph
families remain unchanged; admission and selected later windows use distinct
families containing bounded D2D copies. Graphs retire before snapshot buffers,
slot arenas and expert mirrors. No model/math correction or scheduling change is
introduced. Copies and host dumps can change timing; raw observations do not
prove the observer has no race effect.

Admission is a real first target window at prompt-1. Later capture is a real
run_slot_rows window with at least two active rows after admission. A one-row
batch window cannot satisfy it. Every vector names native PID, protocol22 RID,
observer request generation, slot generation, row index/row count, input token,
absolute position and owning stage/range. Native PID is the native engine
incarnation. API TLS engine_generation and original control queues remain the22
transport contract; this increment does not pretend API engine generation is on
the native wire.

The initial coverage contract deliberately accepts distinct initial BGEN RIDs.
BYIELD/restart/solo migration identities stay protected by22, but raw numerical
qualification of those paths needs a later observer/controller extension. A
request admitted outside an assigned BGEN slot is unobserved. Missing admission,
missing later multi-row execution or fewer than48 layers/full head must fail.
The strict coverage helper rejects empty/partial/duplicate/nonfinite/wrong-stage
or stale-identity raw frames and incomplete per-stage actual prompt evaluation
spans. Span events are emitted after completed Prefill chunks and verifier
windows. RESUME/read_from/reread_to are recorded independently; a read_from
counter does not establish complete cache correctness.

Next steps are actual full source compilation, explicit graph-replay stamps and
host mock recording/replay tests, compile review with forthcoming0023 producer
hooks, then new source390/identity/health/lifecycle gates and matched per-request
serial vs2/4/6 output/full-logit/residual comparisons. Automatic bounded full
root/pin/leaf checkpoint-chain retention and sourcefresh controls for batch
remain a separate mandatory source increment. Fairness qualification requires
ongoing decode beside long prefill with request-local progress/stream timestamps,
P50/P95 and cancellation/teardown evidence; metadata alone cannot satisfy it.
