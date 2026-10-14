# Original MTP existing-sync host-phase result, 2026-10-12

CONFIG -> Original primary+MTP34, same authentic54/GEN64 diagnostic
freshOFF/ON, card0, explicitcommit0/mirror0. Default-off CPU timing
buffer enabled, no additional GPU waits/queries/workspace. Exact
existing synchronization boundaries; no device-only phase timer.

COMMAND -> Native phase compile33774 PASS98s; actual22434 PASS599s;
independent saved readback PASS. All64 outputs/EOS match, native0/
EOF/ownedremove/source/ELF/full5/health/kernel pass.

RESULT -> Decode-only nonnested host phases:

| Phase | OFF count / total ms | ON count / total ms |
| --- | ---: | ---: |
| target_run | 63 / 16972.710 | 33 / 15803.596 |
| target_commit_wait | 63 / 41.121 | 33 / 22.844 |
| own_head_read_argmax | 0 / 0.000 | 33 / 42.242 |
| draft_prepare_submit | 0 / 0.000 | 63 / 3.188 |
| draft_record_wait | 0 / 0.000 | 63 / 120.079 |
| draft_settle_wait | 0 / 0.000 | 63 / 2.062 |

ON target_run is 98.81% of15994.666ms decode wall. Median ON target
window473.794ms versus OFF245.681ms; high draft acceptance does not
make verification constant-cost. This target scope includes its existing
CPU preparation, submission, GPU work and waiting, not GPU-only execution.

OFF8160 vsON7270 resident experts and different mirror footprints remain
explicit; these are not matched-placement arms. Existing verifier counters
are potentially overlapping and exceed wall time, so are not added or
interpreted as CPU active time. This run is materially slower than the
prior6.7s diagnostic pair; no population/clean-serving speed claim.

VERDICT -> Prioritize target T2 work/amortization, not draft micro-tuning.
A source audit of repeated per-row operations and actual hostmapped RAM
paths precedes any kernel proposal. Physical idle/bandwidth, calibrated
device intervals, full512 math and serving/shelf qualification remain open.

F17 original-mtp53-greedy64-phase-card0-run-v1/report.json SHA
fdaf82bfa21dd9e7f6ffd51aa25d96ec7bce06e580f515d651db5dc3394efaaa.
Independent public SHA
3e0e5b7067f7a15a1322df13aca39a74857578fe35d88b5d78aad0eea17daf1d.
