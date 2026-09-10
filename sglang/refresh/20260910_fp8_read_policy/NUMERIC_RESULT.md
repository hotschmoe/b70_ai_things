# SGLang FP8 read-policy numerical result, 2026-09-10

CONFIG -> Upstream current-main bdc on physical card0; read-policy14ee on
physical card1. Same frozen updated fixture/loader, exact per-image source
snapshots, fresh independent caches, 8GiB bounds. Decode q1 and extend q1/q4,
each with correct scales and separately doubled K/V descales. Tolerances
unchanged: rtol0.03, atol0.015. No model was loaded.

COMMAND -> Run prepared14ee strict per-card + compiled pair preflight under
both leases; then the two frozen per-card plans concurrently under selected
leases/device pins. Plans and commands are in paired_oracle_plans.json.

RESULT -> Preflight exit0: both cards healthy; compiled two-rank4x5120, ten
iterations, P2P0 passed. Upstream oracle failed all nine numerical checks,
expected lifecycle exit1. Candidate passed all nine with zero mismatched
elements, lifecycle exit0. Both passed exact store-byte, untouched-slot, clone
and scale-sensitivity checks. Worst maximum absolute error fell from0.137542
in upstream fixture results to0.000921488 in candidate fixture results.

| Mode | Query | Scale | Upstream mismatches | Candidate mismatches |
|---|---:|---|---:|---:|
| decode | 1 | correct | 48 | 0 |
| decode | 1 | K_descaling_x2 | 1006 | 0 |
| decode | 1 | V_descaling_x2 | 499 | 0 |
| extend | 1 | correct | 45 | 0 |
| extend | 1 | K_descaling_x2 | 1039 | 0 |
| extend | 1 | V_descaling_x2 | 488 | 0 |
| extend | 4 | correct | 137 | 0 |
| extend | 4 | K_descaling_x2 | 3165 | 0 |
| extend | 4 | V_descaling_x2 | 1778 | 0 |

All nine rows and full errors are preserved for both arms. Strict selected
pre/post-health returned0 for both. Eight ownership-cleanup records verified
container absence; both lease owner files were empty at allocation return.
No serving or Mamba oracle was launched. Compiled pair health was the
preflight stage; the two independent oracles had selected-card post-health.

VERDICT -> This narrow actual-Triton FP8 store/read fixture now meets its
numerical gates on the read-policy image. Results match the prior CPU policy
prediction. This is not model calibration/quality, native GDN, graph, MTP or
TP2 serving qualification. Different physical cards were used; a device
crossover was not performed. Candidate14ee also inherits the f82 loader
overlay, while upstreambdc does not; this fixture mounts the same loader
helper in both and never loads model weights or model-runner calibration.

The numeric gate permits the next separately scheduled full-model FP8 step;
it does not establish full feature parity or fix the original vLLM Pi defect.

Raw: /mnt/vm_8tb/b70/results/bang_isolation_20260910/sglang-fp8-allnine-upstream-bdc-card0
Raw: /mnt/vm_8tb/b70/results/bang_isolation_20260910/sglang-fp8-allnine-read-policy-14ee-card1
