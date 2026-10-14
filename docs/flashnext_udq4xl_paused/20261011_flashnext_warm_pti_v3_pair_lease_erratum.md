# Warm PTI V3 pair-lease command erratum

CONFIG -> COMMAND -> RESULT -> VERDICT

CONFIG: Frozen warm V3 source plan d72fb6de retains the original paired lease
and per-arm per-card plus compiled-pair health. The model's affinity remains
card0. Its health work touches both cards; a card0-only outer lease is invalid.
The separate tiny paused-list fixture uses card0 lease and card0 health only.

COMMAND: Replace only the outer `bin/gpu-run --card 0` in the frozen guide
with `bin/gpu-run`. All actor arguments remain unchanged:

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 bin/gpu-run python3 strata/flash-next/run_source43_warm_pti_v3.py --leased --plan /mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f17-source37-20261010/source43-normal-cli-v1-prepared-v1/plan.json --output NEW_OUTPUT --phase host-latency --off-receipt /mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f17-source37-20261010/source43-normal-cli-v1-run-v1 --speed-trace ACTUAL_SOURCE40_SPEED_TRACE --pti-intake ACTUAL_PAUSED_FIXTURE_ROOT/report.json --pti-intake-sha256 ACTUAL_PAUSED_FIXTURE_REPORT_SHA --prompt-role short
```

RESULT: Root source review found the card0-only documentation command conflicts
with the unchanged `c.leased([0,1])` parent and paired health path. No warm V3
runtime was executed from that command.

VERDICT: Use paired outer lease for warm V3. Frozen source plan and original
guide remain unchanged. No new runtime or performance authority is asserted.
