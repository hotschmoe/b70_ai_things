# Warm PTI ON-only retry and original OFF component

CONFIG -> COMMAND -> RESULT -> VERDICT

CONFIG: Original warm V3 whole parent f9f99fd3 stays FAILED. Its actual OFF
component completed two matched fresh GEN4 requests and normal retirement,
per-arm full4/pages and health. ON passed the mapped census but stopped before
temporal helper launch because the exec supervisor checked an old install
mount. New exec3 changes only that host mount association. No OFF model rerun.

COMMAND: First produce the named OFF-component public binding. This replays
actual original phase/raw/source/tree/per-arm health/identity/EOF predicates
and preserves the parent's errors. This is historical metadata-only admission;
current SDK/model/health authority and fresh hash authority are false.
New arm current admission and fresh first-arm full4 run after prerequisite reads.

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 strata/flash-next/validate_source43_warm_off_component_v1.py --root /mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f17-source37-20261010/source43-warm-pti-short-v3-run-v1 --report-sha256 f9f99fd39ee12dc82debcc75cf58fc805f8501d78271a1dce533bdcac06bef5f --output NEW_OFF_COMPONENT_PUBLIC
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 bin/gpu-run python3 strata/flash-next/run_source43_warm_pti_v4.py --leased --plan /mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f17-source37-20261010/source43-normal-cli-v1-prepared-v1/plan.json --output NEW_ON_RETRY_OUTPUT --phase host-latency --off-receipt /mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f17-source37-20261010/source43-normal-cli-v1-run-v1 --speed-trace ACTUAL_SOURCE40_SPEED_TRACE --pti-intake /mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f17-source37-20261010/pti-paused-regular-loaded-fixture-v2/report.json --pti-intake-sha256 ead4530510357315c8d2d6874378c92a5f3522f345c333c05b6173ffd6ad8dc9 --prompt-role short --baseline-root /mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f17-source37-20261010/source43-warm-pti-short-v3-run-v1 --baseline-report-sha256 f9f99fd39ee12dc82debcc75cf58fc805f8501d78271a1dce533bdcac06bef5f
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 strata/flash-next/validate_source43_warm_pti_v4.py --root NEW_ON_RETRY_OUTPUT --output NEW_EXTERNAL_ON_BINDING
```

RESULT: Single ON arm still executes original current SDK/config admission,
fresh per-arm full4/three pages/paired health, owned census/resume controls,
normal CLI EOF/removal and clock/H36/public gates. Its IDs/finish compare to
actual original OFF component; original V3 report is never rewritten or viewed
as passed. Component proof does not confer whole-parent/profile/math authority.

VERDICT: Source-only retry pending actual root execution. No full device
coverage/clock alignment/physical idle or clean serving speed claim.
