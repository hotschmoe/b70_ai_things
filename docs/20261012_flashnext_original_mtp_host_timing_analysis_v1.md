# Closed original MTP host timing analysis

CONFIG -> Only actual PASSED protected timing report with source plan90ddca82,
exact caller-supplied report digest, unchanged original artifact roster and
fresh reexecution of the SAME source timing/native collectors. Live/incomplete
root66379 data is not consumed as proof. No current model/GPU health authority
or fresh model/provider scan is transferred to this readonly analysis.

COMMAND -> AFTER root timing report actually passes and closes:

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=strata/flash-next python3 strata/flash-next/analyze_original_mtp53_host_timing_v1.py --root /mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f17-source37-20261010/original-mtp53-host-timing-v1 --report-sha256 ACTUAL_PASSED_REPORT_SHA --output /tmp/original-mtp53-host-timing-analysis-v1.json

New absolute output only. Source/report/raw-roster guards run before and in
protected finally AFTER all consumed reads, including failures. No fullGGUF
scan, source compilation, model inference or GPU query.

RESULT -> Pending actual root report. Per prompt/decode phase: count, nearest
rankP50/P95, total and max milliseconds. Nested stage buckets are explicitly
split bydevice label, never merged into an additive total. Actual(T,position)
joins each decode window with its stage0/1 host spans and committed tokenIDs/
ordinals. Stage0inclusive minus contained same-thread stage1 is HOST exclusive
elapsed (outside that nested call), not physical GPU idle. Accepted/rejected
windows retain actualT and committed-token counts; T1/T2 breakdown avoids
hiding a single-token tail. This conditional observation is not causal
acceptance overhead: position/history/catchup differ. First outputordinal1
comes from the prompt's final target row and is labeled separately.

VERDICT -> Source only until actual report admission. Two tiny tests validate
nested subtraction/token joins and nearest-rank statistics. Host elapsed
includes host work, driver/queue/synchronization and device completion waits;
no deviceidle/bandwidth/calibration/clean latency claim or nested-duration sum.
