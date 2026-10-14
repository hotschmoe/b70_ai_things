# Closed PTI43 observed latency analysis

CONFIG -> Run only after validate_source43_pti_latency_v1 succeeds and its
external admission JSON is saved. Pin exact parent report and public admission
SHA values. This adapter reads closed raw evidence/source metadata only; it
does not rerun current model/health admission or read GGUF payloads.

COMMAND -> ROOT, after actual model capture/public admission:

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=strata/flash-next python3 strata/flash-next/analyze_source43_pti_latency_v1.py --root CLOSED_PTI43_ROOT --public-admission CLOSED_PUBLIC_ADMISSION.json --expected-report-sha256 ACTUAL_REPORT_SHA --expected-public-sha256 ACTUAL_ADMISSION_SHA --output NEW_EXTERNAL_LATENCY_ANALYSIS.json

The report requires the successful public result for the same actual43mode0
root/source/plan, LPdisabled coherence, finalizer close and both H36 coverages.
It pins every consumed file to the original parent artifact roster, exact
source/producer plan snapshots, and before/after full raw byte rosters. It
reexecutes existing typed H36 and clock/Chrome audits and requires their results
equal the original public admission. No saved pass substitutes for raw parsing.

RESULT -> Four native host request/token spans per arm, nearest-rank P50/P95
TEmit gaps, clipped interval unions by host kind, and paired profiler ON/OFF
observations. Nested host kinds are not additive. Request-to-TEmit is not
client first-byte latency; H36 export IO is outside the request span.

Per-device opaque L0 handle groups use actual raw device ticks/frequency/mask,
not host-normalized ordering. Overlapping observed operation intervals merge;
only the merged union is accumulated. Gap distributions are between observed
unions, and unobserved operations may occupy every such gap. Wrapped or
half-range-ambiguous epochs yield null union/gap results. Fractions preserve
exact duration conversion from declared timer frequency. No device union is
clipped to a host token window without an independently proven alignment.

VERDICT -> Observed capture diagnostics only. No transferred fixture counts,
complete device busy/utilization, physical idle, host/device alignment, copy
sizes, bandwidth, calibrated clean serving speed or new model qualification.
The first actual model capture is still pending; synthetic CPU checks test
union/overlap/wrap/type/coverage boundaries without GPU/model reads.
