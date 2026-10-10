# Exact known HC V3 read-only FP log adjudication

CONFIG -> original V3 parent PASS,36 frames/3802788 words/raw/lifecycle/health/
new4; original reader FAIL FP capability observation scope changed.
COMMAND -> NEW adjudicate_hc_composition35_v3_log_v1.py --run-root preserved
hc-composition35-v3-run --output NEW_DIRECTORY_OUTSIDE_ORIGINAL.
RESULT -> CPU6 source/tiny mock controls only. Root actual admission required.
VERDICT -> new read-only view, original reader failure/source/evidence retained.
No primitive, source SDK, fixture, compile, GPU or model re-execution by agent.

CPP printed the FP prefix before device.get_info SINGLE_FP_CONFIG. Under actual
UR2 tracing, one successful query's entry/exit text splits that prefix from the
enum suffix. The recorded leaf SHA is exactly unchanged. The old reader assumes
one line and rejects scope. This new reader admits only the exact known three
lines:original prefix + one successful SINGLE_FP_CONFIG/propSize4 query + exact
32,16,2,4,8,64,65 enum/scope suffix, immediately between DEVICE and CONFIG.
Query device must match the preceding successful driver query. Unknown/extra/
failed/truncated/different-device insertions are refused. Original lines and
global line indices remain in the new report; no logfile is normalized on disk.

The exact completed run path and original parent/raw/free/log SHAs are pinned.
The original frozen V3 reader first runs and must fail only with the known exact
format message. Then one process-local leaf-log observation function is replaced
by the explicit joined view while ALL OTHER frozen finalized V3 gates rerun:
source/SDK/archive/ELFs/compile/input/host, actualraw, USM, journal receipts,
health, normal owned teardown, post4/currentstat/pages and chronology. The
function is restored in finally. Before/after original tree SHA+stat5 equality
and exact source/evidence pins are mandatory. Output must be new and outside
the preserved run tree. Original reader success is explicitly false.

Observed18 analytic frames have both mix variants equal. Observed18 heldout
frames have fused equal and separate unequal. That is a current synthetic
source observation, not selection of a reference policy, universal device
intrinsic or normal fullmodel graph qualification. Host ordinary exp/rsqrt
hypotheses and original mathematical reference remain separate and unqualified.
Future CPP formatting changes belong to a new source version and fresh recipe.
