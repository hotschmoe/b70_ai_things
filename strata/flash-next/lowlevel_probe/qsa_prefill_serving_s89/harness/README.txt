S89 finite same-binary QSA prefill comparison - source-only handoff
CONFIG -> QSA eight-query tiling0/1, one new audited binary both fresh arms.
Q6CENTER0 common; native c1c Q6 host-marker source retained, HC common, S49OFF,
head RPW2OFF, wide/A2/RPW1/PF0/packedOFF, pairhead/depth2, C12288/Q0/FIRST0,
NVMe Direct PLE, two200k slots, all24576 experts, TRACE1 common. CPU ring,
GPU phase timer/profile keys absent. Marker is host selection, not GPU dispatch.

COMMAND -> ROOT ONLY, after source/AOT/build artifact audit and lease coordination:
cd /mnt/vm_8tb/github/b70_ai_things
s89=/mnt/vm_8tb/b70/results/flashnext_ista_gsq_rco_iq3xxs_strata_cachefix_20261014/qsa-prefill-serving-s89
# Set these to root's NEW frozen S89 build-closure values; never generic/stale exe.
bin/cpu-run python3 "$s89/harness-prep/make_configs.py" --exe "$S89_EXE" --sha256 "$S89_SHA256"
# Announce both cards and finite expected duration. Root may reference immediate
# prior full posthealth ONLY with no-intervening-GPU/stack audit; otherwise full prehealth.
# For arm0 then arm1, stop on first failure, preserving all raw observations:
arm=0
bin/cpu-run python3 "$s89/harness-prep/start_arm.py" "$arm"
bin/cpu-run --quiet python3 "$s89/harness-prep/client.py" "$arm" warm
bin/cpu-run --quiet python3 "$s89/harness-prep/client.py" "$arm" prefill
bin/cpu-run --quiet python3 "$s89/harness-prep/client.py" "$arm" functional
bin/cpu-run --quiet python3 "$s89/harness-prep/stop_and_health.py" "$arm"
# After BOTH full arm closures:
python3 "$s89/harness-prep/compare.py"

One ordered C2 coding warm pair,107prompt/1500completion per stream, validates
retainedS16 and exact cross-arm observable work/text/SSE/usage; it is not a
new timedC2 screen. Then EXACT saved S38mode0/screen requests p4096-0..2 and
p32768-0..2, in that order, cap256; actual prompts4059/32728. No arithmetic
precheck or extra32Kwarmed request precedes them. Functionals afterward do
concurrent323/667, reasoning391 and real tool-ID roundtrip.

Native work fields must exist, finite/nonnegative, and match across arms.
Requests reconstructed by existing bench_serving.send must equal savedS38
request objects exactly; fulltext/SSE/usage and cache/prompt/predicted/draft
work are compared, retaining all raw arrivals/hash/counts/timings. Prefill
fresh cache<=64 is explicit and must match between arms. HistoricalS38 text
hash coherence is descriptive only because source/builds differ. ON warm
and each ON prefill request stop on mismatched available cross-arm gates.
Final summary remains NULL for any identity/coherence/work/health mismatch.

QSA marker MUST remain absent through short warm and all4Krequests. After the
FIRST32K request it must appear exactly once with enabled=arm, active domain
4097..50001, first_active4097, nq256/stride50002/tile8/keyWG256/SG32. Later
requests preserve that same single marker. It is NOT a startup-readiness gate.
Native quantum0/TRACE headers provide observed read_part from/to intervals;
all six request log segments are retained. C12288/Q0 subdivisions are explicitly
SOURCE-DERIVED, not observed callback timestamps/device events. Public source
leaves short/ragged/decode unchanged; matched API counts do not prove every
internal per-window layout. No new profiler, source hook or phase timer.

Startup cpu-run releases after loaded; detached systemd service closes fd7
and retains only the GPU pair lease. Client/cleanup quietCPU leases go OUTSIDE
commands. NEVER wrap clients/stop helper in outergpu-run. Exact readyCID and
unique NAME must both disappear and lease wrapper exit0 before any health.
Cleanup remains required after client/functionals failure; stop helper does
not require a PASSfunctional. TERM invokes existing server engine.close QUIT;
no forced kill/restart/retry. An incomplete log/readiness observation preserves
the same healthy server for root inspection, not automatic rerun. Bothcard
and tencompiledTP health remain required per arm, with no nested GPU lease.

RESULT -> Pending; source syntax only checked. No configs generated, HTTP,
server, test, build, GPU or model call by this preparation task. Expected-source
and harness-source-manifest pin all six controllers and borrowed input records.
VERDICT -> ROOT REVIEW/BUILD-CLOSURE REQUIRED BEFORE EXECUTION. Three samples
per shape bound only this fixed-order comparison, not significance/p95 or
fullcontext/modelqualification. No daily promotion/2000tok-s claim.
