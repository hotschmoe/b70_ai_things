S90 CPU analysis preparation (2026-10-15), /root/ple_upstream

CONFIG -> One stopped current32K ON trace, same S89 binary/149+363 ownership;
source-only preparation now. No GPU work by this agent since S89 full pair
posthealth18:01UTC. No raw trace loaded, calibrator/model/build/lease executed.

COMMAND -> Root supplies actual stable JSON filename after collect/normal
QUIT/flush/health and explicit analysis GO. Run ONCE under CPU lease:
cd /mnt/vm_8tb/github/b70_ai_things
bin/cpu-run python3 /mnt/vm_8tb/b70/results/flashnext_ista_gsq_rco_iq3xxs_strata_cachefix_20261014/current32k-profile-s90-analysis-prep/analyze.py /ABSOLUTE/S90/run/trace/ACTUAL.json --output /mnt/vm_8tb/b70/results/flashnext_ista_gsq_rco_iq3xxs_strata_cachefix_20261014/current32k-profile-s90-analysis
Do not execute placeholder or rerun collection/parse. Default --run points
current32k-profile-s90-prep/run; override only to actual retained run root.

RESULT -> NOT RUN. Python syntax parse only. Script reads/hashes stopped JSON
once, reuses summarize_trace.py union_us/percentile; no main tool/source edits.
Existing summarizer invocation can produce the basic census, but drops all
boundary-crossers and reparses JSON perwindow. This one fixed analysis keeps
clipped overlap plus separately counted full-contained service, unions each
metadata device independently, emits exactoperation counts/engine rows,
actual block_scores_prefill_tiled_s89 names/count/service, and all compact GPU
and host submit/wait event timelines. Next source-based phase selection can
reuse CSVs rather than repeatedly loading full raw JSON. Family hints are
only search aids; exactnames remain authoritative. No automatic topk/attention/
GDN/DQ/GEMM/EPwait latency attribution or saturation/occupancy interpretation.

CLOCK -> Actual S90 calibrate.py is existing S31 CPU-only64-bracket sampler,
BOOTTIME-minus-MONOTONIC_RAW. Collector runs it before/after and checks native
UNITRACE_SystemTime absent/0. Script preserves both shifts/bracket widths/drift,
uses before correction, never historical9.7s/10.6s corrections; durations are
unchanged. These are host correction samples, not crosscard GPU clock join.

LIFECYCLE -> Inspected S31 explicit pause/stop and S10 documented _Exit flush
fix. Actual S90 controller resumes only one profiled32K request, pauses/stops,
then sends ONE unprofiled arithmetic request so native observes stop/flush
before normal QUIT. No flush request belongs in captured request counts.
Collection.json nonempty trace is not validity; analysis validates stable
size/mtime/JSON, hashes exact bytes and requires actual gpu_op events.

VERDICT/LIMIT -> Actual window.json contains full request admission/capture/
prefill/32decode. It has no native phase endpoint timestamps. Native PP ms
and p0/T parts are durations/sourcebounds, not an epoch clock join. SSE ttft
has perf_counter-relative send origin without exact epoch sample; do not
subtract native prompt_ms from wall or use TTFT as exact prefill window.
Capture/finalize API candidates are retained, not silently removed. Actual
candidate name's source guard identifies prefill-only work; other source/shape
phase decisions require root inspection of emitted timelines. Cannot promise
exact per-device PREFILL busyunions before those boundaries are supported.
No physicalidle, P2P byte traffic, causal hostwait/criticalpath, additive
device servicebudget, clean speed or S10/S31 matched-config delta claims.
