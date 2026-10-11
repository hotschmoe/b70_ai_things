# Source41 native H36 host interval analysis

CONFIG -> Separate read-only analysis of a caller-specified, closed LP20 V2 or
LPdisabled V1 parent report with its exact SHA. Four useful GEN64 requests and
original OFF/ON/repeat, source, recipe, native namespace, and raw trace gates
remain unchanged. This analyzer does not execute model admission, tokenizer,
Docker, GPU, publisher scans, or profiler tools. It cannot replace the original
actor public qualification. Root must independently close that qualification.

COMMAND -> Run seven tiny CPU controls, then root may invoke:

    python3 strata/flash-next/analyze_source41_h36_latency_v1.py \
      --root CLOSED_ROOT --policy LP20 \
      --expected-report-sha256 EXACT_REPORT_SHA --output NEW_EXTERNAL_JSON

Use policy LPdisabled for its separate actual root. The original strict typed
collector is rerun and must equal the saved original coverage. Saved cfg/env,
recipe, submitted command, individual requests, source snapshots, original
parent result, and current trace hashes/stat5 are joined before reporting.
No parent PASS is invented for an open or failed run.

RESULT -> Native TTFT is Request begin to first TEmit; completion is the original
Request interval. TEmit gaps are marker-to-marker, not client delivery times.
P50/P95 use nearest rank ceil(p*N), separately per request; zero gaps yield null.
Each host kind is reported as an interval union, never an additive duration sum.
LPReadback/LPCPU intervals are also clipped to request-to-first-T and subsequent
T-to-T/last-T-to-end segments, retaining boundary crossings without double
counting. Combined LP union merges both kinds. ReplayWait/Submit and other
scopes are host intervals. Their nesting and overlap prevent additive cost
attribution. The all-kind union includes Request itself and is coverage, not
work utilization. Sampling spans do not prove device sampling execution.

VERDICT -> SOURCE/CPU only until root runs on actual closed inputs. H36 export
IO occurs after the Request span and can delay later queued requests. Optimized
prompt internals retain the original scoped unobserved flag. No device idle,
physical bandwidth, clean serving latency, interactive concurrency, model math,
profiler overhead, or calibrated clock claim follows from these metrics.
