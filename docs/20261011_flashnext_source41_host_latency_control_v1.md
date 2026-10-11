# Source41 mode0 H36 host latency control

CONFIG -> actual fresh SDK41 and completed source41 OFF model actor V3
prerequisite, original model/tokenizer, single device0, ctx2048/prefill64/
PLE65536, cacheOFF, native HC/device plan/nohost, no MTP. Both processes use
materialized mode0. Numeric/P30/input33/QSA observers remain OFF. Initial
requests preserve explicit LP20, including full-vocabulary readback and LP
CPU work. This differs from an ordinary API request without logprobs.

COMMAND -> root uses the existing source41 intakeV2 plan and closed actual
OFF source41 model actor root:

    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python3 strata/flash-next/run_source41_host_latency_v1.py --plan ACTUAL_INTAKE/plan.json --output NEW_HOST_ROOT --phase host-latency --off-receipt CLOSED_MODEL_OFF_ROOT --speed-trace ACTUAL_CHUNK64/engine-token-trace.jsonl

The actor auto-acquires the pair lease for strict/compiled pair health and
pins the model to device0. Do not nest leases. It independently reuses current
tokenization and the actual speed input association. Two separate source41
mode0 processes run the same four fresh absent-PIN GEN64 requests. Only ON
activates STRATA_DECODE_TIMING and the existing bounded H36 host recorder.
Unused ARM is omitted: this avoids the inherited observer-only one-second
post-DONE polling delay when numeric diagnostics are OFF. No new token waits.

The public root command is:

    PYTHONDONTWRITEBYTECODE=1 python3 strata/flash-next/validate_source41_host_latency_v1.py --root NEW_HOST_ROOT --output NEW_EXTERNAL_HOST_BINDING.json

RESULT -> source-only; fifteen tiny controls (seven new, eight frozen model
contract controls) pass. No GPU/model payload read or runtime by this author.
All existing per-arm full4/three-page/health/journal/owned attach+pipe EOF/
normal removal/source/current-prepared gates are retained. Failed traces,
overflow, clock/writer/owner/export errors and missing spans/tokens reject.
Both output IDs and DONE signatures must agree across arms and repeat pairs.

VERDICT -> diagnostic host intervals only. The collector preserves request,
reset/prompt, PLE preparation/gather, capture/finalization, submit/replay wait,
sampling, LP readback/CPU, commit, token printf/flush and DONE ownership.
Durations overlap; do not add them or infer physical device idle. Graph/queue
values are SYCL identities, not Level Zero handles. There is no device clock,
physical expert-traffic or API/shelf/fullmath/speed qualification.

RequestScope ends its recorded Request span before fprintf/fclose export.
That export can delay processing a next GEN already queued after DONE; its IO
cost is outside the H36 span and is not measured as zero. Raw close/session
markers, all exported files and actual EOF remain mandatory.

A separate recipe-only preregistration, preregister_lp_disabled(), omits the
logprobs keyword. Source defaults req_logprobs to -1; explicit logprobs=0 still
enters the full-readback/token-logprob branch. LPdisabled is not LP0. This
next control has no runtime option or PASS yet and does not silently alter the
initial matched LP20 actor. Two-card50tok target is not tested by this single
32GiB mode0 host control.
