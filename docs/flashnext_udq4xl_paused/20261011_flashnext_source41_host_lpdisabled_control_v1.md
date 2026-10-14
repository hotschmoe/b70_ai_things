# Source41 H36 LPdisabled runtime control

CONFIG -> separate control over frozen host V2 LP20. Same actual source41
mode0 SDK/model/inputs/geometry/flags, four fresh absent-PIN GEN64 requests,
numeric/QSA/UR observers OFF. Only the GEN logprobs field is omitted. Native
req_logprobs defaults to -1, disabling LP copy/CPU work. Explicit logprobs=0
would still perform it and is not used. Existing seed option remains omitted;
source falls back to a clock seed, while temperature0 selects greedy. No
explicit seed1 claim or silent seed transport change.

COMMAND -> root runs:

    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python3 strata/flash-next/run_source41_host_lpdisabled_v1.py --plan ACTUAL_INTAKE/plan.json --output NEW_LPDISABLED_ROOT --phase host-latency --off-receipt CLOSED_V3_OFF_ROOT --speed-trace ACTUAL_CHUNK64/engine-token-trace.jsonl

It auto-acquires the pair health lease, pins model device0, and runs separate
H36 OFF/ON processes. Per-arm full4/three-page/health/journal/command/owned EOF/
normal removal/current source gates remain unchanged. Only ON enables H36 and
DECODE_TIMING. Unused ARM remains omitted; no observer polling delay is added.

The public reader requires no LP rows in either arm and no LPReadback/LPCPU
trace events in ON, plus output-ID/finish equivalence and both repeat pairs:

    PYTHONDONTWRITEBYTECODE=1 python3 strata/flash-next/validate_source41_host_lpdisabled_v1.py --root NEW_LPDISABLED_ROOT --output NEW_EXTERNAL_BINDING.json

If a closed LP20 host V2 root is available, append --lp20-root CLOSED_LP20_ROOT.
Its original reader must pass. Exact original intake/model/SDK, native argv
and all environment fields except declared trace binding match; useful IDs
and finish compare across both arms. No equality of LP rows is claimed.

RESULT -> six tiny controls pass, including an actual owned Python protocol
process that rejects any explicit logprobs field. Original stdin confirms
omission, no LP rows are produced, and normal original pipe EOF is retained.
Synthetic H36 positive/negative controls prove mandatory absence of LP scopes.
No model/GPU/Docker payload execution by this author.

VERDICT -> source-only pending peer/root. The actual original request function
is preserved with exactly one command-literal omission through a private
namespace; original module/class globals are unchanged. A separate exact
validator omission retains all old request checks. H36 uses the existing
scoped decode primitive with strict source41 types and PID1 proof; optimized
prompt internals remain unobserved. Export IO remains outside Request spans.
No device clock, physical idle/traffic, API/shelf/fullmath or speed authority.
Frozen host V2 LP20 and all earlier measurements remain unchanged.
