# Bounded cold GEN4 and sampled query brackets: source-only first step

CONFIG -> fresh tool-only overlay on the exact retired model-finalizer PTI
source. Metadata and host bridge must be enabled, plus the NEW exact optional
UNITRACE_CLOCK_QUERY_SAMPLE_512=1. Absent/empty/0 retains the original exporter.
The original six GPU query calls, arguments, returned values and all
GetHostTime instance normalization fields/calculations remain byte-identical.
No native model, SDK, arithmetic, waits, graph lifecycle or registry changes.

COMMAND -> root generates patch with prepare_pti_clock_sampled_queries_v1.py,
then builds a fresh tool installation using the original strict build recipe
with this additional source overlay. Existing installed tool identity is not
accepted as the new tool. The executable cold case constructor is:

    python3 strata/flash-next/pti_bounded_gen4_contract_v1.py --recipe ADMITTED_ARM_RECIPE.json --inputs ADMITTED_INPUTS.json --output NEW_GEN4_RECIPE.json

The constructor verifies original four GEN64 rows and exact useful prompt
lengths54/524, retains ID bytes and short/long/long/short order, and changes ONLY
max_new64 to4 plus truthful scope fields and ON sampling flag. OFF/ON use
identical fresh1, ckpt1, temperature0, omitted LP/seed, absent PIN and natural
EOS/length terminals. Maximum output16 tokens across four requests. Runtime
actor integration is deliberately not claimed by this standalone constructor.
A new outer actor must actually consume the constructed recipes, require fresh
new-tool identity and preserve original per-arm identity/health/EOF/teardown
and matched outputs. Old actor/clock validators cannot admit this new schema.

RESULT -> query sites1/2 retain every initial root/subdevice bracket. Sites3..6
select first/every512th original query globally per site, not per device. Each
selected bracket retains original returned pair, CPU RAW brackets and following
RAW-MONO-RAW bridge. Every original query still executes. Six final summaries
record original calls, selected/completed and suppressed brackets; native close
requires exact selection arithmetic and completion. The independent streaming
row gate requires contiguous owner sequence, exact six ordered summaries,
matching actual query/bridge counts and final close bridge. It supplements,
never replaces, original raw normalization/Chrome ID/owner/clock property joins.
No instance-normalization row is suppressed; each selected scope operation must
still have its full original raw ticks and submit pair. Suppressed diagnostic
brackets are explicit, so an old every-instance query-bracket predicate must
not be silently reused. Fresh parser integration must preserve all remaining
raw-clock predicates and declare per-instance bracket coverage false.

VERDICT -> source/CPU only. No current capture, alignment, calibration, clean
latency, model math or complete operation coverage claim. Sampling cannot prove
host/device alignment or an endpoint on each device. Original512MiB/4096B
limits, serialized exclusive output and exit86 remain unchanged. GEN4 reduces
scope; does not promise fitting within caps until actual capture closes.

The failed PTI3 startup was only about14.5MB; request1 reached420MB clock and
395MB Chrome. Merely pausing startup is insufficient. This first cold GEN4
control is explicitly NOT warm-process decode. After it closes, a distinct
warm process must pause collection before warmup, resume before selected GEN4,
and retain selected graph-replay ownership through normal finalization. Named
session currently initializes RESUMED even with --start-paused; do not assume
that combination excludes startup. An actual owned temporal-control handshake
(or explicit default-off native request hook) is required before warm-window
claims. No such handshake is implemented or inferred here.

CPU COMMAND -> PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=strata/flash-next python3 -m unittest test_pti_bounded_gen4_cpu_v1 -q

CPU tests use tiny fixtures and patch application only: no native compilation,
GPU query, model payload or actual512MB trace scan.
