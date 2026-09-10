# MRV1 TP2 fresh FP8 qualification plan

CONFIG -> Phase plus accepted-count-order image7b107d0e, TP2/P2P0, MTP3,
FP8 e4m3 KV with fresh be02d915 calibration, context100000, batch32768,
max sequences4, prefix caching and FULL_DECODE_ONLY graphs. Preserves prior
fresh-calibration qualification configuration except the isolated MRV1 runner
change and explicit phase-mrv1 startup host instrumentation. Stable primary
model name remains hotschmoe-dd; the detailed alias is in models.yaml.

COMMAND -> Freeze a new raw directory with prepare.py, then separately launch
its plan using vllm/cache800k/run_arm.py. Preparation does not touch a GPU.
The existing campaign server acquires both GPU leases, runs the strict
per-card probe plus compiled P2P0 collective pre-health, owns the server,
verifies teardown, and runs strict post-health. No lifecycle bypass or reset
under a single lease is added. Parent owns launch timing and final evidence.

Jobs: Pi9 reconstructed replay and strict heuristic review; tiny24; early3
reconstructed replay and strict review; three deterministic concurrent rounds
(t0 seed42), followed by three sampled rounds (t0.7 seeds42/43/44). Each tool
round creates four fresh conversation histories, four tool/answer turns each,
32 exact semantic checks and records360. Each round uses a unique cache salt,
shared by all four sessions within that round. This provides a cold namespace
at round start and shared/reused state within the round; it does not imply
all requests are cold or reset the server between rounds. Generated tool IDs
can differ. Only deliberate namespace, sampling and generated IDs may be
normalized for payload comparisons; do not normalize actual SKU/count text.

The separately frozen probe fork changes only sampling options and round
namespace behavior, forbids recovery retries, and leaves the shared live probe
untouched. The offline gate independently checks exact SKU/count semantics,
finish reasons, all32 rows, attempt counts, absence of long repetition, and
positive reported cache hits in EVERY session. Missing usage or zero reuse
fails. Sampled outputs retain the same exact warehouse semantics; temperature
is not a reason to relax correctness. Pi heuristic flags, including budget
exhaustion, fail the automatic gate and require review; absence of flags does
not certify arbitrary Pi reasoning. Tiny controls retain their existing bounded
repetition/response checks. Generated tools are never executed.

RESULT -> Two CPU tests pass: exercise the actual fork through mocked HTTP
responses, verify namespace and sampling on all32 emitted payloads, and prove
the independent gate rejects zero-cache sessions, wrong count and truncated
finish. A frozen-plan test verifies six independent namespaces, records360,
exact image/trace selection and all eight post-workload gates. No GPU run.

VERDICT -> Prepared qualification, not evidence of stability or speed. Require
all job gates, WORKLOADS_PASSED, lifecycle rc0, complete paired startup profile
entry/return records, correct image/model/scales, and strict pre/post health.
Host collective returns do not independently prove device completion; captured
graph replay may bypass Python. Instrumented timings are not a performance
claim. External dependency hashes in manifest.json must still match at launch;
re-freeze if tracked lifecycle/source dependencies change. Preserve failed runs.
