# R276 phase plus accepted-count ordering candidate

CONFIG -> Base is the previously tested GDN phase image
sha256:0328900cf1f8f29f5a8e76ed21a3eff71ef54d0abf495a88fdb86e93e880b077.
Apply only ../patches/r276-accepted-count-order.patch to its original runner.
This waits for the prior D2H before request additions/condensation/reordering,
then consumes accepted counts in their already-current row order. Both parts
are required. There is no Mamba cache-drop guard or native change.

COMMAND -> Read the exact original and prepared candidate source; check
`patch --dry-run --batch --fuzz=0 -p2` against the installed-source export.
Run ../test_r276_accepted_counts.py against baseline source and
../test_r276_accepted_counts_candidate.py against the prepared candidate runner.
The first executes actual baseline row methods/gather with simulated DMA; the
second checks the actual candidate early-wait position and executes its actual
accepted-count branch for swapped, condensed/swapped and new-request mappings.
No patch was applied to the live source or any existing image.

Copy the exact candidate runner to an isolated build context. Tag the immutable
base with the unique full-ID alias in Dockerfile and verify its resolved ID.
Build using docker build --pull=false --network=none --iidfile image.id .
with a .dockerignore allowing only Dockerfile and gpu_model_runner.py.
Run source_identity.py and native_identity.py in baseline and candidate
containers with --network none --cpus 2 --memory 4g and no devices mounted.
Remove the owned CPU containers after each check.

RESULT -> Runnable candidate
sha256:7b107d0e675390fedbbdbe743f104c08072b6175ce8ccbbfdad64dc82bf90ad1.
All 2106 native files and 282 installed distribution identities match baseline.
Exactly one file differs among 2227 vLLM Python files. The GDN phase fix remains
byte-identical. Base54 layers are an unchanged prefix followed by one COPY.
Environment unchanged. Baseline16 CPU ordering outcomes and candidate3 actual
branch checks pass; candidate source SHA is recorded in manifest.json.

VERDICT -> Ready for an isolated runtime comparison, not GPU-qualified and not
proof that this mechanism caused the Pi failure. A stable serial row zero does
not exercise the cross-request permutation defect. Require unequal accepted
counts, observed row transitions and matched cache geometry/configuration;
retain identity, teardown and strict post-health. No GPU work was performed by
this build task. Baseline images and earlier failure evidence remain intact.

The separate startup collective hook currently pins the original runner and
must reject this image until explicitly reviewed/repinned. It observes startup
Python collective boundaries, not live cache or accepted-count transitions.
See host_trace_review.md for its independent review and limitations.

Raw: /mnt/vm_8tb/b70/results/bang_isolation_20260910/mrv1-counts-phase-image

## Prepared runtime plans, not launched

Run `python3 vllm/int4/mrv1_count_candidate/preflight_plan.py` to print the
strict pair plan. Its --run command uses bin/gpu-run for both leases and the
reviewed owned preflight helper; unleased execution fails before GPU work.
The exact image has oneCCL under /opt/venv, matching the compiled probe flag.
Require its PASS before the TP1 serving plan.

The TP1 plan uses card1/port18146, stable primary hotschmoe-dd and separately
registered MRV1 research alias. Preservation Config.json and Mounts.json are
byte-identical to the original card0-prefixon control. Server arguments differ
only in image/card/port/name/alias and fresh preservation/output locations.
Jobs match after normalizing endpoint/output. MTP3, prefixON, FP16KV,
FULL_DECODE_ONLY, context/batch8192, c4, streaming shared-cache tool fixture and
bounds are held. The prepared output does not exist and no cache seed is used.
See tp1_plan_validation.json for the exact run_arm command and plan hash.
Changing cards still requires the original crossover for causal interpretation.
