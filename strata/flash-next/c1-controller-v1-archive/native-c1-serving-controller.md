# Native Strata C1 serving controller

CONFIG -> Exact engine source/build, original four selected GGUF shards, F04
native-source-pack-intake-v4, original tokenizer/template exports, immutable API
identity patch0007, and passing version2 source/upload/logical-free lifecycle.
This is a C1 diagnostic controller, not a verified shelf entry.

COMMAND -> Parent operations, in order:

1. `python3 strata/flash-next/c1_serve_controller.py prepare-runtime --output NEW_RUNTIME_DIR`
2. `python3 strata/flash-next/c1_serve_controller.py prepare --profile one-card --runtime-receipt NEW_RUNTIME_DIR/receipt.json --upload-lifecycle F06/source-upload-v2/receipt.json --oracle-receipt MATCHING_ORACLE_BUILD/receipt.json --verify-model-shards --output NEW_RUN_DIR`
3. Record fresh parent health and acquire the one-card lease:
   `bin/gpu-run --card 0 python3 strata/flash-next/c1_serve_controller.py launch --prepared NEW_RUN_DIR --pre-health CURRENT_HEALTH.json`
4. While that foreground controller retains its lease, run:
   `python3 strata/flash-next/c1_serve_controller.py screen --prepared NEW_RUN_DIR`
5. `python3 strata/flash-next/c1_serve_controller.py stop --prepared NEW_RUN_DIR`
6. Run parent post-health after removal, then:
   `python3 strata/flash-next/c1_serve_controller.py finalize --prepared NEW_RUN_DIR --post-health CURRENT_POST_HEALTH.json`

For the pair, prepare a NEW_RUN_DIR with --profile two-card and use a pair
bin/gpu-run launch with --one-card-receipt ONE_CARD_RUN/qualification.json.
One-card uses context2048, prefill64 and65536 cached PLE rows; the pair retains
context8192/prefill128 and explicit --layer-split32 --split-device1
--trim-stage-weights. Both retain the existing bounded64GiB global mirror cap,
2048MiB stage reserve, original IQ4_NL PLE mmap, FP16KV, MTPoff, C1, adaptationoff
and prompt-cache0. These are sizing hypotheses; model launch must fail if the
stage-correct VRAM/mirror contract cannot cover every required expert.

Parent health receipts must contain passed=true, cards=[leased physical cards],
finished_epoch, and a nonempty files list of actual command/log artifacts with
path andsha256. Pre-health must finish within300 seconds of launch. Post-health
must finish after actual container removal. Health/recovery tools remain parent
operations; this controller does not bypass leases or reset cards itself.

RESULT -> The runtime recipe adds an isolated Python3.12.3 venv with exact
regex2026.9.10, Jinja2 3.1.6 and MarkupSafe3.0.3. The base image has no pip or
these dependencies. A local runtime base tag is accepted only after its imageID
matches sha39992d70; dockerbuild uses --pull=false. A before/after content census
requires all selected LevelZero/SYCL/UR/IGC/GMM native libraries unchanged.
The resulting execution image is pinned by its own sha and receipt. No shared
host Python packages or GPU runtime libraries are replaced.

Preparation validates current executable/patched sources, every intake pack
file, exact exports, registry alias, and optional completed v2 upload lifecycle.
It exports artifact-identity.json in the actual frozen API schema. The primary
served name is hotschmoe-dd, first, followed by the detailed registry alias.
Without runtime receipt, passing v2 lifecycle and fresh full GGUF hashes,
launch_allowed remainsfalse. Pure preparation never discovers or executes GPU.
The --verify-model-shards step reads all four full files and records their
current stat signatures; launch refuses any changed signature or missing hash.

The v2 lifecycle gate rechecks frozen oracle plan/source/binary fingerprints,
engine and pack identities, original roster/inventory fingerprints, every
captured source image's GPU SHA/type/shape/offset/extent, allocation accounting,
actual terminal process state and pre/post-health. It reparses chronological UR
logical-free logs with context/native-context bridges and matches saved ledgers,
requires every owned image/scratch freed inside its owning destructor, an empty
live ledger, and missing/double/failed-free negative controls. USM pointer type
afterfree is explicitly not used as a liveness or lifecycle acceptance gate.

Launch remains foreground and verifies inherited lease descriptors, live
container state and bounded API readiness. Default readiness is600 seconds;
default total supervised runtime is1200 seconds. No detached serving container
survives normal controller exit. Stop only targets the prepared run's labeled
container, allows75 seconds for the API's existing QUIT/terminate/kill ladder,
records terminal state and verifies removal. Forced child termination cannot be
hidden by a zero-exit API process: the tracer records actual child exit codes,
and finalize requires clean engine exit plus matched post-health.

The read-only tracer wraps frozen Service.encode_prompt and StrataEngine.generate.
It records exact rendered/submitted token IDs, generated IDs, hashes, enginePID,
sampling, engineDONE consumption counts, cancellation/error and duration. It
preserves the original methods and all yielded values. The six-request screen
checks both served IDs, literal output, arithmetic, a static user/assistant
history, a factual answer and exact repeats after intervening histories. Each
prompt must match the actual engine boundary and DONE must prove all its IDs
were consumed. Replies and generated token bytes must repeat with one unchanged
engine process. A passing screen is bounded C1 evidence only.

VERDICT -> CPU source syntax, actual source/free prerequisite revalidation and a
pure draft manifest were checked. Incomplete draft launch is rejected. Runtime
image building and all actual model serving remain parent-controlled operations.
HTTP exposes no full raw-logit endpoint; STRATA_STATE_HASH requires prompt-cache>0
and is not enabled in this cache0 control. Full-logit parity, numeric state/reset
and cached prefix behavior remain explicitly unproved. Use the engine's existing
standalone --tokens/--max-new1/--dump-logits route for separate matched teacher
forcing; do not claim HTTP text as full-logit/state parity. Concurrent serving,
complete model fidelity, long-context behavior and shelf promotion remain later
gates, regardless of a passing C1 screen and healthy lifecycle.
