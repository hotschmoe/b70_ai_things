# Flash Next intake build and primitive qualification

This continues the [campaign plan](20261008_flashnext_udq4xl_campaign.md).
The selected checkpoint is Unsloth UD-Q4_K_XL, with unchanged publisher files.
Raw evidence is under `/mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/`.
F01 runs are in its `f01-20261009/` subdirectory. This is research qualification;
no shelf entry or full-model performance result is established here.

## Verified intake and actual storage

CONFIG -> Publisher revision `766911a6b7369840a91dbcd95f9f997acaab6cd6`,
four target shards, publisher Q4_K_M MTP sidecar and BF16 projector.

COMMAND -> `strata/flash-next/fetch_model.py` and the bounded header inspector.

RESULT -> Intake finished 2026-10-08 23:46:22 UTC, all eight selected files
verified. All six GGUF identities match the separate full-file verification
receipt. The header inventory is complete without reading tensor payloads.
The first backbone shard contains metadata only. Main and MTP token arrays
match; their chat templates differ. Keep the main template authoritative.

| Main component | Packed bytes | GiB |
| --- | ---: | ---: |
| Routed experts | 77,017,907,200 | 71.729 |
| PLE embeddings | 28,800,138,240 | 26.822 |
| Dense | 5,254,417,920 | 4.894 |
| Shared experts | 251,166,720 | 0.234 |
| Total tensor payload | 111,323,630,080 | 103.678 |

Separate MTP: 2,775,258,112 tensor bytes, 34 tensors. Projector: 907,523,008
tensor bytes, 334 tensors. Neither is loaded by the first text/no-MTP control.
The main model has 48 layers, 512 experts/top10, width2560, expert width640,
36 recurrent and 12 full-attention layers. The IQ4_NL PLE tensor is stored as
160-wide rows; the graph gathers 16 head rows and reshapes them to width2560.

VERDICT -> Model intake qualified by file identity. Runtime fit, numerical
fidelity, coherent generation and service behavior remain independent gates.
The complete inventory and tokenizer/template hashes are in
`full-gguf-inventory.json`; full hashes are in `model-intake.json`.

## Fresh compiler and aligned GPU runtime

CONFIG -> Unpatched llama.cpp `de7fa0a3c6a2e1b4cd9f22eb8d6bf5b12dbdb63b`,
oneAPI2026.1, SYCL ON, oneDNN ON, SYCL_F16 OFF, no archived native binaries.

COMMAND -> [build.sh](../llamacpp/flash-next/build.sh), then the separate
[runtime layer](../llamacpp/flash-next/runtime/Dockerfile).

RESULT -> Initial compiler job exited3 because the image login shell had
already initialized oneAPI and a second setvars invocation returned3. A
conditional initializer fixed this; build-v2 completed all 336 build steps.
Compiler/MKL are2026.1; oneDNN2026.0.1. Server, benchmark, perplexity and
backend/quantization probes were built and hashed. No GPU devices were exposed
to the compilation containers.

The official compiler image bundled Compute Runtime26.18. The research runtime
therefore installs hash-pinned official Intel NEO26.22, IGC2.36.3 and GMM22.10.0
packages, leaving the compiler stack unchanged. The image Level Zero loader is
upstream1.28.2 with Ubuntu24.04 package revision, while the host has1.28.2-2.
This package revision difference is recorded, not hidden as exact file identity.

The first runtime build command incorrectly supplied a local SHA as a Dockerfile
FROM reference; BuildKit interpreted it as a registry name. The next attempt
hit apt's local-file/no-download path error. The final installer uses only the
six verified local dpkg packages, removes obsolete PPA IGC libraries after the
NEO replacement, and requires a clean dpkg audit and resolved linker paths.
Neither failed attempt altered the host.

Resolved identities:

- Compiler build-v2 image:
  `sha256:64284c46a66e82463a49929fd0cbfca218c2aab7f9f409087f46a60304adce8a`.
- Runtime:
  `sha256:39992d7072aa0557f4e3a5faf7782f83bab8fe4882e9d8fcf3e660d3de50e9e7`.
- Server:
  `f5a0f84c0ec61e64d03006f8bd0952c29f94578a503150a5dee272e7dfac4eea`.
- Production-shape test overlay binary:
  `79fc0b73e2e7ab7a845e658c014ef3795034d6f29bd2b6909be7b4d7a92fd059`.

The test-only overlay adds actual512/top10 expert, PLE and16:48 GDN cases.
It is mounted over the test source in a compilation container. The external
source checkout remains pristine, and the server SHA is unchanged before/after.

VERDICT -> Build and package/linker preparation passed. These are not GPU
compatibility or model-coherence claims. Pre-health on both cards and compiled
two-rank P2P0 collectives passed before the first candidate primitive run.

## Primitive v1 and host crash localization

CONFIG -> Same runtime and test binary, SYCL graphs disabled, per-card affinity,
single worker, CPU-reference comparisons, persistent device-code cache enabled.
Both cards leased; physical card0 tested first. No model server was launched.

COMMAND -> [run_primitives.py](../llamacpp/flash-next/run_primitives.py), initial
combined plan; then isolated last-large and small-multiblock debugger runs.

RESULT -> Card0 printed OK for eight production512/top10 expert comparisons,
three IQ4_NL row gathers and thirteen selected dense comparisons. The process
then exited139 with a host general-protection fault in libc. The full primitive
run failed; card1 was not launched by that run. An OK print precedes local RAII
cleanup, so it does not independently prove successful teardown of that case.
Both cards and compiled ten-iteration P2P0 collectives passed post-health.
No GPU hardware-fault signature was recorded.

The original dense-decode expected count5 was wrong: nine source-generated
cases match the filter. The original plan was archived byte-for-byte under
`primitives-v1/`; plan generation2 records24 small cases plus12 production
cases,36 per card. This is an explicit count correction, not a relabeled pass.

The first debugger setup attempted GPU auto-attach and stopped before any
comparison. The host-only debugger uses `INTELGT_AUTO_ATTACH_DISABLE=1`.
The last large Q8/n2 case completed independently on both cards. The isolated
small-multiblock group reproduced the host crash before completing any case.
Thus accumulation of the large expert cases is not required for that crash.

Captured stack in `diag-small-multiblock-gdb/card0.log`:

```text
strcmp: invalid input pointer 0x8000000000000000
sycl::detail::getSortedImages
PersistentDeviceCodeCache::getItemFromDisc
ProgramManager::getOrCreateURProgram
```

The stack closely matches Intel's
[device-image sorting fix](https://github.com/intel/llvm/commit/49dc9346c2e9efbe171d86117966c52c65501ea7):
the old comparator assumes every device-library image has a named kernel entry.
This is stronger evidence for a runtime cache defect than for an upload-staging
or GPU memory fault. Existing cache files are preserved; no disk corruption is
claimed.

VERDICT -> No F01 pass. Controlled next arm changes only
`SYCL_CACHE_PERSISTENT=0`, keeping runtime, binary, cases and upload staging fixed.
The same full comparison plan must complete, with exact per-group counts and
both-card/collective post-health, before model admission. This cache controls
compiled device-code persistence, not weights, KV, expert placement or arithmetic.

## Initial model control and remaining gates

The [owned model controller](../llamacpp/flash-next/run_control.py) requires a
passing complete primitive receipt from the identical image and binary.
It snapshots model/runtime identity, checks aliases against the evaluation
registry, monitors loading/runtime memory and kernel faults, owns bounded
startup/requests, and retains the pair lease until container removal is proven.
It also requires final fault audit and post-health for a successful receipt.

Initial static placement hypothesis, not measured residency:

- Layer split48/52: blocks0-23 on GPU0;24-47 and output on GPU1.
- CPU PLE plus routed experts from first/last eight blocks.
- Packed GPU weights about25.39/26.23GiB; CPU weights52.06GiB.
- 8K context, FP16 K/V, C1, no MTP, direct-answer reasoning off.
- Reuse disabled; primary aliashotschmoe-dd and precise research alias retained.

Actual allocated memory includes padding/workspaces/staging and will be logged.
The first six-case repeated endpoint suite is a bounded screen; it cannot
establish broad quality, token-ID authority, benchmark speed, concurrency or
long-context qualification. Those later campaign gates remain required.

Current next work: complete cache-disabled primitive qualification, then launch
the static control only if its admission gates pass. Compare RAM-backed GPU
expert caching after obtaining a coherent same-checkpoint baseline.

## Counted cache-disabled qualification

CONFIG -> Identical runtime image, patched test binary and shape filters,
`SYCL_CACHE_PERSISTENT=0`, debugger off. Corrected source counts36 per card.
This changes only compiled-code persistence; data upload staging is unchanged.

COMMAND -> Full `run_primitives.py` plan, `primitives-v2-cache0/`.

RESULT -> Card0 36/36, card1 36/36, exact per-group counts, no unsupported or
zero-case pass. Both processes exited0 and their containers were removed.
Both-card post-health and compiled ten-iteration P2P0 collective post-health
passed. No kernel GPU-fault signature observed in the complete run.

VERDICT -> The declared primitive scope passes with persistent device-code
cache disabled. Combined with the isolated cache-enabled crash and matching
upstream stack, this supports the bounded runtime-cache workaround. It does not
qualify an enabled persistent cache, broad model quality, speed or service
stability. Full-model static-control screening is now admitted and started.
Its results and owned teardown remain pending.

## First full-model static control

CONFIG -> Same image/server, layer split48/52, CPU PLE and first/last eight
expert blocks, C1/8K, FP16 KV, no MTP, reasoning off, reuse disabled,
persistent device-code cache disabled, upstream warmup explicitly disabled.

COMMAND -> `run_control.py`, `static-control-v1/`, six frozen requests repeated
in the same order, temperature0/seed42, followed by owned shutdown/post-health.

RESULT -> Model loaded and live identity matched both registered aliases.
All12 individual screening checks passed: prose keywords/manual coherent text,
exact code AST, arithmetic, JSON, instructions and Spanish translation. All
requests report cached_tokens0. Five outputs repeated exactly; prose did not.
Both prose answers share their first sentence, then differ in the second:
"your backup strategy" versus "the backup". The strict screen and overall
controller receipt therefore remain failed. This failure is preserved.

Both process shutdown and model-container removal completed normally; both-card
and compiled P2P0 collective post-health passed. No GPU hardware-fault signature
observed. Per-card allocation logs were hidden by verbosity3; packed placement
estimates must not be reported as measured residency.

VERDICT -> Full-model loading and coherent bounded responses demonstrated,
but no frozen deterministic authority or speed qualification. The small prose
variation is not evidence of catastrophic corruption, nor a determinism pass.

The next isolated arm removes `--no-warmup` and changes only its identity label.
Source common.cpp runs a tiny BOS/EOS decode, clears all model memory with
clear(true), synchronizes, resets perf counters and resets sampler RNG. Source
recurrent clear(true) zeroes state buffers and resets cell/head metadata.
This is legitimate runtime initialization, not prompt-cache reuse.

A source-based hypothesis is lazy persistent weight reordering: dense reorder
is enabled only at small batches (n<=8), and MoE also reorders at eligible use.
The first35-token prefill may precede packing; later identical prefills can see
the packed path. This is not yet established cause. Do not call it an autotuner:
none was found in the reviewed implementation. Request/state-reset behavior and
small numerical differences near token ties remain alternative explanations.
The same frozen suite and strict checker stay in place for `static-control-warmup-v2/`.

## Default warmup control

CONFIG -> Same model, source, runtime, static placement, precision, scheduler,
sampling and suite as v1. Remove only `--no-warmup`; append the warmup identity
suffix. Persistent device-code caching stays disabled.

COMMAND -> `run_control.py --warmup on`, `static-control-warmup-v2/`.

RESULT -> All12 checks pass and every repeated output matches exactly,
including prose. Both-card and compiled P2P0 collective post-health pass;
container exit0 and owned removal complete. No GPU fault signature observed.

VERDICT -> Bounded repeated-screen scope passes with default warmup. This
supports initializing the backend before establishing the request oracle, but
does not alone prove lazy weight packing caused the earlier variation. Preserve
the failed no-warmup run. No representative speed or broad-quality claim.
A fresh warmup launch with verbosity5 is now running to verify actual tensor
placement/buffer sizes and compare output hashes across independent starts.

## Fresh start and actual placement

CONFIG -> Same warmup control; verbosity5 adds allocation/placement visibility.
This is a screening and placement run, not a timed performance arm.

COMMAND -> `static-control-warmup-fresh-v3/`, same12 request checks, compare
first-round text hashes with v2, then owned teardown and strict post-health.

RESULT -> All12 checks and within-run repeats pass. All six first-round text
hashes match v2 across the independent start. Shutdown/container removal and
both-card/compiled P2P0 collective post-health pass. No GPU fault signature.
`fresh-screen-comparison.json` records the cross-start comparison.

Actual logged buffers:

| Buffer | SYCL0 MiB | SYCL1 MiB |
| --- | ---: | ---: |
| Model | 25,994.56 | 26,861.90 |
| Main K/V | 96.00 | 96.00 |
| Indexer K/V | 24.00 | 24.00 |
| Recurrent/state | 56.46 | 56.11 |
| Reserved compute | 1,735.70 | 76.48 |

Model buffers are25.39/26.23GiB. They do not include every driver allocation or
temporary. CPU-mapped model ranges total about53.22GiB, including the26.82GiB
PLE range; mapping coverage is not physical resident RAM. Logs confirm the
intended first/last-eight expert overrides to SYCL_Host and layer assignment
0-23 to SYCL0,24-48 to SYCL1. Source fit-auto does not silently change the
explicit n_gpu_layers, split or override settings.

VERDICT -> Repeatable six-case C1 screening baseline across two warmup starts,
with actual two-card/host placement recorded. The bounded text authority is
`llamacpp/flash-next/screen-authority.json`. It is not broad model-quality,
token-ID, occupied8K-context, concurrent-serving or speed qualification. Model
servers were stopped after each run; no daily-driver promotion occurred.

## Next experiment and cache compatibility

Compare this baseline with current llama.cpp's RAM-backed GPU expert cache.
Keep checkpoint, tokenizer, FP16 KV, C1, context, warmup and graph settings fixed.
Preregister cache budget and cold/warm conditions, record hits/misses/refills,
and separately check outputs, logits where prefixes agree, and task quality.
Changing compute placement can change arithmetic: do not label it lossless
against the static authority without passing its declared checks.

Source audit found cache banks marked WEIGHTS, but cached decode tensors are
views whose SYCL extra metadata stays null. Current lazy reorder therefore
returns without repacking them. This is a view-behavior safeguard, not an
explicit mutable-buffer contract. Raw cache uploads do not transform layout or
invalidate reorder state. Test evictions/refills and large-batch transitions;
do not infer full compatibility merely from an I32 GET_ROWS implementation.

Only after candidate coherence and identity pass should profiling and balanced
paired cold-request performance comparisons establish an optimization gain.
