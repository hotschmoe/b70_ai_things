# Integrated28+25/26/27 immutable SDK build recipe

CONFIG -> pristine external Strata fb58e0, unchanged image39992d70 and complete
GGML3cf03257 source. The new plan derives actual28 baseline recipe d496d463, then
applies frozen25, final26 and frozen27 after its checked patch order. Existing
plans, builds, binaries and results are preserved. No weight/artifact/tokenizer
file is changed or read by the CPU reconstruction.

COMMAND -> CPU only:

    python3 strata/flash-next/test_integrated_batch_prefix_source_cpu_v1.py

The parent-owned full build recipe, with automatic pair lease and no exposed
container devices, is:

    python3 strata/flash-next/build_native_hc_engine.py \
      --plan strata/flash-next/integrated-batch-public-prefix-engine-build-plan-v1.json \
      --jobs 4

An exact clean complete GGML3cf03257 checkout can be supplied with --ggml-source.
Otherwise the builder fetches that exact dependency pin. Headers alone are
insufficient. Build, source and dependency directories are freshly isolated.
The recipe does not run native probes or inference after compilation.

RESULT -> CPU reconstruction passes all28 patch checks against pristinefb58 and
matches baseline28's51 final files before the new increments. The integrated
ledger covers60 final files,24 consumed header payloads and allsix runtimePython
sources. Every existing pristine file has a separate pristine ledger; patch-created
headers never enter it. All new source hashes differ from build receipts: the
reconstruction receipt is source-only and cannot be mistaken for a compiled result.

VERDICT -> source plan READY for parent full compiler review. Integrated source,
SDK/ABI, weights/source390, model/math/cache/concurrency and health/teardown
qualification remain open. Existing24/28 source/runtime/native8/51 proofs do not
transfer to this integrated generation.

All eight targets rebuild: strata, native_expert_parity,
conversation_snapshot_test, iq_multi_parity, native_grouped_parity,
native_multi_parity, shared_expert_parity and verify_parity. They link newly built
core/prefill/CPU/kernel objects, including optional23 producer callback ABI changes.
Every external upload/ownership/mirror/numerical oracle must also rebuild and bind
this generation's final source/library/binary receipts. Old standalone executables
cannot qualify new library interfaces.

Source flags remain defaultOFF: STRATA_SYCL_NATIVE_HC runtime selection,
STRATA_BATCH_FULL_STATE_CHAIN, STRATA_BATCH_PUBLIC_PREFIX and
STRATA_BATCH_FIDELITY_DIAG. The CMake native-HC primitive option is deliberatelyON
in this reviewed build. BaselineOFF behavior still needs a matched new-generation
control. Full-chain/public cache activation requires exact native2/4/6/group1,
no-MTP/non-pipelined static/no-borrow, resident fixedFP16 KV, original context,
source slot/queue/context identities and real count/byte/physical-RAM bounds.
No guard or requested capacity is removed. Pilot2048/prefill64 does not qualify
production8192, long prefill or fairness.

25 changes complete cache-chain ownership and transfer ordering, including active
immutable checkpoint donors, n-1 prompt leaves and idle slot parking. 27 changes
explicit fresh/pin cache/reset semantics: fresh lookup ceiling0 and actual allmain
stage state/metadata reset, pin exact0..prompt-1 and actual evaluated state atN.
Neither changes weights or request tokens/templates. 26 changes diagnostic
producers/lifecycle only: fresh admitting identity/ARMloss safety and an actual
post-migration solo third raw sample. The six-request three-phase raw quota is
53268480 bytes under64MiB. Its source CPU proofs are not actual GPU replay proofs.

The runtimePython identity closure is exactly six files from
serve.artifact_identity.PYTHON_SOURCES. Five preserve actual28 bytes; server.py
changes only for27 public request fields/resolution and retains28 package-safe
imports. The new plan records the final hashes of allsix, including unchanged
frontend.py and gguf_reader.py. Serving manifests must bind actual final binaries,
arguments, environment, tokenizer/model export and these exact source bytes.

Required actual gates include fresh source390 ownership/account/logical-free
checks, complete four-shard identity and known page sentinel before/after model
runs, new-generation baselineOFF/activated profile controls, exactINFO capacity,
per-request stage/row/epoch/full48/head coverage, private-session isolation,
same-process matched fresh-vs-hit state/output/evaluation counters, root/pin/leaf
and independent conversation/turn/divergence/eviction/cancel/migration cases.
Ongoing decode beside long prefill at1/2/4/6 requires P50/P95 stream fairness,
coherence, throughput and owning cleanup with strict per-card/compiled post-health.
Successful compilation or complete diagnostic metadata does not satisfy them.
