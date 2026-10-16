# Exact-input offline AOT cache reuse, 2026-10-16

S119 establishes a genuine offline cache hit for one repeated sampler IR input and a miss for its changed input. All three generated native images are byte-exact to their respective retained main references. It does not measure a full-build improvement.

CONFIG -> Corrected, versioned setup preserves [S117's failed setup record](20261016_strata_offline_aot_cache.md). Same12064B/S107 and12060B/S114 sampler SPIR-V, BMG-G31 and correct-rounded divide/sqrt options; unchanged byte/counter gates. Precreated owned /out/offline-cache, explicit /out cwd, NEO_CACHE_PERSISTENT=1 and matching environment/CLI/statistics paths. e41/UID1000/fourCPU/noDRI/network-none/diskTMP; SYCL runtime cacheOFF.

COMMAND -> Root authorized one bin/cpu-run cold/warm/changed sequence, handle16905. Wrapper9s/exit0/tee0; no source compile, full main relink, GPU or model read. Actual preflight recorded cwd, UID/GID, cache ownership/writability/existence and environment/CLI path agreement before any ocloc CLI. No flag/tolerance relaxation or input normalization.

RESULT -> Supported cache counters establish reuse rather than elapsed time alone:

| Call | Elapsed s | Cumulative hits/misses | Delta hits/misses |
| --- | ---: | ---: | ---: |
| Cold S107 | 0.141323937 | 0/1 | 0/1 |
| Same S107 | 0.021773478 | 1/1 | 1/0 |
| Changed S114 | 0.150169621 | 1/2 | 0/1 |

For EVERY call, whole native image, complete text, metadata, options, constant data and relocations match the appropriate selected main reference exactly. The changed real SPIR-V causes a cache miss in this case even though native computational prefixes match. Runtime assertions verified frozen input hashes before calls and equal hashes afterward; no separate raw digest files were written. The curated integrity record labels this derivation and performs no new payload hashing.

VERDICT -> Accept this bounded offline-cache mechanism, not a whole-main speed result. Combined S116/S119 actual postreviews by claude-opus-5-5(52.551092s) and claude-fable-5-1(48.834986s), both exit0/root full-read, support a pinned persistent offline cache on the NEXT OTHERWISE-REQUIRED build with source/SDK/options/artifact/cache/link receipts. No standalone146s relink, extra warm run, numerical campaign or GPU test is selected. One warm sample is not a timing population; stats collection can add overhead, and changed IR/compiler/hardware/options may invalidate entries. IR linking, lowering, wrapping and host-link costs remain outside this tiny observation. No saving forecast against S114's146.468s AOT link follows.

[Exact scripts, three calls/results/stats, setup/inverse proof, path/environment/exit evidence, filtered postdecision and external identities](../strata/flash-next/lowlevel_probe/offline_aot_cache_s119/README.md). No binaries, SPIR-V or cache files are curated or newly rehashed. S117 remains an independent negative setup stage.
