# S117 offline AOT cache: negative setup record

CONFIG -> One CPU-only attempt to test explicit offline ocloc caching, aimed at avoiding repeated unchanged AOT work after S114's146.468s final link. Pinned e41/UID1000/fourCPU/noDRI/network-none/diskTMP; SYCL runtime cache OFF. Only two tiny retained sampler SPIR-V inputs were prepared. Input bytes remain external; this subtree contains their hashes/source references, not binaries.

COMMAND -> Root invoked frozen run.sh once, handle48558. The script verified all28 prep checksums before acquiring bin/cpu-run. Actual -### forwarding previously passed; ocloc help/statistics queries were not compiler-result tests. run-cache.py/run.sh here are exact historical source copies, not a runnable standalone package or authorization to repeat.

RESULT -> CPU wrapper9s/exit1, tee0. Only cold ocloc was attempted: exit225 in0.003115452s, reporting "Failed to create directory 'ocloc_cache'." No native compiler output, compiled comparison, cache-hit statistics, warm or changed-input call was produced. Intended absolute cache/output directories were UID1000-owned/writable and empty. Actual child cwd and filesystem errno were unobserved, so a specific permission cause is not proved.

Version-matched source shows both flags are parsed before initialization. Compile initialization uses the default configuration directory unless it is empty; therefore a nonempty relative default can override CLI -cache_dir. Reordering flags alone is not a supported correction. The successful empty-cache statistics query uses a different override path and did not qualify compilation setup.

VERDICT -> HOLD, no cache or speed conclusion and no automatic retry. A later root-selected versioned setup could align a precreated absolute NEO_CACHE_DIR with CLI/cache-stat paths and an explicit owned writable cwd; neither cwd alone nor help success proves correct cache reuse. Byte-comparison/counter gates remain unchanged. No main relink, GPU, source/ABI change or new numerical campaign follows.

See [curated report](../../../../docs/20261016_strata_offline_aot_cache.md), [exact-copy index](source-copy-index.json), [external inputs and version-matched sources](references.json), and [actual HOLD](HOLD.json). Original failed evidence and frozen prep remain external and immutable. No SPIR-V/main/archive/model payloads were copied.
