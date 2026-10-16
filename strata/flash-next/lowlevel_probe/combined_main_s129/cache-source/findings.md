# Next required main build: offline AOT cache adoption (source only)

CONFIG -> Current src-ep/build-sycl-ep, actual target strata, e41/DPC++2026.1,
BMG-G31/SG32/per-kernel/O3/precise, jobs4 unchanged. S127 touches mtp.cpp AND
mtp.hpp; a separate Q5 integration would touch only native_mmvq.dp.cpp.
No live source/build edit, configure, compile, GPU, model read or large hash.

COMMAND -> Read pinned CMake/Ninja files and S119 exact successful command.
Run bounded Ninja -n queries under cpu-run in e41 with /work READ ONLY.
External candidate build.ninja copies change only the existing backend
argument string. Forced phony source inputs simulate pending edits without
changing live file bytes/mtimes. All seven observed live-input stats unchanged.
The first colloquial main-target query failed unknown-target and is retained;
then the observed exact strata target was queried. No configure/build ran.

RESULT -> Actual generated CMake separates frontend _sycl_flags (lines39/78/82)
from link-only _sycl_link (48-49/81/85). CMAKE_CXX_FLAGS is empty, Release
-O3 -DNDEBUG; EXE_LINKER_FLAGS remains -fsycl-max-parallel-link-jobs=4.
AOT backend currently is ONE argument:

    -device bmg-g31 -options -cl-fp32-correctly-rounded-divide-sqrt

Add only this suffix in that same backend argument:

    -allow_caching -cache_dir /out/offline-cache

cmake-cache-option.patch is a concrete OPTIONAL/cache-empty-by-default source
proposal, outside live source. It sets STRATA_SYCL_AOT_CACHE_DIR and appends
only the cache suffix in the AOT link branch. Empty option reproduces the
old backend string exactly. _sycl_flags, definitions, includes and all FP/SDK/
architecture choices stay byte-identical in the external Ninja projection.
Do not add this to CMAKE_CXX_FLAGS or add_compile_options. Do not duplicate
-Xsycl-target-backend: current CMake comments explicitly warn about dropping
repeated backend selectors. Environment alone is not the demonstrated enable
mechanism; S119 passed BOTH the explicit CLI flag and enabled persistent env.

Exact dry-plan result, each Ninja payload0.025-0.031s (container9s):

| Read-only plan targeting strata | Scheduled actions |
| --- | --- |
| Current actual build | 0 |
| Cache-link suffix only | 1 final strata link |
| Cache + S127 cpp/header simulation | 6 |

Those six are mtp.cpp, prefill.cpp, generate.cpp compiles; engine and prefill
archives; one final strata AOT/host link. Header invalidation means this is
NOT a three-action build. Generated file has46 executable link-edge suffixes,
but targeting strata does not build/relink all parity executables. Do not use
all as target. Expected combined S127 + one native-MMVQ TU patch is EIGHT:
the six above plus that TU compile and kernels archive; final link remains one.
This last8 count is a source dependency inference, not an executed fourth dry
plan. A candidate copy for that exact force simulation is retained if needed.
Actual post-configure/source-apply Ninja plan remains the final scope check.

S119 is bounded evidence: same input had hits+1/misses0, changed IR hits0/
misses+1, all3 outputs exact against their respective retained references.
Cold0.141324s, warm0.021773s, changed0.150170s on ONE tiny module are not a
main-link forecast. Driver -### forwarding proof (S117) already passes these
backend CLI controls to ocloc without changing the FP options token. First
otherwise-required main build may mostly populate cold entries; modified IR,
module identity/internal options/compiler/device can miss. No normalization,
cache fabrication, standalone warm-up/relink or extra GPU test is selected.

VERDICT -> Use this link-only option on the NEXT OTHERWISE-REQUIRED main
build, combining independently gated S127/Q5 source if root selects both,
then test separate flag arms on the same frozen binary. One CMake regeneration
would be needed to materialize this source option; it has NOT been run here.
The planned option is:

    -DSTRATA_SYCL_AOT_CACHE_DIR=/out/offline-cache

Future build wrapper precreates /mnt/vm_8tb/b70/cache/strata-aot-bmg-g31 as
UID1000 and verifies ownership/writability before ANY ocloc CLI. Parent is
currently UID1000/writable; proposed child is ABSENT now, not falsely claimed
pre-existing. Bind that stable host directory at /out/offline-cache on every
future build. Set NEO_CACHE_PERSISTENT=1, NEO_CACHE_DIR=/out/offline-cache,
NEO_CACHE_STATS=1 for actual adoption counters, retain SYCL_CACHE_PERSISTENT=0.
Preserve its warmed contents across build result directories. Do NOT mutate
S119's archived experiment cache/stats. Disk TMP remains per-build on disk.

Record actual cache ownership/path/counters, selected source/SDK/options and
artifact/link identities on that required build. Stats add overhead; neither
a cache hit nor this source projection waives compiled/control/numerical or
health gates. No whole-main time saving is claimed.
