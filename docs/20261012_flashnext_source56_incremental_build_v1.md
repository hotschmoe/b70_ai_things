# Direct source55-to56 incremental NoDri build

CONFIG -> Exact completed current source55 SDK, compiler image39992, oneAPI
2026.1, original ggml revision, Release/CMake flags, all headers and existing ABI.
Source56 differs only in sycl/src/kernels/hc_native_composition.cpp. Parent has
124 actual objects and8 archives. Parent source/ggml/build copies use the same
container /src,/ggml,/build paths, preserving unchanged mtimes and Ninja state.
New copies use separate inodes, no hardlinks; parent cannot be modified.

COMMAND -> Root runs the exact argv in
strata/flash-next/source56-incremental-from55-build-recipe-v1.json.
The compiler lease is card1 only; the Docker container exposes no DRI/devices
and checks /dev/dri absent. A live card0 model does not require a pair lease
for this CPU-only compiler container.

Admission hashes parent receipt/plan, all74 patched parent sources, all8 parent
ELFs, all124 objects, all8 archives, compiler/package/cache/Ninja metadata and
complete source/dependency trees. Child applies only the reviewed56 delta.
It deletes the8 copied executables before relinking. Same-image compiler and
package output must match. CMake regenerates at identical paths/flags; all
actual Ninja commands must remain identical. Dependency records must cover
every reused object; source/ggml/generated header bytes must match. The changed
body may appear only in its one object dependency roster. A dry run must show
exactly one composition compilation, one owning archive update and8 ELF links.
Any unexpected rebuild/header/command/dependency fails before compilation.

After build, all123 reused object hashes must be unchanged, the composition
object must change, and all8 required new ELF files must exist/hash. Original
parent proof, external reviewed source and immutable builder/child plan are
rechecked. Failure remains in a separate failed tree; full builder is fallback.
No archive or prior stack objects are admitted.

The receipt says build_strategy=direct55_to56_isolated_incremental_v1,
fresh_all_source_compilation=false and retains exact parent provenance,
reused object hashes, new object hashes, actual container commands/dependency
preflight, all74 child source hashes and all8 newly relinked ELF hashes.
The existing source56 SDK binder requires current74 source/current8 ELF bytes
and a completed isolated NoDri receipt. It does not require74 recompilations,
so these requirements remain unchanged. Its all_fresh_ABI_binaries field means
the8 newly linked executables; it cannot mean all library objects recompiled.
Model arithmetic/runtime qualification remains separate and pending.

RESULT -> Source/CPU parent inspection passed:124 objects,2460 source files,
3608 ggml files,8 archives. Two tiny synthetic controls passed; changed headers
and extra compilation are rejected. No compiler/Docker/GPU/model payload was
executed by the source author. Root execution is still required.

Actual parent55 Ninja timing recorded composition compilation8.080s; its8
links were1.190..35.033s each (parallel schedule). Full parent build was309s.
Incremental elapsed time is unmeasured; copying/hashing/configuration and
relinking remain. No percentage or runtime speed improvement is claimed.

VERDICT -> Opt-in direct current55->56 incremental candidate with truthful
reuse provenance. It is not an all-source fresh compilation or model proof.
