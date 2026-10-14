# Direct55-to56 incremental configuration preservation V3

CONFIG -> Same exact current55 parent/56 body-only delta/compiler image/ABI,
isolated copies, one changed object,123 unchanged objects and8 new ELF links.
V1/V2 and both failed trees remain unchanged.

COMMAND -> Exact root argv is in
strata/flash-next/source56-incremental-from55-build-recipe-v3.json.
V3 retains the original cache/Ninja/configuration and omits redundant CMake
reconfiguration. Parent/child CMake flags, CMake sources, headers, compiler,
packages and full dependency bytes are already admitted as identical. Actual
Ninja dryrun still must schedule exactly one compile, one archive and8 links.
Any scheduled CMake regeneration is rejected. The generated-header equality
gate remains unchanged. All8 ELFs are deleted before relink as before.

V3 additionally checks copied archive hashes, the7 unchanged archive hashes
after build, the owning archive replacement and unchanged cache/build.ninja/
compile_commands/compiler-version/package metadata after execution.
Receipt records CMake_reconfiguration_performed=false and successful unchanged
configuration provenance; it still declares incremental object reuse explicitly.

RESULT -> Actual root V2 session10472 failed19s before compilation. The strict
header gate rejected /build/ggml/src/ggml-version.h: GGML_COMMIT changed from
3cf0325 to3cf0325-dirty during redundant reconfiguration. Dryrun consequently
scheduled12 commands, including unwanted ggml.c compilation/archive update.
The ggml CMake source calls git diff-index --quiet HEAD to set its dirty suffix.
Actual copied Git index has stored inode18600889 for ggml/CMakeLists.txt,
but that copied file's inode is18685551. Original parent stored/current inode
is18600889. Complete source/dependency byte inventories matched before copy;
this is stale Git index stat metadata after creating new owned inodes.
Avoiding unnecessary reconfiguration preserves the original generated header;
its byte hash is never ignored or replaced to bypass dependency admission.

One targeted CPU check constructs the actual V3 script, verifies real newlines,
bash syntax, inherited oneAPI startup skip, absent reconfigure and present strict
Ninja dryrun. No compiler, Docker, GPU or model payload was executed by author.
Runtime incremental reuse/compilation has not yet passed. No speed claim.

VERDICT -> New strict V3 body-only incremental candidate. If any generator,
compiler, header, dependency, object or archive differs, fail and use full build.
