S107 Q6 A4 main incremental build preparation, 2026-10-15

CONFIG -> Derivative of frozen paired-whole-graph-s106-build recipes.
Source-only preparation: no shell build/controller execution, Ninja dry
run, source mutation, Docker launch, GPU or large artifact hash was done.
Small recipe/pin files only were read. Native MMVQ source after root's
reviewed apply must be ae47a632d0cb3a0fc5a992c335bdd1145d4bbb9210c5b481dd45e53cc3835252.
Original c1cdeabe... source is recorded by q6-a4-main-source-prep/source-proof.json.
S106 MTP whole-pair source overlay remains exact; serving must set graph
flag0 in both arms. Build preparation does not select serving flags.

COMMAND -> ROOT ONLY after full script read and source/model GO, and
when heavy CPU lease is available: bash <this directory>/build.sh.
The existing wrapper uses bin/cpu-run with lock timeout0 (no parked
waiter). Same e41 image, DPC++2026.1, O3/precise, SG32, per-kernel split,
BMG-G31 AOT, jobs4, UID1000, cpus4, networknone, noDRI, disk TMP/cache,
SYCL persistent cacheOFF. No configure/cache/flags change or auto repair.
setvars precedes strict shell mode as in working S106 recipe.

RESULT -> Pending actual root execution. Expected live pre-build main is
S106 e3ca8ac25c1f1e6042792906be3e7faa55e14a1cddaeece176f15c52f6b071fd.
Expected actual Ninja dry plan is exactly one native_mmvq.dp.cpp TU,
libstrata_kernels.a archive, and one strata link. The plan parser rejects
other sources/objects/archives, regeneration and unexpected compile/link
flags before compiling. Pre/post source and unchanged-artifact SHA
checks retain engine/prefill/core/spec/CPU/ggml, IQ/attention/fused/verify
objects and the CMakeCachefee542.../build.ninja89cb... inputs. Expected
changed artifact hashes are native MMVQ object, kernel archive and main.
A successful build freezes a NEW build/strata-q6-a4-s107 plus hashes.
Frozen S106 main remains at
paired-whole-graph-s106-build/build/strata-paired-whole-graph-s106;
its external result tree is never mounted writable or modified here.
No claim about selected device images/ISA or serving follows from build
success: original/CENTER/new-candidate selected-image audit remains due.

VERDICT -> FROZEN_PREPARATION_ONLY. Root owns one actual build after
consultation review and CPU availability. No auto retry, application,
ownership fix, serving, decoder or sweep. from-s106.diff records the
complete derivative. Bash syntax and embedded Python syntax were checked
without executing the build.
