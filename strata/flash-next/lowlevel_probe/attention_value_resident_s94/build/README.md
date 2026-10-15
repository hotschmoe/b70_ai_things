# S94 caller-only incremental main build (SOURCE ONLY, NOT EXECUTED)

CONFIG -> Before main551cca, kernel archive5d465c2d, attention object06eb664c,
attention source013e and narrow header87af remain fixed. Only prefill caller
source changes b6f7 ->2a1d. Source/model review and root application required.

COMMAND -> ROOT ONLY after exact source GO; no apply/run/chown/retry here:
  bash /mnt/vm_8tb/b70/results/flashnext_ista_gsq_rco_iq3xxs_strata_cachefix_20261014/attn-value-resident-s94-build/build.sh
Fresh build/ created only at actual execution; absent during preparation.
Reuse S93 e41/noDRI/networknone/UID1000/cpu-run/MAX_JOBS4/diskTMP/precise
CMake recipe. Setvars BEFORE strict. Actual Ninja dryplan must contain ONE
prefill.cpp compile, ONE libstrata_prefill.a archive and ONE main AOT link.
Any attention TU, kernel archive, other host TU, configure or broader rebuild
fails before compilation. Attention source/header and all unrelated source/
core/engine/spec/native/HC/QSA/generate/verify/CPU inputs hash-check pre/post.
Before551cca main/prefill archive/object hashes are inherited from S93's
actual closure and rechecked inside CPUlease, not rehashed during prep.

RESULT -> Scripts syntax/ASCII only, no build. Runtime writable UID1000
preflight checks planned targets/directories, .ninja_log/.ninja_deps. Snapshot
ONLY prefill object plus small Ninja logs/deps; no wholearchive/oldmain copy.
Old main remains frozen in S93. CMakecache/build.ninja fixed, source hashes
before/after. Output once to build/strata-attn-value-resident-s94.

VERDICT -> Root audits actual new main metadata/registered-name ownership
and selected full image hashes AFTER terminal build, before any GPU. Reuse
S93 ISA/resource facts only if corresponding entire selected images match;
unchanged object/archive alone is not same-new-main-GPU-binary proof. No old
36/CAP rerun or new primitive tests. Serving only under separate root GO,
same new executable flag0/1, marker/work/coherence/cleanup/health gates.
