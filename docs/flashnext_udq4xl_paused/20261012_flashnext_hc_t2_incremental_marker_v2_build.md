# Incremental NoDri marker V2 compiler

CONFIG -> Completed markedV1 parent build receipt09292c9a, unchanged SDK52,
compiler/packages/includes/flags/archive inputs. MarkerV2 changes ONLY one
successful early-return hc_record.finish line in verify.cpp. No ABI/header,
Main, projection, composition, device command or arithmetic change.

COMMAND -> Root first prepares the new markedV2 input as already documented;
then hashes its actual compile-recipe.json and executes directly:

    bin/gpu-run --card 0 env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/mnt/vm_8tb/github/b70_ai_things/strata/flash-next python3 strata/flash-next/build_hc_t2_paired_mtp_v4.py --compile-root /mnt/vm_8tb/b70/build/hc-t2-paired-mtp-marked-compile-inputs-v2 --recipe-sha256 ACTUAL_PREPARED_RECIPE_SHA

ROOT ONLY, NoDri2CPU6GiB/card0FD8; no author compiler/GPU/model/archive reads.
Output must be fresh/empty, original parent files remain RO and untouched.

RESULT -> Parent successful compile0/normalEOF/removal receipt is authenticated
by exact SHA. Every recorded parent consumed SDK/header/source/flag/compiler/
package/archive input must still match. New recipe equals parent except marker
plan SHA, relocated overlay/output paths and the SINGLE verify source hash;
actual verify byte removal must recover parent. All other overlay headers and
source hashes, command arrays, compiler image and link arguments stay EXACT.
Three verified objects and TWO verified filtered archives (plus their closed
member rosters) are COPIED from parent. ONLY verify.hc-t2.o is freshly compiled;
then the new paired ELF is freshly LINKED. Receipt explicitly labels copied
objects/archives versus fresh verify compile and fresh link, with actual SHA.
No repeated archive filtering or redundant other-object compilation occurs.

VERDICT -> Source ready pending root actual execution. Same retained-owned
Docker/FD8/normalexit/EOF/removal and original input pre/post closure as full
builder. One targeted CPU control uses actual parent recipe, rejects flag or
header changes, and verifies only one compile command. Original failed65345
model remains FAILED; copying compiler artifacts transfers NO model proof.
