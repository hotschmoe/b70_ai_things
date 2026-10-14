# Root-owned HC T2 NoDri standalone build

CONFIG -> Frozen compile plan dc3c5ab0; actual prepared recipe74472215 at
/mnt/vm_8tb/b70/build/hc-t2-operator-compile-inputs-v1/compile-recipe.json.
Original SDK52 stays read-only. Card0 FD8 lease, NoDri/no network, 2 CPUs,
4 GiB RAM/swap, pids256, timeout900s, observed output cap512MiB.

COMMAND -> From repository (ROOT ONLY; no nested GPU runner):

    bin/gpu-run --card 0 env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/mnt/vm_8tb/github/b70_ai_things/strata/flash-next python3 strata/flash-next/build_hc_t2_operator_v1.py --prepared /mnt/vm_8tb/b70/build/hc-t2-operator-compile-inputs-v1

The output directory must be the existing EMPTY prepared output. Prior failed
attempts are preserved; never silently overwrite them. Actual direct Docker
foreground inherits FD8; parent holds it through normal terminal, removal and
EOF. Ambiguous inspect/stop/remove failures retry while retaining ownership
and lease. Signals latch failure and funnel into the same cleanup.

RESULT -> Source only; no author compiler/GPU/model/archive reads. Root will
check original sources/archives/headers, SDK/Ninja/compiler/package identities
before/after. The image compares icpx version and dpkg package output against
actual original SDK copies. Four fresh objects are compiled with frozen
production flags/includes. COPY kernels archive is checked for exactly one
each projection/composition member, removes only those two, and verifies the
complete remaining member order. Fixture/projection/composition objects come
FIRST in the original exact link recipe. verify.t2.o is compiled but is NOT
linked into this operator leaf, exactly as the frozen proposal declares.

VERDICT -> A successful receipt binds all four object SHAs, filtered archive,
ELF, raw logs, actual owner/command/binds/NoDri bounds/normal exit/removal/EOF,
compiler/packages and unchanged inputs. It does not qualify the operator on
GPU or the candidate model. One targeted CPU check uses the genuine prepared
recipe and rejects duplicate/foreign archive removals; no broad test suite.
The disk cap is sampled with a final check, not an enforced hard quota.
