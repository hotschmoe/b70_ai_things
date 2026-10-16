V2 runtime-wrapper correction - SOURCE ONLY, no execution

ROOT requested this correction after full source/driver review. V1 remains
immutable; every existing payload except leased-run.sh is byte-exact V1.
Probe/build/analyzer/ROOT_RUN/run-container source and their hashes are unchanged.
This is a lifecycle correction, not a second oracle or an actual runtime result.

The attached docker start -a stays on its SAME blocking handle and CPU/GPU
leases until it returns. There is no automatic timeout, signal, stop, kill,
reset or oracle retry. Created/final full CID + NAME + exact image + effective
user are checked. Unknown/live/identity-invalid states write a root-readable
recovery-required.json then HOLD the same leases, with only a sleeping process;
no automatic query/force/removal/health follows in that branch. ROOT must
inspect/decide before releasing a held attempt.

For an authoritative stopped owned container, full native state/log paths are
saved BEFORE normal docker wait and normal rm of that exact CID. Wait/rm exit
codes and raw outputs are retained, wait/native agreement is required, then
full CID and NAME absence is checked. No removal retry or forced cleanup.
Native5/API/strict failure (or IO/unknown diagnostics) receives normal stopped
removal only, then a diagnostic-review receipt and exit2, NO posthealth. There
is no numerical-failure whitelist or automatic recovery classification.
Posthealth runs ONLY after actual native0/dockerstart0, notOOM/dead/error, both
leaf0 statuses, mode closures and overall exact/DOT analyzer PASS, plus completed
normal wait/removal/fullCID+NAME absence. Source counts and no-speed scope stay
unchanged. The old README's180s cap describes historical V1; V2 has no automatic
workload timeout. A hang retains the same handle/leases for ROOT.

Checks performed: bash -n and AST parse of embedded Python only. No container,
compiler, GPU, health, audit, model call or shell-wrapper execution. Actual
Opus/Fable probe-source review is ROOT's separate ongoing consultation; no new
consultation result is claimed here.
