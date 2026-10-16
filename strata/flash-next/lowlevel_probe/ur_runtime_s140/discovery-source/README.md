# Default UR adapter discovery using existing S135 binary

SOURCE ONLY. ROOT executes once, card0, after reading exact files. No build,
forced V1/V2, driver-inorder knob change, profiler or performance measurement.
Original S135 files/build are immutable; direct mounts reuse their exact binary,
source and original per-mode strict/DOT analyzer. Mode0 only, 3A/B/A phases,
8depth each, same integer kernels/buffers/guards and original data validation.

Root exports B70_DISCOVERY_REUSE_BOOT_ID with the known SAME boot as S139 ON
and B70_DISCOVERY_NO_INTERVENING_GPU=1, then `bash ROOT_RUN.sh`. This is explicit
root attestation, not a claim that the historical posthealth JSON recorded a
boot ID. Script checks current kernel7.1, exact root-supplied boot, prior S139
normalteardown plus BOTH-card/collective posthealth exits0, records identities.
No duplicate prehealth; exact one card0 posthealth remains after normal closure.

The ONLY behavior change in the GPU executable invocation is discovery logging:
SYCL_UR_TRACE=1; UR_LOG_LOADER='level:info;output:stderr';
UR_LOG_LEVEL_ZERO='level:info;output:stderr'. All three adapter/driver-list
selection knobs are explicitly UNSET after setvars. Source official logger
syntax: https://oneapi-src.github.io/unified-runtime/core/INTRO.html .
The installed pinned e41 adapters contain EXACT `Skipping L0 V1 adapter: {}` /
`Skipping L0 V2 adapter: {}` strings and default-device-IP reason, retained at
../queue-runtime-readiness-source/installed-discovery-messages.txt.

Proof requires exactly ONE default-IP adapter rejection, surviving adapter's
loader `loaded adapter` evidence, no driver/loader skiplist rejection, plus
successful original checksum-bound executable that explicitly asserts both
actual queues are LevelZero. A list of loaded libraries alone fails the gate.
If logger output cannot prove selection, HOLD without guessing or retries.
Actual selection remains UNKNOWN until this run completes.

S135 queues explicitly carry ONLY in_order, not enable_profiling. This probe
cannot establish mainqueueprofiling, driver-list toggling or a speed effect.
Default mainqueue construction source evidence is separate. No timer, GPU idle,
traffic, reset-cause or overlap interpretation is permitted.

Lifecycle is inherited unchanged: same blocking docker-start handle, exactowned
CID/name/image/user, no timeout/stop/kill/reset/retry, unknown/live holds same
leases. Confirmed stopped diagnostics precede normalwait/rm and CID/nameabsence.
Native5/analyzer/loggerHOLD yields no automatic posthealth; root reviews saved
state. Native0 + strict/DOT + discoveryPASS + normalownedclosure precede card0
posthealth. No new runtime environment is applied to Strata/main serving.

Dev loop: shell syntax + Python AST only. No CPU builds, ELF/ISA work or GPU by
preparing agent. At most ONE finite discovery process; mode1 is intentionally
omitted because S135 already proved its API dependency fixture.
