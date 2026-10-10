# Unchanged CPU screen swap attribution observer

CONFIG

Separate passive observer for exact e6337031/115-file CPU continuation screen.
The failed 95049 screen remains immutable: host SwapFree fell 10592256 bytes,
while all saved owned VmSwap/cgroup swap samples were zero. That does not prove
which process caused the swap movement. No n_ubatch, cap, swap guard, prompt,
seed1234, natural64, all16-before-selection or inference setting is changed.

This source-only observer reads kernel memory metadata, not model/pack bytes.
Root alone runs it under inherited pair exclusion lease FD8/9, without device
grants. Idle30 has no observer model execution and rejects every live container
with the exact frozen CPU-screen ownership label. Root must schedule the whole
idle window without other model work; this cannot prove unrelated CPU processes
are model-free merely by inspecting their names.

COMMAND

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=strata/flash-next python3 -m unittest test_cpu_swap_attribution_cpu_v1

Root future idle (NEW absolute output path):

bin/gpu-run python3 strata/flash-next/observe_cpu_swap_attribution_v1.py idle --output NEW_IDLE

Root future runtime observer inherits the SAME already-held exclusion lease as
screen execution, after successful idle and once exact screen producer PID is
known. Do not acquire a second independent conflicting lease:

python3 strata/flash-next/observe_cpu_swap_attribution_v1.py observe --screen-root NEW_SCREEN_ROOT --producer-pid ACTUAL_PID --idle-receipt NEW_IDLE/report.json --seconds 1800 --output NEW_OBSERVATIONS

Run unchanged screen producer from e633 separately under that owned lease.
Start observer before model launch. SIGINT/SIGTERM requests final observer
export only; it never stops or signals the screen/container. Natural guard
failure and owned stop remain exclusively in the frozen screen producer.
Read-only consumer: observe_cpu_swap_attribution_v1.finalized_binding(ROOT).

RESULT

10 synthetic CPU controls pass: host/per-PID deltas, PID reuse identity,
missing-kernel-field honesty, bounds, no environ/cmdline dependency, owned
smaps/cgroup scopes, foreign owner/image/device/cap refusal, immutable output,
idle live-container refusal and read-only byte mutation. No actual idle,
container inspection, host proc census, inference or model read by agent.

One-second samples retain pswpin/pswpout counters and deltas (kernel pages,
recorded host page size), MemAvailable/SwapFree, and per-process VmSwap deltas
keyed by PID plus start ticks. Only comm/status/stat are read for other PIDs;
no environ, cmdline or descriptor targets are read. Process disappearance is
recorded as incomplete census availability, not attributed to a container.
Kernel processes lacking VmSwap and permission failures are explicit coverage
errors; unavailable values are not silently replaced by zero.

Live owned CPU container samples require exact producer/case names, original
command file, reconstructed frozen e633 CPU image/entrypoint/cwd/user/resources,
no device grants, and exact label. They retain smaps_rollup RSS/PSS/shared/
private/anonymous/Swap/SwapPss and cgroup memory.stat/current/peak/events/swap.
Narrow saved inspections omit environment variables. cgroup swap.max stays0
and original 116GiB cap stays exact. Observation records nonzero memory swap
rather than changing or superseding the original fatal memory guard.

Bounds: at most4096 enumerated PIDs,256 nonzero-swap/delta rows,7201 immutable
sample files and7200 seconds. Missing/excess data fails observer admission.
Duration expiry or an explicit root stop does not prove whole16-case coverage.
Source and raw sample bytes/chronology are rejoined by the read-only consumer.
A passed observer means this diagnostic metadata closed, not that the screen
passed, no global swap occurred or a causal memory attribution was established.

VERDICT

Source/CPU ready for independent review. Actual observation is unexecuted.
Collect unchanged-recipe data first; a lower-peak inference recipe is not
justified by the preserved global-swap event alone. No math/quality/cache/
concurrency/latency qualification is transferred.
