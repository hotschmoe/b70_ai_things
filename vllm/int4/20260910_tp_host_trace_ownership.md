# Opt-in TP startup trace ownership, 2026-09-10

CONFIG -> Root workers created private trace directories/files that the host
reviewer could not read. The active run was handled separately by the parent;
this change applies only to future snapshotted hooks.

COMMAND -> With --tp-host-trace, kv_campaign_server forwards its host UID/GID
as B70_TP_HOST_TRACE_UID/GID. On first trace open, the hook validates both IDs,
chowns only the configured trace directory and fchowns the newly opened file.
The directory/file creation modes remain0700/0600. An ownership failure closes
the unpublished descriptor and fails visibly. Without ownership environment,
the hook retains its old behavior. Tracing-off server argv remain byte-identical.

RESULT ->13 CPU tests pass: four ownership tests, four argv/integration tests
and five existing boundary/source-profile tests. An additional no-device,
no-network1CPU/1GiB container running as root creates a trace owned by host
UID1000/GID1000. Host read succeeds and modes remain0700/0600. The CPU container
exited and was removed. No GPU operation or active snapshot change occurred.

VERDICT -> Host-readable private traces for future opt-in runs. No trace
semantics, source profile, tensor operation, collective wait or default command
change. The200K plan owner was given the updated source hashes before freezing.

Server SHA256: dc5d657190e88ad5bc4e29637f71450f9b30ddd4ad410eb2aed89eb28fd7f4a8
Hook SHA256: e32f74cf4b8c5af8493ac162c0edf711c6c8dfff8c297bb8e02f3fef96c02286
Raw: /mnt/vm_8tb/b70/results/bang_isolation_20260910/tp-host-trace-ownership
