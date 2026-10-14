# PTI3 real-model capture: census fixed, metadata bound reached

CONFIG -> Native43 mode0, LPdisabled, four matched GEN64 requests; H36 and
decode timing in both arms, PTI only ON. Same selected model, ctx2048,
prefill64, PLE65536, MTP0/cacheOFF. Actor3 source plan0e7d6591; existing
normal43 and profiler intake prerequisites unchanged. Sole leased job61440.

COMMAND -> Actual run_source43_pti_latency_v3.py, terminal exit1 after1595s.
Root: `/mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f17-source37-20261010/source43-pti-latency-v3-run-v1`.
Report SHA256:
`943ca683e602217c876b1d6828800747c0c086c1e6d4f1d91f83394400e8a23d`.

RESULT -> OFF completed four requests and H36 session, normal retirement and
post-health/integrity. ON reached READY; the strict Btrfs mapped-library
census passed on the actual model. Census SHA256:
`f20f56500d118886a39e3ebf5af1120463b4eb7636b4af75449556d072ab1dd2`.
ON completed request1; during request2 prefill it emitted
`[B70_CLOCK_EXPORT_ERROR] metadata output bound exceeded`.
Actual tool source contains the512MiB bound. Clock file536870754bytes;
Native terminal exit86, OOMKilledfalse and forced_cleanupfalse; the protocol
records Unexpected engine EOF. Explicit successful finalization was not reached.
bounded tail shows complete sequence778468, host_clock_bridge/site104,
following repeated global_timestamp_query_bracket/query_site4 records.
These tail records do not establish the complete file's record distribution.
Chrome file was492630175bytes at the bounded observation. Saved diagnostic:
F17/source43-pti-v3-metadata-cap-observation-v1.json,
SHA2562dcd45bb51dc94bacd247d76c0f9a6635ad1039f06ebe5e1184f70af0744ff1e.

Both ON post-health and aggregate post-health passed per-card probes and
compiled pair collective. Both corresponding fresh four-shard integrity
scans passed; parent preserves page checks and artifact roster. No reset was
needed. The capture parent correctly remains failed.

VERDICT -> Actual mapped-library association now works. Complete model PTI
capture remains unqualified; do not run the success-only analyzer or infer
physical idle/bandwidth from this incomplete trace. Audit event volume and
prepare an explicitly bounded request/replay-window successor before retry.
Preserve the failed raw files and512MiB enforcement. Original full-model
fidelity, cache/concurrency and clean interactive latency/shelf remain open.
