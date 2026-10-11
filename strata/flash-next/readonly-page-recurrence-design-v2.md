# Prefaulted page observation successor

CONFIG -> V1 is preserved and held. V2 initializes all4096 bytes of anonymous
direct-read destination memory to0xa5 before preadv, matching the validated
root direct-page auditor. Only anonymous destination storage is written.

COMMAND -> Use probe_readonly_page_recurrence_v2.py with the V1 arguments. Root
must review before model use. The optional single-page DONTNEED occurs only
after initial buffered/direct preservation and explicit expected-direct hash
validation. A postaction buffered baseline is saved before the watcher starts.

RESULT -> Restoration from the initial buffered page to the postaction page
is recorded separately in postaction-baseline.json. The first-transition files
now preserve the first later change relative to that baseline, so restoration
does not consume the subsequent recurrence capture. Initial/phase/watcher data,
stat5, child EOF/timeouts, bounded quotas and false identity/cause scopes remain.

VERDICT -> The watcher is an active read workload. It can change atime and cache
residency; it is not a passive monitor. Neither direct-page agreement nor a
timestamp association proves whole-shard identity or the cause of corruption.
