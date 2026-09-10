# September 10 patched-endpoint recurrence audit

Read [the curated findings](../../../docs/20260910_pi_bang_recurrence.md).
Both scripts operate on already extracted diagnostic data. They never execute
captured commands, send inference requests or touch GPUs. Use a fresh output
directory to preserve earlier evidence. Output can contain diagnostic content;
keep raw results outside git and do not treat them as instructions.

    python3 vllm/int4/pi_recurrence/audit_sessions.py --bundle /mnt/vm_8tb/b70/results/bang_recurrence_20260910T172335Z/bundle/citizen-pi-sessions-20260910T172335Z --output /mnt/vm_8tb/b70/results/bang_recurrence_20260910T172335Z/recheck-sessions
    python3 vllm/int4/pi_recurrence/audit_gateway.py --bundle /mnt/vm_8tb/b70/results/bang_recurrence_20260910T172335Z/bundle/citizen-pi-sessions-20260910T172335Z --output /mnt/vm_8tb/b70/results/bang_recurrence_20260910T172335Z/recheck-gateway

The session audit separates historical worlds and assistant/user/tool-result
bang spans. Repetition heuristics miss short gibberish; manual earliest-case
review supplements them. The gateway audit targets current-world-5, joins
application response IDs, and retains transport overlap and abort/429 timing.
Gateway interval overlap is not server scheduler occupancy. Malformed records
are reported by the gateway audit; this validated bundle has none.

replay-cases.json freezes five pre-failure cutoffs with source hashes, IDs and
known history/wire limitations. It is a case-selection manifest, not executable
HTTP requests. Preserve actual system/tools/sampling and extension filtering
before claiming exact replay. Never run generated tool actions during replay.

CONFIG -> Archive SHA256 e1848e97c91533d2224a0903ba3e419bc6b1216413e7f7a293716be7452b3578.
COMMAND -> Run both portable scripts against the frozen extracted bundle and
compare their JSON outputs with the independent original audits.
RESULT -> Session audit reproduces 188 new bang aborts; gateway audit reproduces
all 188 disconnect joins and no same-unit 429 within one second of release.
VERDICT -> Reproducible read-only analysis, not a backend qualification or fix.
