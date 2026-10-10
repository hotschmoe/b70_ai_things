# CPU swap observer V7 exit-transition proposal

CONFIG -> Same frozen finite screen generation3, all16 cases, seed1234,
natural generation cap and original memory/swap guards. Only passive observer
sampling failure classification changes. All original V6 sources and actual
failed wrapper remain unchanged. No prior observer success is invented.

COMMAND -> PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=strata/flash-next python3 -m unittest test_cpu_swap_attribution_cpu_v7

RESULT -> 19 tiny CPU controls pass using temporary synthetic proc/cgroup
files and mocked Docker inspection. No actual proc observation, Docker,
model/pack payload or GPU access occurred. V6 actual report recorded a missing
kernel-byte error during first-case completion. Its missing path was not saved;
status versus smaps attribution and a causal explanation remain unobserved.
The independent screen success does not repair that observer failure.

VERDICT -> V7 is source/CPU preparation for review. Missing owned-PID memory
fields are unavailable, never zero. A second exact recipe/owner/incarnation
inspection must show exact name absence, terminal state, PID transition, or
an explicitly observed same-PID kernel exit state Z/X. A still-live same PID
with missing memory remains failure. Foreign/recreated containers fail.
Saved unavailable rows have exact fields, command association, observed
reason, sample chronology and source checks; the reader rechecks the complete
regular artifact tree and source closure after recursive idle validation.
Unavailable samples cannot establish process retirement, physical memory
release, terminal ownership, causal swap attribution or screen success.
Global PID coverage errors and original complete owned memory observations
retain their prior scope. Existing owned cgroup metadata is carried verbatim;
this proposal does not add a physical-memory conservation or attribution gate.

Actual root execution and wrapper integration need a separate reviewed
consumer. This proposal has no wrapper runtime proof and changes no inference
settings, guard thresholds, model arithmetic or registry.
