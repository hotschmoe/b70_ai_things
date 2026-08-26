# Results

This directory starts clean after the 2026-08-26 archive pass. Historical raw
results and logs are in the gitignored cleanup quarantine.

The ignored `logs/` directory retains only raw evidence from 2026-08-23 through
2026-08-26 because the active journal and current topology/Steve findings cite
those runs. The same rule applies to the retained `oneccl_oracle/` and fused
all-reduce mechanism evidence. These paths resolve into
`archive/research-evidence-20260823-26/`; they are research evidence, not a
runtime dependency or part of the cleanup deletion batch.

For new experiments, record config, command, result, and verdict in `JOURNAL.md`.
Keep only compact summaries in git. Put large traces, profiles, and raw logs under
an ignored results subdirectory and link them from the journal when useful.
