# Structural decode leads S123/S125

Source-only review and actual model consultations; no implementation, build,
GPU run, allocation or timing was performed in these stages.

Read [the root decision](reviews/root-decision.json) before the raw answers:
it records accepted follow-ups and corrections to reviewer claims. S126
private runtime-uniform ones is first; S127 examines same-device owner queues.
Cross-card placement and Q6 pre-expansion remain parked, not disproved.

- s123/: exact frozen dual-card ownership/bridge/fallback source evidence.
- s125/: exact frozen Q6 expansion/FP-order/capacity evidence.
- reviews/: exact shared prompt, commands, answers, root decision and filtered
  completion metadata with actual responding models and wrapper durations.
- source-copy-index.json: original and target hashes, exact/derived distinction.
- external-identities.json: deduplicated external packet/prompt identities.

The raw S125 table covers 43/52/71 matrices. The separately selected 29-output
subset is 488202240B of logical EXTRA compact storage with originals retained;
it is not an allocation/free-memory/residency result. Historical headroom and
static instruction/geometry counts do not establish current serving cost.

The review harness and commands are archival evidence, with original external
paths. Their presence does not authorize execution. Cumulative session usage
is deliberately omitted from filtered completions.
