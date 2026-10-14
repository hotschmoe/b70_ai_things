# Campaign identity scans for future performance actors

CONFIG -> Source/CPU-only change. Live source47 actor, old reports, model
payloads, GPU leases and registry are untouched. User authorized fewer tests.

COMMAND -> Inspect source45/47 actors and their inherited source41 main and
public reader. Read completed full4 receipt epochs. Port only identity scope
to new V2 files and run one meaningful synthetic parent-flow control.

RESULT -> Existing matched two-arm parent performs four111,334,654,784-byte
scans: OFF pre, OFF post, ON pre, ON post. Source45 OFF post took161.134s and
immediately adjacent ON pre77.957s:239.091s and222,669,309,568 bytes of internal
work that a shared experiment boundary can remove. Source47 OFF post took
172.488s before starting another ON pre scan. These are observed identity-scan
times, not inference throughput improvements. Source45 and source47 failures
remain failures; no old evidence is rewritten.

NEW actor `run_source47_grouped_f16_model_v2.py` provides
`boundary_ports(namespace)`. It runs one fresh publisher full4 before all arms
and one in the outer finally after the last attempted arm and post-health,
including failed experiments. Each arm retains current source/all ABI binary
checks, exact config/command/input evidence, stat/inode/size/mtime/ctime
snapshots, all three known page guards and active guardian, strict/compiled
pre/post health, kernel journals and owned normal teardown/removal evidence.
No arm may independently claim qualification before the campaign closes.

NEW reader `validate_source47_grouped_f16_model_v2.py` provides
`boundary_reader_ports(active_actor)`. It validates the saved whole-experiment
publisher hashes, receipt/artifact hashes, current stat and preserved/current
three-page guards. It preserves existing source/config/raw-output/health/
journal/namespace/retirement predicates and omits the old public reader's
additional111GB scan. No historical recursive requalification is added.

Default source47 construction is SAFE OFF-only, groupedF16=0. It is a CPU
construction control; no additional baseline GPU run or unchanged source47
grouped ON retry is requested. The helpers are reusable for future matched
source46/48 controls after actual SDK receipt admission; no new SDK is baked in.

One CPU test `test_source47_campaign_identity_cpu_v2.py` executes the actual
new compiled parent with tiny four-file synthetic weights and fake native/
health calls. It verifies exactly two scans for a matched two-arm experiment,
rejects an inter-arm file/stat mutation, proves a native failure still performs
the final close, and rejects missing final identity or changed arm stat. PASS,
approximately0.018s. This is flow/identity validation, not GPU/model fidelity.

VERDICT -> Use the new boundary helpers for the next justified performance
control. Preserve all page-fault and lifecycle evidence. This removes measured
duplicate I/O without changing weights, inference math or the current actor.

Owned source plan: `strata/flash-next/source47-grouped-f16-model-source-plan-v2.json`.
