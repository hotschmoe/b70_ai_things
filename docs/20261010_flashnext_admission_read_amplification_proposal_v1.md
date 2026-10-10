# Operation-local admission read reduction proposal

CONFIG

Frozen parent43/40 behavior and actual runtime evidence stay unchanged. Root
observed prelease admission83s, then the same whole admission after gpu-run exec,
and leased process rchar124161860824 bytes with physicalread8192. This is observed
host admission overhead, not a GPU/model qualified latency or speed result.

COMMAND

Read-only source call study and stat/receipt census, no payload reads by agent.
Largest explicit loop: c1_serve_controller_combined_v137.py:512-513 rehashes every
intake pack file for every validate_prepared call. Nested manifest -> two topology
baselines +chosen baseline +collector.parent_arm +first49.finalized_binding ->
more current baseline/source/artifact/tree checks repeats it. First49 additionally
rechecks source joins, collector admission, artifact hashes and original tree.

RESULT

Intake binds10 files total1495822427 bytes, dense.bin1485688320. SDK eightELFs total
138226240 bytes; source64 total3072048. Failed serial0 tree69854357 bytes, including
combinedlog41084542 and rawJSON24507781. ON tree105009474, log91696153. One pack is
not115GiB; roughly80 repeated dense passes could account for that rchar volume.
Exact attribution/count requires future bounded read/hash/parse instrumentation.

Harness44 immediately removes duplicate whole prelease/leased/startup traversal:
cheap source/shape before exclusion lease, full current semantic once before any
health/GPU touch, unchanged postchecks. Lease acquisition is exclusion only.

Further NEW gate API proposal: build an operation-owned immutable metadata view
and explicit fresh payload-digest witnesses for the unique current dependency
corpus. Stream each payload once per pre phase, with before/after stat5 and actual
SHA; parse small JSON/source metadata from owned byte snapshots. NEW semantic gate
functions consume that explicit same-operation witness, never an old receipt PASS
or hidden globally cached sha function. Then independently stream every current
file again after the semantic phase and compare fullbytes/stat5/epochs. Never
reuse by stat alone or persist witness authority across operations/processes.
Do not snapshot1.49GiB payload data into RAM while serving; retain only bounded
metadata and fresh streamed witness records. Original all4 scans/pages remain.

VERDICT

Source proposal and CPU mutation/order controls only. No optimized runtime/speed
qualification. Witness API requires its own reviewed generation, negative mutation,
ABA/replacement/duplicate-key/phase-escape controls and actual read-call evidence.
