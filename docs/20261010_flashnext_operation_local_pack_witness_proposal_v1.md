# Explicit operation-local pack hash witness prototype

CONFIG

H46 preparation79819 remained CPU-only after eight minutes with roughly364GiB
rchar while its actual pack is1.496GB (dense member1.486GB). The earlier leased
parent similarly showed115.6GiB rchar. These are host read-call observations,
not physical bandwidth measurements or qualified runtime speed. H46 parent-only
admission deduplication leaves nested baseline/adjudication/upload/private
semantic calls intact. All executed and frozen helpers remain unchanged.

COMMAND

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=strata/flash-next python3 -m unittest test_operation_pack_hash_witness_cpu_v1

No actual pack/model bytes, proc records, runtime, Docker, compiler or GPU are
read/executed by this source prototype. Ten tmp byte-fixture controls run.

RESULT

operation_pack_hash_witness_v1.PackEpoch accepts a bounded explicit current
path/expectedSHA roster. It performs a complete fresh byte read at creation,
then lets deliberate semantic callers consume that in-memory result after
joining their own complete current expected roster. It performs another
complete fresh byte read at the predevice seal, then a third after operation.
Each read binds path/descriptor stat5 before/after and checks expected SHA.
Identity changes, current roster drift, saved stale digests, process ownership,
expired lifetime, duplicate paths and invalid phase transitions are refused.

The80-call tiny control performs exactlythree actual full-byte reads. This is
synthetic evidence for the API behavior, not integrated pack optimization or
actual preparation speed. The object is process-local and has no save/load
admission constructor. Reports cannot be imported as future current-byte proof.

The explicit consume_prepared_pack adapter authenticates current pack intake
receipt bytes and exact complete file roster. Its future use replaces only
validate_prepared's pack loop (source37 controller lines512-513). All other
combined generation, ordered patches, source/ELF/runtime/current upload,
layout/model/known-page/health/journal/ownership gates remain mandatory.

Integration requires NEW deliberate caller generations with a named optional
operation witness argument through baseline/adjudication/upload/private
helpers that actually consume pack checks. It cannot be integrated by swapping
module.sha, importing a fake baseline view or silently skipping inherited gates.
Source37/H46 existing helpers currently ignore this prototype and still rehash.

Each process creates its own current-byte witness. A child may share the
expected roster, but must read bytes itself; a parent's saved digest is not
child admission. Every operation publishes final qualification only after its
fresh postoperation boundary closes. Mutation between the explicit read
boundaries remains unobserved, and is labeled as such rather than treated as
stat-only validation. A future parent must seal immediately before first
health/device action and preserve its current source/plan/identity guards;
root must decide whether that declared interval is acceptable before adoption.

VERDICT

SOURCE/CPU prototype only, not drop-in admission or a speed qualification.
It targets repeated pack byte reads while retaining fresh predevice/post byte
verification. Review the explicit boundary semantics and full caller graph
before authorizing a new integrated harness; current H46 remains unchanged.
