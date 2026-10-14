# Fast diagnostics, GPU lease closure, and PLE control

CONFIG -> COMMAND -> RESULT -> VERDICT

CONFIG: Read-only source/receipt audit on 2026-10-12. No compiler, GPU,
Docker, model payload reads, process interruption, or frozen source changes.
The actual61 loaded actor is sealed separately. This note proposes future
scope; it changes no existing reader or qualifying requirement.

COMMAND: Read AGENTS.md and the complete
20261009_flashnext_goal_prompt.md; inspect source41 main/native_phase,
source47 V2 boundary_ports, current59 fidelity V3 and actual59 SDK PLE
sources; read completed current59 V3 scan/health/report JSON metadata.

RESULT: User rules require pinned model identity, coherence, health and
owned teardown. Neither standing AGENTS nor the full goal explicitly
requires rereading all publisher bytes before and after every short
startup/selector diagnostic. Fresh complete scan chronology is a concrete
current harness/reader qualification rule. Source47 V2 already replaces
old source41 per-arm scans with one matched-campaign pre/post scan.
Modern actors using those ports must not be described as scanning per arm.

Measured current59 mode0 fidelity V3 root:

- pre-full4: 111334654784 bytes, 72.939451 seconds.
- post-full4: same bytes, 112.516567 seconds.
- terminal to post-health close: 60.531518 seconds.
- report: FAILED, actual exact fresh4/MAX1/absentPIN reader check.
- Both whole scans passed. Their combined 185.456018 seconds bought no
  qualifying model result on this failed diagnostic.

Evidence root:
/mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f17-source37-20261010/source59-mode0-fidelity-v3-run-v1

These are actual receipt durations, not a general 20-minute scan estimate.
The earlier process rchar observation alone cannot distinguish duplicate
whole scans from one pre-scan plus a partially completed post-scan.

VERDICT: Implement a future explicitly unqualified diagnostic mode first.
Authenticate a previously completed passing whole-scan receipt by its exact
SHA, lock/revision, complete roster and whole-file hashes. Require each
current canonical resolved file's dev/inode/size/mtime_ns/ctime_ns to equal
that receipt's stat_after, before launch, during existing guards, and after
owned close. Keep known-page, current source/all8 ELF/config/input identity,
per-card/collective health, kernel-fault and owned EOF/inspect/remove gates.
Declare fresh_model_bytes_read=false, identity_basis=prior_whole_hash_plus_current_stat_pages,
clean_performance_qualified=false, shelf_qualified=false. Do not pass the
old fresh-full4 reader or fabricate current scan timestamps. A missing
receipt, changed stat/path/lock, failed page/source guard, artifact change
or uncertain writer ownership refuses this shortcut. Metadata/page checks
are an explicit bounded diagnostic identity assumption, not a new whole
byte proof. No automatic upgrade to speed or production qualification.

Use fresh complete scans for changed publisher/tokenizer artifacts and
final qualifying campaign boundaries. Any writer touching original model
files invalidates the diagnostic basis. New SDK-only diagnostics still
require their exact new source/ELF binding; previous GPU health never
transfers. Start with a startup/selector-only diagnostic pilot, then run
its successful useful observation under the normal qualifying boundary
when promotion is needed. No redundant operator/math sweep is implied.

For fully qualifying future runs, split the existing lease owner from
closing CPU work. An unleased coordinator launches the bounded GPU child
through gpu-run. The child alone owns all native processes and the GPU
lease; it records typed EOF, owned inspect/removal, actual per-card and
compiled pair post-health plus final kernel journal. Only when all GPU
owners and health children are retired can the child return. The outer
coordinator authenticates that sealed close receipt, performs the fresh
post-whole scan and final source/raw closures, then publishes final PASS.
Before this final CPU close, qualification remains pending. Unknown
retirement or failed health retains the existing recovery/lease discipline.
No live run is released early. This permits NoDri builds while CPU hashing
continues; concurrent hash/model IO can change residency, so clean timing
experiments must still serialize that IO or explicitly label it diagnostic.
Do not use a magic marker file to release a lease owned by the current
monolithic actor.

Smallest seams: source47 V2 boundary_ports campaign post-health/journal
versus post_before_pages/full_buffered_identity/post_pages/campaign_stats;
source41 native_phase already owns protocol.close/safe_retire/remove_owned.
Keep current source61 on its complete existing pre/post boundary.

PLE candidate: existing flags, no engine patch

Actual60 frozen recipe is --ple-io mmap --ple-row-cache 65536. In actual59
src/kernels/ngram.cpp:314, only Direct opens PleReader with cache_rows;
mmap gather_batch at 547 loops read_row serially, which decodes mapped rows.
Thus the configured 65536 bounded row cache does not cache this mmap path.
Direct gather_batch at 507 uses a single ticket for the whole row batch.
PleReader::issue at sycl/src/ngram/ple_reader.cpp:435 caches raw rows,
deduplicates aligned pages and sorts jobs by offset. Default IO worker owns
DirectFile calls; --ple-sync-submit would change that and is not this A/B.
Linux src/platform/direct_file.cpp opens O_RDONLY|O_DIRECT and uses a bounded
parallel pread pool. Same validated table offset, quantized row decoder and
output ordering are retained; actual model coherence remains to be tested.

The failed60 partial observation of incoming RID2/stage0 has a 4678.427868
ms ple_future_get interval; repeated RID4 is near zero. This is unadmitted
partial host evidence. It does not prove storage saturation or GPU idle.

Next runtime control: retain one admitted paired current SDK, authentic
P2048 A512/C64 after A256, single_window/cooperation/copy/mirror/share/input
flags fixed; change only --ple-io mmap versus direct with cache65536 and
inflight256, default worker. Two fresh owned processes are needed because
PLE mode is startup configuration. Record cold and warm cohorts, native
PLE requests/cache_hits/dedup_rows/bytes_read and first512 future wait rows,
full512/64 histories and current fullhead fidelity under declared gate.
Use source61 actual chunk tracing for the wait census, retain ordinary
qualifying identity/health/retirement. Do not use literal prefix1 NUM capture
as a representative P2048 timing workload. No cold-cache assertion without
actual residency evidence; whole scans can warm mmap pages while Direct
bypasses them. No drop_caches or hidden filesystem flush.

Storage is a separate next factor, not combined with the IO-mode switch:
/dev/sdc1 btrfs /mnt/vm_8tb is Samsung SSD 870 SATA (serial S5SSNF0WA05872J);
/dev/nvme1n1p1 btrfs /mnt/cache is SPCC M.2 PCIe SSD NVMe
(serial A0D6079502A700037914). Root reported 720221593600 bytes available on
NVMe. Root may stage the exact PLE shard2 only after active hash/model IO
closes, authenticate its complete unchanged hash, and A/B that path with
all other mode/cache flags held constant. No model substitution or SATA
HDD assumption. This note executes neither the copy nor model test.
