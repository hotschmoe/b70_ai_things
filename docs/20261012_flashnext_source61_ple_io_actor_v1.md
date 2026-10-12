# Actual61 PLE mmap/direct comparison

CONFIG -> COMMAND -> RESULT -> VERDICT

CONFIG: Existing immutable actual61 SDK, exact original UD-Q4_K_XL artifacts,
paired native2, mirror/copy0, cooperation/cap1, default decode share0.5.
Two owned processes differ only in --ple-io mmap/direct. Each runs authentic
A512/C64 after A256 twice; cache65536/inflight256/default IO worker fixed.
Both arms require all512/64 histories equal to accepted52/58 references and
one another, and exactly eight complete current PFH61-selection host runs.
No engine patch, GPU operator sweep, new quantization or fullhead claim.

COMMAND: Use the command array in
/tmp/b70-source61-ple-io-v1-launch-final-agent.json.
Run independent reader_command only after the producer closes. Default
qualifying mode retains fresh campaign pre/post complete publisher hashes.
Optional --split-GPU-CPU-close uses one unleased coordinator and the same
GPU child. The child holds the lease through typed EOF, owned removal,
closing card/compiled pair health and kernel journal. Unknown retirement
or failed health retains its foreground lease owner for coordinator recovery.
CPU final identity follows released GPU ownership; public reader rejects a
pending child report. CPU/source/raw closure remains mandatory before PASS.
No clean timing claim while a different workload competes with this hash IO.

Fast diagnostic: separate command array in
/tmp/b70-source61-ple-io-v1-launch-diagnostic-agent.json authenticates one
completed prior whole receipt by SHA and exact lock/revision/four-file hashes.
Current canonical dev/inode/size/mtime_ns/ctime_ns must match every prior
stat_before/stat_after. Existing known-page/source/ELF/config/input/health/
owned retirement guards remain. Changed paths/stats/lock fail closed.
Distinct diagnostic-identity files state fresh_model_bytes_read=false and
prior path/SHA/time. This is explicitly unqualified for speed/shelf/fullmath;
no old reader or fresh scan timestamp is fabricated. Final qualifying runs
retain fresh whole scans. Model artifact writes invalidate prior identity.
Source61 queued single run and all frozen evidence remain unchanged.

RESULT: Actual SDK command construction, CPU-only generated ownership-flow
controls, and independent diagnostic reader port checks are recorded in the
handoff. Controls exercise unchanged tiny synthetic files, inter-arm mutation,
native failure, missing final identity, GPU-child pending close and a final
CPU scan after typed release. No model/GPU/compiler/Docker execution by author.
Runtime comparison remains pending root execution.

VERDICT: Useful existing-engine A/B without another compilation. Direct's
bounded raw row cache is 65536*90=5898240 payload bytes for original IQ4_NL;
metadata and aligned inflight slab are additional. Mmap bypasses this bounded
cache and uses OS page cache. Neither initial cohort is claimed cold: publisher
hashing can warm mmap pages, Direct O_DIRECT bypasses page cache. No drop_caches.
Native serve does not expose PleTable::io_report (printer is non-serving stats
at generate.cpp13157); absence is not zero IO. Physical exact byte/cache-hit
counts remain unqualified; preserve any actually emitted original counters.
The host future waits and token timings can identify a useful mode effect.

Source-only equality audit in actual61 src/kernels/ngram.cpp: Direct and mmap
share Impl::decode, selected by the same validated tensor format. Load validates
data_start+tensor offset and extent before Direct drops the parsing mapping.
Direct gathers one raw batch and decodes raw+i*rowbytes in original input order;
mmap invokes read_row for each original row. PleReader cache keys are original
row IDs; open resets cache for its exact table instance, completed page reads
insert matching raw row bytes and scatter each requested destination. Invalid
row indices produce zero raw rows in Direct, while mmap zeroes F32 output;
original IQ4_NL zero-half scales decode those zero bytes to zeros. Direct collect
failure returns failure from gather_batch; scalar read_row's existing failure
zeroing is unchanged and not newly qualified. Existing direct_file_async.cpp
covers the asynchronous file layer, native_dense_ple_key_test.cpp covers PLE
key projection, neither is a model-level mmap/direct equality result. This
source audit supports the A/B, not an invented fullhead/mixed-format proof.
