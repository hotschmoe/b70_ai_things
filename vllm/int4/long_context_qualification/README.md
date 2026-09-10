# Separate 200K long-context qualification

CONFIG -> Immutable image7b107d0e (GDN phase plus MRV1 accepted-count repair),
fresh FP8 KV scale artifactbe02d915, TP2/P2P0, MTP3, cache ON,
FULL_DECODE_ONLY graphs, max context200000, c4, batch32768 and original0.96
memory utilization. Stable primary hotschmoe-dd plus the detailed research
alias in models.yaml. The scales were collected at <=96K; longer validation
is a separate test and does not retroactively expand calibration coverage.

COMMAND -> Run tokenize_corpus.py using the exact serving image's CPU tokenizer
with model mounted readonly, no devices and network disabled. Freeze prepare.py
only after that output exists. Launch later through launch.py, which verifies
the successful100K predecessor (WORKLOADS_PASSED and lifecycle0), all frozen
hashes and a fresh output path before delegating to existing run_arm.py. The
existing server owns both GPU leases, strict per-card and compiled-pair
pre/post health, timeout handling and verified teardown. No GPU launched here.

RESULT -> Exact chat-template token counts, thinking disabled and generation
prompt enabled: retrieval185011/185012/185007/185005 and corresponding
guide185065/185066/185061/185059. Every prompt plus requested output fits below
200000. Corpus records token-ID hashes, prompt hashes, tokenizer version/class
and model tokenizer-file hashes. The initial CPU attempt failed because this
transformers version returns BatchEncoding by default; explicit return_dict=False
corrected token extraction. No failed count was used. No-device torch import
reported zero XPU devices; no device was exposed to either CPU container.

The jobs run four concurrent independent archives containing three exact secrets
at25/50/75 percent depth, then identical prompt reuse, then four concurrent long
LRU guides. They share a new arm namespace but distinct archive prefixes.
Afterwards one long guide stream is closed after eight content chunks, running
and waiting gauges must drain within60 seconds, and four exact retrievals must
recover with positive cache hits. Client closure and gauges are bounded recovery
evidence, not a worker cancellation trace. No model-generated tools or code
are executed. A final startup host-boundary gate remains separate from device
completion or graph-replay counts.

Every complete long response must report precisely the CPU token count, valid
usage, EOS rather than cap exhaustion, and no long repeated output unit.
Retrievals must equal all three expected secrets as an exact JSON array;
reuse and recovery require positive per-request cached_tokens. Guides require
five requested sections, balanced Python code fences, sufficient text and
existing degeneration checks. This tests structure and degeneration, not full
semantic correctness of generated implementations. Each request is bounded to
1200 seconds, each workload to1350; the arm stops on a failed stage. Four
concurrent185K contexts are intentional memory pressure, not a claim they fit
or perform well before measurement.

Three CPU tests pass: wrong secret, tokenizer mismatch, zero reuse and truncated
finish reject; cancellation is distinguished from successful completion and
repetition rejects; the launcher refuses an unqualified100K prerequisite.

VERDICT -> Prepared optional validation only. No shelf promotion or stability,
long-context quality or speed claim. Preserve failures, actual resolved cache
geometry, live model identity, source/native identity, trace results, health
and teardown before interpreting outcomes. Scales remain unchanged.
