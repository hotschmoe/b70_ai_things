# S49: live batch PLE reuse, default retained OFF (2026-10-15)

CONFIG -> The CPU prefetch path is now reachable and successful aggregate
reuse increases. Exact outputs, functionals and health pass. The complete
work-matching gate fails, and the two fully matched timed repetitions show
no useful application gain. Keep batch prefetch OFF; do not rerun this screen.
The qualified152055a6/headpairON/depth2 binary remains unchanged.

S47 showed that fresh batch slots lacked the table/hash metadata needed for
MTP prefetch. S49 borrows only validated main table/consts into those slots,
leaving their GPU PleRun unbound. CPU prefetch guards require an open table;
GPU consumers retain ready(). The exact opt-in requires string "1", serve,
batch MTP and main PLE readiness. Startup confirms OFF wired0, ON wired2,
slot_ple_ready0, depth2 and vector budget6 against cap16.

The same executable includes the S47 key-match/drain-all helper and two
successful fast-path counters in both arms. OFF is therefore a new control,
not the old qualified binary. Normal QUIT optionally prints one final report
after the existing GPU drain, before _Exit. No server/proxy/token tee is added.
Shared table access is serialized in this tested SYCL/no-layer-split path;
the asynchronous prefill gather joins before batch dispatch. This does not
qualify independently threaded per-slot access or every other backend path.

Direct PLE, EP cards0+1,149/363 experts per layer (all24576 resident), KVint8,
max_context200000, batch2/depth2/headpairON/spec4 and32GiB/four-entry parking
remain fixed. Existing buffered S32 CPU markers are ON in both arms; there is
no external profiler. No200K request, C4/mixed-prefill sweep or daily restore.
Only batch-prefetch0/1 and log paths differ between the two fresh servers.

COMMAND -> Actual claude-opus-5-5/claude-fable-5-1 source reviews exit0
(69.37/65.84s). Root confirms Direct mode, source scopes and exact final hashes.
Incremental build compiles only ngram/mtp/generate, updates two archives and
links. A10s initialization attempt ends before compile; an18s attempt compiles
ngram then fails on inherited root-owned .ninja_log. Both are preserved.
Root initializes oneAPI before strict shell options and repairs ownership of
only the log/executable/two relink archives. Successful build takes202s,
unchanged flags/cache/server, no configure/clean/DRI/network/stack change.

Root runs full per-card and compiled TP10/P2P0 health, then OFF and ON:
startup under cpu-run, one128x2 warm pair, three fresh128x2 pairs under quiet
CPU, concurrent math/reasoning/tool continuation, normal TERM->server QUIT,
then per-card/compiled health. The live server owns both-card GPU lease;
HTTP clients never nest another GPU lease. CPU lease is not held for the
whole server lifetime. Original prepared harness is preserved; runtime review
fixes cleanup after failed checks and marker/native count reconciliation.

Raw P:
`/mnt/vm_8tb/b70/results/flashnext_ista_gsq_rco_iq3xxs_strata_cachefix_20261014/batch-ple-prefetch-s49-prep/`.
Actual build `bash P/build-root.sh`; `P/harness-prep/{start_arm,client,
stop_and_health}.py` run with the wrappers above. Strict parser runs once per
saved segment into analysis-mode{0,1}-{warm,repeat1,repeat2,repeat3}; then
`python3 P/harness-prep/compare.py` exits2, correctly retaining unmatched work.
Root independently requires BOTH card and collective records/exit0.

| Identity | SHA256 |
| --- | --- |
| Frozen experimental executable | 0c30e8aa9625dbd35966752d1bfcf4191f2c4fb1b028335dd4f78170c407ca23 |
| Complete repository patch | 50d23e71f2f645a4f55ef919f6813f47317f413fdca4ecbbfbee2b4ca330b598 |
| ngram.cpp | 53035b1723d038f699b62175f106861ed9be705b5e29d29160c27ff194a2c9b2 |
| mtp.cpp | 5b46c431c0dbfa57cc3f1d087794a5e3970836069b284370a6948b18476d805b |
| generate.cpp | 6433d750e414aeb9c71135257ae40a8f9e38ea661537c251b50cabdbc2217d83 |

RESULT -> Eight corresponding OFF/ON outputs (16 coding responses) match
full text, concatenated SSE content and usage; all128-token counts and warm
S16 retained prefixes pass. Both arms pass323/667 concurrent arithmetic,
391 reasoning and lookup_build_status tool continuation. All marker dumps
are complete/dropped0 and reconcile positive windows and paired calls with
native summaries. Both servers QUIT normally, wrappers exit0, containers are
removed and per-card plus compiled collective health exit0. No reset.
Startup observations are172.223s OFF and82.127s ON, fixed order; no causal
startup comparison follows from these two observations.

| Process-cumulative reader counter | OFF | ON |
| --- | ---: | ---: |
| Successful prefetch gathers | 93 | 284 |
| Successful requested tokens from prefetch | 372 | 1491 |
| Requested rows | 46240 | 46288 |
| Demand SSD-read counter | 15475 | 15475 |
| Keepalive reads | 1561 | 1868 |
| Blocked ms | 160.478 | 150.500 |

The+1119-token difference proves additional successful reuse in aggregate.
These totals include admission/prefill/warm/timed/functionals and duplicate
requested tokens; they do not attribute every hit to a batch slot or window.
Zero-token gathers cannot create the positive token delta. Equal demand-read
counters are not physical-I/O equivalence, especially with different keepalive
totals. Logical requested rows and bytes are not physical transfer bytes.

| Pair | OFF wall s | ON wall s | Exact observed work/capture match |
| --- | ---: | ---: | --- |
| warm | 4.909422 | 4.853700 | no |
| repeat1 | 4.824458 | 4.831223 | yes |
| repeat2 | 4.794172 | 4.815717 | yes |
| repeat3 | 4.806594 | 4.831691 | no |

Warm accepted-draft counts are80/79 versus79/80 per logical request; native
capture/layout comparison also differs. Repeat3 has49/45/3 windows/pair/solo
OFF versus48/46/1 ON. Admission order is a hypothesis, not an observed cause.
Do not sort/relabel requests, drop these observations or normalize by tokens.
The two fully matched timed pairs are slightly slower ON, without a meaningful
benefit signal. They are not a replacement pooled sample or stability claim.

In matched repeat1/2, replay-window stage_PLE medians are19.85/22.31us OFF
versus13.18/13.12us ON; whole-window medians46.245/46.250ms versus
46.165/46.190ms. Counts41/40 agree across arms. This is a measured hot-row-cache
case, not a universal savings ceiling or a full causal latency decomposition.
CPU queue waits do not prove GPU occupancy, physical idle or producer readiness.
Fresh128-output client decode intervals include admission/prefill pauses;
SSE event gaps are not individual token ITL. Raw native token IDs are consumed
by the existing pump and were not observed. Root predeclared that limitation
before requests, correcting the original overstrong source-only plan.

VERDICT -> Actual post-review claude-opus-5-5 (32.83s) and claude-fable-5-1
(34.39s) independently favor defaultOFF, no C2 retry, an isolated HC candidate
with S49OFF fixed in both arms, and a separate RPW-head probe. Root accepts
that decision and rejects their stronger physical-I/O equivalence, GPU-bound
and universal-ceiling interpretations. Scoped correctness and reachability
are useful findings; speed and shelf promotion remain unqualified. Absolute
serving targets remain unmet; prior true-RAM qualification is not repeated.

Loop: CPU mechanism test first, one incremental build, two finite serves,
no routine daily restore, one parser pass/segment, all negative/unmatched
results retained. Next fresh HC comparison can control native request order
using existing trace markers, identically enabled in both arms; S49 timings
cannot be reused under that changed protocol. Future NVMe placement is
recorded separately and never changes a path between these S49 arms.

Frozen raw manifest:224files/14,290,280bytes, SHA256
`cbfa6b190e2f587adebcdafde76d32cff3369b090a9d2903e3466dd9fd3ef616`.
