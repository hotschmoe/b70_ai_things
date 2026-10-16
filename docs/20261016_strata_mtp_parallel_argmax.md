# S114: MTP parallel argmax main integration and serving discriminator

S114 integrated the strict S112 private parallel top-1 candidate behind the
default-OFF `STRATA_MTP_PARALLEL_ARGMAX` flag, then completed one matched C2
serving discriminator. Aggregate output median moved from 103.073716 to
103.673778 tok/s (+0.582168%); per-stream decode medians moved from
53.485314/55.836291 to 53.858032/56.240725 tok/s. All reported matching,
functionals, normal QUIT, owned-container removal and post-health gates passed.
This is one OFF-then-ON sequence with three nested pairs per process. It does
not establish significance or stability and does not promote a default or
shelf. Actual post-serving Opus/Fable replies and root close S114 with the
flag retained default OFF, no immediate repeat and no cache campaign.

CONFIG -> The same new main `d1c4968ffbb62e66753e0c016d830d7af53154d0de6c6b8f1b966fec4d518e92`
and pinned e41 image served both arms. Only this flag changed 0/1, apart from
arm-local names/log paths. Q6 A4/center/packed, IQ2 multi, attention value
pipeline, whole-pair graph, probability omission, head RPW2, QSA tiled/prompt
XMX and KV ROT candidates were OFF. The existing paired head at depth two,
Direct PLE, EP149/363 residency, parallel2/context200000, prefill12288, INT8
pool/32768MiB/four parked conversations and model identity remained fixed.
Native work flags were suffix0/lookup0/spec4/mtp-max-t4/min-p0 in both arms;
HTTP explicitly overrode only `spec_min_p=0`. TRACE1 supplied prompt-start
admission evidence; no profiler or CPU ring ran in these measurements.

COMMAND -> Actual source reviews preceded one incremental CPU-only build.
Root and independent audit read the selected compiled kernels and host guards,
resolved the initial default-image HOLD additively, then obtained actual
compiled reviews. V2 controllers ran one 107-prompt/256-output C2 warm pair,
then three fresh 107/1500 C2 pairs per arm, concurrent math/thinking/tool
functionals, and normal stop plus fresh full health. Request1 was released
only after request0's native prompt-start while request0 remained outstanding;
client lifetimes overlapped. ON warm and each ON pair matched the control
before further requests. The finite root driver used `finally` cleanup.

RESULT -> The comparison is `PASS_MATCHED_DISCRIMINATOR`, with every reported
check true. Both arms and all finite phases returned zero. The only numerical
claims here come from the actual
[comparison](../strata/flash-next/lowlevel_probe/mtp_parallel_argmax_serving_s114/serving-actual/comparison.json).

| Timed measure | OFF | ON |
| --- | ---: | ---: |
| Aggregate median output tok/s | 103.073716 | 103.673778 |
| Pair median wall seconds | 29.105383 | 28.936922 |
| Stream0 median decode tok/s | 53.485314 | 53.858032 |
| Stream1 median decode tok/s | 55.836291 | 56.240725 |
| Stream0 median TTFT seconds | 0.913302 | 0.914998 |
| Stream1 median TTFT seconds | 2.096913 | 2.090230 |

All three paired aggregate comparisons were slightly positive, approximately
+0.582%, +0.667% and +0.464%. They share one process per arm and fixed order;
they are not independent process replications. SSE content-event gaps are
client metrics and may span multiple tokens, not token ITL or device service.
The S112 private two-owner chain observation is not multiplied into a serving
budget or used to forecast model throughput.

VERDICT -> CLOSED: measured bounded main integration passed, source/evidence
retained, default OFF. No immediate repeat, cache campaign, default or shelf
promotion. The 150 tok/s/stream goal remains unmet.

The implementation changes only SYCL `mtp.cpp` and one local header. The
ASCII [patch, inverse and proof](../strata/flash-next/lowlevel_probe/mtp_parallel_argmax_serving_s114/integration/)
preserve the original source identity without copying its existing UTF8 into
the repository. Four inverse changes reconstruct prior MTP source
`13500a3af44f30793617d51aff6639344d2d943a50b53bdfe3c60ba81081967f`;
candidate source is `2d26e686c2aa6065d33461464efa39975db8e0cc5196f8962a216b4c5fa2a589`.
The private header is byte-exact S112 v2
`c1652c6aaba20b7cfd83320b86234c6fbcf25bcf985fa966873ee737bd5b90ba`.
Its first historical comment is retained verbatim; S114 now compiles it into
the opt-in main. Generic source, shared sampler/verify TUs and public headers
are unchanged.

The process-static flag accepts exactly the one-character string `1` and is
initialized at bind before captures. Existing architecture-gated
`argmax_rows_wanted()` stays first. The new path is only inside the old
sampler else branch, gated by full head/no subset, T1, vocabulary248320,
both coupled states false, and the unchanged local greedy/temp0/default
penalty parameters. False gates call the original sampler with identical
arguments. Probability, ID mapping and select remain downstream, unchanged.
This is not the separate S111 probability-omission proposal.

Each owner reuses its existing private `ArgRow` scratch: 1040-byte rows,
counter offset0, values16, indices528. Bind checks pointer/capacity/ABI before
capture and fails explicitly on mismatch. The candidate uses row0 only,
61 blocks, per-block4071, local256/global15616. A/B pair tails forward the
respective owner scratch/output IDs and separate logits rows on the same
pair queue. Source arena allocation/queue idle/drain/destruction establishes
the lifetime argument; compiled owner-field loads do not independently prove
heap disjointness. No scratch reset, retry or redraft recovery is introduced.
An aborted replay is terminal for this protocol.

The [independent compiled host audit](../strata/flash-next/lowlevel_probe/mtp_parallel_argmax_serving_s114/host-independent-audit/note.md)
pins actual gate and call anchors. Helper0x98f0 implements guarded exact-string
initialization; bind's inlined gate precedes ABI/capacity checks. At
`record_post_head`0xdd5b the old argmax predicate remains first; the candidate
call0xde2b and default sampler0xde50 join at0xde59 before probability.
Component-tail calls0x16e39/0x16e5e forward owner A/B and second logits at
`+4*nv`. Source constant T1/greedy/penalty checks are compiler-folded, not
request-variable runtime tests. Four unique Info/cache/guard/name bindings
are established by selected host relocations, metadata and image mapping.
The once marker is HOST selection only, possibly emitted during startup
capture; observed OFF0/ON1 does not count GPU dispatches or cover every owner.

The actual Ninja plan rebuilt only MTP object, engine archive and main.
Incremental Ninja elapsed159.932s, full CPU wrapper174s/exit0, under four CPU
cores/jobs, pinned e41, UID1000, no network/DRI, disk TMP. MTP object is
`c430dcd6f274371345fbb1aec1cd38ad6182a8ab1b046e704b8300cc76aad385`.
The [build receipts](../strata/flash-next/lowlevel_probe/mtp_parallel_argmax_serving_s114/build-actual/)
retain changed and unchanged identities; the kernels archive stayed
`cf710683aec404599d1cc6173bf8364e6d30731bda82b315df286bf58aed9b38`.
No complete archive/model rehash was performed for this curation.

The first compiled audit correctly returned HOLD on eight changed native
text bytes each in sampler and probability. That original result and exit
remain unchanged. One CPU-only two-selected-image decode (exit0/11s) then
proved sampler native prefix through EOT7536 and probability through EOT3696
byte-exact to S107. All label references stay before EOT, with no unaccounted
indirect/call/return transfer. The eight differing bytes are two immediate
fields in null-destination MOVs after EOT. The analyzer first misclassified
explicit direct `jmpi L<label>` as indirect; its failure and corrected bounded
predicate are retained, without repeating a decoder. Metadata/options/data/
relocations are equal. Embedded SPIR-V/trailer values remain unclassified:
raw whole-image/text equality is false, not silently rewritten as true.

The candidate's 9728-byte computational prefix through EOT is byte-exact to
S112, with matching metadata/options/data/relocations. This reuses its reviewed
release/returning-atomic/leader device-acquire-before-WG-barrier publication
chain, comparisons/lower-index tie policy, sentinel and counter reset, without
another candidate decode. Select is whole-image exact. Candidate resources
are SG32/GRF128/SLM544/barrier_count1/eu_threads8; absent spill/scratch fields
are UNKNOWN. These are metadata, not runtime resource traffic or a count of
static barrier instructions. Whole-pair graph stayed OFF; its depth-two caller
bodies were not disassembled. Bounded computational reuse is not whole-image,
SPIR-V or universal program equivalence.

Actual source consultations were Opus `claude-opus-5-5`40.294855s and Fable
`claude-fable-5-1`45.957781s, both terminal0/CPU_BUILD_GO. Root checked real
sampler fields/defaults and the inverse, then applied exactly the reviewed
source. Actual compiled consultations were the same responding IDs,
24.335633s/23.709831s, terminal0/finite GPU_GO conditional on hardened v2
cleanup. Root read all six controllers and the diagnostic-before-health delta.
Filtered receipts retain actual IDs/sessions/local elapsed/exit and omit
cumulative CLI usage/cost/duration. The replies remain historical evidence;
root corrections take precedence over model text that overstates HTTP fields,
requires a historical rerun, predicts a small gain, or infers behavior after
an aborted counter protocol.

Actual post-serving consultations completed in the same responding models'
own resumed sessions: Opus68.061816s/Fable86.535391s, both terminal0. Root read
both complete answers and agreed to close S114/default OFF/no repeat. Exact
replies, prompts, commands, filtered receipts and the61186-byte combined
packet are retained in
[postreviews](../strata/flash-next/lowlevel_probe/mtp_parallel_argmax_serving_s114/postreviews/).
The combined packet also reviews S116's private HC source, not an S116 GPU
result. Root accepted ONE CPU-only private six-class build followed by a
selected compiled audit; there is no GPU or production-integration approval.

Root explicitly rejects Fable's universal claims that FP32 stored-forwarded
operands can never differ across contexts or that division and mul0.25 always
agree for finite/subnormal values. Actual materialization, operand/order,
CR0 and lowering must be inspected, with a differing form held for an explicit
decision rather than tolerance. Fable's declaration that probability,
selector and submission changes are an experimentally closed class is also
too broad: S111 probability omission is still SOURCE PREP ONLY, unexecuted.
These individual small comparisons close; every mechanism does not.

S116 anchor provenance was closed independently before CPU build approval:
36 actual mode1 hashes matched retained records; fresh mode0 hashes matched
mode1 and the S39 cross-mode record. The original provenance note's420096-byte
checker observation concerned unbuilt S39 v1. Its retained addendum identifies
actual built S39 v2 source
`ae2fb80d8e43d4fde0bf638d19c64213f651a263662a7a47bb18c045b341c2dc`,
matching the original build closure,747776-byte R/lo/rs/inject/mixed/xn layout
and0xff poisoning. This is a source-version clarification, not a historical
evidence inconsistency; no old source or result was rewritten. S116 build or
later GPU outcomes belong to that separate stage.

Observable match scope is complete request/text/concatenated SSE content,
usage, per-request cache/prompt/predicted/draft/accepted counts, native windows,
paired/solo counts, printed rounded rows and observed capture logs. Warm work
was95 windows/92paired/2solo; timed pairs were567/562/4,562/559/2 and556/553/2
respectively. Raw token IDs and
the unprinted per-window work sequence are not observed. Raw SSE chunk
segmentation, response IDs and timings are not claimed equal.

Whole-process PLE aggregates deliberately remain unmatched: OFF/ON gathers
129/128, prefetch_tokens429/428 and rows199200/199184. They include
prefill/admission/functionals, outside matched coding windows. Both raw
aggregates and final native logs are kept; no full-process work-equality claim
is made. Normal teardown observed pre-stop Running=true/OOMKilled=false,
TERM->native QUIT, wrapper0 and both exact CID and name absent. Final aggregate
diagnostics were checked before post-health, which passed per-card and
compiled collective0 in both arms. Removed-container final OOM state is not
observable. Pre-health was an explicit same-boot contextual reuse from closed
S113 full health with no intervening coordinated GPU work; not a new probe.

The loop reused the S107 six-controller design, retained v1 history, and made
only the v2 ownership/diagnostic cleanup correction. Warm took about7s/arm,
timed phases87.758/87.373s, functionals6.157/6.149s, stop/full health59.282/
63.293s. No private-oracle rerun, redundant candidate decode or routine daily
server restoration was needed. Original large SSE records remain on disk
with exact identities; deterministic compact records preserve full requests,
text and concatenated content while omitting redundant chunk metadata.

The [bundle and index](../strata/flash-next/lowlevel_probe/mtp_parallel_argmax_serving_s114/README.md)
retain exact small source/build/audit/controller/closure records. Full UTF8
MTP sources, binaries/archives/objects/device images/ISA and CLI event streams
remain external. See the [data map](../strata/flash-next/data-map/README.md)
for the wider path; S114 measures this isolated integration, not every stage.
