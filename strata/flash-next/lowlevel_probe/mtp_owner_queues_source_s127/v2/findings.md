# S127 V2 source-only candidate: paired MTP forwards on existing owner queues

CONFIG -> Default-OFF STRATA_BATCH_MTP_OWNER_QUEUES=1 (exact string1).
Same device, same immutable shared weights/head, same paired eligibility and
kernel arithmetic/arguments. One code patch, SYCL mtp.cpp+mtp.hpp only. S123
cross-card and S125 Q6 pre-expansion source designs stay parked; no architecture judgment
or claimed6ms ceiling. No apply, compiler, GPU, model call or prototype here.

COMMAND -> Read actual current forward/capture/caller/lifetime code and narrow
callee buffer contracts; make external candidate+exact diff; git apply --check
only, exit0. Base source SHA2d26e686; exact full SHA/bytes in source-pins.json.
Exact snippets include the PLE/host-service question, not a general map.

RESULT -> Minimal scheduling seam is feasible at source level. Each drafter
load creates its OWN in-order cs_ and side_ queues (mtp502..541). Existing
pair pregraphs use A.cs_ for BOTH owners (2130..2187), although owner B already
captures its solo work on B.cs_. The candidate records B's head-free pregraphs
on B.cs_, leaves A on A.cs_, and submits both at a depth before blocking on
both queues. Only after both completions does it submit the UNCHANGED primary
paired tail. It waits that tail before either owner's next depth, establishing
B's token/Rin_ dependency too. Host completion + later submission is explicit;
no unsupported cross-queue graph dependency API or removed wait is assumed.
Whole-pair graph is explicitly bypassed while this immutable mode is ON.
Marker is once/object HOST selection only; it does not prove concurrent GPU
execution or execution counts. Kernel/graph scheduling may still serialize.

Ownership audit: generate4061..4100 creates a distinct SessionState for each
slot;4165..4183 loads each drafter with that slot's session. MTP shares only
read-only dense_/experts_/tensor metadata from the global owner(278..303).
Each load allocates own state_arena_(393..429), own mapped records, arena and
all activation/attention/router/hit/shared/output/argmax scratch(434..478).
QsaState's pools/page table/status/map/host mirror are per-state. RoPE tables
may be borrowed, and are read-only during forwards(layer839..879/901..952).
Even record_rest's fallback ss.block.gr is distinct by slot; shared_ and
sh_scratch_/sh_fork_/sh_join_ belong to each drafter. Grouped/shared expert
callees carve the supplied owner's scratch. HC additionally uses queue-keyed
function-static device buffers, as the separate static audit establishes.
Native embedding and head are common read-only tables; the tail owns xq2_/
logits2_, writes each owner's independent selection/scratch serially on A.
These are source ownership facts, not a runtime pointer-disjointness receipt.

Critical PLE check: NO GPU host-service pump is reachable in these MTP
forwards. Full record_front/rest825..1165 has no PleRun/table/ple_ss_ or PLE
request ring argument. ss_ supplies K and fallback GR scratch. Front's
NativeEmbed::gather_dev calls iq_embed_rows over fixed table bytes; MTP experts
are resident. Attention uses supplied pools/ids/steps and caller-owned scratch;
KV append writes each owner's existing mapped host mirror directly. This is
separate from main verifier PLE, which MUST NOT be generalized from this audit.
Only pair2366..2383 references PLE: optional CPU ngram prefetch_rows hints for
the future verifier, after submissions. These are not a completion-service
callback for MTP. Actual retained S114 log17 reports requested0/wired0/
slot_ple_ready0. Thus proposed blocking joins do not strand a source-visible
MTP PLE GPU request waiting for this later loop. If a future path adds such a
request, these waits need a progress pump and this source argument expires.

Mapped staging, seq_cst fence, final h_out_ reads/probabilities/counters and
scratch stay intact. The CPU-only PLE lambda/state moves before submissions;
OFF callbacks stay in their original output loop. ON issues the initial hints
AFTER both forwards are submitted and BEFORE first joins, then hints each
owner's completed head token BEFORE the next depth. It skips later duplicate
callbacks, preserving the same per-owner token order/count and both-owner
serial host access. No unnecessary progress pump. Complete pair timing must
include every hint, host join and tail; no wait or prefetch cost is timed away.
No CPU/peer clock join or idle inference is made.

Lifetime/cache/error: mode latched once at init; caches[T]/[j] cannot switch
record/replay queue later. Existing same-device/head/vocab/geometry/owner/
forcing/coupled/rebound-source/equal-step guards stand. Distinct owner queues
are checked only for ON. Solo/unequal-step/one-active fallback stays exactly
in generate9480..9510 and original MtpDrafter methods. Captures/launches failing
return false, never redraft after possible KV writes. ON joins use
wait_and_throw; errors drain BOTH owner queues AND each owned side branch;
ON pre-staging readiness explicitly calls wait_and_throw on each owner's cs_
AND distinct side_; it is not the noexcept cleanup drain. Original idle waits
only cs_ with wait(), not side_ or asynchronous error delivery. Its DPCT macro
catches ordinary std::exceptions and returns false; the outer fatal catch does
not make every ordinary idle error fatal. Pair
teardown drains before deleting graphs/scratch; generate4133..4139 declares
pair after drafters so it dies first. Owners/queues are borrowed, never freed
by this patch. OFF graph/call ordering and arithmetic remain the original;
q selection resolves to cs_, drain helper is no-op, original launch arm runs.

VERDICT -> SOURCE REVIEW ONLY, not build/GPU GO. Smallest strict oracle must
exercise actual component graphs on A/B owner queues + real paired tail,
fresh A/B/A input replays at depth2, unequal effective T and multiple cache
keys, consumed intermediates/sample/KV and exact full tokens/probabilities vs
single-queue control, read-only/guard checks, solo/unequal-step fallback, and
injected launch-failure/drain if feasible without damaging context. Include
cold/pending optional-PLE prefetch (not just warmed prefetchOFF), plus both
shared-branch settings if claiming support for both; preserve source guards.
No numerical proof follows from unchanged arithmetic under new concurrency.
Only after strictness/health/cleanup gates can finite completed-pair timing
answer whether hardware overlap beats added host joins/contention and changed
prefetch timing. No new explicit owner arena, IPC, heads, graph API, counters or test framework.
A previously unused owner queue may materialize existing lazy HC scratch.

V2 error/readiness correction -> Actual source consultations reported by root:
claude-opus-5-5 (56.542s) and claude-fable-5-1 (92.956s), both SOURCE_DESIGN_GO
with required corrections. Root read both; this is a narrow source follow-up,
not a new model consultation or runtime result. The existing SYCL exception
handler remains first, preserving OFF behavior. A new outer std::exception
handler rethrows when OFF; ON drains both owners/branches BEFORE allocating
its error string and returns false. The existing SYCL handler also performs
ON drain before error-string assignment (no-op OFF). Completed tail missing
output includes exact owner index and depth. Non-std exceptions are not newly
claimed handled. Header/device routines/kernel arguments are unchanged from V1.

Bounded static audit -> See ../mtp-pair-owner-queues-s127-static-audit/findings.md
and its frozen manifest670851cf718aacc75c2a59c05769c5aca7e2ed5b004d89fb9fd48726ed78865e.
No unkeyed mutable GLOBAL DEVICE SCRATCH blocker was found on the audited
head-free paths; this is not a runtime alias/concurrency proof. Existing
fused_gr down_partials and slice_partials maps are keyed by queue pointer and
mutex-protected. First use of an unseen queue can add40960+829440=870400B
(850KiB, source-default constants); earlier B solo/prefill may already have
initialized them. No no-allocation claim or measured memory delta follows.
These static maps retain allocations/process-lifetime keys, with no erase/free
here; owner queues and captured pointers must stay live, without queue-address
reuse while users exist. A lookup table may also initialize lazily per device;
it is immutable after initialization, not owner working scratch. Grouped state
is host atomic diagnostics/selectors, while its GPU scratch is caller-owned.
The legacy unkeyed QSA host-scalar step buffer is not reached by MTP's explicit
*_step/*_steps arguments. Audit excludes arbitrary verifier/legacy callers and
future feature paths. Existing read-only registries/selectors must remain fixed
during overlapping replay. Any failure still requires root health/recovery
judgment, never a second draft or a silent scratch reset.
