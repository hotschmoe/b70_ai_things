# SYCL full-state prefix reuse: v6 audit and next qualification

CONFIG -> CPU read-only audit of engine v6 at
`/mnt/vm_8tb/b70/build/strata-native-hc-engine-20261009T065210Z-e6m4q9uo/source`,
base fb58e0 plus 0001..0012. Selected original Unsloth UD-Q4_K_XL is unchanged.
COMMAND -> Inspect SYCL generate/session/verify and shared conversation_state,
conversation_cache and server request paths; prepare isolated source increments.
RESULT -> Complete staged in-memory checkpoint and parked snapshot paths exist;
explicit server shared-prefix pin was silently ignored by the SYCL GEN parser.
VERDICT -> Source support is sufficient for a bounded in-memory two-GPU design,
but no cache correctness, isolation, speed or concurrent serving qualification
is established by this audit or by source-upload tests.

## State that actually survives

`src/core/conversation_state.cpp:154` saves the entire owned GDN recurrence and
convolution allocation, PLE normalized-row history, and each owned QSA indexer's
tail, dead/spare row and block position. Restore validates geometry/ownership,
restores these bytes, reconstructs the moving pooled spare row from the dead
row, and derives PLE predecessor token IDs from the checkpoint token prefix.
Completed attention K/V and indexer pooled rows are not copied into every small
checkpoint: they remain in the session arena, valid only on its retained chain.
The request removes checkpoints beyond the rewind or outside the token/image
prefix before any suffix overwrites those cells.

For switching branches/sessions, a SavedConversation includes the live running
checkpoint plus complete retained K/V/scales, completed and spare pooled indexer
rows and checkpoint chain. Global QSA ordinals select the owned layers, not an
assumed dense local ordinal. The SYCL park/restore loops explicitly save, validate
and restore each stage under its OnDevice context. Split/merge carries every
checkpoint's stage parts and pin metadata; a mismatch discards or rejects it.
State restored on stage 0 alone is insufficient; the acceptance fixture must
exercise the actual two-stage path.

Strict native HC completes both residual halves for every layer/token. Its
pending flag is local to the per-token path; the native route does not use the
legacy fused delayed FFN route. No persistent HC history missing from the
checkpoint has been established. Scratch residuals are rebuilt from the next
input token. Qualification must still prove quiescent save/restore boundaries
and full logits; this source observation is not a numerical state proof.

`sycl/src/core/session.cpp:141` resets residual, every owned GDN recurrence/conv,
QSA state/cache/indexer, PLE history and predecessor/token window. The existing
resume=0 route runs this reset and waits on each stage stream. Terminal commit
wait already recursively completes all stages before finished-session parking.

## Identity, session semantics and remaining limits

In-process matching uses exact token and image prefixes and control-vector
mode; source/model/precision configuration is fixed by the process. API patch
0007 binds tokenizer/model/artifact/runtime identity and rejects stale host
session contracts. It injects identity into disk cache keys. The engine's
SAVE/RESTORE command currently rejects layer split and batch: two-GPU disk
persistence remains unsupported. It is optional for the current in-memory goal.
Message-boundary extra checkpoints are explicitly disabled for multi-GPU, but
root/turn/checkpoint chains and full parked snapshots are not disabled.

There is no dedicated tenant ID in the native cache match. Sharing an identical
prefix may be safe math, but authorization for private cross-tenant caching is
an API policy question. Fixtures must use independent server sessions and
verify divergent inputs never match by session ID alone. Root/turn tokenization
must come from the exact exported tokenizer/template, not guessed IDs.

RESUME and DONE report selected candidate reuse. STRATA_CKPT_REREAD only applies
to a non-live checkpoint restore; it does not force fresh work on live or all
parked paths. Where it applies, it resets every stage but retains the old resume
counter and forces batched prompt geometry. `ckpt=0` also still reuses matching
prefixes. Neither is an authoritative uncached negative control. Existing
STATE_HASH only sees the main session and omits block-position from its printed
aggregate; it cannot establish complete staged state equality.

## Separate source increments

0014 adds an armed, host-only completed-work ledger. Default off, at most 64
armed requests, no extra GPU buffers, copies or waits. It records exact request
ID hash, selected resume versus actual read_from, completed evaluated prompt
spans, generated/decoder rows, stage count and bounded cache bytes/entries and
evictions. Supported scope is serial text, no MTP or pipeline. A failed/cancelled
prefill may have done partial device work before returning failure: completed
span counters are lower bounds for that case, not a claim of zero work.

0015 implements explicit `pin=N`. It rejects malformed/out-of-scope requests
before state changes, requires cache capacity >=3 and 0<N<prompt-1, splits the
normal prompt at that absolute boundary and uses existing complete staged
checkpoint_save. A newly saved pin is protected during insertion/eviction;
only one checkpoint is explicitly pinned per chain. Root plus pin plus rotating
leaf use the existing five-argument bounded policy. Parked images use the
existing RAM byte budget: when only pins remain, parking skips rather than
allocating without bound. If a resumed chain has no requested earlier boundary,
the request explicitly fails instead of pretending it pinned state. Start
fresh to establish that boundary. No pin key leaves the usual math/cuts intact.

0016 implements `fresh=1` on serial text requests. It bypasses live, checkpoint,
inactive-slot and parked candidate selection, asserts resume=0 and executes the
existing all-stage session_zero/wait path. It preserves normal root/turn/pin
cuts and fresh evaluation includes the final prompt token in verifier row 0.
It does not clear other parked sessions, so a subsequent non-fresh request can
still reuse them. Absent/fresh=0 retains existing behavior. Source preparation
is not qualification; compare against a newly started process too.

Patches apply in order after 0013, with ordinary offset-only hunks. Use a new
immutable engine plan and source/build generation; do not alter v6 evidence or
frozen C1 controller. Direct native GEN fixtures can first use fresh/pin without
changing API behavior. Exposing fresh through the API needs a later explicit
request-contract update and provenance, not silent unknown-key acceptance.

## First correctness fixture and acceptance

1. Pin all image/source/model/tokenizer/launch identities. Use one card then
   explicit/static 24/24 two-stage with no MTP, adaptation or prefill borrowing.
   Keep expert placement, KV precision, graph mode, capacity, turn/root IDs,
   prefill chunk size and root/pin/turn cut geometry fixed.
2. In a newly started process, establish exact token prefix P with a suffix A
   and explicit pin=len(P). Choose boundaries around a QSA indexer block and
   prefill chunk (below/on/above), and PLE predecessor history of at least two
   tokens. Preserve actual token IDs, not only rendered text.
3. For every target P+B, run fresh=1 and the cached target under ABBA ordering.
   Add a second fresh-process reference with the same cuts. Require fresh
   actual_reused=0 and evaluated_prompt_rows=prompt_tokens. For a hit require
   actual_reused=len(P) and evaluated_prompt_rows=prompt_tokens-len(P), including
   the first verifier's final prompt token. RESUME alone is not a gate.
4. Compare all first-generated raw F32 vocabulary logits from 0013, exact greedy
   generated token IDs and natural finish; declare bitwise equality for these
   matched fixed-geometry runs. Any difference fails this initial contract and
   triggers first divergent layer/state diagnosis, not a posthoc wider epsilon.
   Preserve finite counts, max abs error, RMSE, top-token margin and raw arrays.
5. Exercise root-only/turn checkpoint, explicit pin, live continuation and parked
   branch restore separately; alternate independent sessions A/B, divergent
   suffixes and prefixes differing by one early token, same prefix across new
   sessions, longer/shorter continuations, replacement pin and absent boundary.
   Require wrong token/image/runtime identity rejection or cold reconstruction.
6. Force bounded checkpoint and RAM snapshot eviction. Confirm bytes never
   exceed budget and evicted requests reevaluate with fresh reference equality.
   Pins can make new parking skip; that is a bounded miss, not an unlimited hit.
7. Cancel during prefill and decode, then issue same target and unrelated fresh
   target. Compare to fresh process, require no stale state publication and
   healthy teardown. Do not use cancellation's incomplete span counters for an
   exact-token-work claim until partial-work instrumentation is implemented.
8. Only after serial gates pass, qualify real one/two/four streams and six as
   bounded stress with separate supported batch instrumentation. Current new
   serial pin/fresh/ledger explicitly reject batch: small-concurrency coverage
   remains required work, not implied by this fixture.

Diagnostic first-logit/layer copies and state hash runs cannot supply clean
latency. After correctness gates, run matched diagnostic-off ABBA hit/miss
requests and report client/server P50/P95 TTFT and token gaps, cache bytes and
expert residency before/after, and decode interference from long prefill.
Follow docs/20261009_flashnext_latency_methodology.md: queue/submission/device/
wait/transfer/cache/sampling/stream intervals need ownership and valid timestamp
coverage without added barriers. Lifecycle and post-health remain mandatory.

CONFIG -> Apply 0013 then 0014/0015/0016 to a new minimal source overlay.
COMMAND -> Pinned-image icpx, actual v6 defines/includes/SYCL device-code flags,
O1 object compilation, network none, no device mounts or runtime execution.
RESULT -> All patches applied; complete generate.cpp host/device object compiled
exit 0 in 43.28 s. Standalone 0014..0016 object also compiled exit 0 in 37.92 s.
VERDICT -> Source application and compilation pass. Full rebuild, flag-off
equivalence, fresh reset and numerical cache fixture remain unqualified.
Receipts are in /mnt/vm_8tb/b70/build/strata-prefix-combined-source-20261009/
and /mnt/vm_8tb/b70/build/strata-prefix-fresh-source-20261009/.

CPU ledger/retention controls also PASS: reread candidate versus actual reused,
completed prompt/decoder/stage row totals, 64-request arming cap, default off and
missing ARM, and actual pinned eviction policy. Fresh reconstruction of frozen
0013 SHA5589910e plus these increments exactly matches all eight combined
overlay files. These controls do not test GPU reset or numerical state reuse.
