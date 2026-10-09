# V5 serial prefix coverage against the serving goal

CONFIG -> read-only audit while parent owns the GPU/basic fixture. V5 controller,
exact token fixture and frozen patches0001..0018 are unchanged. Actual compiled
source is /mnt/vm_8tb/b70/build/strata-native-hc-engine-20261009T085407Z-pzbw4jgp/source,
with successful build receipt and plan SHA
97243b44806418ce81557923405e8b219b33082ebb4a8f3e7b72a903413df08d.
Its sycl/src/program/generate.cpp SHA is
c98a6af356ded6f7212626158d724f7c378988bbef52ef3ac7cac9acba64d37c.

COMMAND -> inspect serial_prefix_qualification_v5.py, serial_prefix_prompt_fixture.py,
actual generate.cpp, conversation_state.cpp, conversation_cache.hpp and consumed
prefix_diagnostic.hpp against docs/20261009_flashnext_goal_prompt.md and
prefix-full-state-source-audit.md. No GPU touch or consumed file/plan edit.

RESULT -> V5 is a useful bounded serial internal-consistency gate. It preserves
complete finite first-logit/all48 residual coverage, actual prompt span accounting,
exact IDs/LP20/natural finish, identity and lifecycle gates. Its report deliberately
keeps complete_prefix_qualified=false (controller line346). The following gaps
must remain explicit even if every V5 group passes.

| Priority | Existing evidence and exact reference | Missing gate | Separate next action |
| --- | --- | --- | --- |
| 1 | Controller lines269-277: nonfresh replay follows prefill STOP, but decode STOP is followed only by unrelated fresh=1 | Fresh reset can mask stale live/checkpoint publication after decode cancellation | Separate six-request cancel-decode replay group below |
| 2 | Token fixture lines23-32: rendered turn_B user question is replaced by turn_A plus assistant-prefix space | No real generated-assistant/new-user continuation or natural live chain comparison | Four-request real continuation group below |
| 3 | Controller lines263-268: global eviction count grows and target returns reused0 | Target admission/victim identity and memory/entry budgets are not asserted; an unadmitted target plus unrelated evictions could satisfy those checks | Keep six-request eviction roster, require keyed admission/eviction telemetry and budget checks |
| 4 | Controller lines255-262: parked A/B share the same system text and differ in one user question | This exercises branches within one native GEN process, not independent HTTP sessions, client identity isolation or private tenant policy | Later API session/branch fixture with stable served ID and exact token traces |
| 5 | PrefixDiagnostic::finish line35 emits stage_prompt_rows=prompt_rows*stages; controller line135 checks the same multiplication | Count is a topology-derived total, not independent per-stage work measurement | Default-off per-stage entered/completed span events at existing stage boundaries |
| 6 | Fixed root/pin/turn boundaries, controller lines242-254; token fixture lines28-34 | No deliberate neighboring QSA indexer/prefill boundaries, changed earlier token, replaced pin, or missing pin refusal | Separate <=6-request groups for each boundary/admission class |

## Cancellation: six armed requests per process

For decode cancellation use separate exact original useful prompts A and B, with
parking disabled and the same fixed geometry/warmups as V5:

1. Fresh A reference, natural completion.
2. Fresh B reference, natural completion.
3. Fresh A, STOP on first emitted T, require actual cancellation.
4. Nonfresh A replay, compare all first logits/all48 residuals/IDs/LP20/finish to1.
5. Nonfresh B replay, compare to2; this tests unrelated state isolation before reset.
6. Fresh B confirmation, compare to2 and5.

Require actual request/position/route/coverage records for all noncancelled rows;
cancelled coverage is explicitly unobserved. Request4/5 must not silently set
fresh=1. A wrong checkpoint/live state must fail numerical comparison, not be
hidden by recovery or relaxed tolerance. Repeat this as a distinct prefill-cancel
group if independent same/unrelated replay after partial prefill is needed.

V5 STOP-on-PP plus cancelled=true does not by itself prove STOP was consumed
inside prefill: PP is emitted at completed progress boundaries (actual
sycl/src/program/generate.cpp lines7706-7710 and10026/10080); the receiver can
lag. Record host-only cancellation phase and last successful stage/span/commit
boundary, without calling partial in-flight work zero. Existing cancellation
publication guard at generate.cpp lines11394-11399 preserves checkpoints taken
while reading and suppresses incomplete-prefill live publication; decode STOP
sets finish=cancel at line11365. Inspecting those guards is not exercising them.

## Real assistant/new-user continuation: four armed requests

1. Run an exact-template seed request fresh, preserving its actual generated
   assistant text/IDs and natural stop; do not insert a canned assistant reply.
2. Render seed messages plus that actual assistant reply and a new useful user
   question through the exact tokenizer/template; run this continuation fresh.
3. Repeat the seed fresh; require exact seed output equality with1.
4. Run the same continuation nonfresh, require the intended live route and the
   exact valid consumed-prefix boundary; compare its outputs/full first logits/
   all48 residuals to2.

The renderer's closing/stop tokens need not already be consumed engine state.
Determine expected reuse from actual committed native tokens and the rendered
continuation, not from decoded-text length or assumed EOS consumption. Preserve
bounded committed-prefix length/digest/token IDs and the generation commit
boundary. Current request telemetry describes GEN input, not the complete
post-generation live chain; add host-only metadata rather than inferring it.
Later separate groups should alternate two real conversations, fork suffixes,
shorten/extend histories and change one early source token. Native GEN gives no
HTTP session or /v1/models claim.

## Eviction: keep six requests, require exact ownership

Existing reference B, independent A/C/D, nonfresh B, fresh B roster stays within
six armed requests. Add default-off host-only events at actual successful cache
admission and removal. Bind process/request, snapshot ID, exact source-prefix
SHA256 and length, stage ranges, retained/checkpoint payload sizes, pin state,
admission result/refusal reason, eviction victim ID and retained budget/slots.
Do not dump weights. Check B was successfully admitted when it was parked,
then B's exact ID was evicted before replay. Global count growth alone is weak.

At every observed mutation and selection/finish assert bytes<=512MiB and
entries<=1 for this declared group, including retained reuse accounting. The
actual ConversationCache budget and bytes() logic is in
include/strata/core/conversation_cache.hpp lines191-203,219-230,261-282,324-325;
generate.cpp estimates/admission checks precede capture at lines7489-7563 and
prints global stored/skipped statistics at7566-7571. Those messages omit exact
victim identity. Also preserve read-only host/cgroup peak memory evidence:
logical retained-byte budget is not total transient allocation or OS headroom.
Admission skipping under budget/pins must be a declared bounded miss, not an
unlimited hit or eviction of an image that never existed.

## Remaining source-state and goal gates

Actual src/core/conversation_state.cpp lines154-191 saves/restores owned GDN,
PLE history and QSA tail/dead/block-position, reconstructs the moving pooled
spare row and derives ple_prev from prefix tokens. Full parked capture/restore
also carries retained KV and pooled indexer rows, lines263-285 and328-355.
The staged caller must continue to own each device's part. Matching48 residuals
and one full vocabulary row is strong output evidence at those boundaries;
0013 still does not observe complete persistent GDN/PLE/QSA/KV state bytes.
If a numerical failure occurs, localize the first divergent layer and inspect
that family's bounded state before adding reset/math switches.

The actual QSA indexer block is4 (include/strata/kernels/qsa.hpp lines95-112).
V5 prefill is64. Deliberately pin below/on/above both boundaries, including PLE
predecessor histories of at least two tokens. Use exact exported IDs and fixed
cut geometry for paired fresh/hit comparisons. Separate groups can cover pin
replacement, invalid absent boundary rejection without state mutation, live
shorter/longer prefixes, one early-token mismatch and root/pin coexistence.

Goal work still includes independent mathematical reference/coherence, stale
runtime/artifact/cache identity rejection on the actual API, hotschmoe-dd first
in /v1/models, production8192 capacity/long prefill, real one/two/four streams
and bounded six-stream stress, plus diagnostic-off latency/fairness. The current
serial fresh/pin diagnostic rejects batch and cannot qualify those paths.

VERDICT -> V5 scope/pass contracts remain unchanged. Propose a NEW controller
version and new immutable token fixture/source plan, not edits to V5. Cancel
nonfresh replay and real continuation can use frozen model math; exact eviction
and committed-prefix/phase provenance require a separately reviewed default-off
host telemetry increment. Pin its patch/source/header/binary/plan hashes, actual
stage bounds and tokenizer/template/runtime identity; requalify same-version
observer off/on equivalence and lifecycle before new attribution. Do not claim
complete prefix or concurrency qualification from the existing bounded groups.
