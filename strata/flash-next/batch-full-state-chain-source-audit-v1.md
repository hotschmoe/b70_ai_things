# Batch full-state chain source audit and increment 0025

CONFIG -> corrected20 + final21 +22-v2 + frozen24-v2. This source audit uses
owned overlay /mnt/vm_8tb/b70/build/strata-batch-observer24v2-ozauc3tl.
Final23 independently applies without cache-region edits. No consumed build,
source, controller, patch or model bytes change during this audit.

COMMAND -> inspect sycl/src/program/generate.cpp BSlot/copy_to_slot/
copy_from_slot/slot_source/periodic/root/turn checkpoint/admission paths, plus
sycl/src/core/conversation_state.cpp and conversation_snapshot.cpp and shared
include/strata/core/conversation_cache.hpp and program/conv_cache.hpp.

RESULT -> these missing paths are source-anchored:

- generate.cpp11660..11682 selects only the deepest checkpoint for new BSlot.
  Root, earlier shared boundaries and an explicit pin are lost at admission.
- generate.cpp9784..9796 normally imports only the selected checkpoint, dropping
  valid older members of the donor chain. BYIELD is a partial exception.
- generate.cpp9720 skips active slots completely. An immutable root of a
  conversation that is still decoding cannot serve another conversation.
- generate.cpp10442..10506 stops prefill at n-1 but saves there only when it also
  matches another boundary/periodic rule. Short repeated prompts may have no
  usable leaf; live ids after first target consumption have length n and cannot
  mount a repeated n-token prompt because the final token must run again.
- generate.cpp8800..8834 restores running checkpoint state before restoring each
  QSA KV image. conversation_snapshot.cpp includes the moving pooled spare row.
  conversation_state.cpp193..215 checkpoint restore reconstructs that spare from
  checkpoint dead state. Later KV restore can overwrite it with the donor's
  later moving/completed row. Whole-session restore in conversation_state.cpp
  357..367 correctly restores KV first and running checkpoint last.
- Slot host checkpoint copies have count bounds through prompt_cache, but are
  outside ConversationCache.bytes(). Blindly multiplying full chains into six
  slots needs an explicit additional host-memory budget, transient admission
  bounds and real capacities rather than a count-only claim.
- Public pin and fresh requests still reject batch. This increment will retain
  those guards. A cache-bypass reset/control plus public pin semantics still
  require subsequent actual safe-state qualification; ckpt=0 is not currently
  a true bypass of batch cache lookup and must not stand in for fullfresh.

VERDICT -> batch1/2/4/6/cache qualification remains incomplete. A dedicated
opt-in0025 must fix actual full-chain retention, source selection, transfer
ownership/order and leaf generation, then compiler/device tests must qualify it.
These gaps are not evidence that the frozen serial source/results are invalid.
The pooled-spare order finding is a source defect, not a claimed explanation for
an observed numerical drift or model-pagecache incident.

Complete state means every stage's GDN recurrence and conv history, PLE F32
history plus last-two-token window, QSA tail/dead/block position and reconstructed
moving pooled row, authoritative KV/completed indexer rows, and committed token
identity. Checkpoint payloads omit immutable weights and per-token residual/
projection scratch. Under strict native HC, every layer's FFN HC write is immediate,
and every target window/commit completes before source transfer. The next token
rebuilds residual scratch from its embedding; scratch cannot be mistaken for
persistent prefix state. These claims still need actual cached/fresh raw-logit and
per-slot residual qualification, not just RESUME counters.

The next source design is default-off STRATA_BATCH_FULL_STATE_CHAIN=1, restricted
to exact strict native batch2/4/6/group1 text, static no-MTP/non-pipelined scope.
It will preserve every complete token-prefix checkpoint within the configured
count and host-byte budgets, including root and pinned metadata, between main and
slots. Active slots can donate only immutable saved checkpoints, never their
mutable live running state. All stages quiesce; selected checkpoints and source/
target geometry are prevalidated before writes. Source slot RID/generation is
snapshotted and checked around transfer. KV copies precede the authoritative
running checkpoint restore. Partial transfer cannot be advertised as reuse.

Each completed admission adds an exact n-1 prompt leaf without changing token
IDs or rewriting a template. The root/shared/turn/leaf chain follows existing
root-pinned/LRU retention. Global checkpoint-byte capacity covers all main/slot
chains plus preflighted transient copies; KV transfer has a separate bounded
single-layer image allowance and physical RAM admission. Unsupported capacity
fails explicitly, never silently reduces requested slots/checkpoint geometry.

CPU controls must reconstruct exact tracked source and exercise real chain
policy and extracted actual restore/transfer bodies with host mocks. In particular,
a donor KV spare999 restored over checkpoint dead111 must end111 with KV-first/
state-last, while original ordering ends999. Controls must also reject partial
stage payloads, wrong tokens, stale generations, mutable active-live donors,
wrong stage carve/format, budget failure, and partial transfer before publication.
GPU/math/cache/fairness qualification remains the parent's responsibility.
