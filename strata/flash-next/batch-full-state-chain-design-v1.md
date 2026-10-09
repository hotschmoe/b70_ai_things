# 0025 bounded complete batch prefix chain: CPU source READY

CONFIG -> actual compiled20 source ledger, frozen21,22-v2,24-v2 and separate0025.
Final23 independently applies between21 and22-v2 without conflict. Existing live
source/builds/controllers and frozen patches are unchanged. No model payload is
read for this work. The parent's source/model identity and GPU qualification gates
remain mandatory before any inference result can be trusted.

COMMAND -> python3 strata/flash-next/test_batch_full_state_chain_cpu_v1.py
reconstructs tracked inputs, verifies every consumed source hash, and host-compiles
the actual new policy plus extracted actual startup, copy_to_slot, copy_from_slot,
park_slot and conversation_checkpoint_restore bodies. Ordinary g++ C++20 in
image39992 uses ASan/UBSan with explicit queue/state/KV transfer mocks. No SDK,
SYCL runtime, devices or model math execute.

RESULT ->22 source/ownership/transfer controls and7 malformed capacity settings
pass, plus startup scope/capacity negatives. The original restore order ends with
pooled spare999/1999; KV-first/state-last ends with checkpoint dead111/211 on two
stage carves. Both recurrence and conv float rows, PLE history/token window,
indexer tail/dead/block position and reconstructed spare restore from the chosen
checkpoint. Source active recurrence is never saved or modified. The same actual
admission/parking body retains the overwritten idle branch in both stage images
before replacing its private target state. A device transfer failure leaves the
imported chain unpublished and returns failure; no inference/reuse is permitted
from the partial destination.

VERDICT -> CPU source READY for full compiler review; no whole-TU SDK build,
actual model/cache numerical or concurrent speed/fairness qualification.

STRATA_BATCH_FULL_STATE_CHAIN=1 opts in. It requires exact native2/4/6 slots,
group1, no-MTP/non-pipelined greedy text, static expert placement/no prefill
borrowing, fixed KV allocation, no CKPT_REREAD and checkpoint capacity2..8.
The flag changes actual cache behavior. Disabled mode preserves existing cache
selection/retention and original transfer order so old source/evidence is intact.
A new research alias/build generation is required when enabling it; the primary
client name remains hotschmoe-dd.

Main-to-slot admission retains every valid checkpoint, including root, pin
metadata, turn and leaf. Slot-to-main restoration imports all valid earlier points
through the selected boundary. Values are owned deep copies; equal token IDs
alone never deduplicate independently computed state bytes. All token IDs and
stage parts are exact, and no template is rewritten.

Active slots can donate immutable saved checkpoints only. Their live GDN/PLE
running state is never captured for a different request. Source RID/generation is
captured at selection, checked before dereferencing the selected pointer, and
checked after staged quiescence and complete transfer. The checkpoint pointer
must belong to that donor's chain. Every retained stage payload, source/target
carve, token identity and KV format is prevalidated before destination writes.
Every stage drains before its authoritative KV prefix is read. KV writes happen
first; checkpoint restore happens last and reconstructs the moving pooled spare.
A stale identity or partial transfer cannot be advertised as a cache hit.

A completed cache-enabled prompt always saves a real leaf at n-1 before its first
target window. That makes an identical n-token prompt eligible to reuse its whole
prefix while re-evaluating the last token. Existing root-pinned/LRU checkpoint
count retention still decides which non-root points survive. An idle cached slot
is parked as a complete per-stage conversation image before overwrite, using the
existing bounded ConversationCache. Refused parking and real eviction counts
are explicit; a budget miss may require a later re-evaluation and cannot count as
successful retention in qualification.

Checkpoint storage is a separate explicit host budget:
STRATA_BATCH_CHAIN_MIB defaults8192, maximum16384. Transfer image capacity is
STRATA_BATCH_TRANSFER_MIB, default512, maximum1024. Positive decimal values are
required. Startup computes a conservative point bound from each real session
carve's GDN/PLE/indexer bytes, maximum token IDs and metadata, with capacity slack.
It reserves (slots+3)*checkpoint_cap+3 such point allowances: all main/slot chains,
owned admission/source clones and bounded asynchronous stage-part transients.
Insufficient capacity fails before READY; requested slots/count/geometry remain
unchanged. Runtime checks actual vector capacities, overflow, physical RAM floor,
three partial stage-point directories and actual KV image capacities. KV image
preflight includes doubled logical payload plus metadata slack. Parked whole
conversation images retain their existing separate byte/entry budget and actual
post-capture RAM admission.

CPU tests prove this source ordering and ownership with synthetic states; they do
not establish actual device transfers, scheduling isolation or model equivalence.
Next steps are final source compilation, new source390/model identity/library ABI
qualification, then same-geometry cached/uncached full heads/all48 residuals and
actual per-stage evaluation spans for repeat/shared/divergent/turn/eviction/cancel
and independent conversations while decode overlaps long prefill.

Public pin=N and fresh=1 batch guards remain unchanged. ckpt=0 is not a genuine
cache-lookup bypass and must not be used as a fake fresh reference. A validated
same-process batch fullfresh/reset control and public pin semantics are mandatory
following work before complete cache qualification. Existing serial controls or
cache-disabled numerical pilots do not substitute for that requirement.

Observer26 separately repairs frozen24 stale admitting/span attribution and adds
post-migration raw observation. It must compose in a fresh source generation.
No change here touches24 observer headers or Prefill observer code. A successful
2/4/6 metadata trace alone cannot qualify full-state cache continuity or fairness.
