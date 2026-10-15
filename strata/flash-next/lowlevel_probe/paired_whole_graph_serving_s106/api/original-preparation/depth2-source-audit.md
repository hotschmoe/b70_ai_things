# Conditional source audit: per-paired-call depth2

This is a bounded SOURCE argument for root review, not an executed counter or
an automatic measured 5*bt_pair claim. The analyzer retains total collected API
counts and rejects unconditional per-call attribution.

1. The fixed config pins --spec4, batch depth2 and two slots. Slot drafters load
   with o.spec and receive set_max_drafts(batch_depth) at generate.cpp4174.
   mtp.hpp51 assigns max_drafts_=k directly. Searching actual production caller
   finds only this slot setter. Calls8939/8941/12967 target the separate global
   mtp object. Copy/restore updates slot prompt length, not max_drafts. Reply
   remaining is applied by batch_keep/slot active; it does not lower this field.
2. steps_of is min(max_t_-1,max_drafts_,st_.max_cells-(p+a)). Here the first two
   terms are3 and2. MTP load inherits the target logical max_cells, and
   qsa_state_init preserves logical max_cells even if physical KV uses a ring.
   Target and slot sessions initialize with o.max_context (fixed200000).
3. The request's n is its full parsed token-ID count, and generation starts
   p=n-1. Every committed emitted token advances logical p once. Admission
   clears BSlot and copies that request's p; each batch carries pos=sl.p and
   a=keep-1 (0..2 because G<=1+depth=3). Bounded requests here are n107/cap64,
   warm cap256. Even a deliberately loose bound p+a<107+256+8=371 leaves far
   more than2 logical cells. Successful native prompt/usage/cache counts and
   full request records are required; this is not a proof for arbitrary cached
   long conversations or request mutation. EOS/length completion removes a slot
   from active drafts; output remaining does not truncate the slot setter.
4. Therefore, IF actual runtime config/request invariants and exact built source
   identity hold, every successful paired call in these bounded windows has
   max_steps2. Effective cache key coordinates still depend on acceptance and
   lie in1..3; a once-marker observes only the first key, not every key/cache hit.

Even accepting this source proof does not classify every OTHER main L0 call:
profiled prefill/admission, graph finalization, slot copies, solo windows and
remaining setup/runtime activity all stay in the host summary. A five-times
paired population is a SOURCE contribution hypothesis for a matched difference;
only actual API-family totals and reconciliation can establish the observed
main submission effect. Do not equate host API counts with GPU operations,
physical wait time, or uninstrumented latency. Root decides whether this proof
is sufficient before any numerical per-call attribution; no counter code added.

Exact narrow snippets and source hashes are in source-audit.txt/dependencies.json.
