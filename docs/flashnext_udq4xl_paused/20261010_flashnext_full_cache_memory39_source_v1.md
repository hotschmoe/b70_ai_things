# Full-cache logical memory and native lifetime successor39

CONFIG -> Immutable source38 engine548d978e and source plan8c8cac3a are a
bounded main/admission diagnostic checkpoint. New patch0039 adds one shared
host-only header; resulting source closure66/30/39/8/6. It inherits source38
default-OFF and explicit <=6 captured actual RID selection, unchanged raw64MiB
and graph16 bounds. No cache budget, victim policy, math or dispatch is changed.

COMMAND -> CPU reconstruct all66 files, apply patch0039 to pristine source38
in /tmp, compare exact bytes, test logical arithmetic/overflow, transferred
reuse, estimate overlap, sampled peaks and actual-shaped native lifetime.

RESULT -> 15 CPU controls pass. No native compilation, model payload read,
Docker or GPU operation was executed by the agent. Source38 files remain exact.

Accounting scopes are explicit:

- cache_total = parked entries + reusable KV retained inside ConversationCache;
- owned_pending = local reusable KV outside cache + allocated incoming image;
- observed owned snapshots = cache_total + held restore image + owned_pending;
- logical reservation = cache_total + held + max(incoming estimate, owned_pending).

Incoming estimates include retained KV that moves into the image. Adding both
the estimate and owned_pending would double-charge that storage. Estimates are
reservations, never fabricated allocations. A refused over-budget reservation
may be observed without asserting that the allocation occurred. Arithmetic
fails on uint64 overflow; sampled peaks are per actual cache instance/PID and
recorded event sequence. Peaks mean maximum observed scope samples, not a
continuous allocator high-water mark or whole-process/device peak.
Resource records use current_main_body_command/current_main_body_rid only as
the last main-body context, not ownership of a slot being parked by BSTEP.
Their cache scope is the one stable ConversationCache instance for this served
process; pointer reuse across different object lifetimes is unsupported.

Hooks observe reusable candidate/release/transfer, held take and restore release,
main and later-stage capture before/after actual saves, put before/after, reserve
and reusable release, victim removal, and main-body completion. Active main and
batch slot checkpoint totals and configured chain/transfer limits are separate
resources. Expert live/full arena sizes, segment and mapped-arena bookkeeping
augment source38 actual host slot maps. They are metadata observations, not a
device residency sensor. Source38 actual process VmRSS remains distinct from
snapshot capacities and expert arenas; these scopes must not be added to RSS.

Native later batch lifetime has explicit actual RID/slot-generation records:
BSTEP_before, BT_emitted, BDONE_emitted and BSTOP_applied. DONE_emitted and
BADM_emitted cover the main leg and its actual continuation decision. In this
grammar a BSTEP window may emit multiple accepted BT candidates. Each raises
the generated count by exactly one in the same window; the next BSTEP requires
at least one prior BT and a strictly newer actual window. BDONE requires a
completed emitting window and the final exact generated count. Empty windows,
changed ownership, count gaps, duplicate windows and postterminal BT fail closed.
The initial single-BT proposal is preserved under
/tmp/source39-initial-review-1ff324-20261010; no native experiment used it.
In this
successor the inherited checkpoint label becomes main_body_complete. Native
BDONE does not itself prove an HTTP client received terminal content. Runtime
V2 must join actual frontend/client disconnect, finish/DONE and cancellation
receipts, complete raw49/current-source observations and owned parent teardown.
STOP delivery, API cancellation and completed snapshot disposal are separate.

Selected direct GEN uses a unique actual RID initially. Repeated same-RID GEN
requires its actual intervening BGEN admission/later/cancel solo lineage;
unsupported repetitions fail the existing source38 ledger. No synthetic slot,
event, protocol, captured state or reuse is fabricated to admit a repeat.

VERDICT -> Reviewable SOURCE/CPU proposal only. Fresh source39 compile/eightABI,
oracle, upload390, whole4/current pages, new strong baseline and matched OFF/ON
native requalification are mandatory. Source38/source37 runtime authority is
not transferred. No full-cache, full-model, physical-free, HTTP terminal or
speed qualification follows from this proposal.

Remaining required physical evidence is explicit: actual host/process peak and
minimum available RAM around every actor phase; GPU/device allocations and
residency on each card; transient transfer/clone buffers; expert/host-RAM tradeoff
across matched configurations and complete history/eviction/cancel/restart/stale
families. The resource rows expose scopes for those joins but do not qualify
missing physical measurements. Full runtime V2 and its fresh49 controls remain
unfinished. No remaining scenario is reduced to fit six selected raw requests;
complete independently recreated families with preregistered capture rosters
must retain all unselected real work, lifecycle, victim and memory observations.
