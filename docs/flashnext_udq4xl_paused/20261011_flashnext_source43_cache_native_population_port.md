# Minimum fresh43 mode1 cache and native population port

CONFIG -> existing V12 is a source40, two-slot cache suite. Its prepared whole
suite has nine independent actors: shared-capture0, independent-capture0,
cancellation-capture0, eviction-capture0/1, history-capture0, stale-capture0/1,
and restart-shared-capture0; plus the separate six-phase batch0 persisted actor.
Every actor has independent fresh49 controls. Eviction retains all14 HTTP calls
and12 armed requests in each complete actor. Up to six selected capture RIDs is
an observation quota across an actor, not six simultaneously active streams.
The mode1 API43 baseline is batch0/FIFO with two overlapping HTTP lifetimes; it
also does not establish native batch2.

COMMAND -> source-only inspection and a deliberately narrow recipe prototype:
source43_cache_port_recipe_v1.recipe(config,capacity,selected_rids) accepts only
native capacities2/4/6. It deliberately ports V12's sole configuration-file read
to an explicit immutable cfg argument and its one batch2 literal to capacity.
All remaining recipe flags/budgets stay unchanged. proposal() records desired
actual populations1/2/4/6 and sets cfg.parallel equal to native capacity, avoiding
the current frontend's explicit batch/parallel conflict. For one active batch
stream, capacity remains2 and the actual cohort has one owner; --batch1 is not a
supported strict native path. Original batch0 GEN/persisted session behavior is
a separate mandatory control. The prototype always sets API_launch_allowed=False.

RESULT -> actual current source43 supports strict native2/4/6 without a kernel
change. Anchors under the current SDK source root:

- sycl/src/program/generate.cpp:3833 requires requested_batch2/4/6, group1,
  greedy text, no MTP and no pipeline windows. Lines3841-3858 require full-state
  chain/public-prefix, prompt-cache2..8, static no-borrow/fixedKV and bounded
  startup/transfer bytes; capacity admission cannot be waived.
- generate.cpp:9001 iterates active slots into one actual batch window;
  line9052 emits per-owner FC39 BSTEP_before with the same window identity.
  Actual N-way proof requires N distinct live slot/RID/generation owners in
  the SAME window, matching BSTEP/BT/BDONE and raw49 stage observations.
  Six RIDs accumulated over time or six HTTP clients waiting on two slots fails.
- serve/server.py:736-739 requires actual INFO batch_slots equal requested
  strict capacity. Lines2277-2280 reject cfg.parallel inconsistent with --batch.
  Thus use explicit --batch N --batch-groups1 and cfg.parallel=N for2/4/6.
- strata/flash-next/full_cache_shared_runtime_v12.py:38 hardcodes batch2 and
  conversation-cache-slots2; lines51-60 bind source40/currentC1401403, counts67/31
  and registry173. These gates must be deliberately replaced by current43
  SDK70/32/43/all8, actual API43 public baseline and explicit current registry
  preservation. No old current-global registry/hash view is acceptable.
- full_cache_shared_suite_v12.py:16-28 preserves all nine actors and both entire
  eviction passes. full_cache_shared_phase_contract_v2.py:13 hardcodes slots2;
  all existing cohorts have at most two actual rows. It cannot qualify4/6 by
  changing plan.slots or selecting more RIDs.
- full_cache_shared_batch0_persisted_v12.py:11-20 is the separate batch0 session
  save/wrong-fingerprint/valid-restore family, with all3 full49 plus fresh49.

The smallest implementation order after the genuine API43 baseline closes:

1. Port the unchanged two-slot whole suite and persisted/fresh49 families to a
   NEW source43 mode1 controller/parent/public routing layer. Retain every
   original raw/text/tokenizer/lifetime/waiter/cancellation/victim/EOF/health/
   full4/three-page gate. Sign the actual new cache configuration and record its
   truthful detailed method/profile; any new alias is a proposal until root
   reviews it. Old V12/source40 results remain historical only.
2. Add a small separate capacity/cohort actor for actual1/2/4/bounded6 using
   current43 SDK and the same mode1 math. Start with native cacheOFF fresh
   requests, then shared-prefix/independent/cancel/survivor and solo comparisons.
   Authenticate and freshly render the retained four/six-case messages with
   the current tokenizer rather than import old source37 execution proof.
   Require exact same-window populations, all49 per consumed prefix and fresh
   source43 solo controls, actual empty-state/owner/counter and normal retirement.
3. Extend the shared/divergent/history cache schedules with newly authenticated
   four/six-row cohorts. Preserve the whole two-slot eviction and persisted
   families; scaling the population is additional evidence, not replacement.
   Keep fixed8GiB chain/512MiB transfer bounds initially and refuse startup
   capacity reduction or overflow. If six cannot fit, report the bounded refusal.

Retained corpus inputs (software messages only, no source37 runtime authority):
batch-numerical-case4-source37-v11.json SHA189f6df8..., four59-token target
prompts; batch-numerical-case6-source37-v11.json SHAa81dbe15..., six59-token
prompts. Both have native diagnostic budgets32 each and cancel_index0. These are
bounded diagnostics, not natural-completion/load qualification. Their already
completed H54 groups6+2/392 comparisons cannot qualify source43, mode1 or six
simultaneously active cache streams.

VERDICT -> no source43 cache family has run in this port. Genuine43 whole9,
persisted/fresh49, native active1/2/4/bounded6, changed-loaded-model refusal,
physical host/device/per-expert residency and transient peaks remain missing.
The minimal port does not alter kernels, quantization, canonical registry,
weights or historical receipts. Root owns every actual model/GPU action.
