# Original publisher MTP: paired serving ownership seams

CONFIG -> Audit of genuine SDK52 target source and the compiled two-stage V4
original publisher adapter. Target24/24, original drafter on GPU1, capacity4096.
Actual controls passed; natural P2048/GEN512 is pending. Neither authorizes
API/cache/concurrent serving. Primary served model must remain hotschmoe-dd,
with a truthful secondary original-publisher-MTP/model/quant/source alias.

COMMAND -> Read source only: SDK52 source/sycl/src/program/generate.cpp,
source/sycl/src/core/mtp.cpp, source/sycl/include/strata/core/mtp.hpp, current
original-mtp53/loaded_main_adapter_pair_v3.hpp, draft_step_pair_v1.hpp and
qsa_state_segment_v2.hpp. No model/provider reads, compiler, GPU or source edits.

RESULT -> Existing hooks are reusable scheduling boundaries, not an already
compatible backend:

- generate.cpp3826-3832: stock batch_mtp refuses a layer split; mtp.cpp271
  refuses native HC. Dropping these guards would not integrate original MTP.
- generate.cpp3974-3992: stock drafter load runs on LAST GPU; slot drafter load
  passes &mtp to share weights. mtp.cpp281-292 aliases dense/experts/tensors
  and disables their ownership; mtp.cpp741-749 similarly shares draft head.
  This is the ownership pattern to port, without its RT/repacked tensor loader.
- generate.cpp7242-7260: private slot hidden-row buffers and binds; current
  stock buffers use current-device allocation/copy. A paired original backend
  must explicitly OnDevice1 for creation, final_R_all row copy and retirement.
- generate.cpp8897-8911 / 8982-8996: solo-to-slot and slot-to-solo draft cache
  transfers currently copy legacy draft KV only. conversation_snapshot.hpp13
  explicitly says draft include_index=false. That cannot represent original
  compress4 indexer state or original hidden recurrence.
- generate.cpp9017-9038: each speculative slot contributes two target rows;
  kVerifyMaxT=8 (include/strata/kernels/verify_kernels.hpp21) permits at most
  four slots per window, rotating larger active populations. Six active slots
  is meaningful but is not six simultaneous two-row slots. Existing optional
  batch observation array at generate9049 has six entries: eight observed
  rows would need a separately correct bounded observer, not an ignored gate.
- generate.cpp9072-9078: per-slot keep=1/2 from actual target pick, EOS, stop,
  budget and context; commit_slot_prefixes is the correct all-stage boundary.
  generate9109-9121 only prepares another draft while sl.active. Original own
  recurrence must catch up committed rows BEFORE terminal cache publication,
  including EOS/cancel/length. It need not propose another token then.
- generate.cpp11837-11861: admission transfers solo state to the slot and seeds
  its proposal. BSlot8755 stores rid/generation, active/stop, ids, position,
  draft_ready and branch checkpoints. BSTOP8788-8813 matches rid/generation.
  New draft ownership must obey the same identity; stale cancellation cannot
  discard or mutate a new owner's banks.

Smallest deliberate refactor (not implemented):

1. Extract OriginalDraftWeights53: explicit GPU1 queue/context, the exact 34
   original tensors, lossless expert concatenation, StepWeights53 and their
   owner. Upload once BEFORE expert cache sizing. Current constructor lines
   25-40 are the weight boundary. Hold a shared immutable owner until every
   private session and captured graph has drained and retired.
2. Extract OriginalDraftSession53: existing lines41-48 allocations, BOTH
   DraftBanks, DraftQsaTransaction, StepScratch53, own committed hidden,
   incoming rows, token/staging, per-session cells/failure/transaction flags
   and graph owner. One admission session plus one per live BSlot; no mutable
   scratch or staging can be shared concurrently. Each bank at capacity4096
   is 8,915,972 bytes; two are 17,831,944 bytes per session, before private
   scratch/hidden/graphs and target slot state. Shared weights stay
   2,775,258,112 bytes, not that amount per slot. Allocate all intended slots
   before sizing BOTH target caches; do not infer available headroom.
3. Use the existing prompt LAST-prefill callback for the admission session;
   copy complete committed draft state on actual admission. For each batch
   verify, copy that slot's actual LAST residual rows into its owned GPU1
   staging before the next verifier overwrites them. Apply only actual keep
   rows and paired target decisions; never call current verify_actual helper
   once per slot (it invokes the global non-slot verifier itself).
4. Add an explicit original-draft checkpoint payload to branch/current/slot/
   parked/persisted snapshots. Required: committed K,V, tail384, dead128,
   pooled(max_cells/4+2)*128, block_pos, committed_cells, committed_hidden
   (10240 F32), accepted-token/prefix identity and capacity/arithmetic/model
   contract. Snapshot only with no tentative transaction after actual drain;
   rebuild working bank from committed and clear draft_ready/pending state.
   To propose immediately after restore, preserve source-owned logits or
   recompute the SAME own head from committed hidden; stale scratch logits
   cannot be used. Existing state class has no reset/export/import API yet.
5. Fresh resets clear both banks/hidden/counters. Shared/divergent restore
   must match the exact TARGET prefix/checkpoint on all stages AND original
   recurrence at that prefix. A latest bank cannot reconstruct an older
   prefix just by reducing cells. Partial prompt/BYIELD, cancelled requests,
   conversation rewind and persisted reload need their actual matching draft
   snapshots; explicitly fail closed until those paths are integrated.

Cache identity: generate.cpp7473-7476 currently fingerprints RT dense.txt,
experts.bin and draft_vocab.bin only when o.mtp is nonempty; 7494 sets the
legacy MTP window. A new original backend must bind the real original GGUF
identity, unchanged 34-tensor dtype/recurrence, original compress4 selection
contract, capacity and actual target source/math/KV/config identity. It must
not call the original sidecar an RT directory or reuse an old cache version.
Current diagnostic argmax is greedy-only; sampled serving needs a separately
correct target acceptance contract, not a claim from greedy CLI parity.

Cancellation/teardown: drain the slot's actual committed work before caching
or resetting it; reject stale RID/generation; invalidate proposals on ownership
change. Finish/cancel must not free rows referenced by active target/draft
commands. Retire private graphs/state under their GPU1 queue first, then shared
weights, then target contexts, retaining the current failure/EOF/health gates.

VERDICT -> No tiny guard removal is sufficient. The weight/session split is
small and concrete; complete original-draft snapshot/restore plus paired
BSlot acceptance/terminal ownership is the blocking serving work. Next proof
must exercise fresh/shared/divergent/rewind/cancel/persisted and true active
1/2/4/bounded6 requests under the actual API and hotschmoe-dd manifest, with
all-stage target and original-draft prefix equality. CLI512 is not that proof.
