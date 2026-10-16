# MTP-only hardened parallel argmax design, 2026-10-16

CONFIG -> SOURCE ONLY; no patch, source generation, main edit, build, GPU or
model invocation. S113 no-build prefill remains first in the execution queue.
Parent reports S11248 strict cases PASS and two nested ABBA two-owner completed
host-chain medians about165.5->116.4us. This is not isolated argmax GPU service,
serving gain or a budget to multiply by historical cycle counts.

COMMAND -> Read pinned MTP/caller/argmax/sampler/private-v2 source and bounded
Ninja dependency stanza. No object/archive/executable/model/trace payload read.

RESULT -> Recommend isolated unique header included ONLY by sycl/src/core/mtp.cpp.

| Alternative | Default-path exposure | Build/maintenance tradeoff |
| --- | --- | --- |
| A: private MTP-only header with exact reviewed S112 v2 helper/body/launch, unique namespace/class | sampler.dp.cpp and verify_kernels.dp.cpp remain byte untouched; old dispatch and kernel IDs retained | One bounded clone, about8KB including migrated comments, one include consumer. Smallest promotion surface; explicit source-projection checks needed to prevent drift. |
| B: share/parameterize original argmax helper, add hardened template/new API | Can source-project false branch to original statements, but changes the shared TU/helper source and requires new default-image proof | Less duplication long-term; verify TU + MTP TU/public-header dependencies, additional archive surface. No present serving gain justifies this refactor now. |

A is an opt-in measured-integration proposal, not a permanent separate backend.
Use a local SYCL-only header such as sycl/src/core/mtp_argmax.hpp, an internal
namespace and NEW registered kernel class (not old argmax_rows_kernel_d5bcf6
or S112 private class). Copy v2 arithmetic, last-block election/acquire-before-
last-publication, existing fences/barriers, reset, shape mapping, SG32,256threads
and use_root_sync properties exactly. No public API/class layout changes.
Only helper namespace/name and registered class change from the reviewed leaf.
Keep a source inverse/projection to the pinned private body. Generic sharing
can be considered upstream after a meaningful production result, separately.

Default-path proof/design:
- record_post_head(mtpcpp1138..1158) fixes T=1. Preserve its FIRST existing
  if(argmax_rows_wanted()) and old argmax call exactly. Do not change the
  major==8 predicate (verify_kernels.cpp1586..1600) or global sampler flags.
- Inside its current ELSE, keep SamplerParams construction/greedy=true/temp0
  statements. Nested opt-in replaces ONLY that local sample_tokens call for
  full head(dhead_==nullptr), nv==248320,T1, noncoupled and old-wanted=false.
  Everything else executes the original sample_tokens statement unchanged.
  This is the locally unpenalized greedy primitive (nullptr history,len0;
  default penalty_last_n0/repeat1/frequency0/presence0). It does not infer
  every outer request is globally greedy: coupled requests have their own path.
- Preserve subsequent row_top_prob_split/row_top_prob, map_ids and every
  mtp_select argument/order. Do not combine probability omission S111.
- One process-static default-OFF flag, proposed STRATA_MTP_PARALLEL_ARGMAX,
  fixed before first capture. Optional once eligible HOST marker, no counters.
  Environment changes cannot alter existing graphs; no new graph-cache key
  or mutable setter. OFF keeps current ordering and registered default names.

Callsite coverage and stride:
- record_rest1124..1135 returns through coupled_draft_sample for coupled_rec_;
  otherwise record_post_head handles solo chain. Pair eligibility2074..2075
  rejects coupled/chain-live, and tail2158 invokes each owner's same method.
- Pair logits are contiguous two full-vocab columns: ownerA(logits2_),
  ownerB(logits2_+nv_). Each new launch scans ONE row from its passed pointer,
  writing its owner's out_ids_[0]; do not pass2rows/shared scratch or j offsets.
- capture_pair_depth2 also calls the same method2223. The immutable flag would
  affect its captures consistently, although future discriminator keeps whole-
  pair graphs OFF. A source change must not silently skip component/solo tails.
- Full layer record_forward790 rejects T!=1. max_t bounds come from load271;
  max_t4 is capacity, not four active argmax rows. nv==248320 excludes zero,
  subset4097 and integer-overflow domains; n+4095 and b*per are safely int here.
  nb=61,per=4071; last chunk bounded by min(n,lo+per). IDs fit int32.

Owner scratch/ordering/lifetime:
- Every load, including shared-weight slots, carves OWN arg_scratch_ at456,
  sized argmax_rows_scratch_bytes(max_t); arena zeroed once at468. Sharing
  dense/head weights does not copy arena/scratch pointers. bind727 allocates
  owner logits; global/slot head binding does not unify scratch.
- New ArgRow must assert counter0/v16/i528,size1040,align4. Check public scratch
  bytes(1)==sizeof(newArgRow) and available max_t rows before capture; layout
  mismatch under opt-in is an explicit startup/bind error, never silent reuse.
  max_t4 already reserves4160 logical bytes (rounded4352 in Bump), no new VRAM.
- T1 uses row0 only. Scratch starts zero and normal last block resets counter0;
 61 increments cannot uint-wrap when starting at0. Persistent scratch reuse,
  not per-launch memset. Failure drains/reports/stops; NEVER zero after failed
  replay, retry, wrap a bad counter or mask a failed numerical gate.
- Bump76..82 aligns each carve256. MTP cs queues505..519 use create_queue(true);
  dpct/device.hpp616..621 selects in-order with this USM build. Pair idles BOTH
  owner queues2272..2273 before its in-order queue reuses owner scratch; chain
  live is rejected. No concurrent same-owner launch or new queue introduced.
- Drafter destructor140..161 waits/deletes graphs before arena free. Pair2000..
 2010 drains/deletes its captures; generate4138 explicitly declares pair after
  slot owners so it dies first. Preserve these lifetimes, queue choice and
  mapped publication. S112 snapshots after completion do not establish an
  early host-poll timing theorem; do not broaden that result.

Build/proof cost, not authorization:
Header A included only by MTP implies planned mtp.cpp.o -> libstrata_engine.a ->
main link. Existing Ninja792..798 already compiles that TU with SYCL/AOT/precise,
so no CMake change or kernel archive rebuild should be needed; actual future
ninja-n plan must confirm. New main AOT/host binding must independently match
reviewed candidate arithmetic/resources and preserve old selected sampler/
probability/select images (allow only separately proved metadata/trailer drift).
Do not substitute private symbol identity for actual main launch ownership.
No new private rerun is automatic. Only if root judges build/loop cost justified,
one later fixed-policy matched C2 OFF/ON serving window, unchanged downstream,
exact work/text/capture/marker/normal health, can determine a bounded effect.

VERDICT -> DESIGN A queued, default OFF, root selects after actual consultations.
Parent reports actual claude-opus-5-5(29.882s)/claude-fable-5-1(23.814s) prefer A
but prioritize S113 first. Their agreement is not correctness or speed proof.
Reject counter-reset-after-failure and historical-cycle gain extrapolation.
No patch or integration/build/serving action follows from this document.
