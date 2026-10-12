# Source52 native two-stream cache probe

CONFIG -> One existing pair24/24 source52 native process, original UD-Q4_K_XL
weights, current accepted P2048 A and authentic divergent B (2023 tokens),
shared positive pin1536, ctx4096, pref512, native batch2/group1, no MTP,
static no-borrow, cache3, no RAM conversation parking, mirror-stage-T1 OFF,
copy0, grouped/ESIMD OFF, existing two-stage expert mirrors unchanged.
The existing full-state chain must admit its original 8 GiB chain and
512 MiB per-transfer limits without reducing the requested two slots.
H36, PTI and UR trace are OFF. Existing bounded SBF/FC38 diagnostics are ON.

One process receives six greedy, natural-EOS, cap8 requests:

1. Serial GEN A fresh1, RID1.
2. Serial GEN B fresh1, RID2.
3. Cold BGEN A slot0 fresh1, RID3.
4. Cold BGEN B slot1 fresh1, RID4 (sent before draining RID3).
5. Cached BGEN B slot0 fresh0, RID5.
6. Cached BGEN A slot1 fresh0, RID6 (sent before draining RID5).

All requests use ckpt1/pin1536. Expected reuse is 0,0,0,0,1536,1536.
Expected new prompt rows are 2048,2023,2048,2023,487,512. The owner swap
makes a retained wrong suffix observable. Whole generated histories must match
the corresponding serial control and cold cohort. Each paired cohort must
complete an actual rows2/mask3 native event; queue admission alone cannot pass.
Native prefill admission is serial. This does not claim concurrent prefill.

Current source already provides strict native2/4/6 batching, public fresh/pin
and complete-state chain transfers. No engine patch/rebuild is needed.
The scalar mirror1 optimization requires batch0 and is intentionally OFF.

The existing SBF observer selects exactly six RIDs. It captures all48 residuals
and full248320 logits for each direct serial first input, each batch admission,
and each batch first later decode input. Its original 64 MiB static process
quota is unchanged; the exact requested raw extent is 29,593,600 bytes.
Both stages' immutable graph frame/epoch/row/current-owner association is
required. Serial/cold/hit first arrays and cold/hit later arrays must be bitwise
equal, finite, and captured full heads must select the actual greedy token.
This observes two mathematical input positions per batch request, not every
decode position or complete family qualification.

COMMAND -> /tmp/b70-source52-native-batch2-cache-v1-launch-agent.json contains
the exact root runtime command, image/SDK/corpus/fixture pins and environment.
Root runs through the actor's existing gpu-run pair owner. The inherited owner
keeps source/all8 binary admission, whole-experiment pre/post complete4 scans,
active model page/stat checks, per-phase health/kernel journals, normal QUIT,
merged-pipe EOF/hash/quota proof and owned retirement/removal. Failure must close
the same owner and post guards; no success is inferred from partial cohorts.

Public read-only close, after successful producer terminal:

  PYTHONPATH=strata/flash-next python3 strata/flash-next/validate_source52_batch2_cache_v1.py --root /mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f17-source37-20261010/source52-native-batch2-cache-v1-run-v1 --output NEW_EXTERNAL_OUTPUT.json

The new reader ports exactly one former serial protocol-partition assumption
into an explicit co-admission partition. All original command, source, saved
identity, health, kernel, page, stdout quota, launch, EOF and teardown predicates
remain. Original raw observations and exact native send chronology are retained.
No extra public whole-model read is added.

RESULT -> Actual52 genuine construction passed, exactly one phase and six
requests. Two tiny CPU transport controls passed: both BGEN sends precede drain,
and a stale slot generation is rejected. Private reader AST construction passed.
No GPU, model payload access, build or native CPU model process was performed
by the source author. Runtime/source admission and all mathematical gates still
require actual root execution and independent saved-evidence reading.

VERDICT -> Runnable bounded current52 native two-stream shared/divergent cache
candidate. It cannot qualify HTTP/API behavior, cancellation, eviction, stale
model identity, restart, native4/6, clean latency, shelf promotion, original full
model math, or the full goal. Instrumented receive times are diagnostic only.
