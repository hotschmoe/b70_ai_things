# DD feature map against stock v0.2.16

Static inventory, 2026-11-19. Upstream 58ee3227720629ff2e41d640165ebe89c880dfef; night e753e9cbd7d38dac1a7851c75ef6fc0ae016d5c0. Classification is a porting plan, not proof of runtime equivalence. Same-name controls require validation. The opt-in carries listed below compile as individual objects; assembled runtime validation is pending.

## Core configuration

| DD feature | Classification and evidence |
| --- | --- |
| Two-card expert parallel, --expert-parallel-device 1 | PORTED: stock lacks EP; the base port loaded and passed mirrored verification. Night carries require assembled validation. |
| Four streams, --batch/parallel 4 | UPSTREAM: batching remains in SYCL generate.cpp; verify startup and concurrent serving. |
| Batch MTP depth 1 and solo --spec 4 | NEEDS PORT for two cards: stock has --batch-mtp on one GPU but generate.cpp:4314-4318 explicitly disables it for layer split or helper. Single-stream MTP exists. |
| --max-context 200000, --kv int8, --kv-resident 65536 | UPSTREAM: options and RAM streaming retained; four-slot capacity and restore need live validation. |
| --conversation-cache-mib 32768 --conversation-cache-slots 4 | UPSTREAM: options and split parking exist; slot-preservation behavior must be checked against #2526. |
| IQ3 native GGUF, PLE, --stream-experts, expert-cache, profile, reserve, prefill, no-prefill-borrow, MTP draft vocabulary | UPSTREAM options exist; old pack/MTP compatibility remains untested. Stock profile and vocabulary files now reside under terminal/data. |
| Vision | UPSTREAM Python serving and separate vision encoder need compatibility and image test. |
| API names and server config | Server feature, preserve hotschmoe-dd first and test /v1/models; use :18134. |

## Environment inventory

UPSTREAM = control exists in stock; equivalence and setting-specific behavior are unverified. NEEDS PORT = enabled old control absent, with no proven equivalent; retained only if useful/required after S1. OBSOLETE = disabled absent control or old diagnostic; no claim about a replacement. An old name present only in comments is not counted as a control.

| Variable | DD value | Classification |
| --- | --- | --- |
| STRATA_EP_PREFILL_REDUCE | 1 | PORTED: opt-in carry compiles; assembled runtime validation pending |
| STRATA_OWNED_PRICE | exact | UPSTREAM: control exists |
| STRATA_EP_PREFILL_PUSH | 1 | PORTED: opt-in carry compiles; assembled runtime validation pending |
| STRATA_BATCH_MTP | 1 | UPSTREAM: control exists; disabled for stock multi-GPU layer split |
| STRATA_BATCH_MTP_DEPTH | 1 | UPSTREAM EQUIVALENT: one proposal is upstream default; depth control also ported |
| STRATA_BATCH_MTP_PAIR_HEAD | 1 | PORTED: four-slot paired subset head; preliminary EP run confirmed pairing |
| STRATA_BATCH_PREFILL_QUANTUM | 0 | OBSOLETE: disabled or removed; no active stock control |
| STRATA_PREFILL_FIRST | 0 | UPSTREAM: control exists; zero explicitly disables upstream default of 256 |
| STRATA_BATCH_MTP_PLE_PREFETCH_DIAG | 1 | OBSOLETE: old diagnostic; no active stock control |
| STRATA_BATCH_MTP_PLE_PREFETCH | 0 | OBSOLETE: disabled or removed; no active stock control |
| STRATA_TRACE | 1 | UPSTREAM: control exists |
| STRATA_MMVQ_WIDE | 1 | UPSTREAM: control exists |
| STRATA_MMVQ_WIDE_MIN | 0 | UPSTREAM: control exists |
| STRATA_MMVQ_WIDE_MAXC | 8 | UPSTREAM: control exists |
| STRATA_Q6_PACKED | 0 | UPSTREAM: control exists |
| STRATA_MMVQ_WIDE_K | 1 | UPSTREAM: control exists |
| STRATA_MMVQ_WIDE_K_RPW | 1 | UPSTREAM: control exists |
| STRATA_HEAD_Q5K_RPW2 | 0 | OBSOLETE: disabled or removed; no active stock control |
| STRATA_MMVQ_WIDE_RPW | 1 | UPSTREAM: control exists |
| STRATA_MMVQ_A2 | 1 | UPSTREAM: control exists |
| STRATA_MMVQ_PF | 0 | OBSOLETE: disabled or removed; no active stock control |
| STRATA_Q6_CENTER | 1 | UPSTREAM: q6_sub32 uses equivalent arithmetic unconditionally |
| STRATA_QSA_PREFILL_TILED | 1 | UPSTREAM CANDIDATE: SELECT_GEMM on Battlemage handles same regime; floating-point path differs |
| STRATA_ATTN_VALUE_PIPELINE | 1 | PORTED: opt-in prefill value pipeline; full build and runtime validation pending |
| STRATA_PROMPT_ATTN_XMX | 0 | UPSTREAM: control exists |
| STRATA_KV_ROT | 0 | UPSTREAM: control exists |
| STRATA_EXPERT_SPLIT | 0 | UPSTREAM: control exists |
| STRATA_GU_LANES | 8 | UPSTREAM: control exists |
| STRATA_IQ2XS_MULTI | 1 | PORTED: opt-in IQ2_XS multi-entry dispatch; compiled, runtime validation pending |
| STRATA_BATCH_MTP_PAIR_GRAPH | 1 | PORTED BUT INACTIVE: pair graph requires depth 2; DD depth is 1 |
| STRATA_Q6_A4 | 0 | OBSOLETE: disabled or removed; no active stock control |
| STRATA_BATCH_MTP_PAIR_SKIP_PROB | 0 | OBSOLETE: disabled or removed; no active stock control |
| STRATA_MTP_PARALLEL_ARGMAX | 1 | PORTED BUT INACTIVE: DD uses the subset vocabulary head, not the full head |
| STRATA_BATCH_MTP_OWNER_QUEUES | 0 | OBSOLETE: disabled or removed; no active stock control |
| STRATA_Q5_HEAD_RUNTIME_ONES | 0 | OBSOLETE: disabled or removed; no active stock control |
| STRATA_Q5_HEAD_RUNTIME_ONES_C6 | 0 | OBSOLETE: disabled or removed; no active stock control |
| STRATA_GDN_COMMIT_LAYERS | 1 | PORTED: opt-in grouped GDN layer commit; compiled, runtime validation pending |
| STRATA_QSA_COMMIT_GROUP_STEPS | 0 | OBSOLETE: disabled or removed; no active stock control |
| STRATA_ROPE_TABLE | 0 | UPSTREAM: control exists |
| STRATA_GDN_SPLIT | 0 | UPSTREAM: control exists |
| STRATA_GDN_COMMIT_SPLIT | 1 | UPSTREAM: control exists |
| STRATA_DBG_GDN | 0 | OBSOLETE: disabled or removed; no active stock control |
| STRATA_HC_Q8 | 1 | PORTED: #2810 BF16 conversion and Q8 read on dd-0.2.16; runtime parity pending |
| STRATA_EP_PREFILL_GATHER_ALIAS | 1 | PORTED: opt-in carry compiles; assembled runtime validation pending |
| STRATA_NIGHT_SH_LATE | 1 | PORTED: opt-in carry compiles; assembled runtime validation pending |
| STRATA_NIGHT_EP_PSUM | 1 | PORTED: opt-in carry compiles; assembled runtime validation pending |
| STRATA_NIGHT_EP_PLAN1 | 1 | PORTED: opt-in carry compiles; assembled runtime validation pending |
| STRATA_NIGHT_EP_ROWS | 2 | PARTIAL: the compiled carry covers IQ4_NL down and upstream has several row-reuse defaults; IQ2_XXS/IQ2_XS row reuse is only on the local #2802 refresh branch |
| STRATA_NIGHT_HC | q8 | PORTED: use STRATA_HC_Q8=1 (#2810); runtime validation pending |
| STRATA_NIGHT_ATTN_BATCH | 1 | UPSTREAM: STRATA_BATCH_ATTN_ROWS=1; streamed KV uses per-row path, as did the old batched path |
| STRATA_NIGHT_ATTN_VPIPE | 1 | PORTED: opt-in decode value pipeline; full build and runtime validation pending |
| STRATA_NIGHT_ATTN_MERGE_ACTIVE | 1 | UPSTREAM: STRATA_ATTN_MERGE_ACTIVE is default on Battlemage; verify selected dispatch |
| STRATA_NIGHT_TOPK_SCAN | 1 | PORTED: use STRATA_QSA_TOPK_SCAN=1 (#2806); runtime validation pending |
| STRATA_NIGHT_QSA_SCORES_T | 1 | PORTED: use STRATA_QSA_SCORES_TRANSPOSED=1 (#2806); runtime validation pending |
| STRATA_NIGHT_MMVQ | 1 | PORTED: use STRATA_MMVQ_KREG=1 (#2804); runtime validation pending |
| STRATA_NIGHT_PREFILL_EARLY | 1 | PORTED: opt-in carry compiles; assembled runtime validation pending |
| STRATA_NIGHT_PREFILL_F16SUM | 1 | PORTED: opt-in carry compiles; assembled runtime validation pending |
| STRATA_NIGHT_IQ | 1 | PARTIAL: upstream #4185 supplies faster decode helpers; IQ2_XXS/IQ2_XS and down row reuse remainder is only on unvalidated refresh/pr-2802-0216, not the measured DD binary |
| STRATA_NIGHT_GDN_SLOTS | 1 | UPSTREAM: use STRATA_BATCH_GDN_GROUPS=1; assembled validation pending |
| STRATA_NIGHT_ADMIT_D2D | 1 | PORTED: use STRATA_SLOT_COPY_DIRECT=1 and STRATA_SLOT_CKPT_KV_DIRECT=1; snapshot test passed |
| STRATA_NIGHT_MMVQ2 | 1 | PORTED: IQ3_S path for configured mask 2; full build and runtime validation pending |
| STRATA_NIGHT_MMVQ2_MASK | 2 | PORTED: configured mask 2 selects IQ3_S path; full build and runtime validation pending |
| STRATA_NIGHT_PREFILL_GROUPED | 1 | PORTED: opt-in carry compiles; assembled runtime validation pending |
| STRATA_NIGHT_PATTN_XMX | 2 | PORTED: use STRATA_PROMPT_ATTN_UNION=1 (#2808); runtime validation pending |
| STRATA_NIGHT_TOPK10 | 1 | PORTED: use STRATA_ROUTER_TOPK_TREE=1; compiled, runtime validation pending |
| STRATA_NIGHT_GDN_AB | 1 | PORTED: opt-in carry compiles; assembled runtime validation pending |
| STRATA_NIGHT_GDN_GRF | 1 | PORTED: opt-in carry compiles; assembled runtime validation pending |
| STRATA_NIGHT_PARK_MOVE | 1 | PORTED: opt-in carry compiles; assembled runtime validation pending |
| GLIBC_TUNABLES | glibc.malloc.hugetlb=1 | UPSTREAM: host allocator setting, retain only if appropriate |
| STRATA_NIGHT_SHEXP | 1 | PORTED: shared-expert gate/up and down paths; full build and runtime validation pending |
| STRATA_NIGHT_PREFILL_HALVES | 1 | PORTED: opt-in carry compiles; assembled runtime validation pending |
| STRATA_NIGHT_PREFILL_DDQ | 1 | PORTED: opt-in carry compiles; assembled runtime validation pending |
| STRATA_NIGHT_PREFILL_DQ32 | 1 | PORTED: opt-in carry compiles; assembled runtime validation pending |
| STRATA_NIGHT_PREFILL_DQCACHE | 1 | PORTED: opt-in carry compiles; assembled runtime validation pending |
| STRATA_NIGHT_BF16 | 1 | PORTED: use STRATA_BF16_KREG=1; compiled, runtime validation pending |
| STRATA_MTP_SHARED_BRANCH | 0 | UPSTREAM: control exists |

## PR status and triage

GitHub gh pr view on 2026-11-19: #2802, #2804, #2806, #2808, #2810, #2812, #2526, #2511 and #3273 all OPEN; #2526 and #3273 are drafts. Open status alone does not prove the code is still needed. PR #2802 is at least partly superseded by upstream PR #4185. Check every remaining change semantically before carrying; no refreshed branch exists yet.

## Measurement restriction

The stock two-GPU layer-split arm and old expert-parallel arm are matched by model, prompt and requested workload only, not by placement or batch-MTP behavior. Report both explicitly; do not attribute differences to the version alone.
