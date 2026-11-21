# Status (each lead keeps its own section current)

## Lead A (model path)  [updated 17:46 UTC]
- SERVE ROW int-rf2 (fn-a3 372bf1b2 = 2f022b5f + bench C2 knob; fn-int ff'd), two-card quick bench, 128 tok, oracle_equal true, health true:
  C1 109.5 (109% of 100) | C2 139.5 (93% of 150; 69.7/stream) | C4 168.0 (56% of 300) | C8 209.0 (42% of 500) | prefill 2k 1560 (104% of 1500)
  vs int-mtp2 e9a37df: C1 92.2 -> 109.5 (+19%), C4 149.5 -> 168.0 (+12%), C8 187.7 -> 209.0 (+11%), PF 1362 -> 1560 (+15%).
  Evidence: workers/S/int-rf2-results.json (+ int-rf-results.json, same build without C2: 109.5/-/166.5/210.1/1567).
  C1 and prefill targets MET. Open: C2 (-7%), C4 (-44%), C8 (-58%). C4/C8 min stream 42/26 tok/s; ITL p99 81/152 ms.
- fn-a3 = 5cfa39f7 ONE-CARD GATE PASS (+ PF d1a19df6 kregGroups table: q4_k/q6_ks persistent G=256 at c1/c4/c8, draft head q5_k c1 cap removed 77.8 -> 67.1 us;
  about -45 us per C1 window, 0 at C4 where DPAS is already at best).
- fn-a3 = e8e21a4f ONE-CARD GATE PASS: + PF 2b0de0ed (Q4_K persistent groups at R3-4, ssm_out R4 15.6 -> 14.4 us), + HS 092c82f0 GDN rows J=2
  (lead commit e8e21a4f makes J=2 the default, bitwise; XPU_GDN_ROWS_J=1 restores; gdn_rows lanes1 R1/4/8/16 4.06/6.35/9.89/16.67 -> 3.23/5.42/7.71/13.33 us),
  + HS 07c01aab XPU_HC_PREFETCH_AR (default off; flag off = unchanged stream; judged only by the HS block in H/r11/card1-queue.sh).
  recOOO [444.7, 583.1, 789.8, 1178.2] us (best so far; 988effc was 463/623/854/1434).
- LY MTP draft step R1 table (workers/LY/mtp/tail-t1.log): 44 launches, 488 us device vs 360 ideal, device-bound (gaps ~0.3 us).
  Fixes in flight: wide rmsnorm_rows (+16.6), mtp_chain_head argmax (+12.9), draft head q5_k_c1 N=20263 (+18.1).
- fn-a3 = 5e92440d ONE-CARD GATE PASS (+ PF 2b0de0ed: Q4_K KREG at R3-4 runs 256 persistent groups, same bits; GDN ssm_out R4 15.6 -> 14.4 us, about -43 us/C1 window).
  PF: IQ4_XS codebook alternatives (__constant, shuffle, early loads) all slower than the SLM stage; it stays.
- fn-a3 = c78c82b7 ONE-CARD GATE PASS (+ LY 517d7c47: IQ2 expert scale unpack via hardware half->float, removes the divergent denormal branch, same bits;
  fixture up R4 iq2_xs 77.4 -> 75.1, iq2_xxs 69.4 -> 64.7, R8 -4..-9%). HS norm2+down fusion (743e11d7) and HC L2 prefetch (6373c2c7): bitwise, window flat or slower, not merged.
- fn-a3 = 82d548e8 ONE-CARD GATE PASS (+ PF dc26dccd fn_fused_project_c4_h2 at R3-4, two quads per WG, same bits; R4 layers 2+3 70.9 -> 65.8 us,
  about -120 us per C1 window; XPU_FN_FUSED_HV=1 restores the old kernel). fused_project excess = iq4_xs KREG streaming at ~290 GB/s (decode-limited per LY).
- fn-a3 = 72d50dd0 ONE-CARD GATE PASS (+ HS 84ffb7f0 XPU_HC_BLOCK default on: hc_down b2/b4 + upmix_b with 64/32-bit subgroup block reads, same bits;
  slice per call R4/R8/R16 down 17.6/20.8/28.7 -> 16.1/17.2/27.0, upmix 16.6/19.1/28.3 -> 16.1/18.1/23.2 us). HC R1 is at the launch floor (3 us + bytes/590 GB/s);
  the next HC lever is overlap (L2 prefetch of the next HC weights, or fusing norm2 into the down prologue).
- fn-a3 = 32ca6f77 ONE-CARD GATE PASS (+ LY 32ca6f77: IQ4_XS R=2 RP8 kernel fn_q8_linear_iq4_xs_w_c2 for N>512, bitwise;
  attn_gate R2 13.3 -> 10.9, ssm_out 12.5 -> 10.4, attn_qkv 20.4 -> 18.8 us). IQ4_XS row split + arithmetic codebook measured, REJECTED (fused path regresses).
- fn-a3 = 3fe36cf3 ONE-CARD GATE PASS (+ PF 6823d59d: DENSE_DPAS 1024 persistent WGs for the main LM head, bitwise; op bench R16 572 -> 548 us).
  recOOO [455.4, 607.7, 822.6, 1236.7] us. PF non-bitwise q5_k finish DECLINED (~0.1% of window).
  C1 PLAN OF RECORD = PF's C1-window ranking (workers/PF/c1-0.log): kernel sum 15.6 ms/window, about 5.9 ms over byte ideal.
  Top excess: fused_project_c4 +954 (PF), hc_down +879 / hc_upmix +782 / hc_writeback_norm2 +244 (HS), ups_iq2_xs +530 (LY),
  moe_compact +479 / route_topk +279 / router_shared +401 (KF r11), q4_k ssm_out +308 (PF), iq4_nl shared-down floor +188 (PF).
  Assigned: PF items 1/7/10, HS HC kernels, KF r11 routing glue, LY expert ups.
- fn-a3 = 06dfcb1d ONE-CARD GATE PASS (+ LY 38e2d99c IQ3_S rs-only incl. DPAS + PF unpack readers, bitwise; -65 MB/card weights;
  attn_q R1 18.3 -> 13.6, R16 40.7 -> 32.3 us; project_fused (iq3_s+q6_k) R1 -14%). KF r11: fuse route_topk+moe_compact, fold q8 quant
  into producers (bitwise). KF r10 prefetch rejected (IQ2_S -5% at R8).
- fn-a3 = 8f2258aa ONE-CARD GATE PASS (+ LY e6ea2005 Q6_K row-split rs-only, bitwise, default on; raw not uploaded (+5.7 MB/card vs
  raw-only; keep-raw would be +434 MB); KREG/fused/DENSE_DPAS/PF unpack all read rs; attn_qkv R1 22.9 -> 20.9, ssm_out R1 16.8 -> 14.0,
  project_fused R1 -19%). Card-1 to-do: PF prefill greedy hash (TP2) for the Q6_K rs prefill unpack.
- fn-a3 = 1144114c ONE-CARD GATE PASS (+ KF f3382a6f: router M4 tiles (bitwise, default) + FN_GGUF_EXPERT_DPAS union (off)).
  Layer slice recorded OOO us R1/4/8/16: 453/611/821/1245 (e9a37df-era 503/671/864/1386 -> about -10%; serving immediate default path
  also improved). KF dispersed table (rotated, real GGUF): rank-0 rows/expert 1.07-1.19 at R4-R16; GU/down 63-79% BW at R8/R16 ->
  C8 expert headroom <= ~25%. KF r10: one-chunk software prefetch in split GU (IQ2_S first; stop if < 5%).
- LY: Q6_K row-split (bitwise; attn_qkv R1 77 -> 86% BW, project_fused R1 -22%) going rs-only (no raw upload; saves the +412 MiB/card
  a raw copy would cost) after porting PF prefill unpack + DENSE_DPAS q6_k readers; then IQ3_S rs-only (-62 MiB).
- fn-a3 = a4da5a1f ONE-CARD GATE PASS: + PF cfb630a1 (sampler fn_logits_part_sg*, bitwise, default on; lm stage-1 R16 91 -> 30 us)
  + HS 47021875 (XPU_ATTN_DPAS v2: 8-SG / 4-SG slice split + tree merge, off; p2048 attn+merge R1 31.3 -> 10.5 us, R16 157 -> 59;
  real slice window p2048 R1 561 -> 460 us; tokens change -> KL on card 1).
- fn-a3 = c2ae97b8 ONE-CARD GATE PASS (incl. LY 9ed8a9a4 load 11.6 -> 5.1 s). New since b4dce2a1:
  - H dd5aeb7c XPU_MTP_HC_Q8 (drafter HC on Q8; never used before; R1 body 391 -> 371 us; off, acceptance-gated) + MTP overlap probe.
  - LY 4f0878e4 dense IQ3_S row-split (bitwise, DEFAULT ON; attn_q R1 18.4 -> 13.6 us = 81% BW; shared_fused R1 -26%) + XPU_MTP_Q8ACT
    (MTP experts on q8_1, off; R1 50.8 -> 36.3 us; proxy top-1 97.95%, card-1 acceptance decides).
  - LY 9ed8a9a4 staged parallel split repack: load 11.6 -> 5.1 s (raw 7.8).
  - HS af8c82b9 XPU_ATTN_MERGE_WIDE (bitwise, mixed speed, off).
- KF cafb9ec6 (opt-in expert XMX union + router M4 tiles) NOT merged: 13-file conflict (fn-prep carried an old LY commit). KF round 9:
  merge fn-a3 first; then characterize DISPERSED routing (KF: no union byte saving when C8 streams pick disjoint experts; concentrated
  R16 1.85-1.98x, dispersed -2.8..-10.9%) and propose one dispersed-case change.
- CARD-1 RETURN PLAN: bash workers/H/r11/card1-queue.sh (H, syntax-checked, not run; PASS criteria in H/r11/PASS-GATES.txt).
- STRATEGIC NOTE: C4/C8 expert cost is per-distinct-expert bytes (dispersed routing) -> the levers are %BW per expert at 1-2 rows
  (LY/KF) and the non-expert multi-row kernels (DPAS dense/attn/HC done or flagged), not union.
- fn-a3 = b4dce2a1 (+ PF fa466559 fn_gguf_linear_bf16w for MTP o_proj K=3072 / shared down K=320 at R>=5, bitwise; tail launches
  R8/R16 112/128 -> 110/122; PF: mtp_body R16 1210 -> 1075, tail window R16 3467 -> 3339). ONE-CARD GATE PASS.
- fn-a3 = 78d34e79 (+ HS eae9a448 XPU_ATTN_DPAS=1 DPAS decode attention, default off, row-invariant; op p2048 1seq R16 140.6 -> 48.3 us,
  p8192 149.4 -> 55.3; slice lanes8 window R16 1519 -> 1426 us). ONE-CARD GATE PASS (layer + tail + keep batch, health OK).
  Card-1 queue adds: ATTN_DPAS serve + KL. HS next: fn_desc_attn_merge (12.5 us at R1, serial 65-slice loop).
- CODEGEN LESSON (HS): a uniform debug kernel argument with branches destroyed codegen (DPAS kernel looked 3x slower). Diagnostic
  variants must be separately compiled kernels, not runtime-branch args.
- fn-a3 = e8d534c0, ONE-CARD GATE PASS (layer slice hashes == reference; TAIL slice SLICE_BITWISE + SLICE_LAUNCHES + TAIL_KEEP_BATCH PASS;
  health OK). New since fc200f43, all bitwise:
  - PF 5601aefb weight-stationary bf16 K=2560/640 kernel (fn_bf16t, default on): N=24576 R16 513 -> 321 us; MTP body R16 1247 -> 1117.
  - H 9d9eef8e batched keep -> XPU_FN_KEEP_BATCH DEFAULT ON (Lead A e8d534c0): 73 launches/card regardless of lanes; keep C4 1.13 -> 0.52 ms,
    C8 2.26 -> 0.90 ms (one-card matched probe).
  - LY 42662d33 FN_TIMING_ROTATE fixture.
  Layer slice recorded OOO us R1/4/8/16: 466/620/819/1293.
- H r10: C1 MTP chain (drafter HC Q8?, cross-step overlap). Card-1 queue unchanged (card1-validate.sh first).
- BENCH HYGIENE #2 (LY): the expert timing fixture at R <= 4 was L2-resident (one routing plan reused, 1-2 MB). Use FN_TIMING_ROTATE=1
  (25 disjoint expert sets, merged in fn-a3 aba6c7e3) for any expert us claim. DRAM-realistic split up %BW: R1 43-51%, R2 60-76%,
  R4 57-70%; down R1 53-55%. R=1 is per-thread chain latency (one under-filled round of ~100 WGs per rank), not bandwidth.
  KF: re-check round-8 expert numbers with rotation.
- MTP drafter numerics change only acceptance, never output tokens (greedy main-model verify): drafter changes are gated by draft
  top-1 agreement + real acceptance (card 1), not KL. LY -> MTP experts on q8_1 dp4a (XPU_MTP_Q8ACT=1).
- PF working: weight-stationary bf16 K=2560 kernel (~400 us excess at R16, MTP layer), K=3072/320, sampler fn_logits_part.
- PF 13:28 UTC, FYI H/R: fn_logits_part (argmax+LSE stage 1) is now same-bits fn_logits_part_sg{1,2,8} via Device.logitsPart
  (all 11 call sites; b70/fn-pf cfb630a1). PV/PI/PS/token/best are bit-identical (invariance-mtp and TAIL hashes); R16 91 -> 30 us,
  R4 23 -> 9. XPU_FN_LOGITS=0 restores the old kernel. New launch code should call d.logitsPart, not d.k.logits_part.
- fn-a3 = fc200f43 (+ PF 15c82f77 DPAS v2: LM/draft head q5_k R16 145 -> 96 us (ideal 58.7), R2-8 1.21-1.24x ideal; q6_k/q4_k/small-N
  shapes on DPAS from R=9 by per-shape policy; all bitwise). ONE-CARD GATE PASS (hashes ok, health ok). Slice recorded OOO us
  R1/4/8/16: 462/623/824/1298. PF next: rank non-expert/attn/HC kernels by us - ideal from the slice TAIL table, attack top 2-3.
- H r8 window reconciliation (78e52926 merged; slice TAIL harness): conserved ledger window (train1-imm card 0) C1 31.0 / C4 44.2 ms
  = 48 layers 22.1 / 29.8 + outside 8.9 / 14.5 (embed 0.5/0.8, head+sampling 1.0/1.4, verify/keep 2.7/2.8, draft chain 4.7/9.4).
  One-card C8 proxies: serial keep 1.95-2.22 ms (73 launches per lane) -> H r9 batched keep (bitwise, flag); MTP chain 0.86-1.33;
  head+sampling 1.02 (PF on q5_k head). Needs a fresh two-card C8 ledger when card 1 returns (Lead B).
- HS: decode attention is chain/ALU-bound (no-load variant same time), 4-5.5x off ideal at prefix 2048; no bitwise fix -> DPAS
  flash-decode attention behind XPU_ATTN_DPAS=1 (KL-gated), reusing PF fn_pf_attn_gather per-row design (row invariance).
- fn-a3 = d97096cb, ONE-CARD GATE PASS (slice hashes == reference, health OK). New since 1c15dcd:
  - PF a91c2670 DENSE_DPAS (int8 DPAS dense projections R >= 8, BITWISE == KREG) -> DEFAULT ON (Lead A 3299501d). R16 us: attn_qkv
    60.4 -> 34.5, attn_gate 41.8 -> 22.4, attn_q 56.5 -> 28.2, LM head 207.7 -> 145.1 (ideal 58.7). Sibling fusion bypassed at R >= 9.
  - LY 922bedd7: routed down Q2_0/IQ4_NL + MTP q2_0 experts split layout (bitwise; MTP up R1 49.8 -> 21.9, down 27.5 -> 13.8 us; the
    16-lanes-same-16B finding confirmed); load 17.0 -> 11.5 s (raw 7.8).
  - H 4376fe4d: recorded capture now uses fused/packed paths; launches equal in all modes.
  Slice medians us R1/4/8/16: recorded OOO 465/615/822/1308; immediate in-order (serving default) 676/697/928/1432.
- Worker tasks: H r8 = window budget reconciliation (non-layer ms at C1/C4/C8, slice TAIL); LY = experts R1-4 toward 80% BW + dense IQ3_S;
  PF = LM head / draft head q5_k + q6_k/q4_k DPAS; HS = decode attention/GDN at R8/16; KF = grouped XMX experts R8-16; R parked.
- FIRST on card-1 return: H/r7/card1-validate.sh (rows default / OOO / RECORD+OOO) then the rest of the queue.
- fn-a3 = 1c15dcdf, ONE-CARD GATE PASS (slice hashes == reference at R1/4/8/16 in all 4 modes, health OK). Since e9a37df (last serve row):
  PF 4974075 + KF 23fb20b fusion + H 7aa50080 slice harness + HS ddc31a9 (HC_Q8 off) + PF 17d2d7bf / 4035abb2 (PF_ATTN/GDN/ATTNS off)
  + LY 8930f006 (expert gate/up SLM grids + aligned split layout, DEFAULT ON, bitwise; kernel_labels 256 -> 512)
  + H e0fa1982 (XPU_FN_OOO=1 immediate out-of-order descriptor windows, default off; RECORD+evpool load fix = second registerCleanup slot).
  Slice window medians us R1/4/8/16 (card 0, not quiet): immediate in-order 641/693/933/1552; immediate OOO 555/653/878/1459;
  recorded OOO 502/671/864/1386. Recorded capture still uses the non-fused path (68 vs 56 launches) -> H round 7.
- Card-1 queue (H preparing H/r7/card1-validate.sh): serve rows 1c15dcd default, XPU_FN_OOO=1, RECORD+OOO; chunk-solo 1024; HC_Q8 + KL;
  PF_ATTN+ATTNS (+GDN) first-token/greedy/KL; R MTP dump 2 x 3 min.
- fn-a3 = e50dec2d (+ PF 4035abb2 XPU_FN_PF_ATTNS=1 sparse-regime DPAS prefill attention, default off; op 4.5x per 64-row tile,
  flat 2k..32k; est -180 ms per sparse 2048 chunk per card). Builds clean. Card-1 queue adds: PF_ATTN+ATTNS greedy-32 + KL at 2k/8k.
- PF prefill AR wait = wire time (2.0 GB per 2048 chunk at 11.2 GB/s ~ 180 ms). Lead A decision: f16 AR payload NOT taken (AR
  compression out of scope per 05:55 LITERATURE actions; orchestrator may reopen). 2-microbatch layer wavefront parked (prefill ~91%).
  PF reassigned to decode: DPAS dense projections for R >= 8 (XPU_FN_DENSE_DPAS, default off) -- the C4/C8 gap.
- LY 6a037833 (expert up SLM grids + split layout, bitwise, R8 iq3_xxs 252 -> 184 us) conflicted with PF up3 path; LY rebasing on fn-a3.
- ONE-CARD GATE PASS on fn-a3 988effc (card 0, 33 s, health OK): slice default hashes == H reference at R=1/4/8/16 (all merges since
  e9a37df are bitwise on the default path). Kernels per slice window 68 -> 56 (R16 84 -> 60) from KF fusion. Slice window medians (us):
    ('immediate', 'inorder', 1) 602.9
    ('immediate', 'inorder', 4) 796.2
    ('immediate', 'inorder', 8) 978.5
    ('immediate', 'inorder', 16) 1548.6
    ('immediate', 'ooo', 1) 605.2
    ('immediate', 'ooo', 4) 661.7
    ('immediate', 'ooo', 8) 905.5
    ('immediate', 'ooo', 16) 1488.3
    ('recorded', 'inorder', 1) 518.5
    ('recorded', 'inorder', 4) 681.5
    ('recorded', 'inorder', 8) 907.7
    ('recorded', 'inorder', 16) 1522.1
    ('recorded', 'ooo', 1) 463.0
    ('recorded', 'ooo', 4) 622.8
    ('recorded', 'ooo', 8) 854.4
    ('recorded', 'ooo', 16) 1433.7
- fn-a3 988effc = e9a37df + PF 4974075 + KF 23fb20b (fusion) + H 7aa50080 (slice harness xpu-fn-slice) + HS ddc31a9 (XPU_FN_HC_Q8,
  default off) + PF 17d2d7bf (XPU_FN_PF_ATTN / XPU_FN_PF_GDN, default off). Card-1 queue: serve row 988effc; chunk-solo 1024 arm;
  HC_Q8 serve + teacher KL; PF_ATTN / PF_GDN first-token + greedy-32 + KL; R dump (2 x 3 min).
- HS in-window HC (slice, recorded OOO): Q8 down 28.7 -> 14.9 us, upmix 28.5 -> 14.8 us at R16; ~-0.95 ms (R1) / -2.25 ms (R16) per
  card per window projected. PF_ATTN op: 2048 rows 9.95 -> 1.09 ms (est ~-120 ms per 2048 chunk). PF_GDN 1.36x at 2048, slower < 512.
- H round 6: decompose recorded vs OOO gain; XPU_FN_OOO serving flag; RECORD+evpool load failure.
- R e82d561 (b70/fn-rec2-mtp): XPU_FN_DRAFT_RES_F32=1 arm (drafter residual + HC writebacks fp32; default off). Card-0 op check: bitwise vs
  host f32 model, residual rel err 1.6e-3 -> 2.5e-8, f32 kernels faster (writeback_norm R4-16 4.3 -> 2.8 us). MTP-only acceptance scorer
  (mtp_accept.py, removes copy-draft confounder). run_dump.sh (2 x ~3 min two-card leases) QUEUED for card 1 return.
- fn-a3 = 113ee5c (builds clean). Unbenched, waiting on card 1: + PF 4974075 (HC upmix prefill retile) + KF 23fb20b
  (sibling projection fusion qkv/z/beta/alpha and q/k/v/indexer as one launch; router+shared gate/up/SwiGLU as one launch;
  static -288 (R1) / -336 (R4/R8) / -624 (R16) launches per rank per window; KF 2-card quick gate PASS 07:22, hashes unchanged;
  FN_GGUF_FUSE=0 / FUSE_ROUTER=0 revert). KF's serve-bench script rewrite was NOT taken. fn-int stays at e9a37df (last benched).
  LEAD B: build tag for ledger regen when card 1 returns = fn-a3 113ee5c.
- R: code read found NO structural drafter mismatch vs Strata (norm scope, fc_e/fc_h + Gemma +1, embedding, hidden handoff,
  rope pos, catch-up/KV all match). fp32 draft logits change nothing (identical accept counts). Remaining = numeric (bf16 drafter
  residual/HC, bf16 drafter KV, bf16 embedding, bf16 main-R handoff) + copy-draft confounder. Dump + FP64 numpy reference + an
  f32-residual arm are queued for a ~6 min two-card lease when card 1 returns.
- [A-KF 12:48 -> A] MERGE-READY opt-in asset cafb9ec6 on fn-prep; native+harness build PASS, final API gate PASS/healthOK/reset0, final kernels embed exact gated/timed C12 image. Default OFF: hotR16 chain1.85-1.98x, dispersedR8/16 +2.8-10.9% slower. A/B60332e35 already delivered. Report8 and KF_report8.md updated. Request pending: allow two fixed sparse/XMX kernels per stage (+96 launches/rank/window) to avoid one-kernel sparse-mode overhead, or retain one-launch opt-in. No dependent split implementation yet. When cherry-picking, retain integration kernel_labels512 over branch320.
- [A-KF 12:46 -> A] Final native+harness build PASS278s; embedded expert image byte-identical to fully gated/timed C12. Expert union is DEFAULT OFF (FN_GGUF_EXPERT_DPAS=1 opt-in): hotR16 1.85-1.98x faster, dispersedR8/16 2.8-10.9% slower. C13 grid and C14 <=16KiB SLM variants passed bits/FP64 but did not remove penalty; rejected. Final API binding gate queued card0, then code commit. Report KF/report8.md now has complete selected per-op R1/4/8/16 us/GBs/%608/ideals and conditional48-layer projections. Asked Lead A whether one-GU/one-down constraint may relax to two fixed sparse/XMX kernels per stage (+96 launches/rank/window); no dependent implementation before answer. A/B60332e35 remains merge-ready.
- [A-KF 11:25 -> A] C union bitwise + sampled FP64 PASS all5 up/2down, public prepared/unprepared map PASS, claim41s/healthOK. Frozen C9 latency capture has waited for CPU-quiet/card0 overlap since11:02; builds and GPU claims keep alternating. Please leave the next ~60s overlap for KF (hard GPU bound250s); no GPU held while waiting. Offline C10/C11 probes still spill, so no unmeasured default change. A/B60332e35 remains merge-ready; full selected per-kernel/byte-ideal table now KF/round8/selected-fusion-records.tsv. C remains dirty until measured.
- [A-KF 08:30 -> LY] Fable5.2/Opus5.5 high independent reviews complete: consume split chunks, integer DPAS, 8 canonical K parts; no full-weight SLM copy for up. Need published shared decoder ABI/commit (currently see uncommitted QE.split/q_split_part on fn-ly). Please keep qs rw*K/4+(c*8+sub)*16, sg rw*K/8+(c*8+sub)*8, mt (rw*8+sub)*Q_SPLIT_MP(T), d rw*K/128+4c, and accessor q_split_part stable or post changes here. KF will not edit loader/repack or QE layout. Draft kernel uses these read-only planes and overlays metadata/chunk/partial SLM <=31KiB, native128GRF,16-row reuse; integer probe/down prototype independent of up-layout publication. IQ4_NL DPAS masks use interleaved8+8, exact old float association/tree. Fusion ROW_BITS PASS all48layers/37rows/everyposition/two permutations; cold timing still queued for CPU quiet (since08:11), no GPU owned while waiting.
- [A-KF 09:00 -> LY/A] Consuming published LY 01dd0e3b accessor/producer read-only (cherry-picks 54c4d2de/37b18d33 after merge of accepted fn-a3 113ee5c). No loader/layout edits by KF. Resolved prefill G2/up3 context by retaining integration span logic and LY split branches. Integer XMX packet probe PASS random/onehot, all R1/2/3/4/8/16, every position37, masks5. Candidate down bit gate caught 1ULP on first tiny-scale row; candidate remains disabled, diagnostic partial-sum probe building. Fusion ROW_BITS passed all48layers, cold timestamp capture still queues outside GPU for quiet CPU; no serve/two-card work.
- [A-KF 09:03 -> A] Pending A/B table needs one CPU-quiet + card0 ~3min slot; queued outside both leases since08:12, first quiet wait expired1800s, retry took quiet08:52 but withdrew immediately because GPU0 busy. Frozen tested binary ready; no rebuild needed. Please reserve the next quiet/GPU0 overlap if coordinating builds. Default quiet rule remains honored; optional pinned device-only exception question pending, no exception used.
- [A-KF 09:13 -> A] Cold A/B capture PASS181s card0-only quiet CPU, alltimedbits0, pre/posthealthPASS. Table report8.md + round8/fusion-records.tsv. Per-rank48layer projection+router/shared sumus R1 3101.9->2175.3,R4 3817.4->2841.8,R8 5096.0->4632.3,R16 8645.1->9492.8 (REGRESSION). Fixing dispatch: old R16 siblings, oldrouter+fusedshared R8/16 (shared alone improvesallR). No C8 speed claim from23. C integerpacket PASS; IQ4NLdown1ULP diagnosis+all5upbits building. No expert layout changes.
- [A-KF 09:30 -> A/LY] Exact1ULP cause PROVEN withoutGPU: IGC oldq_reduce8 emits adjacent1/2/4 tree despite sourcexor4/2/1. CPU replay real blk0down row2 + exactZigPRNG/q8 reproduces old35ca56c6 adjacent and candidate35ca56c7 literalxor. Corrected newSLMtree to emittedoldtree (no oldkernel/hashchange). Evidence KF/round8/down-tree-reproduction.json + isa-debug. C7build queued: newexpertintegerXMX tree, wideGRFup toremove scratch, nativeM4router4rowtiles/32Bblock-read weightloads, legacyR16siblings. Noexpertupload/layoutmods.
- [A-KF 10:25 -> A] MERGE-READY 60332e35 A/B follow-up only (4 files; no bench changes). Native+bit gates PASS, cold card0/quiet CPU PASS. Router+shared48layer sums R1/4/8/16 old1188/1370/1730/2529 ->471/712/1157/1245us. R16 siblings use legacy fallback6046->6042(noise). Launch delta -288/-336/-336/-240 perrank. KF_report8.md handoff updated; C expert work continues dirty, do not merge it yet. C7 all5up/2down direct rowbits PASS; C8 hybrid128 still has3types spilling, SLM accumulator fix next.
- [A-KF 07:39] 23fb20b accepted/merged113ee5c. Card0-only from now; no further TP2 gate or kf9 bench. Restored Lead A bench script locally (de6fc283); no bench changes proposed. Fusion op matrix/timing pending; fixing fixture to use random timing rows and device timestamp resets per HARDWARE.md before any capture (none has run). Priority C now DPAS integer dots, preserving old eight K-part chains/tree and exact q8_1 tail. LY owns upload/layout; KF will only consume QE.split/q_split_part. LY coordination: please keep/expose stable per-row split offsets and decoded signed int8 lo/hi plus d/a/b scales for DPAS K32; no KF expert-upload/repack edits. Proposed WG256 = 8 K parts x 2 N16 tiles, shared across all selected token rows (two M8 DPAS calls for up to16), fixed grids, SLM codebooks/activation tile, one layer-wide grouped GU launch and one down. Numerical contract needs two masked K16 dots for IQ2_S/XS scales, one K32 integer dot for IQ3_S/XXS/IQ2_XXS; exact old float finish/order retained. Details follow after independent review.
- [A-PF 09:25 -> B, cc A] PREFILL AR WAIT IS WIRE TIME: one 2048-row AR = 2048 x 2560 x 4 B = 21 MB, 96 per chunk = 2.0 GB; at the
  11.2 GB/s P2P push ceiling (TRANSPORT-B table) that is ~180 ms of the measured 202-222 ms, so it is not skew. Two levers:
  (1) small, transport-local: prefill-only (rows >= 256) half-width payload. Each rank converts its f32 send to f16 locally (~40 us),
  pushes half the bytes, and a symmetric reduce reads both f16 operands in rank order -> f32 dst (bit-identical on both ranks).
  Est. -90..-100 ms per 2048 chunk (~8% at 2k). Not bitwise, KL-gated, decode untouched, default off (XPU_FN_PF_AR16).
  (2) larger: 2-microbatch layer wavefront + peer push on a copy-engine list. The AR of half A (~1 ms) overlaps compute of half B
  (~4.6 ms per half segment), hiding ~all ~200 ms (~16%). Needs forwardPass row-offset sub-batches plus a copy-engine list in tp.zig.
  ASK B: may PF implement (1) in tp.zig behind the flag on b70/fn-pf, or do you want to own it? PF will not touch transport until you answer.
- CARD-0-ONLY WINDOW (orchestrator 07:25, [INK] holds card 1). Lead A workers redirected: one-card merge gate = bitwise unit/op test
  (or KL/top-1 op test) + per-kernel us R=1/4/8/16 on real GGUF tensors; serve rows queued until card 1 returns.
  - H round 5 = SINGLE-CARD WINDOW HARNESS (GDN+QSA layer slice, MoE+HC, real card-0 TP half, other half simulated by local
    buffer, immediate + recorded, per-kernel us + output hash as a one-card bitwise gate). Lane-depth re-tune paused.
  - LY: R1 expert-up SLM grid + split layout (card 0). HS: Q8 HC. PF: chunked GDN + DPAS attention prefill (card 0 op level).
  - R: fp32 draft-logit one-card op test + trace analysis (full-model parity paused).
- SERVE ROW #9 (07:17, int-mtp2 = fn-int e9a37df: + H4 multi-lane device MTP + HS writeback_norm2/down-once):
  C1 92.2 / C4 149.5 / C8 187.7 / PF 1362. target C1 92% C4 50% C8 38% PF 91%. p0 exact, health OK.
- QUEUED for serve when card 1 returns: fn-a3 (+PF 4974075 HC upmix retile, merged 07:23), XPU_CHUNK_SOLO=1024 arm
  (PF engine-level 2k 1580 / 8k 1423 vs 512: 1378/1240; 2048: 1689/1517; check C8).
- AU AUDIT DONE (/mnt/vm_8tb/b70/results/tf-flashnext/workers/AU/audit-input-for-LY.txt). Answer to orchestrator 1+2:
  - EVERY quantized weight is uploaded as RAW GGUF blocks; only the router is repacked (DPAS tiles).
  - Dense KREG/QROW kernels stage their grids into SLM. Decode expert-up QUP and prefill fn_pf_moe_up read IQ grids from __constant.
  - Top layout-fixable (est. ms per card per window, C1/C4):
    - R1, expert gate/up IQ3_S/IQ3_XXS/IQ2_*: -0.88 / -1.58
    - R2, expert down Q2_0/IQ4_NL 18 B blocks: -0.31 / -0.60
    - R3, dense IQ3_S: -0.28 / -0.17
  - Dense K-quants (q6_k/q4_k/q5_k/iq4_xs) gain ~0 from a repack. Their gap is kernel structure (1 tile in flight, dp4a ALU at R=8), so it needs a persistent/DPAS rewrite.
  - Ledger overstates expert %BW: it assumes 19.4 distinct experts per launch; live is ~13.
  - MTP fn_gguf_up/down_q2_0: all 16 lanes load the same 16 B (likely bug).
  - Net R1-R3: about -1.5 ms at C1 (+4 tok/s) and -2.35 ms at C4 (+8 agg).
  - Non-layout: HC down (few subgroups, weights re-read every 4 rows); HC upmix 4 B/lane; router should use fn_router_dpas_mm from 2 rows.
- LY worker (Claude Opus, b70/fn-ly) launched: SLM-grid check, then R1, then R2 (incl. the MTP q2_0 lane bug).
- PF 4974075 merge-ready (HC upmix retile 1.8 -> 0.6 ms @2048 rows, 2 expert groups per weight decode; bitwise). Engine-level real-weight
  prefill: 2k 1378 (chunk 512) / 1689 (chunk 2048); 8k 1240 / 1517. Remaining per card @2048-row chunk: wall 1244 ms vs device
  887 ms; TP AR WAIT 202-222 ms (LEAD B: prefill all-reduce wait is now the biggest single prefill gap), host 73, experts ~200,
  gdn_rows 116, attn_sparse 113, HC down 62 + writeback_norm 52. PF next: chunked GDN scan + DPAS flash attention (KL-gated), chunk 1024 check.
- HARDWARE.md items 1+2: AU (Claude Opus audit, agent a274d0942de6df219, ~1 h, read-mostly) audits every decode kernel for raw GGUF
  block reads vs aligned split layout, __constant vs SLM grids, load width; ranked table + top-3 loader repack sketches in
  workers/AU/report.md. Bench-hygiene rule (real weights / random data only) sent to PF/HS/R; KF gets it on its next resume.
- DEVLOOP BOTTLENECK (for orchestrator): bin/cpu-run allows ONE heavy job at a time, and quiet waits for it. With ~10 agents
  each doing ~5 min -j4 zig builds, integration builds queue 15-25 min and quiet benches wait behind builds. Suggest: let cpu-run
  admit 2-3 concurrent heavy -j4 builds (RAM cap 90G each is the real limit; check host RAM), or a separate build slot for the
  integration line ([A-S]). bin/ change = orchestrator's call; Lead A did not touch bin/.
- SERVE ROW #8 (06:44, int-pf2 = fn-int 2133d91: + PF 7973882 dense prefill GEMM retile + pair-major expert gather; p0 exact, health OK):
  C1 83.4 / C4 151.0 / C8 187.8, PREFILL 2k 1351 tok/s. target C1 83% C4 50% C8 38% PF 90%.
- Next row building: fn-a3 e9a37df = + H4 multi-lane device MTP (156e010) + HS writeback_norm2 / down-once (cbb488c), both bitwise.
- HS findings: HC split across cards NO-GO by analysis (rmsnorm/silu between AR and sublayer -> needs a new hop per sublayer, hop
  >= 9.4 us vs <= 8 us saved; Q8 dominates). Single fused norm->down->upmix needs a grid barrier -> not worth it; writeback_norm fusion
  into the AR consumer = Lead B arfuse. Done (bitwise, cbb488c, smoke queued): writeback_norm R1-16 5.2 -> 2.2-2.8 us, down R8 24.0 ->
  19.6, R16 35.1 -> 27.8 us (ideal 10.9). Next: Q8 HC behind a flag, KL gate.
- PF 7973882 merge-ready (dense prefill GEMM retile 15 -> 105-128 TFLOPS, pair-major expert gather, bitwise): engine 2k prefill
  chunk 512 1009 -> 1309, chunk 2048 -> 1525. Merged into fn-a3; int-pf2 quick bench running.
- R: drafts differ both ways vs Strata (both 223, Strata-only 52, TF-only 27); first candidate = bf16 draft logits / bf16 drafter
  residual (Strata fp32) -> testing fp32 draft logits.
- LITERATURE.md actions routed (05:55): HS = Q8 HC weights (Strata HC v2 Q8) fused with norm->down->upmix, KL-gated, env flag;
  R = MTP acceptance parity: diff drafter logits vs Strata on identical prefixes, find first divergent op; PF = expert-sorted tokens +
  one grouped DPAS GEMM per projection (187 -> ~22 GB/card/chunk); KF = after projection fusion/router, decode experts at >= 8 rows
  via DPAS/XMX grouped by expert across streams (C8 path).
- LEDGER-train1-imm decision (stay on fork, fusion first) applied: HS = fused HC kernel per sublayer (norm->down->upmix, ~200 fewer
  launches/window) THEN HC split across cards; KF round 8 = (0) fn-a3 merge of 820cb20, (A) sibling projection fusion (qkv+z+gate,
  shared up+gate, ...), (B) router (35% BW) fused with shared-expert gate, (C) grouped-expert union R=8..16. Each report quotes
  per-kernel us at R=1/4/8/16 vs byte ideal + launch-count delta. LEAD B: please regen the markers-only ledger table on each Lead A
  merge (next: KF 820cb20 rebase, then HS fused HC) so we report kernel deltas.
- SERVE ROW #7 (05:16, int-pf-512 = fn-int aa66709: train1 + PF prefill DPAS d8baa8d, XPU_CHUNK_SOLO=512; quick bench; p0 exact,
  streams == solo, health OK; S/int-pf-512-results.json): C1 83.1 / C4 151.1 / C8 191.3, PREFILL 2k 1024 tok/s (was 503).
  % of target: C1 83%, C4 50%, C8 38%, prefill 68%. Chunk 2048 (int-pf-2048): prefill 1147 but C8 147.9 -> default stays 512.
  KF7 820cb20 conflicts with evpool/PF in xpu_store.zig/trace.zig; KF round 8 merges fn-a3 first, I take its next commit.
- TARGETS (user, BRIEF): C1 >= 100 / C2 >= 150 / C4 >= 300 / C8 >= 500, prefill >= 1500. Lead A staffing to them (multi-row first):
  KF round 8 (Codex) = expert up/down grouped by distinct expert at R = 8..16 (weights read once per window) + GDN/QSA at R = 8/16;
  HS (new Claude Opus, agent a4258554ec0395b67, b70/fn-hs) = HC split across the two cards riding an existing AR (-0.66 GB/card/window)
  + one HC weight read per window; PF = XMX HC/experts/attention for prefill; H4 = multi-lane device MTP + lane-depth re-tune
  (Lead B: H4 already covers the lane policy item; we will share results); R = MTP drafter acceptance (matched t256, depth 4, IQ3_S/IQ4_NL drafter).
- Merged into fn-a3: PF d8baa8d (prefill DPAS, decode untouched, hashes equal) -> aa66709; bench default XPU_CHUNK_SOLO=512.
  KF7 820cb20 (cached kernel args) merge-ready, merging after the running PF benches.
- C1 GAP ANALYSIS, CORRECTED 04:45 (R, CPU only): Strata DD uses the SAME q2_0 MTP experts (its "mtp-bf16" rt is converted from
  mtp-q2_0.gguf), same depth 3, same draft vocab, suffix/lookup/min-p OFF. Matched per prompt (Strata oracle-eight DONE lines vs
  ours): acceptance per draft p0 0.69 vs 0.62, p1 0.95 vs 0.88, p2 0.92 vs 0.88, p3 0.72 vs 0.69 (ours 3-7 points lower).
  On p0: tokens/window 2.50 vs 2.24 (+12%), window time 23.2 vs 27.7 ms (+19%) -> window time is the slightly larger lever;
  Strata's "C1 ~141 / 70%" figures come from its own (easier) prompt, not p0. My 04:36 "drafter is the main lever" read was wrong.
  R next: matched 256-token run, C1 depth 4, and an IQ3_S/IQ4_NL MTP drafter test (b70/fn-rec2-mtp).
- Per orchestrator 04:32: RECORD x evpool PAUSED (R does a CPU-only Strata window-protocol summary, R/strata-window-protocol.md,
  input for the structural decision). Waiting on Lead B's markers-only ledger of train1-imm 3edbc4d (C1/C4). PF (prefill XMX port),
  KF8 (GDN host setup) and H4 (multi-lane device MTP + lane depth) continue.
- SERVE ROW #6 = MERGE TRAIN 1 immediate (04:16, train1-imm, fn-a3 3edbc4d: DESC default + H3 device MTP + B evpool + R5 F1/drain,
  RECORD off; QUICK bench = 128 tokens; p0 == previous build prefix, C4/C8 stream 0 == solo; health OK; S/train1-imm-results.json):
    C1 82.6 / C4 151.4 / C8 191.6 agg tok/s; prefill 2k 503. (vs int-desc 256-token: C1 70.7 / C4 139.6 / C8 176.0 / 482.)
  Note: 128-token quick runs are not exactly matched to the earlier 256-token rows (shorter decode, smaller KV).
  RECORD on (train1-rec) FAILED AT LOAD: "could not load ... ZeFailed" right after evpool init (events_created=2048) -> recording
  x evpool conflict (owner: R rec2 + Lead B evpool). RECORD with XPU_TP_EVPOOL=0 (train1-rec-noevpool, p0 exact, health OK):
    C1 83.7 / C4 136.5 / C8 163.8 / prefill 492 -> recording is NO LONGER a win over immediate+evpool (C4 -10%, C8 -15%).
  DECISION: integration default stays immediate + evpool + DESC + device MTP. Recorded path only returns via R's Strata-style
  epoch-flag transport (b70/fn-rec2), which must be evpool-compatible and beat 151/192 at C4/C8.
  Target at this decision point: C1 >= 110, C8 >= 220 -> not met by immediate (82.6 / 191.6).
- QUICK LOOP (user 03:45): tools/flashnext_serve_bench.sh FN_BENCH_QUICK=1 default (fn-a3 0df935c): C1/C4/C8 x 128 tokens + prefill 2k,
  p0 == previous build (prefix of S/expected-current.json) and every C4/C8 stream on p0 == solo p0; one lease ~3 min incl. load +
  health pre/post. This IS the integration gate now (no separate smoke). I/smoke.sh unchanged (~145 s lease, one load) for workers.
  FN_BENCH_QUICK=0 restores the full ladder for qualification. All Lead A workers re-briefed with the directive.
- MERGE TRAIN 1 (fn-a3 3edbc4d) = int-desc 6d92e0f + H3 device MTP chain (85feb5b) + B evpool (b70/fn-rt-evpool) + R5 F1/drain
  (b70/fn-rec 3404c06). Building; quick bench train1-imm then train1-rec under PRIORITY.hold.
- GPU PRIORITY HOLD (orchestrator 03:25): file /mnt/vm_8tb/b70/results/tf-flashnext/PRIORITY.hold exists while an integration
  smoke+bench waits/runs; removed when [A-S] logs "released". All Lead A workers (R, PF, KF8, H4) told: before every gpu-run,
  while [ -e PRIORITY.hold ]; do sleep 10; done. LEAD B: please give your workers (codex-park, codex-arfuse, codex-vplan,
  codex-ttft, codex-evpool) the same rule now; codex-arfuse claimed at 03:26 over the hold.
- R5 HANG REPRODUCED + FIXED off-server (worker R, Claude Opus): real recorded forward soak (tf-xpu-fn-record-soak, no-MTP C4 bucket 4):
  OLD 7660564 HUNG 2/2 (windows 458, 308; L0 0x1); F1 4b1e0f4 PASS 5000/5000 + 3000/3000 shape-cycle with MTP interleaved.
  F1 = recorded Ready/Done/head/gate events never attached to a copy/kernel walker (payload -> barrier -> explicit AppendSignalEvent).
  Mechanism inferred (walker-attached completion signals on peer-reset events), not proven. Evidence R/fwd/{old-c4,old-c4b,
  fixed-c4,fixed-cycle}, R/report5.md. Next: + explicit immediate drain (B4g), smoke, serve pair r5-rec/r5-imm, then fn-a3 merge.
- SERVE ROW #5 (02:23, int-prep-desc = same build + XPU_FN_DESC=1; p0 exact, health OK; S/int-prep-desc-results.json):
    C1 72.0 / C2 109.3 / C4 141.2 / C8 177.6 agg tok/s; no-MTP C4 146.9; prefill 501 / 483 / 460. C4 +14%, C8 +21% vs DESC off.
  -> 6d92e0f makes descriptor windows the DEFAULT (XPU_FN_DESC=0 = legacy; paged KV keeps legacy); build+smoke+bench int-desc now.
- SERVE ROW #4 (02:06, int-prep = fn-int/fn-a3 bf8415b: hc2 + KF round 5 kernels + KF round 6 prepared inputs/pitched copies
  + kv8 off; RECORD off, DESC off; smoke-int-prep PASS hashes unchanged; p0 exact; health OK; S/int-prep-results.json):
    C1 71.7 / C2 101.4 / C4 123.4 / C8 146.2 agg tok/s; prefill 497 / 484 / 460 at 512 / 2k / 8k.
    vs comb-imm: C1 +3%, C4 +7%, C8 +9%, prefill +13%.
  Bench fix 2f6b5eb: rg is not on PATH in bench subshells -> the GPU-free pre-wait always ran its full 30 min (the "35 min wait")
  and the DEVICE_LOST check never matched; now grep.
- STAFFING (02:09, user execution mode): PF = new Claude Opus worker (agent aa613b80b8493625e, b70/fn-pf from bf8415b) porting
  Strata XMX prefill + HC v2 Q8 large-M + B's DPAS. R5 Claude Opus worker told: drain fix (B4g) first, then mirror Strata's
  one-graph window cross-card protocol. H3 (device MTP) and KF8 (GDN host setup) in flight.
- GPU PRIORITY ack (orchestrator 02:00): [A-S] integration serve_bench first, then B's ledger regen on that build, then workers.
  Current integration build: fn-a3 bf8415b (= hc2 + KF round 6 prepared inputs 103cfc2/9bbe16c + kv8 off + R2 code, RECORD off),
  smoke-int-prep PASS (hashes unchanged); serve_bench int-prep / int-prep-desc queued. Lead A takes fn-rt-evpool + evict round 2
  into fn-a3 as soon as B reports smoke PASS (merge -> smoke -> bench).
- R5 handed to a Claude Opus worker (00:08): the Codex session died (false-positive content filter, then the thread was lost). The work
  is intact: CPU model (R/report5.md) finds no H1/H2 source defect; F1 candidate transport fix (Ready/Done published by explicit
  SignalEvent after copy+barrier) + always-on pair manifest guard + standalone microrepro (r5-old vs r5-fixed) are uncommitted in
  fn-rec. Next: micro old must FAIL / fixed must PASS, then the serve pair r5-rec/r5-imm.
- LEDGER-hc2 OWNERS (orchestrator 23:50): Lead A AGREES.
  host.fn.mtp -> H3 (re-scoped 23:45: fully device-side draft chain, on-device argmax feeding the next embedding, no inter-depth
  readback, immediate path first). host.fn.gdn -> KF8, queued as soon as KF7 lands (KF7 is mid-build on b70/fn-prep:
  prepared inputs + pitched copies replace the per-row project() memcpy appends; re-measure launch_gap/d2d on it). launch_gap via
  recorded windows -> R5 once the hang is fixed. Lead B: verify_plan off the critical path, AR fused into consumer prologue,
  ledger chain binding + markers-only host-phase split.
- R2 RECORDED WINDOWS: NO-GO under the 2-failure stop gate (R/report.md, R_report3.md). Two serve hangs at no-MTP C4 + KF's at
  bucket 8 C2: card 0 waits in layer 1 MoE AR peer-ready while card 1 is already at the next attention AR (cross-card desync, L0 0x1
  after 30 s, rebind recovered, health OK). Cause unproven. NEW PATH (R5): CPU state model of the recorded AR event/slot lifecycle,
  Fable+Opus consult, then a device microrepro (not serve) that must fail before / pass after the fix. LEAD B: this touches your
  M-B2c recorded transport (beginRecording/recordAllReduceF32/device-reset events) - please review R's model when it lands
  (R/report5.md) and post any known replay-reuse constraints here. Orchestrator 23:45 hypotheses relayed to R (first checks):
  H1 ABA stale ready/ack across back-to-back replays (fix: epoch-tagged flags via a device replay counter, or parity double-buffered
  slots + final cross-card ack, or per-hop sequence numbers); H2 ranks choosing different buckets / AR counts in one window (add an
  always-on host assert that aborts cleanly instead of a 30 s hang).
- SERVE NUMBERS #3 (22:55, COMBINED fn-int/fn-a3 18be9ac = binary of 99e31ff: hc2 + kv8 off + R2 recorded windows; smoke-comb PASS
  hashes unchanged; p0 exact; pre/post HEALTHY; S/comb-r2-results.json, S/comb-imm-results.json):
    XPU_FN_RECORD=1: C1 86.4 / C2 108.1 / C4 143.2 / C8 183.4 agg tok/s; ITL p50 C1 8.9 ms; TTFT C1 85 ms; C8 TTFT 470 ms;
                     no-MTP C1 66.1 (15.1 ms/token); prefill 439 / 428 / 413 tok/s at 512 / 2k / 8k.
    immediate (off): C1 69.3 / C2 98.0 / C4 115.2 / C8 134.3; no-MTP C1 57.7, C4 115.8.
  Recording = +25% C1, +37% C8. KNOWN BUG: recorded no-MTP C4 (bucket 4, no drafts) HUNG/ZeFailed on R's run -> bench skips that
  control under RECORD (default FN_BENCH_NO_MTP_LEVELS=1); R is fixing it, so RECORD stays default-off until it is fixed.
  23:11 KF's prep-rec (fn-prep, RECORD=1) hit the same abort at bucket=8 during policy C2 (KF/round6/serve/prep-rec-server.log):
  the recorded-replay ZeFailed (L0 0x1) is INTERMITTENT and not tied to no-MTP C4. comb-r2's clean pass was lucky; RECORD unqualified.
  Gap to Strata DD: C1 86 vs ~141, prefill 428 vs ~2200.
- FOR LEAD B LEDGER (builds/tags as they land): int-hc3 = fn-int 186329a; hc2 = b70/fn-hc 9fb00f5 (S/hc2-results.json:
  C1 69.2 / C2 96.9 / C4 115.1 / C8 135.9 agg, prefill 2k 430).
  COMBINED BUILD READY: fn-a3 99e31ff = hc2 + kv8 (off) + R2 recorded windows (fn-rec 99bda20, incl. C2 cross-stream keep fix
  bc39d16). Worktree /mnt/vm_8tb/b70/build/tensorfold/fn-a3; build+smoke running 22:07, then serve_bench comb-r2 (XPU_FN_RECORD=1)
  and comb-imm (off). Ledger it with XPU_FN_RECORD=1. Lead A takes kernel work from B's ledger: KF6 = BYTES-hc2 items
  (one quantize per input, no per-row project() memcpy); H3 = MTP draft step (35 ms depth-3 window vs 18.4 ms forward).
  KF5 (fn-model 6cb4f56: GDN/QSA decode, expert dots) folds in via KF6's b70/fn-prep.
- SERVE NUMBERS #2 (21:19, fn-int 186329a = KF4 + HC kernels + serve wiring; S/int-hc3-results.json; smoke PASS, health OK):
    C1 65.5 tok/s (was 45.8, +43%), C2 92.0 agg, C4 110.5 agg, C8 128.9 agg; ITL p50 C1 11.6 ms; TTFT C1 97 ms;
    prefill 386 / 379 / 367 tok/s at 512 / 2k / 8k (was ~300). no-MTP: C1 54.2 (18.4 ms/token), C4 110.9.
  Read: one C1 forward is now ~18 ms, but an MTP depth-3 window costs 35 ms -> draft steps ~5 ms each (host-driven), so MTP
  only buys +21% at C1 and nothing at C4. Next levers: recorded windows (R2, host lag) and device draft loop (R3).
- 20:20: HC kernels (worker H, ee86b66) PASS full gates (teacher, 8-prompt serial==T2==T4, C4 T2/T4 == solo); profiled C1 T2
  kernel time/card 31.7 -> 17.4 ms (HC 20.1 -> 5.8 ms), C4 T4 68.8 -> 51.7 ms (HC 30.2 -> 12.9). hc_down R1 ~190 -> 16.6 us.
  Merged into fn-a3 b329393 with KF4; smoke + serve_bench running now (results -> S/summary.tsv, tag int-hc).
- fn-int = 8524431 (KF4 + serve wiring + bench baseline selector). R (recorded windows): R1 PASS, R2 in progress but Codex
  503s 20:12-20:20 were transient: at 20:52 all Lead A Codex workers (R4, KF6, H2) are active (logs < 1 min old), so no
  Claude handoff yet; will hand off any worker idle on 503 for > 10 min.
- FIRST SERVE-LEVEL NUMBERS (18:26, worker S, workers/S/report.md + first-results.json): tensorfold-native OpenAI serve,
  TP2 GGUF IQ3_XXS + MTP, fn-serve c3ac6cf (= fn-int 46afffb + server wiring), :18150, unprofiled, greedy,
  256 tok, p0 tokens == solo serial (exact), chat coherent, health pre OK:
    C1 42.4 tok/s (T2) / 45.5 (T4); C2 66.5 agg (33-38/stream); C4 89.9 agg; C8 111.8 agg (T2; T4 worse at C>=4);
    TTFT C1 107 ms (short prompt); prefill 309 / 299 / 286 tok/s at 512 / 2k / 8k.
  Not merged yet into this number: HC kernels (H; ~18 ms/window GPU), recorded windows (R; ~17 ms/window host lag), KF4/KF5.
  Strata DD reference: C1 ~141, C2 ~92/stream, prefill ~2200. Re-measure after each merge with S/serve_bench.sh.
- Milestones: M-A1..M-A4 DONE. M-A5 (speed) in progress: correctness on the fast path DONE (16:35).
- Integration: b70/flashnext-2card (fn-int) = 46afffb = M-A4 + push transport + Strata-parity q8_1 kernels (row-invariant)
  + device PLE + per-card kernel-timestamp profiler. All gates pass on it: teacher vs Strata oracle 547/554 top-1, KL <= 0.008,
  deterministic; serial == T2 == T4 (8 prompts); C4 / stagger / 3-run soak == solo. q8_1 default.
- Critical path (diagnostic, profiled), C1 T2 55 ms/window per card: kernels 31.7 ms (HC 20.1 ms: fn_hc_down 16.7 ms = naive
  324-work-item GEMV; dense 5.0; experts ~3) + host submission lag 16.8 ms (~1720 appends/card) + event churn 5.2 ms; exposed
  AR 1.65 ms. C4 T4 116 ms/window. Prefill p7 287 tok/s (no XMX prefill yet).
- USER PRIORITY (17:25): serve-level numbers first. Worker S builds/benchmarks the OpenAI serve on :18150 (first numbers
  today); I6 = 3-min smoke gate + KF4 merge; merges to fn-int on smoke pass, full gates only at milestones.
- Workers now: R (fn-rec) recorded windows: R0 GO (recorded 32.3 ms vs 31.8 ms floor, host submit 13.8 ms -> 39 us, bitwise);
  R1 invariant-immediate in progress. H (fn-hc) HC kernels (target hc_down <= 20 us/call). KF (fn-model) round 4: Q6_K,
  IQ4_NL down (82.5 -> 70.1 us, now beats Strata 72), prepared shared q8_1 activations.
- Lead B REQUEST (20:10, dev-loop speed): cpu-run --quiet blocks every build on the host (my xpu-native build waited
  behind a 28-min quiet park session). Please keep --quiet to the timed windows only, not whole GPU sessions/correctness runs;
  Lead A workers have the same rule (GPU.md note 19:25).
- Next GPU need: both cards in short leased runs (R, H gates), card 0 (KF, H microbench).

- [A-R R5 CPU model] Model/source audit published at workers/R/report5.md and fn-rec tools/xpu_trace/record_model.py;
  results workers/R/r5-model.json. Ideal binary-event semantics + actual both-queue drain show no H1 ABA/reset race in bounded
  same/alternate-bucket exploration; injected lost-ready matches marker10/14 but is NOT a demonstrated source defect. One b
  selects both rank lists, static96 ARs, dedicated channel1; no per-window host append touches recorded events in7660564.
  Adding cheap bucket/list-id/per-rank AR-count guard before either submit. Lead B: please review whether any legal device-reset/
  in-order event completion detail is missing, and known replay constraints beyond4e. Attribution currently UNKNOWN; no GPU yet.

- [A-R R5 response to B4g] Agreed H1/H2 are excluded under the modeled contracts; no epoch spin planned. Check item4's
  caller chain: the only Cache.execute caller is wf.forward, immediately preceded by Runtime.prepare -> Runner.syncAll ->
  TpComm.syncAll, which ALWAYS waits both immediate lists even when step==0 (tp.zig:163). No commands are appended between
  prepare and execute. Thus an execute-entry drain is useful API hardening, not a demonstrated hole in7660564. Will add
  event-status-at-abort and all-reset-after-drain diagnostics in the microrepro. Fable/Opus reviews pending, no GPU yet.

- [A-R R5 00:01] Consults complete (claude-fable-5-2 / claude-opus-5-5, read-only high). Both exclude app ABA/H2. Fable
  identified an unverified NEO lowering hypothesis: producer also waits on its walker-attached completion event; peer reset
  can strand that hidden waiter within the SAME window. Augmented CPU model reproduces exact rank0 hop3 / rank1 hop4 pattern;
  standalone event publication removes it. F1 candidate in fn-rec: copy(null event)->barrier->Signal(Ready), sum(local TS or
  null)->barrier->Signal(Done); same for recorded head/gate copies. Immediate unchanged. Not yet hardware-proven. Micro build
  queued, with exact766 tp.zig negative control, four shape phases and event-status dump. NO GPU/serve yet. Report5 updated.
  Fable also found B's old replay script UNSETS NEO workaround keys, unlike serve; new micro retains required keys throughout.

## Lead B (runtime, latency, features)  [updated 07:42 UTC]
- [B 07:42] OUT-OF-ORDER DISPATCH, card 0 (B/ooo/OOO-3edbc4d.md, b70/fn-rt-ooo 23f4cb17). Real GGUF weights, proxy activations,
  29 s lease, bitwise equal across modes.
  One layer's sibling kernels, device p50: C1 (4 rows) 35.4 us immediate vs 33.7 us OOO replay (-4.7%, no overlap observed);
  C4 (8 rows) 41.5 vs 35.5 us (-11.9%, all kernels overlapped).
  => about 0.1 ms per C1 window and 0.3 ms per C4 window over 48 layers. Below the 15% bar; NOT worth wiring now. Fusion (A) wins
  over concurrency.
- [B 07:42] B worker state: codex-arfuse PAUSED until [INK] frees card 1 (peer-store wiring built on b70/fn-rt-arfuse, untested
  on GPU). codex-trace, codex-park and codex-kv8: idle or paused. Lead B is at a milestone boundary; report sent to the orchestrator.
- [B 07:28] CARD-0-ONLY window (orchestrator 07:25). codex-arfuse is stopped: its peer-store wiring needs both cards, and gpu-run would
  have held card 0 idle while waiting for card 1. Its build is committed on b70/fn-rt-arfuse; it resumes when [INK] frees card 1.
  codex-trace -> card-0 out-of-order-list dispatch experiment on real GGUF tensors (b70/fn-rt-ooo, B/ooo/). The one-card harness
  is Lead A's H5.
- [B 07:10] HOP-LATENCY MICROBENCH (B/arfuse/peerstore, cc77748, 256 verified samples per case, 61 s lease). Producer start -> peer
  consumer end, p50/p99 us:
    40 KiB: copy-engine push 16.0/19.3; producer peer-only stores 11.2/20.2; producer local+peer stores 13.0/13.8.
    10 KiB: the copy case is a cold-order outlier, so it is inconclusive.
  => Peer stores save ~3-5 us per 40 KB hop, i.e. ~0.3-0.5 ms per C1 window (~100 hops), or 1-2%. Small but real. codex-arfuse is
  wiring it at the MoE producer epilogue (default off); quick gate + pinned A/B next. TRANSPORT-B section 5 "EU peer stores vs
  copy-engine push" is now measured.
- [B 07:08] ACK HARDWARE.md.
  Item 9: consistent with B's eviction design (D2H 4.5 GB/s, so the write-behind mirror stays).
  Item 8: confirms evpool; no bulk host reset anywhere on B's paths.
  Item 6: B will try an out-of-order side list for MTP/verify side work after the peer-store latency result.
  Item 7: B's earlier DPAS standalone prefill numbers must be re-checked for real-weight input before they are cited again. The e2e
  serve prefill numbers (+3.6-4.4%) used real GGUF weights and stand.
- [B 06:50] MTP LANE POLICY RE-TUNE on 3edbc4d (b70/fn-rt-lane 18c50fe; quick gate passed). Single runs, 128 tokens, tok/s (% of target):
    C2: depth 1 116.3 / depth 2 116.3 / depth 3 118.2 (78.8%); the old policy measured 116.9.
    C4: d1 151.4 (50.5%) / d2 140.5 / d3 136.5.
    C8: d1 191.9 (38.4%) / d2 165.7 / d3 157.4.
  => Deeper drafts LOSE at C4/C8: multi-row windows are GPU-bound, so extra verify rows cost more than the extra accepted
  tokens. The only change is C2 -> depth 3 (+1.1%, inside noise). Optional merge, low value. MTP depth is NOT the C4/C8 lever;
  multi-row kernel efficiency is (Lead A).
- [B 06:43] Lease fairness fix (bin/, uncommitted, tested on fake locks; orchestrator to commit): non-priority waiters now poll
  instead of blocking in flock. While PRIORITY.hold exists, the A-S/A-I run takes the next free lock. Before this, A-S int-pf2 sat
  25 min in the CPU queue behind builds that were already waiting in flock (flock is not FIFO).
- [B 06:15] ARFUSE (b70/fn-rt-arfuse b3e7e8e, default off; pinned A/B in one lease; tokens equal at C1/C4/C8).
  Result: AR launches per round dropped 204 -> 12 (C1), 208 -> 16 (C4), 224 -> 32 (C8), but lane-round p50 did not move:
  C1 33.27 -> 33.13 ms, C4 48.75 -> 49.00, C8 80.80 -> 82.09. Same base as da583ba.
  FINDING: the ledger's ~21 us per tp2_allreduce_f32 is waiting for the peer push to arrive, not launch or reduce cost. Fusing
  only moves the wait into the consumer. The lever is per-hop push LATENCY (~100 hops per window).
  Next (codex-arfuse): microbench copy-engine push vs producer-epilogue peer stores (EU writes straight into the peer inbox).
  Wire it if it saves >= 5 us per hop. NOT for merge yet.
- [B 06:05] CHEAP CHECKS (B/route/ROUTE-train1.md, 3edbc4d, 144 s lease, token gate OK; ranks agree on all picks):
  VERIFY-ROW OVERLAP IS REAL. At C1 the 4 verify rows hit 26.0 distinct experts per layer out of 40 assignments (uniform: 38.8).
  Per card: C1 13.0 (uniform 19.4); C4 30.2-30.7 (uniform 37.4).
    => A: grouped-by-expert decode (each expert read once for all its rows) saves ~1/3 of the expert bytes at C1 and ~20% at C4,
       on top of the uniform estimate.
  ROUTING SKEW across the 256/256 card split is SMALL: window imbalance mean 1.5-1.9%, p90 3.0-3.7%. Per layer at C8 it is 10.5%
  mean, 22.5% p90, but static rebalancing is modelled at only ~0.07 ms/window. => DROP expert rebalancing.
  C8 with 8 distinct prompts (4 full 16-row windows): 54.7 / 54.5 distinct experts per card per layer, vs 69.3 uniform (-21%).
  Per-layer imbalance is 8.3% mean, 17.5% p90. So the C8 expert union is ~4.6 GB/card/window, not the 5.85 GB in LITERATURE.
- [B 05:58] keep/settle (b70/fn-rt-keep 03a319a): neutral. C1 27.51 -> 27.62 ms, C4 40.18 -> 39.37 ms, one sample under load. NOT for
  merge. The "peer copies" are the AR pushes themselves (96 x 40 KB per window), so they are covered by arfuse.
  codex-trace -> cheap checks (orchestrator 05:55): verify-row expert overlap and routing skew across the 256/256 split
  (B/route/ROUTE-train1.md).
- [B 05:33] ACK orchestrator 05:32 (stay on the fork; kernel efficiency + fusion). B owns item 1. AR -> consumer prologue: codex-arfuse
  prototype (b70/fn-rt-arfuse, XPU_FN_AR_FUSE=1) passed the C1/C4/C2 token gates; its timing is queued behind PRIORITY.hold. Next step
  after the timing: issue the push right after the producer (before the local tail work), so the consumer prologue never waits.
  The ledger kernel-table delta (tp2_allreduce_f32 102 calls -> 0) comes with that report.
- [B 05:05] USER TARGETS (C1 100 / C2 150 / C4 300 / C8 500, prefill 1500). B plan, value-ranked:
  (1) MTP lane-policy re-tune, depth 1..3 at C2/C4/C8 on 3edbc4d (codex-park, b70/fn-rt-lane).
  (2) AR fused into the consumer (codex-arfuse; 3.7 ms per C1 window summed over the cards).
  (3) host keep/settle + peer copies (codex-trace, b70/fn-rt-keep; 2.1 + 2.0 ms).
  (4) RECORD vs evpool: NOT either/or any more. a9c6725 lets them coexist. R's rec2 epoch transport is the one-submit-per-window
      path; B hands R the fix and reviews rec2. ttft paused, per the orchestrator.
- [B 05:00] DECISION LEDGER, train1-imm 3edbc4d (B/ledger/LEDGER-train1-imm.md; one 246 s lease, health OK, tokens equal across modes):
  ANSWER: the C1 window is now GPU-BUSY-BOUND, not host-bound. Busy is 68% of the traced window on each card. Stay on the fork.
  Per card per window, traced (full trace adds +12.6% at C1, +12.1% at C4), ms:
    | load | card | wall | busy | host CPU | launch gap | peer | other |
    | C1 | 0 | 31.0 | 21.1 | 3.1 | 3.1 | 0.17 | 3.6 |
    | C1 | 1 | 31.0 | 20.9 | 3.6 | 3.2 | 0.08 | 3.3 |
    | C4 | 0 | 44.2 | 29.8 | 4.9 | 3.7 | 0.00 | 5.8 |
    | C4 | 1 | 44.2 | 29.1 | 5.8 | 3.7 | 0.18 | 5.4 |
  Markers-only (untraced) lane-round p50: C1 27.4 ms for ~2.03 tokens; C4 39.4 ms for ~6.85 tokens. Host MTP phase 4.1 / 3.6 ms.
  ARITHMETIC: removing ALL non-busy time (27.4 -> ~21 ms) caps C1 at ~2.03 / 21 ms = ~97 tok/s. C1 >= 110 therefore NEEDS
  (a) fewer busy ms per window (kernels) and/or (b) more tokens per window (Lead A: 2.27 now vs Strata ~3.3). At 21 ms and
  3.3 tokens, C1 would be ~157. Recorded windows or the epoch transport are worth at most ~1.2x now.
  Top-8 C1 exposures, summed over both cards: launch gaps 6.3 (R); hc_down 3.7, tp2_allreduce 3.7, hc_upmix 3.5, q6 dense 2.5
  (A kernels; the AR is B's arfuse); host MTP 2.4, host keep/settle 2.1, peer copies 2.0 (B runtime).
- [B 05:00] B next, value-ranked: arfuse (AR fused into the consumer, 3.7 ms) continues. B takes host keep/settle (2.1 ms) and
  host MTP glue (2.4 ms, with H3/H4).
- [B 04:55] ROOT CAUSE of train1-rec "ZeFailed at load" = evpool bug (B): the recorded-window channel creates a SECOND TpComm
  (record_comm), whose evpool init calls gpus[0].registerCleanup again, and ze.Gpu has a single cleanup slot, so it returns
  ZeFailed (right after events_created=2048). FIX: b70/fn-rt-evpool-int a9c6725 evpool: share one process event-pool manager across TpComm channels. One process-wide manager is
  shared by every TpComm; it registers once and is cleared at shutdown. tp/evpool unit tests 5/5 pass.
  Cherry-pick of a9c6725 onto 3edbc4d is clean (checked with merge-tree).
  For R (rec2), when RECORD returns: cherry-pick this commit. No GPU run yet.
- [B 04:35] DECISION LEDGER on train1-imm 3edbc4d STARTED (codex-trace, orchestrator 04:32): one <= 5 min lease as B70_AGENT=A-S-ledger
  (passes PRIORITY.hold), markers + one full pass at C1/C4, answer GPU-bound vs host-bound. ETA ~05:05, gated by the tracer
  merge/build (~10 min CPU). Output B/ledger/LEDGER-train1-imm.md.
- [B 04:10] VALUE CHECK applied (orchestrator 04:08). codex-kv8 WAS codex-ttft (the same worker): PAUSED. codex-ledger (regen on the
  bf8415b/evpool side build): PAUSED; next run is markers-only, once, on train1 when it exists. codex-park: C8/C16 round DONE
  (2c5d885), now idle, no GPU claims. codex-arfuse: continues (the only B worker that may claim). It yields to PRIORITY.hold
  automatically, because bin/gpu-run waits on the file.
- [B 04:03] EVICT ROUND 3 READY TO MERGE: b70/fn-rt-evict 2c5d885, already merged with fn-a3 6d92e0f; post-merge 2-prompt C2 gate
  passed. Parked streams now RELEASE their physical state slot, so logical streams can exceed physical slots. P=2/L=6 smoke == solo,
  with 10 restores into a different slot. Env: XPU_FN_EVICT=1 XPU_FN_KV=int8 XPU_FN_EVICT_SLOTS=N XPU_FN_EVICT_QUANTUM=N.
  Single runs, 2K prompts, 256 tokens, pre-merge build:
    C4/P4 44.9 agg; C8/P8 54.0; C16/P16 78.9 (16 slots fit sixteen 32K contexts).
    C8/P2 q4 54.0 (146 parks); C16/P2 q4 53.5 (283 parks), same aggregate as full fit. ITL p99 2.3-4.4 s at q4; q1 has a better
    tail but loses aggregate (20 agg).
    TTFT p50 is 12 s (C4), 35 s (C8), 77 s (C16): PREFILL-BOUND. Prefill is the lever for C8/C16 (Lead A prefill worker +
    B ttft scheduler).
- [B 03:42] CORRECTION (orchestrator 03:40): evpool was HANDED to Lead A at 02:16, not merged; it is not in 6d92e0f. For the merge
  train use b70/fn-rt-evpool-int 740aca0 (bf8415b + 4 commits: tp.zig, tp_event_pool.zig, ze.zig). merge-tree onto 6d92e0f:
  CLEAN_ON_6d92e0f. Build needs PATH=/mnt/vm_8tb/b70/build/tensorfold/toolchain/zig-0.17.0:.../toolchain/bin.
  XPU_TP_EVPOOL=0 gives the old path, so serve_bench can A/B it in the same build.
- [B 03:30] PRIORITY HOLD LIVE (orchestrator 03:25). bin/gpu-run and bin/cpu-run (uncommitted edit, tested on fake locks) now
  check /mnt/vm_8tb/b70/results/tf-flashnext/PRIORITY.hold before taking a NEW lease. While it exists, only agents whose
  B70_AGENT starts with a prefix listed in it (now "A-S A-I") get through; everyone else waits (bounded by their lock timeout,
  exit 124). A hold older than 60 min is ignored. Mid-lease runs are unaffected. A watcher removes the hold when [A-S] logs a
  release in GPU.md. Lead A: for every later integration bench, echo 'A-S A-I' > PRIORITY.hold while queued and rm it after.
  B also freed the CPU lease (cancelled its own compile check). codex-arfuse is mid-lease until <= 03:38.
- [B 02:16] EVPOOL READY FOR THE INTEGRATION LINE (orchestrator item 1). Lead A, merge it: b70/fn-rt-evpool-int 740aca0 = bf8415b
  plus 4 commits touching only zig/src/xpu/tp.zig, tp_event_pool.zig and ze.zig (clean cherry-pick, no tracer). Worktree:
  /mnt/vm_8tb/b70/build/tensorfold/fn-rt-evpool-int (compile check queued behind the CPU lease). Default ON; XPU_TP_EVPOOL=0 gives
  the old path. Fresh events are made in 32-step chunks (128 events) by a worker thread off the critical path and retired after
  syncAll. No host reset or reuse (TRANSPORT 4c kept).
  Measured on da583ba+evpool (fn-rt-evpool 3404c06). Markers-only, one sample each, same lease. Tokens identical.
    C1 lane-round p50 33.17 -> 28.01 ms (-15.6%), 71.1 -> 82.2 tok/s.
    C4 50.25 -> 43.36 ms (-13.7%), 58.4 -> 61.6 tok/s.
    Zero spare waits. 94,592 events created, the same number destroyed.
  Please run serve_bench [A-S] on fn-int + evpool + evict round 2 (a139dab) and append the row.
- [B 02:02] ACK GPU priority (orchestrator 02:00). HANDOFF TO LEAD A for the integration line NOW: evict round 2 = b70/fn-rt-evict
  a139dab (already merged with fn-int bf8415b, smoke passed; B/evict/HANDOFF.md). evpool (b70/fn-rt-evpool) will be handed over
  the moment its smoke passes; ETA ~02:30. After the next [A-S] bench, B regenerates the ledger on that build (regen.sh).
- [B 01:55] EVICT ROUND 2 READY FOR LEAD A MERGE: b70/fn-rt-evict a139dab (already merged with fn-int bf8415b; post-merge C4 smoke has
  four outputs == solo, 11 parks and 11 restores). Enable: XPU_FN_EVICT=1 XPU_FN_KV=int8 XPU_FN_MIRROR=1 --prompt-cache-gib N.
    Park p99 586 -> 118 ms: the outlier was first-use pinned-frame allocation (324 of 395 ms), now preallocated.
    Write-behind mirror: park wall 14.4 / 14.4 / 14.6 ms at 1K / 8K / 32K, vs 32.0 / 30.0 / 76.3 off. Bytes per park per card are
    a flat 62 MB (state + open tail). Decode ITL p50 cost -0.1..+0.2%.
    RAM-tier prompt cache: an idle snapshot is parked, not dropped. A repeat prompt restores 21/22 rows, tokens are identical,
    TTFT 25.4 ms (restore 5.9 ms).
  Open: ITL p99 under forced 2-slot pressure is still 0.55-0.84 s (time-slice quantum); untested cache fault paths.
  codex-park -> parked streams release their state slot, so logical streams > physical slots; C8/C16 bench.
- [B 01:50] ORCHESTRATOR ITEM 2 CLOSED. host.verify_plan EXCLUSIVE time is 0.06 us at C1 and 0.18 us at C4 (b70/fn-rt-vplan
  814ef7f, XPU_FN_PHASE_STAMPS=1). The ledger marker was inclusive of the whole verify forward and its waits, so there is
  nothing to move. The real host cost there is L0 API calls: event churn (B, fn-rt-evpool), launches (R5), per-row copy appends
  (A KF7). Under-load sample da583ba: C1 33.1 ms lane-round / 71.0 tok/s, C4 50.4 ms / 58.5 tok/s (32-token streams).
  codex-kv8 -> B3-7 cold C4 TTFT / scheduler (b70/fn-rt-ttft).
- [B 01:25] HOST SPLIT on da583ba (B/ledger/HOSTSPLIT-int-da583ba.md): per card per C1 window, L0 API time 10.7-12.1 ms (C4 17-19.6),
  our Zig code 1.5-1.9 ms, kernel-argument API calls 1.2-1.3 ms, ~1,870 launches and ~900 copies (800 of them unsignalled).
  BIGGEST SINGLE ITEM: TpComm.syncAll destroys and recreates ALL MAX_PIPE_STEPS*4 shared events on every sync. That is 8,192
  events per C1 window (10,240 at C4) and ~8 ms of traced API time per window (C4 ~9.8 ms). FIX in progress, B-owned (codex-trace,
  b70/fn-rt-evpool): lazy chunked pools, fresh-event rule kept, replacements pre-created off the critical path.
  Note for Lead A / R5: pre-baked arguments alone do not remove the host bound. Recorded windows remove the appends; evpool
  removes the churn.
- [B 00:30] LEDGER v2 on fn-int da583ba: B/ledger/LEDGER-int-da583ba.md (12 KB) + .json. The timelines conserve wall; unknown idle
  is < 5% per card (was 85-96%). Markers-only overhead vs off: +0.76% C1, +0.35% C4. USE THE MARKERS FOR A/B. Full trace is +29%.
  Per card per window (traced), C1 / C4: busy 22.0-24.5 / 34.7-38.8 ms of 42.9 / 65.4; host CPU 12.4-13.5 / 16.9-19.5;
  launch 3.8-4.0 / 6.6-6.7; peer <= 0.48 ms. Untraced markers p50 (inclusive spans, so they overlap), C1 / C4:
  MTP 9.2 / 12.7 ms; GDN host 6.9 / 14.0 ms; verify span 23.2 / 35.3 ms. The verify span INCLUDES the whole verify submit,
  so "verify_plan" alone is small and the exclusive cost is per-layer host submit. Readback 0.14 / 0.27 ms.
  Regenerate: B/ledger/regen.sh <worktree> <tag>.
- [B 23:55] TO LEAD A / R5 (hang review, TRANSPORT-B 4g): stale-signal ABA and bucket/AR-count mismatch are EXCLUDED by
  construction (each event is signalled and device-reset once per replay, HOST scope, both queues drained; one entry serves both
  ranks; exactly 96 hops). The signature is a lost or stale signal. Do first: (4) drain the immediate lists at execute() entry,
  a real hole when staggered C4 prefill runs; (1) abort-time event status dump; (2) post-drain all-reset assert (diag);
  (3) parity double-buffered entries as hardening and ABA test. Epoch flags would need device spins; not recommended.
- [B 23:55] ACK orchestrator 23:50 owners: B takes verify_plan off the submission path, an AR-into-consumer fusion study, and the
  ledger tool fix (chain binding + an unprofiled markers-only phase split). Workers (23:58): codex-kv8 -> verify_plan
  (b70/fn-rt-vplan, :18155); codex-pushar -> AR fused into consumer prologue (b70/fn-rt-arfuse, :18156, XPU_FN_AR_FUSE=1);
  codex-trace -> ledger v2 (timelines, unprofiled phase split, md summary) regenerated on da583ba. All from fn-int da583ba.
  LEDGER-hc2 delivered: B/ledger/LEDGER-hc2.md/.json (tracer overhead: full +20% C1 / +28% C4 vs markers; markers ~= off).
- CRITICAL-PATH LEDGER (user 21:50, orchestrator ask): codex-trace on fn-hc e12b9610 + tracer v2 (b70/fn-rt-ledger, serve :18154),
  C1 and C4 steady windows, per card: every kernel (ns, GB/s, % of 608, bytes needed), every queue gap with cause, every data
  movement, totals, top-10 recoverable us with owner. Output B/ledger/LEDGER-hc2.{md,json} + regen.sh for each Lead A merge.
  ETA ~23:00 UTC. Lead A: one short both-card lease per capture.
  BYTES MODEL READY (B/ledger/BYTES-hc2.{md,json}, CPU model of fn-hc e12b9610 [I], Claude claude-opus-5-5 worker): bytes a window
  SHOULD move per card: C1 5.26 GB -> 8.65 ms at 608 GB/s vs ~32.7 ms measured (26%); C4 7.37 GB -> 12.1 ms vs ~55 ms (22%).
  Largest: routed experts 1.64/3.16 GB; HC readout weights 1.33 GB/window/card (13.2 MB per sublayer, replicated on both
  cards, read regardless of rows = 25% of C1 bytes); dense 0.95 GB; MTP head 170 MB/pass; LM head 218 MB/verify; GDN state
  226 MB/stream. Code-level waste: fn_hc_down_bf16_1 re-reads its 6.6 MB weight per 4-row grid row (+~705 MB/card at C4);
  project() re-quantizes the same input per matrix (3x/layer) + 768 (C1) / 1536 (C4) per-row memcpy appends per window.
  FOR LEAD A: HC weights replicated per card -> split HC readout across ranks (TP) or fuse; one quantize per input.
- HEADLINE (B3-5, b70/fn-rt-park2 6b6af10, fn-int 46afffb + kv8, two cards, push transport, int8 KV, unprofiled, 1 sample):
  park to host RAM / restore without recompute, per card (both in parallel):
    1K: 66 MB, park 17.1 ms, restore 6.1 ms | 8K: 121 MB, park 29.9 ms, restore 10.2 ms | 32K: 310 MB, park 75.6 ms, restore 24.9 ms
  (D2H ~4.2 GB/s, H2D ~13 GB/s; restore to first window 89-110 ms incl. one window). K=7 T2 parked == unparked tokens.
- EVICTION ON THE REAL SERVE (b70/fn-rt-evict 4275ddc, fn-int 8524431 + kv8 + park2 + admit; XPU_TP=2 XPU_FN_KV=int8
  XPU_FN_EVICT=1, XPU_FN_EVICT_PAGES=2 to force pressure; :18153, unprofiled, 1 sample, 256 tok greedy):
    C2 uncapped 48.8 agg (ITL p50 40.0 ms); C4 on a 2-stream cap 43.4 agg (89% of C2), 44 parks + 44 restores, all four outputs
    == solo, each prompt prefilled once (no recompute). Park wall p50/p99 43.7/585.9 ms, restore 37.1/42.6 ms. C4 TTFT
    117/307/1352/1768 ms; ITL p99 1.1-1.6 s = time-slicing quantum. READY for Lead A merge (HANDOFF in B/evict/).
    Limits: prompt cache off in eviction mode; parked streams keep their state slot. Next: p99 park fix, write-behind mirror,
    idle Snaps as RAM-tier prompt cache.
- DPAS prefill smoke (b70/fn-rt-dpas, XPU_FN_DPAS=1, Q6_K dense only): off == fn-int hashes; on: first-token top-1 equal,
  KL <= 0.0047, greedy agrees 11-32/32 tokens (non-bitwise path, expected), chunk 512 vs 2048 bitwise equal, C2 == solo.
  Prefill (fn-int 46afffb, unprofiled): off 362 / on 378 tok/s at 2048 ids (+4.3%), 20K: 350 -> 363 (+3.6%). Profiled
  128-row chunk, sum of kernel time per card 310 ms: HC down 80.7 + HC up 68.0 = 48%, Q6 DPAS 20, BF16 K2560 14, IQ4_XS 13,
  IQ3_S expert up 12. => FOR LEAD A / worker H: HC is also the prefill critical item; H's hc_down/up must be fast at M=128-2048
  rows, not only decode R<=16. Lead B can put HC on the DPAS path for rows > 16 if H does not cover it - say which.
- Ack Lead A 20:10: Lead B workers told to use cpu-run --quiet only around timed windows.
- 20:52 Codex 503 check: both Lead B Codex workers (prefill DPAS, serve eviction) are live (WebSocket -> HTTPS fallback, logs
  advancing); no handover needed. Will switch to Claude workers if either stalls > 10 min.
- READY TO MERGE (Lead A): B3-3 paged KV, b70/fn-rt-kv8 36738f3 (= fn-int 46afffb merged at 004ef6e), default off,
  XPU_FN_KV=int8|bf16page. Smoke: off == your 46afffb hashes (3 x 32), bf16page == off. Teacher 2420 pos: int8 dNLL +0.0001,
  top-1 99.0% vs bf16; 32K 8/8 vs Strata. KV bytes/token/card: int8 7,696, bf16page 14,144 (legacy bf16 more; see HANDOFF).
  int8 smoke c8d48ab: 3 hashes recorded, C2 == solo; +1 MiB/card allocated during C2 (unexplained, small). int8 chunk 512 != 64:
  KV append is per-token absmax (same kernel both widths); the >16-row passes select different ROUTER kernels - likely a
  forward-path (Lead A) row-invariance gap at 512 rows, not KV; check bf16page/legacy at 512 vs 64 to confirm. B/kv8/HANDOFF.md.
- Lead A first serve (C1 42 tok/s, prefill ~300 tok/s): prefill is the Lead B lever -> DPAS wired in b70/fn-rt-dpas (flag
  XPU_FN_DPAS=1), smoke + prefill tok/s pending a lease.
- SPEED DIRECTIVE (17:25) applied: kv8 and park2 wrapping up now -> merge fn-int 46afffb, one fast smoke each, HANDOFF.md for
  Lead A to merge (B/kv8/HANDOFF.md, B/park2/HANDOFF.md, ETA ~18:45). park2 delivers unprofiled park/restore ms at 1K/8K/32K.
  DPAS prefill next gets wired into fn-int prefill behind a flag (Lead B branch) for a real prefill tok/s on the serve.
- FOR LEAD A, ready now:
  * Tracer for your path: b70/fn-rt-trace2 d1a8a93 (based on fn-int bfb427f; default off, XPU_TRACE=<fresh dir>,
    XPU_TRACE_MODE=markers for baselines; analyzer: tools/xpu_trace/analyze.py TRACE --output OUT --window a4_window --streams N
    --rows R). Overhead 1.3-3.6%. First profile: JOURNAL [B] 13:20, B/trace2/RESULTS.md (C4: card 0 idles ~28 ms/window waiting
    for card 1, whose per-row GDN/attention appends are late).
  * Push all-reduce: b70/fn-runtime 611566d (TRANSPORT-B 4d).
  * Recorded-list transport (M-B2c) in b70/fn-runtime (TRANSPORT-B 4e: beginRecording / recordAllReduceF32 / finish, device-reset
    events, 10,000 replays clean at 10-160 KB x 96-192 hops, ~6 us host submit per window, flat p99). Counter-based events FAILED.
  * PLE host memory ANSWERED (TRANSPORT-B 4f, B/pushar/PLE_HOST.md): file-mapping import fails even with unlimited memlock
    (anonymous 4/64 MiB imports work). Use one zeMemAllocHost filled by O_DIRECT reads (3.85 GB/s measured, ~7.5 s for
    28.8 GB) + fadvise DONTNEED on the file range; the page cache is reclaimable, so the one USM copy is the only hard cost.
- Milestones: M-B1 DONE (one card 0.3-0.5% overhead; two-card profile done). M-B2, M-B2b, M-B2c DONE.
- M-B3: B3-1 + B3-2 DONE (b70/fn-runtime 620851f): 27B park/restore without recompute, bit-identical (bf16/int8, DFlash2,
  same/other slot, K=1/7/64); park 16K 155 ms (D2H 4.5 GB/s bound), restore 53 ms; concurrent T p50 +2.1% (gate 2%: narrow miss).
  B3-3 paged KV (codex-kv8, b70/fn-rt-kv8 from fn-int): bf16page chunk invariance (1/3/5/16/64) + 512-token seam bitwise ==
  legacy, 32/32 ids == legacy; int8 drafted == serial (p0, p1 full 256); M-A4 suites + teacher-forced quality gate running.
  B3-5 (Flash-Next two-card park/restore on the page pool): codex-park, b70/fn-rt-park2 from b70/fn-rt-kv8.
  B3-6 host side: b70/fn-rt-admit 853d273 accepted as prototype (18/18 host tests; sim [I]: C4x200K 0.25% lost, C8/C16x200K
  15.5% lost with full-copy parks -> write-behind mirror needed); fairness fixed + mirror model e0db34d [I: C16x200K lost 20.6% -> 9.3% with mirror].
  Prefill (B/prefill/CRITICAL_PATH.md): same lease, 2048 ids, chunk 512: pre-KF 55.8 -> fn-a3 integration 362.2 tok/s
  (6.49x; first token equal). Pre-KF 20K trace: dense GEMM 60% of busy, experts 25-28%, HC 6%, AR 2-5%, GDN 0.4%; the
  16-row tile kernels re-read weights per tile. ~6x more needed for 2200 tok/s.
- PROPOSAL TO LEAD A (prefill DPAS): DESIGN-A lists "DPAS tiles for R > 16 (prefill)" under M-A5. Since cold C4 TTFT is in
  M-B3 and KF is on decode gaps, Lead B has started a standalone, default-off DPAS prefill GEMM (dense first, then grouped
  experts) on b70/fn-rt-prefill3 with parity, determinism and row-invariance tests, card 1 only, no edits to your trees; you
  wire it. Say so in your section if you would rather own it or have KF do it, and Lead B will hand over the branch.
  DPAS (standalone, card 1, device timestamps): Q6_K dense M=512 7.4x / 8.5x vs Q8_1 (QKV 2560x5120 295 us, out 3072x2560
  156 us; GEMM 69-97 TOPS), M=2048 9.9-10.7x; fp64 rel L2 2.1e-4, bitwise row-invariant, repeatable. Grouped IQ2_XS/IQ3_XXS/
  IQ4_NL expert projections correct; expert speed vs baseline being measured.
- Next GPU need: both cards in short runs (kv8 gates, park2 gates/latency), card 1 short runs (DPAS prefill microbench).

[A-KF 2026-11-21 11:04 UTC] Round8 C9 expert union: full direct ROW_BITS all5 gate/up formats and both down formats PASS, plus prepared/unprepared public API bits PASS; fresh sampled FP64 dequant(W)*dequant(q8_1(X)) errors6.0e-8..9.3e-8. Card0 claim41s, health pre/postPASS/reset0; no card1 or serve. A/B60332e35 already delivered. C timing queued under cpu-run--quiet, waiting normal LY CPU build; fixed-grid256WG, 1GU+1down launch; no KF upload/layout edits.

[A-KF 2026-11-21 11:13 UTC -> A] C9 union timing has waited for cpu-run--quiet since11:02 (builds LY/PF/HS). Next free CPU+card0 overlap needed for frozen5-type real-bank capture, ~1min GPU, max250s; no GPU is held while waiting. Offline C10 register-pressure probe also queued as normal heavy CPU work. No priority/lock bypass. Existing bit+FP64 gates PASS; C remains uncommitted until latency selection.

- [A-H -> A-PF, round10] I am leaving fn_q8_linear_q5_k_c1 unchanged. Your q5_k head path starts at R=2; is R1 draft-head work yours, or may H test a candidate? H owns drafter-only HC Q8 wiring and cross-step dependency analysis on fn-hc.
- [A-H -> A-HS, round10] MTP loader currently never fills GHcW.down_q8/up_q8, so drafter HC cannot use your Q8 path. I will reuse q8Split and existing HC kernels under a drafter-only default-off flag; no main HC/kernel edits. Please flag any Q8 HC caveat.

[A-KF 2026-11-21 13:17:34 UTC] Round9 integration merge cb11ab12 in progress on fn-prep: took the entire integration tree then reapplied only router M4 tiles and default-off expert union. No LY producer/layout diff; split down explicitly falls back to integration kernels. CPU build queued normally since13:08; ROW_BITS/Q8_EXPERT/slice gates prepared, card0 only. No GPU held while waiting.

[A-KF -> A 2026-11-21 13:38:25 UTC] ROUND9 MERGE-READY f3382a6f on b70/fn-prep (parents cafb9ec6 + cb11ab12). Entire integration tree retained, then only router M4 and opt-in union reapplied. No LY producer/layout diff; split-down fallback explicit. All15 dense ROW_BITS both activation modes zero differences; Q8_EXPERT all5pairs off/on PASS; union/fallback99 ROW_BITS cases + sampledFP64 PASS; exact supplied slice core48/tail32 hashes + keep-state checks PASS. Worktree clean. Handoff workers/KF_report9.md; dispersed FN_TIMING_ROTATE=1 capture continues. Card0 only, all healthPASS, no reset/push.

[A-KF -> A 2026-11-21 13:44:50 UTC] Round9 merge-ready f3382a6f already handed off; report9 correctness complete (exact S hashes, all ROW_BITS/Q8_EXPERT gates). Only final rotated dispersed R1/4/8/16 timestamp bundle remains, ~20s card0 / <=250s cap, waiting cpu-run--quiet since13:41 behind INK build. Card0 is NOT held while waiting. New test-only TP2 fixture prints both rank occupancies and rank0 working-set size; default serving kernels/layouts unchanged.

[A-KF -> A 2026-11-21 13:48:03 UTC] Timing capture completed rc0 in7s (all5pairs real split bank FN_TIMING_ROTATE1). Post-health 20s probe timed out, no DEVICE_LOST in capture. GPU.md released. xe-reset tool mutates/stops BOTH cards so NOT run under INK card1 reservation. One card0-only health recheck at standard60s queued; timings remain unqualified until it passes. No kernel reruns. Merge f3382a6f correctness gates remain clean.

[A-KF -> A 2026-11-21 13:48:41 UTC] Card0 standard60s post-health recheck PASS in16s, no reset/card1 touch; capture rc0 in7s and all hashes completed. Original20s timeout retained in raw logs, kernel journal has no xe/GuC/reset fault lines in capture+health window. Timings now qualified with explicit health-recheck caveat. f3382a6f remains merge-ready; final report9 table/proposal being written.

[A-KF -> A 2026-11-21 13:52:08 UTC] ROUND9 COMPLETE f3382a6f, clean fn-prep. Report workers/KF/report9.md; final handoff workers/KF_report9.md. Real split bank, FN_TIMING_ROTATE1 + dispersed global512 seed: R8 rank0 40pairs/37experts (34x1+3x2); R16 80/67 (54x1+13x2). Default already reads weights once. IQ3_S R8/R16 GU113.02/198.85us (76.4/78.6% of608 split bytes); IQ2_S100.99/182.76us (63.2/63.3%). All5pairs R1/4/8/16 table and byte ideals included. ONE proposal: one-chunk-ahead register pipeline for E1/E2 split GU, not implemented. Capture7s quiet/card0, no reruns. Initial20s post-health timeout preserved; standard60s recheck PASS16s, no reset/card1 touch or xe fault journal lines. All required merge gates and exact S hashes PASS; merge-ready remains unchanged.

[A-KF -> A-LY 2026-11-21 14:01:01 UTC] ROUND10 claim: touching fn_q8_expert.cl fn_q8_ups_* / QUPS_PASS only for one-chunk-ahead E1/E2 split GU software pipeline, IQ2_S first. No layout producer, down, Q6_K rs, forward or recorded changes. Will stop if IQ2_S R8/R16 gain is <5%; depth1 only, inspect ISA for spills. Card0 short leases only.

[A-KF -> A/A-LY 2026-11-21 14:21:04 UTC] Round10 IQ2_S depth1 pipeline ready: 128GRF/SLM9216/no spills; actual next loads overlap32/64 dp4a in E1/E2. Split ROW_BITS+FP64+48 exact core slice hashes PASS. Need only ~3s card0 + cpu-run--quiet for frozen baseline/candidate rotated-disperse A/B; queued since14:13, first CPU opening rejected75 because card0 busy, released immediately. No other-type port/timing sweep until >=5% at BOTH R8/R16. No GPU held while waiting.

[A-KF -> A/A-LY 2026-11-21 14:23:50 UTC] ROUND10 COMPLETE, STOP (>=5% rule missed). Real rotated dispersed IQ2_S GU baseline->pipeline R8 100.938->106.146us (+5.16%), R16 183.750->189.375us (+3.06%); R1 +20.32%. One matched quiet3s A/B, no further types or sweep. Actual E1/E2 overlap verified32/64 dp4a; 128GRF/SLM9216/zero spills. Full split ROW_BITS+FP64+48 exact slice hashes+timing hashes PASS, health pre/post PASS. Rejected production fn_q8_ups pipeline RESTORED to f3382a6f; release expert QUPS_PASS edit claim. Commit275b9643 contains ONLY direct split-GU bitwise test mode/docs. Reports workers/KF/report10.md and workers/KF_report10.md; clean fn-prep, no push/card1/reset.

[A-KF -> A/H/HS/R/LY 2026-11-21 14:27:36 UTC] ROUND11 claim: route_topk+descriptor moe_compact fusion and q8_1 producer-side fusion. Updating fn-prep to accepted integration8f2258aa first, omitting R10 test-only275 content as requested. Will add fused kernels/helpers and host selection in xpu_ops/store/window bindings; HC arithmetic unchanged, only q8 side-output where whole32-value blocks are owned. No expert layout/Q6rs/head/attention changes. Card0 only, short leases; direct plan/q8-byte bits before slice timings.

[A-KF -> A/H/HS/R 2026-11-21 15:06:11 UTC] ROUND11 candidate now building after normal CPU lease wait: standalone fn_moe_latency/fn_hc_producer/fn_shared_producer files, no edits to HC original arithmetic or LY layouts. Fixed one-WG topk+plan; exact q8 tails from HC mixed/GDN norm/QSA gate/shared SwiGLU. Two stable prepared banks with append-time generation/expiry checks; recorded grid/arg contract retained. Card0-only byte+plan fixture and slice immediate/recorded-OOO ablation prepared; no GPU touched yet. MTP BF16 input norms retain old path (no redundant q8 output).

[A-KF -> A 2026-11-21 15:50:14 UTC] ROUND11 v2 correctness complete: 2101 direct plan/producer/shared cases plus HC54 row cases, all15 dense types both activation modes 11640 ROW_BITS cases, all5 expert fp64 gates PASS. v1 was byte-clean but rejected on measured latency (+260us slice R16); v2 uses 16B/lane HC block reads, NC1/2/4 shared tails and adaptive route SG allocation, all active ISA resources zero scratch. Final 17s four-variant slice timestamp capture ready; cpu-run--quiet has waited normally since15:41 behind PF/HS/LY builds. No GPU held while waiting. No merge-ready claim until this capture validates net latency. Card0 only; all healthPASS, no reset, no two-card/serve run.

[A-KF -> A/LY/HS/PF 2026-11-21 16:22:31 UTC] Coordination: R11 final producer candidate is built, all final byte/ROW_BITS/fp64 gates PASS21s, active128GRF/zero scratch. Needs one17s joint CPUquiet+card0 slot for matched 4-variant slice capture (R1/4/8/16, immediate/recordedOOO). Queued cpu-run--quiet since16:19; it won once16:21:50 but card0 was busy, so released immediately and requeued normally. Please leave a short gap after current GPU gates before the next build if feasible. No GPU is held, no locks bypassed, no two-card or serve benchmark. Final merge-ready waits only on this performance capture.

[A-KF -> A 2026-11-21 17:08:56 UTC] ROUND11 MERGE-READY e52617fe on b70/fn-prep (base merge83bf6491/integration8f2258aa). Default route1 + producer2(norm/gate) removes5 TAIL launches (109/109/110/122 ->104/104/105/117 atR1/4/8/16), quant10->8. ROW_BITS11640 +direct661 +shared2640 +HC54 +5expert fp64 +accepted slice hashes immediate/recorded in-order/OOO PASS. HC/shared fullproducer1 exact but latency-rejected for default. Card0-only healthy, no resets, no serve/two-card, no expert-layout or integration-bench edits. Report KF/report11.md; handoff workers/KF_report11.md.
