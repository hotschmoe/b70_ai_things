# LocalMaxxing overnight -- dead-end packets

Closed paths for `docs/20260820_lmx_overnight_plan.md`.
Newest packet at the **bottom**. Retry only if **Retry if** is now true.

Carry-ins from the W8A8/Ornith loops (do not re-open casually):

| id | path | retry if |
|---|---|---|
| D18 | emul NVFP4 G1 (bangs even with KV_FP8=0) | fused path also bangs; it does not |
| L63/L64 | Python sticky/M1 fused apply | 33.1/33.3 vs hold 34.9. Kernel ESIMD only |
| L65 GRAPH | draft-INT4 on Ornith GRAPH Half!=BF16 | RETRIED LOOP 10. Opaque b70::int4_gemm_w4a16_cast + wipe stale eagle_head cache. GRAPH boots. code c1 21.7 < 34.9 |
| PRE.1 | P2P=1 in vLLM TP>1 | RETESTED 7.1 LOOP 4. Hang remains. Wedge (follow-on TP=2 DEVICE_LOST) CURED. Keep P2PACCESS=0. |
| 101.9/D16 | Steve AutoRound MTP5 photocopy | parked; not this overnight first pick |

LocalMaxxing numbers that are **not** C1 (do not chase):

| tok/s | why ignore |
|---|---|
| 1139 | SergiioB C64 aggregate, no MTP |
| 438 | bosd concurrency=64 unlabeled |
| 248 | palmfuture MTP4 p4096/o32 |

## PRE.1b -- vLLM TP>1 CCL_TOPO_P2P_ACCESS=1 hang -- 2026-08-20 -- LOOP 4

Tried: kernel 7.1.0-070100. Operator override. I_KNOW_P2P_WEDGES=1.
  zeDeviceCanAccessPeer 0<->1 True/True.
  Arm A: W8A8-gptq TP=2 GRAPH=0 P2P=1 PUSH_AR=1. Hung 900s
  at encoder-cache / shm_broadcast after 16.74 GiB load.
  Arm B: same PUSH_AR=0 (plain oneCCL). Same hang 900s.
  Arm C: P2P=0 PUSH_AR=1 immediately after both hangs:
  HEALTHY 147s, G1 Paris/391 GO.
Result: 7.1 cured the H.13 *wedge* (matmul+TP=2 P2P=0 still
  work). 7.1 did not make P2P-in-vLLM-serve boot.
Why it is closed: oneCCL P2P=1 still hangs at worker warmup
  all_reduce. Production comm stays PUSH_AR L0-IPC at
  P2PACCESS=0.
Retry if: oneCCL/vLLM multiproc P2P patch lands, or a
  measured non-hang P2P serve on this box.
Related JOURNAL: ### 2026-08-20az

## P1b -- Q8 fused doors 0 vs 1 -- 2026-08-20 -- LOOP 6

Tried: Pliny Q8_0 TP2, ignore-eos g128 n=5.
  Q8_DOORS=1 (COMM_FUSED_Q8 / SWIGLU_Q8 /
  ATTN_Q8 / GDN_Q8 + COMM_DIRECT=2) vs
  Q8_DOORS=0 (those off, COMM_DIRECT=0).
  MMVQ pair/triple/quad ON both.
Result: **31.78** vs **31.56** post_first
  (+0.7%). G1 Paris/391 both. Prefill
  570 vs 568. Already mmq_q8_reorder=1,
  PRIORITIZE_DMMV=0, not a 4x DMMV miss.
Why it is closed: fused Q8 doors are not
  the 43.8 Q4_K_M gap (~1.38x remains).
Retry if: a new Q8 door kernel lands, or
  COMM_DIRECT/1x-vs-2x/DP4A2 changes the
  decode mix so doors could matter.
Related JOURNAL: ### 2026-08-20bb

## P1c -- COMM_DIRECT_Q8 2 vs 0 -- 2026-08-20 -- LOOP 7

Tried: Pliny Q8_0 TP2, Q8_DOORS=1 both,
  ignore-eos g128 n=5. COMM_DIRECT=2 vs 0.
  Never 3 (lab DEVICE_LOST).
Result: **31.99** vs **31.83** post_first
  (+0.5%). G1 Paris/391 both. Prefill
  570 vs 568. Matches P1b doors=1 31.78.
Why it is closed: llama.cpp USM comm
  direct is not the 43.8 Q4_K_M gap
  (~1.37x remains).
Retry if: COMM_DIRECT=3 becomes safe, or
  1x vs 2x shows comm-bound decode.
Related JOURNAL: ### 2026-08-20bc

## P1d -- GPU_COUNT 1 vs 2 -- 2026-08-20 -- LOOP 8

Tried: Pliny Q8_0, Q8_DOORS=1 COMM=2,
  ignore-eos g128 n=5. GPU_COUNT=2 vs 1.
  1x booted (29GB Q8 on 1x B70).
Result: 2x **32.03** vs 1x **17.93**
  (scale 1.79x). Prefill 566 vs 573.
  G1 Paris/391 both.
Why it is closed: 1x does not close
  the 43.8 gap; it is slower. TP2 is
  compute-split, not the limiter.
Retry if: a 1x-only kernel (DP4A2) is
  measured faster than 32.03 2x.
Related JOURNAL: ### 2026-08-20bd

## P1e -- MMVQ_SG32=1 and lab DP4A2xSG24 -- 2026-08-20 -- LOOP 9

Tried: 0xSero JIT libggml-sycl.so
  258f4729 (no QUAD_SG24/DP4A2 strings).
  Forced GGML_SYCL_MMVQ_SG32=1 on both
  B70s (q8_0 ne1=1 kernel). Q8_DOORS=1
  COMM=2 GPU=2 ignore-eos g128 n=5.
Result: G1 Paris/391. **30.11** vs hold
  32.03 (-6%). Lab DP4A2xSG24 .so
  e75b9603 not present. AOT 2026.1.1
  UR does not enumerate on public stacks
  (0xSero). Our 32.03/43.8 ratio matches
  lab Q8/Q4 (36.8/49.7).
Why it is closed: in-image SG32 is a
  regression. Closing the 13% to lab Q8
  needs that AOT binary, not an env door.
Retry if: a BMG JIT build of DP4A2xSG24
  exists, or 2026.1.1 AOT enumerates.
Related JOURNAL: ### 2026-08-20be

## O3 -- MTP routed-expert slot INT4 -- 2026-08-20 -- LOOP 11

Tried: pack FusedMoE routed_experts w13/w2
  E=256 INT4 (1688->435 MB). Slot apply
  with NT-view qweight (.t() no contig).
  GRAPH=1 MTP3. G1 Paris/391.
Result: boots. bench_code c1 **21.1** vs
  O2 dense-only 21.7 vs hold **34.9**.
Why it is closed as a speed path: slot
  int4_gemm loops did not beat Triton
  BF16 experts. VRAM win only.
Retry if: grouped INT4 MoE gemm, or
  Triton reads packed INT4 weights.
Related JOURNAL: ### 2026-08-20bg

## O4b -- WG/SLM occupancy on M=1 NVFP4 GEMV -- 2026-08-20 -- LOOP 13

Tried: same Ornith expert shapes as LOOP 12.
  WG=1 1D block_load vs WG=16 (no SLM)
  vs SLM-broadcast x WG=16/32.
  Grouped 8x up SLM16 vs WG=1.
Result: numerics still 1.6e-7.
  up: wg1 0.014 ms 109 GB/s;
  wg16 0.986x; slm16 0.967x; slm32 0.922x.
  down slm16 0.427x (SLM setup on K=512).
  grouped slm16 0.978x (0.069 vs 0.067).
Why it is closed: occupancy is not the
  84-vs-528 gap on these shapes. WG=1
  1D load stays the proto. 2.3x vs
  oneDNN still stands; needs a torch op
  for e2e, not a bigger work-group.
Retry if: fused layerlet / larger K
  makes SLM x reuse show up, or a
  persistent decode kernel changes
  launch mix.
Related JOURNAL: ### 2026-08-20bi

## O4d -- sidecar M1_KERNEL e2e GRAPH -- 2026-08-20 -- LOOP 15

Tried: Ornith NVFP4 fused GRAPH no-MTP
  STICKY=0 M1=0 M1_KERNEL=1 sidecar
  b70_nvfp4_m1.gemv. Same ckpt as 34.9.
  G1 + bench_code c1 256 n=3.
Result: m1k=1. dispatch N=1024 K=2048
  at capture (4s). G1 Paris/391.
  bench_code c1 **32.2** vs hold **34.9**
  vs L64 Python M1 33.3.
Why it is closed as a speed path:
  isolated 2.36x vs oneDNN does not
  beat GRAPH oneDNN in the T x top_k
  slot loop. Extra launches dominate.
Retry if: fused up+silu+down layerlet
  (one launch / expert), or a
  persistent decode kernel.
Related JOURNAL: ### 2026-08-20bk

## O4e -- fused up+silu+down layerlet -- 2026-08-20 -- LOOP 17

Tried: one-launch ESIMD layerlet
  S=8 H=2048 I=512 vs seq 8x
  (up GEMV + silu + down GEMV).
  WG=1024 aborted (ESIMD WG<=64).
  WG=64, 16 serial up rows / WI.
Result: numerics max_rel 1.4e-3 PASS.
  fused **0.133 ms** vs seq **0.137**
  (1.036x) vs eager oneDNN **1.159**
  (8.71x). 21 lsc_load, 0 dpas.
Why it is closed as a speed path:
  launch fusion does not beat the
  slot GEMV, so it cannot beat GRAPH
  oneDNN (O4d 32.2 vs 34.9). ESIMD
  WG=64 serializes N=1024.
Retry if: non-ESIMD SYCL WG>=1024, or
  a persistent decode kernel.
Related JOURNAL: ### 2026-08-20bm

---
