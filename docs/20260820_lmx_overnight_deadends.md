# LocalMaxxing overnight -- dead-end packets

Closed paths for `docs/20260820_lmx_overnight_plan.md`.
Newest packet at the **bottom**. Retry only if **Retry if** is now true.

Carry-ins from the W8A8/Ornith loops (do not re-open casually):

| id | path | retry if |
|---|---|---|
| D18 | emul NVFP4 G1 (bangs even with KV_FP8=0) | fused path also bangs; it does not |
| L63/L64 | Python sticky/M1 fused apply | 33.1/33.3 vs hold 34.9. Kernel ESIMD only |
| L65 GRAPH | draft-INT4 on Ornith GRAPH Half!=BF16 | opaque bf16-out op or dtype=fp16 dummy_run fix |
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

---
