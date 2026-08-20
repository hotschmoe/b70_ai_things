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

---
