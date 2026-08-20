# LocalMaxxing overnight -- dead-end packets

Closed paths for `docs/20260820_lmx_overnight_plan.md`.
Newest packet at the **bottom**. Retry only if **Retry if** is now true.

Carry-ins from the W8A8/Ornith loops (do not re-open casually):

| id | path | retry if |
|---|---|---|
| D18 | emul NVFP4 G1 (bangs even with KV_FP8=0) | fused path also bangs; it does not |
| L63/L64 | Python sticky/M1 fused apply | 33.1/33.3 vs hold 34.9. Kernel ESIMD only |
| L65 GRAPH | draft-INT4 on Ornith GRAPH Half!=BF16 | opaque bf16-out op or dtype=fp16 dummy_run fix |
| PRE.1 | P2P=1 in vLLM TP>1 | reboot window + `I_KNOW_P2P_WEDGES=1` |
| 101.9/D16 | Steve AutoRound MTP5 photocopy | parked; not this overnight first pick |

LocalMaxxing numbers that are **not** C1 (do not chase):

| tok/s | why ignore |
|---|---|
| 1139 | SergiioB C64 aggregate, no MTP |
| 438 | bosd concurrency=64 unlabeled |
| 248 | palmfuture MTP4 p4096/o32 |

---
