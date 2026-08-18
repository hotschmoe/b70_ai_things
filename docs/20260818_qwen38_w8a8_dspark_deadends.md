# Qwen3.8 W8A8 + DSpark -- dead-end packets

Closed paths for `docs/20260818_qwen38_w8a8_dspark_campaign.md`.
A looping agent must read this before retrying anything that
"maybe works now".

Newest packet at the **bottom**. Do not rewrite old packets.
Retry only if the packet's **Retry if** line is now true, and
then write a new packet (or a LOOP ledger note that the retry
condition fired).

Packet shape:

```
## D<n> -- <short name> -- YYYY-MM-DD -- LOOP N

Tried:
Command / config:
Result:
Why it is closed:
Retry if:
Related JOURNAL:
```

Honest ugly numbers (e.g. P0.4 pos0 in the 20% band) are **not**
dead-ends -- they go in the loop ledger + JOURNAL and they make
the train mandatory. A dead-end is a path we will not walk again
without a stated condition.

---

## Pre-closed from prior lab (do not re-open casually)

These were closed before this campaign. They are listed so a loop
does not "just try it". Details live in JOURNAL / P2P_GPU.md /
the campaign standing-list (section 6).

| id | path | retry if |
|---|---|---|
| PRE.1 | `CCL_TOPO_P2P_ACCESS=1` in vLLM TP>1 | a reviewer demands a 7.1 retest **and** there is a reboot window. `I_KNOW_P2P_WEDGES=1`. Never chain two tries. |
| PRE.2 | FATTN_MMA=1 on llama.cpp JIT | we have an AOT 2026.1.1 image and Paris-first on that image. JIT already crash-looped. |
| PRE.3 | method=dflash on vLLM 0.26 | `DFlashQwen3DSparkModel` is registered in the serve image. Today it is not. Use method=dspark. |
| PRE.4 | Adaptive verify on GDNAttentionBackend | vLLM grows a GDN-safe adaptive path. Today it rejects. |
| PRE.5 | DeepSpec 38 TB offline cache | never. SpecForge offline + tens of GB hiddens is the recipe. |
| PRE.6 | llm-scaler 0.21 / rmacy v10-slim as a vehicle | we need an 8k / 17-22 tok/s curiosity retest. Not a campaign vehicle. |
| PRE.7 | Q4K reorder-family on a *new* JIT without Paris-first | never skip Paris-first. We got lucky on llama.cpp once. |
| PRE.8 | PCIe ASPM=performance | never on this box (lab kernel panic). |
| PRE.9 | Peer-pair comm mode 3 | never on this box (lab device-lost storm). |
| PRE.10 | oneCCL 2021.15 in a TP>1 serve | never. Overlay 2021.17. |
| PRE.11 | `xpu_shard_top1` default-on for NVFP4 | already e2e-negative (c1 48.9 -> 32.5). Re-A/B only on W8A8 DSpark, explicitly, once. |
| PRE.12 | Inventing a PSpark checkpoint / DeepSeek sibling | never. Prefill arm is SpecPrefill / PFlash-class (campaign section G). |
| PRE.13 | FP8 GEMM on Xe2 | never. Repack FP8 weights to s8. No systolic FP8/FP4. |
| PRE.14 | Overwriting `models/files/qwen3.8-27b/w8a8-gptq` | never. New scheme = new dir. |
| PRE.15 | Entering Phase 2 (torch 2.13) before a Phase 0+1 W8A8+DSpark number | the living header has that number **and** a written 0.27-only feature list. |
| PRE.16 | Long DSpark train before P1.2 10-sample overfit | overfit accepts the full block against the same W8A8 target. |
| PRE.17 | Speed work after HE+ plus < 0.90 | quality is back above the gate (campaign A). |

Campaign-origin packets start at D1 below, once a loop closes
something new.

---

## Campaign packets

(none yet -- LOOP 0 authored the campaign, no GPU)
