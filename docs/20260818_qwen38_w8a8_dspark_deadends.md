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

## D1 -- GRAPH=1 DSpark k=7 @131k UTIL=0.90 KV OOM -- 2026-08-18 -- LOOP 9

Tried: S0 exact LOOP 8 recipe with GRAPH=1 (SPECTOK=7
  method=dspark THINK_BUDGET=0 MAXLEN=131072 UTIL=0.90
  MAXSEQS=2 TP=2 SERVED=qwen3.8-27b-W8A8-gptq-dspark7
  CGRECLAIM=0 P2PACCESS=0 IMG=int8g-v0260).
Command / config:
  GRAPH=1 SPECTOK=7 MAXLEN=131072 PORT=18080
    NAME=qwen38_w8a8_dspark
    ./bin/gpu-run bash vllm/dflash/serve_qwen38_w8a8_dspark.sh start
Result: EngineCore ValueError at KV init. Need 6.98 GiB
  for 131072, available 6.84 GiB, estimated max 128128.
  Capture/profile already counted (enforce_eager=False).
  Workers died. No DEVICE_LOST. xpu-health card 0 OK.
Why it is closed: GRAPH=1 DSpark draft+verify graphs take
  extra memory vs GRAPH=0 @131k (LOOP 8 loaded). Same
  131k UTIL=0.90 will fail the same way.
Retry if: UTIL>0.90 measured, or MAXLEN<=128128, or a
  smaller capture/graph memory path is in the image.
  LOOP 9 already retried MAXLEN=122880 (loads, G1 hold,
  c1 26.2). Do not retry 131k at UTIL=0.90.
Related JOURNAL: ### 2026-08-18n
  log /mnt/vm_8tb/b70/qwen38-w8a8-dspark/loop9_graph1_131k_crash.log

## D2 -- GRAPH=1 DSpark k=3 G1 "duct" / 0% accept -- 2026-08-18 -- LOOP 11

Tried: leftover k-sweep first cell. Same LOOP 9 recipe
  (method=dspark THINK_BUDGET=0 MAXLEN=122880 UTIL=0.90
  MAXSEQS=2 TP=2 CGRECLAIM=0 P2PACCESS=0 IMG=int8g-v0260)
  with SPECTOK=3 GRAPH=1.
Command / config:
  GRAPH=1 SPECTOK=3 MAXLEN=122880 PORT=18080
    NAME=qwen38_w8a8_dspark
    ./bin/gpu-run bash vllm/dflash/serve_qwen38_w8a8_dspark.sh start
Result: HEALTHY 147s. G0 id
  qwen3.8-27b-W8A8-gptq-dspark3. G1 thinking-off and
  completions both emit "duct" (finish_reason=length,
  64/16 completion_tokens). Spec accept_len 1.00,
  pos0 0.000, 0 accepted. Loaded compile cache
  /vllm_cache/torch_compile_cache/b3f7e9e010 (k=7
  GRAPH=1 leftover). No DEVICE_LOST. Revert GRAPH=0
  k=3: G1 Paris / 391 / fib hold.
Why it is closed: do not publish GRAPH=1 k=3 speed.
  G1 fail is fail-closed. Same k+cache combo will
  replay "duct".
Retry if: wipe host
  /mnt/vm_8tb/b70/vllm_cache/torch_compile_cache/b3f7e9e010
  (backbone + dspark_head) then GRAPH=1 k=3 G1 only.
  If still duct after a cold compile, k=3 GRAPH=1 is
  a real dead-end; stay GRAPH=0 / try k=4.
Related JOURNAL: ### 2026-08-18p
  log /mnt/vm_8tb/b70/qwen38-w8a8-dspark/loop11_graph1_k3_g1fail.log

## D3 -- GRAPH=1 DSpark k=3 still duct after cold compile -- 2026-08-18 -- LOOP 12

Tried: D2 retry. Wiped host
  /mnt/vm_8tb/b70/vllm_cache/torch_compile_cache/b3f7e9e010
  via docker --entrypoint /bin/rm (root-owned; host sudo
  needs a tty). Then GRAPH=1 SPECTOK=3 MAXLEN=122880.
Command / config:
  docker run --rm --user 0:0 --entrypoint /bin/rm
    -v /mnt/vm_8tb/b70/vllm_cache:/vllm_cache
    vllm-xpu-env:int8g-v0260
    -rf /vllm_cache/torch_compile_cache/b3f7e9e010
  GRAPH=1 SPECTOK=3 MAXLEN=122880 PORT=18080
    NAME=qwen38_w8a8_dspark
    ./bin/gpu-run bash vllm/dflash/serve_qwen38_w8a8_dspark.sh start
Result: Cache GONE then rebuilt. Logs: Compiling a graph
  for compile range (1, 2048) takes 1.59 s -- NOT
  "Directly load". HEALTHY 147s. G1 content="duct"
  finish_reason=length ct=32. accept_len 1.00 pos0 0.000.
  No DEVICE_LOST. Revert GRAPH=0: G1 Paris / 391 / fib.
Why it is closed: k=3 GRAPH=1 is broken even on a cold
  compile. Not a stale-cache-only bug. Compile key hash
  b3f7e9e010 is shared across SPECTOK (k=7 and k=3 used
  the same dir) so a k=3 graph must not be left on disk
  for later k. LOOP 12 wiped it again after revert.
Retry if: capture sizes / DSpark graph path change, or
  compile key includes num_speculative_tokens. Do not
  retry the same recipe.
Related JOURNAL: ### 2026-08-18q
  log /mnt/vm_8tb/b70/qwen38-w8a8-dspark/loop12_graph1_k3_cold_g1fail.log

## D4 -- P1.5 W8A16_M_MAX>0 @122880 KV OOM -- 2026-08-18 -- LOOP 15

Tried: default-on small-M w8a16 at the live long-ctx
  DSpark recipe (GRAPH=1 SPECTOK=4 MAXLEN=122880
  UTIL=0.90). W8A16_M_MAX=8 (covers verify M=k+1=5).
Command / config:
  W8A16_M_MAX=8 GRAPH=1 SPECTOK=4 MAXLEN=122880
    SERVED=qwen3.8-27b-W8A8-gptq-dspark4-w8a16
    ./bin/gpu-run bash vllm/dflash/serve_qwen38_w8a8_dspark.sh start
Result: EngineCore ValueError. Model load 27.31 GiB/card
  (clone ON). Available KV cache memory: **-0.93 GiB**.
  No cache blocks. Workers died. No DEVICE_LOST.
  xpu-health card 0 OK. Revert W8A16_M_MAX=0: HEALTHY
  147s, G1 Paris exact.
Why it is closed: NT layout clone doubles s8 weight
  residency. Any W8A16_M_MAX>0 costs the same clone.
  122880 GRAPH=1 DSpark cannot spare ~9 GiB/card.
Retry if: int8_gemm_w8a16 consumes the s8s8 [K,N]
  layout (no NT clone), or a measured MAXLEN/UTIL
  where KV stays positive with the clone. Do not
  retry W8A16_M_MAX>0 at 122880 UTIL=0.90.
Related JOURNAL: ### 2026-08-18t
  log /mnt/vm_8tb/b70/qwen38-w8a8-dspark/loop15_w8a16_oom.log
