# 27b_w8a8_research.md -- per-step disassembly of TP=2 W8A8 Qwen3.6-27B (PP + TG)

**Created:** 2026-06-24 - **Box:** sec-0 rig, 2x Arc Pro B70 (Battlemage G31), 1950X, PCIe Gen3 cross-die,
Unraid kernel 6.18.33. **Image:** `vllm-xpu-env:int8g` (vLLM 0.23.0 + custom oneDNN s8s8s32 INT8 kernel).
**Model:** `Qwen3.6-27B-W8A8-sqgptq-mtp-graft` (compressed-tensors INT8wxINT8a + BF16 MTP graft), TP=2.

Goal (Isaac): disassemble **every step** of prompt-processing (PP/prefill) and token-generation (TG/decode)
into cycles / time / latency / bandwidth / compute, so each optimizable piece is named and quantified ahead of
the **Linux 7.0** migration (drm/xe pcie-p2p fast-interconnect -> real B70<->B70 P2P, the TP lever). All numbers
below are **measured on this box** (scripts 112/113 + allreduce bench, 2026-06-24) unless marked [calc] or [ref].

---

## 0. TL;DR -- the two regimes are bottlenecked by DIFFERENT things

```
                  what dominates              measured              kernel-7.0 P2P helps?
  PREFILL (PP)    TP all-reduce (host-staged) 745 tok/s, TTFT 2.75s  YES, big   (~4x TTFT)
  DECODE  (TG)    weight bandwidth (GEMM+quant) 18.1 t/s (no-MTP)     A LITTLE   (~1.1-1.2x)
                                                34.8 t/s (MTP spec=3)
```

- **PREFILL is ~84% collective-bound.** The 64-layer forward fires **128 all-reduces**; at M=2048 each moves a
  21 MB bf16 activation, and our cross-die host-staged oneCCL all-reduce runs at **only 1.16 GB/s** -> ~18 ms
  each -> **~2.3 s of all-reduce** out of a 2.75 s TTFT. The actual int8 GEMM compute is only **~0.30 s (11%)**.
  This is why prefill sits "~10x below the int8-XMX compute ceiling" -- the gap is the wire, not the math.
- **DECODE is weight-bandwidth-bound.** Per token the model reads ~16 GB of weights/card; at 581 GB/s that is a
  ~28 ms floor, and the measured captured token is 55 ms (no-MTP). The GEMM+activation-quant device budget is
  **~40 ms (73%)**; the **128 decode all-reduces are only ~7-11 ms (~13-20%)** because each moves just 10 KB
  (latency-bound, not bandwidth-bound). Activation-quant alone is ~6.5 ms.
- **Therefore Linux-7.0 P2P is a prefill/TTFT + concurrency win, not a single-stream decode win.** It attacks the
  1.16 GB/s all-reduce directly. Decode's levers are different: smaller weights (int4), fusing the activation
  quant into the GEMM prologue, graph capture, and MTP.

Clock for cycle conversions: **gt0 = 2.8 GHz** (measured `tile0/gt0/freq0/max_freq=2800`). 1 us = 2800 cycles.
One decode token (55 ms) = **154 M cycles**; one 2048-prefill pass (2.75 s) = **7.7 G cycles**.

---

## 1. Hardware facts (measured on this box, 2026-06-24)

| quantity | value | source |
|---|---|---|
| GPU | Intel Arc Pro B70, Battlemage **G31** (`0xe223`), Xe2 | `lspci`, `sycl-ls` |
| Xe-cores / EU | **32 Xe-cores x 8 XVE = 256 EU**; 32 subslices | `torch.xpu.get_device_properties` |
| sub-group sizes | 16, 32 | device props |
| core clock (gt0) | **2.8 GHz** max (gt1 media = 1.5 GHz) | sysfs `tile0/gt0/freq0/max_freq` |
| VRAM | 32 GB GDDR6 (30.3 GiB usable) | device props |
| read BW roofline | **~581 GB/s** (bf16 GEMV hits it at M=1) | docs/kernel/23 |
| int8 XMX GEMM peak | **~290-305 TOPS** @ M=2048 (spec 367) | script 113, this run |
| bf16 XMX GEMM peak | **~139-145 TFLOP/s** @ M=2048 | script 113, this run |
| native FP8 | **none** on Xe2 -> INT8 is the low-precision systolic path | docs/kernel/14 |
| per-card PCIe | Gen3 x16 (~15.8 GB/s wire); cross-die (separate 1950X dies) | docs/P2P_GPU.md |
| **GPU P2P** | **NONE on kernel 6.18** (`zeDeviceCanAccessPeer=False`, 12-variant probe) | P2P_GPU H.9 |
| TP all-reduce (host-staged oneCCL, **measured**) | **1.16 GB/s** @ SYCL-kernels / 0.68 GB/s eager | allreduce bench |
| all-reduce small-msg latency | **88 us** @ SYCL-kernels / 250 us eager (10 KB) | allreduce bench |

The single most important hardware number in this whole document is **1.16 GB/s** -- the effective cross-die
host-staged all-reduce bandwidth. It is ~13x below the Gen3 x16 wire and is what makes prefill slow.

---

## 2. Model identity + the TP=2 sharding map

Qwen3.6-27B is a **hybrid** model (`config.json`): 64 decoder layers, `full_attention_interval=4` ->
**16 full-attention layers** (idx 3,7,...,63) + **48 Gated-DeltaNet (GDN / `linear_attention`) layers**. Every
layer has a dense **MLP** (intermediate 17408). hidden=5120, vocab=248320, head_dim=256.

**W8A8 quant is selective** (ignore-list `["lm_head","re:.*linear_attn.*","re:.*visual.*","re:.*mtp.*"]`):

```
  int8 W8A8 (int8 weight x int8 dynamic-per-token act):  ALL 64 MLPs + the 16 full-attn q/k/v/o projections
  BF16 (kept full precision):  ALL 48 GDN in_proj/out_proj, lm_head, MTP head, vision tower
```

So the 48 GDN layers run **bf16** projections (no activation quant there); only MLP + full-attn are on the int8
systolic path. This matters: ~5.6 GB/card of the weights (the GDN bf16 projections) are read at 2 bytes/elem.

### 2.1 Sharding + collective placement (vLLM default TP, confirmed by per-op profile)

Column-parallel = split the OUTPUT dim, no collective. Row-parallel = split the CONTRACT dim, **all-reduce after**.

```
 LAYER KIND   GEMM            parallel   full [K->N]        per-card [K->N]     dtype   collective
 ----------   ----            --------   ----------         --------------      -----   ----------
 MLP (x64)    gate_up_proj    COLUMN     5120 -> 34816      5120 -> 17408       int8    -
              down_proj       ROW        17408 -> 5120      8704 -> 5120        int8    ALL-REDUCE
 FULL-ATTN    qkv(+out-gate)  COLUMN     5120 -> 14336      5120 -> 7168        int8    -
   (x16)      o_proj          ROW        6144 -> 5120       3072 -> 5120        int8    ALL-REDUCE
 GDN (x48)    in_proj_qkvz    COLUMN     5120 -> 16384      5120 -> 8192        bf16    -
              in_proj_ba      COLUMN     5120 -> 96         5120 -> 48          bf16    -
              conv1d(k=4)     (per-head, on sharded qkv mixed dim 10240->5120)  bf16    -
              out_proj        ROW        6144 -> 5120       3072 -> 5120        bf16    ALL-REDUCE
 HEAD         lm_head         COLUMN     5120 -> 248320     5120 -> 124160      bf16    (gather logits)
```

**Collective count per forward pass = 2 all-reduces x 64 layers = 128 all-reduces** (one after each
attention/GDN out_proj, one after each MLP down_proj). This count is identical for prefill and decode -- only the
message SIZE differs (M=2048 -> 21 MB vs M=1 -> 10 KB). That size difference is the whole story (sec 5).

MTP spec-verify adds one **all_gather** per verify step (the spec tokens), which is why the recipe needs the
capture-safe all-reduce-of-padded shim; see sec 7.

---

## 3. ASCII data-flow: ONE DECODE TOKEN (TG), per card, captured

Times are measured per-op DEVICE time (script 113, back-to-back kernels = the captured per-op cost). `[AR]` =
all-reduce (host-staged, ~88 us eager / ~50-70 us in-graph at 10 KB). A token traverses all 64 layers once.

```
 DECODE (M=1)  ---- one token, one card, 2.8 GHz ----            dev_us   GB/s   %roof   cycles
 input hidden [1,5120] bf16
   |
   +-- x16 FULL-ATTN layers ------------------------------------
   |     RMSNorm                                                   ~3      -      -        8k
   |     act_quant(K=5120)  [serial per-row reduce]                41     0.4     -      115k
   |     qkv+gate GEMM int8 [5120->7168]                           62    593    ~100%    174k
   |     RoPE(partial 0.25) + flash-decode vs KV (16 layers)       ~15     -      -       42k
   |     act_quant(K=3072)                                         ~30     -      -       84k
   |     o_proj GEMM int8 [3072->5120]                             51    310     53%     143k
   |     [AR] all-reduce [1,5120] 10KB ......................... ~50-88  (latency-bound)  ~196k
   |     RMSNorm + residual                                        ~4      -      -       11k
   |       (MLP block follows, shared below)
   |
   +-- x48 GDN layers -----------------------------------------
   |     RMSNorm                                                   ~3      -      -        8k
   |     in_proj_qkvz GEMM bf16 [5120->8192]                      144    581    ~100%    403k
   |     in_proj_ba bf16 [5120->48] + conv1d(k4) + SiLU gate      ~70     -      -      196k
   |     gated-delta RECURRENCE (state [128x128] x 24 heads/card) ~est within residual bucket
   |     out_proj GEMM bf16 [3072->5120]                          57    550    ~95%     160k
   |     [AR] all-reduce [1,5120] 10KB ......................... ~50-88  (latency-bound)  ~196k
   |     RMSNorm + residual
   |
   +-- x64 MLP blocks (one per layer, after attn/GDN) ----------
   |     act_quant(K=5120)                                        41     0.4     -      115k
   |     gate_up GEMM int8 [5120->17408]                         157    567     90%     440k
   |     SiLU(gate)*up                                            28      -       -       78k
   |     act_quant(K=8704) [serial per-row reduce, ~5ns/elem]     51     0.5     -      143k
   |     down_proj GEMM int8 [8704->5120]                         77    577     99%     216k
   |     [AR] all-reduce [1,5120] 10KB ......................... ~50-88  (latency-bound)  ~196k
   |
   +-- final RMSNorm
   +-- lm_head GEMM bf16 [5120->124160]  (vocab-parallel)        2149   592    ~100%   6.0M   <- 1.27 GB read!
   +-- sample

 PER-LAYER device (GEMM+quant+conv): MLP 354us | FULL-ATTN 154us | GDN 271us
 PER-TOKEN device budget (GEMM+quant+conv only): 354*64 + 154*16 + 271*48 + 2149(head) = ~40.3 ms
 + 128 all-reduces (~7-11 ms captured) + recurrence/attn-math/norms/residual (~5-8 ms)
 = MEASURED captured no-MTP token ~55 ms  (18.1 t/s).   Pure weight-read floor = 28 ms (50.9 t/s).
```

Note the **lm_head is one op but 2149 us (~4% of the token)** -- a 1.27 GB/card bf16 vocab read at the roofline.
And note act_quant: `MLP.act_quant_in` measured 157 us in the raw log was a **first-op warmup artifact**; the
steady-state K=5120 quant is **41 us** (`ATTN.act_quant_in`), K=8704 is **51 us** -- a serial per-row reduction
(~5 ns/elem) that does NOT parallelize over M and persists under capture (docs/kernel/23).

---

## 4. ASCII data-flow: ONE PREFILL PASS (PP), M=2048, per card, captured

Same ops, M=2048. Now the int8 GEMMs fill the XMX array (compute-bound, ~290-305 TOPS) and the all-reduce moves
**21 MB** instead of 10 KB -- so the [AR] cost EXPLODES from ~88 us to ~18 ms each.

```
 PREFILL (M=2048)  ---- one pass, one card ----                 dev_us    TOPS/  note
                                                                          TFLOPs
   FULL-ATTN x16:  act_quant 123 | qkv+gate int8 535 (281 TOPS) | attn(quadratic, but flat in practice)
                   | o_proj int8 240 (268 TOPS) | [AR] 18 ms          ~= 899us GEMM + 18ms AR / layer
   GDN x48:        in_qkvz bf16 1239 (139 TFLOPs) | conv 430 | out_proj bf16 445 (145 TFLOPs)
                   | [AR] 18 ms                                        ~= 2114us GEMM + 18ms AR / layer
   MLP x64:        act_quant 103+225 | gate_up int8 1236 (295 TOPS) | SiLU 677 | down int8 599 (304 TOPS)
                   | [AR] 18 ms                                        ~= 2841us GEMM + 18ms AR / layer

 PER-CARD GEMM budget:  MLP 2841*64=182ms | FULL-ATTN 899*16=14ms | GDN 2114*48=101ms = ~298 ms total
 COLLECTIVES:           128 all-reduce x 21 MB / 1.16 GB/s = ~18 ms each = ~2304 ms   <==== 84% of TTFT
 OTHER (attn math, GDN chunk-scan, norms, launch): ~150 ms
 -------------------------------------------------------------------------------------
 = MEASURED captured TTFT @2048 = 2.748 s (745 tok/s).  Compute-only ceiling would be ~0.30 s (~6800 tok/s).
```

The journal already noted prefill is "~10x below the int8-XMX compute ceiling" and "flat ~380-396 tok/s from 2K
to 131K (NOT attention-bound)". This decomposition **names the cause**: the 128 host-staged 21 MB all-reduces at
1.16 GB/s. Prefill throughput is flat vs context because the per-layer all-reduce cost is fixed per token, not
quadratic -- the hybrid GDN keeps attention cheap, so the wire dominates at every context length.

---

## 5. The interconnect deep-dive (the measured all-reduce, the crux of TP)

`scripts/allreduce_bench.py`, both cards, xccl, 2026-06-24. algbw = busbw at world=2.

```
  msg size      SYCL-kernels=1 (captured serve)     eager (SYCL=0)
                lat_ms    algbw GB/s                lat_ms   algbw GB/s     regime
  10 KB (dec)   0.088     0.10  (latency-bound)     0.25     0.03           DECODE all-reduce
  1 MB          0.85      1.23                       1.55     0.68
  16 MB         14.12     1.19                       24.9     0.67
  21 MB (pre)   ~18 [interp]  1.16                   ~32      0.68          PREFILL all-reduce
  256 MB        231.8     1.16                       396      0.68
```

Reads:
- **Bandwidth plateaus at 1.16 GB/s** (SYCL-kernels) / 0.68 GB/s (eager). This is the cross-die host-staged
  ceiling -- GPU0 -> host RAM (Gen3) -> CPU reduce -> host RAM -> GPU1 (Gen3), un-pipelined, across two 1950X
  dies over Infinity Fabric. It is ~13x below the Gen3 x16 wire (~15.8 GB/s) and matches the H.8 torch-d2d
  1.35 GB/s. `CCL_ENABLE_SYCL_KERNELS=1` (the captured-serve setting) is ~1.7x the eager BW and ~3x lower
  small-msg latency -- a real, free win we already take.
- **Decode all-reduce is latency-bound** (10 KB transfers in 8.6 us at 1.16 GB/s, but the call costs 88 us ->
  ~80 us is fixed launch/round-trip latency). 128/token = ~11 ms eager-equiv (less in-graph).
- **Prefill all-reduce is bandwidth-bound**: 21 MB / 1.16 GB/s = 18 ms, x128 = 2.3 s. This is the prefill wall.

---

## 6. Where the time goes (apportionment)

```
  DECODE token (55 ms, captured no-MTP, per card)        PREFILL pass @2048 (2748 ms, captured)
  ------------------------------------------------       --------------------------------------
  int8/bf16 GEMM weight BW   ~34 ms   62%   <- int4!     TP all-reduce (1.16 GB/s)  ~2304 ms  84%  <- P2P!
  activation-quant (serial)  ~6.5 ms  12%   <- fuse      int8/bf16 GEMM compute      ~298 ms  11%
  all-reduce x128 (10KB)     ~7-11ms  13-20% <- P2P sm   attn math + GDN scan+norms  ~150 ms   5%
  GDN recur + attn + norms   ~5-8 ms  ~12%               activation-quant            (in GEMM bucket)
  ------------------------------------------------       --------------------------------------
  weight-read roofline floor = 28 ms (50.9 t/s)          compute-only floor = ~0.30 s (~6800 tok/s)
```

Inside the decode "GEMM weight BW" bucket, the **BF16 GDN projections are ~10 ms (~18% of the token)** on their
own (in_proj_qkvz 144 us + out_proj 57 us, x48 layers) -- a first-order term *because GDN is left unquantized*.
We cannot int4 it (the gated-delta recurrence needs BF16 for stability), so unlike the MLP/attn int8 GEMMs this
~5.6 GB/card of bf16 weight is a fixed decode tax. That makes the GDN projections, after the MLP, the second
biggest single decode cost -- worth flagging for any future "int8 GDN projection with bf16 recurrence" experiment.

**Decode levers, ranked:** (1) **smaller weights** -- W4A8/W4A16 halve the 34 ms GEMM-BW bucket (int4 decode is
~2x int8, docs/kernel/19/23); this is why W4A16 decodes ~2x W8A8. (2) **fuse activation-quant into the int8 GEMM
prologue** -- removes most of the 6.5 ms (one opaque node, parallel K-reduction; standalone swaps already proven
to regress under capture, docs/kernel/23). (3) **MTP** -- amortizes the whole per-token cost over ~accept-length
tokens (measured 18.1 -> 34.8 t/s at spec=3). (4) P2P only trims the ~7-11 ms small-msg all-reduce bucket.

**Prefill levers, ranked:** (1) **P2P / faster all-reduce** -- attacks 84% of TTFT directly. (2) fewer/fused
collectives (Seguin's clone-safe + delay-allreduce patches, docs/literature/10). (3) compute is already near
ceiling -- do NOT tune the GEMMs.

---

## 7. MTP interaction (the production path = spec=3, 34.82 t/s)

The shipped recipe runs **captured + MTP spec=3** = 34.82 t/s @51% accept (vs 18.10 captured no-MTP = 1.92x).
MTP changes the per-step picture:
- The 1-layer BF16 MTP head drafts spec tokens; the body then **verifies a batch of M=1+spec** (=4 at spec=3) in
  one forward. The GEMMs go from M=1 (BW-bound) to M=4 (still BW-bound -- intensity ~4 ops/byte << the ~800
  int8 ridge), so the weight bytes are read **once per ~accept-length accepted tokens** instead of per token.
  That is the win: MTP cuts the NUMBER of bandwidth-bound passes, it does not make decode compute-bound.
- It adds **one all_gather per verify** (the spec tokens across ranks). oneCCL 2021.17 cannot SYCL-graph-record
  its native allgather -> the recipe replaces it with an **all-reduce-of-padded** shim so it captures
  (`patches/sitecustomize.py`). Ejecting collectives to eager instead breaks the captured-piece input-address
  contract -> garbage (this was "Bug B", root-caused 2026-06-24).
- Decode decreases monotonically with spec (51%/37%/26% accept at spec 3/4/5) -- the 1-layer head over-drafts
  past ~3, so spec=3 wins. (The old "spec=5 climbing to 63 t/s" was degenerate-garbage measurement.)

At TP=2 specifically MTP is a **bigger** lever than single-card because it also amortizes the all-reduce tax:
1 collective round per ~accept-length verified tokens instead of per token.

---

## 8. Linux-7.0 P2P: what gets faster, quantified

The 7.0 drm/xe "pcie p2p as fast interconnect" patch (Battlematrix / Arc Pro B-series, merged drm-xe-next
2025-12-30) is the gate to replace the **1.16 GB/s host-staged all-reduce** with direct B70<->B70 transfer.
Our box is on the AMD-Zen whitelist for cross-die pci_p2pdma (1950X family 0x17); the load-bearing unknown is
whether `iommu=pt` voids it (P2P_GPU A.2). Assume P2P lands and the all-reduce approaches the Gen3 x16 wire.

```
  IF all-reduce goes 1.16 GB/s -> ~10-13 GB/s (Gen3 wire, direct or behind a switch):

  PREFILL @2048:  collectives 2304 ms -> ~210-270 ms.  TTFT 2.748 s -> ~0.65-0.75 s  =  ~4x faster prefill.
                  (compute 298 ms now becomes the new floor -> the NEXT bottleneck is the GEMMs/GDN.)
  DECODE no-MTP:  all-reduce 128x(88->~15 us latency, P2P kills the host round-trip)  ~11 ms -> ~2 ms.
                  token 55 ms -> ~46 ms  =  18.1 -> ~21.7 t/s  =  ~1.2x.   MODEST -- decode is weight-bound.
  CONCURRENCY c8: all-reduce scales with batch tokens (8x the bytes) -> currently agg 34 t/s is allreduce-
                  capped; P2P should lift c8 aggregate substantially (the 2x-worse-than-TP1 gap in H.6 closes).
```

**Verdict:** Linux-7.0 P2P is a **prefill/TTFT and throughput-at-concurrency** win (~4x prefill, large c>1 gain),
and only a **~1.1-1.2x single-stream decode** win. It does NOT change the fact that single-stream decode is
weight-bandwidth-bound -- that needs int4 + quant-fusion + MTP, which are orthogonal and stack with P2P. The
right framing for the migration: **P2P makes TP=2 finally worth it for prefill and multi-user serving**, where
today (H.6/H.7) the 35 GB W8A8 only uses TP=2 because it must (does not fit one card), and pays a 3.3x-worse-TTFT
tax for the privilege.

Concrete first experiments once on 7.0 (queued, reboot-gated -- P2P_GPU I.1/I.2):
1. `71_run_ze_matrix.sh` -> does `zeDeviceCanAccessPeer` flip True? (today False on all 12 variants).
2. Re-run **this** allreduce bench -> does 1.16 GB/s climb toward the wire? (the single number that gates the 4x).
3. Re-run scripts 95/96 prefill -> does 745 tok/s climb toward ~2500-3000?
4. A/B `iommu=off` first (cheap; tests the Zen-whitelist-vs-IOMMU question without a kernel change).

---

## 9. Optimization target board (ranked, with expected payoff)

```
  #  lever                                   regime        expected         status / where
  1  Linux-7.0 P2P all-reduce                PREFILL/conc  ~4x TTFT, big c8 GATED on 7.0 reboot (sec 8)
  2  MTP spec=3 (already shipped)            DECODE        1.92x (live)     rdy_to_serve recipe
  3  int4 weights (W4A8/W4A16) for decode    DECODE        ~2x decode BW    W4A8 single-card 2.03x proven
  4  fuse act-quant into int8 GEMM prologue  DECODE        ~+12% (6.5ms)    NOT built (kernel/23 option 1)
  5  Seguin clone-safe + delay-allreduce     PREFILL/conc  cut collective#  not cherry-picked (lit/10)
  6  capture (already on)                    BOTH          eager 4.1->18.1  shipped (GRAPH=1)
  -  do NOT tune the GEMMs                    -            near-roofline    int8 99%/90%, bf16 ~100% roof
```

The int8/bf16 GEMM kernels are within 5-12% of the BW/compute roofline at every shape measured -- there is no
meaningful headroom there. Every real lever is either **fewer bytes** (int4), **fewer launches/collectives**
(capture, P2P, fusion), or **fewer passes** (MTP).

---

## 9.5 Independent cross-check (codex / gpt-5.5, analytical, no measurement)

A second model was given only the architecture + hardware roofline (no measured per-op times) and asked to
derive the sharding map and roofline independently. It agreed on the structure and validated the microbench:

- **128 all-reduces/forward** -- derived identically (16 full-attn x2 = 32, plus 48 GDN x2 = 96).
- **Per-op roofline matched the measurement:** MLP gate_up 153 us (measured 157), GDN in_qkvz 144 us (measured
  144), lm_head 2.19 ms (measured 2.15), down-quant K=8704 ~50 us (measured 51) -- all within a few percent.
- It independently flagged the **BF16 GDN projections as a first-order decode term** (sec 6 callout) and the
  serial activation-quant.

Where it **differed is exactly where the measurement was decisive** -- the interconnect, which it could only
estimate from the literature:
- It used Seguin's **15-17 us** decode all-reduce (a PCIe-4.0 box) -> 2 ms/forward, concluding "decode is not
  collective-bound." Our box **measures 88 us** (Gen3 cross-die) -> ~11 ms; still not dominant, but ~5x its est.
- It assumed prefill all-reduce at **8-16 GB/s** -> ~170-340 ms of collectives. Our box **measures 1.16 GB/s**
  -> ~2.3 s. That single measured number is what flips prefill from "compute+comm balanced" (its guess) to
  "84% collective-bound" (reality). **This is the headline that only measurement could produce**, and it is the
  strongest possible argument for the Linux-7.0 P2P migration.

---

## 10. Appendix: provenance, raw data, caveats

**Scripts (this session):** `scripts/113_w8a8_perstep_microbench.py` (per-op device times, card 0),
`scripts/allreduce_bench.py` (collective BW/latency, both cards), `scripts/112_*` (the vLLM-profiler attempt --
see caveat). CSVs: `results/perstep_microbench_*.csv`. E2E numbers [ref] from recipe README + JOURNAL
scripts/95/96/99/111 + P2P_GPU H.6/H.7 (same model, same image).

**Method:** per-op device time = N=100 kernels enqueued back-to-back between two `torch.xpu.Event`s / N (launch
overlaps -> approximates the in-captured-graph per-op cost). A second perf_counter+synchronize timer gives the
eager per-call cost; the gap is the per-op launch overhead (~50-70 us/op) that graph capture removes. The
all-reduce bench is eager `dist.all_reduce`; in-graph the small-msg latency is somewhat lower (dispatch removed,
host round-trip remains).

**Caveats / honesty:**
- The vLLM **torch-profiler endpoint is absent** on `:int8g` (`VLLM_TORCH_PROFILER_DIR` is an unknown env;
  `/start_profile` -> 404). So the per-op decomposition is a **component microbench at the exact per-card TP=2
  shapes**, not an in-situ server Kineto trace. The component sums (40 ms decode / 298 ms prefill GEMM) are
  cross-validated against the measured E2E (55 ms / 2748 ms) and reconcile within the named residual buckets.
- The **GDN gated-delta recurrence + gating** (the `gdn_attention_core_xpu` custom op) is not isolated -- it
  lives in the ~5-8 ms decode "recurrence/attn/norms" residual and the ~150 ms prefill "other" bucket. Its
  bf16 in/out projections ARE measured (144/57 us decode). A follow-up could isolate it via the GDN .so directly.
- Cycle figures use the **2.8 GHz max clock**; sustained clock under sustained load may be lower, so cycle counts
  are upper-ish bounds on "work" and the time numbers are the ground truth.
- Decode all-reduce captured cost (7-11 ms) is bracketed, not directly measured in-graph (the eager bench gives
  88 us/call; capture removes some fixed overhead). Prefill all-reduce (18 ms/call, BW-bound) is robust.
- All compute numbers are **per card** (TP=2). Both cards run the same shapes in lockstep; wall-clock = one card
  + the all-reduce barriers.

**Open questions for the 7.0 window:** (1) does the 1.16 GB/s all-reduce actually climb with P2P, or is the
cross-die Infinity-Fabric path still the limiter even with peer access? (2) does `iommu=pt` void the Zen
whitelist? (3) at what context length does prefill attention finally stop being flat (the GDN scan cost)?
