# Strata EP prefill: where the time goes, and the path to 2000+ tok/s

Date: 2026-10-16. Read-only analysis. No build, no source edit, no GPU work.
Model: Qwen3.8-Flash-Next ISTA GSQ-RCO IQ3_XXS. Engine: Strata SYCL, src-ep tree.
Setup: two B70s, EP 149 experts/layer on card0 and 363 on card1, all experts
resident, prefill chunk 12288, REDUCE=1, PUSH=1, INT8 KV.

## 0. Evidence and method

| Source | What it gives |
| --- | --- |
| S90 32K trace (`cachefix/current32k-profile-s90-analysis/gpu-timeline.csv`, 323K GPU events) | Per-kernel intervals on both cards for one 32,728-token request. Main chunks are 12288 / 12288 / 8145 tokens. S89 binary b2a610ea. |
| S10 4K trace (S34 `selected-events.json.gz`, 102K GPU events) | Same data for a 4,044-token chunk. Older S9 binary 932d2f34, 155/357 split, otherwise the same EP/PUSH path. |
| S38 host spans (`peer-gather-serving-s38/host-spans.csv`) | Uninstrumented CPU timestamps per layer on a 4,049-token prompt: local/peer rows, active peer experts, peer enqueue loop time. |
| S9/S38/S113 serving medians | Uninstrumented totals: 4K about 3.12 s (1300 tok/s); 32K about 20.6-21.1 s (1550-1590 tok/s). |
| Source | `src-ep/sycl/src/prefill/prefill.cpp` (cited as `pf:LINE`), `gemm.dp.cpp`, `kernels.dp.cpp`, `kernels/cuda/*.dp.cpp` |

Method:

- The new script `prefill_analysis_work/s90_layers.py` cuts card0's timeline
  into 48 layer windows per chunk. Each window runs from one `ep_prefill_finish`
  to the next.
- Inside a window it finds the activation push (`copy_f32_wide`), the primary
  gather, the first primary dequant, and the sum read (`ep_copy_uncached`).
  It assigns card1's kernels to the same window by timestamp.
- The card clocks are not formally joined. Causal order still holds in all 192
  windows: card1 starts 0.8-5.4 ms after card0's push ends, and card0's sum read
  starts after card1's reduce ends.
- Outputs are `prefill_analysis_work/s90_layers.json` and `s10_layers.json`.

Profiler inflation, which matters for reading the tables:

- Unitrace stretched the 4K prefill from 3.1 to 6.7 s, and the 32K prefill
  from about 20.8 to 27.6 s.
- Almost all of the extra time sits in host-enqueue-bound regions. GPU kernel
  durations are trustworthy; host gaps are not.
- I rebuilt an uninstrumented per-layer model from three inputs: trace kernel
  durations, the S38 uninstrumented host cost (about 55-58 us per peer expert;
  median 15.8 ms per layer for 272 active peer experts), and the source order.
- That model gives about 3.0 s at 4K and about 20.0 s plus about 0.6 s of
  non-layer work at 32K. Both are within about 3% of the measured
  uninstrumented totals.

## 1. Per-layer timeline

### 1a. Program order per layer (source)

Card0 runs everything in one in-order queue. Two host blocking points split each layer.

1. HC read for the attention half: `gr_norm_rs` or the previous `gr_write_norm_rs`,
   then BF16 down, `gr_silu`, BF16 up into `gated` (T x 10240 FP32), BF16
   inject, and `gr_mix_r` (pf:2955-3030).
2. Either the GDN block (pf:3032-3055): dequant + FP16 GEMMs for qkv and z,
   BF16 alpha/beta, gates, conv, `gdn_recurrence`, dequant + FP16 ssm_out.
   Or the QSA block (pf:3057-3356): k/v/q projections, indexer k/q, rms/rope,
   KV append, indexer append, block scores + top-k in 256-query batches, then
   attention through `qsa_decode_attn_batch` 32 queries at a time (pf:3345),
   gate, and o projection.
3. `gr_write_norm_rs`, then the HC read for the FFN half (same kernels as step 1).
4. MoE pre-work: BF16 router and `route` (pf:3365-3366), shared expert as
   3 GEMMs + SwiGLU (pf:3368-3371), shared-gate GEMV, ids copy, and
   `copy_f32_wide`. The last one pushes 2 bytes/elem activations to card1 over
   PCIe as a compute kernel on card0 (pf:3487).
5. **Host block 1:** `m.cs->wait()` (pf:3494) waits for all of the layer's
   attention/router work.
6. Host routing: count, offsets, slot and src tables over T*10 entries
   (pf:3512-3600), then upload them and run the primary `gather_rows16` (pf:3633).
7. Host enqueues card1's work first (pf:3636-3767):
   - `ep_copy_uncached`, card1's local copy of the pushed activations (pf:3676);
   - src/slot/weight memcpys;
   - per active expert, 5 launches (pf:3747-3762): `gather_rows16`,
     `iq_dequant_expert_f16`, oneMKL GU (M x 2560 -> 1280), `swiglu_interleaved`,
     oneMKL down (M x 640 -> 2560);
   - `ep_prefill_reduce<push>`, which sums each token's peer rows and writes FP32
     straight into card0's memory (pf:3765).
8. Host enqueues card0's experts: 4 launches each (dequant pf:4328, GEMM pf:4383,
   SwiGLU pf:4384, GEMM pf:4386), driven from pf:4407.
9. **Host block 2:** `m.pp->s->wait()` (pf:4474) waits for card1 to finish
   everything.
10. Card0: `ep_copy_uncached` sum read (pf:4477), `ep_prefill_finish` (pf:4488),
    then `gr_write_norm_rs` (pf:4658). The next layer starts.

### 1b. Measured medians and the uninstrumented model (ms per layer)

| Phase | 4K GDN layer | 4K QSA layer | 12288 GDN, ctx 12-24K | 12288 QSA, ctx 12-24K |
| --- | ---: | ---: | ---: | ---: |
| card0 pre-MoE GPU work (steps 1-4 minus push) | 20.3 | 72.7 | 54.7 | 292.7 |
| - QSA attention family (scores, top-k, attention, merge) | - | 60.0 | - | 259.2 |
| - GDN kernels (conv, recurrence, norms) | 8.3 | - | 22.7 (recurrence 17.9) | - |
| - HC elementwise (`gr_write_norm_rs`, `gr_mix_r`, ...) | 3.6 | 3.6 | 11.0 | 11.1 |
| - all GEMMs before the push (dense FP16 + BF16) | 6.7 | 6.5 | 18.9 | 17.2 |
| activation push over PCIe (card0 compute engine) | 2.0 | 2.0 | 5.75 | 5.77 |
| host wake + routing gap | ~1-3 | ~4 | 0.6 | 4.6 |
| host enqueue of card1 work, card0 idle (uninstr.) | ~15 | ~16 | ~13.5 | ~13 |
| card1 expert kernels (gather + dequant + GEMM + SwiGLU) | 12.6 | 14.8 | 18.9 | 20.0 |
| card1 reduce + PCIe push of the FP32 sum | 4.2 | 4.2 | 12.7 | 12.7 |
| card0 own experts: GPU busy (host enqueue) | 5.1 (~9) | 5.9 (~9) | 8.2 (~7) | 8.4 (~7) |
| card0 sum read + finish | 0.8 | 0.8 | 2.4 | 2.4 |
| **modelled uninstrumented layer period** | **~49** | **~103** | **~97** | **~339** |
| instrumented median period | 125.9 | 180.3 | 144.8 | 382.9 |
| active experts card1 / card0 | 261 / 121 | 277 / 117 | 245 / 103 | 236 / 106 |

Chunk 1 (context 0-12K) is similar. It has more active experts (297/129),
QSA attention of 214 ms per layer, and modelled periods of about 98 ms (GDN)
and 296 ms (QSA). In chunk 3 (8145 tokens, context 24-32K), attention is
185 ms per layer.

### 1c. ASCII timeline, 12288-token GDN layer (uninstrumented model, ms)

```
t:      0 ............ 54.7  60.5 61   74.6     81     83      93.7  96.5
card0:  [HC|GDN|HC|rtr|shx ][push][g][..idle..][own exp][.idle.][sum+fin]
card1:  [..........idle..............][act][experts 18.9 ][reduce+PCIe 12.7][idle]
host:   [enqueue layer][cs->wait   ][grp][enq card1 ][enq card0][pp->s->wait][enq next]
```

At 4K the host enqueue is the critical path:

- Card1's 12.6 ms of expert kernels trickle out behind 15 ms of host enqueue.
- Card0's own 5.1 ms of experts start only after that and take about 9 ms of host time.
- The MoE phase is about 29 ms per layer, of which about 24 ms is pure host
  submission.

### 1d. Idle and why

The 32K request (about 20.6 s uninstrumented), by modelled bucket:

| Bucket | s | share |
| --- | ---: | ---: |
| QSA attention + selection (12 layers x 3 chunks) | ~7.9 | 38% |
| MoE exchange phase (push, host gaps, card1 critical path, sum/finish) | ~5.6 | 27% |
| - of which PCIe kernels (act push + reduce/push) | 2.35 | 11% |
| dense GEMMs before the push (dense FP16 at about 115-125 TFLOPS) | ~2.4 | 12% |
| GDN (36 layers) | ~2.2 | 11% |
| HC elementwise | ~1.4 | 7% |
| other (dense dequant 0.17, PLE, embeddings, tails) | ~0.6-0.9 | 4% |

- **Card0 idle:** about 25 ms per layer-chunk at 12288. That is about
  3.2-3.4 s per 32K request (about 16%). On top of that, card0 spends 0.74 s
  running the PCIe push kernel. At 4K card0 is idle about 21 ms per layer,
  about 1.0 s of 3.1 s (33%).
- **Card0 idles for two reasons:**
  1. The host is enqueueing card1's experts (pf:3636-3767), and card0's own
     experts are queued only afterwards.
  2. Card1's critical path continues after card0 runs out of work: 18.9 ms of
     experts plus 12.7 ms of PCIe reduce/push, while card0 has only about
     8 ms of experts. Host block 2 (pf:4474) then keeps card0 empty until
     card1 is done.
- **Card1 idle:** card1 works only in the MoE phase, about 32.5 ms per
  layer-chunk at 12288 and about 17 ms at 4K. Over the request it is about
  4.4-4.9 s busy of 20.6 s at 32K and about 0.8 of 3.1 s at 4K: **75-80% idle**.
  It always waits for card0's whole pre-MoE (attention, GDN, HC, dense) and
  for the activation push.
- **PCIe:** both cards sit on Gen3 x16 under different Threadripper root dies
  (docs/20260825_xe_nonreboot_recovery_and_pcie_topology.md). S8 measured
  posted-write P2P at 11.2 GB/s, and serving achieves it: 63 MB in 5.75 ms
  outbound, 126 MB in 12.7 ms back. The link runs at its ceiling, so only
  fewer bytes or overlap can help.
- **Host:**
  - Per 12288 layer-chunk, about 1,700 kernel appends across both cards.
    S90 recorded 255,208 appends per 32K request.
  - Plus two full queue drains per layer. Some pageable-memory memcpys also
    block inside `zeCommandListAppendMemoryCopy`: 681 calls with a 14.4 s
    union in S90, nearly all overlapping card0 GPU activity.
  - S34 also counted 117,240 `zeMemGetAllocProperties` calls, two per launch,
    inside the peer dispatch spans.

## 2. GEMM paths and XMX use

Every prefill GEMM is a oneMKL call through `dpct::blas::gemm`
(gemm.dp.cpp:822-828 BF16, 861-867 FP16). The types are `<half,half,float,float>`
or `<bf16,bf16,float,float>`, "tn", with FP32 accumulate. There is no
`gemm_batch`, DP4A or custom joint_matrix in the serving path.

The oneMKL kernels (`gemm_*_nocopy_tn_<tile>`) are XMX systolic kernels:

- Grid `{32;1;1}` is a persistent launch, one work-group per Xe core.
- Large expert GU calls (M 961-1920) reach 107-136 TFLOPS. That exceeds the
  roughly 46 TFLOPS vector-engine ceiling, so they must be on XMX.

| Weights | Path | Dequant fused? | Achieved |
| --- | --- | --- | --- |
| Dense quantized (GDN qkv/z/out, QSA q/k/v/o, shared expert; IQ4_XS/Q4_K/Q5_K/Q6_K/IQ3_S, IQ4_NL/Q2_0) | `Gemm::native`: full-matrix dequant to a 64 MiB FP16 scratch, then one oneMKL FP16 GEMM (gemm.dp.cpp:961-1013) | No | about 115-125 TFLOPS for the GEMMs, already near the measured 126-128 TFLOPS BF16 peak. Dense dequant `dequant_kernel_ec9311` reaches only 93-151 GB/s because of scalar 2-byte stores (dequant_bf16.dp.cpp:33-37): 171 ms per 32K request. |
| BF16 (HC down/up/inject, router, indexer, alpha/beta, shared gate) | oneMKL BF16 | n/a | about 53 TFLOPS, memory-bound. HC up writes 503 MB per 12288 chunk. N=4 and N=1 GEMVs re-read the activations. |
| Routed experts. Gate/up: IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S. Down: Q2_0, IQ4_NL. | Per expert: `iq_dequant_expert_f16` (both matrices, 9.83 MB FP16 out) + oneMKL GU + SwiGLU + oneMKL down. Card0 has a 2-slot ring; card1 has 1 slot and no XMX/grouped option. | No | GEMMs about 70 TFLOPS on card1 in aggregate, ranging from under 29 TFLOPS at M<=32 to 136 at M 1600-1920. Dequant runs at 515 GB/s (bandwidth-bound). Including dequant, gather and SwiGLU the expert pipe is about 40 TFLOPS. |
| QSA prompt attention | `qsa_decode_attn_batch` (qsa_decode_attn.dp.cpp:644): FP32 SIMD, one work-group per (query, KV head, 64-cell chunk), 32 queries per launch, then a merge | n/a | Per 12288-query layer: 619 GFLOP in 259 ms is about 2.4 TFLOPS, and the logical K/V gather (2.1 MB per query) is about 100 GB/s. XMX is not used. |
| GDN recurrence | `gdn_rec_cols_pipe_kernel` (kernels.dp.cpp:878): FP32, 192 work-groups walking the tokens in sequence | n/a | about 1.45 us per token step, so 17.9 ms per 12288 tokens. XMX is not used. |

Peak references:

- About 183 TFLOPS FP16 and 367 TOPS INT8 theoretical (memory note
  b70-int8-xmx-roofline, 32 Xe cores and 256 XMX at 2.8 GHz).
- 125.9 / 127.5 TFLOPS BF16 and 530 GB/s measured per card (memory note
  xpugraph-stable-on-b70).
- The often-quoted 96 TFLOPS is not supported by these notes.

Opt-in XMX code exists, but all of it is off by default and was measured slower on B70:

- `STRATA_PF_XMX=1/2`: grouped joint_matrix expert GEMM, card0 only.
  Mode 2 decodes weights in local memory. Upstream INTEL.md:879 measured it
  11-15% slower; only 29/48 layers have eligible formats.
- `xmx_gemm_iq`: fused dequant + GEMM, 4-5x slower.
- `STRATA_PROMPT_ATTN_XMX=1`: about 2x slower than the FP32 kernel. It keeps
  one query per work-group (16-row tile with 4 rows of padding, hi/lo Q split).

The FP16 tile round trip:

- At 12288 tokens, experts write about 3.2 GB of FP16 per layer and read it
  back. That is about 462 GB per 32K request, against about 82 GB of
  quantized weight reads.
- Dense layers add about 5.8 GB per chunk.
- This is the DPAS-feeding inefficiency. It only costs latency where it sits
  on a critical path, which today is card1 (4-5 ms of dequant per layer).

## 3. EP split and pipelining

Nothing overlaps across cards: card1 never works during card0's attention
or GDN. Within the MoE phase the two cards overlap only partly. Card0's
experts start after card1's enqueue and finish long before card1's
reduce/push.

**Two-microbatch pipelining** (tokens A = first half, B = second half of the
chunk) is legal:

- Layer L+1 for half A needs only layer L's output for half A.
- GDN is a forward scan, so A then B in order is fine.
- QSA queries in B need KV from A, which is already appended in queue order.

The schedule:

```
card0: preMoE(L,A) push(A) | preMoE(L,B) push(B) own-exp(L,A) fin(L,A) | preMoE(L+1,A) ...
card1:                     | exp(L,A) reduce/push(A)                   | exp(L,B) reduce/push(B) ...
```

Per 12288 GDN layer:

- card0 work is about 54.7 + 5.75 + 8.2 + 2.2 (duplicated dequant) + 2.4,
  about 73 ms;
- card1 work is about 2 x (~11.5 + 6.35), about 36 ms, which hides behind
  card0;
- the layer drops from about 97 to about 73 ms;
- a QSA layer drops from about 339 to about 321 ms;
- **total: about -1.0 s per 12288 chunk (-13 to -14%), about -2.6 s per 32K request.**

Preconditions:

1. Host submission must not block. Replace `m.cs->wait()` (pf:3494) and
   `m.pp->s->wait()` (pf:4474) with waits on specific events (route-done(A),
   reduce-done(A)), or with device flags. The `ep_push`/`ep_wait` doorbells
   already exist for decode (ep_kernels.dp.cpp:88-120).
2. Host enqueue per layer doubles (two halves of about 245 experts each,
   about 27 ms plus about 14 ms on card0). That still fits inside about 73 ms
   of GPU time at 12288, but not at 4K. So item 1 in section 5 (fewer
   launches) comes first.
3. Dequant is duplicated per half: card1 +4 ms per layer (hidden), card0
   +2.2 ms (exposed). A fused dequant GEMM (item 8) removes this cost.

The activation push stays on card0's compute engine either way:

- About 5.75 ms per layer at 12288. A copy-engine P2P memcpy measured only
  1.65 GB/s (S5), so moving it there is worse even when asynchronous.
- With a single compute engine, a second queue does not run truly concurrently (INTEL.md:879).

**Rebalancing experts:**

- Today the MoE critical path is card1 (19 + 13 ms) against card0 (8 ms).
  Moving about 60 experts per layer to card0 would balance the phase, but
  needs about 5 GB on card0, which does not exist with two 200K INT8 slots.
- After pipelining the balance flips: card0 becomes the bottleneck at about
  73 ms per layer, card1 at about 36. The useful move would then be expert
  work away from card0, and card1 has only about 2.4 GB free.
- Conclusion: little room by reassigning weights. Overlap is the lever.
- Routing skew is high: about 326 of 512 experts are active per chunk, and
  52% of active experts get 64 rows or fewer. Rebalancing by count is noisy;
  any rebalance should weight by measured popularity.

**Giving card1 non-MoE work:**

- QSA has 2 KV heads with 12 query heads each, so splitting attention by KV
  head is natural.
- Card1 would need its head's KV for the 12 QSA layers: about 6 KB per token,
  so 0.2 GB per 32K slot, but about 2.4 GB for two 200K slots.
- Traffic is Q for 12 heads FP16 plus output back, about 150 MB per 12288
  layer: about 14 ms of PCIe against about 130 ms of attention saved per
  QSA layer.
- Workable only up to about 64K context, or with the KV-streaming machinery.
  It overlaps with item 2, so it is ranked below it.

## 4. What the host does per layer

- Two full drains per layer: pf:3494 and pf:4474.
- About 0.6 ms of routing loops over 122,880 entries at 12288.
- Enqueue of about 1,135 card1 kernels per layer-chunk: 5 per active expert
  plus about 6 fixed.
- Enqueue of 4 kernels per active card0 expert.
- The S34 historical 4K count: 58,716 card1 appends in 48 layers (median
  1,307 per layer), on one thread, plus two `zeMemGetAllocProperties` per
  launch.
- Uninstrumented S38: 15.8 ms per layer median for 272 experts, about
  58 us per expert or about 12 us per launch on average. oneMKL submissions
  are likely the expensive ones.
- Experts are appended one at a time. No grouped launch exists on card1. The
  card0 opt-in grouped XMX is slower per kernel.
- 52% of active experts have M<=64. Each still pays about 48 us of GPU time
  (22.5 us dequant, 7 us GU, 5.5 us down, gather, SwiGLU) and about 55 us of
  host time. These small experts make up most of the launch storm.

Prior evidence to respect:

- S7: per-layer `std::async` card1 submission was slower (4K 1046.8 -> 1002.3).
  Do not retry it unchanged.
- S38: the one-gather alias (`STRATA_EP_PREFILL_GATHER_ALIAS=1`) removes one
  launch per card1 expert, worth +1.8% at 4K and +0.65% at 32K. Default OFF.
- S42: the oneMKL group `gemm_batch` for GU differs only at rounding level
  (max 7.45e-8, median 2 ULP). It failed the bitwise gate; speed was never
  measured. Per the S42 dev-loop correction, a changed reduction tree needs a
  declared accuracy contract, not bitwise equality.

## 5. Ranked opportunities

Gains are modelled from the numbers above, not measured, and they are not
additive. "Per chunk" means one 12288-token chunk at 12-24K context. Each
32K request has about 2.66 chunk-equivalents.

| # | Opportunity | Main code | Est. gain | Risk |
| --- | --- | --- | --- | --- |
| 1 | Remove the MoE launch storm and host serialization | pf:3494, 3512-3600, 3636-3767, 4319-4407, 4474 | 4K: -0.6 to -0.9 s (to about 1750-1950 tok/s); 32K: -0.3 to -0.6 s | Medium-high |
| 2 | QSA prompt attention: query-tile union on XMX | pf:3329-3350; qsa_decode_attn.dp.cpp:644-770; qsa_prompt_attn_xmx.dp.cpp:223-308 | 32K: -3 to -4.5 s per request (attention 6.4 s -> 2-3 s); 4K: -0.3 to -0.5 s | High |
| 3 | Two-microbatch cross-card pipeline | run_impl layer loop pf:2887-4720 | 32K: about -1.0 s per chunk (-2.6 s per request); 4K: -0.3 to -0.4 s after #1 | High |
| 4 | Cut PCIe bytes on the critical path | ep_kernels.dp.cpp:29-52, pf:3765, 4477; pf:3487 | FP16 return: -6.3 ms per layer, -0.3 s per chunk, -0.8 s per 32K; -0.1 s at 4K | Low-medium |
| 5 | Quick, low-risk host and order fixes | pf:3636-3767 vs 4407; env | 4K: -0.1 to -0.25 s; 32K: -0.2 to -0.6 s | Low |
| 6 | GDN chunkwise-parallel recurrence on XMX | kernels.dp.cpp:878, 2624; pf:3051 | -13 ms per GDN layer: -0.47 s per chunk, -1.2 s per 32K; -0.17 s at 4K | Medium-high |
| 7 | HC traffic elimination | pf:2955-3030, 4658; gr_* in kernels.dp.cpp | -4 to -6 ms per layer at 12288: -0.6 to -0.8 s per 32K; -0.1 s at 4K | Medium |
| 8 | DPAS-fed experts: decode in SLM or int8 MMQ instead of the FP16 round trip; direct grouped path for small experts | pf:3747-3762, 4319-4387; iq_kernels.dp.cpp:4744; xmx_moe.dp.cpp | card1 -4 to -6 ms per layer while card1 is critical: -0.6 to -0.8 s per 32K. Removes #3's duplicated dequant. | High |

**1. Remove the MoE launch storm and host serialization.** This is the largest
lever at 4K and the enabler for #3. Steps, cheapest first:

1. Small experts (M<=64, about 52% of active experts): send every one of them
   to a single grouped launch per card per layer. Use a direct-from-quantized
   kernel driven by a device-side expert list, in the spirit of decode's
   `native_expert_grouped`/`resident_plan_ep`. This removes about 600 of
   about 1,135 card1 launches per layer, plus their 22.5 us dequants
   (about 2.7 ms per layer on card1, 1.2 ms on card0).
2. Large experts: waves of 16-32 experts. One batched dequant per wave into
   16 slots (157 MB on card1, which has 2.4 GB free), one oneMKL group
   `gemm_batch` for GU, one SwiGLU, one group GEMM for down. The S42 adapter
   already exists (lowlevel_probe/grouped_peer_oracle.cpp). That is 4 launches
   per wave instead of 5 per expert.
3. End state: compute counts and offsets on the GPU (histogram + scan of the
   routing ids), keeping the same row order so the reduction is unchanged.
   Signal card1 with the existing doorbell kernels (`ep_push`/`ep_wait`,
   ep_kernels.dp.cpp:88-120). The host then never drains a queue inside a
   layer.

At 4K this turns the about 24 ms per layer of host submission into GPU-bound
work. Card1 has about 12.6 ms of experts plus 4.2 ms of reduce, overlapping
card0's 5 ms.

**2. QSA prompt attention: query-tile union on XMX.** This is the largest lever at 32K.

The current kernel is gather- and latency-bound:

- Each (query, KV head) streams its own 2,051 selected cells, 1.05 MB of
  INT8 K+V.
- 12 heads share each load, all on FP32 SIMD at about 2.4 TFLOPS.

The proposal:

- Tile 8-16 consecutive queries per KV head, giving 96-192 matrix rows
  instead of 12-16.
- Build the union of their selected 4-cell blocks with a per-query mask.
  Upstream measured the union of 8 positions at 3.3x one position's cells,
  and 16 positions at 5.1x, at 80K context. At 4-32K it is likely tighter.
- Run masked QK^T and PV with online softmax on joint_matrix. Reuse the hi/lo
  FP16 split and per-64-dim INT8 scale handling from qsa_prompt_attn_xmx.
- K/V traffic falls 2.4-3.1x: about 10.6 GB per layer instead of 25.8 GB
  logical. Arithmetic rises 3.3-5x, but on XMX: about 2 TFLOP per layer is
  about 34 ms at 60 TFLOPS.
- Target about 40-80 ms instead of 215-259 ms per 12288-query layer.
- Upstream rejected grouping by costing the extra math on the FP32 path. That
  objection does not hold once the math is on XMX.
- First step: a cheap CPU check of union sizes on our prompts, using
  `STRATA_DUMP_SEL` (pf:3312), before writing the kernel.
- The top-k scorer (`qsa_block_scores`, about 0.38 s per chunk, growing with
  context) is a separate, later item.

**3. Two-microbatch cross-card pipeline.** See section 3. It hides card1's
experts, the 12.7 ms return push and most host gaps behind card0's other half.

- Needs event or doorbell waits instead of pf:3494 and pf:4474, and #1 so the
  host keeps up.
- Numerics: expert M per call changes, so oneMKL tile choice and rounding may
  change. Needs an accuracy contract.

**4. Cut PCIe bytes on the critical path.**

- Return the per-token peer sum as FP16, or BF16 with care: 126 -> 63 MB per
  layer at 12288. The relative error on the peer contribution is about 5e-4,
  added into the FP32 residual, so it needs a quality gate.
- The activation push is already FP16. Int8/FP8 per-row activations
  (63 -> 32 MB, -2.9 ms per layer) change the peer experts' inputs. That is
  MMQ-style numerics with a higher quality risk; optional.
- Keep the posted-write kernels: the copy engine is 7x slower (S5/S8).
- The link is the topology ceiling (Gen3 x16, cross-root). A same-root slot
  move is a hardware A/B with at most a small gain.

**5. Quick, low-risk host and order fixes.** Measure as one bundle.

- Enqueue card0's experts before card1's loop, or interleave about 1 card0
  expert per 2.4 card1 experts. At 4K card0's experts then overlap the card1
  enqueue instead of following it. At 12288 interleaving keeps card1
  GPU-bound from the start.
  - Note: card0-first costs card1 a delayed start, so interleaving is safer.
  - This differs from the rejected S7 async: it stays on one thread with
    the same calls, only reordered.
- Turn `STRATA_EP_PREFILL_GATHER_ALIAS=1` on (measured +1.8% at 4K).
- Turn `STRATA_ATTN_VALUE_PIPELINE=1` on (S113: +2.7% at 32K, outputs matched).
- Give `dequant_kernel_ec9311` vector stores (171 -> about 60 ms per 32K).
- Fold HC inject (N=4) into HC down (N=320), and the shared gate (N=1) into
  the router: about 0.17 s per 32K.
- Remove the blocking pageable memcpys (pf:3484, 3489, 3685-3689) by using
  pinned staging.

**6. GDN chunkwise-parallel recurrence on XMX.**

- Replace the token-sequential column walk (17.9 ms per 12288 tokens) with
  the chunked gated-delta-rule form, as in flash-linear-attention with chunk
  size 64.
- Intra-chunk products and state updates become 64x128x128 matmuls. Only 192
  sequential inter-chunk steps per layer remain.
- About 100 GFLOP per layer on XMX: target 3-5 ms.
- FP32-level differences need a state-continuity oracle. The S145/S147
  GDN-commit harness can be reused.

**7. HC traffic elimination.** The kernels already run at 75-100% of DRAM
bandwidth, so tuning them is pointless. Remove traffic instead:

- Never materialize `gated` (T x 10240 FP32, 503 MB written and read per
  half at 12288). Fuse the HC up-projection epilogue into `gr_mix_r`, as
  upstream's `gr_upmix` does (`STRATA_HC_UPMIX`, gfx11 only; port it).
- Merge the N=4 inject GEMV into the down GEMM.
- This is about 2 GB less DRAM traffic per layer at 12288.

**8. Feed DPAS straight from quantized experts.**

- Option A: an MMQ-style int8 path. q8_1 activations times the IQ codebook
  as int8 into int8 DPAS (2x the FP16 rate), with per-32 scales in an FP32
  epilogue. S74 confirmed the int8 DPAS instruction on BMG AOT.
- Option B: decode IQ blocks into SLM FP16 tiles inside an oneMKL-class
  tiled kernel.
- Either removes the 3.2 GB per layer FP16 write/read and the per-expert
  dequant launch.
- Upstream's simple fused kernel lost 4-5x, so this needs real tiling work.
- Worth it only while card1 is critical (before #3), or to make #3's
  per-half duplicated dequant free.

**Projection (modelled, not additive certainty):**

- **4K, 3.12 s today:** #1 (-0.8) and #5 (-0.15) give about 2.2 s. With #2
  at 2x on attention (-0.36), about 1.85 s, roughly 2,150-2,200 tok/s.
- **32K, 20.6 s today:** #2 at 2-3x (-3.2 to -4.3), #4 (-0.8) and #1 (-0.4)
  give about 15.1-16.2 s, roughly 2,000-2,150 tok/s. Adding #3 (-2.6) gives
  about 12.5-13.6 s.
- **The single biggest items:** #2 at 32K, #1 at 4K.

## 6. Cheapest next measurements (queue between serves)

1. CPU only: dump QSA selections (`STRATA_DUMP_SEL`) for the 4K and 32K
   fixtures. Measure the union size per 8/16-query tile per KV head.
   Go/no-go for #2.
2. One bundled serve for #5 (alias, value pipeline, interleaved enqueue) with
   S38-style buffered host markers. Confirms the 4K host-bound model:
   peer-loop ms against card0 idle.
3. Standalone card1 timing of S42-style grouped GU/down at M<=64 and M>=256
   waves, against per-expert oneMKL. Record host enqueue time alongside
   device time: the win is host-side.
4. A FP16-return variant of `ep_prefill_reduce` in the existing PUSH oracle:
   bytes, time and error distribution against FP32 (#4).

Files: per-layer extraction script and JSON in
`/mnt/vm_8tb/b70/results/night-20261016/agents/prefill_analysis_work/`
(`s90_layers.py`, `s90_layers.json`, `s10_layers.json`, `s10_gpu.csv`). GEMM/kernel census
by the helper agent in `/mnt/vm_8tb/b70/results/night-20261016/agents/gemm_scratch/`.
