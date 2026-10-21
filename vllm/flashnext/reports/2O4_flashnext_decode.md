# O4 -- Flash-Next no-MTP decode latency on the vLLM sidecar

Owner: O4 (Claude Opus 5.5). Date: 2026-10-20. Layout: EP sidecar as H1e
(world size 1; card 0 = trunk + 138 experts/layer, card 1 = 374
experts/layer via `b70_ep_xpu`), `FULL_DECODE_ONLY`, FLASH_ATTN,
max_model_len 2048, no MTP, `B70_QSA=0`. Model ISTA GSQ-RCO IQ3_XXS
(qwen4_exp, 48 layers, 512 experts top-10, H=2560, hc=4, PLE on layer 1).
Engine v0.32.1rc1.dev94+g5fd62688f, fork branch b70/flashnext (uncommitted
working tree; see "Code" below). Raw data:
`/mnt/vm_8tb/b70/build/vllm-flashnext/o4/run-*` and `serve-*`.

Bar: Strata no-speculation C1 66.2 tok/s (15.1 ms, localmaxxing Verified),
the task's "13-14 ms" estimate, and the H1e starting point (C1 ITL p50
60.2 ms on the server bench).

## Result

| run | change (cumulative) | C1 step ms | C1 graph ms | C2 step ms | C2 graph ms |
|---|---|---:|---:|---:|---:|
| base | H1e recipe, o3f MoE lib | 80.39 | 24.11 | 32.24 (mean 45.2) | 29.47 |
| ringmlock | X1 PLE ring + `B70_PLE_MLOCK` | 26.72 | 23.81 | 32.14 | 28.61 |
| f1 | fused HC, shared-expert overlap, lean | 20.81 | 18.27 | 24.65 | 21.41 |
| f2 | MoE lib o3g (O3) | 20.92 | 18.32 | 24.59 | 21.39 |
| f3 | shared expert fused into card-0 EP call | 19.98 | 17.36 | 23.57 | 20.37 |
| f4 | faster PLE host stage | 19.78 | 17.35 | 23.15 | 20.38 |
| f5 | split-K HC down GEMV | 19.04 | 16.65 | 22.48 | 19.71 |
| f6 | PLE fully on device (no host sync) | 17.43 | 16.55 | 20.82 | 19.66 |
| f7 | GGUF GEMV bf16 epilogue into one output | 16.68 | 15.80 | 19.92 | 18.78 |
| f8 | Triton top-k, GDN b|a one GEMM | 16.43 | 15.59 | 19.70 | 18.58 |
| f9 | small dense via split-K, bf16 EP sum | 16.29 | 15.43 | 19.53 | 18.37 |
| f10 | f9 + `B70_EP_Q0=150` (not kept) | 16.31 | 15.46 | 19.54 | 18.38 |
| f11 | f9 + Q8 HC GEMVs (numerics change) | 15.70 | 14.85 | 18.93 | 17.86 |
| f12 | + PLE tail as 2 Triton kernels | 15.49 | 14.60 | 18.79 | 17.65 |
| f13 | + IQ3_S dense on the row kernel | 15.24 | 14.40 | 18.68 | 17.54 |
| f14 | + attn combine/MLP norm fused, no y0 fill | 15.22 | 14.34 | 18.61 | 17.49 |

"step" = in-process execute_model-to-execute_model interval (p50 of 94
decode steps, `harness/o4_decode_profile.py`); "graph" = card-0 event pair
around the FULL graph replay. C2 is two identical prompts; its step yields
one token per stream. Every row: greedy 17*19 -> 323 and 23*29 -> 667,
C2 streams identical, health HEALTHY before and after. From f6 on the
C1/C2 token ids over 96 tokens are identical run to run (f6 = f5; f7 = f6;
f8-f14 = f7 chain): the device PLE, bf16 epilogue, top-k, dense GEMV, Q8
HC, fused PLE tail and IQ3_S row-kernel changes did not move a token in
this test (Q8 HC and top-k are not bitwise at the op level).

HTTP server (harness/o4_serve.sh, `bench_latency.py` under `cpu-run
--quiet`, decode 64, prefill 1024; identity first `hotschmoe-dev` then
`qwen3.8-flash-next-ista-gsq-rco-iq3-xxs-o4`; greedy 323/667; coherence
OK/OK same; health HEALTHY before and after):

| serve | recipe | C1 ITL p50 / p90 / p99 ms | C1 tok/s | C2 ITL p50 ms (per stream) | TTFT C1 ms |
|---|---|---|---:|---|---:|
| H1e (start) | H1e sidecar | 60.16 / 89.40 / 98.68 | 18.4 | 32.06 / 32.06 (means 56.3) | 265 |
| s1 | o4_recipe at f9 (bf16 HC) | 16.22 / 16.53 / 16.86 | 61.7 | 20.26 / 20.25 | 224 |
| s2 | s1 + Q8 HC | 15.57 / 15.97 / 16.35 | 64.0 | 19.84 / 19.84 | 227 |
| s3 | o4_recipe at f12 | 15.42 / 15.64 / 15.81 | 64.9 | 19.68 / 19.69 | 230 |
| s4 | o4_recipe at f13 (current) | **15.25** / 15.48 / 16.26 | 65.6 | **19.62** / 19.62 | 228 |
| Strata no-spec (bar) | night-final, localmaxxing Verified | 15.1 | 66.2 | | |

C1 no-MTP went 60.2 -> 15.25 ms (3.9x). It is 1% above Strata's no-spec
15.1 ms, not below it; C2 per stream is 19.6 ms (Strata's 10.9 ms C2 is
with MTP; no no-spec C2 reference was found). Prefill on these serves is
151 tok/s (eager prefill, X3/F1 scope; C2 TTFT ~3 s is the second
stream's prefill). s1's script first benched F1's server, which answered
on the same port before this container was ready; that run was discarded
and repeated by hand (serve-s1/NOTE.txt); o4_serve.sh now uses port 18147
and waits for its own alias.

## Step timeline, before

Base (H1e recipe, `o4/run-base`, C1, ms per step, host clock):

| phase | p50 | note |
|---|---:|---|
| prep (inputs, metadata) | 3.1 | overlaps the previous graph |
| PLE: `input_ids.to(cpu)` | 20.2 | waits for the previous graph + sample |
| PLE: hash | 0.18 | Python loop |
| PLE: host row gather | 53.3 | cold rows of the 26.8 GiB memmap: 16 major faults, 0-86 ms; 0.18 ms when cached |
| graph replay (device) | 24.1 | card 0 |
| sample (device) | 0.83 | lm_head Q5_K 0.77 ms |

So ~2/3 of H1e's 60 ms ITL was page faults on the PLE table, and the
staged-PLE path also baked capture-time conv slots into the graph (C2
streams diverged; fixed by X1's ring).

Card-0 graph, unitrace kernel census (`o4/run-ut2`, ringmlock config;
tracing inflates gaps, kernel sum 24.15 ms): 4827 kernels per step.
oneDNN gemm 460 calls 5.23 ms (HC down/up/inject 291), dense GGUF GEMV
~6.4 ms, dtype casts 757 calls 2.41 ms, HC elementwise/reduce ~3.5 ms
(~46 kernels per layer), `ep_wait` 48 x 26 us = 1.26 ms (card 0 idle,
waiting for card 1), card-0 MoE ~1.5 ms, TopKGating 0.59, q8 activation
quant 0.53, GDN core 0.71, FA 0.27, dead `residual.mean` 0.17. Card 1:
576 kernels (48 x 12), mostly spinning on flags.

## Step timeline, after (f7, `o4/run-f7ut`)

2259 card-0 kernels per step, kernel sum 16.54 ms including lm_head:
gemm 2.30 (HC up 97, router, b|a), dense GGUF GEMV ~5.1 (Q6_K ~460 GB/s,
IQ3_S worst), HC down split-K 1.51, card-0 MoE incl. fused shared ~3.1,
TopKGating 0.61, GDN 0.72, PLE ~0.6 (about 150 small kernels), q8 quant
0.30, casts 0.33, `ep_wait` 0.34 (card 1 now ~7 us/layer later than
card 0). GPU idle between steps ~0.05 ms after f6 (was ~3 ms of
host-serial work plus the faults). Graph covers the whole decode step;
no eager pieces remain except sampling bookkeeping.

## Root causes and fixes

1. PLE host gather page faults (53 ms). First fix `B70_PLE_MLOCK=1`
   (fork `gguf/ple.py` `pin_host_table`, mlock of the memmap). Final fix
   `B70_PLE_DEVICE=1` (fork `gguf/ple_device.py`): the table is copied
   once into 26.82 GiB of pinned host USM (`b70_o4.host_table`, a DataPtr
   storage because `from_blob` rejects host USM on an XPU device); the
   n-gram hash (int64 wraparound + unsigned modulo), a position-indexed
   device token ring, the per-token slot/start/end metadata and the row
   gather (GPU index_select over PCIe) all run inside the graph; X1's conv
   ring is reused. This removes the per-step D2H sync. Outputs identical
   to the host ring. Cost: 122 s extra at load (cold copy; madvise
   SEQUENTIAL/WILLNEED added afterwards, unmeasured), 26.8 GiB pinned RAM.
2. Hyper-connection kernel storm (~7.9 ms of the graph). `B70_DECODE_HC=
   triton` (fork `xpu/hc_decode.py`): norm, silu(lo/hc), sigmoid-gate
   collapse and combine as Triton kernels with torch's bf16 rounding
   points (0 ulp vs eager HC, `harness/o4_hc_test.py`); inject rows
   appended to the down weight (one GEMM, memory neutral). 97 mix + 96
   combine at T=1: 7.88 -> 4.03 ms. `B70_DECODE_HC_GEMV=1`: the
   [324, 10240] down GEMM on `b70_o4.bf_gemv` (split-K 4, 15.5 vs 19.7
   us). `B70_DECODE_LEAN=1` drops the dead per-layer mean.
3. Shared expert serialized before the EP dispatch. `B70_EP_SHARED_OVERLAP
   =1` (moe_ep_begin/end split) then `B70_EP_FUSE_SHARED=1`: card 0's own
   experts and the gated shared expert in one O3 `b70_moe_layer2` call
   inside `moe_ep`, overlapping card 1 (b70_ep_xpu `ep_register_shared`,
   `moe_ep_sh`). `B70_EP_BF16_OUT=1` writes the card-0 + card-1 sum as
   bf16 in the add kernel (`moe_ep_shb`).
4. Per-linear glue: fp32 GEMV output + cast + cat per shard. O4 GGUF
   library variant `_C.o4g.so` adds `gemv_out` (bf16 epilogue, output
   stride; bitwise equal to gemv().to(bf16), `harness/o4_gemv_out_test.py`)
   and fork `gguf/linear.py` writes all shards into one bf16 output
   (`B70_GEMV_BF16=1`). `B70_LINEAR_CAT=1` (GDN b|a one GEMM),
   `B70_DENSE_GEMV=1` (router and b|a on `bf_gemv`).
5. Host stage (after mlock): one D2H per metadata tensor and Python
   hashing between the sampled token and the replay (`B70_PLE_FAST=1`,
   fork `gguf/ple_fast.py`; superseded by item 1's device path).
6. Router top-k 12.7 us/call: `B70_DECODE_TOPK=1` (fork
   `xpu/topk_decode.py`), 5.6 us, same expert ids on 600 random cases,
   weights within 1-3 ulp (not bitwise).

Not kept: Triton bf16 GEMV (slower than oneDNN); a 4-deep load unroll of
the O4 GEMVs (slower); `B70_EP_Q0=150` (no change). Q8 HC (Strata-style,
numerics change) -- see f11 below.

Q8 HC (f11, `B70_DECODE_HC_Q8=1`): the HC down|inject and up GEMVs use
Q8_0-rounded int8 copies (`b70_o4.w8_gemv`, ~12 us floor per call vs
15.5/13.4 us bf16), -0.6 ms per step, +0.67 GB on card 0, ~0.5% relative
error per HC GEMV output vs bf16. This matches Strata's DD numerics
(STRATA_NIGHT_HC=q8); it is in the recipe and can be dropped for a
bf16-exact path (s1 row).

Later steps: `B70_PLE_FUSED=1` (ple_device.py `_ple_gate`/`_ple_conv`,
-0.21 ms), `B70_IQ3S_ROW=1` (IQ3_S dense ran at 128-206 GB/s; the
existing row kernel with one expert id is 22-37 vs 50-66 us per call and
bitwise equal; -0.25 ms), `B70_DECODE_HC_FUSE=1` + `b70_ep_C.o4d.so`
(-0.02 ms, optional).

## Strata comparison

Strata (`/mnt/vm_8tb/b70/build/strata-ab/src-ep`, docs/20261016_night_lab.md)
already does the things vLLM lacked: PLE host gather once per token
before the graphs with an O_DIRECT async reader and a row cache
(`sycl/src/ngram/ple_reader.cpp`), so no page faults; fused HC v2
(norm+down+reduce, Q8 HC weights); shared expert after card-0 experts and
before the peer wait (N5); KREG MMVQ dense GEMV (76-94% of roofline);
batched GPU-side EP exchange. Strata no-spec is 15.1 ms/token C1. This
vLLM sidecar is now at 15.25 ms on the server bench (1% above) and uses
fewer host round trips (the PLE is fully in-graph). Remaining card-0
costs vs the ~6.7 ms byte floor: ~2260 kernels with 1.3-3 us fixed cost
each, HC GEMVs at a ~12-15 us latency floor per call, card-0 MoE ~3 ms.

## Next steps (ranked)

1. Router + top-k + EP prelude (bump, dispatch, plan) is ~20 us per
   layer serialized before card 1 can start, and card 1 is now ~7 us per
   layer later than card 0: fuse top-k into the router GEMV or into the
   EP dispatch kernel (~0.2-0.3 ms).
2. HC GEMVs sit at a ~12-15 us per-call latency floor for 3-6 MB; a
   streamed kernel covering a layer's four HC GEMVs (~0.5 ms).
3. PLE device stage still ~40 small kernels (~0.06 ms); IQ4_XS dense at
   N=6144 runs at ~290 GB/s (17 tensors, ~0.1 ms).
4. q8 activation quant shared between shards of one input (~0.1 ms).
5. MoE kernel wins (O3/O5) move C1 directly; `B70_EP_Q0=150` alone did
   nothing.
6. PLE table pin at load: 122 s -> 88 s with madvise; pread into the
   pinned buffer would cut it further. The pinned table is 26.8 GiB of
   host RAM per Flash-Next process.

## Code (uncommitted)

Fork `/mnt/vm_8tb/b70/build/vllm-flashnext/vllm`:
new `vllm/models/qwen4_exp/xpu/hc_decode.py`, `xpu/topk_decode.py`,
`gguf/ple_device.py`, `gguf/ple_fast.py`; edits in `xpu/model.py`
(opt-in hooks, positions to PLE, combine/mix fusion), `gguf/linear.py`
(gemv_out, dense cat, dense GEMV, IQ3_S row kernel), `gguf/ple.py`
(mlock). All default off.
Plugins: new `vllm/flashnext/plugins/b70_o4_xpu` (bf_gemv, w8_gemv,
host_table; `build.sh`, AOT); `b70_ep_xpu` csrc/method/ops additions
(moe_ep_begin/end, ep_register_shared, moe_ep_sh/shb, B70_EP_SO);
`b70_gguf_xpu` csrc additions (gemv_out, gemv2). Built libraries (in
`/mnt/vm_8tb/b70/build/vllm-flashnext/o4/`): `_C.o4g.so` (GGUF + O3
stage 3 + gemv_out, sha 2eac42c3), `b70_ep_C.o4c.so` (sha 10495833),
`b70_ep_C.o4d.so` (sha 66975d09), plugin `b70_o4_xpu/_C.so` (sha
b9268747). Building `_C.o4g.so` with the plugin script also rewrote
`plugins/b70_gguf_xpu/libdequant_cpu.so` from unchanged source.
Harness: `o4_decode_profile.py`, `o4_window.sh`, `o4_serve.sh`,
`o4_recipe.env`, `o4_trace.py`, probes `o4_hc_test.py`, `o4_bf_bench.py`,
`o4_w8_bench.py`, `o4_gemv_out_test.py`, `o4_hosttab_test.py`,
`o4_ple_hash_test.py`, `o4_topk_test.py`, `o4_dense_bench.py`,
`o4_gemv_bench.py`.

## Serve recipe

`TAG=x bash vllm/flashnext/harness/o4_serve.sh` (copy the script first;
it reads `harness/o4_recipe.env`). Sidecar base as H1e plus
`--no-enable-prefix-caching --mamba-cache-mode align` (required by the PLE
ring), served names `hotschmoe-dev` then the alias, max_len 2048,
FULL_DECODE_ONLY capture 1,2,3,4,8, FLASH_ATTN, util 0.98, max_num_seqs 4,
batched 2048, SYCL_CACHE_PERSISTENT=0. Flags (all opt-in):
`B70_MTP_RT` (ring), `B70_PLE_DEVICE`, `B70_PLE_FUSED`, `B70_DECODE_HC=
triton`, `B70_DECODE_HC_GEMV`, `B70_DECODE_HC_Q8`, `B70_DECODE_LEAN`,
`B70_GEMV_BF16`, `B70_IQ3S_ROW`, `B70_LINEAR_CAT`, `B70_DENSE_GEMV`,
`B70_DECODE_TOPK`, `B70_EP_SHARED_OVERLAP`, `B70_EP_FUSE_SHARED`,
`B70_EP_BF16_OUT`, `B70_EP_SO=o4/b70_ep_C.o4c.so`,
`B70_GGUF_SO=o4/_C.o4g.so`, PYTHONPATH + `plugins/b70_o4_xpu`. Do not
combine `B70_PLE_DEVICE` with `B70_PLE_MLOCK`/`B70_PLE_FAST`. Host RAM:
the pinned PLE table adds 26.8 GiB; card 0 KV after load 10.9 GiB at
max_len 2048 (f10 log).
