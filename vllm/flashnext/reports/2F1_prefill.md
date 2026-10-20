# 2F1 -- Flash-Next prefill on the EP sidecar (vLLM, world size 1)

Agent F1 (Claude Fable 5.1, claude-fable-5-1), 2026-10-20. Takes over X3's
prefill findings (reports/1X3_prefill.md) and moves them onto O4's sidecar
layout (reports/1H1_ep_sidecar.md, JOURNAL O4 f1..f4). Target 2000+ tok/s;
bar Strata night-final-v3 ~2200 tok/s (QSA on) on the same model and cards.

## 1. Why the sidecar prefill was weak

`b70_ep_xpu` (H1) routes every MoE call, whatever T, through `moe_ep` ->
`b70_moe_layer`, the decode kernel (M chunked by 8, one weight pass per
8 rows). H1e measured 106 tok/s at 1K on that path. The dense trunk on
card 0 was also on the decode GEMV unless `B70_PREFILL=1` (X3's dispatch
fix, +19x on the 35B proxy). Nothing shipped prefill activations to card 1
except that decode push, and card 1 ran its 374 experts with the GEMV.

## 2. Design (what was built)

Layout unchanged: card 0 = trunk, KV, PLE host table (mlocked), router,
shared expert, 138 routed experts (Strata traffic snake, q0=138); card 1 =
374 routed experts and nothing else. Decode keeps O4's f4 recipe and its
captured sidecar graphs untouched. Prefill (T > 8) is a new eager path.

Native, `plugins/b70_ep_xpu/csrc/prefill_ep.cpp` (new file; `moe_ep.cpp`
only gains the accessor `ep_transport_path()`):

- Own regions (inbox on card 1, psum on card 0, one 4 KiB flag page per
  layer each way), own epoch counters/done words/host error words. Same
  `region_alloc` + page touch as the decode sidecar, path probed by
  `ep_init` (same-context peer pointer, path 2).
- All kernels enqueue on torch's current streams
  (`c10::xpu::getCurrentXPUStream(0|1)`), so torch ops on either card
  order against them without events. The decode plugin queue is not used.
- `ep_prefill_dispatch(x bf16 [T,K], w fp32 [T,topk], ids i32, map0, map1,
  n0, e1, layer) -> (ids0 on card 0, x1/w1/ids1 fresh tensors on card 1)`:
  card 1 first bumps its epoch, EU-spins on `flag_in[layer]` (bounded,
  default 1e9 spins, error word on timeout) and copies the inbox uncached;
  card 0 then runs `ep_dispatch` (H1 kernel: remaps global ids to each
  card's local slot, -1 elsewhere, posted-writes ids/w/x into the inbox,
  system-scope barrier, uncached flag store).
- `ep_prefill_return(y1 on card 1 fp32|bf16, y0 on card 0 fp32, layer) ->
  y`: card 1 `ep_push_ctr`s its partial sum into card 0's psum (done
  counter, last group raises `flag_ret[layer]`); card 0 spins, then adds
  psum with uncached loads (`ep_add_uncached`, or a new bf16 variant).
- Error words are checked (host-mapped, no sync) at every dispatch; a
  timeout raises instead of hanging. `ep_prefill_sync` exists for probes.

Python, `b70_ep_xpu/prefill_ep.py`, hooked at the top of
`B70EPMoEMethod.apply` behind `B70_EP_PREFILL=1`:

1. dispatch (above);
2. card 0: shared expert (when the method owns it, O4's overlap flags),
   then its 138 experts through X3's dispatcher
   `serve_prefill.moe_prefill` (C13 wide dequant + bmm above
   `B70_PREFILL_DQ_T`, power-of-two row buckets, tile 64, grouped Triton
   pad/combine; fused DPAS kernel below the threshold);
3. card 1 (under `torch.xpu.device(1)`, since every b70_gguf op takes the
   current device's stream): its 374 experts with the scratch-free fused
   DPAS grouped op `prefill_moe_layer`, in token sub-chunks
   (`B70_EP_PREFILL_C1_CHUNK`, default 2048) because the fused op allocates
   about 32 KB of fp32 scratch per expanded row and card 1 has about
   1.1 GiB free beside 29.2 GiB of experts;
4. return + add; output cast to the model dtype.

Host order is card 0 first: its grouping `.cpu()` wait only covers work
card 0 needed anyway (the dispatch), card 0 then computes while the host
enqueues card 1, and card 1's wait was already released by the push.

Chunk size: 12288 (vLLM `max_num_batched_tokens`), the C13/X3 point where
dequant+bmm wins and the GEMM-only budget is 3255 tok/s. The exchange is
sized by `B70_EP_PREFILL_MAX_T` (12288: inbox 63 MB, psum 126 MB).

Not built in this pass: HALVES-style overlap of card 1's experts under
card 0's next-layer attention (Strata `prefill.cpp:5216`), and the QSA
indexer prefill (`B70_QSA=1` + X3's `B70_PREFILL_QSA` helpers) -- the
dense FA path is measured first because it is what O4's decode recipe
serves.

## 3. Recipe

`harness/f1_window.sh` (health -> serve -> `harness/f1_probe.py` -> stop
-> health). Serve = O4 f4 env (ring + mlock PLE, PLE_FAST, DECODE_HC
triton, LEAN, shared-expert fused/overlap, MoE `o4/_C.o4pin-o3g.so` sha
38f7176a) plus:

```
B70_EP_SO=/mnt/vm_8tb/b70/build/vllm-flashnext/f1/b70_ep_C.f1.so   (sha 3d7b231c)
B70_EP_PREFILL=1  B70_EP_PREFILL_MAX_T=12288  B70_EP_PREFILL_C1_CHUNK=2048
B70_PREFILL=1  B70_PREFILL_WIDE_SO=.../_C.x3wide.so  B70_PREFILL_DQ_T=8192
B70_PREFILL_REDUCE=grouped  B70_PREFILL_BUCKET=1  B70_PREFILL_EXPERT_TILE=64
B70_PREFILL_DQ_OVERLAP=0  B70_PREFILL_PLE_HASH=numpy  B70_PREFILL_HC=triton
vllm serve ... --max-model-len 147456 --max-num-seqs 2 --max-num-batched-tokens 12288
  --no-enable-prefix-caching --mamba-cache-mode align --gdn-prefill-backend sycl
  --attention-backend FLASH_ATTN  FULL_DECODE_ONLY [1,2,3,4,8]
```

Build: `cpu-run /mnt/vm_8tb/b70/build/vllm-flashnext/f1/build_ep.sh`
(89 s, MAX_JOBS=1). The package `_C.so` is not replaced.

## 4. Results

(filled per window below)
