# vllm_push_allreduce -- hand-rolled PUSH all-reduce for dual B70 TP=2

Replaces vLLM-XPU's oneCCL all-reduce with the hand-rolled posted-write (PUSH) collective proven in
`docs/P2P_GPU.md` J.8-J.12. On our cross-die PCIe-Gen3 dual-B70 box it BEATS oneCCL on BOTH:

- decode latency: ~34-45 us vs oneCCL ~85 us
- prefill bandwidth: ~10 GB/s (bf16) vs oneCCL ~9.4 GB/s

and it does its own Level-Zero IPC P2P, **independent of `CCL_TOPO_P2P_ACCESS`**, so the serve runs
`P2PACCESS=0` (oneCCL's host-staged warmup all_reduce succeeds -> NO H.13 `DEVICE_LOST` wedge) while the
model's allreduces go over the 11 GB/s posted-write fabric.

## Pieces

- `scripts/106_xpu_push_ar_torch.cpp` -> `libxpu_push_ar_torch.so` (build with `icpx -fsycl ... -lze_loader -lrt`).
  Runs IN torch's L0 context (grabs `torch.xpu.current_stream().sycl_queue`), so it operates directly on
  vLLM tensor `data_ptr()`s. C-ABI: `ar_setup_torch / ar_exchange / ar_allreduce_ptr_dt / ar_teardown`.
- `_push_ar_patch.py` -- monkeypatches `XpuCommunicator.all_reduce`. Engages only for world_size==2,
  contiguous bf16/fp16/fp32, size <= `PUSH_AR_MAXB`; otherwise falls back to the original oneCCL path.
- `usercustomize.py` -- import shim so the patch coexists with the rdy_to_serve MTP `sitecustomize.py`
  (Python loads `usercustomize` in addition to the first `sitecustomize` on the path).
- `sitecustomize.py` -- same, for standalone use (no other sitecustomize on the path).

## Activate

```
PYTHONPATH=<mtp_shim_dir>:<this_dir>   # this_dir must be reachable as a top-level import dir
PUSH_AR_SO=/path/to/libxpu_push_ar_torch.so
```
Env: `PUSH_AR_DISABLE=1` (kill switch), `PUSH_AR_MAXB` (scratch bytes, default 128 MiB),
`PUSH_AR_SOCKDIR` (IPC socket dir, default /tmp).

Driver script: `scripts/108_serve_push_ar_ab.sh` (27B-W8A8 TP=2 A/B vs oneCCL).

## CAVEATS

- **Graph capture**: the op uses a host barrier + ctypes call -> NOT SYCL-graph-capturable. Run the serve
  EAGER (`GRAPH=0`), or mark the TP all_reduce as a graph splitting op. With `GRAPH=1` the capture will
  break exactly like the original oneCCL `sched` allreduce did.
- **Only all_reduce** is accelerated. reduce_scatter / all_gather (used by MoE / MTP-spec) fall back to
  oneCCL. Dense models (e.g. 27B-W8A8) are pure-allreduce in TP and benefit fully.
- 2-rank pairwise algorithm only (world_size==2). world>2 falls back.
