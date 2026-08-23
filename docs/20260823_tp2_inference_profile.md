# TP=2 inference profile on 2x Intel Arc Pro B70

Date: 2026-08-23

## Answer

TP=2 communication boundaries are the largest sglang W8A8 decode bottleneck,
but raw peer bandwidth is not the main decode problem. Decode messages are
small and frequent. Their fixed synchronization, queue, and host/runtime cost
dominates their byte-transfer cost.

Large prefill is different: its collectives are bandwidth-bound. The existing
Level Zero IPC push transport is about 9x faster than oneCCL at 16-64 MiB and
has already delivered the large-prefill win.

The current llama.cpp Q4_K_M daily driver must not be conflated with the
sglang result. It already uses a single-kernel, fused direct-Q8 collective and
is primarily a weight-streaming/GEMM lane. Its safe TP=2 profiler is not
available on this build, so this campaign measured its served baseline and
used the source-level collective census rather than enabling the known-unsafe
TP=2 SYCL profiler.

## Scope and identities

- Hardware: 2x Intel Arc Pro B70, 32 GiB each, kernel 7.1.0-070100, Intel
  Compute Runtime 26.22.38646.4. The cards sit below separate AMD root
  complexes (`0000:00` and `0000:40`). Each exposes a 32 GiB BAR2 aperture.
- Production baseline: `hotschmoe-dd`, stock Qwen3.8-27B Q4_K_M, llama.cpp
  SYCL TP=2, F16 KV, 262144 context, MTP off, lab doors off.
- Detailed trace target: `qwen36-27b-w8a8-mtp`, compressed-tensors GPTQ W8A8,
  sglang TP=2, NEXTN MTP10, eager decode, 8192 context.
- Stable transport policy: `CCL_TOPO_P2P_ACCESS=0`. No unsafe P2P-on serve or
  cross-device output-write experiment was run.

## Commands and artifacts

All GPU commands ran inside one both-card `gpu-run` lease.

```bash
docker run --rm --device /dev/dri --ipc=host --shm-size 16g \
  -v "$PWD/scripts/allreduce_bench.py:/allreduce_bench.py:ro" \
  -e CCL_TOPO_P2P_ACCESS=0 --entrypoint bash sglang-xpu:mtp \
  -lc 'source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1; python /allreduce_bench.py'
IMG=vllm-xpu-env:int8g-v0251 bash scripts/106_run_ar_torch.sh
bash sglang/profile_w8a8_0515_vs_0506.sh
PUSH_AR_MIN_NUMEL=1048576 bash rdy_to_serve/sglang/qwen36-27b-w8a8/serve.sh run
PUSH_AR_MIN_NUMEL=0 bash rdy_to_serve/sglang/qwen36-27b-w8a8/serve.sh run
```

The historical `scripts/60_allreduce_bench.sh` default image and host mount
were stale. The successful run used its current Python payload,
`scripts/allreduce_bench.py`, mounted into `sglang-xpu:mtp` with only
`CCL_TOPO_P2P_ACCESS=0` added.

Primary artifacts:

- `results/logs/20260823T213306Z_tp2_oneccl_allreduce_profile.log`
- `results/logs/20260823T213459Z_tp2_push_ar_profile.log`
- `results/logs/20260823T213607Z_sglang_w8a8_stage_profile.log`
- `results/logs/20260823T215451Z_sglang_w8a8_decode_push_gate1m.log`
- `results/logs/20260823T220620Z_sglang_w8a8_decode_push_all.log`
- `results/logs/profile_sglang_w8a8_pushall_20260823_221727_parsed.log`
- `/mnt/vm_8tb/b70/sgl_cache/profile_sglang_w8a8_ab_20260823_213607`
- `/mnt/vm_8tb/b70/sgl_cache/profile_sglang_w8a8_pushall_20260823_221727`

## Transport microbench

oneCCL used torch 2.12.0+xpu from the current sglang image. Push used the
existing two-process Level Zero IPC implementation in torch's L0 context.

| size | oneCCL latency / bandwidth | push latency / bandwidth | reading |
| --- | ---: | ---: | --- |
| 4 KiB | 81 us | not run | fixed-latency regime |
| 8 KiB | 78 us | not run | fixed-latency regime |
| 10 KiB | about 80 us by neighboring sizes | 40.0 us | push is about 2x lower latency |
| 64 KiB | 111 us / 0.59 GB/s | 40.6 us / 1.62 GB/s | latency still dominates |
| 1 MiB | 853 us / 1.23 GB/s | 124.8 us / 8.40 GB/s | transition to bandwidth regime |
| 16 MiB | 14.54 ms / 1.15 GB/s | 1.57 ms / 10.66 GB/s | push is 9.3x faster |
| 64 MiB | 58.16 ms / 1.15 GB/s | 6.30 ms / 10.66 GB/s | push is 9.3x faster |

Verdict: the safe push path fixes large-message transport and halves the
isolated tiny-message latency. It does not remove the per-boundary host and
framework synchronization in eager serving.

## Real sglang W8A8 stage profile

The default 1,048,576-element gate keeps decode on oneCCL and sends large
prefill to push. Each decode trace contains five batches.

### Decode with oneCCL

| device bucket | rank 0 | rank 1 | share rank 0 / rank 1 |
| --- | ---: | ---: | ---: |
| all-reduce, 795 calls | 319.4 ms | 878.4 ms | 41.3% / 65.9% |
| BF16 GEMM | 218.4 ms | 218.4 ms | 28.2% / 16.4% |
| INT8 GEMM | 96.6 ms | 96.8 ms | 12.5% / 7.3% |
| activation quant | 31.9 ms | 31.9 ms | 4.1% / 2.4% |
| copy/reshape | 32.8 ms | 32.4 ms | 4.2% / 2.4% |
| GDN recurrent | 8.0 ms | 7.9 ms | 1.0% / 0.6% |
| total device kernels | 773.7 ms | 1332.6 ms | 100% / 100% |

There are 795 all-reduces over five decode batches, or 159 per batch. Their
traced device time averages 402 us/call on rank 0 and 1105 us/call on rank 1.
The payload-identical ranks have essentially identical math time but a 2.75x
collective-time imbalance. The imbalance reproduces on sglang 0.5.15, where
the same 795 calls take 553.7/892.2 ms. It is not specific to the 0.5.6 API
route.

### Prefill

At approximately 2K input tokens, the default push gate is already active.
The 0.5.6 rank traces spend about 293 ms in 528 push all-reduce kernels. INT8
GEMM is 164-170 ms, BF16 GEMM 118-132 ms, copy/reshape 156-161 ms, GDN
recurrent 74-77 ms, and activation quant 34-35 ms. Transport, math, recurrent
work, and tensor materialization all matter. The prior controlled length
sweep remains the right result for prefill: push gives 2.1-3.1x cold-prefill
gains from 512 through 32K tokens.

## Decode push A/B

Only `PUSH_AR_MIN_NUMEL` changed. Both arms used the same shelf wrapper,
model, MTP depth, eager mode, prefill push path, and coherence/soak gates.

| measure | 1M production gate | push every all-reduce | delta |
| --- | ---: | ---: | ---: |
| c1 decode | 21.36 tok/s | 22.48 tok/s | +5.2% |
| c4 aggregate output | 19.76 tok/s | 19.81 tok/s | +0.3% |
| c1 TTFT | 583.98 ms | 594.73 ms | 1.8% slower |
| 2K-token soak | 16.36 tok/s | 17.35 tok/s | +6.1% |
| coherence | pass | pass | unchanged |
| first/last soak window | 1.01x | 1.01x | stable |

This is a positive research result, not a shelf promotion. It is one ordered
A/B and has not passed the required full serve-sweep gate.

### Mechanism trace

The push-all trace proves that the throughput signal is real transport
engagement:

| device measure, five decode batches | oneCCL rank 0 / 1 | push rank 0 / 1 |
| --- | ---: | ---: |
| collective kernel time | 319.4 / 878.4 ms | 8.0 / 8.0 ms |
| total device-kernel time | 773.7 / 1332.6 ms | 469.2 / 468.8 ms |
| BF16 GEMM | 218.4 / 218.4 ms | 218.2 / 218.3 ms |
| INT8 GEMM | 96.6 / 96.8 ms | 96.5 / 96.7 ms |
| XPU runtime trace duration | 598 / 608 ms | 668 / 663 ms |

Push removes 97.5-99.1% of the collective device time and removes the rank
imbalance, but end-to-end decode rises only 5-6%. The push path launches 1,590
small `do_ar` kernels over 795 logical boundaries and uses an eager host
barrier. Device work is no longer the limiting part of those boundaries; the
serial dependency, Python/ctypes call path, queue waits, and XPU runtime are.

## Production llama.cpp lane

The restored stock Q4_K_M server measured 37.88 tok/s on a fresh 128-token
request, with 163 prompt tok/s for the short 33-token prompt. The model file
is 18.96 GB, or about 9.48 GB/card under equal TP. A one-pass ideal weight
traffic model therefore implies about 359 GB/s/card at the measured decode
rate, about 59% of the nominal 608 GB/s/card roofline before accounting for
non-weight traffic.

The source-level Qwen3.8 TP=2 census is 128 already-fused 20 KiB hidden-vector
boundaries per target token, about 2.5 MiB/token. The current server enables
single-kernel communication plus fused all-reduce/add/RMS/multiply and direct
Q8 handoff. This makes its communication payload small and its main remaining
target weight bandwidth, MMVQ efficiency, and launch fusion. Do not apply the
sglang oneCCL percentages directly to llama.cpp.

## Engineering order

1. Repeat and gate eager push-all.
   Run a position-balanced A-B-B-A plus `bin/serve-sweep --smoke` and the
   mixed-load coherence gate. If the 5-6% c1/soak gain survives with c4 and
   TTFT non-regressing, lower the shelf gate. Do not promote from this one A/B.
2. Remove host synchronization from the eager push path.
   The device collective is now about 10 us/logical boundary in the trace,
   but the wall gain is small. Replace the per-call host shared-memory barrier
   and ctypes/queue waits with a correct command-streamer event path, or batch
   multiple boundaries under one device-driven submission. This is the
   highest-value TP=2 communication project.
3. Reduce boundary count structurally.
   Evaluate PP=2 for concurrency-oriented serving and prototype fused
   GEMM-output push/reduce/residual-norm boundaries. TP has a hard dependency
   after every row-parallel projection; making a faster copy does not remove
   those 128-159 synchronization points.
4. Optimize the post-push math stack.
   After push-all, BF16 GEMM is 46.5% of device time, INT8 GEMM 20.6%,
   activation quant 6.8%, and copy/reshape about 8%. The BF16 MTP/draft and
   remaining non-INT8 linears are the next concrete kernel census target.
5. Keep raw P2P toggles and broad peer writes off the critical path.
   `CCL_TOPO_P2P_ACCESS=1` in TP>1 serve remains guarded, and Steve's broader
   peer-output write prototype caused device-lost/reset storms. The safe
   Level Zero IPC push path already demonstrates the available approximately
   11 GB/s direction. More raw link bandwidth cannot remove serial boundary
   latency.

## Final verdict

- sglang W8A8 decode: communication synchronization is bottleneck number one.
- sglang W8A8 prefill: collective bandwidth is bottleneck number one and the
  push transport is the proven fix.
- After push-all decode: BF16/INT8 GEMM is the device bottleneck, while host
  runtime synchronization still limits end-to-end realization of the device
  savings.
- llama.cpp Q4_K_M production: weight bandwidth/MMVQ and launch fusion deserve
  more effort than P2P transport.
- Best next project: eliminate or amortize the 128-159 decode synchronization
  boundaries; do not chase unsafe P2P-on transport knobs.
