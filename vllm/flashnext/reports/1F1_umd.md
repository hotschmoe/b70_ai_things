# F1 container-scoped UMD A/B

Agent F1. Card 0 only. Card 1 stayed with E1 (`vllm-fn-lowbit-e1`,
port 18084). Kernel stayed `7.1.0-070100-generic`. No apt, no dpkg,
no host library replacement.

Stack under test: image `vllm-flashnext-xpu:py312`, venv
`/mnt/vm_8tb/b70/build/vllm-flashnext/venv`. Ornith-1.5-35B-A3B GPTQ
int4 mixedcal-v2, TP=1, dtype float16, language-model-only,
max-model-len 33792, gpu-memory-utilization 0.90, max-num-seqs 4,
`B70_XPU_CG_RECLAIM=500`, async scheduling on, port 18091. Served
name `hotschmoe-dev`, alias
`ornith-1.5-35b-a3b-GPTQ-int4-mixedcal-v2`. Harness
`bench_latency.py`, decode 64, prefill 1024/8192/32768. JSON and
logs: `/mnt/vm_8tb/b70/build/vllm-flashnext/logs/f1-*`.

Claude consult at the close: `claude --bare -p --model fable-5.1` and
`--model claude-5.5` both returned `unrecognized_model` and
"Not logged in". Same result D1 recorded. No second-model review.

## Recommendation

Keep the host userspace as the campaign runtime: compute-runtime
26.22.38646, IGC 2.36.3, Level Zero loader 1.28.2.

On Ornith GPTQ FULL_DECODE_ONLY FlashAttention, C1 ITL p50 is 12.80 ms
(host), 12.77 ms (26.27), and 12.60 ms (26.35). C2 ITL p50 is about
13.73, 13.75, and 13.58 ms. TTFT is 84.7, 89.7, and 92.1 ms. Prefill
at 1k/8k/32k stays inside about 3 percent. The 0.20 ms decode gain on
26.35 does not offset the TTFT regression, and it does not justify
leaving the qualified host stack. 26.35 did serve, stay coherent, and
leave card 0 healthy, so `xpu-run.sh --umd` can be used when a later
kernel needs that compiler.

Triton-attention decode is the same shape (C1 ITL p50 13.13 / 13.18 /
12.97 ms). Triton 32k prefill was 1224 / 1868 / 1705 tok/s. The host
Triton cache was already warm (server ready in 106 s). The 26.27 and
26.35 arms cold-compiled into fresh per-UMD caches. Do not treat that
prefill gap as a clean IGC result.

The AOT `b70_gguf_xpu` `_C.so` (6391032 bytes, 2026-10-20 04:32) was
not rebuilt. Dense GEMV at M=1 matches across the three runtimes
(Q4_K 129-132 us, Q5_K 810-836 us). Fused moe_layer at T=8 is
1409 / 1425 / 1441 us on the behind timer. Swapping the container
UMD does not retune that binary. C5's compile-time IGC A/B is the
lever for the plugin.

## Identity

Recorded inside the container with `sycl-ls` and `stat` of the
mounted files. Device query was a 4x4 torch.xpu matmul, sum 64, on
one visible device (`ZE_AFFINITY_MASK=0`).

| Arm | NEO / L0 (sycl-ls) | IGC | Loader | Clang soname |
| --- | --- | --- | --- | --- |
| host | OpenCL NEO 26.22.38646.4, L0 1.15.38646+4 | 2.36.3, 95845480 bytes | 1.28.2, 1773224 bytes | libopencl-clang2.so.17 |
| 26.27.39122.11 | OpenCL NEO 26.27.39122.11, L0 1.15.39122+11 | 2.38.2, 96015048 bytes | 1.32.0, 1036264 bytes | libopencl-clang2.so.17 |
| 26.35.39758.10 | OpenCL NEO 26.35.39758.10, L0 1.17.39758+10 | 2.41.5, 104543000 bytes | 1.32.0, 1036264 bytes | libopencl-clang2.so.22.1 |

26.35.39758.10 is the newest intel/compute-runtime GitHub release
(2026-09-17). It was built with IGC 2.41.5 and Level Zero v1.32.0.
Loader 1.34.0 is newer and is not the matching loader, so both
non-host trees use 1.32.0. gmmlib 22.10.0 is the same bytes as the
host `libigdgmm.so.12.10.0`. Intel's 26.35 note is Ubuntu 26.04
kernel 7.0. This host is kernel 7.1. The trial was userspace-only.

GPU sonames: host `libze_intel_gpu.so.1.15.38646` (62177352 bytes),
26.27 `libze_intel_gpu.so.1.15.39122`, 26.35
`libze_intel_gpu.so.1.17.39758` (63522968 bytes). `ldd` of loader,
NEO, libigc, libigdfcl, and libiga64 had no "not found". Process maps
for the vLLM engine and API server resolved `libigc.so.2`,
`libigdfcl`, `libiga64`, `libopencl-clang`, `libze_intel_gpu`, and
`libze_loader` under `/host-umd` on every serve, including the Triton
arms. No oneAPI RPATH copy of libigc showed up in those maps.

Trees (extracted, not installed):
`/mnt/vm_8tb/b70/build/vllm-flashnext/umd/26.27.39122.11/` and
`umd/26.35.39758.10/`.

## Harness opt-in

D2 owns `harness/`. The only edit is additive:

- `vllm/flashnext/harness/xpu-run.sh`: `--umd DIR` or `XPU_RUN_UMD`.
  Unset keeps the host mounts. The host `--check` still requires
  `libze_loader.so.1.28.2` and still rejects a resolved 1.32 or 26.27
  path.
- `vllm/flashnext/harness/xpu-entrypoint.sh`: `check_loader_override`
  when `XPU_UMD_MODE=override`. It checks ldd, mapped paths and sizes,
  and, when `/dev/dri` is present, `sycl-ls` plus a torch.xpu matmul.

Override Triton and SYCL caches are
`cache/triton-<label>` and `cache/sycl-<label>`. The host arm kept
the shared default caches. `--check` (no GPU) exited 0 for host,
26.27, and 26.35. Logs: `f1-check-host.log`, `f1-check-2627.log`,
`f1-check-2635.log`.

## Method

`bin/xpu-health --card 0` before the first lease: HEALTHY
(img `vllm-xpu-env:int8g-v0251`). Each arm then ran:

1. Device probe (`f1_probe_inner.sh`).
2. `serve_ornith_gptq.sh` with `GRAPH=1 GRAPH_MODE=FULL_DECODE_ONLY
   ATTN=FLASH_ATTN` (SYCL GDN). Then `bench_latency.py`.
3. Same graph with `ATTN=TRITON_ATTN GDN_BACKEND=triton`. Then the
   same bench.
4. `plugins/b70_gguf_xpu/tests/gpu_check.py` and
   `tests/moe_layer_bench.py` via `xpu-run` (not `scripts/c1_run.sh`,
   which would prepend another `LD_LIBRARY_PATH` and pin card 0 from
   inside).

Health after each stage. The one skip is below. No `xe-reset`.
Driver: `logs/f1_umd_driver.sh`. Final card 0 health after the 26.35
plugin retry: HEALTHY.

## FlashAttention FULL_DECODE_ONLY

Identity ok on every bench (`hotschmoe-dev` first, alias present).
Coherence ok: both greedy strings
` Paris. The capital of France is Paris.` and `same=true`.

Ready time is from container start to `/v1/models`.

| Arm | Ready | C1 ITL p50/p99 ms | C1 tok/s | C1 TTFT ms | C2 ITL p50 ms | Prefill 1k/8k/32k tok/s |
| --- | ---: | --- | ---: | ---: | --- | --- |
| host 26.22 | 107 s | 12.80 / 13.13 | 78.27 | 84.68 | 13.73 / 13.74 | 5956 / 7581 / 6182 |
| 26.27 / IGC 2.38.2 | 135 s | 12.77 / 13.25 | 78.40 | 89.67 | 13.76 / 13.74 | 5800 / 7574 / 6171 |
| 26.35 / IGC 2.41.5 | 136 s | 12.60 / 13.00 | 79.36 | 92.07 | 13.58 / 13.58 | 5845 / 7622 / 6193 |

C2 columns are stream 0 / stream 1. C2 TTFT follows the same order
as C1 (host 167/173 ms, 26.27 176/183 ms, 26.35 181/188 ms).

D1's same-day host FlashAttention row was C1 ITL p50/p99 12.81/13.12
ms, 78.23 tok/s, TTFT 88.59 ms, prefill 6030/7888/6414. This host arm
reproduces that decode number.

JSON: `f1-bench-host-fa.json`, `f1-bench-2627-fa.json`,
`f1-bench-2635-fa.json`.

## Triton attention plus Triton GDN

Same graph mode. Fresh IGC for 26.27 and 26.35 (cache suffix). Host
used the shared Triton cache and was ready in 106 s. 26.35 was ready
in 467 s. 26.27 spent about 4 minutes queued behind C5
(`run_measure.sh` on card 0) and was ready 693 s after the serve
process started, about 450 s after the container was running.

| Arm | C1 ITL p50/p99 ms | C1 tok/s | C1 TTFT ms | C2 ITL p50 ms | Prefill 1k/8k/32k tok/s |
| --- | --- | ---: | ---: | --- | --- |
| host 26.22 | 13.13 / 13.58 | 76.34 | 140.79 | 14.24 / 14.26 | 3706 / 3043 / 1224 |
| 26.27 / IGC 2.38.2 | 13.18 / 13.64 | 75.96 | 157.48 | 14.36 / 14.37 | 4163 / 3900 / 1868 |
| 26.35 / IGC 2.41.5 | 12.97 / 13.21 | 77.29 | 154.11 | 14.33 / 14.34 | 4424 / 4066 / 1705 |

Decode moved by about 0.2 ms. The 32k prefill gap is real in these
logs and is confounded by the warm host cache. D1's host Triton C1
ITL p50 was 13.20 ms, which this host arm matches.

JSON: `f1-bench-host-triton.json`, `f1-bench-2627-triton.json`,
`f1-bench-2635-triton.json`.

## b70_gguf_xpu microbench (for C5)

Same AOT `_C.so` on every arm. Not relinked. Correctness failures
were 0 on the runs below. Shape notes: dense Q4_K is blk.11 attn_q,
N=12288 K=2560. Q5_K is the output head, N=248320 K=2560. moe_layer
is E=32 topk=10, I=640 K=2560, gate IQ2_XS / up IQ2_XXS / down IQ4_NL.
`behind_us` is the test's queued timer.

| Arm | Q4_K M=1 us | Q5_K M=1 us | moe_layer T=1 behind us | moe_layer T=8 behind us |
| --- | ---: | ---: | ---: | ---: |
| host 26.22 | 129.4 | 810.6 | 370.0 | 1408.5 |
| 26.27 / IGC 2.38.2 | 132.3 | 809.5 | 589.4 | 1424.8 |
| 26.35 / IGC 2.41.5 | 128.9 | 835.7 | 254.4 | 1441.3 |

M=1 GEMV and T=8 moe_layer agree. T=1 moe_layer does not (370 / 589 /
254 us) and the a2 timer on that point is just as scattered, so it is
not a UMD ranking. Q5_K M=8 was 4309 / 4310 / 3857 us, one sample.

The first 26.35 plugin run failed before any bench line:
`operator b70_gguf::reserve_decode does not exist`. That was
`ops.py` registering a fake for a symbol that exists only in
`_C.graph.so`. The default `_C.so` does not export it. A later edit
of `ops.py` guards that fake with `hasattr`. The retry
(`f1-gguf-2635-retry.log`, 114 s, exit 0) is the 26.35 row above.
Card 0 was HEALTHY before and after the retry.

Logs: `f1-gguf-host.log`, `f1-gguf-2627.log`, `f1-gguf-2635.log`
(failed attempt), `f1-gguf-2635-retry.log`.

## Health and lease

Card 0 HEALTHY at 06:16 UTC before the campaign, after each completed
stage, and at 07:08 UTC after the plugin retry. After the 26.27
FlashAttention server stopped, C5 took card 0 (`run_measure.sh`).
Health was skipped for that one gap. The 26.27 Triton serve queued,
then ran, and the following health checks passed. No wedge, no
`xe-reset`. No F1 container left running.
