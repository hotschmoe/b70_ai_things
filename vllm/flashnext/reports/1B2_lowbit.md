# 1B2 -- low-bit WNA16 XPU kernels

Plugin: `vllm/flashnext/plugins/b70_lowbit_xpu`. Ops: `torch.ops.b70_lowbit.*`.
Quant method name: `b70_lowbit`, entry point group `vllm.general_plugins`,
key `b70_lowbit_xpu` = `b70_lowbit_xpu:register`. No vLLM core edit.

Shipped binary (the numbers below, unless a row says otherwise):

```
/mnt/vm_8tb/b70/build/vllm-flashnext/b70_lowbit_xpu/libb70_lowbit_xpu.so
773928 bytes, mtime 2026-10-20 01:31:36 UTC
```

AOT for `intel_gpu_bmg_g31`. The `.so` is not in git. Previous SLM-only
binary, kept for the comparison in "Earlier binaries":

```
.../libb70_lowbit_xpu_slm.so   692008 bytes, 2026-10-20 01:21:38 UTC
```

## CONFIG

- Card 0 only. Launcher is `vllm/flashnext/harness/xpu-run.sh --card 0`.
  That script takes `bin/gpu-run` for card 0 and sets `ZE_AFFINITY_MASK`.
  A second `bin/gpu-run` around it deadlocks. Host Python with
  `LD_LIBRARY_PATH=torch/lib` saw zero XPU devices; the container path is
  the one that ran the kernels. Image `vllm-flashnext-xpu:py312`.
  `--memory 32g`, container name `b70-lowbit-bench`.
- Card 1 was left alone (B1). No `xe-reset`. `bin/xpu-health --card 0`
  before and after the official suite: HEALTHY, exit 0.
- Torch 2.14.0+xpu, vLLM `0.32.1rc1.dev94+g5fd62688f.xpu` at
  `5fd62688ffe995c0c11c8708fe51dbff53235b74`, tree
  `/mnt/vm_8tb/b70/build/vllm-flashnext/vllm`. `vllm_xpu_kernels` 0.1.15.4
  supplies the int4 baseline op.
- Compiler: `icpx` 2026.1.0 inside
  `b70/flashnext-llamacpp-build:oneapi2026.1-de7fa0a`, wrapped in
  `bin/cpu-run` (not `--quiet`). Flags that actually compiled:
  `-std=c++20`, `-fsycl-targets=intel_gpu_bmg_g31`,
  `-fsycl-device-code-split=per_kernel`. PATH must include
  `/opt/intel/oneapi/compiler/2026.1/bin/compiler` (`llvm-foreach`).
  Host `ocloc` 26.22.38646.4 and `libocloc.so` / `libocloc.so.2` are
  bind-mounted; the image's ocloc is the VTune copy and does not compile
  this target. `setvars` is sourced only after `set +e; set +u`.
- Build script: `plugins/b70_lowbit_xpu/build.sh` (mode 664; launch with
  `bash`, because `cpu-run` execs the file).
- Ops, all XPU, bf16 activations, symmetric weights, group 128:
  - `gemv(x, plane2, plane_hi, scale, bits)` dense, M=1..8
  - `moe_gemv(...)` top-k routed, fused gate/up SiLU-mul + down, router
    weighted reduce
  - `prefill_wna16(...)` unpack to bf16 tiles, XMX `dpas<8,8,float>`
  - `prefill_wxa8(...)` per-group int8 activations, `dpas<8,8,int32_t>`,
    prefill only
- Decode kernel (`csrc/lowbit_kernels.cpp`): ESIMD, work-group 16, SLM
  holds the activation (cap M<=8 and K<=2560 for dense; one row K<=2560
  for MoE). Even groups share one 64-byte plane2 load (`accum_pair`,
  line 176). Odd tail (K=640 has 5 groups) uses a 32-byte load. Output
  is one row per work-item. No `queue.wait()` inside the op.
- Prefill kernel: one work-item per (M-tile of 8, N-tile of 16), local
  size 1. Each tile unpacks its own B panel with scalar byte loads and
  scatters into VNNI. Weights are re-read `ceil(M/8)` times. The int8
  VNNI-4 index is `b[(kk/4)*64 + c*4 + (kk%4)]`. Numerics match the
  reference, so that index is the one this GPU executes.
- Roofline used for GB/s: 608 GB/s. Numerator for `weight+scale` is the
  tight packed payload (not the 64-byte row pad) plus bf16 scales.
  `all` adds the unique activation and the output. Prefill multiplies
  weight bytes by `ceil(M/8)`. MoE counts the top-k experts, not the
  resident table. Timing is `torch.xpu.Event` elapsed_time over the
  loop, divided by iters (`tests/bench_lowbit.py` `_elapsed_ms`). The
  host-clock fallback did not print.
- Shapes. Ornith / Qwen3.6-35B-A3B: H=2048, I=512, E=256, top-8, 40
  layers, vocab 248320. Flash-Next: H=2560, I=640, E=512, top-10, 48
  layers. GEMV cases are M,N,K:
  square (1,2048,2048) and (1,2560,2560); gate/up (1,1024,2048) and
  (1,1280,2560); down (1,2048,512) and (1,2560,640); M=8 gate/up;
  lm_head (1,248320,2048) int3 only inside the suite.

### Two layouts

The kernel reads **b70lb1**, defined in `b70_lowbit_xpu/format.py`:

- Codes are signed. int3 is [-4, 3], int2 is [-2, 1], int4 is [-8, 7].
  Scale is bf16 `[N, K/128]`, `amax/maxq`.
- `plane2` is the low 2 bits, 4 codes per byte, 32 bytes per group of
  128. `plane_hi` is the extra bits (int3: 16 B/group, int4: 32 B/group).
  int2 has no hi plane.
- Each row is padded so the next row starts on 64 bytes
  (`plane2_stride`, `plane_hi_stride`). Low K sits in the low bits of
  a byte.

`plugins/LOWBIT_FORMAT.md` is B1's contract, written after the kernel
layout existed. It specifies `auto_round:auto_gptq`:

- `qweight` int32 `[K * bits / 32, N]`, little-endian along K.
- `scales` fp16 `[K/128, N]`.
- `qzeros` int32, constant lane `maxq - 1` (7, 3, or 1). Signed recovery
  is `unsigned - maxq`, not `unsigned - qzeros`.
- Checkpoint names `{prefix}.qweight`, `.qzeros`, `.scales`, `.g_idx`.
  `lm_head` stays bf16. Router, norms, conv, embeddings, MTP stay bf16.

`format.py` `gptq_to_b70lb1` (line 185) implements that recovery and
repacks into b70lb1. A CPU roundtrip of a hand-rolled AutoGPTQ packer
(bits 2/3/4, K in {128, 512, 640, 2048}, N=13, including the int3
straddle at index 10) printed `VERDICT gptq_to_b70lb1 ok`. That script
lives outside the repo at
`/mnt/vm_8tb/b70/build/vllm-flashnext/b70_lowbit_xpu/test_gptq_repack.py`.
It was not compared to a real `auto_round.pack_bitstream` blob or to
the on-disk Ornith expert B1 used.

`linear.py` `create_weights` registers `weight`, `weight_plane_hi`,
`weight_scale`. `moe.py` registers `w13_weight` / `w2_weight` and the
matching `_plane_hi` / `_weight_scale`. `process_weights_after_loading`
only calls `contiguous()`. It does not call `gptq_to_b70lb1`.
`B70LowbitConfig` rejects any `pack` other than `b70lb1`
(`quant_config.py` line 65). A B1 checkpoint (`quant_method` auto-round,
GPTQ tensor names) does not load through this method today.

## COMMAND

```
B70_AGENT=B2 B70_CPU_LOCK_TIMEOUT=0 \
  bin/cpu-run bash vllm/flashnext/plugins/b70_lowbit_xpu/build.sh
# log: /mnt/vm_8tb/b70/build/vllm-flashnext/b70_lowbit_xpu/build_pair.log
# compile rc=0, "=== BUILD OK ==="

echo '=== health before ==='
bin/xpu-health --card 0

B70_AGENT=B2 B70_CPU_LOCK_TIMEOUT=0 B70_GPU_LOCK_TIMEOUT=0 \
  bin/cpu-run --quiet \
  vllm/flashnext/harness/xpu-run.sh --card 0 --memory 32g \
    --name b70-lowbit-bench -- \
  bash vllm/flashnext/plugins/b70_lowbit_xpu/tests/run_xpu.sh
# log: .../b70_lowbit_xpu/bench_pair.log

bin/xpu-health --card 0
```

`tests/run_xpu.sh` sets `PYTHONPATH` to the plugin source (ahead of the
stale site-packages copy), `B70_LOWBIT_SO` to the `.so` above, and
`B70_LOWBIT_PORT=29671`, then runs `python3 -u tests/bench_lowbit.py`.
Pass rule: rel L2 < 0.02 and cosine > 0.999 against an fp32 reference
of the dequantized weights. WXA8's reference quantizes activations the
same way the kernel does (round, clamp -127..127, scale amax/127 per
group of 128).

Isolated lm_head probe, same binary, bits 4 then 3 then 2, 2 warm / 6
iters, plus 9 exact GEMV checks (K=640 tail, M=8):

```
.../b70_lowbit_xpu/lmhead_probe.py
```

via the same `xpu-run.sh --card 0` wrapper. Health 0/0. That probe is
the bit-width comparison. The suite's single int3 lm_head row is the
matched-run number. They disagree by about 0.45 ms; both are reported.

## RESULT

Official suite, 2026-10-20 01:35:28. `BENCH_RC:0`. `HEALTH_BEFORE:0`.
`HEALTH_AFTER:0`. `gpu-run: [B2] done in 109s (exit 0)`.
`VERDICT all correctness checks passed`.

### Correctness (verified)

Pack roundtrip, bits 2/3/4, K in {128, 512, 640, 2048, 2560}: all PASS.

Dense GEMV, bits 2/3/4, M in {1,2,4,8} at K=128 and the K=640 / 512 /
2048 spots: rel L2 0, cosine 1, max abs 0. Includes the K=640 odd-group
tail and M=8.

MoE GEMV, bits 2/3/4, small shapes (T=1..2, topk=2..4, H=256..512):
rel L2 0, cosine 1.

Prefill WNA16, bits 2/3/4, M up to 32, K up to 512: rel L2 0, cosine 1.
DPAS bf16 on this GPU matches the fp32 reference at these sizes.

Prefill WXA8, same shapes: rel L2 from 0 to 1.707e-3, cosine >= 0.999999,
max abs up to 0.25. That is the int8 activation quant error. The integer
DPAS partial itself is exact inside a group (max product magnitude
65024, inside int32 and exact in float32); the scale is applied every
128 K.

### Plugin registration (verified, same process)

```
PASS get_quantization_config -> b70_lowbit bits=3
PASS bits map and skip list
PASS entry point vllm.general_plugins b70_lowbit_xpu
PASS ReplicatedLinear quant=_B70Linear weight=(64, 64) torch.uint8
PASS linear.apply M=1 rel_l2=0.0000e+00 cos=1.000000 max_abs=0.0000e+00
PASS MoE method decode (2, 256) prefill (9, 256) monolithic=False
```

`weight=(64, 64)` is N=64, K=128: plane2 payload 32 bytes, padded to
the 64-byte row stride. `_B70Linear` subclasses `LinearMethodBase`.
`_B70MoE` subclasses `FusedMoEMethodBase`. `is_monolithic` is false, so
the qwen3_5_moe runner calls `select_experts` then `apply`. `apply`
returns routed states only. Skip list hits a path component: `gate`
matches `mlp.gate` and does not match `gate_proj`. Longest `extra_bits`
key wins. The linear construct sits inside `set_current_vllm_config`
plus a gloo world of size 1; without that, `initialize_model_parallel`
raises. TP is 1. A packed-K row split for TP>1 is wrong
(`linear.py` lines 53-55) and is not implemented.

Import warnings (not failures): `torch._opaque_base` deprecated,
`register_opaque_type` deprecated, `typ='value'` deprecated.

The installed copy under the venv `site-packages` predates the MRO fix
and `gptq_to_b70lb1`. The suite sees source because `PYTHONPATH` puts
the plugin directory first. A vLLM process without that `PYTHONPATH`
still loads the stale module. Reinstall is a follow-up, done when card-1
jobs are not importing this venv.

### Decode latency, official suite

Times are GPU-event milliseconds, mean of 20 launches after 5 warmups
(lm_head: 8 after 3). `w+s` is weight+scale GB/s.

| op | bits | ms | w+s GB/s | % of 608 |
|---|---:|---:|---:|---:|
| ornith square 2048 | 3 | 0.0256 | 64.0 | 10.5 |
| ornith gate/up 1024x2048 | 3 | 0.0245 | 33.4 | 5.5 |
| ornith down 2048x512 | 3 | 0.0244 | 16.8 | 2.8 |
| flashnext square 2560 | 3 | 0.0411 | 62.3 | 10.3 |
| flashnext gate/up 1280x2560 | 3 | 0.0250 | 51.2 | 8.4 |
| flashnext down 2560x640 | 3 | 0.0360 | 17.8 | 2.9 |
| ornith gate/up M=8 | 3 | 0.0261 | 31.4 | 5.2 |
| ornith square | 2 | 0.0403 | 27.6 | 4.5 |
| ornith gate/up | 2 | 0.0308 | 18.1 | 3.0 |
| ornith down | 2 | 0.0295 | 9.4 | 1.6 |
| flashnext square | 2 | 0.0305 | 57.1 | 9.4 |
| flashnext gate/up | 2 | 0.0307 | 28.3 | 4.7 |
| flashnext down | 2 | 0.0404 | 10.8 | 1.8 |
| ornith gate/up M=8 | 2 | 0.0309 | 18.0 | 3.0 |
| ornith square | 4 | 0.0356 | 60.7 | 10.0 |
| ornith gate/up | 4 | 0.0248 | 43.5 | 7.2 |
| ornith down | 4 | 0.0357 | 15.1 | 2.5 |
| flashnext square | 4 | 0.0283 | 119.4 | 19.6 |
| flashnext gate/up | 4 | 0.0257 | 65.7 | 10.8 |
| flashnext down | 4 | 0.0265 | 31.8 | 5.2 |
| ornith gate/up M=8 | 4 | 0.0243 | 44.5 | 7.3 |
| ornith lm_head 248320x2048 | 3 | 2.1263 | 93.4 | 15.4 |

MoE, T=1, full expert table resident, top-k experts touched. Mean of 20
after 5 warmups.

| model | bits | topk | ms | w+s GB/s | % of 608 |
|---|---:|---:|---:|---:|---:|
| ornith E=256 H=2048 I=512 | 3 | 8 | 0.1071 | 91.8 | 15.1 |
| flashnext E=512 H=2560 I=640 | 3 | 10 | 0.1761 | 109.0 | 17.9 |
| ornith | 2 | 8 | 0.1127 | 59.3 | 9.8 |
| flashnext | 2 | 10 | 0.1148 | 113.7 | 18.7 |

Stacking the measured per-layer MoE time, with no attention, shared
expert, router, or lm_head: Ornith int3 is 40 * 0.1071 ms = 4.28 ms of
routed-expert GEMV per token. Flash-Next int3 is 48 * 0.1761 ms = 8.45 ms.
A 150 tok/s ITL budget is 6.67 ms. These sums are op totals, not a
serving measurement. B1 keeps `lm_head` in bf16; the int3 lm_head row
(2.13 ms in this suite) is a bandwidth probe, not the serving path.

### Large-N probe, same binary (verified)

lm_head M=1 N=248320 K=2048, weight+scale only, 6 iters. Nine smaller
GEMVs in the same process were exact (max abs 0), including K=640 and
M=8. Health 0/0.

| bits | ms | w+s GB/s | % of 608 | payload |
|---:|---:|---:|---:|---:|
| 4 | 1.1904 | 220.3 | 36.2 | 262.2 MB |
| 3 | 1.6708 | 118.9 | 19.6 | 198.7 MB |
| 2 | 0.9586 | 140.9 | 23.2 | 135.1 MB |

The suite's int3 lm_head (2.1263 ms, 93.4 GB/s) is the same kernel and
the same shape, timed later in a long run. Both numbers stand. The probe
is the cleaner bit-width comparison because bits 2/3/4 ran back to back.

Inferred from that probe, not from a hardware counter: int2 and int4 fit
`time ~= 0.71 ms + bytes / 549 GB/s`. int3 sits about 0.60 ms above that
line. The extra time lines up with the strided hi-bit extract inside
`dequant_regs` (an 8-step `select` over the 1-bit plane). The 0.71 ms
floor on a 248k-row GEMV is fp32 dequant-dot plus dispatch of about 15k
work-groups, not a failure to issue the load. Expert rows at ~25 us are
a different floor: an Ornith int3 down is 0.41 MB, which is 0.7 us at
608 GB/s, and it measured 24.4 us.

### int4 baseline, same suite (verified)

`torch.ops._xpu_C.int4_gemm_w4a16` after `import vllm_xpu_kernels._xpu_C`.
GPTQ nibble layout, `qzeros` replaced by the scalar int8 tensor `[8]`,
matching the INC symmetric path. Same logical M,N,K. Layouts differ, so
the GB/s columns are not the same byte stream. Latency is comparable.

| shape | our int4 GEMV | xpu int4 GEMM |
|---|---:|---:|
| ornith gate/up | 0.0248 ms | 0.0453 ms |
| ornith down | 0.0357 ms | 0.0440 ms |
| flashnext gate/up | 0.0257 ms | 0.0435 ms |
| flashnext down | 0.0265 ms | 0.0442 ms |

Our GEMV is faster on all four expert shapes. The historical ~552 GB/s
W4A16 figure (report 06, 27B, oneDNN, a large payload) is a different
kernel. This run did not reproduce it. Our best measured int4 point is
the lm_head probe at 220 GB/s.

### Prefill (verified)

WNA16 / WXA8 on gate/up. Weight GB/s counts the reload. Mean of 5 after
2 warmups.

| op | shape | bits | ms | w+s GB/s | note |
|---|---|---:|---:|---:|---|
| wna16 | ornith M=512 | 3 | 2.3532 | 22.3 | weights x64 |
| wxa8 | ornith M=512 | 3 | 2.7052 | 19.4 | |
| wna16 | ornith M=2048 | 3 | 9.4373 | 22.2 | weights x256 |
| wxa8 | ornith M=2048 | 3 | 11.9003 | 17.6 | |
| wna16 | flashnext M=512 | 3 | 5.6464 | 14.5 | weights x64 |
| wxa8 | flashnext M=512 | 3 | 5.1754 | 15.8 | |
| wna16 | ornith M=512 | 2 | 1.5897 | 22.4 | |
| wxa8 | ornith M=512 | 2 | 1.8771 | 19.0 | |
| wna16 | ornith M=2048 | 2 | 6.3714 | 22.4 | |
| wxa8 | ornith M=2048 | 2 | 8.7613 | 16.3 | |
| wna16 | flashnext M=512 | 2 | 4.0552 | 13.7 | |
| wxa8 | flashnext M=512 | 2 | 3.2519 | 17.1 | |

WXA8 is slower than WNA16 on every Ornith gate/up row and slightly
faster on both Flash-Next M=512 rows. Int8 activations do not shrink
the weight read. Report 06's conclusion that W2A8/W3A8 do not help
decode still holds: decode never calls `prefill_wxa8`. Neither prefill
kernel is near the roof. The cause, read from the launch, is scalar
unpack plus a fresh B-panel load for every 8 rows of M.

### Earlier binaries (verified, not the shipped table)

WG=1 kernel (`bench.log`, ~01:10): every numeric check passed. Linear
construct failed because `initialize_model_parallel` ran without
`set_current_vllm_config`. That is fixed. Expert GEMV latencies were
the same band as the table above (Ornith gate/up int3 0.029 ms,
down 0.025 ms). int4 baseline was skipped in that log (op not imported).

SLM + work-group 16, no paired load (`bench_slm.log`, 01:24, the 692 KB
binary): full suite green, including the plugin lines. Expert latencies
did not drop versus WG=1. Isolated lm_head on that binary: int4 1.225 ms
/ 214 GB/s, int3 1.741 ms / 114 GB/s, int2 1.002 ms / 135 GB/s.

Paired 64-byte loads on the shipped binary moved that probe by about
3 to 4 percent (int4 214 to 220, int3 114 to 119, int2 135 to 141).
Wide loads are not the remaining wall. SLM broadcast of a 4 KB
activation is not the wall either: the activation already hits, and the
tax is the fp32 dequant-dot plus, for int3, the bit extract.

## VERDICT

The plugin builds, registers, and is numerically exact for WNA16 GEMV,
MoE GEMV, and prefill DPAS at the shapes above. WXA8 prefill matches
its int8 reference within the quant error. Card 0 was healthy before
and after. Expert-shape int4 GEMV is faster than `int4_gemm_w4a16` on
the four shapes measured.

It does not serve an Ornith / Qwen3.6-35B-A3B checkpoint yet. B1 exports
auto_gptq `qweight` / `scales`. The kernel reads b70lb1 planes named
`weight` / `w13_weight`. `gptq_to_b70lb1` is CPU-tested against a
synthetic packer and is not called from the weight loader. TP=1 only.
Do not quote tok/s from these op times.

### Next work (size, not a blocker)

1. Wire `gptq_to_b70lb1` into linear and MoE
   `process_weights_after_loading`, and register the checkpoint names
   (`qweight`, `qzeros`, `scales`, `g_idx`) so the stock loader fills
   them. Then one GPU plugin test with a synthetic GPTQ tensor through
   `ReplicatedLinear.apply`. About a day, Python only, no kernel rebuild.
   Until a real B1 blob is diffed, the unpack is synthetic-only.
   HF config for a serve would set `quant_method` to `b70_lowbit` (the
   constructor name), `pack=b70lb1`, `bits`, `group_size=128`, and
   `modules_to_not_convert` covering everything B1 left in bf16,
   including `lm_head`. Mixed per-expert bits (`expert_bits.json`) need
   one method per bit width or a per-expert bits tensor; one fused
   scheme cannot carry them. `uv pip install` the plugin into the
   flashnext venv once those Python files are frozen.
2. Large-N GEMV: overlap the next group's dequant with the current fp32
   dot, and replace the int3 extract. Worth it for a quantized head
   (probe is 1.19-1.67 ms). Expert shapes are already at a ~25 us launch
   floor and beat oneDNN int4; more expert-GEMV tuning will not move
   Ornith ITL in a way this table can see.
3. Prefill: one unpacked B panel applied across many M tiles, simd
   selects instead of scalar `load_codes16`. Current gate/up GEMM is
   14-24 GB/s with the reload counted. That is the TTFT kernel. Decode
   stays on the GEMV.

### Requests

- B1: keep the auto_gptq export. A single on-disk int3 (or int2) expert
  `qweight` plus its fp16 `scales`, or a confirmation that int3 uses the
  same `pack_bitstream` as the int4 expert you already byte-matched,
  lets `unpack_gptq_unsigned` be checked against your file. The signed
  rule this code uses is `unsigned - maxq`.
- A3: a serve config that names `b70_lowbit` will not load B1 weights
  until item 1 lands. `lm_head` should stay on the unquantized method.
- A1: registration needs no core edit on this vLLM commit for
  `qwen3_5_moe` `RoutedExperts` + `LinearBase`. The measured path is
  TP=1, `is_monolithic=False`.
