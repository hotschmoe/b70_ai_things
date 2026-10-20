# C11 expert prefill (2026-10-20)

Card 0, Intel Arc Pro B70. The published expert layer stays C10's
dequant + `torch.bmm` at T>=12288 (66.97 ms). Wider SGN, the SLM X
kernel, `grouped_mm`, and a two-stream matmul-plus-copy probe left
that number in place. The 30 ms target is unmet. The lowest sum of
measured kernels that would compose a dequant-then-GEMM layer at
T=12288 is 35.55 ms, and that sum was not executed as one layer.

GEMM-only Flash-Next budget stays the C10 table: 4255.71 ms at a
12288-token chunk, 2887 tok/s. `DQ_T_MIN` stays 12288.
`B70_PREFILL_SGN` unset stays 4. `B70_PREFILL_XSLM` unset stays on
the s64 kernel. `_C.so` and `_C.c10.so` were not replaced.

## CONFIG

Shape is the Flash-Next routed layer: E=512, top-10, H=2560, F=640.
Gate and up are IQ3_XXS, down is Q2_0. One layer is
`6 * T * 10 * 2560 * 640` FLOPs (1.208 TFLOP at T=12288).

Binaries, side by side. Neither is the daily `_C.so`.

| file | bytes | mtime | what it contains |
| --- | ---: | --- | --- |
| `b70_gguf_xpu/_C.c11.so` | 1591952 | 09:48 | SGN 4/8/16/40, no XSLM, no MT>4 work-group cap |
| `b70_gguf_xpu/_C.c11b.so` | 1760272 | 10:04 | same SGN set plus `B70_PREFILL_XSLM=1` |
| `b70_gguf_xpu/_C.c10.so` | 8294752 | 08:28 | C10 dequant library, untouched |
| `b70_gguf_xpu/_C.so` | 7574840 | 07:40 | C8 merge, untouched |

Build: icpx 2026.1, `-DB70_PREFILL_SMOKE=1`, sources
`grouped_prefill.cpp`, `dequant_fast.cpp`, `c11_bind.cpp`.
`c11_bind.cpp` is not linked with `bind.cpp`. Smoke mode throws on
`mode != 0`.

Shared-file diff, already small: `csrc/bind.cpp:182` defines
`prefill_wide_sgn() -> Tensor` and `:199` implements it. The op
returns int32 40, meaning the binary was built with the wide-SGN
source. `csrc/grouped_api.hpp` declares it. After the measurements,
`csrc/prefill_sched.inl:340` caps an MT>4 work-group at 512 and falls
back to SGN=4. That cap is source only. It is in neither `.so`.
Default SGN stays 4, so a later full rebuild does not change the
unset path.

Dispatch code, behavior unchanged:

- `b70_gguf_xpu/prefill_dq.py:34` `DQ_T_MIN = 12288`
- `prefill_dq.py:94` `moe_wants_dq`
- `prefill.py:101` calls `moe_dq` when `mode == 0` and `moe_wants_dq`

Activations in the C11 bench are `randn * 0.001` fp16. Scale-1 IQ3
packs (dequant max abs 480) overflow the fp16 swiglu store. See
Correctness.

Logs, all under `/mnt/vm_8tb/b70/build/vllm-flashnext/logs/`:

- `c11-build.log`, `c11-build-b.log`
- `c11-bench.log` (NaN gate, sweep skipped)
- `c11-nan.log`
- `c11-bench2.log` (SGN table, `_C.c11.so`)
- `c11-dnn.log` (bf16 reference, dequant GB/s, oneDNN, overlap)
- `c11-xslm.log` (SGN plus XSLM, `_C.c11b.so`)

## COMMAND

```
B70_AGENT=C11 B70_CPU_LOCK_TIMEOUT=7200 bin/cpu-run -- \
  bash vllm/flashnext/plugins/b70_gguf_xpu/scripts/c11_build.sh
```

Second build is the same script with `B70_PREFILL_OUT` pointed at
`_C.c11b.so` (`c11-build-b.log`).

```
B70_AGENT=C11 B70_GPU_LOCK_TIMEOUT=21600 bin/gpu-run --card 0 \
  vllm/flashnext/plugins/b70_gguf_xpu/scripts/c11_run.sh
```

That is `c11-bench2.log` (`_C.c11.so`). The oneDNN rerun added
`--ref-debug --bench-only --dnn-only` (`c11-dnn.log`, same `.so`).
The XSLM run set `B70_GGUF_SO=.../_C.c11b.so` and passed `--sgn-only`
(`c11-xslm.log`). The NaN probe is `logs/c11_nan_run.sh` on card 0
against both `_C.c11.so` and `_C.c10.so`.

`scripts/c11_run.sh` pins card 0, runs `bin/xpu-health --card 0`
before and after, and does not call `xe-reset`. Every C11 GPU log
has `HEALTH_BEFORE:0` and `HEALTH_AFTER:0`.

Lease holds after acquire, from the logs: bench2 41s (python 9s),
dnn 110s (python 5s, plus a quiet-window wait on G5b), xslm 43s
(python 8s), nan probe 51s. The long flock waits in front of those
acquires were H1's both-card health and an `xe-reset`. C11 did not
kill them.

## RESULT

### Correctness

Schedules match each other. On the small layer (E=4, T=8, topk=2),
`c11-xslm.log`:

```
OK sgn=8 vs sgn=4 max_abs=0.000e+00 cols=128 want=128
OK sgn=16 vs sgn=4 max_abs=0.000e+00 cols=256 want=256
OK sgn=40 vs sgn=4 max_abs=0.000e+00 cols=640 want=640
OK xslm vs sgn=4 max_abs=0.000e+00 cols=640
```

`cols` is the launched strip width. SGN=40 really launched (640
columns) on the MT=4 small layer.

The bf16-weight gate is not green. `check_small` compares the SGN=40
output (bit-identical to SGN=4) to `torch.mv` of `prefill_dequant`
bf16 weights, SiLU in fp32, and requires `max_rel <= 5e-2`.
`c11-xslm.log`:

```
FAIL sgn=40 vs bf16-weight max_abs=2.246e+01 max_rel=1.990e+00
```

`c11-dnn.log` `--ref-debug` on `_C.c11.so`, SGN=4, same scaled
input. The `at` pair is the argmax of absolute error
(`tests/c11_bench.py:459`):

```
REF bf16 max_abs=2.5277e+01 max_rel=1.9815e+00 med_abs=2.4749e+00 frac_gate=0.9368 at got=-1.1742e+03 ref=-1.1490e+03
REF fp32 max_abs=1.4741e+01 max_rel=1.9186e+00 med_abs=1.4082e+00 frac_gate=0.9638 at got=3.8224e+02 ref=3.9698e+02
REF fp16 max_abs=1.4879e+01 max_rel=1.9552e+00 med_abs=1.4258e+00 frac_gate=0.9631 at got=3.8224e+02 ref=3.9711e+02
```

`frac_gate` is the C8 combined check
`abs <= 5e-2 + 5e-2 * max(|got|, |ref|, 1e-6)`. Against the bf16
mv, 93.68% of elements pass. The largest absolute miss is 25.3 on a
value of magnitude 1174 (about 2.1% on that element). `max_rel` 1.98
is a different element. The same kernel misses the fp32 mv and the
fp16 mv of the dequantized weights. Every schedule, including XSLM,
agrees with SGN=4 at max abs 0, so this is a kernel-versus-matmul
disagreement on the scale-1 synthetic pack, shared by all of them.
The gate was not weakened. This report does not claim the fused
kernel matches a bf16-weight reference. C8's 5e-2 checks on real
GGUF rows are a different input distribution and were not re-run.

Unscaled fp16 `randn` is all NaN on this pack, on both libraries.
`c11-nan.log`, small layer, health 0/0, `C11_NAN_RC:0`:

```
PROBE_SO c11 .../_C.c11.so
STAT y_randn n=20480 finite=0 nan=20480 inf=0
STAT y_x0.02 n=20480 finite=0 nan=20478 inf=2
STAT gate_bf16 n=6553600 finite=6553600 max_abs_finite=4.8000e+02
PROBE_SO c10 .../_C.c10.so
STAT y_randn n=20480 finite=0 nan=20480 inf=0
STAT gate_bf16 ... max_abs_finite=4.8000e+02
```

`fast_pack` writes a scale of 1.0. IQ3 dequant reaches 480. The fp16
swiglu store overflows, and a Q2_0 zero times that overflow is NaN.
`x * 0.001` is finite (small-layer max abs 4832). The schedule does
not depend on the values. C11's SGN=4 time at T=512 is 9.87-10.00 ms
against C10's published 10.82 ms. That gap is the scaled input. It
is not a faster kernel, and it does not replace C10's published
fused times. The first bench (`c11-bench.log`) skipped the sweep on
the NaNs (`VERDICT failures=4 skip_bench=1`, `C11_GPU_RC:1`) with
health still 0/0.

`c11-xslm.log` prints `VERDICT failures=1` and the process still
exits 0 (`tests/c11_bench.py:519` returns 0). The 1 is the
bf16-weight gate. `c11-dnn.log` is `VERDICT failures=0` because
`--bench-only` skips `check_small`; the REF lines above are from
that same process.

### Routing

Every expert is live. `T * topk` divides E, so the mean row count is
exactly 10 / 80 / 240. Two draws:

| log | T=512 mmax / empty | T=4096 | T=12288 |
| --- | --- | --- | --- |
| c11-bench2 | 22 / 0 | 105 / 0 | 293 / 0 |
| c11-xslm | 20 / 0 | 116 / 0 | 291 / 0 |

Skipping inactive experts saves no dequant bytes. A uniform bmm
(rows_per = mean) has no pad. `moe_dq` still pads to mmax
(`prefill_dq.py:174` `_project_bmm`).

### Wide SGN

`launch_s64` (`csrc/prefill_sched.inl`) is the Strata 64-column
schedule, templated `<TYPE, MT, SGN>`. One group per expert per
N-strip. Each subgroup loads X from global itself. The number of
subgroups is N/16 no matter how they are packed, so a wider strip
does not cut X reads. The header comment that used to say otherwise
was corrected after this measurement. Gate+up is already one GEMM
of N=1280 (even column gate, odd column up). Down is a second launch,
N=2560, K=640. Auto policy: experts use s64; MT=4 when average rows
<= 32, else 16.

Device `max_work_group_size` is 1024 (`C11_SGN` lines, both logs).
MT=4 (grf 128) launched a group of 640. MT=16 (grf 256) throws
`Total number of work-items in a work-group cannot exceed 512 for
this kernel`. SGN=40 is 640, so it runs at T=512 and throws at
T=4096 and T=12288 in both measured binaries. Source now falls back
to SGN=4 in that case. The fallback was not rebuilt, because SGN=4
is already timed.

Event time, `_C.c11b.so`, `c11-xslm.log`. C10 published layer is
10.82 / 26.76 / 74.34 ms fused, and 66.97 ms bmm at T=12288
(`reports/1C10_prefill_dq.md`).

| T | sched | event ms | TF | gu ms | down ms | cols |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 512 | SGN=4 | 10.00 | 5.03 | 6.91 | 2.54 | 64 |
| 512 | SGN=8 | 10.12 | 4.97 | 7.14 | 2.61 | 128 |
| 512 | SGN=16 | 10.00 | 5.03 | 6.84 | 2.65 | 256 |
| 512 | SGN=40 | 14.76 | 3.41 | 10.43 | 3.69 | 640 |
| 512 | XSLM | 14.80 | 3.40 | 10.63 | 3.53 | 640 |
| 4096 | SGN=4 | 28.26 | 14.25 | 17.06 | 6.92 | 64 |
| 4096 | SGN=8 | 27.87 | 14.45 | 16.68 | 6.92 | 128 |
| 4096 | SGN=16 | 27.98 | 14.39 | 16.69 | 7.08 | 256 |
| 4096 | SGN=40 | throw | | | | |
| 4096 | XSLM | 28.97 | 13.90 | 19.48 | 5.14 | 512 |
| 12288 | SGN=4 | 77.78 | 15.53 | 46.02 | 18.98 | 64 |
| 12288 | SGN=8 | 76.54 | 15.78 | 44.73 | 18.78 | 128 |
| 12288 | SGN=16 | 76.28 | 15.84 | 44.32 | 18.91 | 256 |
| 12288 | SGN=40 | throw | | | | |
| 12288 | XSLM | 70.02 | 17.25 | 45.11 | 11.94 | 512 |

`_C.c11.so` in `c11-bench2.log` is the same story within 0.4 ms on
every cell that launched: T=12288 SGN 4/8/16 = 77.92 / 76.42 / 76.14 ms.
Best fused cell is SGN=16 at 76.14-76.28 ms. C10's bmm layer is
66.97 ms. Widening SGN is not the expert win.

### SLM X kernel (idea a, inner-loop dequant removed from X only)

`grouped_prefill.cpp:420` `xslm_on`. `B70_PREFILL_XSLM=1` calls
`launch_gemm<TYPE, MT, 0>` (`:152`), the C2 kernel: one X panel in
SLM per K step, shared by the work-group. `choose_layout` (`:70`)
picks the widest strip that divides N and fits. MT>4 caps the group
at 512, so the down GEMM (N=2560) lands on cols=512 and the gate+up
GEMM (N=1280) on the widest divisor at or under 512. The split
record keeps the down launch, which is why T=12288 prints cols=512
and the small layer (down N small enough) prints cols=640.

XSLM is bit-exact with SGN=4. At T=12288 it cuts the down GEMM from
18.98 ms to 11.94 ms and leaves gate+up at 45.11 ms. The layer is
70.02 ms, 17.25 TFLOPS. That is faster than fused SGN=16 and slower
than C10's 66.97 ms bmm. At T=512 and T=4096 XSLM is slower than
SGN=4/8 (14.80 vs 10.00, 28.97 vs 27.87). It stays opt-in. Unset
env stays on s64.

The bench printer adopts the fastest C11 kernel when `XSLM_OK` is
set (`tests/c11_bench.py:271`). `c11-xslm.log` therefore prints
`BUDGET T=12288 best_sgn=xslm512 layer_ms=70.02` and a GEMM-only
2791 tok/s. That line is the fastest kernel inside the C11 binary.
It is slower than the C10 bmm layer, and it is not the dispatch.

### Dequant bandwidth (idea a, active experts)

`c11-dnn.log`, `_C.c11.so`, in+out bytes, roof 608 GB/s. One expert
projection in bf16 is E*2560*640*2 = 1.678 GB. Three packs are
5.03 GB. Forty-eight layers are 241 GB.

| buffer | event us | GB/s |
| --- | ---: | ---: |
| IQ3_XXS gate, all experts | 7827.5 | 255.4 |
| IQ3_XXS up | 7820.2 | 255.6 |
| Q2_0 down | 7396.9 | 258.7 |
| sum of the three | 23044.6 | |
| copy of one gate bf16 (read+write) | 6347.6 | 528.6 |

`c11-bench2.log` measured the same three dequants at 7807 / 7815 /
7407 us and the copy at 6376 us, 526.3 GB/s. Dequant holds about
42% of the 608 GB/s roof and about half the measured copy rate.
Empty experts are 0, so dequanting only the active set is the same
transfer.

### oneDNN ceilings (idea a, grouped GEMM on bf16)

Uniform shapes: every expert has `rows_per = T * topk / E` rows, so
`torch.bmm` has no pad. `dense_ceil` is one `torch.mm` of the
concatenated tokens, same FLOPs as the gate bmm.
`torch.nn.functional.grouped_mm` uses equal-length offs. `c11-dnn.log`:

| T | op | event us | TFLOPS |
| --- | --- | ---: | ---: |
| 4096 | dense mm gate | 1108.6 | 121.07 |
| 4096 | bmm gate | 3439.2 | 39.03 |
| 4096 | grouped_mm gate | 21979.7 | 6.11 |
| 4096 | bmm down | 3603.5 | 37.25 |
| 4096 | bmm gate+up N=1280 | 6467.4 | 41.51 |
| 12288 | dense mm gate | 3236.3 | 124.42 |
| 12288 | bmm gate | 4719.2 | 85.32 |
| 12288 | grouped_mm gate | 21525.6 | 18.71 |
| 12288 | bmm down | 4649.4 | 86.60 |
| 12288 | bmm gate+up N=1280 | 7860.8 | 102.45 |

`grouped_mm` does not scale from T=4096 to T=12288 (22.0 ms, then
21.5 ms) and loses to uniform bmm by 4.6x at T=12288. The dense mm
at 124 TFLOPS is the ceiling for that gate's FLOPs (3.24 ms). The
uniform gate bmm is 4.72 ms. Grouping costs about 1.5 ms there.

Kernel sums, not layer times:

- T=12288, three dequants + uniform gate+up bmm + uniform down bmm
  = 23.045 + 7.861 + 4.649 = 35.55 ms.
- T=4096, same dequants (the weight size does not change) + the two
  bmms = 23.045 + 6.467 + 3.604 = 33.12 ms.

35.55 ms is already above 30 ms, before sort, gather, scatter, SiLU,
reduce, pad, or a host sync. 33.12 ms is already above the fused
layer at T=4096 (C10 26.76 ms, this run's SGN=8 27.87 ms). A second
gate-shaped bmm was not timed; up has the same shape as gate, so
`moe_dq`'s separate gate and up bmms are about 4.72 + 4.72 + 4.65 +
23.05 = 37.1 ms of kernels at T=12288. C10's measured `moe_dq` layer
is 66.97 ms. The gap between that layer and these kernel sums was
not broken out here. The extra work in the shipped path is
`_project_bmm`: pad to mmax (293 vs 240 at T=12288), `zero_`, pad,
bmm, unpad, and `_Group` pulling `starts` back to the host.

### Overlap (idea c)

Two streams, small tensors, not the expert packs. `torch.mm` of
`[4096, 2560] x [2560, 4096]` bf16 against a copy of `[4096, 2560]`.
`c11-dnn.log`:

```
BENCH overlap mm_us=635.5 copy_us=123.8 both_us=726.5 wall_mm=737.7 wall_copy=234.5 wall_both=842.3
```

635.5 + 123.8 = 759.3. The paired launch is 726.5, so 33 us of the
copy is hidden (about a quarter of the copy). Wall time of the pair
(842 us) is above the matmul alone (738 us). Streams did not overlap
this matmul with this copy in a useful way. A dequant kernel overlapped
with a bmm was not run. Pipelining layer L+1 dequant with layer L
GEMM was not coded. One layer of three bf16 projections is 5.03 GB;
forty-eight layers are 241 GB, so resident bf16 for every layer does
not fit on the card.

### Gate+up fusion (idea b)

The fused s64 layer already multiplies gate and up as one N=1280
GEMM. SiLU is a separate kernel (`swiglu_kernel` in
`grouped_prefill.cpp`), not an epilogue. The bmm path dequants and
multiplies gate and up separately. Fusing those two bmms, on the
uniform shape, is the 7.86 ms row against two 4.72 ms rows: about
1.6 ms at T=12288. That does not move a 23 ms dequant.

### Full-model GEMM budget

Unchanged from `reports/1C10_prefill_dq.md`. 48 layers = 36 GDN +
12 full attention. Attention, GDN recurrence, conv, norms, PLE, and
indexer top-k are outside the sum. Routed experts are the measured
winner times 48: fused at 512 and 4096, dequant + bmm at 12288.

| piece | T=512 | T=4096 | T=12288 |
| --- | ---: | ---: | ---: |
| routed experts | 519.38 | 1284.53 | 3214.53 |
| GDN qkv+gate+out | 44.46 | 185.00 | 523.37 |
| full attn q+k+v+output | 12.80 | 53.26 | 150.67 |
| shared expert | 10.23 | 31.74 | 79.75 |
| head (12288 projected) | 12.39 | 55.32 | 153.67 |
| HC + output mixer | 4.66 | 27.08 | 118.43 |
| router + indexer | 0.79 | 4.97 | 15.29 |
| GEMM sum | 604.71 | 1641.90 | 4255.71 |
| GEMM-only tok/s | 847 | 2495 | 2887 |

Per layer at T=12288 the expert piece is 67.0 ms, about 75% of the
4.26 s GEMM sum. 2000 tok/s on a 12288-token chunk is 6.14 s, so
1.89 s remains for attention and the GDN recurrence. That slack is
not a measurement of those kernels. End-to-end 2000 tok/s is not
claimed.

Shipping the 70.02 ms XSLM layer would move the expert sum to
3360.9 ms and the GEMM sum to 4402.1 ms (2791 tok/s). That is a
regression against 2887 tok/s. Shipping SGN=16 at 76.3 ms is a
larger regression (bench2 printed 2617 tok/s). Neither replaces
the table.

`max_num_batched_tokens` stays 12288. It is the only measured T
where expert bmm beats the fused layer, and it is the best GEMM-only
rate in the table.

## VERDICT

The expert layer at T=12288 is still C10's dequant + `torch.bmm`,
66.97 ms. The goal, well under 30 ms and a strong T=4096, is unmet.

What was measured and left opt-in or unused:

- Wider s64 strips do not cut X reads. Best fused T=12288 is
  SGN=16 at 76.3 ms. SGN=40 throws at MT=16 (work-group 640, kernel
  limit 512). Source falls back to SGN=4; the measured `.so` files
  still throw, and the bench catches it.
- XSLM (`B70_PREFILL_XSLM=1`) is bit-exact with SGN=4, helps only
  the down GEMM, and lands at 70.02 ms. Slower than C10 bmm. Off
  unless the env is set.
- `grouped_mm` is 21.5 ms for one gate at T=12288. Uniform bmm is
  4.72 ms.
- Dequant of the three packs is 23.05 ms at 255-259 GB/s. A bf16
  copy of one gate is 6.35 ms at 529 GB/s. Empty experts are 0.
- Three dequants plus a fused gate+up bmm plus a down bmm add to
  35.55 ms of kernels at T=12288 and 33.12 ms at T=4096. Those sums
  are not layer times. The shipped `moe_dq` layer remains 66.97 ms.
- Two streams hid about a quarter of a small copy behind a small
  matmul. Layer pipelining was not built.

Dispatch stays `moe_wants_dq` at T>=12288. The GEMM-only budget stays
4255.71 ms, 2887 tok/s. The bf16-weight gate on the scale-1 synthetic
pack fails (`max_rel` 1.98, `frac_gate` 0.937) while every schedule
matches SGN=4 at max abs 0. The gate was not weakened.

No xe-reset. Health before and after every C11 card-0 run was 0.
