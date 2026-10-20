# C8 prefill, continuing C6

Card 0, Arc Pro B70, icpx 2026.1, host IGC left as C5 left it.
Flags stayed `-fsycl -O3 -fPIC -fexceptions -fsycl-device-code-split=per_kernel`
(`scripts/build.sh:48-49`). C5's 06:38 journal entry is still the last C5
entry: IGC 2.41.5, immediate lists, precise, and AOT move kernels about 1 us.
No flag change, no IGC mount.

Shipped package `b70_gguf_xpu/_C.so` is now the merged library
(7574840 bytes, sha256
`086595a01babbc366b3d5366147464ebd24022fba0b65d83b1ff530489c70d55`),
copied from `_C.c8full.so` after the CPU dequant check. The 04:32 decode
binary it replaced was 6391032 bytes, sha256
`1e484faeeade967701bf56b8abeb24acd230b28eec79212b28a101ffe4bc3cac`.
`_C.c6.so`, `_C.c8.so`, `_C.graph.so`, and `_C.c8full.so` are unchanged
side copies.

Speed numbers below are event minima from a quiet `bin/cpu-run` window
inside `bin/gpu-run --card 0`. Health was HEALTHY before and after every
GPU run (`HEALTH_BEFORE:0`, `HEALTH_AFTER:0`). Correctness gate is
`failures=0` (prefill atol=rtol=5e-2; dequant is bit-exact).

## What changed in the prefill source

`csrc/prefill_sched.inl` is the C6 kernel (dequant one 16-column panel
into SLM, subgroups walk M, X from global, fp16 DPAS 8x16x16) with two
host-side changes:

- Dense (`E==1`) launches N as the fast grid axis (`prefill_sched.inl:195`).
  Consecutive groups share one M tile's X. Experts stay one group per
  expert, N fast, which C6 already did.
- Auto policy (`prefill_sched.inl:24-29`): `B70_PREFILL_SCHED=reuse|s64`
  still overrides. Otherwise dense uses reuse rm64 and experts use s64.
  `B70_PREFILL_MT` still defaults to 4 when average rows per expert are
  <= 32, else 16. `B70_PREFILL_RM` still defaults to 64.

`tests/c6_bench.py` names the arm with `_f=0` on the guarded closure
(`c6_bench.py:223` and `:259`). The first smoke log printed `BENCH dense 0`
because `guard` passed `failures` as the first positional. Arm order in
that log is `c6_bench.py:128-133`: s64-mt4, s64-mt16, reuse-rm32, reuse-rm64.

Decode sources (`kernels.cpp`, `kreg.hpp`, `row_moe.hpp`, `bind.cpp`) were
not edited. The full link picked up `kernels.cpp` as it stood at
2026-10-20 07:32:53. That mtime was the same after the link.

## Task 1. Dense GEMM

Shape is Q4_K, N=12288, K=2560, the full-attention `attn_q` tile.
C2 on the old one-work-group schedule (`reports/1C2_prefill.md`):
5295.8 / 42015.7 / 126036.2 us at M=512 / 4096 / 12288.

### C6 smoke, before this schedule change

CONFIG: `_C.c6.so` (5445576 bytes, 2026-10-20 06:23), `c6_run.sh --full`.
COMMAND: `bin/gpu-run --card 0 bin/cpu-run --quiet scripts/c6_run.sh --full`
with `B70_GGUF_SO` pointing at `_C.c6.so`. Log
`/mnt/vm_8tb/b70/build/c8-prefill-smoke.log`. Exit 0, failures=0.
RESULT: reuse-rm64 event-min us / TFLOPS, then torch.mm bf16 on the same
shape:

| M | reuse-rm64 us | TFLOPS | torch.mm us | torch TFLOPS |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 1226.8 | 26.26 | 301.9 | 106.69 |
| 4096 | 12171.9 | 21.17 | 1853.0 | 139.07 |
| 12288 | 44877.8 | 17.23 | 5461.6 | 141.55 |

M=512 already matches a single 512-row tile, so the grid axis cannot
matter there. From M=512 to M=12288 the C6 reuse rate fell 26.3 to 17.2
TFLOPS. s64-mt16 on the same binary was 2317 / 18249 / 54660 us
(13.9 / 14.1 / 14.1 TFLOPS).

### Rejected: wider panel plus an SLM accumulator

A BN=128 kernel that staged both X and W in SLM, and an s64 path that
stored the fp32 accumulator to SLM every K step, compiled and passed
`failures=0` (`/mnt/vm_8tb/b70/build/c8-prefill-c8bench.log`). Reuse-rm64
fell to 4795 / 37843 / 112695 us (6.7 / 6.8 / 6.9 TFLOPS). The 64-row
tile re-dequants the weight panel once per 64 rows (8 times at M=512,
192 times at M=12288). The C6 kernel dequants once per 512 rows at M=512.
The SLM-accumulator chunk at s64-mt16 was 14161 / 129536 / 388447 us
against the register s64-mt16 path at 2313 / 18250 / 54669 us on the
restored kernel. Both schedules were removed. The binary that held them
was overwritten by `_C.c8.so`; the log is the record.

### N-fast reuse, the dense result

CONFIG: `_C.c8.so` (5445760 bytes, sha256
`82f8e4296849c1c3d9f09592e225923d47d44763279a6da2bd8116dbbe69e93c`),
built from the restored C6 device kernel with the N-fast dense grid.
This smoke binary predates the auto-policy edit; the bench forces each
arm through the environment.
COMMAND: same `c6_run.sh --full` lease, `B70_GGUF_SO=_C.c8.so`. Log
`/mnt/vm_8tb/b70/build/c8-prefill-c8bench2.log`. gpu-run exit 0,
failures=0, health 0/0. Card 0 was waited out from g1-kv; that lease
was left alone.
RESULT: event-min us / TFLOPS.

| M | s64-mt4 | s64-mt16 | reuse-rm32 | reuse-rm64 | torch.mm bf16 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 15856.7 / 2.03 | 2313.0 / 13.93 | 1598.2 / 20.15 | 1227.1 / 26.25 | 323.0 / 99.74 |
| 4096 | 127881.7 / 2.02 | 18250.2 / 14.12 | 12364.3 / 20.84 | 10243.9 / 25.16 | 1855.5 / 138.89 |
| 12288 | 386898.9 / 2.00 | 54669.0 / 14.14 | 36938.4 / 20.93 | 30689.3 / 25.19 | 5481.5 / 141.04 |

reuse-rm64 is the dense winner at every M. Versus C2: 4.32x / 4.10x /
4.11x. Versus the C6 smoke binary the grid flip is 1.00x / 1.19x / 1.46x,
and TFLOPS stays flat at about 25 from M=512 to M=12288. Versus torch.mm
the same kernel is 3.80x / 5.52x / 5.60x behind (26 TFLOPS against
100-141).

The flat rate is the wall. At M=512 the winning kernel already dequants
each 16-column panel once (MTILE=512 covers every row) and still takes
1.227 ms against torch 0.323 ms. GRF 256 holds about eight fp32 8x16
accumulators (RM=64). Covering the same 512 rows with a wider N strip
doubles the accumulator file or drops M coverage and repeats the dequant.
The BN=128 measurement above is that second outcome. The next dense
kernel, not built here, is a one-shot dequant of the weight tile to fp16
global memory and then a packed XMX GEMM. The bench weight tile is
N*K*2 = 63 MB, about 0.1 ms at the 608 GB/s roof.

## Task 2. Grouped expert prefill

Layer is IQ3_XXS gate/up plus Q2_0 down, E=512, top-10, H=2560, F=640.
C2 (`reports/1C2_prefill.md`): 13884.8 / 27598.9 / 66129.1 us at
T=512 / 4096 / 12288, expert-only 48-layer tok/s 768 / 3092 / 3871.
C2 splits: T=512 GU 10268.4 down 3440.1; T=4096 GU 19562.5 down 5085.0;
T=12288 GU 45207.1 down 11710.2.

Same log as the dense result (`c8-prefill-c8bench2.log`), event-min:

| T | s64-mt4 us | s64-mt16 us | reuse-rm32 us | reuse-rm64 us | winner expert48 tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 9716.9 | 12892.6 | 39306.9 | 39471.6 | 1097.7 (s64-mt4) |
| 4096 | 36075.5 | 28335.7 | 60092.2 | 90708.1 | 3011.5 (s64-mt16) |
| 12288 | 109257.1 | 76670.8 | 76389.3 | 101428.5 | 3351.3 (reuse-rm32) |

T=512 s64-mt4 split, one call: sort 91.0, gather 36.6, GU 7060.0,
swiglu 71.6, down 2584.2, reduce 187.3. Both GU and down beat C2.
T=4096 s64-mt16 split: GU 17737.4, down 7172.0. GU beats C2; down does
not, and the layer is 28.34 ms against C2 27.60 ms.
T=12288 s64-mt16 split: GU 46602.0, down 19165.4. Down is 19.2 ms
against C2's wide-tile 11.7 ms. reuse-rm32 is 76.39 ms, tied with
s64-mt16 at 76.67 ms. Both sit behind C2's 66.13 ms.

Auto policy therefore stays s64 for E>1. reuse wins only the T=12288
photo finish (0.3 ms) and loses T=512 by 4x (39.3 ms vs 9.72 ms) and
T=4096 by 2x. Average rows are 10 / 80 / 240, so the MT default is 4
at T=512 and 16 at the two larger T, which matches the s64 winners.

A fp16 scratch plus a batched GEMM for the small-T experts was considered
and not built. T=512 GU is already 7.06 ms. Expanding gate+up to fp16
for 512 experts is about 3.4 GB, and the write alone is about 5.5 ms at
608 GB/s. The large-T lever that was also left unbuilt is C2's wide
`launch_gemm` on the K=640 down projection. Routing K<=1024 there would
have to beat 19.2 ms at T=12288 without giving back the T=512 down win
(2.58 ms vs C2 3.44 ms). That measurement is the next grouped experiment.

## Task 3. One shipped library

CONFIG: full link of `kernels.cpp` (mtime 07:32:53), `bind.cpp` /
`kreg.hpp` / `row_moe.hpp` (06:57:12), `grouped_prefill.cpp` (06:16:21),
`prefill_sched.inl` (07:38:54). `B70_PREFILL_OUT=_C.c8full.so`.
`MAX_JOBS=4`. cpu-run waited on C9's `build.sh`, then compiled 89 s,
exit 0, 120 warnings, log `/mnt/vm_8tb/b70/build/c8-full-build.log`.
The link also rewrote `libdequant_cpu.so` from the existing
`host_dequant.cpp`.
COMMAND: one card-0 lease, `/mnt/vm_8tb/b70/build/c8-task3.sh`.
The script unsets `B70_PREFILL_SCHED`, `B70_PREFILL_MT`, `B70_PREFILL_RM`,
and `B70_PREFILL_SPLIT`, so `prefill_check.py` hits the auto policy.
Log `/mnt/vm_8tb/b70/build/c8-task3.log`. `TASK3_FAIL:0`, health 0/0.

Decode, same moe_layer bench, gate IQ2_XS, up IQ2_XXS, down IQ4_NL,
E=32, topk=10, I=640, K=2560. `behind_us` is the fused layer.
04:32 `_C.so` first, then `_C.c8full.so`:

| T | 04:32 behind us | merged behind us | 04:32 a2 us | merged a2 us |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 472.7 | 367.3 | 209.5 | 199.0 |
| 2 | 417.0 | 315.0 | 419.0 | 320.3 |
| 4 | 713.9 | 515.1 | 728.0 | 577.0 |
| 8 | 1383.3 | 815.1 | 1413.3 | 852.0 |

Both benches printed `VERDICT failures=0`. The merged layer is faster
because the .so contains the 07:32 decode sources. The prefill schedule
is a different entry point. This is a decode-time comparison, and it
cleared the "compare before replace" check.

Also on `_C.c8full.so`, all `VERDICT failures=0`:

- `tests/gpu_check.py`: bf16 dequant mismatches=0 for Q4_K, Q5_K, Q6_K,
  IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL, IQ4_XS, Q2_0, Q2_K, Q3_K.
- `tests/prefill_check.py`: real IQ3_XXS shard, mode 0 dense and grouped
  for those types, plus mode 1 bf16 and mode 2 int8 on the instantiated
  subset. Env was unset, so this is the auto policy.
- `tests/c6_bench.py --full`: failures=0. Large-M reuse-rm64 split gemm
  times were 1222.4 / 10235.3 / 30690.1 us, matching the clean bench.
  The M=512 event-min inside this long suite was 1341.1 us; the dedicated
  bench's 1227.1 us is the number used above.

CPU dequant, no GPU: `tests/compare_cpu.py` with `PYTHONPATH` set to the
vllm-flashnext venv, under `bin/cpu-run`. Log
`/mnt/vm_8tb/b70/build/c8-compare-cpu.log`. 13/13 types, mismatches=0,
`VERDICT failures=0`. Q2_0 is the local transcription; the other types
are gguf-py. Q2_K and Q3_K came from the Qwen3.6 UD GGUFs under
`/mnt/cache/b70/devmodel`. A first invocation omitted `PYTHONPATH` and
exited in 0 s on `import numpy`; the log file is the retry.

Replace, only after that verdict, and only while `_C.so` still hashed to
the 04:32 sha. `cp -a` to `_C.so.c8tmp` on the same filesystem, re-check
the live sha, then `mv`. After the mv, `_C.so` and `_C.c8full.so` both
hash to `086595a0...`. `_C.graph.so` (6987192 bytes) and `_C.c6.so`
(5445576 bytes) were not touched. `_C.so.c8tmp` is gone.

## Task 4. Full-layer prefill budget

Attention, the GDN recurrence, conv, norms, PLE lookup, and indexer
top-k are outside this sum. The tok/s column is GEMM time only.
Expert-only 48-layer tok/s from task 2 is a different quantity.

Shapes read from IQ3_XXS shard 1
(`Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf`). ggml shape
is (K, N). Block 0 is a GDN layer, block 3 is a full-attention layer:

| tensor | shape (K, N) | type on this block |
| --- | --- | --- |
| blk.0.attn_qkv | (2560, 10240) | IQ4_XS |
| blk.0.attn_gate | (2560, 6144) | IQ3_S |
| blk.0.ssm_out | (6144, 2560) | IQ4_XS |
| blk.3.attn_q | (2560, 12288) | IQ4_XS |
| blk.3.attn_k | (2560, 512) | IQ4_XS |
| blk.3.attn_v | (2560, 512) | Q6_K |
| blk.3.attn_output | (6144, 2560) | Q6_K |
| output.weight | (2560, 248320) | Q5_K |
| shared gate/up/down | (2560, 640), (2560, 640), (640, 2560) | mixed per layer |
| hc down/up/inject | (10240, 320), (320, 10240), (10240, 4) | BF16 |
| router ffn_gate_inp | (2560, 512) | BF16 |

Quantized mats are scaled from the measured Q4_K reuse-rm64 time at the
same M: 1.2271 / 10.2439 / 30.6893 ms at M=512 / 4096 / 12288, by
`(N*K)/(12288*2560)`. That is a Q4_K-rate projection. Real dense types
are the mix in the table. C2's old kernel at M=512 took 7.53 ms on Q5_K
and 8.35 ms on Q6_K against 5.30 ms on Q4_K, and 1.77 / 1.45 ms on
IQ4_XS / IQ3_S. Those ratios are not applied here. BF16 mats (HC,
router, indexer q/k) use the torch.mm time from the same bench
(0.3230 / 1.8555 / 5.4815 ms) scaled the same way. Indexer shapes are
from `reports/02_model_support.md` (BF16, 12 full layers only).

Model milliseconds, 48 layers (36 GDN, 12 full attn):

| piece | T=512 | T=4096 | T=12288 |
| --- | ---: | ---: | ---: |
| routed experts, measured auto | 466.4 | 1360.1 | 3680.2 |
| GDN qkv+gate+out, Q4 rate | 81.0 | 676.1 | 2025.5 |
| full attn q+k+v+output, Q4 rate | 23.3 | 194.6 | 583.1 |
| shared expert, Q4 rate | 9.2 | 76.8 | 230.2 |
| head, Q4 rate | 24.8 | 207.0 | 620.2 |
| HC + output mixer, torch rate | 4.7 | 26.7 | 111.5 |
| router + indexer, torch rate | 0.8 | 4.9 | 14.4 |
| GEMM sum | 612.1 | 2557.3 | 7265.0 |
| GEMM-only tok/s | 836 | 1602 | 1691 |

Per layer at T=12288: routed experts 76.7 ms (measured), one GDN layer's
three projections 56.3 ms, one full-attention layer's four projections
48.6 ms, shared expert 4.8 ms, HC attn+ffn 2.3 ms. The head is once,
620 ms at the Q4 rate. If the old Q5/Q4 ratio of 1.42 carried onto the
new kernel, the head would be about 880 ms. That case is still smaller
than the expert sum.

2000 tok/s on a 12288-token chunk is a 6.14 s budget. This GEMM sum is
7.26 s, 1691 tok/s, before attention and the GDN recurrence. Routed
experts plus the GDN projections alone are 5.71 s (about 2150 tok/s).
Full-attention projections plus the head bring the sum under 2000.
Strata's ~2200 tok/s at 32K is a full-model rate; a linear 12288-token
slice of that rate is 5.59 s. The GEMM projection is already slower
than that slice.

At T=512 the measured expert layer is the win: 9.72 ms and 1098
expert-only tok/s, against C2 13.88 ms and 768. The GEMM-only 836 tok/s
at T=512 is the small-M projection, and launch plus attention overhead
sit on top of it. At T=4096 the expert layer is 28.3 ms, expert-only
3012 tok/s, beside C2's 3092.

## VERDICT

Dense prefill works, matches the fp16 reference, and holds 25 TFLOPS
from M=512 through M=12288. That is 4.1x C2 and 5.6x behind torch.mm
bf16 at M=12288. Grouped experts work. T=512 is faster than C2
(9.72 ms vs 13.88 ms). T=4096 and T=12288 are slower (28.3 / 76.7 ms
vs 27.6 / 66.1 ms), and the gap is the K=640 down projection.
The merged library is `_C.so`, with decode moe_layer faster than the
04:32 binary on the same bench and with compare_cpu, gpu_check,
prefill_check, and c6 --full green.

The 2000 tok/s Flash-Next prefill goal is not met by this budget.
GEMM-only at T=12288 projects to 1691 tok/s. Attention is still extra.

Next kernel, not started: one-shot fp16 dequant plus a packed XMX GEMM
for the dense tiles, and a measured wide down schedule for K=640 that
has to keep the T=512 win.
