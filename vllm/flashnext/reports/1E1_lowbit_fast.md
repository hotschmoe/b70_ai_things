# 1E1 -- b70_lowbit graph decode and prefill

Plugin `vllm/flashnext/plugins/b70_lowbit_xpu`. Quant name `b70_lowbit`.
No vLLM core edit. Card 1 only. Every serve used
`FULL_DECODE_ONLY`, FlashAttention, `B70_XPU_CG_RECLAIM=500`,
`B70_LOWBIT_MOE_LAUNCHES=2`, bf16, TP=1, max-len 33792, util 0.90,
served name `hotschmoe-dev` first. MTP off. Identity and coherence
passed on every serve below.

The 8k prefill target of 5000 tok/s was not met. The best measured
int3 prefill is 366 tok/s. Decode for the same checkpoint is 11.3 ms
ITL inside the graph.

## Libraries

```
libb70_lowbit_xpu.so       1442856 bytes  2026-10-20 04:05  templated GEMV
libb70_lowbit_xpu_e1.so    1696504 bytes  2026-10-20 06:35  grouped DPAS
libb70_lowbit_xpu_e1mix.so 2280328 bytes  2026-10-20 06:56  moe_gemv_mixed
libb70_lowbit_xpu_e1mm.so  2345192 bytes  2026-10-20 08:25  unpack_bf16
```

The 04:05 file is the int4 and int3 graph arm. Later libraries were
new files. Nothing overwrote a mapped `.so`.

## 1. Graph decode

int3 checkpoint `qwen36-w3a16-g128-experts-w4a16-trunk`, templated
GEMV, eager reference C1 ITL 51.7 ms.

| | C1 ITL p50 | C2 ITL p50 | C1 TTFT | 8k prefill |
|---|---:|---:|---:|---:|
| int3 graph | 11.29 ms | 15.65 / 15.65 ms | 2388 ms | 112 tok/s |

C1 p90/p99 11.52/11.97 ms, 90.34 tok/s, n=57. Prefill at 1k/8k/32k:
99.33 / 112.32 / 115.78 tok/s. Two graph captures, about 1 s each.
Both coherence texts: " Paris, a city renowned for its rich history,
culture, and iconic landmarks."

Logs: `e1-int3-graph-bench.json`, `e1-int3-ppl.json`.

## 2. Expert bits, same graph flags

Bytes are a full expert-weight read for every selected slot. 8 slots
times 40 layers, no dedup. Trunk int4 is the same on every checkpoint
and is not in the byte column.

| Arm | Library | C1 ITL p50 | C2 ITL p50 | 8k prefill | Expert bytes / token | Served ppl |
|---|---|---:|---:|---:|---:|---:|
| int4 | 04:05 templated | 10.23 ms | 13.65 / 13.61 ms | 109 tok/s | 519045120 (495 MiB) | 8.858 |
| int3 | 04:05 templated | 11.29 ms | 15.65 / 15.65 ms | 112 tok/s | 393216000 (375 MiB) | 9.120 |
| mixed int3/int2 | e1mix `moe_gemv_mixed` | 10.63 ms | 14.26 / 13.97 ms | 127 tok/s | 368050176 (351 MiB) | 10.159 |

int2 would be 267386880 bytes per token. The mixed figure is the
design mix, 80 percent of the mass on int3 and 20 percent on int2.
A count-weighted average (7573 int2 experts, 2667 int3) is
300158976 bytes and understates routed traffic, because the hot
experts are int3.

int4 moves 32 percent more expert bytes than int3 and the C1 step is
1.06 ms faster. If the whole step were that read, the rates would be
about 51 GB/s and 35 GB/s. This decode step is not expert-bandwidth
bound.

The mixed arm is a different kernel from the templated GEMV. It is
not an expert-bits-only delta. All 40 layers are mixed. Storage width
is int3 (qweight rows 192 and 48). Shorter int2 rows are sliced in
`gptq_experts_mixed_to_b70lb1`. The vLLM module prefix is
`model.language_model.model.layers.N.mlp.experts`. Checkpoint keys
omit the inner `.model`. Lookup uses the `layers...` tail. CPU check
`MIXED_BITS_OK` against `expert_bits.json`. GPU check `MIXED_PROBE_OK`:
uniform int3 matches the template with max_abs 0, and an int2/int3
mix at H=2048 I=512 matches per-expert prefill with rel_l2 1.0e-3.

int4 text starts " Paris, a city renowned for its rich history, art,
and culture. Sit". Mixed text repeats "The capital of France is
Paris." int4 load 18.13 GiB, KV 7.22 GiB. Mixed load 14.38 GiB, KV
10.78 GiB. Both fit at util 0.90. Weight-load wall time was 664 s and
522 s on a busy host. That is not the ITL.

Logs: `e1-int4-graph-bench.json`, `e1-int4-ppl.json`,
`e1-mix-graph-bench.json`, `e1-mix-ppl.json`.

## 3. Prefill

Same int3 checkpoint. 8k is 8185 tokens.

| Path | 1k | 8k | 32k |
|---|---:|---:|---:|
| per-expert WNA16 | 99 tok/s | 112 tok/s | 116 tok/s |
| grouped bf16 DPAS, one submit per projection | 165 | 174 | 175 |
| unpack to bf16, vendor GEMM | 288 | 366 | 385 |

Grouped DPAS is numerically the per-expert loop (`GROUPED_PROBE_OK`,
64 KiB SLM accepted, bits 2/3/4). It is 1.55x and far from 5000.

One expert GEMM M=256 K=2048 N=1024: bf16 DPAS 1.118 ms, int8 DPAS
1.156 ms, vendor matmul 0.113 ms. The int8 tile is not faster. It was
not served.

One q-proj shape M=8192 N=2048 K=2048: WNA16 72.659 ms, unpack plus
matmul 0.575 ms, rel_l2 0 against WNA16. Bits 2 and 4 unpack match
`format.dequant`.

The served matmul path is coherent and the Paris text matches the
int3 graph serve. C1 ITL stays 11.34 ms. 8k wall time is 22.34 s.

One synthetic int3 layer at T=8192, not a serve number:

| Piece | Time | Times the model |
|---|---:|---:|
| expert w13 unpack | 12.93 ms | |
| expert layer, both unpacks and 256 matmuls | 144.64 ms | 5.79 s |
| one dense GEMM N=2048 | 0.655 ms | 0.10 s for 160 calls |

About 5.9 s of the 22.3 s is those lowbit GEMMs. The other 16 s is
outside them. Thirty of the forty layers are GDN linear attention and
ten are full attention. Stock int4 eager prefill in the brief is
about 6000-7800 tok/s, so this GPU can do the whole step much faster
than 16 s. This run did not time `linear_attn` and `self_attn`
inside a real forward.

Next prefill measurement: XPU events around `linear_attn`,
`self_attn`, and `mlp` on one real 8k forward. Not Kineto
(`PTI_ERROR_INTERNAL` on this stack). Do not report that instrumented
step as a steady ITL.

## 4. Decode floor

Graph C1 ITL is 10.23 ms (int4), 11.29 ms (int3), 10.63 ms (mixed).
Eager int3 was 51.7 ms. The graph removed the launch overhead.

Eager device events, not the graph ITL: `moe_gemv` 0.168 ms per layer
(6.72 ms times 40), `gemv` N=2048 0.098 ms (15.76 ms times 160). The
sum is larger than 11.3 ms because those events include launch
overhead that the captured graph collapses. A per-layer split of the
captured command list was not taken.

The int4 versus int3 gap says the remaining kernel time is the
unpack, not the extra int4 bytes. A faster `moe_gemv` is the next
decode change. Another launch-count change is not.

## 5. Perplexity

Same 4x512 wikitext windows, `tests/ppl_serve.py`, prompt logprobs.
2044 scored tokens.

| Arm | ppl | nll | window nll |
|---|---:|---:|---|
| int4 | 8.858 | 2.181 | 1.789, 2.441, 2.097, 2.399 |
| int3 | 9.120 | 2.211 | 1.813, 2.441, 2.215, 2.373 |
| mixed | 10.159 | 2.318 | 1.912, 2.540, 2.346, 2.476 |

int3 unpack reference on this slice is 9.19. The served value is
9.120. int4 and mixed were not run through the on-the-fly unpack.

## Verdict

Graph decode works. int3 C1 ITL is 11.29 ms, int4 is 10.23 ms, mixed
is 10.63 ms on a different kernel. Served perplexity moves with the
expert bits in the expected direction.

Prefill improved from 112 to 366 tok/s at 8k. The ESIMD tile and the
int8 tile are not the path to 5000. Vendor GEMM is 126x one dense
projection and the whole served prefill is still 13x short of the
target, with most of the 22 s outside the lowbit GEMMs that were
timed.
