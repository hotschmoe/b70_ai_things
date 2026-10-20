# 1E2 -- lowbit prefill: where the 8k time went

Plugin `vllm/flashnext/plugins/b70_lowbit_xpu`. Card 1. Checkpoint
`qwen36-w3a16-g128-experts-w4a16-trunk` (int3 experts, int4 trunk).
Library `libb70_lowbit_xpu_e2u.so` (not rebuilt this pass).
`FULL_DECODE_ONLY`, FlashAttention, `--gdn-prefill-backend sycl`,
bf16, TP=1, max-len 33792, util 0.90, max batched tokens 16384,
prefix cache off, `B70_LOWBIT_MOE_LAUNCHES=2`, `B70_LOWBIT_GEMV_ROWS`
unset. Served name `hotschmoe-dev` first, alias
`qwen36-w3a16-b70-lowbit`. MTP off.

The 8k target of 5000 tok/s was not met. The official bench, with
the diagnostic sync off, is 345 / 1961 / 3390 tok/s at 1k / 8k / 32k.
Decode C1 ITL p50 is 11.34 ms, in line with E1's 11.29 ms.

## Official bench

`bench_latency.py` tag `e2-int3-matmul-fdo`. Log
`/mnt/vm_8tb/b70/build/vllm-flashnext/logs/e2-bench.json`.
Mode file `matmul`: unpack every active expert to bf16 and call
`F.linear`. Dense linears take the same unpack plus `F.linear`.
Bench rc 0 at 11:54:23 on 2026-10-20.

Identity: `hotschmoe-dev` first, alias present. Coherence both
times: " Paris, a city renowned for its rich history, culture, and
iconic landmarks."

| | tokens | TTFT ms | tok/s |
| --- | ---: | ---: | ---: |
| 1k | 1017 | 2945 | 345 |
| 8k | 8185 | 4173 | 1961 |
| 32k | 32762 | 9664 | 3390 |

32k is two chunks at max batched tokens 16384. Time grew from 2.95 s
at 1k to 9.66 s at 32k while the token count grew 32x. The expert
loop is mostly launch overhead.

| | C1 ITL p50/p90/p99 ms | C1 tok/s | C2 ITL p50 ms |
| --- | --- | ---: | --- |
| this bench | 11.34 / 11.54 / 11.70 | 89.9 | 15.76 / 15.76 |
| E1 int3 graph | 11.29 / 11.52 / 11.97 | 90.3 | 15.65 / 15.65 |

## Where the old 22 s went

E1's served 8k step was 22.3 s at the default chunk of 2048, with
the expert loop about 5.8 s and dense GEMMs about 0.1 s on a
microbench. One chunk of 8186 tokens, prefix cache off, env
`B70_LOWBIT_PREFILL=bmm`, was still 31.3 s. XPU events on that
step (serve log split, diagnostic sync inside the forward):

| span | ms | count |
| --- | ---: | ---: |
| forward | 31263 | 1 |
| linear_attn | 19519 | 30 |
| self_attn | 5312 | 10 |
| experts (matmul file) | 6390 | 40 |
| shared_expert | 2272 | 80 |
| router | 4.4 | 40 |
| norm | 22 | 80 |

Host counters on that step: `item` 7, `tolist` 44, `synchronize` 1.
The ~25 s outside the expert loop was not a host sync. `linear.py`
took unpack plus `F.linear` only when the env was exactly `matmul`.
Env `bmm` sent every GDN projection, attention projection, and
shared expert through `prefill_wna16`. E1's 0.1 s dense figure was
the vendor path. The XPU GDN logger returns without a line, so the
backend was not identified from the log. Later serves pass
`--gdn-prefill-backend sycl` (`additional_config` shows that key).

After dense `matmul` and `bmm` both unpack and call `F.linear`, the
warm instrumented 8186-token step on the bench serve was:

| span | ms | count |
| --- | ---: | ---: |
| forward | 4479 | 1 |
| linear_attn | 243 | 30 |
| self_attn | 126 | 10 |
| experts | 4067 | 40 |
| shared_expert | 35 | 80 |
| router | 4.4 | 40 |
| norm | 22 | 80 |

`linear_attn` fell from 19.5 s to 0.24 s. `self_attn` fell from
5.3 s to 0.13 s. The shared expert fell from 2.3 s to 0.035 s.
The forward is the per-expert `F.linear` loop. The official 8k
number (1961 tok/s, sync off) matches this split (1814 tok/s with
the sync on).

D1's stock GPTQ int4 eager / FDO FlashAttention + SYCL GDN line is
6001/7888/6398 and 6030/7888/6414 tok/s at 1k/8k/32k
(`reports/1D1_overhead.md`). That is float16 GPTQ, not this
checkpoint. Our attention plus GDN span is 0.37 s. It does not
explain the gap to 7888 tok/s. The expert loop does.

## Expert bmm

Empty-card one-layer bmm (earlier E2 probe, full expert cube, no
KV cache) was 34 ms, which would be 1.36 s across 40 layers. A
single `torch.bmm` of all 256 experts raises Level Zero error 40
(`UR_RESULT_ERROR_OUT_OF_RESOURCES`) beside the KV cache. The
per-expert loop fits.

Tiled unpack plus tiled `torch.bmm` (32 experts, 256 MiB operand
cap) completed one 8186-token request at 769 tok/s and died on the
next. That step's expert spans summed to 9.9 s: group 4695 ms,
`bmm_w13` 1687, `bmm_w2` 1071, pad 453, unpack 524, reduce 315.
Slower than the 4.1 s loop. The death was `torch.bmm` of the down
tile, same error 40.

`torch.xpu.empty_cache` at the start of a large step made the next
first 8k request die in `_group_slots` at `torch.cat`, before any
bmm. That call was removed.

Reused scratch for the transposed tile and the bmm outputs still
died after the official matmul bench, at `y.index_add_` in
`expert_prefill_bmm`, error 40. The matmul bench leaves the
caching allocator at the 1.5 GB expert-unpack high water. The bmm
path does not allocate on top of that.

C10b's dequant plus oneDNN bmm is the same idea on a GGUF
checkpoint. Their journal (09:16, and C11 at 10:26) keeps fused
kernels at smaller T and bmm at T>=12288, with a GEMM-only budget
around 2887 tok/s. That budget is not this serve. This plugin's
served bmm has not produced a second successful 8k request.

## Decode

Eager `moe_gemv` at the 35B shape, `MOE_LAUNCHES=2`, rows=0
(`e2-bmm-probe.log`): int4 0.0724 ms, int3 0.0973 ms, int2 0.0663 ms.
int3 is slower than int4 by 0.025 ms. Times 40 layers that is 1.0 ms,
which matches E1's served ITL gap (11.29 vs 10.23). The int3 kernel
is unpack-bound, not byte-bound.

rows=2 on int4: max_abs 1.0 and 0.0748 ms vs 0.0645 ms for rows=0
(`e2-fast-probe.log`, `GEMV_MISMATCH`). Not shipped. `GEMV_ROWS`
stayed unset, and the served C1 ITL did not move.

Container `vllm-fn-lowbit-e2` was stopped after the bmm trial.
`xpu-health --card 1` then printed `card 1: OK` and `HEALTHY`,
rc 0 at 11:55:30. No xe-reset.

## Verdict

5000 tok/s at 8k was not met. Official 8k is 1961 tok/s. Official
32k is 3390 tok/s. The 16 s outside E1's expert loop was dense
`prefill_wna16`, and that part is fixed. What is left is a
256-launch expert loop of about 4.1 s. Grouped vendor bmm is
faster on an empty card and does not survive in this serve next
to the KV cache.
