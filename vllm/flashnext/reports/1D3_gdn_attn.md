# D3: GDN decode fusion and attention decode

Date: 2026-10-20. Card 1. Ornith-1.5-35B-A3B GPTQ int4 MixedCal-v2,
one card, dtype float16, max-model-len 33792. vLLM
0.32.1rc1.dev94+g5fd62688f, torch 2.14.0+xpu, image
vllm-flashnext-xpu:py312. Roof used below is 608 GB/s.

D2's FULL_DECODE_ONLY graph (FlashAttention, SYCL GDN) is 11.4 ms.
The split D3 was given: GDN module 4.79 ms (30 layers, 0.160 ms each,
including qkvz, ba, out, and the core), MoE 4.91 ms (E1/C9; left
alone), full attention 1.43 ms (10 layers, 0.143 ms, short KV),
norms 0.18 ms. D1 matched serve, no MTP: C1 ITL p50/p90/p99
12.81/12.99/13.12 ms, 78.23 tok/s, TTFT 88.59 ms; C2 ITL p50
13.75/13.75 ms, 73.40/72.96 tok/s. JSON `d1-bench-graph-fa.json`.

Claude consult: `claude --bare -p --model fable-5.1` and
`--model claude-5.5` both exited `unrecognized_model` and
"Not logged in". Not retried. Same result D1 and D2 recorded.

## Verdict

The non-GEMM GDN piece is launch-bound, not bandwidth-bound. Fusing
conv + gated delta + RMSNorm into one graph-capturable plugin op is
correct and faster in isolation (0.041 ms vs 0.064 ms at T=1, core
cosine 1). Inside the full model it moves C1 ITL p50 from 12.81 ms
to 12.60 ms (79.39 tok/s) and C2 p50 from 13.75 ms to 13.50/13.48 ms.
Identity and coherence passed. Prefill tok/s matches D1, so the
overlay stayed off the prefill path. Turn it on with `B70_D3_GDN=1`.
The flag stays default off.

Attention at short context is two fp16 GEMVs plus a 12 us
FlashAttention core. Auto split-k is already on and is the long-context
path (1.13 ms/layer at 200k, 363 GB/s). The custom paged kernel matches
FA to fp16 and is slower at every length. `B70_D3_ATTN` stays off.

MTP verify (spec T=3) matches stock in the microbench (16.1 us vs
17.8 us). The Shisa spec=2 serve with `B70_D3_GDN_SPEC=1` reproduced
D2: C1 ITL 18.67 vs 18.72 ms, 121.67 vs 121.73 tok/s, acceptance
113/170 on both. `B70_D3_GDN_SPEC` stays off.

## Byte model

Ornith GDN, per layer, T=1, non-spec. State dtype is the model field
`mamba_ssm_dtype=float32`.

| piece | shape | bytes |
| --- | --- | --- |
| conv state | fp16 [3, 8192] | 49152 |
| conv weight | fp16 [8192, 4] | 65536 |
| SSM state | fp32 [32, 128, 128] | 2097152 |
| SSM read+write | | 4194304 |
| qkvz activation | fp16 [12288] | 24576 |
| ba activation | fp16 [64] | 128 |
| core + norm out | fp16 2 x [32, 128] | 16384 |

T=1 traffic is 4399232 B (4.40 MB). Roof is 7.24 us. Thirty layers of
state read+write are 125.83 MB, roof 0.207 ms. The dominant move is
the 2 MiB SSM state, loaded once and stored once. Fusion removes a
second launch and the q/k/v/z round trip (about 24 KB). It does not
remove the state traffic.

D2's 0.160 ms/layer module is the projections. qkvz alone is 0.093 ms
at 540 GB/s, out_proj is 0.036 ms. The non-GEMM piece was hidden
inside that region.

qkvz is packed [q|k|v|z] = [2048|2048|4096|4096]. `reorder_input=True`:
the SYCL conv walks a virtual grouped space. Conv bias is None. The
conv cache layout is SD, (state_len, dim), state_len = kernel 4 - 1 = 3.
With MTP spec=2 the conv cache grows to 5 rows.

`bind_kv_cache` (`mamba/abstract.py`) slices conv and SSM out of one
padded int8 page. The views are strided. Inner dims stay packed.
`stride(0)` is the page gap. The kernel multiplies the slot index by
`stride(0)`.

Attention, read-once fp16 KV, GQA 2 KV heads x (K 256 + V 256) x 2 B
= 2048 B/token. Per full-attention layer: S=32 about 0.1 us, S=8192
about 27 us, S=32768 about 110 us, S=131072 about 442 us, S=200000
about 674 us. Ten layers at 200k are about 6.7 ms at the roof. The
probe's `roof_us` print on the attention sweep is milliseconds
mislabeled; the figures here are the corrected ones. The GDN time line
`roof_us 0.01` is the same mislabel (0.007 ms). The byte-model line in
`d3-probe-gdn.json` prints roof_us=7.24.

## GDN kernel

Plugin `vllm/flashnext/plugins/b70_gdn_xpu`, op
`torch.ops.b70_gdn.decode_fused`. One work-group of 1024 (subgroup 32)
owns one (sequence, value head): conv window, gated delta, and the
group RMSNorm. State stays in registers across the token loop. The
schedule follows `strata::kernels::fused_gdn_step_norm` in
`sycl/src/kernels/cuda/fused_gdn.dp.cpp` (Flash-Next / Strata
night-20261016, tree `/mnt/vm_8tb/b70/build/strata-ab/src-ep`). The
math is vLLM's: beta = sigmoid(b), g = exp(-exp(A_log) *
softplus(a+dt_bias)), L2 eps on the sum of squares, scale 1/sqrt(128)
on q only, RMSNorm-gated-silu with norm_before_gate. Strata's
sigmoid(z), native::exp, and the extra rsqrt(S) are not copied. The
ba GEMV stays on oneDNN (`in_proj_ba`, about 5.5 us). qkvz and
out_proj stay on oneDNN.

Non-spec and spec are separate templates. Spec stores the SSM state
after each token into `cache_indices[batch, t]` and rolls the conv
line. Graph capture: no alloc, no printf, no wait inside the op. The
wrapper caches the fp32 norm weight and the fp16 dt_bias before
capture. Pure decode leaves `has_initial_state` as None (`gdn_attn.py`,
the `num_prefills == 0` branch). Stock treats a null mask as "every
row has state". The wrapper passes a stable all-true buffer.

`.so`: `/mnt/vm_8tb/b70/build/vllm-flashnext/b70_gdn_xpu/libb70_gdn_xpu.so`,
322256 bytes, rebuilt 2026-10-20 09:23 (compile rc=0) so the host
check accepts strided mamba pages. Host strings `conv channel stride`
and `ssm head packed` are in that file.

Wrapper `harness/patches/d3_gdn.py`, installed from sitecustomize only
when `B70_D3_GDN=1`. It wraps `QwenGatedDeltaNetAttention.forward_xpu`,
skips `RMSNormGated` because the kernel writes the normed output, and
calls `out_proj`. Prefill and wrong geometry call the stock SYCL op.
Spec runs only when `B70_D3_GDN_SPEC=1` and `qkvz` rows equal
`slot_rows * slot_width`. Eager MTP warmup is ragged (slot width 3,
fewer query tokens). Launching the spec kernel on those batches
indexed past `core` and lost the device (`UR_RESULT_ERROR_DEVICE_LOST`,
log `d3-serve-shisa.log` first attempt, 09:29). The guard falls back
to stock on ragged batches. Uniform capture batches stay on the fused
op. Card 1 health after that stop was HEALTHY. No xe-reset.

## GDN microbench

CONFIG: card 1, XPU graph replay, `harness/d3_probe.py --only gdn`.
COMMAND: `B70_AGENT=D3 bin/cpu-run --quiet` around `xpu-run.sh --card 1`.
gpu-run 27 s, exit 0. JSON
`/mnt/vm_8tb/b70/build/vllm-flashnext/logs/d3-probe-gdn.json`.
The CONFIG lines that say `op gdn_attention False` are a `dir()` miss.
The calls succeeded. `logs/d3-probe.json` is the attention sweep; its
GDN section is the import failure from before `vllm_xpu_kernels._xpu_C`
was imported.

| path | core vs stock | conv state | SSM state | graph time |
| --- | --- | --- | --- | --- |
| T=1 non-spec | max_abs 7.63e-6, cos 1 | exact | max_abs 2.87e-5, cos 1 | stock 0.0640 ms, fused 0.0411 ms |
| T=3 non-spec | max_abs 7.63e-6, cos 1 | vs stock cos 0.152; vs CPU ref exact | max_abs 5.85e-5, cos 1 | stock 0.0170 ms, fused 0.0144 ms |
| spec T=3 | max_abs 7.63e-6, cos 1 | exact | max_abs 6.06e-5, cos 1 | stock 0.0178 ms, fused 0.0161 ms |

Post-norm vs stock RMSNorm max_abs is 3.906e-3 on every row. That is
one fp16 ulp at magnitude 4 (2^-8). Cosine is 1. Mean abs is about
6e-5. The core match says the subgroup-32 dot and the RMS reduce are
right. The out gap is the final fp16 store.

T=3 non-spec conv state disagrees with stock on purpose. With
`num_prefills==0` and T>1 the stock decode kernel does not write conv
state. The fused kernel does, and it matches the CPU reference
(max_abs 0, cos 0.999998). MTP verify does not use this path. vLLM
keeps `num_decodes` and `num_spec_decodes` mutually exclusive, so a
spec=2 verify batch is the spec kernel, which matched stock conv
state exactly.

Fused T=1 moves 4.38 MB in 0.0411 ms, 107 GB/s. Both sides are
launch-bound. The roof is 7 us. The 23 us win is the removed second
launch plus the norm, which stock's 0.064 ms does not include.

Thirty layers times (0.0640 - 0.0411) ms is 0.69 ms, about 6% of the
11.4 ms step, if the isolated gap survives inside the full graph.
The end-to-end row below shows about 0.20 ms of that. T=3 is already
17 us stock vs 14 us fused, so the spec path has little launch left
to remove. Thirty layers of the fused core are 1.23 ms. D2's 4.79 ms
module is still qkvz (30 x 0.100 ms) plus out_proj (30 x 0.036 ms)
plus ba.

## Attention decode

The 143 us/layer figure is the whole attention region. D2's split at
short KV is qkv 69 us + core 38 us + o 35 us. Two of those are fp16
GEMVs already near the roof (D2 floor: attn qkv 0.0708 ms at 533 GB/s,
attn o 0.0367 ms).

The installed serve path is
`vllm_xpu_kernels.flash_attn_interface.flash_attn_varlen_func`, the
long `varlen_fwd` schema. `num_splits` in Python is a different
parameter from C++ `num_splits_kv`. The serve forwards None, which
calls `get_num_splits` in
`vllm-xpu-kernels/csrc/flash_attn/flash_api.cpp`. It returns 1 when
the KV tile count is under 16, otherwise a parallelism-plus-bandwidth
choice capped at 32. Pure paged decode forces `is_causal=false` inside
the kernel because `seqused_k` already limits KV. T>1 causal paged
batches with `max_seqlen_q<=16` are rewritten into independent qlen=1
decodes (`_spec_decode_varlen_fwd`). That is the MTP verify path.

KV layout is LBNHC. After `kv_cache.transpose(1, 2).split`, K and V
are [blocks, block, kv_heads, head_dim] and strided. Probe: shape
(1, 64, 2, 256), stride (65536, 512, 32768, 1). Dim stride 1, token
stride 512, head stride 32768, page stride 65536. The serve sets
block size 64 for FlashAttention and then attention block size 1088
so the page matches the mamba page.

CONFIG: card 1, XPU graph, attention section of `harness/d3_probe.py`.
JSON `logs/d3-probe.json`. Custom op `b70_gdn.paged_decode` (same
.so): one subgroup of 32 per (sequence, KV head, split), online
softmax, strides honored. `fa_serve` is the real wrapper. `fa_splits1`
forces `num_splits_kv=1`. Cosine of ours vs FA is 1.000000 at every
length below, including S=1 (both exactly 0.25) and S=200000. Block
64 and 1088 agree at long S. Times below are T=1, block 64, graph
replay. Block 1088 is in parentheses when it differs.

| S | ours splits 1 / 8 / 32 | FA serve | FA splits=1 |
| --- | --- | --- | --- |
| 32 | 0.426 / 0.098 / 0.084 ms (1088: 0.100 / 0.025 / 0.028) | 0.0315 ms (1088: 0.0120) | 0.0353 (1088: 0.0119) |
| 8192 | 23.26 / 2.92 / 0.755 ms | 0.0715 ms, 234 GB/s | 0.241 ms, 70 GB/s |
| 32768 | 107.4 / 13.43 / 3.38 ms | 0.218 ms, 308 GB/s | 1.198 ms, 56 GB/s |
| 131072 | 429.9 / 53.7 / 13.46 ms | 0.765 ms, 351 GB/s | 5.05 ms, 53 GB/s |
| 200000 | 655.8 / 81.9 / 20.46 ms | 1.127 ms, 363 GB/s | 7.68 ms, 53 GB/s |

At S=32, `get_num_splits` returns 1 (kv tiles < 16). Auto and
splits=1 match. Isolated FA at the serve block size 1088 is 0.012 ms.
The 0.038 ms D2 called "attn core" is that 12 us plus other work in
the region (KV write, metadata). At 8k and above, auto split-k is
3.4x to 6.8x faster than forced splits=1 and reaches about 60% of the
608 GB/s roof at 128k-200k (roof at 200k is 0.67 ms; FA is 1.13 ms).

T=3 FA serve at S=200000 is 1.087 ms. It does not triple. The spec
rewrite runs three qlen=1 decodes that reuse the same pages, so DRAM
is about 1x. The custom kernel re-reads KV per query token. Its best
200k time is 20.5 ms, about 20 GB/s.

No Triton decode was run. FA's own split-k answers the launch-config
question, and the custom kernel is the slower correct implementation.
`B70_D3_ATTN` stays off. Ten layers at 200k are 10 x 1.13 ms = 11.3 ms
of attention core before the two GEMVs. Block 1088 is the better
short-context choice (0.012 vs 0.032 ms) and does not change long
context. The serve already uses 1088.

## MTP verify, microbench

Spec T=3, `num_spec_tokens=3`, conv rows 5, `num_accepted=[1]`,
identity `token_indx`. Core, conv line, and SSM slots match stock
(table above). 16.1 us vs 17.8 us. The non-spec flag does not cover
MTP verify, because `num_decodes` and `num_spec_decodes` are never
both positive. A spec=2 serve needs `B70_D3_GDN_SPEC=1` to hit this
kernel.

## End to end, no MTP

CONFIG: card 1, `B70_D3_GDN=1`, `B70_XPU_CG_RECLAIM=500`, `GRAPH=1`,
`GRAPH_MODE=FULL_DECODE_ONLY`, `ATTN=FLASH_ATTN`, `MTP=0`. Spec flag
off. Attention flag off. Container `vllm-fn-d3-gdn`, port 18081.
Log `logs/d3-serve-gdn3.log`. Captured decode graphs called the fused
op: `capturing=1 spec=0` for ntok=1 and ntok=2 on the real cache
`conv=(245, 3, 8192)`. No layout fallback and no exception fallback.
Prefill still takes the stock path (`skip path pre=4`). Two earlier
serves captured stock and were torn down before any bench: the first
treated `has_initial_state is None` as missing metadata; the second
rejected the strided page views. The `.so` rebuild and the wrapper
fix are what this row measures.

COMMAND: `bin/cpu-run --quiet python3 harness/bench_latency.py
--tag d3-gdn-fused --decode-tokens 64
--out logs/d3-bench-gdn.json`. cpu-run 11 s, exit 0.
`bin/xpu-health --card 1` after the later teardown: HEALTHY.

RESULT: identity first_ok alias_ok. Coherence ok, both temp-0 texts
the same Paris continuation.

| | C1 ITL p50/p90/p99 | C1 tok/s | TTFT | C2 ITL p50 | C2 tok/s | prefill 1k/8k/32k tok/s |
| --- | --- | --- | --- | --- | --- | --- |
| D1 stock | 12.81/12.99/13.12 | 78.23 | 88.59 | 13.75/13.75 | 73.40/72.96 | 6030/7888/6414 |
| D3 fused GDN | 12.60/12.83/13.07 | 79.39 | 94.75 | 13.50/13.48 | 74.82/74.36 | 6076/7883/6415 |

C1 p50 moved 0.20 ms. C2 p50 moved about 0.25 ms. Prefill matches D1.
TTFT on the 29-token bench prompt was 94.75 vs 88.59. That prompt's
prefill does not run the fused decode op.

## End to end, Shisa spec=2

CONFIG: same graph flags, `MTP=1`, `MTP_TOKENS=2`, `B70_D2_SHISA_MTP=1`,
`B70_D3_GDN=1`, `B70_D3_GDN_SPEC=1`. Container `vllm-fn-d3-shisa`.
Log `logs/d3-serve-shisa.log` (second attempt, after the ragged guard).
Real cache `conv=(147, 5, 8192)`. `capturing=1 spec=1` for ntok=3
slot=(1, 3) and ntok=6 slot=(2, 3). The ragged warmup `ntok=4
slot=(2, 3)` logged `skip spec ragged` and stayed on stock. That is
the shape that lost the device on the first attempt.

COMMAND: `bench_latency.py --tag d3-gdn-shisa-spec2 --decode-tokens 64
--skip-prefill --out logs/d3-bench-shisa.json`. cpu-run 2 s, exit 0.
Server stopped after the bench. Card 1 HEALTHY.

RESULT: identity and coherence passed (same Paris text). Acceptance
during the bench is 113/170 = 0.665, the same count as D2.

| | C1 ITL p50/p90/p99 | C1 tok/s | TTFT | C2 ITL p50 | C2 tok/s | acceptance |
| --- | --- | --- | --- | --- | --- | --- |
| D2 Shisa spec=2 | 18.72/18.97/19.26 | 121.73 | 115.12 | 20.93/20.99 | 110.85/123.42 | 113/170 |
| D3 fused spec | 18.67/19.15/19.41 | 121.67 | 109.76 | 20.96/21.02 | 110.92/123.44 | 113/170 |

C1 p50 differs by 0.05 ms. That is the size of 30 layers times the
1.7 us microbench gap, and it sits inside the run-to-run spread.
Effective ms/token is 8.22 on both rows (1000/121.7). The spec fusion
does not move the MTP chunk.

## What to leave off

- `B70_D3_ATTN`: custom paged decode is correct and slower than FA
  auto split-k at every context from 32 to 200k.
- `B70_D3_GDN_SPEC`: correct on uniform spec batches, no ITL change
  on Shisa spec=2, and ragged eager batches must keep the fallback.
- MoE: still E1/C9. 4.91 ms of the 11.4 ms step, 108 GB/s.
- qkvz and out_proj: already 89% of the 608 GB/s roof in D2. The
  fused op does not replace them.
