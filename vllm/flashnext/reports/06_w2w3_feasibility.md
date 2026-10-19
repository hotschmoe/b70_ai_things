# W2 / W3 feasibility for Flash-Next on vLLM-XPU

Task 06. CPU only. No GPU, no `hotschmoe-dd` change. Line numbers are the
vLLM tree at `/mnt/vm_8tb/b70/build/vllm-flashnext/vllm`, HEAD
`5fd62688ffe995c0c11c8708fe51dbff53235b74`, read 2026-10-19. Published
Flash-Next checkpoint inventory is report 05. This file is the W2/W3
quality, format, recipe, and kernel question.

Labels: **verified** means a file, card, or this run. **Inferred** means
arithmetic on those measurements, not a new serving number.

## Rank

1. Do not build W2A8 or W3A8. No export scheme, no XPU kernel, no DPAS
   type for 2-bit or 3-bit weights. Decode is weight-byte bound, so an
   int8 activation does not raise decode. Prefill is already an XMX stack
   at about 2200 tok/s (report 03). An int8 activation on the expert GEMM
   is a few percent of that token at most, and only after an unpack.
   Inferred, section 2.
2. Mixed or uniform W3A16 on the routed experts is the only packed format
   that is size-legal beside 2x200k BF16 KV and, on the two layers measured
   here, a better weight reconstruction than the IQ2 and Q2 tensors inside
   the local IQ3_XXS file. It loses to IQ4_NL on the one down projection
   ISTA promoted. Serving it needs a new XPU MoE kernel (L). Until that
   kernel exists, Strata plus the ISTA IQ3_XXS GGUF remains the quality and
   size bar: published task-average retention 99.4% (section 1).
3. Mixed W2A16 fits with room. It is the only large-MoE recipe with a
   published task table (DeepSeek-R1 int2-mixed, 97.9% on four zero-shot
   tasks). On these Flash-Next layers, round-to-nearest int2 is a worse
   reconstruction than ISTA IQ2_XS and IQ2_XXS. SignRound was not run.
   The XPU MoE kernel is still int4-only. Dense int2 is closer: the scheme
   sends it to ARK.
4. Uniform int4 does not fit (report 02). Dequantizing IQ3_XXS and
   requantizing spends the error twice and still has no int3 MoE kernel.
5. If a packed checkpoint is pursued, the kernel to write is a symmetric
   int3 MoE GEMV (decode), then a grouped GEMM (prefill). A full-body
   round-to-nearest int3 can be done on this host one layer at a time
   after the 360 GB BF16 download, with the PLE table skipped. Tuned
   AutoRound is a later CPU job. Do not start it as a serving plan while
   the MoE kernel is missing.

## 1. Published quality

### This model, ISTA GSQ-RCO, the IQ3_XXS bar

Primary card, fetched 2026-10-19:
https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF

The local daily-driver file is this IQ3_XXS build
(`/mnt/cache/b70/qwen3.8-flash-next/ista-gsq-rco/IQ3_XXS`). Task average
is the mean of AIME25, GPQA-Diamond, and LiveCodeBench v6. Zero-shot
average is ARC-E, ARC-C, HellaSwag, WinoGrande, PIQA. bpw is the
transformer weights. GB is the full download, including the n-gram shard.
The card's own caveat: zero-shot recoveries above 100%, and IQ3_S above
BF16 on GPQA, are benchmark variance.

| Variant | bpw | GB | ZS avg | recovery | AIME25 | GPQA-D | LCB v6 | Task avg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BF16 | 16.00 | 354 | 76.94 | 100.0% | 100.00 | 91.92 | 87.43 | 93.12 |
| GSQ-RCO Q2_0 | 2.40 | 66.4 | 78.00 | 101.4% | 96.67 | 89.39 | 81.14 | 89.07 |
| GSQ-RCO IQ2_XS | 2.50 | 68.0 | 77.16 | 100.3% | 96.67 | 87.37 | 83.43 | 89.16 |
| GSQ-RCO IQ3_XXS | 3.00 | 75.8 | 77.23 | 100.4% | 100.00 | 91.41 | 86.29 | 92.57 |
| GSQ-RCO IQ3_S | 3.50 | 83.6 | n/a | n/a | 100.00 | 92.93 | 86.86 | 93.26 |

IQ3_XXS keeps 92.57/93.12 = 99.4% of the task average. LCB is -1.14,
GPQA-Diamond is -0.51. IQ2_XS keeps 95.7%. Q2_0 holds AIME25 at 96.67.
BF16, IQ3_XXS, and IQ3_S all print AIME25 = 100.00, so that column does
not separate them. Papers: GSQ https://arxiv.org/abs/2604.18556 , RCO
https://arxiv.org/abs/2605.00649 .

Their llama.cpp timing (55 prompts, their GPU, not a B70 result): Q2_0
prompt 367.49 tok/s, decode 93.79, latency 6.70 s; IQ2_XS 108.19 / 70.30
/ 12.68 s. The card attributes the gap to lookup-table formats.

The search covers 352 tensors. `ffn_down_exps` can only be Q2_0 (block 64)
or IQ4_NL (block 32), because 640 is not divisible by 256. IQ3_XXS lifts
18 of 48 down layers to IQ4_NL. Shard 1 resident sizes on the card:
Q2_0 37.6, IQ2_XS 39.2, IQ3_XXS 47.0, IQ3_S 54.8 GB. The n-gram shard is
28.8 GB, IQ4_NL, identical across variants.

Report 02 measured the local IQ3_XXS tensor payload at 43.797 GiB and the
uniform packed estimates at int3 45.747 GiB, int2 31.108 GiB, int4 60.386
GiB (`02_model_support.md:320-327`). Int4 plus 2x200k BF16 KV (9.155 GiB)
does not fit in 64 GiB. Int3 and int2 do, with PLE left in host RAM.

### DeepSeek-R1 mixed int2, the large-MoE W2 recipe with a task table

Card, fetched 2026-10-19:
https://huggingface.co/OPEA/DeepSeek-R1-int2-mixed-sym-inc

AutoRound, bits=2, group 64, symmetric, iters=400, batch 4, seqlen 512,
nsamples 512, `enable_torch_compile=False`, format `auto_round`,
calibration `NeelNanda/pile-10k`. Fallback, from the generate script on
that card: non-expert Linear to 4-bit group 128; shared experts to 4-bit
group 128; first 3 layers all Linear to 4-bit group 128; layer 60
`down_proj` stays 16-bit; `lm_head` stays 16-bit. Routed experts otherwise
int2 group 64. The card says generation wanted 5x80 GB GPU and 1.4-1.6 TB
RAM, that int2 may be slower than int4 on CUDA because of the kernel, and
that the file could not run on other serving frameworks. Scores are CUDA
with overflow protection; the card says the CPU path should be more accurate.

| task | BF16 | INT2-mixed |
|---|---:|---:|
| mmlu | 0.8514 | 0.8302 |
| arc_challenge | 0.6212 | 0.6084 |
| hellaswag | 0.6935 | 0.6657 |
| winogrande | 0.7932 | 0.7940 |

Mean of the four: BF16 0.7398, INT2-mixed 0.7246, retention 97.9%. These
are zero-shot tasks, not AIME or LiveCodeBench. Paper cited on the card:
https://arxiv.org/abs/2309.05516 (SignRound).

### Other large MoE families, this pass

Qwen3-235B-A22B in the AngelSlim table has BF16, FP8, and INT8 rows
(BF16 CEVAL 89.60, MMLU 86.28, GSM8K 85.29, HumanEval 27.44) and no int2
or int3 row. The int4 rows in that table are the smaller dense Qwen3
models. https://angelslim.readthedocs.io/zh-cn/latest/performance/quantization/benchmarks.html

Qwen3 dense, not MoE, is a warning rather than a Flash-Next result.
https://arxiv.org/html/2505.02214v1 and the Springer copy
https://link.springer.com/article/10.1007/s44267-026-00114-4 : Qwen3-8B
MMLU 74.7 to 69.3 at 4-bit; at 3-bit AWQ keeps some accuracy and loses
most of the full-precision advantage; at 2-bit only calibration methods
such as GPTQ keep a floor. Same AWQ W3A16G128 setting: Llama3-8B C4 PPL
9.2 to 11.6, Qwen3-8B-Base 10.4 to 23.8. The ISTA Flash-Next table above
shows a searched mixed GGUF can still hold the 3-task average. A uniform
RTN int3 should not be assumed to do the same. The pilot below is weight
error, not those tasks.

Kimi-K2.5's published low-bit method is native INT4, the same method as
Kimi-K2-Thinking (https://huggingface.co/unsloth/Kimi-K2.5-GGUF). No
int2/int3 task table was found. GLM-4.5 community GGUF perplexity
(https://huggingface.co/ubergarm/GLM-4.5-GGUF): BF16 PPL 3.1788 +/- 0.01790,
IQ2_KS at 2.671 bpw PPL 4.3196 +/- 0.02652. That is a llama.cpp imatrix
quant, not a W3A16 kernel result. No GLM int2/int3 AutoRound or GPTQ task
table was found. QuIP# and AQLM remain dense codebook methods; no
large-MoE task table was found, and neither format is a vLLM-XPU kernel.
EXL3 was not found as an XPU path.

## 2. Which formats exist

| Format | Real scheme or kernel | Where | XPU MoE today |
|---|---|---|---|
| W4A16 | yes | oneDNN `int4_gemm_w4a16`, ARK MoE | yes, symmetric int4 |
| W4A8 | yes, linear | oneDNN `int4_gemm_w4a8`; falls back under 512 tokens | MoE path is still W4A16 |
| W8A16 / W8A8 | yes | `kernels/int8_gemm_w8a16.h`, `int8_gemm_w8a8.h` | int8 weights, not 2/3-bit |
| W2A16 | scheme yes, XPU linear via ARK, MoE no | AutoRound export; INC parser | no |
| W3A16 | scheme yes on CUDA humming and in the INC parser; XPU kernel no | AutoRound export; humming on CUDA | no |
| W2A8 | no scheme string found |  | no |
| W3A8 | no scheme string found |  | no |

Verified in this tree:

- INC accepts bits `{2,3,4,5,6,7,8}` (`inc/inc.py:37`).
- XPU WNA16 linear bits are `{2, 4}` (`inc/schemes/inc_wna16_scheme.py:21`).
  Bits 3 raise `NotImplementedError` on XPU (`:116`).
- oneDNN backends are `("w4a16", "w4a8")` (`:29-30`). Bits other than 4
  raise (`:81-86`).
- Int2 linear on XPU requires ARK. If `auto_round_kernel` is missing, bits
  2 raise (`:101-107`). `envs.py:2207-2216`: "The two oneDNN backends are
  int4-only; ARK also serves int2." This task did not import ARK (no GPU,
  no serving venv).
- XPU MoE requires GPTQ, bits == 4, symmetric (`inc_wna16_scheme.py:161-166`).
  `INCARKWNA16MoEMethod` raises unless `weight_bits == 4` and there is no
  zero point (`inc/schemes/inc_wna16_moe.py:38-40`).
- CUDA humming covers linear and MoE at bits `{2, 3, 5, 6, 7}`
  (`inc_wna16_scheme.py:26`, `:139-146`, `:211-216`). That is CUDA.
- Compressed-tensors WNA16 maps bits 2 through 8, including `uint3b4`
  (`compressed_tensors/schemes/compressed_tensors_wNa16.py:41-58`).
  `pack_factor = Fraction(32, num_bits)`, so int3 is not an integer pack.
  CUDA, not XPU.
- `vllm-xpu-kernels` bindings in
  `/mnt/vm_8tb/b70/vllm-xpu-kernels-nvfp4-v028/csrc/xpu/torch_bindings.cpp:48`
  and `:66` are `int4_gemm_w4a16` and `int4_gemm_w4a8`. A search of that
  `csrc` tree found no int2 or int3 GEMM symbol.
- `kernels/README.md:35-44`: Xe2/Battlemage XMX is INT8 and INT4 DPAS.
  No native FP8 or FP4. Repo kernels are int8 W8A16 (decode) and W8A8
  (prefill), plus upstream int4. `nvfp4_gemm_w4a16` is a decompress, not
  an FP4 tensor core.

AutoRound 0.16 export table, README fetched 2026-10-19
(https://github.com/intel/auto-round/blob/main/README.md):

- `auto_round` and `auto_gptq`: W4A16, W2A16, W3A16, W8A16, W2A16G64,
  W2A16G32. Gray rows in that README have no kernel or only a reference
  kernel.
- `auto_awq`: W4A16.
- `llm_compressor`: NVFP4, MXFP4, MXFP8, FP8, INT8, W4A16, W8A16. No W2,
  no W3, no W2A8, no W3A8. GPTQ-through-llm-compressor is not a W2/W3 path.
  The W2/W3 path is AutoRound `auto_gptq` or `auto_round`.
- Model-free RTN (no full model load) presets are W2A16, W2A16G32,
  W2A16G64, W4A16, W8A16. W3A16 is not in that list
  (https://github.com/intel/auto-round/blob/main/docs/step_by_step.md).

### What A8 would buy on this host

Decode. JOURNAL 2026-08-21b (`JOURNAL.md:232-236`), Qwen3.8-27B shapes,
not Flash-Next serving: M=1 W4A16 hit 552-565 GB/s, 95-97% of 581 GB/s,
and M=1 W4A8 tied W4A16. An int8 activation does not shrink the weight
read. Active routed weights are 2.359e9 params/token
(`02_model_support.md:82`). At 3 bits that is 0.885 GB. At 581 GB/s the
expert-only GEMV roof is about 650 tok/s on one card. Daily-driver C1 is
141.4 tok/s (report 03). Report 03 row 4: IQ expert decode was
arithmetic-bound, because the IQ grids are lookups. A packed int3 GEMV
would look more like the 27B W4A16 bandwidth case than like that IQ
decode. That speed is unmeasured. Do not quote 650 tok/s as a serving
result.

Prefill. The ~2200 tok/s number is already an XMX stack: grouped XMX
expert prefill, XMX QSA attention, then the dequant stack
(`03_strata_inventory.md:59-68`). 2200 tok/s is 455 us/token. Routed
expert math is about 4.72e9 FLOPs/token. JOURNAL Path S s8 was 208 TOPS
on a 27B microbench (`JOURNAL.md:239`); at that rate the expert math is
~23 us, about 5% of 455 us. The same journal's M=2048 gate_up numbers
were w8a8 260 TOPS, w4a8 197, w4a16 135, so an int8 activation does move
large-M TOPS on int4/int8 weights. int2 and int3 have no DPAS type, so a
W3A8 or W2A8 kernel would unpack to int4 or int8 first. S74
(`JOURNAL.md:23951-23969`) compiled one signed int8 DPAS and did not run
it. S75 (`JOURNAL.md:23973-23990`) rejected one canonical-Q8 mapping of
IQ3_S: gate median 70.8 us to 557.9 us. That rejects that mapping. It is
not a measurement that every int8 DPAS is slow.

Inferred: A8 buys no decode. On this prefill it can only speed the expert
GEMM, which is already on XMX and is a small slice of the 455 us. A
generous reading is well under 15% of prefill, and the 32K expert-kernel
delta in report 03 was +30 tok/s (1708 to 1738) on an earlier stack.
W3A8 and W2A8 are not worth a kernel.

## 3. Recipe on this host, and the pilot

Host: 121 GB RAM, 32 cores, 2x B70 32 GB (both held by the daily driver;
this task did not touch them), NVMe `/mnt/cache` with about 600 GB free
before the download.

### Source

`Qwen/Qwen3.8-Flash-Next` is public. The index
(`/mnt/cache/b70/w2w3-pilot/index-only/model.safetensors.index.json`,
170726 bytes) lists 131 safetensors shards. The card's BF16 line is 354
GB; report 02's element math is 329.58 GiB / 353.89 decimal GB including
PLE. The FP8 sibling is the wrong source.

Expert tensors are fused, one per layer, not one per expert:

- `model.language_model.layers.N.mlp.experts.gate_up_proj` BF16
  `[512, 1280, 2560]` (3.125 GiB of weights)
- `model.language_model.layers.N.mlp.experts.down_proj` BF16
  `[512, 2560, 640]` (1.5625 GiB)

Gate is the first 640 of the middle axis, up is the second 640. Verified
by cosine against the GGUF dequant (section below). Layer 0 gate/up is
the whole of `model-00002`. Layer 0 down shares `model-00003` with the
router, shared expert, and hyper-connection weights. Do not
`from_pretrained` the repo. The PLE table is 51.2B params, 95.368 GiB in
BF16 (report 02). It OOMs a naive load inside 121 GB. Skip it and keep
the existing IQ4_NL host table. MTP experts live in later shards; the
extracted MTP BF16 under the strata build is a separate file. Leave MTP
and the vision tower out of the first quant.

### How to quantize

Devices AutoRound documents: CPU, Intel XPU, CUDA, Gaudi
(https://github.com/intel/auto-round). This host has no CUDA GPU. XPU
quantization would need `bin/gpu-run` after the daily driver is stopped.
Not done here. CPU AutoRound is the path that fits the current lease.

`low_cpu_mem_usage` (immediate save) is the default. Turn it off only
with `--disable_low_cpu_mem_usage`. `low_gpu_mem_usage` is irrelevant on
the CPU path.

W2A16 can use model-free RTN: stream a shard, quantize, write, delete.
Peak is a few GB. W3A16 cannot. It needs the calibration AutoRound path,
one layer resident at a time. SignRound keeps a V per weight. One expert
layer is 4.69 GiB BF16; an fp32 V is about 9.4 GiB. That fits. All 48
layers plus V do not. AutoScheme's overhead (about 1.1x-1.5x BF16 RAM, per
the AutoRound README) does not fit this host for the whole model.

Suggested layer config, copied from the R1 card and aimed at the ISTA
allocation: attention, shared expert, hyper-connection, embeddings, and
the first few layers at 4-bit group 128 or left BF16; `lm_head` BF16;
routed gate/up at int3 group 128; routed down at int4 group 128 on the
layers where ISTA used IQ4_NL, int3 elsewhere. Uniform int3 of every
shard-1 element is the 45.747 GiB estimate. Promoting all 48 downs from
int3 to int4 adds about 4.7 GiB (one extra bit on 1/3 of the expert
elements) and gets tight next to 9.155 GiB of KV. Promoting only the 18
IQ4_NL downs adds about 1.8 GiB. Inferred from report 02's element
counts, not a packed file.

Calibration: `NeelNanda/pile-10k`. The R1 run used nsamples 512, seqlen
512, batch 4, iters 400, `enable_torch_compile=False`. Default AutoRound
is nsamples 128, seqlen 2048. With 512 experts and top-10, 128 sequences
touch on the order of a thousand expert slots per layer. Cold experts
fall back toward RTN. Prefer the larger sample count, and quantize one
layer at a time so the activations fit.

Wall time. The four-shard fetch below moved 11.082 GiB in 237 s,
unauthenticated, about 48 MiB/s. At that rate the 360 GB BF16 is about
two hours. Inferred from one transfer. Round-to-nearest of six expert
matrices (two layers, all 512 experts) took 557 s under `bin/cpu-run`
heavy (`nice -19`). A 48-layer expert RTN is a few hours at that pace.
Tuned AutoRound at iters 400 was not timed. The R1 card does not publish
a CPU hour count, and this host is not 1.4 TB. Do not invent one. Expect
hours to days for a tuned 48-layer CPU run, dominated by calibration
forwards, and plan to delete each BF16 shard after its layer is saved.

Risks. The ISTA card says the quantized weights inherit the base-model
license (the card YAML says apache-2.0). Do not redistribute. A checkpoint
does not unlock vLLM-XPU serving while MoE is int4-only. Int2 on CUDA is
slower than int4 in the R1 card's own note. RTN at 2-bit is the weak
floor (pilot, and the Qwen3 dense paper). Loading PLE in BF16 OOMs.
Quantizing from the IQ3 dequant instead of BF16 spends the error twice.

### Pilot

CONFIG -> official BF16 shards of layers 0 and 1 (11.082 GiB) against
ISTA IQ3_XXS shard 1. CPU, `/usr/bin/python3` 3.14, numpy, safetensors,
no torch. gguf-py from
`/mnt/vm_8tb/b70/build/strata-ab/llama-pinned/gguf-py`. RTN is AutoRound
`quant_tensor_rtn_sym` (https://github.com/intel/auto-round/blob/main/auto_round/data_type/int.py):
`maxq = 2 ** (bits - 1)`, codes in `[-maxq, maxq - 1]`, fp16 group scale,
group along the last axis (K). int3 group 128 uses 7 levels `[-4, 3]`.
int2 uses 4 levels `[-2, 1]`, the full signed range, not the 3-level
`[-1, 1]` grid. This is iters=0 RTN, not SignRound.

IQ types in the file: layer 0 gate/up IQ2_XS, down IQ4_NL; layer 1
gate/up IQ2_XXS, down Q2_0. gguf-py has no Q2_0 dequant. The local
decoder matches this tree's Q2_0: block 64, 18 bytes, fp16 scale, four
2-bit codes per byte, low bits first, value `(code - 1) * d` so the grid
is `{-d, 0, d, 2d}` (`gguf/constants.py:5713` and `:5906`;
`src-ep/tools/gguf_writer.py:64`; `q2_spread4` in
`src-ep/sycl/src/kernels/cuda/iq_kernels.dp.cpp`). A one-block self-check
passed.

Layout, four experts, cosine vs BF16: gate = `gate_up[:, :640, :]`
0.9449; the swapped half 0.0014; up = `gate_up[:, 640:, :]` 0.9466; down
direct 0.9927. Shapes already matched `(512, 640, 2560)` and
`(512, 2560, 640)`. No transpose.

COMMAND ->

```
B70_AGENT=w2w3-pilot B70_CPU_LOCK_TIMEOUT=0 \
  bin/cpu-run python3 -   # hf_hub_download of four shards into
                          # /mnt/cache/b70/w2w3-pilot/bf16
B70_AGENT=w2w3-pilot B70_CPU_LOCK_TIMEOUT=0 \
  bin/cpu-run /usr/bin/python3 /mnt/cache/b70/w2w3-pilot/rtn_vs_iq.py
```

The first quant run (480 s) finished five matrices and raised
`NotImplementedError: Dequantization for Q2_0`. The second run (557 s,
exit 0), after the Q2_0 decoder, recomputed all six and matched the first
five to 6 decimals. Script and JSON:
`/mnt/cache/b70/w2w3-pilot/rtn_vs_iq.py`,
`/mnt/cache/b70/w2w3-pilot/rtn_vs_iq_result.json`.
Shard bytes: 3355443368, 1832518264, 3355443368, 3355888504.

RESULT -> relative L2 `||recon - BF16|| / ||BF16||`, all 512 experts,
838860800 elements each. Expert p50 is within 0.01 of the tensor value
on every row, so one bad expert is not driving the mean.

| Layer | Mat | GGUF | IQ vs BF16 | RTN W3 g128 | RTN W2 g128 | RTN W2 g64 |
|---|---|---|---:|---:|---:|---:|
| 0 | gate | IQ2_XS | 0.352 | 0.237 | 0.471 | 0.428 |
| 0 | up | IQ2_XS | 0.329 | 0.231 | 0.462 | 0.422 |
| 0 | down | IQ4_NL | 0.129 | 0.216 | 0.437 | 0.403 |
| 1 | gate | IQ2_XXS | 0.392 | 0.224 | 0.451 | 0.413 |
| 1 | up | IQ2_XXS | 0.365 | 0.220 | 0.445 | 0.409 |
| 1 | down | Q2_0 | 0.443 | 0.211 | 0.429 | 0.398 |
| both | pooled |  | 0.350 | 0.224 | 0.450 | 0.413 |

Cosine vs BF16, same order: IQ 0.943 / 0.944 / 0.992 / 0.929 / 0.931 /
0.902. RTN W3 g128 cosine is 0.973 to 0.978 on every matrix. Pooled
relative L2 of each RTN against the IQ dequant is 0.420 (W3 g128), 0.556
(W2 g128), 0.530 (W2 g64): the two quantizers miss BF16 in different
directions. Peak RSS was not logged. The process completed under the
heavy CPU lease on this host.

VERDICT -> On these two layers, untuned symmetric int3 group 128 already
beats IQ2_XS, IQ2_XXS, and Q2_0 on weight error, and loses to IQ4_NL
(0.216 vs 0.129 on layer-0 down). Untuned int2 is worse than the IQ2
grids and only slightly better than Q2_0 on the one Q2_0 down; group 64
beats group 128 and still does not catch IQ2_XS. SignRound would be
expected to beat this RTN. It was not run, and weight L2 is not AIME /
GPQA / LCB. Two early layers are not the whole model. The result is
enough to prefer int3 over int2 as the packed experiment, and enough to
keep IQ4-class precision on the down projections ISTA already promoted.

## 4. Kernel work, vLLM-XPU

S = a thin wiring of a kernel that already exists. M = a new linear
kernel or a serious packing path, no model-runner rewrite. L = a new MoE
kernel plus the fused-expert integration. 96% of shard-1 elements are
routed experts (report 02), so the MoE row is the one that matters.

| Format | Dense decode | Dense prefill | MoE decode | MoE prefill |
|---|---|---|---|---|
| W2A16 | S/M. Scheme calls ARK for symmetric int2 linear. oneDNN cannot. Confirm ARK imports before calling it S. | M. Same kernel if ARK GEMM covers large M; otherwise a dequant fallback. | L. `get_moe_method` and ARK MoE are int4-only. Power-of-two pack. This is the decode kernel. | L. Grouped GEMM, same pack. |
| W3A16 | M. Parser accepts bits 3. XPU scheme rejects it. CUDA already uses humming. Need an unpack-to-BF16 path or a real kernel. | M. Same, large M. | L. No kernel, and int3 is not a DPAS type. Pack is `Fraction(32, 3)`. Harder than int2. | L. Grouped GEMM after the same pack. |
| W2A8 | L to invent, low value. No format, no oneDNN op. | L, low value. | L, low value. | L, low value. |
| W3A8 | L, low value. | L, low value. Unpack, then the existing int8 or int4 DPAS. | L, low value. | L, low value. |

W4A16 MoE is the kernel that already exists and the byte budget that does
not. W3A16 is the byte budget that fits and the kernel that does not.

## What this run did not do

No GPU, no serve, no task-score eval, no SignRound, no full-model
download, no ARK import check, no peak-RSS log. Layers 2-47 are
unmeasured. Report 05 owns the published-checkpoint survey; this file
does not repeat it.
