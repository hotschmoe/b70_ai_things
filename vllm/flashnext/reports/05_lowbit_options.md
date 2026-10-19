# 05 -- sub-4-bit vLLM-native weight options

Date: 2026-10-19. CPU only. No GPU. No weight download.

## Verdict

Keep the local ISTA IQ3_XXS GGUF as the size and quality bar (43.797 GiB
shard-1 payload, 26.822 GiB PLE in host RAM; report 02). Do not start an
AutoRound job, and do not download a 4-bit, MXFP4, or NVFP4 checkpoint for
the 2x32 GiB target.

No published checkpoint that vLLM can load at or under ~3.3 bits/weight is
runnable on this XPU tree. The one file whose language-model tensors sit
under the 46 GiB bar (Haberstroh int2-mixed, 36.144 GiB of safetensors) is
2-bit on every routed expert. This tree's XPU MoE kernels are symmetric
int4 and MXFP4 only. int2 linear exists only through the ARK backend, and
`auto_round_kernel` is not installed. int3 has no XPU linear or MoE kernel.

The artifact to build later, after an XPU int2 MoE kernel exists, is int4
on the non-routed shard-1 pool and int2 on the routed experts, PLE left in
host RAM. That layout is 32.259 GiB. It matches the only Intel-shaped
recipe that fits. The strata expert profile cannot assign those bits: it
is a residency ranking spread across all 48 layers, not a hit-rate curve,
and the fused MoE kernel takes one scheme per layer.

`Qwen4Exp` still refuses XPU (`vllm/models/qwen4_exp/__init__.py:30-31`).
That port is a separate blocker. A fitting weight file does not remove it.

## How this was checked

vLLM tree `/mnt/vm_8tb/b70/build/vllm-flashnext/vllm` at
`5fd62688ffe995c0c11c8708fe51dbff53235b74` (2026-10-19 14:37 -0700).
vllm-xpu-kernels at `19c6ddca8f6ef7a0f1de1bce1bc4dea9505cf8d4` (2026-10-16).
Line numbers below are that tree.

Hugging Face metadata: `GET /api/models/{repo}?blobs=true`, then
`config.json` when that sibling was at most 12 MB, plus README text for
the shortlist. Sibling `size` fields are the byte evidence. `usedStorage`
is recorded only where it disagrees. No LFS payload was fetched.

Targeted API searches on 2026-10-19, all HTTP 200:

| Query | Hits |
|---|---|
| `Qwen3.8-Flash-Next W2A16` | none |
| `Qwen3.8-Flash-Next W3A16` | none |
| `Qwen3.8-Flash-Next AQLM` | none |
| `Qwen3.8-Flash-Next QuIP` | none |
| `Qwen3.8-Flash-Next int2` | HaberstrohSystems int2-mixed only |
| `Qwen3.8-Flash-Next int3` | SecureAI-SE and codelight-ai NVFP4-PLE-INT3 only (experts stay NVFP4) |
| `Qwen3.8-Flash-Next HQQ` | spectator2026 W6/INT8 only |
| `author:Qwen` search `Flash` | `Qwen/Qwen3.8-Flash-Next`, `Qwen/Qwen3.8-Flash-Next-FP8` |
| `author:Intel` search `Flash-Next` | `Intel/Qwen3.8-Flash-Next-W4A16-AutoRound` |
| `author:intel-ai` search `Flash-Next` | seven MXFP4/MXFP8 repos, no int2/int3 |
| `hipfire Qwen3.8` | `hipfire-models/qwen3.8-flash-next` (library `hipfire`, not vLLM) |

Every `config.json` parsed here has `model_type: qwen4_exp` and
`Qwen4ExpForConditionalGeneration`, 48 layers, hidden 2560, 512 experts,
top-10, expert intermediate 640. The exception is azampatti, which sets
`num_experts_per_tok` to 5. No second official name turned up a low-bit
checkpoint.

Host at survey time, from `/proc/meminfo`: MemTotal 127317900 kB
(121.42 GiB), MemAvailable 76691872 kB (73.14 GiB), SwapTotal 8 GiB.
Disk free: `/mnt/vm_8tb` 6229028229120 bytes (5801 GiB), `/mnt/cache`
632482041856 bytes (589 GiB). `importlib.util.find_spec` is None for
`auto_round` and `auto_round_kernel`. Those modules were not imported.

Local weights are still the GGUF set in report 02. A depth-3 search for
`*.safetensors` and `config.json` under `/mnt/cache/b70/qwen3.8-flash-next`,
`models/files/qwen3.8-flash-next`, and `/mnt/vm_8tb/b70/models` returned
nothing.

## Size bar

From report 02, which read the GGUF headers. Shard-1 reader total is
125,743,653,760 elements. Routed experts are 120,795,955,200 (48 layers x
512 x 3 x 2560 x 640). The non-routed remainder is 4,947,698,560. Report
02 also prints a shared-expert row of 235,929,600. That row sits inside
the remainder: adding all three printed pools overshoots the reader total
by exactly 235,929,600. Layouts below quantize routed and non-routed as
two pools, so the shared expert is counted once, inside the non-routed
pool.

PLE is 51,200,245,760 elements. IQ4_NL stores it in 26.822 GiB. BF16 would
be 95.368 GiB. Budgets leave it in host RAM. 2x200k BF16 KV on the 12 full
layers is 9.155 GiB (report 02). 2x32 GiB is 64 GiB. The campaign bar for
routed plus dense weights is ~46 GiB, which is 3.142 stored bits/element
on the shard-1 total, scales included. ISTA's 47,028,835,840 tensor bytes
are 2.992 stored bits/element on that same total.

Uniform group-128 with FP16 scales, recomputed from those element counts
(report 02 prints 60.386 / 45.747 / 31.108; the 0.002 GiB gap is rounding
of the parts):

| Layout | Bytes | + 9.155 GiB KV | Under 46 GiB weights? |
|---|---:|---:|---|
| Uniform int4 g128 | 60.385 GiB | 69.54 GiB | No. Experts alone are 58.008 GiB. |
| Uniform int3 g128 | 45.745 GiB | 54.90 GiB | Yes, by 0.3 GiB. No XPU kernel. |
| Uniform int2 g128 | 31.107 GiB | 40.26 GiB | Yes. No XPU MoE kernel. |
| Non-routed int4 g128 + routed int3 g128 | 46.321 GiB | 55.48 GiB | 0.3 GiB over the bar. No XPU int3 MoE kernel. |
| Non-routed int4 g128 + routed int2 g128 | 32.259 GiB | 41.41 GiB | Yes. This is the future file. |
| Non-routed int8 g128 + routed int2 g128 | 34.563 GiB | 43.72 GiB | Yes. Same kernel gap. |
| Non-routed BF16 + routed int2 g128 | 39.099 GiB | 48.25 GiB | Yes. Same kernel gap. |
| Non-routed int4 + routed int3 g64 (bf16 scales) | 48.079 GiB | 57.23 GiB | No. |
| MXFP4 routed experts only (4-bit + 1-byte scale / 32) | 59.766 GiB | over | No, before the dense tail. |
| NVFP4 routed experts only (4-bit + 1-byte scale / 16) | 63.281 GiB | over | No, before the dense tail. |

A raw 3.3 bits/element with no scales is already 48.307 GiB. Any 4-bit
expert format, integer or float, misses the bar before scales and before
the dense tail. The fraction of routed experts that can stay int4 g128,
with the rest int2 g128 and the non-routed pool int4 g128, and still land
at 46 GiB, is 0.49. That split is not a stock fused-MoE layout (one
scheme per layer).

## Published checkpoints

Body is safetensors bytes with one obvious PLE file removed, when the
file is a single shard or `ple/*.bin` whose size matches the BF16 or FP8
table (102,400,491,520 or 51,200,245,760, plus a small header). Other rows
are the full safetensors sum. "vLLM method" is the `quantization_config`
seen in `config.json`. "XPU" is dispatch on this tree, not a load test.

### At or under the bar, or built to be

| Repo | Safetensors or noted file | Body | PLE side file | Method | XPU on this tree |
|---|---:|---:|---:|---|---|
| ISTA IQ3_XXS (local, report 02) | 47028835840 tensor bytes | 43.797 GiB | 28800138240 (26.822 GiB, IQ4_NL) | GGUF, not in this vLLM tree | No IQ/XPU kernels (report 02) |
| HaberstrohSystems/...-int2-mixed-AutoRound-24GB-SGLang | 38809545259 | 36.144 GiB | `ple/ple.f8_e4m3.bin` 51200245760 (47.684 GiB) | `auto-round`, bits 2, g128, sym, `packing_format=auto_round:auto_gptq` | int2 MoE rejected. Card says stock vLLM will not load it. |
| klee100/...-AutoRound-3bpw-MTP | 153036884475 | 47.159 GiB after removing shard 1 | shard 1 is 102400512288 (95.368 GiB, BF16 table) | `auto-round`, default bits 2 g64; 36888 extra entries at 3 | int2 and int3 MoE rejected. Card requires a private patch. |
| HamboneLabs-AI/...-uint3-g64 | 55335958816 (51.536 GiB) | that is the body | `ple/qwen38-flash-next.ngram.bf16.bin` 102400491520 | `modelopt` on the dense tail; experts are uint3 g64 files | NVFP4 has no XPU kernel. Card ships 8 patched vLLM files. |
| EliovpAI/...-W3A8-Paiton-RDNA4 | repo sum 120897167562 | card: LM resident on 2x32 GB | filename role `ple` sums to 52480294912 | no `quantization_config`; `library_name: paiton` | Not a vLLM quant method. |
| r0b0tlab/...-EXL3-2.50bpw | 64452230773 | 41.548 GiB after `ngram_embedding.safetensors` 19840121928 | that n-gram file, card says 3 bpw | `quant_method: exl3`, bits 2.52 | No exl3 loader. |
| hipfire-models/qwen3.8-flash-next `qwen3.8-flash-next-base.mq2-xts` | one file 46895167000 (43.675 GiB) | that file | separate `ple-q4` 29120146432, `ple-q8` 54400269312 | library `hipfire` | No loader. |
| hipfire `qwen3.8-flash-next-base.mq3-xts` | one file 59109362200 (55.050 GiB) | that file | same PLE sidecars | library `hipfire` | No loader. Over the bar. |

Haberstroh's card text says the repo ships 38,755,352,600 bytes of
language-model tensors. The API safetensors sum is 38,809,545,259
(54,192,659 bytes higher). Both are 36.1 GiB. The card's 2.572 bpw figure
is the AutoRound output before a later INT8 re-pack of `lm_head` and 84
dense projections.

klee100's card says ~47.3 GiB resident and a 95.37 GiB BF16 PLE table on
SSD. The API split is 47.159 GiB plus a 95.368 GiB shard. That body is
1.16 GiB over 46 GiB. HF `usedStorage` for this repo is 306201779694,
about 2x the sibling sum; the sibling sizes are what a download would
move. HF's parameter total (53.9B) does not match the tensor census and
is not used.

### 4-bit and float references (all miss 46 GiB)

| Repo | Safetensors bytes | GiB | What the config says |
|---|---:|---:|---|
| Qwen/Qwen3.8-Flash-Next | 360000363614 | 335.276 (360.000 decimal GB) | Unquantized BF16. Largest shard 3510240000. |
| Qwen/Qwen3.8-Flash-Next-FP8 | 185540727598 | 172.798 | `quant_method: fp8`. 943 ignored modules. |
| Intel/Qwen3.8-Flash-Next-W4A16-AutoRound | 181219801476 | 168.774 | `auto-round` bits 4 g128 sym. PLE shard 102400512256. Body 73.406 GiB. Report 02's earlier header fetch was 181196007016. |
| intel-ai/...-MXFP4-Mixed-CT-AutoRound | 179781189951 | 167.434 | compressed-tensors mixed: 73728 expert targets at float 4-bit g32, other Linear at float 8-bit g32. Card label 168 GiB. PLE left BF16. |
| intel-ai/...-MXFP4-BF16SharedExperts-CT-AutoRound | 180028753243 | 167.665 | Same, shared expert ignored (BF16). |
| intel-ai/...-MXFP4-BF16Attn-CT-AutoRound | 182619025915 | 170.077 | MXFP4 on Linear, attention in the ignore list. |
| intel-ai/...-MXFP8-CT-AutoRound | 240407189565 | 223.897 | float 8-bit g32. |
| INCModel3/...-MXFP4-Mixed-CT-RTN-AutoRound | 180009710061 | 167.647 | Same mixed MXFP4/MXFP8 pattern, model-free recipe in the card. |
| devan-carlin/...-W4A16 | 77418311322 | 72.101 | compressed-tensors int4 g128 on Linear. PLE is `ple_table_qwen4exp.pt` at 102400493290, not in the safetensors sum. Attention, shared, PLE, MTP, GDN are ignored. |
| wtdcode/...-AWQ-W4A16 | 180750853530 | 168.337 total, body 72.969 after the 102400512288 PLE shard | compressed-tensors int4 g128. Ignore list is attn, shared, PLE, GDN, MTP, vision. Experts are the quantized set. |
| nvidia/Qwen3.8-Flash-Next-NVFP4 | 132711524896 | 123.597 total, 73.569 after `model-fp8-mtp-ple.safetensors` 53717551730 | modelopt, experts float 4-bit group 16. |
| nvidia/Qwen3.8-Flash-Next-FP4-Local | 102239292690 | 95.218 | modelopt. No separate PLE name. Still over 64 GiB before KV. |
| RadixArk/...-NVFP4 | 135229562658 | 125.942 | modelopt float 4-bit group 16. FP8 PLE shards in the same sum. |
| RedHatAI/...-NVFP4 | 186432353250 | 173.629 | compressed-tensors nvfp4-pack-quantized, experts float 4-bit group 16. BF16 PLE shard included. |
| Inferact/...-NVFP4 | 182814991793 | 170.260 | modelopt. BF16 PLE shard included. |
| amd/...-Quark-MXFP4 | 182272111132 | 169.754 | `quant_method: quark`. PLE left in the shards (3200016008-byte pieces). |
| ISTA-DASLab/...-P48NVFP4-MoESQ | 160027935503 | 149.038 | compressed-tensors nvfp4-pack-quantized, 4-bit group 32 on Linear, attention and GDN ignored. |
| Minachist/...-INT4-Mixed-AutoRound | 175285755028 | 163.248 | int4 g128 experts, int6 g64 on 300 dense targets, int8 on hyper-connections and a few projections. Card: 2x24 GB serving keeps ~26 GiB of experts in host RAM. |
| cyankiwi/...-AWQ-INT4 | 188319714364 | 175.386 | compressed-tensors int4 g32, asymmetric, targets Linear. |
| btbtyler09/...-GPTQ-4bit | 187697766349 | 174.807 | `quant_method: gptq`, bits 4, group 32, sym. PLE filenames account for 102400513104 inside that sum. |
| spectator2026/...-W6-INT8-HQQ-g32 | 218756445985 | 203.733 | compressed-tensors int6 g32 on experts, int8 g32 on attn and shared. Card: custom 6-bit Marlin kernel, SM80. |
| azampatti/...-125B-A5B-INT4-AutoRound | 130816765050 | 121.833 | `quant_method: gptq`, bits 4 g128, `num_experts_per_tok: 5`. Pruned routing, not the trained top-10 model. Card says stock vLLM cannot serve the int4+fp8 mix. |

MXFP4 and NVFP4 are 4-bit on the expert pool. Quantizing the PLE table
(SecureAI-SE and codelight-ai, INT3 PLE beside NVFP4 experts, repo sums
106921679176 and 99383823704) does not pull the experts under 46 GiB.
Both cards require a vLLM patch for the INT3 table.

## Bit layout of the sub-4 candidates

Haberstroh, verified from `config.json` (`quant_method=auto-round`,
`bits=2`, `group_size=128`, `sym=true`, `data_type=int`,
`packing_format=auto_round:auto_gptq`, `autoround_version=0.14.2`,
`nsamples=48`, `batch_size=1`, `gradient_accumulate_steps=8`,
`low_gpu_mem_usage=true`) and from the card's component table. Default
scheme is the experts. `extra_config` has 2342 entries: 303 at 8 bits and
2039 at 16 bits. Classified by key:

| Class | Bits in extra_config | Card assignment |
|---|---|---|
| Routed experts | default 2, g128, symmetric. Not listed in extra. | 2-bit, all three projections, all 512 experts, all 48 layers |
| Shared expert | 145 keys at 8, 49 at 16 | card: shared expert 8-bit g128; gate stays 16 |
| Attention / GDN projections | 157 at 8, 232 at 16 | card: QSA q/k/v/o, GDN in_proj_qkv, in_proj_z, out_proj at 8 after a post-hoc INT8 re-pack; `in_proj_a`/`in_proj_b` stay 16 because N=48 is not a multiple of 32 |
| lm_head | 8, g128 | 8 |
| Routers, hyper-connections, indexer, norms, embeddings, PLE projections | 16 | 16 |
| PLE table | separate fp8 file, 51,200,245,760 bytes | fp8 lookup, mmap |
| MTP | 1554 extra keys at 16 | card: MTP head not in the serving config |

The card states the checkpoint does not load in Transformers, vLLM,
llama.cpp, or unpatched SGLang. All expert projections in a layer share
one `(bits, group_size)` because SGLang `FusedMoE` requires one scheme
per layer. No task-accuracy number is published for the weight quant.
The NLL table on the card is a KV-cache comparison against a bf16 KV
pool (INT4 KV delta +0.009 nats/token), not a weight-quant KLD.

klee100, from `config.json` plus the card. Default bits 2, group 64,
symmetric, `quant_method=auto-round`. `extra_config` length 39232:

| Class | Count at each bit width |
|---|---|
| Routed experts | 36888 keys at 3, against 73728 expert projections (48*512*3). About half the expert tensors are overridden to 3; the rest stay on the default 2. That is one bit width per block, not per expert. |
| Attention | 176 at 8, 219 at 16 |
| Shared expert | 194 at 16 |
| Router and other | 16 |
| MTP | 1536 at 4, 7 at 8, 13 at 16 |
| PLE projections | 16. The table itself is the 95.368 GiB BF16 shard. |

Card: "~2.998 effective bpw" on the routed experts, 8-bit main
projections, BF16 sensitive paths, 4-bit MTP experts. "KLD vs original
weights: 0.1208." The protocol (corpus, tokens, full vocab vs top-k) is
not in the card. Serving needs a patch on vLLM `a5a30471` for PLE SSD
offload, mixed-bit expert overrides, and MTP remapping
(github.com/iIIusi0n/qwen3.8-flash-next-cmp170hx).

HamboneLabs, from the card and the file list. Routed experts uint3,
group 64, bf16 scales, re-quantized from BF16 (the card says a second
quant of an NVFP4 file compounds error). 73,728 expert tensors. Dense
tail, including attention and shared experts, is the source build's
`W4A16_NVFP4`. MTP experts stay NVFP4 group 16. PLE is bf16, mmap.
`lm_head` and the vision tower are unchanged. Effective expert width
with bf16 scales is 3.125 bits (3 + 16/64). The card's export audit:
median relative Frobenius error 0.212, median cosine 0.978. Paired
accuracy on 564 items (GSM8K 200, MATH-500 200, HumanEval 164):

| Benchmark | Reference arm | This build | Delta |
|---|---:|---:|---:|
| GSM8K | 0.9750 | 0.9650 | -1.00 pp |
| MATH-500 | 0.9250 | 0.9000 | -2.50 pp |
| HumanEval | 0.9268 | 0.8780 | -4.88 pp |
| Pooled | 0.9433 | 0.9167 | -2.66 pp (SE 0.92, McNemar p 0.0071) |

The card does not name the reference checkpoint in that table. A separate
"lossless reference" token check reports mean KL 0.149 and top-1
disagreement 0.0639. Load needs eight overlaid vLLM files, including a
humming MoE path and a modelopt patch. Model-load size on the card is
52.04 GiB, next to the 51.536 GiB safetensors sum.

Eliovp W3A8, from the card. 512 routed experts at 3.125 bits in a rotated
8-level codebook; everything else 8-bit or bf16; PLE in pinned host RAM.
Served by Paiton on two Radeon AI PRO R9700 (32 GB). Card: full-vocab KL
vs bf16 is 0.060 (0.041 with bf16 routing), top-1 agreement 90.7%. Peak
VRAM on the measured run was 30.6 GiB and 30.5 GiB. That is evidence a
~3.1 bpw expert codebook plus an 8-bit trunk can sit in 2x32 GB with PLE
in host RAM. The file is not `auto-round`, GPTQ, AWQ, or
compressed-tensors. `library_name` is `paiton`.

Other non-loaders, for the record:

- EXL3 (ExLlamaV3). r0b0tlab 2.50 bpw: `quant_method: exl3`, bits 2.52,
  card recipe `convert.py -b 2.50` on routed experts, higher bits on
  attention (`-hq`), vision 6, MTP 4, n-gram 3. JRamirez 2.25 bpw repo
  sum 66855420804, n-gram file 26240152672. VinceTrune abliterated 2.05
  bpw repo sum 50044873738 (config fetch returned 401). No `exl3` string
  under `vllm/model_executor/layers/quantization/`.
- MLX. `mlx-community/...-OptiQ-2bit` safetensors 80765335007. Card:
  2-bit routed experts, 4-bit attention, GDN, shared expert, embeddings,
  lm_head, and n-gram table. No capability score. Not a vLLM file.
- GGUF Niwaki. `neopolita/...-99B-A5B-Niwaki-3bit-gguf`: UD-Q3K file
  42457159040, UD-Q4K file 54074469120. Card: experts pruned (routed
  experts on 24 of 48 layers, intermediate 512 with 448 used), not the
  full 512-expert model.
- GLQ. `xv0y5ncu/...-GLQ-trellis-3inst-3bpw-ple4`: one safetensors file,
  83279953072 bytes, `quant_method: glq`. Not a vLLM method.
- HQQ. The only hit is the W6/INT8 repo above. `hqq` is not in
  `QuantizationMethods` (`quantization/__init__.py:15-49`).
- AQLM and QuIP#: zero repos, and neither name is a vLLM method.
- hipfire GPTQ3-in-mq4 file `qwen3.8-flash-next-gptq3.mq4` is
  125288544792 bytes. The mq3-xts-coder metrics file is a pruned pack
  (243 experts/layer, 29112225304 bytes) and compares itself to an ISTA
  GSQ-RCO Coder GGUF, not to the full IQ3_XXS daily driver. Runtime is
  hipfire.

## XPU kernels on this tree

`XPUPlatform.supported_quantization` (`platforms/xpu.py:113-131`) is:
awq, gptq, moe_wna16, auto_awq, auto_gptq, inc, fp8, deepseek_v4_fp8,
mxfp4, mxfp8, fp8 per-tensor/block/channel, online, gpt_oss_mxfp4,
modelopt, compressed-tensors. It does not list gguf, aqlm, hqq, quip,
or exl3. `QuantizationMethods` (`quantization/__init__.py:15-49`) has
the same gap.

| Quant | Width that actually dispatches on XPU | Linear or MoE | File:line |
|---|---|---|---|
| INC config accept list | 2, 3, 4, 5, 6, 7, 8 | config only | `inc/inc.py:37` |
| `auto-round` renamed to `inc` | when `quant_method == "auto-round"` | both | `inc/inc.py:364-367` |
| INC XPU WNA16 linear | symmetric 2 and 4 | linear | `inc/schemes/inc_wna16_scheme.py:21`, `:71-72` |
| oneDNN w4a16 / w4a8 | int4 only; int2 raises | linear | same file `:29-30`, `:81-86` |
| INC int2 linear | requires ARK (`auto_round_kernel`); missing ARK raises | linear | `:99-107` |
| ARK linear call | passes `bits=self.weight_bits` | linear | `inc/schemes/inc_wna16_linear.py:406-409` |
| Anything else on INC XPU linear (int3, asymmetric) | raises `INC on XPU: unsupported config` | linear | `inc_wna16_scheme.py:116` |
| INC XPU MoE entry | only `is_gptq` and bits == 4 and sym | MoE | `inc_wna16_scheme.py:161-166` |
| ARK MoE | symmetric int4 only; `weight_bits=4` hardcoded in `prepare` | MoE | `inc/schemes/inc_wna16_moe.py:38-40`, `:135-149` |
| Fallthrough `MoeWNA16Method` | int4 or int8, else `ValueError` | MoE | `moe_wna16.py:242-252` |
| GPTQ into moe_wna16 | bits in {4, 8}; AWQ bits == 4 | MoE | `moe_wna16.py:178-185` |
| `XPUwNa16LinearKernel` | `uint4` and `uint4b8`, group multiple of 32 | linear | `kernels/linear/mixed_precision/xpu.py:15`, `:33-38`; GEMM `int4_gemm_w4a16` at `:100` |
| `XPUW4A8IntLinearKernel` | symmetric int4 | linear | same file `:112-135` |
| XPU WNA16 kernel list | those two kernels only | linear | `kernels/linear/__init__.py:548-550` |
| `XPUExpertsWNA16` | `kInt4Static`, `kInt4Static32` | MoE | `fused_moe/experts/xpu_moe.py:273-308` |
| WNA16 MoE backend order on XPU | `WNA16MoEBackend.XPU` only | MoE | `fused_moe/oracle/int_wna16.py:130-131` |
| compressed-tensors WNA16 type map | uint2/uint3/.../uint8 exist as scalar types | linear map | `compressed_tensors/schemes/compressed_tensors_wNa16.py:41-58` |
| compressed-tensors MoE WNA16 | builds a weight key for bits other than 4 and 8, then `select_wna16_moe_backend` | MoE | `compressed_tensors_moe_wna16.py:81-109` |
| That selection on XPU | `XPUExpertsWNA16._supports_quant_scheme` is int4-only, so uint2/uint3 fail the support check | MoE | `xpu_moe.py:305-308` |
| Humming (the CUDA odd-bit path) | bits 2, 3, 5, 6, 7 | linear and MoE, CUDA | `inc_wna16_scheme.py:26`, `:142-146`, `:212-216`; `can_implement` requires `is_cuda` at `kernels/linear/mixed_precision/humming.py:31-32` |
| Exllama | listed types uint4b8 and uint8b128; comment says uint2/uint3 are untested and not listed | linear, CUDA/ROCm | `kernels/linear/mixed_precision/exllama.py:19-33` |
| AutoGPTQ `TYPE_MAP` | `(4, True)` and `(8, True)` only | linear | `auto_gptq.py:99-102`; other pairs raise at `:152-155` |
| AutoAWQ `TYPE_MAP` | 4 only | linear | `auto_awq.py:182-184` |
| MXFP4 linear | 4-bit float | linear | XPU kernels at `kernels/linear/__init__.py:629-632` |
| MXFP4 MoE | 4-bit float | MoE | `fused_moe/oracle/mxfp4.py:446-447`; INC path `inc/schemes/inc_mxfp4_moe.py:78-79` |
| NVFP4 linear | 4-bit float | linear | `_POSSIBLE_NVFP4_KERNELS` has CUDA and ROCM only (`kernels/linear/__init__.py:584-605`). The selector reads that map at `:1212`. `modelopt` is on the XPU quant list and still has no NVFP4 kernel. |
| vllm-xpu-kernels fused MoE recipes | mxfp8, fp8 block, mxfp4, int4, fp8, else bf16 | MoE | `vllm_xpu_kernels/fused_moe_interface.py:56-68` |
| vllm-xpu-kernels GEMM ops | `int4_gemm_w4a16`, `int4_gemm_w4a8` | linear | `csrc/xpu/torch_bindings.cpp:48-56` |

A compressed-tensors or INC checkpoint at int2 or int3 will parse.
Kernel selection then fails. The only sub-4-bit path that reaches a
kernel call on XPU is symmetric int2 linear via INC+ARK, and the ARK
package is absent. MoE at int2 or int3 does not reach a kernel.
Symmetric int4 MoE does, and so does MXFP4 MoE. Both are 4-bit on the
expert pool, which is the part that misses 46 GiB.

Haberstroh's `packing_format` contains `gptq`, so `INCLayerConfig.is_gptq`
is true (`inc/config_parser.py:28-29`). The XPU MoE branch still requires
bits == 4. The expert default is bits 2, so the call falls through to
`MoeWNA16Method`, which raises.

## Producing our own mixed AutoRound

Source. There is no local BF16 or FP8 safetensors. Official BF16 is
360000363614 bytes (335.276 GiB, 360.000 decimal GB), which matches the
card's 360 GB line and is larger than report 02's shard-1-plus-PLE figure
(329.58 GiB) by the MTP and vision tensors. Official FP8 is 185540727598
bytes (172.798 GiB). Largest BF16 shard is 3.27 GiB. Largest FP8 shard
is 1.99 GiB. Either repo fits on `/mnt/vm_8tb` (5801 GiB free). Neither
fits in 121 GiB of RAM as a resident load, and MemAvailable at survey
time was 73 GiB. FP8 also fits on `/mnt/cache` (589 GiB free) with room
left; BF16 does not leave a comfortable margin there.

AutoRound memory, from the upstream `step_by_step.md` fetched this
session (not a local measurement on Flash-Next):

- Tuned quantization is block-by-block and "typically 10 times the block
  size" of GPU memory. One expert layer is 120,795,955,200 / 48 elements,
  4.688 GiB in BF16. Ten times that is 46.9 GiB, which is over one 32 GB
  B70. `low_gpu_mem_usage` plus batch 1 and gradient accumulation is the
  documented way down. Haberstroh finished that path from the Qwen FP8
  release on one 32 GB GPU and 32 GB host RAM in 3 days 8 hours
  (AutoRound 0.14.2, scheme W2A16 g128, `--enable_alg_ext`, 48 samples x
  2048, 200 iters, batch 1, grad accum 8, four subset rounds over the
  512 experts). Our host has more RAM and two 32 GB B70s. The package is
  not installed, and the source is not on disk. That recipe is evidence
  the tuned int2 job can finish. It is not a reason to start it.
- Model-free mode (`--iters 0 --disable_opt_rtn`, automatic since
  v0.13) quantizes safetensors shard by shard and does not instantiate
  the model. Documented INT presets are W2A16, W2A16G32, W2A16G64,
  W4A16, W4A16_MIXED, W8A16. W3A16 raises `ValueError` in model-free
  mode. AutoScheme with `model_free=True` rejects W3A16 and rejects a
  mix of INT and MXFP. Published peak-memory examples are a few GB for
  Qwen3-Next-80B and Qwen3-235B W4A16. Those are not Flash-Next numbers.
  Intel's own MXFP4-mixed card is this model, model-free, and the file
  is 167 GiB because the experts stay 4-bit and the PLE stays BF16.
- The doc recommends auto-round for W4A16 and auto-round-best (about
  4-5x slower) for W2A16. Model-free INT is plain RTN. RTN at 2 bits is
  the weak end of that ladder.
- The same doc says vLLM and SGLang fuse MoE and QKV, and recommends
  against different bit widths inside those fused layers.

Expert profile. `/mnt/vm_8tb/b70/build/strata-ab/src/data/expert-profile-sycl.bin`
is 196632 bytes, magic `STRP`, version 1, 48 x 512, 24576 slots, 24576
unique pairs. Reader: `src-ep/sycl/src/core/expert_cache.cpp:152-207`.
Writer: `src-ep/tools/make_profile.py`. The ranking is not the same file
as `expert-profile.bin` (same length, first pair already differs).

Spread of the sycl ranking, counted here:

| Prefix of the ranking | Pairs | Layers touched | Experts per layer |
|---|---:|---:|---|
| 5% | 1229 | 48 | 6 to 42 |
| 25% | 6144 | 48 | 103 to 149 |
| 49% | 12042 | 48 | 209 to 327 |
| 100% | 24576 | 48 | 512 |

`make_profile.py` says a layer-split cost model that reads this ranking
as a frequency curve claimed 94.9% coverage against 69.2% measured at
5805 pairs. Hot experts are already spread across every layer, so a
per-layer bit choice cannot isolate them. A per-expert bit choice inside
one fused expert tensor is not what `XPUExpertsWNA16` or Haberstroh's
fused MoE accepts. klee100's workable mixed layout is half the blocks at
W3 and half at W2, every expert in a block on the same scheme. That is
the layout AutoRound will actually emit. It is not a hotness quant, and
W3 cannot be produced model-free.

Quality against ISTA IQ3_XXS. No head-to-head exists. These published
numbers are different protocols and mostly different checkpoints:

| Source | What was measured | Number | Comparable to ISTA IQ3_XXS? |
|---|---|---|---|
| Intel W4A16 card | GSM8K / MMLU / PIQA / HellaSwag avg vs their BF16 run | 0.8332 vs 0.8362 (99.64%) | No. 4-bit experts, file is 169 GiB. |
| intel-ai MXFP4-mixed card | same four tasks, model-free RTN | 0.8326 vs 0.8362 (-0.35 pp). GSM8K 0.9621 vs 0.9674 | No. Experts stay MXFP4. |
| Hambone uint3 card | paired GSM8K+MATH-500+HumanEval | -2.66 pp pooled, -4.88 pp HumanEval; KL 0.149 vs a lossless reference | Closest published cost of a 3-bit expert cut. Dense tail was left at NVFP4, not matched to ISTA. |
| klee100 card | "KLD vs original weights" | 0.1208, protocol not stated | Cannot be ranked against the others. |
| Eliovp W3A8 card | full-vocab KL vs bf16, Paiton codebook | 0.060 | Different format and runtime. |
| Haberstroh card | KV-cache NLL, not weight KLD | no weight-accuracy table | The 2-bit expert choice is unpublished as a task score. |
| Unsloth UD-IQ3_XXS (their docs) | top-1 85.414, mean KLD 0.165 | a different GGUF mix, total ~82 GB including PLE | Not the ISTA GSQ-RCO file. |

Inference, not a measurement: an int2 expert pool at 2.125 stored
bits/weight covers 96% of shard 1, against ISTA's 2.992 stored
bits/element with the importance layout report 02 histogrammed (expert
downs are Q2_0 or IQ4_NL, gate/up are IQ2/IQ3). That file should be
expected to land below ISTA. Hambone's -2.66 pp is the published cost of
stopping at uint3 with the dense tail left expensive, and it still
missed the 46 GiB bar (51.5 GiB body). Re-quantizing the ISTA GGUF would
add a second loss on top of that. Report 02 already rejects that path.

## Ranked recommendation

1. Keep serving quality and the size bar on ISTA IQ3_XXS. Getting those
   bytes to run is an XPU Qwen4Exp and GGUF-kernel problem (report 02).
   It is not a quantization problem.

2. Do not download Intel W4A16, intel-ai MXFP4, nvidia/RadixArk/RedHat
   NVFP4, wtdcode AWQ, or devan-carlin W4A16 for this target. Expert
   int4 or MXFP4 or NVFP4 is 58 to 63 GiB before the dense tail. The
   published files are 72 GiB of body once PLE is split out, or 167 to
   360 GiB with PLE left in the shards. XPU can run symmetric int4 MoE
   and MXFP4 MoE. The bytes still miss 2x32 GiB plus 2x200k KV.

3. Do not adopt Haberstroh, klee100, or Hambone as the vLLM weight.
   Haberstroh fits (36.144 GiB) and is 2-bit on every expert, with no
   XPU MoE kernel and a card that says stock vLLM will not load it.
   klee100 is the only AutoRound file aimed at vLLM with a published
   KLD, and the body is 47.159 GiB plus a private patch, with both int2
   and int3 experts. Hambone is the honest 3-bit quality measurement
   and is 51.5 GiB plus NVFP4 dense, humming, and eight patched files.

4. Do not start a local quant run until an XPU int2 or int3 MoE kernel
   exists. When it does, the file to produce is non-routed int4 g128
   (attention, shared expert, the rest of shard 1) and routed int2 g128,
   one scheme for every expert in a layer, PLE left in host RAM.
   Computed size 32.259 GiB, 41.41 GiB with 2x200k BF16 KV. Source is
   the official FP8 (172.8 GiB) or BF16 (335.3 GiB), streamed. The
   quality path is Haberstroh's tuned low-memory recipe, not model-free
   RTN, and not a bit mask taken from `expert-profile-sycl.bin`. W3
   cannot be made model-free; a tuned int3-expert file is 46.321 GiB
   with an int4 dense tail, just over the bar, and still has no kernel
   today.

5. Eliovp's 2x32 GB residency is a data point that a 3.125 bpw expert
   codebook can fit. It is a Paiton format. Porting it is a new runtime,
   not a weight choice.

## Not done

No GPU, no quantization, no download above the config/README cap, no
change to hotschmoe-dd, no edit to reports 01-04. Kernel conclusions are
from static dispatch code. Nothing was loaded.
