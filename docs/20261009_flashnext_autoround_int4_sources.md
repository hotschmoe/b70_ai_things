# Qwen3.8-Flash-Next Intel AutoRound INT4 research, 2026-10-09

CONFIG -> Read-only investigation of the exact Flash-Next model and Intel's W4A16
artifact. No serving, GPU access, model download, or change to the selected
Unsloth UD-Q4_K_XL campaign. The only weight-file request read at most 64 KiB
from a safetensors header; it did not download weight tensors.

COMMAND -> Search current Hugging Face and primary Intel/AutoRound sources;
read HF model API, pinned configuration and model card; inspect upstream
vLLM/SGLang quantization dispatch and the published four-B70 reference. Read
shard 16's safetensors header with HTTP Range bytes=0-65535. Retrieval date:
2026-10-09 UTC.

RESULT -> An official Intel artifact exists:
[Intel/Qwen3.8-Flash-Next-W4A16-AutoRound](https://huggingface.co/Intel/Qwen3.8-Flash-Next-W4A16-AutoRound),
revision `4c67bf686b7f7fd386bae6b07ab59e8ff1d5b897`, last modified
2026-08-31. The Intel-author HF search returned this one matching repository.
Its declared base is `Qwen/Qwen3.8-Flash-Next`. Architecture is
`Qwen4ExpForConditionalGeneration` / `qwen4_exp`: 48 text layers, hidden size
2560, 512 experts, top 10, expert intermediate size 640. This is the model
family in our campaign, not the differently named Qwen3-Next-80B-A3B.
[HF identity API](https://huggingface.co/api/models/Intel/Qwen3.8-Flash-Next-W4A16-AutoRound?blobs=true),
[pinned config](https://huggingface.co/Intel/Qwen3.8-Flash-Next-W4A16-AutoRound/raw/4c67bf686b7f7fd386bae6b07ab59e8ff1d5b897/config.json).

The configuration declares `quant_method=auto-round`,
`packing_format=auto_round:auto_gptq`, integer weights, 4 bits, symmetric,
group size 128, AutoRound version 0.15.0. Routed experts are quantized;
attention/GDN, shared experts, router, PLE, MTP, vision, embeddings, and head
are excluded. The embedded config has 2340 extra_config entries; the separate
quantization_config.json has 784. Do not substitute one for the other without
checking loader precedence. This is packed safetensors W4A16, not GGUF
UD-Q4_K_XL or NVFP4.
[Pinned quantization config](https://huggingface.co/Intel/Qwen3.8-Flash-Next-W4A16-AutoRound/raw/4c67bf686b7f7fd386bae6b07ab59e8ff1d5b897/quantization_config.json).

Intel's card supplies a 200-iteration AutoRound tuning command and this
same-model BF16 comparison:

| Metric | BF16 | INT4 | Difference, percentage points |
| --- | ---: | ---: | ---: |
| GSM8K | 96.73 | 96.82 | +0.09 |
| MMLU | 86.51 | 85.60 | -0.91 |
| PIQA | 81.93 | 82.15 | +0.22 |
| HellaSwag | 69.27 | 68.73 | -0.54 |
| Average | 83.62 | 83.32 | -0.30 |

The card reports 99.64% relative retention of the benchmark average. It does
not establish universal quality retention, exact coding/agent parity, or B70
inference fidelity. The card/config inspected do not pin calibration dataset,
samples, seed, sequence length, base-model revision, or evaluation harness and
prompt settings. Default calibration settings cannot be assumed to be the
actual calibration provenance.
[Pinned Intel model card](https://huggingface.co/Intel/Qwen3.8-Flash-Next-W4A16-AutoRound/blob/4c67bf686b7f7fd386bae6b07ab59e8ff1d5b897/README.md).

Size is materially different from our GGUF choice:

| Measurement | Bytes | GiB, approximately |
| --- | ---: | ---: |
| All 17 safetensors files, HF file metadata | 181196007016 | 168.752 |
| Raw tensor bytes, HF dtype census | 181165326328 | 168.723 |
| Shard 16 file | 102400512256 | 95.368 |
| Shard 16 PLE tensor payload | 102400491520 | 95.368 |
| Files excluding shard 16 | 78795494760 | 73.384 |

Shard 16 contains only 128 BF16 PLE tensors, each shape `[2500012,160]`.
The header is 20728 bytes. HF's displayed 75B parameter count counts packed
storage elements; it is not evidence that this is a smaller base model.
Full host residency exceeds our 121 GiB RAM. Excluding PLE still exceeds the
two-card device capacity before KV/workspace. That is a capacity calculation,
not a claim that mmap/offload is impossible. Nominal combined RAM plus VRAM
is about 185 GiB, so split residency is arithmetically possible but tight
before the OS, staging, KV, and workspace. It needs measured residency and
an explicit PLE mmap/disk-tier plan; it is not a direct TP=2 fit. Our selected
main GGUF is approximately 103.69 GiB, before auxiliary files.
[Exact shard](https://huggingface.co/Intel/Qwen3.8-Flash-Next-W4A16-AutoRound/blob/4c67bf686b7f7fd386bae6b07ab59e8ff1d5b897/model-00016-of-00017.safetensors).

XPU support is more than a generic HF deployment button. At vLLM revision
`a1cd0ec650852bf2ce124b3dea930a38f0d243da`, INC WNA16 recognizes GPTQ-packed
AutoRound and has explicit XPU dispatch; GPTQ MoE resolves through
AutoGPTQMoEMethod to the XPU WNA16 expert backend when supported. Linear XPU
code accepts this symmetric GPTQ packing. This supports a plausible XPU
loader/kernel route, not a qualification of Flash-Next, TP=2, host offload,
or our current environment.
[vLLM scheme dispatch](https://github.com/vllm-project/vllm/blob/a1cd0ec650852bf2ce124b3dea930a38f0d243da/vllm/model_executor/layers/quantization/inc/schemes/inc_wna16_scheme.py),
[XPU linear implementation](https://github.com/vllm-project/vllm/blob/a1cd0ec650852bf2ce124b3dea930a38f0d243da/vllm/model_executor/layers/quantization/inc/schemes/inc_wna16_linear.py).

Intel ARK documents Battlemage weight-only kernels, but its MoE INT4 API is
still labeled work in progress. ARK support alone is not proof of the vLLM
fused-expert route. SGLang's generic AutoRound support likewise is not exact
Flash-Next-on-B70 evidence: inspected revision
`438df9a2a4645d39c87e33c4a27792e568ac2701` delegates GPTQ MoE to other
quantization implementations; this audit did not establish the complete XPU
architecture/kernel/offload path.
[ARK at inspected revision](https://github.com/intel/auto-round/blob/63895ebf68975e8f0f13bb84eac25e175fba83a4/auto_round_extension/ark/README.md),
[SGLang dispatcher](https://github.com/sgl-project/sglang/blob/438df9a2a4645d39c87e33c4a27792e568ac2701/python/sglang/srt/layers/quantization/auto_round.py).

A relevant operator report exists for this exact Intel artifact on four B70s.
However, its reference setup uses 256 GiB host RAM, TP=4 plus expert parallel,
patched vLLM, and an INT8 NVMe PLE table substituted from another checkpoint.
It does not qualify our two-card machine or preserve the original Intel BF16
PLE artifact unchanged. The repository explicitly says Intel-versus-donor PLE
row identity was not checked. Its known-issues document also identifies F16
scale conversion to BF16, missing MTP-on/off parity, and an unbooted combined
release branch. These caveats override a casual reading of the issue's
enthusiastic stability claim.
[Operator issue](https://github.com/intel/auto-round/issues/2440),
[pinned weight provenance](https://github.com/Lumnus/b70-flash-next/blob/b103e0f4522316b6368e135586b69a8bd3d37cf0/docs/weights.md),
[pinned known issues](https://github.com/Lumnus/b70-flash-next/blob/b103e0f4522316b6368e135586b69a8bd3d37cf0/docs/known-issues.md).

VERDICT -> Yes: Intel provides a tuned, symmetric group-128 INT4 AutoRound
checkpoint of our exact model family, with encouraging vendor BF16 comparisons
and a concrete four-B70 vLLM reference. It merits separate research. Neither
the published quality numbers nor the modified four-card deployment establish
quality or feasibility for our two-card/RAM configuration. No format switch,
large download, or GPU experiment was performed or proposed as an automatic
next step.
