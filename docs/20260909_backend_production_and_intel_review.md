# Production inference backends and Intel B70 evidence

Reviewed 2026-09-09. Read-only web/source research; no serving experiments.
Companion: [next campaign](20260909_next_serving_campaign.md).

## Decision for this machine

Keep vLLM as the measured serving reference and SGLang as the primary
alternative/research backend. Public production adoption supports taking
both seriously; it does not rank their Intel hybrid-attention implementations.
OpenVINO GenAI/Model Server is the most relevant additional Intel-native
implementation to audit if the two main candidates cannot meet correctness
and capacity. No third-backend GPU campaign is implied by this review.

No source reviewed establishes a matched winner for two B70s, Qwen3.8 INT4,
FP8 KV, MTP3, GPU prefix caching and four active 200K tool conversations.
Our maintenance campaign must establish that result.

## What companies publicly disclose

| Organization | Public evidence | Limits |
| --- | --- | --- |
| xAI / SpaceXAI | SGLang's project README lists xAI as an adopter. A joint MTP article credits the xAI team. | Project-side evidence of adoption/contribution, not proof that every Grok endpoint uses stock SGLang, or which current fork/hardware path it uses. |
| LinkedIn | Its engineering article explicitly describes vLLM production deployments for generation and embeddings, with custom integration and operational work. | Deployment-specific; does not imply exclusive use across LinkedIn. SGLang also lists LinkedIn as an adopter. |
| Baseten | Its Qwen 3 article explicitly says it deployed SGLang in production for customers. Its documentation offers vLLM, SGLang and optimized TensorRT-LLM engines. | A concrete example of choosing multiple engines by workload rather than a single company-wide winner. |
| OpenAI | Official documentation explains self-hosting gpt-oss using vLLM. | This is not disclosure of the backend serving proprietary ChatGPT/API models. No reviewed official source established a specific current engine for those products. |

Sources:

- [SGLang adoption statement](https://github.com/sgl-project/sglang)
- [Joint SGLang MTP work](https://www.lmsys.org/blog/2025-07-17-mtp/)
- [LinkedIn production vLLM account](https://www.linkedin.com/blog/engineering/ai/how-we-leveraged-vllm-to-power-our-genai-applications)
- [Baseten Qwen 3 production deployment](https://www.baseten.co/blog/day-zero-benchmarks-for-qwen-3-with-sglang-on-baseten/)
- [Baseten engine options](https://docs.baseten.co/examples/deploy-a-llm)
- [OpenAI gpt-oss vLLM guide](https://developers.openai.com/cookbook/articles/gpt-oss/run-vllm)

Do not conflate an OpenAI-compatible API with OpenAI's internal runtime;
model support, conference attendance or a contributor affiliation is not
proof of a particular production deployment. At our scale, portable lessons
are pinned artifacts, workload-specific qualification, admission control,
cache correctness, observability and tested recovery. Large GPU fleet
architecture and NVIDIA benchmark rankings do not transfer automatically.

## Intel-specific engineering literature

### vLLM XPU and Intel LLM Scaler

[vLLM's XPU support matrix](https://docs.vllm.ai/en/stable/models/hardware_supported_models/xpu/)
names Intel Arc Pro B-Series as validated hardware and lists model/precision
combinations. This is stronger evidence than generic GPU support, but does
not qualify our exact hybrid model, MTP/cache combination or 800K requirement.

[Intel LLM Scaler](https://github.com/intel/llm-scaler) targets Arc Pro B60/B70
and maintains model-serving recipes and patches. Intel's published B70
[launch measurements](https://download.intel.com/newsroom/2026/ClientComputing/Intel-vPro-Arc-Pro-Press-Desk.pdf)
use intel/llm-scaler-vllm:1.3. They establish Intel's investment in this route,
not a measured advantage over SGLang on our workload. Model, driver, image,
precision, number of cards and context differ from our current campaign.

### SGLang on Intel: a concrete source lead

The [Intel LLM Scaler SGLang directory](https://github.com/intel/llm-scaler/tree/main/sglang)
contains an actual Battlemage LLM recipe, not only SGLang Diffusion support.
It describes Qwen3.6-35B-A3B TP2 online FP8 e5m2, ESIMD attention/MoE/GDN
fast paths and a GSM8K harness. It deliberately disables XPU graphs for
accuracy. Its optional FP8 router changes some top-8 routing decisions.

This is useful source for the CPU audit and possible deliberate ports,
especially the MoE alternative. It is not calibrated E4M3 KV evidence:
online FP8 model/GEMM quantization and FP8 KV storage are different features.
Do not enable its graph/router settings or copy its state precision into
our recipe without matched correctness tests.

SGLang's [hardware documentation index](https://docs.sglang.io/llms.txt)
also lists XPU. The direct XPU markdown fetch failed during this review;
the concrete Intel source recipe is the stronger inspected reference here.

### OpenVINO GenAI and Model Server

[Hybrid Attention Model Cache Management](https://openvinotoolkit.github.io/openvino.genai/docs/concepts/optimization-techniques/hybrid-attention-models-cache/)
explains separate KV and linear-attention state allocation. With prefix
caching, checkpoint intervals affect both state memory and reuse granularity.
This is directly relevant to our 800K accounting and complete-state reload
requirements, even if we keep another engine.

[Long-context optimization guidance](https://docs.openvino.ai/nightly/model-server/ovms_demo_long_context.html)
describes low-precision cache and hybrid-state tuning in Model Server. The
nightly documentation is not a qualification of a pinned stable release.
Its low-precision cache routes are not interchangeable with our FP8 scales.

OpenVINO is worth a feature/source audit, but exact model conversion,
two-card parallelism, MTP, tool/reasoning parsing and hybrid offload would
all need verification before allocating a GPU campaign. Existing constraints
keep llama.cpp and ZML outside the live tree; this review does not restore
either backend.

### Research papers

- [Leveraging Speculative Sampling and KV-Cache Optimizations Together for Generative AI using OpenVINO](https://arxiv.org/abs/2311.04951)
  (2023): useful background on composing speculation and cache optimization.
  It predates our model and B70 stack; it is not a backend ranking or a fix
  for current hybrid-state corruption.
- [Xe-Forge: Multi-Stage LLM-Powered Kernel Optimization for Intel GPU](https://arxiv.org/abs/2605.26118)
  (2026): reports Intel Arc Pro B70 kernel experiments and emphasizes
  hardware-specific constraints and on-device verification. Kernel-level
  results do not establish end-to-end serving speed or correctness.

Treat vendor marketing, project support matrices, kernel papers and
deployment reports as different evidence classes. None replaces a matched
multi-client agent workload with identity, health and teardown evidence.

## Alternatives that do not answer the Intel question directly

[TensorRT-LLM](https://docs.nvidia.com/tensorrt-llm/) is an NVIDIA-oriented
LLM runtime with its own hardware support matrix; it is not an Intel XPU
replacement. Serving layers such as Triton Inference Server, KServe or
Ray Serve can host/manage engines but should not be confused with the
attention/GDN kernels responsible for our current numerical/state behavior.

## Follow-up before maintenance

1. Audit the exact Intel SGLang source/model/precision paths alongside our
   retained implementation. Start with a source ledger, not an upgrade.
2. Confirm support for FP8 KV scales, hybrid prefix state and RAM offload
   on each candidate's actual XPU path, rather than its generic feature list.
3. Preserve the vLLM reference and run the companion campaign's matched
   quality/capacity matrix. Prefer the engine that meets it with acceptable
   latency and maintenance cost, regardless of company adoption lists.

CONFIG -> User-requested production/backend literature review; live endpoint
reported in use; CPU/read-only scope.
COMMAND -> Search and open primary project, vendor, company-engineering and
paper sources; inspect local campaign evidence and model config geometry.
RESULT -> Both engines have production adoption evidence; Intel supplies
concrete vLLM and SGLang BMG paths; no matched public 800K result found.
VERDICT -> Keep both in the next campaign. Production engine choice remains
an empirical local decision. No service, driver, package or GPU change made.
