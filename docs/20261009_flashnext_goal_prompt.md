# Flash-Next continuous research goal prompt

Copy the block below into a new session. Saved on 2026-10-09.

```text
/goal Get Qwen3.8-Flash-Next, using our selected Unsloth UD-Q4_K_XL artifact, running correctly and efficiently across both Intel Arc Pro B70 GPUs, then profile and optimize latency for real interactive serving.

Work in /mnt/vm_8tb/github/b70_ai_things as hotschmoe. Read AGENTS.md first and preserve unrelated dirty changes. Follow the existing GPU lease, health, recovery, model-identity, and checkpoint rules. You have permission to use both GPUs and stop current serving containers as needed.

Continue autonomously with agents working in parallel. Keep one coordinator responsible for GPU experiments and lifecycle ownership. Research agents can inspect sources and prepare patches independently. Commit and push coherent checkpoints often, document failures as well as successes, and give concise progress updates.

Read the existing campaign evidence before repeating work:
- docs/20261008_flashnext_udq4xl_campaign.md
- docs/20261009_flashnext_gpu_expert_cache_comparison.md
- docs/20261009_flashnext_strata_next_steps.md
- docs/20261009_flashnext_latency_methodology.md
- strata/flash-next/native-q8-hc-design.md
- JOURNAL.md

Current checkpoint: d786633, pushed. Verify subsequent changes before proceeding.

Model and workload:
- Keep the pinned UD-Q4_K_XL model, tokenizer, and chat-template identity.
- Hardware: two 32 GiB B70s and approximately 121 GiB usable system RAM.
- Usually one stream, frequently two, rarely four, never more than six.
- Optimize time to first token, inter-token latency, completion latency, and fairness between streams. Aggregate throughput is secondary.
- Support prefix caching for repeated conversation history and shared prefixes, avoiding unnecessary prefill recomputation.
- Keep hotschmoe-dd as the first client-facing served model name.

Backend selection:
Any backend or custom engine is acceptable. Assign an agent to investigate mature alternatives to Strata that support this exact model, mixed GGUF quantization, Intel GPUs, tiered expert placement, prefix caching, and small-concurrency serving.
Compare actual source support and implementation cost, not advertised capabilities. Distinguish KV/prefix-cache offload from model-weight/expert offload.
Use current backend sources, pin revisions, and rebuild ABI-specific code deliberately.
Do not silently change quantization or substitute model components to make a backend work.

Existing findings:
- Fresh llama.cpp loads the exact GGUF across both cards and supports expert caching, but repeated-request correctness remains unresolved.
- Static placement also exhibits history-dependent output/probability drift. Disabling optimization, checkpoints, or fusion has not established a fix. Stop stacking switches; localize the first divergent state or activation.
- Strata supports concurrent slots and cross-GPU conversation groups, but native Q8 hyper-connection fidelity and second-stage RAM expert coverage need work.
- A default-off Strata Q8/F32 projection primitive patch passes CPU tests and SYCL object compilation. GPU numerical qualification and complete model integration are outstanding.
- Neither backend has earned a production speed/stability claim for this configuration.

Memory hierarchy:
Aim to keep dense/shared computation and useful experts in VRAM, with overflow experts in RAM. Evaluate PLE/ngram tables on NVMe with a bounded RAM cache.
The 8 TB model filesystem is SATA SSD with approximately 5 TB free. /mnt/cache is NVMe with approximately 671 GiB free; recheck capacity.
Move or stream data only when matched latency measurements show a benefit. Do not assume disk offload improves performance when RAM residency is available.

Prefix-cache qualification:
Verify actual reused and newly evaluated token counts, cache-hit TTFT, and cached-versus-uncached outputs/logits using declared correctness tolerances.
Cover recurrent/GDN, attention, QSA/indexer, and PLE state; KV reuse alone may be insufficient.
Test independent sessions, shared prefixes, divergent suffixes, eviction, cancellation, fresh starts, and concurrent requests. Prevent state leakage and stale cache identity.
Measure cache memory costs and their effect on expert residency and active-stream latency.

Profiling methodology:
Use the user's newer reference:
https://github.com/hotschmoe/x2-nvfp4-lab
Local read-only reference: /mnt/vm_8tb/b70/research/x2-nvfp4-lab
Recorded revision: e0756d2115dd22af0b7399515de41d9390004422.
Specula is the old repo and is not the requested methodology.

Separate host preparation/submission, queued work, device execution, synchronization, transfers, cache bookkeeping, sampling, and streaming.
Correlate these intervals to the request/token critical path. Avoid adding barriers that destroy the overlap being measured.
Validate trace coverage, timestamps, ownership, and output equivalence. Separate instrumented diagnostics from clean latency runs.
Use interleaved A/B or ABBA comparisons, matched configurations, useful prompts, natural completion, full-logit/intermediate checks where appropriate, and negative controls.
Report per-stream distributions, including P50/P95 TTFT and token gaps. Test an ongoing decode while another request introduces a long prefill.
Bandwidth saturation is not the optimization objective.

Progression:
1. Choose a justified primary development backend and retain a trustworthy reference.
2. Establish model fidelity, deterministic coherence, state isolation, and healthy teardown.
3. Implement and qualify two-GPU tiered placement and prefix caching.
4. Qualify one and two streams, then four, with six as a bounded stress case.
5. Profile the critical path and optimize measured bottlenecks using kernels, scheduling, prefetching, caching, placement, or runtime changes.
6. Requalify correctness and lifecycle after every meaningful optimization.

Completion:
Produce a reproducible serving configuration, verified shelf entry, launch/stop instructions, pinned identities, prefix-cache evidence, concurrent coherence checks, and matched latency results showing what improved and why.
Do not promote an engine merely because it loads, gives readable answers, or produces an attractive tokens/second number.
```
