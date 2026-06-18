# Best / Most-Current Upstream vLLM-XPU Config for Intel Arc Pro B70 (Battlemage)

*Research snapshot: 2026-06-17. Scope: the **upstream from-source `vllm-project/vllm` (XPU)** axis for a **dense Qwen3-14B FP8** workload on a single Arc Pro B70 (Battlemage BMG-G31, 32 GB). This is a different axis than [`04_vllm_versions.md`](./04_vllm_versions.md), which covers the Intel `llm-scaler` Docker distribution and the Qwen3.6 / GDN-DeltaNet MTP path. Where the two overlap, 04 is the authority on the Docker/MTP story; this doc focuses on "build newer upstream vs stay on c51df4300," compilation, dense-model spec-decode, and FP8-KV.*

*Caveat up front: Battlemage/B70 upstream support is young and moving weekly. Anything older than ~30 days is potentially stale. Re-check the [vLLM Releases](https://github.com/vllm-project/vllm/releases) and [XPU doc](https://docs.vllm.ai/en/stable/getting_started/installation/gpu.html#intel-xpu) before pinning. Items flagged ⚠️ are single-source or cross-architecture (CUDA-derived) and should be benchmarked on your actual B70 before you trust them.*

---

## 0. Executive answer

| Question | Verdict |
|---|---|
| **Build newer upstream vs stay on c51df4300?** | **Conditional yes — target `v0.23.0` (tag), but stage it; do not blind-bump `main`.** c51df4300 (0.20.2rc1.dev2) is fine and proven for Qwen3-14B FP8. v0.23.0 is the first *tagged* release that lands the XPU GDN-MTP kernels (#43565) + fused GDN (#43534) **and** keeps dense Qwen3/Llama/Mistral on the new default Model-Runner-V2 path (#43458). For a **dense** Qwen3-14B you gain little from v0.23.0 itself (MTP/GDN is a Qwen3.5/3.6 feature, not dense); the reason to move is newer XPU FP8/quant kernels (vllm-xpu-kernel v0.1.7) and bug-fix accumulation. **Bump to the `v0.23.0` tag, not random `main`** — XPU has a documented history of per-release regressions (GPTQ broke in 0.19.0, #39474). |
| **Compilation (`--compilation-config`)?** | **No real win on B70 today. Keep `--enforce-eager`.** XPU has *no* CUDA-graph-equivalent capture, so `cudagraph_mode=PIECEWISE` and `use_inductor_graph_partition=true` are silently **no-ops** (vLLM logs "XPU Graph is not supported … disabling cudagraph_mode"). Inductor-only `-O1` compile is still maturing and a known breakage source. |
| **Spec-decode for dense Qwen3-14B?** | **Yes — draft-model with Qwen3-0.6B is the most-likely single-stream win.** Qwen3-0.6B is vocab-compatible (shared 151936-vocab `Qwen2Tokenizer`). ngram is a near-free second attempt for code/RAG. EAGLE3 is officially "supported on B-series" but needs a Qwen3-14B-specific trained head. |
| **Top other levers?** | **(1)** keep the default **Flash-Attn** XPU backend; **(2)** `--block-size 64` + `--max-num-batched-tokens 8192` + `--gpu-memory-utilization 0.9`. **FP8-KV is conditional** (forces Triton backend — see §4). |

---

## 1. Which vLLM tag/commit builds + runs well on XPU/Battlemage

### 1a. Version landscape (upstream, from-source)

| Upstream version | XPU-relevant content | Relevance to dense Qwen3-14B FP8 on B70 |
|---|---|---|
| **c51df4300** = `0.20.2rc1.dev2` (current) | Your proven baseline. Qwen3-14B FP8 works. | **Known-good. The thing to beat.** |
| **v0.22.0 / v0.22.1** | XPU GPTQ-int4 (#37844), mxfp8 MoE, FP8 block-scaled, MXFP4 MoE fallback; GDN output-proj flatten for Qwen3.5/3.6 (#42311). | Quant maturity; nothing dense-Qwen3-14B *needs*. |
| **v0.23.0** (newest tag) | **vllm-xpu-kernel v0.1.7**, block-FP8-MoE (#42139), block-scaled W8A8 FP8 (#39968), RMSNorm/act-quant fusions (#43963), **XPU GDN-attention MTP (#43565)**, fused GDN gated-delta-rule kernels (#43534), Triton selective-scan (#43421), XPU DeepSeek-V4 decode (#42953), CPU/tiering offload on XPU (#36423). **Model-Runner-V2 became default for Qwen3 + Llama + Mistral dense (#43458).** | **The recommended target.** Dense Qwen3 stays working under MRv2; you pick up newer XPU FP8/quant kernels + fixes. |
| **`main` (post-v0.23.0)** | Active churn. DeepSeek-V4 was *detached* from `torch.compile` (#43746/#43891) — a hint that compile paths are still being reworked. | **Not recommended for a "just works" box.** XPU regressions land and get fixed out of band; you'd be QA. |

### 1b. Yes/No: build newer?

**Recommendation: YES, move to the `v0.23.0` *tag* — but treat it as a staged upgrade, keep c51df4300 as the rollback.** Rationale:

- **For your specific workload (dense Qwen3-14B FP8), the headline v0.23.0 features (GDN-MTP #43565, fused GDN) do nothing** — those are for Qwen3.5/3.6 Gated-DeltaNet models, not dense Qwen3. So the upgrade is *not* urgent on feature grounds.
- **What you actually gain:** newer `vllm-xpu-kernel v0.1.7`, block-scaled W8A8 FP8 path, RMSNorm/activation-quant fusions (#43963) — these are the kernels that move FP8 throughput on Battlemage, and they are the legitimate reason to move forward.
- **Does v0.23.0 keep Qwen3/Gemma working on XPU?** Qwen3-14B is explicitly listed as XPU-supported in the [XPU supported-models doc](https://docs.vllm.ai/en/stable/models/hardware_supported_models/xpu/), and #43458 made Model-Runner-V2 the default specifically for **Qwen3 + Llama + Mistral dense** — i.e., the project intends dense Qwen3 to work on the new default path. ⚠️ *Gemma on XPU under MRv2 is plausible but not explicitly confirmed in a source I could pin; validate Gemma separately if you need it.*
- **Why NOT just build `main`:** XPU is a second-class CI target and has a **documented pattern of per-release regressions** — e.g. GPTQ models stopped loading on XPU in v0.19.0 because XPU branches were dropped from `gptq.py` ([#39474](https://github.com/vllm-project/vllm/issues/39474)), and dual-B70 TP=2 GP-faults on `intel/vllm:0.17.0-xpu` ([#41663](https://github.com/vllm-project/vllm/issues/41663)). A tagged release has at least had a release-gate pass; arbitrary `main` has not.

**Net:** Bump to `v0.23.0` to bank the newer FP8/quant kernels and fixes; **benchmark it head-to-head against c51df4300 on your exact Qwen3-14B FP8 run before retiring the old build.** If v0.23.0 regresses your throughput or correctness, stay on c51df4300 — it owes you nothing.

### 1c. Build notes / pins (Dockerfile.xpu, from-source)

Per the [XPU install doc](https://docs.vllm.ai/en/stable/getting_started/installation/gpu.html#intel-xpu):

- **Python 3.12 is mandatory** — "The provided vllm-xpu-kernels whl is Python3.12 specific so this version is a MUST."
- **PyTorch:** v0.23.0's `requirements/xpu.txt` targets **torch 2.11 (xpu)**; the matching triton is **triton-xpu==3.7.0**. (c51df4300 was on the torch 2.10+xpu line — a torch bump is part of moving to v0.23.0, and is the riskiest single change. Pin torch exactly to what `requirements/xpu.txt` specifies for the tag you check out.)
- **Replace stock triton with triton-xpu** *after* installing requirements:
  ```bash
  pip uninstall -y triton triton-xpu
  pip install triton-xpu==3.7.0 --extra-index-url https://download.pytorch.org/whl/xpu
  ```
- **Build:** `VLLM_TARGET_DEVICE=xpu pip install --no-build-isolation -e . -v`
- **Docker:** `docker build -f docker/Dockerfile.xpu -t vllm-xpu-env --shm-size=4g .`
- ⚠️ The doc does **not** pin IPEX / oneCCL / Level-Zero versions. If you hit init crashes, **match Intel's validated BOM** (per #41663: Ubuntu 25.04 / kernel 6.14, oneCCL 2021.15.7.8) rather than whatever your host happens to have — the GP-fault reproducer ran a *different* host BOM (PyTorch 2.10.0+xpu, oneAPI 2025.3.2, oneCCL 2021.17.2-5, L0 25.48.36300.8) and that mismatch was judged the root cause, not vLLM itself.

> **Alternative worth knowing:** Intel's `llm-scaler-vllm:0.14.0-b8.3.1` Docker image (June 2026) is the *officially B70-validated* stack, but its internal base wraps an **older** upstream vLLM (~0.11.1 / PyTorch 2.9 lineage per the b8.2.1 notes) — i.e. **behind** your c51df4300. So for *dense Qwen3-14B FP8* the llm-scaler image is **not** "newer/better" than what you have; it's a hardened-but-older path. Use it only if you want a turnkey validated BOM, not for being on the latest kernels. (Full llm-scaler analysis: [`04_vllm_versions.md`](./04_vllm_versions.md).)

---

## 2. Compilation on XPU

**Bottom line: keep `--enforce-eager`. The compilation knobs the prompt asks about are no-ops or breakage on Battlemage.**

- **No CUDA-graph-equivalent on XPU.** PyTorch-XPU lacks a graph-capture API analogous to CUDAGraph, so vLLM logs `"XPU Graph is not supported in the current PyTorch version, disabling cudagraph_mode"` and turns cudagraphs off **regardless of what `cudagraph_mode` you pass** ([#36350](https://github.com/vllm-project/vllm/issues/36350), [#26970 XPU graph roadmap](https://github.com/vllm-project/vllm/issues/26970), [vllm-xpu-kernels #141](https://github.com/vllm-project/vllm-xpu-kernels/issues/141)). So **`cudagraph_mode=PIECEWISE` / `FULL_AND_PIECEWISE` is silently inert on B70** — it won't error, it just won't capture anything.
- **`use_inductor_graph_partition=true` is also a no-op** — it only does work when cudagraphs are active (it partitions the captured graph). With no XPU graphs there is nothing to partition. No measured B70 benefit exists.
- **Inductor-only torch.compile (`-O1`, no cudagraphs) is still maturing and is a real breakage source on XPU.** Reported failures: linear-attention custom ops calling `torch.cuda._exchange_device()` on the XPU build (crashes Qwen3.5/Gemma-class models on first request, [#36350](https://github.com/vllm-project/vllm/issues/36350)); Dynamo/Inductor compile-stage failures under TP on B70 ([#41663](https://github.com/vllm-project/vllm/issues/41663)); modules falling back to eager when the platform lacks TorchInductor support ([vllm-omni #2374](https://github.com/vllm-project/vllm-omni/issues/2374)). Intel's own B-series blog says torch.compile is supported only on **FP16/BF16 paths** ([Intel Arc Pro B blog](https://vllm.ai/blog/2025-11-11-intel-arc-pro-b)) — not the FP8 path you run.
- **Every working B70 config in the wild uses `--enforce-eager`** — Intel's own sample launch command does ([Intel Arc Pro B blog](https://vllm.ai/blog/2025-11-11-intel-arc-pro-b)), and so does the #41663 reproducer. On B70, **TTFT/throughput is dominated by the eager kernel quality in `vllm-xpu-kernels`, not by compilation.**

### Recommended `--compilation-config`

**Primary (recommended): don't use one — pass `--enforce-eager`.**

**If you want to *experiment* with Inductor fusion** (dense Qwen3-14B is the model class most likely to survive it, since the crashes are in linear-attn ops it doesn't have):
```bash
--compilation-config '{"level":1,"cudagraph_mode":"NONE","use_inductor_graph_partition":false}'
```
⚠️ **Unverified on Battlemage.** No published B70 numbers show eager-vs-Inductor-`-O1` benefit. Benchmark against `--enforce-eager`; if you see compile-stage errors or no TTFT/throughput gain, revert. Do **not** ship `cudagraph_mode=PIECEWISE` expecting graphs — you will not get them on XPU.

---

## 3. Speculative decoding on XPU for dense Qwen3-14B (no native MTP)

Dense Qwen3-14B has **no native MTP head** and is **not** a Gated-DeltaNet model, so the `gdn_attention … spec_sequence_masks` blocker that kills MTP on Qwen3.5/3.6 ([llm-scaler #386](https://github.com/intel/llm-scaler/issues/386), see [`04_vllm_versions.md`](./04_vllm_versions.md)) **does not apply to you.** You have the easier, better-supported draft-model + ngram paths available.

### 3a. Draft-model spec-decode — recommended, most likely single-stream win

- **Syntax:** `--speculative-config '{"model":"Qwen/Qwen3-0.6B","num_speculative_tokens":4}'`
- **Draft choice: Qwen3-0.6B is correct and vocab-compatible.** All dense Qwen3 models (0.6B/1.7B/4B/8B/14B/32B) share the identical `Qwen2Tokenizer` with vocab **151936**, so no token-remap (t2d/d2t) is needed. vLLM's own draft-model docs pair Qwen3-0.6B as the drafter for Qwen3-8B ([draft_model docs](https://docs.vllm.ai/en/latest/features/speculative_decoding/draft_model/)) — same family, same logic for the 14B verifier. (The only quirk: 0.6B/1.7B tie embeddings while 14B doesn't — irrelevant to drafting.)
- **`num_speculative_tokens`:** start at **3–4** for single-stream (lower acceptance variance), test 5.
- **XPU status:** ⚠️ The generic draft-model path is **not flagged as XPU-blocked anywhere**, and runs over the same V1 spec-decode machinery as ngram, which Intel lists as B-series-supported. But the Intel B-series blog enumerates **ngram / EAGLE / EAGLE3** explicitly and does *not* name `draft_model` — so draft-model on XPU is **plausible but not explicitly source-confirmed.** Benchmark batch=1 against an `--enforce-eager` no-spec baseline before trusting it.

### 3b. ngram — near-free second attempt

- **Syntax:** `--speculative-config '{"method":"ngram","num_speculative_tokens":4,"prompt_lookup_min":2,"prompt_lookup_max":5}'`
- **XPU status:** explicitly listed as supported on Arc Pro B-series ([Intel Arc Pro B blog](https://vllm.ai/blog/2025-11-11-intel-arc-pro-b)). No model attention change; drafting is CPU/Python — lowest-risk thing to try.
- **When it wins:** repetitive / long-context / code / RAG workloads. Little to no gain on free-form chat. The known ngram corruption bug ([#39273](https://github.com/vllm-project/vllm/issues/39273)) is a **GDN-state-rollback** issue on hybrid Qwen3.5/Qwen3-Next models — **not** dense Qwen3-14B, **not** XPU-specific. You are unaffected.

### 3c. EAGLE / EAGLE3 — highest ceiling, most effort

- Officially "supported on B-series" ([Intel Arc Pro B blog](https://vllm.ai/blog/2025-11-11-intel-arc-pro-b)). EAGLE3 needs a **trained Qwen3-14B-specific EAGLE head**; if no good public head exists, this is more work than draft-model. General EAGLE caveat: lower-than-expected speedup is a known issue across backends ([#9565](https://github.com/vllm-project/vllm/issues/9565)). ⚠️ One search hit claimed "vLLM 0.19.0 on Arc Pro B70 runs EAGLE3"; could not confirm the primary source — treat as unverified.

### 3d. Which actually gives single-stream (batch=1) speedup on B70?

Spec-decode helps **most** at batch=1 (decode is memory-bound, the B70 is underutilized). Ranked for dense Qwen3-14B FP8:

1. **Draft-model Qwen3-0.6B** — most general, vocab-matched, simplest. **Start here.**
2. **ngram** — only wins on repetitive/long-context/code/RAG, but near-free to try.
3. **EAGLE3** — highest ceiling, needs a matching trained head, most fragile.

**Known failure modes:** spec-decode regresses at *high* batch (verify cost > draft savings — keep it for low-concurrency single-stream); ⚠️ XPU rejection-sampler/kernel maturity lags CUDA *(this is inference, not a cited source)* — **always benchmark batch=1 against the `--enforce-eager` no-spec baseline before trusting any speedup number.**

### Recommended spec-decode recipe
```bash
vllm serve <Qwen3-14B-FP8> \
  --enforce-eager \
  --speculative-config '{"model":"Qwen/Qwen3-0.6B","num_speculative_tokens":4}'
# measure TPOT/ITL vs the same command WITHOUT --speculative-config.
# For code/RAG/long-context, also A/B the ngram config from §3b.
```

---

## 4. Other B70 throughput levers in current vLLM-XPU

### 4a. FP8 KV cache (`--kv-cache-dtype fp8`) — conditional, with a backend gotcha

- **It works on XPU, but ONLY with the Triton attention backend.** vLLM's XPU platform **hard-asserts** `"XPU only support fp8 kv cache with triton backend"` (supported: `fp8_e4m3` / `fp8_e5m2` / `fp8`) ([XPU platform API](https://docs.vllm.ai/en/v0.11.0/api/vllm/platforms/xpu.html)). **Enabling FP8-KV therefore forces you off the default Flash-Attn path onto Triton** — a real speed/correctness tradeoff, not a free win.
- **Precision:** default online FP8 dtype is **E5M2**; switch to higher-precision **E4M3** with `VLLM_XPU_FP8_DTYPE=e4m3` ([env vars](https://docs.vllm.ai/en/stable/configuration/env_vars/)). Prefer **`--kv-cache-dtype fp8_e4m3`**. E5M2 silent-corruption bugs exist on some models ([#41343](https://github.com/vllm-project/vllm/issues/41343)).
- **Benefit on B70:** mainly **~halved KV memory** → more context / larger batch in 32 GB, **not** a guaranteed decode speedup (and you pay the Flash→Triton switch). ⚠️ No B70-specific FP8-KV accuracy data exists; the only published study is NVIDIA-only ([FP8-KV blog](https://vllm.ai/blog/2026-04-22-fp8-kvcache)). **Verify accuracy on your eval set before relying on it.** For a single 14B FP8 model in 32 GB you likely have KV headroom already — only reach for FP8-KV when you need long context or higher batch.

### 4b. Attention backend

- **XPU default is Flash Attention** (IPEX `flash_attn_varlen_func` / SYCL-TLA kernels); Triton is the fallback and is **forced** for fp32 and for fp8-KV ([XPU platform API](https://docs.vllm.ai/en/v0.11.0/api/vllm/platforms/xpu.html), [Triton backend deep-dive](https://vllm.ai/blog/2026-03-04-vllm-triton-backend-deep-dive)).
- Override via `VLLM_ATTENTION_BACKEND=FLASH_ATTN` (or `TRITON_ATTN`). **For Qwen3-14B with FP16/FP8 weights, keep the default Flash Attention** for best perf; only move to Triton if you adopt FP8-KV.

### 4c. Chunked prefill / batching / memory

Intel's official B70 examples ([llm-scaler vLLM README](https://github.com/intel/llm-scaler/blob/main/vllm/README.md), [Intel Arc Pro B blog](https://vllm.ai/blog/2025-11-11-intel-arc-pro-b)):

- **`--block-size 64`** — XPU forces block_size 64 in the V1/IPEX chunked-prefill path. Use it.
- **`--max-num-batched-tokens 8192`** — Intel's default. **Lower → 2048–4096 for better single-stream ITL/TPOT; raise for better TTFT/batch throughput** ([optimization docs](https://docs.vllm.ai/en/stable/configuration/optimization/)).
- **`--gpu-memory-utilization 0.9`** for a 32 GB B70.
- **`-tp 1`** for single GPU.

### 4d. XPU env vars that matter for single-GPU

| Env var | Effect | Verdict |
|---|---|---|
| `VLLM_XPU_FP8_DTYPE=e4m3` | Higher-precision FP8 KV/quant | Use if enabling FP8-KV |
| `VLLM_WORKER_MULTIPROC_METHOD=spawn` | XPU default/required | Keep |
| `ZE_AFFINITY_MASK=0` | Pin the single device | Keep (single-GPU) |
| `VLLM_OFFLOAD_WEIGHTS_BEFORE_QUANT=1` | Avoid OOM during online FP8 quant load | Keep if you OOM at load |
| `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1` | Allow long max-model-len | As needed |
| `VLLM_XPU_ENABLE_XPU_GRAPH`, `VLLM_XPU_USE_SAMPLER_KERNEL` | Experimental XPU graph / sampler kernel | ⚠️ Unverified single-GPU levers; leave default (`0`) unless you A/B-test them |
| `CCL_*`, `TORCH_LLM_ALLREDUCE`, `SYCL_UR_*` | Multi-GPU/TP collective tuning | **Irrelevant for a single B70.** Ignore. |

⚠️ No SYCL_* / IGC_* perf var is documented as validated for single-GPU throughput. The practitioner guide ([roger.lol](https://www.roger.lol/blog/accessible-ai-vllm-on-intel-arc)) uses `LD_LIBRARY_PATH` hygiene only (keep SYCL/UR from the venv torch), not perf tuning.

### Top 2 other levers (beyond compilation/spec-decode)
1. **Keep the default Flash-Attn XPU backend** (don't force Triton unless you adopt FP8-KV).
2. **`--block-size 64` + `--max-num-batched-tokens 8192` + `--gpu-memory-utilization 0.9`** (tune `max-num-batched-tokens` down to 2048–4096 if optimizing single-stream ITL).

---

## 5. Consolidated recommended launch (dense Qwen3-14B FP8, single B70)

```bash
# Target build: vLLM v0.23.0 tag, from source (Python 3.12, torch 2.11+xpu, triton-xpu 3.7.0)
# Keep c51df4300 as rollback; A/B both on your exact workload before retiring it.

export VLLM_WORKER_MULTIPROC_METHOD=spawn
export ZE_AFFINITY_MASK=0
export VLLM_OFFLOAD_WEIGHTS_BEFORE_QUANT=1     # if you OOM during FP8 quant load

vllm serve <Qwen3-14B-FP8> \
  --dtype float16 \
  --enforce-eager \
  --block-size 64 \
  --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.9 \
  -tp 1 \
  --speculative-config '{"model":"Qwen/Qwen3-0.6B","num_speculative_tokens":4}'

# Optional, only if you need long context / bigger batch AND verify accuracy:
#   --kv-cache-dtype fp8_e4m3   (forces Triton backend → set VLLM_XPU_FP8_DTYPE=e4m3)
# Do NOT add --compilation-config expecting cudagraphs (silently disabled on XPU).
```

---

## 6. Confidence / caveats

- ✅ **Well-sourced:** v0.23.0 XPU content & PR numbers (release notes); `--enforce-eager` is the working B70 norm; XPU has no cudagraph capture; Qwen3-0.6B vocab-compat (151936) for drafting; FP8-KV requires Triton backend on XPU (hard assert); default backend = Flash-Attn; Intel's `block-size 64 / 8192 / 0.9` defaults.
- ⚠️ **Unverified / cross-arch (benchmark on B70):** Inductor-`-O1` benefit; draft-model (vs ngram/EAGLE) explicit XPU support; FP8-KV accuracy on Battlemage; `VLLM_XPU_ENABLE_XPU_GRAPH` / sampler-kernel as perf levers; Gemma-on-XPU under MRv2; EAGLE3-on-B70 claim.
- 🟥 **Moving target:** B70 upstream support is young; `main` ships regressions out of band (e.g. GPTQ broke in 0.19.0, #39474). Pin to the **v0.23.0 tag**, re-check Releases before each bump, and always keep c51df4300 as rollback.
- ↔️ **Scope boundary:** For Qwen3.6 / Gated-DeltaNet **MTP**, the llm-scaler Docker path, and AutoRound W4A16 — see [`04_vllm_versions.md`](./04_vllm_versions.md). That MTP-on-GDN blocker does **not** apply to dense Qwen3-14B (this doc's subject).

---

## Sources

- vLLM v0.23.0 release notes: <https://github.com/vllm-project/vllm/releases/tag/v0.23.0>
- vLLM v0.22.0 release notes: <https://github.com/vllm-project/vllm/releases/tag/v0.22.0>
- vLLM Releases index: <https://github.com/vllm-project/vllm/releases>
- XPU install / build doc (Python 3.12, torch 2.11, triton-xpu 3.7.0, Dockerfile.xpu): <https://docs.vllm.ai/en/stable/getting_started/installation/gpu.html#intel-xpu>
- XPU supported-models (Qwen3-14B listed): <https://docs.vllm.ai/en/stable/models/hardware_supported_models/xpu/>
- XPU platform API (fp8-KV requires Triton; default Flash-Attn): <https://docs.vllm.ai/en/v0.11.0/api/vllm/platforms/xpu.html>
- vLLM env vars (VLLM_XPU_FP8_DTYPE etc.): <https://docs.vllm.ai/en/stable/configuration/env_vars/>
- Optimization / chunked-prefill tuning doc: <https://docs.vllm.ai/en/stable/configuration/optimization/>
- Speculative decoding — draft model (Qwen3-0.6B drafter): <https://docs.vllm.ai/en/latest/features/speculative_decoding/draft_model/>
- Intel Arc Pro B-series vLLM blog (ngram/EAGLE/EAGLE3, --enforce-eager, FP16/BF16-only torch.compile, sample launch): <https://vllm.ai/blog/2025-11-11-intel-arc-pro-b>
- FP8 KV-cache & attention quant blog (NVIDIA-only accuracy study): <https://vllm.ai/blog/2026-04-22-fp8-kvcache>
- Triton backend deep-dive: <https://vllm.ai/blog/2026-03-04-vllm-triton-backend-deep-dive>
- vLLM torch.compile integration doc: <https://docs.vllm.ai/en/latest/design/torch_compile/>
- #36350 — XPU graph not supported / linear-attn `torch.cuda._exchange_device` crash: <https://github.com/vllm-project/vllm/issues/36350>
- #26970 — XPU graph support roadmap: <https://github.com/vllm-project/vllm/issues/26970>
- vllm-xpu-kernels #141 — XPU graph roadmap: <https://github.com/vllm-project/vllm-xpu-kernels/issues/141>
- #41663 — dual-B70 TP=2 GP fault / xe BCS reset on intel/vllm:0.17.0-xpu (BOM mismatch): <https://github.com/vllm-project/vllm/issues/41663>
- #39474 — GPTQ regression on XPU in v0.19.0 (per-release XPU breakage example): <https://github.com/vllm-project/vllm/issues/39474>
- #39273 — ngram corruption on GDN/hybrid models (NOT dense Qwen3): <https://github.com/vllm-project/vllm/issues/39273>
- #41343 — FP8 E5M2 KV silent corruption: <https://github.com/vllm-project/vllm/issues/41343>
- #9565 — EAGLE lower-than-expected speedup: <https://github.com/vllm-project/vllm/issues/9565>
- #35638 — XPU best-practices request (B-series args): <https://github.com/vllm-project/vllm/issues/35638>
- vllm-omni #2374 — torch.compile bypassed where TorchInductor unsupported on XPU: <https://github.com/vllm-project/vllm-omni/issues/2374>
- intel/llm-scaler vLLM README (B70 validated stack, block-size 64, env): <https://github.com/intel/llm-scaler/blob/main/vllm/README.md>
- Phoronix — llm-scaler 0.14.0-b8.2 official B70 support: <https://www.phoronix.com/news/Intel-LLM-Scaler-vllm-0.14-b8.2>
- roger.lol — vLLM on Intel Arc practitioner guide: <https://www.roger.lol/blog/accessible-ai-vllm-on-intel-arc>
- Companion doc (llm-scaler / Qwen3.6 / MTP / AutoRound): [`04_vllm_versions.md`](./04_vllm_versions.md)
