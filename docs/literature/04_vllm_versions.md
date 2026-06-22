# vLLM-XPU / Intel LLM-Scaler Version Landscape for Qwen3.6 on Arc Pro B70 (Battlemage)

> **[!] SUPERSEDED PIN (2026-06-22): use `vllm-xpu-env:v0230` (vLLM 0.23.0), NOT the 0.14.x llm-scaler image below.**
> Per CLAUDE.md + the 2026-06-22 results, v0230 is our newest/most-capable stack: native GDN-MTP (#43565, +79% MTP),
> Triton fused-MoE (int8 + int4 35B-A3B serve), Quark int8 dispatch, graph capture. The 0.14.x llm-scaler image has NO
> `_moe_C` op suite (int8 MoE hard-fails) and burned multiple agent-days (docs/kernel/20). Newest-first: v0230 > :tf (0.20.2rc1) > 0.14.x.
> The version-landscape analysis below is kept for history; treat the "pin 0.14.0-b8.3.1" recommendation as OBSOLETE.

*Research snapshot: 2026-06-17. Author: literature review. Battlemage/B70 inference support is young and moving fast — treat anything older than ~30 days as potentially stale, and re-check the [llm-scaler Releases](https://github.com/intel/llm-scaler/blob/main/Releases.md) page before pinning.*

---

## 0. TL;DR recommendation

- **Pin this image:** `intel/llm-scaler-vllm:0.14.0-b8.3.1` (June 2026). This is the version Intel's own README explicitly recommends for "Qwen3.5/3.6-27B, Qwen3.5/3.6-35B-A3B and Qwen3.5-122B-A10B." `b8.3` (May 2026) is the fallback; `latest` currently aliases `b8.3` — **do not use `latest`**.
- **Two different "vLLM versions" are in play and people conflate them:**
  - **Intel `llm-scaler-vllm`** — Intel's hardened fork/distribution shipped as Docker images tagged `0.14.0-b8.x` (the `0.14.0` is Intel's internal base, *not* the upstream vLLM version). This is the only thing with documented, validated **B70** support. **Use this.**
  - **Upstream `vllm` (xpu)** — community/`intel/vllm:*-xpu` images built from `vllm-project/vllm`. The user's "vLLM 0.20.1 / 0.17.0-xpu" references are this lineage. Battlemage TP=2 on these is **buggy/unvalidated** (see GP-fault issue [#41663](https://github.com/vllm-project/vllm/issues/41663)).
- **Quant for B70 today = online `--quantization fp8` or `--quantization sym_int4`** inside llm-scaler. Pre-quantized **AutoRound W4A16** is *not* a first-class llm-scaler path (no XPU AutoRound kernel is documented in the README); the documented INT4 route is Intel's `sym_int4` online quant via `vllm_int4_for_multi_arc.so`.
- **MTP on XPU for Qwen3.6 (Gated-DeltaNet) is NOT production-ready.** It hits `XPU gdn_attention does not yet support 'spec_sequence_masks'` ([llm-scaler #386](https://github.com/intel/llm-scaler/issues/386)). The upstream fix (`#43565 GDN-attention MTP`) only landed in **upstream vLLM v0.23.0**, which is *newer than the base inside b8.3.1* — so on B70 today MTP is a **manual patch / negative-EV experiment**, not a supported feature.

---

## 1. Version timeline

### 1a. Intel `llm-scaler-vllm` Docker tags

Source: [Releases.md](https://github.com/intel/llm-scaler/blob/main/Releases.md), [Docker Hub tags](https://hub.docker.com/r/intel/llm-scaler-vllm/tags), [vllm/README.md](https://github.com/intel/llm-scaler/blob/main/vllm/README.md), [Phoronix b8.2](https://www.phoronix.com/news/Intel-LLM-Scaler-vllm-0.14-b8.2).

| Image tag | Date | What it added (relevant to B70 / Qwen3.6) |
|---|---|---|
| `1.0` (PV stable) | 08/2025 | First "Project Battlematrix" PV release; Arc Pro B-series (B60) focus. No Qwen3.6. |
| `0.10.2-b6` / `1.2` | 11–12/2025 | Earlier Battlemage B-series serving; multi-Arc TP. |
| `0.11.1-b7` / `1.3` | 01/2026 | Iterative model + perf updates. |
| `0.14.0-b8` | 03/2026 | Base bump to internal `0.14.0`; broader Qwen3 family. |
| `0.14.0-b8.1` | 03/2026 | The image used in [#386](https://github.com/intel/llm-scaler/issues/386) where MTP crashed (`gdn_attention ... spec_sequence_masks`). FP8 path present. |
| **`0.14.0-b8.2` / `1.4` / `0.14.0-b8.2.1`** | **05/2026** | **Official Arc Pro B70 (BMG-G31, 32 GB, `8086:e223`) support announced** (Phoronix, wccftech). New platform image `intel/llm-scaler-platform:26.18.8.2`. |
| `0.14.0-b8.3` | 05/2026 | Perf improvements for **Qwen3.5/3.6** series & Qwen3-Coder-Next; **model streaming load** to cut peak memory. `latest` → this. |
| **`0.14.0-b8.3.1`** | **06/2026** | **FP8 KV Cache** enabled; bug fixes for Qwen3/Qwen3.5/3.6. **README-recommended for Qwen3.6.** |

> The `1.x` tags are "stable PV" aliases and the `0.14.0-b8.x` tags are the beta channel; `1.4` and `0.14.0-b8.2.1` share a digest, as do several other pairs. Intel's README: *"Do **NOT** use the `latest` tag. Instead, go to the Releases page and pull the exact beta version."*

### 1b. Upstream `vllm-project/vllm` (XPU) feature landing

Source: vLLM release notes [v0.22.0](https://github.com/vllm-project/vllm/releases/tag/v0.22.0), [v0.23.0](https://github.com/vllm-project/vllm/releases/tag/v0.23.0); [XPU supported-models doc v0.20.0](https://docs.vllm.ai/en/v0.20.0/models/hardware_supported_models/xpu/).

| Upstream area | Where it landed | Notes for B70/Qwen3.6 |
|---|---|---|
| XPU baseline (Arc Pro B-series listed; FP16 + Dynamic FP8 only; AWQ/GPTQ/AutoRound *not* in the XPU table) | ≤ v0.20.0 | Doc lists "Intel Arc Pro B-Series" only — **Battlemage/B70 not explicitly enumerated**; Qwen3-Next/3.6 absent. |
| XPU **GPTQ int4** (#37844), **mxfp8 MoE** (#41918), **FP8 block-scaled** (#42952), MXFP4 MoE fallback (#42951) | **v0.22.0** | XPU quant maturing upstream, but ahead of the base inside llm-scaler b8.3.1. |
| GDN attention work: **GDN output-proj flatten for Qwen3.5/3.6** (#42311), fused GDN AMX-CPU (#42707) | v0.22.0 | DeltaNet plumbing for Qwen3.5/3.6. |
| **XPU GDN-attention MTP (#43565)**, fused GDN gated-delta-rule kernels (#43534), XPU DeepSeek-V4 decode (#42953) | **v0.23.0** | **This is the upstream fix that makes MTP+GDN work on XPU.** It is *newer* than the upstream base shipped in llm-scaler `0.14.0-b8.3.1`, so it is **not** in the recommended B70 image yet. |

**Mapping the user's "vLLM 0.20.1 / 0.17.0-xpu" numbers:** these are upstream `vllm` versions (or `intel/vllm:*-xpu` images), *not* llm-scaler tags. The dual-B70 working run on "0.20.1, INT4 AutoRound W4A16, TP=2, flash_attn, MTP (patched fallback)" is a **hand-built upstream-vLLM stack with a manual GDN/MTP patch**, distinct from the Intel-supported `llm-scaler-vllm` path. That explains why MTP needed patching: the supported GDN-MTP fix only exists upstream from v0.23.0.

---

## 2. MTP / speculative decoding state on XPU for Qwen3.6 (DeltaNet)

**Bottom line: immature on XPU. Single-source-positive, multi-source-negative.**

- **The blocker:** Qwen3.6 uses Gated-DeltaNet (linear/recurrent attention interleaved with full attention). Speculative decoding requires rolling back the recurrent state on token rejection, which standard pipelines can't do. On XPU specifically, enabling `speculative-config {"method":"mtp",...}` throws **`NotImplementedError: XPU gdn_attention does not yet support 'spec_sequence_masks'`** raised in `vllm/model_executor/models/qwen3_5.py` (`forward_xpu`). Reported on `intel/llm-scaler-vllm:0.14.0-b8.1`, Qwen3.5-9B, FP8 — [llm-scaler #386](https://github.com/intel/llm-scaler/issues/386). As of this snapshot the issue shows **no posted fix/workaround** in llm-scaler.
- **Is the "patched MTP fallback" upstreamed?** Not into the recommended B70 image. The genuine upstream support is **`#43565 GDN-attention MTP` in vLLM v0.23.0** ([release notes](https://github.com/vllm-project/vllm/releases/tag/v0.23.0)). On the user's dual-B70 0.20.1 stack, "MTP (patched fallback)" = a **manual patch** to the XPU GDN path (back-porting/forcing a fallback), *not* a feature toggle. Treat it as experimental.
- **The MTP head itself must be preserved in BF16** for native vLLM MTP to work. Pre-quantized Qwen3.6 AutoRound checkpoints keep `mtp.fc` (and `linear_attn.in_proj_a/b`, layernorms, router gates) in BF16 precisely for this reason — see [Lorbus/Qwen3.6-27B-int4-AutoRound](https://huggingface.co/Lorbus/Qwen3.6-27B-int4-AutoRound). If your INT4 path quantizes the MTP head, MTP won't draft.
- **Single-GPU vs TP2 (matches the user's negative result):**
  - On CUDA, MTP gives ~2x (e.g. ~60 tok/s vs ~30 on RTX 5090, ~85% draft accept) per the Lorbus card.
  - On B70, the user found **TP2+MTP slower than TP2 non-MTP and slower than single-B70 MTP** — consistent with vLLM **disabling XPU graph capture for TP2 communication ops**, so the per-step overhead of verify+collective dominates the draft savings. This is a known structural limitation, not a tuning miss.
- **Which spec-decode methods work on XPU right now:**
  - **ngram** — most likely to "just work" (no model attention change; CPU/Python drafting). Lowest-risk speedup attempt on B70, though community benchmarks (e.g. RTX 3090 A3B) show *no net speedup* on MoE+A3B even on CUDA, so don't expect much.
  - **MTP** — needs the v0.23.0 GDN-MTP fix (XPU `#43565`) or a manual patch; broken on stock b8.3.1. Best theoretical method for Qwen3.6 since the head is trained against the verifier.
  - **EAGLE / EAGLE3** — upstream EAGLE3 speculator work exists (#42764) but no XPU+DeltaNet validation found; assume unsupported on B70.

---

## 3. Quantization formats on vLLM-XPU for B70

Source: [llm-scaler vllm/README §2.2 + §3.2](https://github.com/intel/llm-scaler/blob/main/vllm/README.md), [intel/auto-round](https://github.com/intel/auto-round), [vLLM AutoRound blog](https://vllm.ai/blog/2026-06-02-vllm-omni-autoround), HF model cards.

| Format | Status on llm-scaler B70 | How invoked | Uses XMX well? |
|---|---|---|---|
| **Online FP8** (W8A8 dynamic) | **Supported, primary** | `--quantization fp8` (+ optional `--kv-cache-dtype fp8` on b8.3.1) | Yes — FP8 is the best-exercised XMX path on Battlemage; this is what Intel benchmarks. |
| **Online INT4** (`sym_int4`, dynamic symmetric) | **Supported** (Intel's own kernel) | `export VLLM_QUANTIZE_Q40_LIB=/usr/local/lib/python3.12/dist-packages/vllm_int4_for_multi_arc.so` then `--quantization sym_int4` | Partially — Intel's multi-Arc INT4 kernel; INT4×FP16 (W4A16-like) but produced *online*, not from an AutoRound checkpoint. |
| **MXFP4** | Supported, **gpt-oss only** | `--quantization mxfp4` | n/a for Qwen. |
| **Pre-quantized AWQ / GPTQ** | Auto-detected from `config.json` (no flag) | just point `--model` at the repo | GPTQ-int4 XPU kernel exists upstream (#37844); community reports (#371) of *very low* perf with Qwen3.5 GPTQ-Int4 on llm-scaler — **verify before relying on it**. |
| **AutoRound W4A16 (auto_round format)** | **Not a documented llm-scaler-vllm path** | — | The README lists AWQ/GPTQ auto-detect but **not** AutoRound. AutoRound's own XPU support targets generic `torch-xpu`; no Battlemage-validated AutoRound-in-llm-scaler recipe was found. **This is the gap between the user's upstream 0.20.1 stack and the Intel-supported stack.** |

**On the user's "INT4 AutoRound W4A16" working config:** that ran on the **upstream/community vLLM-XPU** stack (which *does* accept `auto_round` checkpoints via the auto-round loader kernels), not llm-scaler. If you want to stay on llm-scaler's supported path, prefer **online `sym_int4` or `fp8`**. If you want AutoRound W4A16, you're committing to the upstream-vLLM-XPU build (less validated on B70, but is where MTP+GDN actually got upstreamed in v0.23.0).

### How to obtain / produce an AutoRound W4A16 Qwen3.6 checkpoint

**Option A — pre-quantized (fastest):**
- [`Lorbus/Qwen3.6-27B-int4-AutoRound`](https://huggingface.co/Lorbus/Qwen3.6-27B-int4-AutoRound) — W4A16, `auto_round:auto_gptq` packing, group size 128, **`mtp.fc` dequantized to BF16** so native MTP works (~85% accept on CUDA). Dense 27B — fits one B70 (32 GB).
- [`atbender/Qwen3.6-VL-REAP-26B-A3B-W4A16`](https://huggingface.co/atbender/Qwen3.6-VL-REAP-26B-A3B-W4A16) — VL/MoE variant.
- Browse: [HF `base_model:quantized:Qwen/Qwen3.6-27B`](https://huggingface.co/models?other=base_model%3Aquantized%3AQwen%2FQwen3.6-27B).

**Option B — produce it yourself with [intel/auto-round](https://github.com/intel/auto-round):**
```bash
pip install torch --index-url https://download.pytorch.org/whl/xpu
pip install auto-round-nightly      # AutoRound 2026.04+ for Qwen3.6/DeltaNet
auto-round \
    --model Qwen/Qwen3.6-27B \
    --scheme "W4A16" \
    --format "auto_round" \
    --output_dir ./Qwen3.6-27B-int4-AutoRound
```
Keep these in BF16 (AutoRound skips / you should exclude): `mtp.fc` (MTP head — required for spec-decode), `linear_attn.in_proj_a/b` (DeltaNet, also not divisible by group size 128), all layernorms/RMSNorms, router gates. *(~54 GB BF16 → ~18 GB INT4 for the 27B.)*

> ⚠️ **Single-source / cross-arch caveat:** the AutoRound recipe and the 85% MTP-accept numbers come from **CUDA (RTX 5090/3090)** write-ups ([dasroot.net](https://dasroot.net/posts/2026/04/efficient-quantization-qwen3-6-autoround-int4/), Lorbus card). Whether the `auto_round` checkpoint loads and uses **XMX** efficiently on B70 via upstream vLLM-XPU is **unverified** here — validate the kernel actually hits XMX (vs. a dequant-to-FP16 fallback) before trusting throughput.

---

## 4. Known landmines / required flags for B70 vLLM

From [llm-scaler vllm/README](https://github.com/intel/llm-scaler/blob/main/vllm/README.md) and [vllm #41663](https://github.com/vllm-project/vllm/issues/41663), [llm-scaler #382](https://github.com/intel/llm-scaler/issues/382).

**Always-on flags / env (llm-scaler path):**
- `--enforce-eager` — **mandatory** in every Intel B70 example. XPU graph capture is unreliable, and for TP2 vLLM disables graph capture for communication ops anyway. Expect to run eager.
- `--dtype float16` (or `--dtype half`) — not bf16.
- `--block-size 64`, `--gpu-memory-util 0.9`, `--trust-remote-code`.
- `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1`, `VLLM_WORKER_MULTIPROC_METHOD=spawn`.
- `VLLM_OFFLOAD_WEIGHTS_BEFORE_QUANT=1` — prevents OOM during **online** FP8/INT4 quant (see §4.2 OOM below). Set `=0` only if you have memory headroom.
- For online INT4: `export VLLM_QUANTIZE_Q40_LIB="/usr/local/lib/python3.12/dist-packages/vllm_int4_for_multi_arc.so"`.
- `ZE_AFFINITY_MASK=0,1` to select the first two Arc GPUs for TP2.

**TP / multi-GPU landmines:**
- **`CCL_TOPO_P2P_ACCESS=1` (P2P) vs `=0` (USM):** P2P gives ~15% higher throughput at large batch (e.g. batch 30); negligible at small batch. Try P2P first; fall back to USM if you hit collective hangs.
- **Head-count divisibility:** `--tensor-parallel-size 2` requires attention head count divisible by 2. Qwen3.6-35B-A3B works at TP2 in Intel's own example; verify for any non-standard model.
- **Graph capture is disabled for TP2 comm ops** — this is *why* TP2+MTP underperformed for the user. Don't expect CUDA-graph-like speedups at TP2.
- **OOM on big models / VL profiling:** [#382](https://github.com/intel/llm-scaler/issues/382) — Qwen3.6-35B-A3B FP8 on 2×B70 hit `UR_RESULT_ERROR_OUT_OF_RESOURCES` (40) during the dummy-encoder profiling phase. Mitigations: lower `--max-model-len` / `--max-num-batched-tokens`, set `VLLM_OFFLOAD_WEIGHTS_BEFORE_QUANT=1`, reduce `--gpu-memory-util`. Open issue, no confirmed fix.

**Upstream-vLLM-XPU (non-llm-scaler) B70 TP2 landmines — [#41663](https://github.com/vllm-project/vllm/issues/41663):**
- Symptom: `general protection fault` in workers + `xe ... Engine reset: engine_class=bcs` on `intel/vllm:0.17.0-xpu`, 2×B70, Qwen3-30B-A3B FP8, TP2, Ubuntu 24.04 HWE kernel 6.17. Root cause judged **host-stack**, not vLLM-only (Intel's own image reproduces it; host differs from Intel's validated BOM).
- **Stable workaround profile (~362 tok/s @ 50 concurrency):**
  ```
  CCL_ENABLE_SYCL_KERNELS=0
  CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0
  SYCL_UR_USE_LEVEL_ZERO_V2=0
  VLLM_XPU_ENABLE_XPU_GRAPH=1
  ```
- Debug-only fallback (very slow, ~0.5 tok/s): `--enforce-eager` + `VLLM_XPU_ENABLE_XPU_GRAPH=0` + `UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1` + `CCL_ALLREDUCE=ring`.
- **Software BOM that reproduced the fault:** PyTorch `2.10.0+xpu`, oneAPI compiler `2025.3.2`, oneCCL `2021.17.2-5`, Level-Zero driver `25.48.36300.8`, GuC firmware `70.44.1`. Intel's **validated** BOM differs: **Ubuntu 25.04 + kernel 6.14.0**, oneCCL `2021.15.7.8`. *Takeaway: match Intel's validated BOM, or just use the llm-scaler image which bundles a validated stack (RDC platform `26.18.8.2`, Ubuntu 24.04 server minor 3/4, fresh install).*

---

## 5. Docker run recipes

### 5a. Single-B70, Qwen3.6 4-bit (recommended, llm-scaler online INT4)

This is the supported path. `sym_int4` is Intel's online W4A16-style INT4. (If you specifically need an AutoRound checkpoint, see the note after.)

```bash
# 1) Start the container (host)
sudo docker run -td --privileged --net=host --device=/dev/dri \
  --name=lsv --shm-size="32g" \
  -v /path/to/models:/llm/models/ \
  --entrypoint /bin/bash \
  intel/llm-scaler-vllm:0.14.0-b8.3.1

# 2) Exec in and serve Qwen3.6-27B INT4 on ONE B70
sudo docker exec -it lsv bash

export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_OFFLOAD_WEIGHTS_BEFORE_QUANT=1
export ZE_AFFINITY_MASK=0
export VLLM_QUANTIZE_Q40_LIB="/usr/local/lib/python3.12/dist-packages/vllm_int4_for_multi_arc.so"

vllm serve \
  --model /llm/models/Qwen3.6-27B \
  --served-model-name Qwen3.6-27B \
  --dtype float16 \
  --enforce-eager \
  --quantization sym_int4 \
  --max-model-len 40000 \
  --max-num-batched-tokens 8192 \
  --block-size 64 \
  --gpu-memory-util 0.9 \
  --trust-remote-code \
  --disable-log-requests \
  --host 0.0.0.0 --port 8000 \
  -tp 1
```
- Swap `--quantization sym_int4` (+ the `VLLM_QUANTIZE_Q40_LIB` export) for `--quantization fp8` to run FP8 instead (often the safer/faster XMX path; add `--kv-cache-dtype fp8` on b8.3.1 to save KV memory).
- Do **not** add `--speculative-config` here — MTP will crash on `gdn_attention ... spec_sequence_masks` on this image.

**If you must run an AutoRound W4A16 checkpoint instead of online INT4:** mount e.g. `Lorbus/Qwen3.6-27B-int4-AutoRound`, drop the `--quantization`/`VLLM_QUANTIZE_Q40_LIB` lines (it's auto-detected), and be aware this is the **upstream-vLLM-XPU** loader path — less validated on B70 inside llm-scaler; confirm it loads (not all auto_round packings are XPU-supported) and benchmark vs. `sym_int4`.

### 5b. Dual-B70, Qwen3.6-35B-A3B, TP2 (next week)

Mirrors Intel's README §3.2 example.

```bash
sudo docker run -td --privileged --net=host --device=/dev/dri \
  --name=lsv2 --shm-size="32g" \
  -v /path/to/models:/llm/models/ \
  --entrypoint /bin/bash \
  intel/llm-scaler-vllm:0.14.0-b8.3.1

sudo docker exec -it lsv2 bash

export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_OFFLOAD_WEIGHTS_BEFORE_QUANT=1
export ZE_AFFINITY_MASK=0,1
export CCL_TOPO_P2P_ACCESS=1            # try P2P; set 0 (USM) if collectives hang

vllm serve \
  --model /llm/models/Qwen3.6-35B-A3B/ \
  --served-model-name Qwen3.6-35B-A3B \
  --dtype float16 \
  --enforce-eager \
  --quantization fp8 \
  --tensor-parallel-size 2 \
  --max-model-len 40000 \
  --max-num-batched-tokens 8192 \
  --block-size 64 \
  --gpu-memory-util 0.9 \
  --trust-remote-code \
  --disable-log-requests \
  --host 0.0.0.0 --port 8000
```
**Dual-B70 cautions:**
- Expect **no MTP** (broken on this image) and **no graph capture at TP2** — so TP2 buys you capacity (fit 35B-A3B / longer context), not necessarily latency. Matches the user's finding that TP2+MTP was a negative result.
- If you hit OOM during profiling (the [#382](https://github.com/intel/llm-scaler/issues/382) failure), lower `--max-model-len`/`--max-num-batched-tokens` and keep `VLLM_OFFLOAD_WEIGHTS_BEFORE_QUANT=1`.
- For INT4 on TP2, add the `VLLM_QUANTIZE_Q40_LIB` export and use `--quantization sym_int4`.
- If you stay on an **upstream `intel/vllm:*-xpu`** image instead (e.g. to get AutoRound + the v0.23.0 GDN-MTP fix), bring the [#41663](https://github.com/vllm-project/vllm/issues/41663) stability env: `CCL_ENABLE_SYCL_KERNELS=0`, `CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0`, `SYCL_UR_USE_LEVEL_ZERO_V2=0`, and match Intel's validated BOM (Ubuntu 25.04 / kernel 6.14, oneCCL 2021.15.7.8).

---

## 6. Confidence / caveats flags

- ✅ **Well-sourced (Intel primary docs):** recommended tag `0.14.0-b8.3.1` for Qwen3.6; online FP8/`sym_int4` quant; `--enforce-eager` mandatory; B70 official since b8.2; the exact TP2 Qwen3.6-35B-A3B recipe.
- ⚠️ **Single-source / cross-arch (verify on B70):** AutoRound W4A16 recipe & MTP 85%-accept numbers are CUDA-derived (dasroot.net, Lorbus HF card); XMX efficiency of `auto_round` checkpoints on B70 is unverified.
- ⚠️ **Upstream-only, not yet in the B70 image:** GDN-MTP fix `#43565` is in upstream **vLLM v0.23.0**, *newer* than the base inside llm-scaler b8.3.1 — so MTP+GDN on the supported B70 image remains broken/manual-patch.
- 🟥 **Open/unresolved:** llm-scaler [#386](https://github.com/intel/llm-scaler/issues/386) (XPU gdn_attention spec_sequence_masks) and [#382](https://github.com/intel/llm-scaler/issues/382) (2×B70 35B-A3B FP8 OOM) and vllm [#41663](https://github.com/vllm-project/vllm/issues/41663) (TP2 GP-fault) show **no merged fix** as of this snapshot.
- 🟥 **Immature area in general:** Battlemage/B70 is *not* enumerated in the upstream vLLM v0.20.0 XPU doc; B70 support effectively lives in Intel's llm-scaler fork. Pin images, re-check Releases weekly.

---

## Sources

- Intel llm-scaler repo: <https://github.com/intel/llm-scaler>
- llm-scaler Releases (tag index): <https://github.com/intel/llm-scaler/blob/main/Releases.md>
- llm-scaler vLLM README (recipes, quant, env, B70, Qwen3.6 §3.2): <https://github.com/intel/llm-scaler/blob/main/vllm/README.md>
- Docker Hub tags: <https://hub.docker.com/r/intel/llm-scaler-vllm/tags>
- llm-scaler #386 — MTP `gdn_attention ... spec_sequence_masks` crash: <https://github.com/intel/llm-scaler/issues/386>
- llm-scaler #382 — 2×B70 Qwen3.6-35B-A3B FP8 OOM: <https://github.com/intel/llm-scaler/issues/382>
- llm-scaler #371 — low perf Qwen3.5-27B GPTQ-Int4: <https://github.com/intel/llm-scaler/issues/371>
- vllm #41663 — dual-B70 TP2 GP fault / xe BCS reset on `intel/vllm:0.17.0-xpu`: <https://github.com/vllm-project/vllm/issues/41663>
- vLLM v0.22.0 release (XPU GPTQ-int4, GDN, MTP): <https://github.com/vllm-project/vllm/releases/tag/v0.22.0>
- vLLM v0.23.0 release (XPU **GDN-attention MTP #43565**): <https://github.com/vllm-project/vllm/releases/tag/v0.23.0>
- vLLM XPU supported-models doc (v0.20.0): <https://docs.vllm.ai/en/v0.20.0/models/hardware_supported_models/xpu/>
- Phoronix — llm-scaler 0.14.0-b8.2 official B70 support: <https://www.phoronix.com/news/Intel-LLM-Scaler-vllm-0.14-b8.2>
- wccftech — Intel confirms Arc Pro B70 via llm-scaler release notes: <https://wccftech.com/intel-confirms-arc-pro-b70-workstation-gpu-via-llm-scaler-vllm-ai-release-notes/>
- intel/auto-round (W4A16 CLI, XPU install): <https://github.com/intel/auto-round>
- vLLM AutoRound blog (2026-06-02): <https://vllm.ai/blog/2026-06-02-vllm-omni-autoround>
- dasroot.net — AutoRound INT4 for Qwen3.6 recipe: <https://dasroot.net/posts/2026/04/efficient-quantization-qwen3-6-autoround-int4/>
- HF `Lorbus/Qwen3.6-27B-int4-AutoRound` (W4A16, mtp.fc BF16): <https://huggingface.co/Lorbus/Qwen3.6-27B-int4-AutoRound>
- HF `atbender/Qwen3.6-VL-REAP-26B-A3B-W4A16`: <https://huggingface.co/atbender/Qwen3.6-VL-REAP-26B-A3B-W4A16>
- HF quantized-of-Qwen3.6-27B index: <https://huggingface.co/models?other=base_model%3Aquantized%3AQwen%2FQwen3.6-27B>
