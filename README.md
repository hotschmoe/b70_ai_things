# B70 AI Inference Optimization

> **PICKING UP THIS PROJECT? START AT [HANDOFF.md](HANDOFF.md)** — current state, what's running, box
> access (incl. the Linux SSH note), immediate next steps, and the roadmap.


Optimizing LLM inference on an **Intel Arc Pro B70** GPU (32 GB Battlemage) hosted in
an Unraid server (Threadripper). A public research log for fellow Team Blue tinkerers.

> ### Latest wins (2026-06-22)
> - **MTP proved POSITIVE on a single card:** Qwen3.6-27B W4A16 + native MTP (spec=4, PIECEWISE) = **1.79x**
>   (55.28 t/s vs 30.84) -- refutes the prior "-19% / not viable". Dense lever (MoE MTP is flat). [MTP_TODO.md](MTP_TODO.md)
> - **PIECEWISE XPU graph capture ~doubles int4 decode** (27B 7.9 -> 30.8 t/s); FULL still blocked on stock v0230.
> - **Int8 MoE now SERVES:** 35B-A3B Quark-W8A8 on 2x B70 TP=2 (v0230 Triton fused_moe); int4-AutoRound 56.8 t/s captured.
> - **AutoRound int4 on the qwen3_5 VLMs unblocked** (the MLLM-calib wall is beaten): Qwable W4A16 int4 producing. [QUANTS_TODO.md](QUANTS_TODO.md)
> - **Default image is `vllm-xpu-env:v0230` (vLLM 0.23.0)** -- NOT the deprecated 0.14.x llm-scaler.

> **Start here:** [FINDINGS.md](FINDINGS.md) — what actually works on the B70 (and what doesn't).
> [JOURNAL.md](JOURNAL.md) — full experiment log. [archive/RESULTS.md](archive/RESULTS.md) — older raw tables (Qwen3-14B), superseded by FINDINGS.
> [docs/COMMUNITY_CONFIGS.md](docs/COMMUNITY_CONFIGS.md) — other people's B70 configs we're chasing.
>
> **Headline contribution:** [docs/kernel/02_int8_w8a8_status.md](docs/kernel/02_int8_w8a8_status.md) — we
> wrote the **first working INT8 W8A8 kernel for Battlemage** in vLLM (oneDNN s8s8s32 + a fused per-token
> int8 quant). It beats FP8 ~1.6x in prefill and nearly matches it in decode. Source in `contrib/vllm_int8_xpu/`.

## Goals

- Run Qwen3.6-27B at 8-bit and 4-bit quantization on the Arc Pro B70.
- Exploit Intel XMX **INT8 fast paths** where possible.
- Evaluate backends: Intel vLLM (IPEX-LLM / `vllm-xpu`), upstream vLLM SYCL,
  llama.cpp (SYCL / Vulkan), and others.
- Explore speculative decoding and MTP (multi-token prediction) for throughput.
- Measure: tokens/s (prefill + decode), TTFT, VRAM use, quality at each quant.

## Environment / Constraints

- **Host:** Unraid @ `192.168.10.5`, login `root`, alias `b70` (see `~/.ssh/config`).
- **GPU:** 1x Intel Arc Pro B70 (Battlemage, BMG). Passed into Docker containers.
- **Storage:** ALL work/models/containers MUST live on the **8TB VM SSD**, NOT the
  array HDDs. See `docs/storage.md` for the exact path once confirmed.
- **Execution:** Everything runs in **Docker containers** with the Intel GPU passed
  through (`--device /dev/dri`). No bare-metal installs on the host.

## Layout

| Path | Purpose |
|------|---------|
| `JOURNAL.md` | Running log of every experiment: what we tried, results, verdict. |
| `docs/` | Hardware, storage, and environment notes (ground truth). |
| `docs/COMMUNITY_CONFIGS.md` | External/community B70 configs to chase + beat (primary target: 27B BF16 TP4 MTP, 54.2 t/s). |
| `docker/` | Dockerfiles / compose files per backend. |
| `scripts/` | Helper + benchmark scripts. |
| `results/` | Raw benchmark output, logs, metrics. |

## Quick connect

```
ssh b70           # passwordless once the key is authorized in Unraid GUI
```
