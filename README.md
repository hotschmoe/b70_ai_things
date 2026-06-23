# B70 AI Inference Optimization

Research and engineering notes for local LLM inference on Intel Arc Pro B70
GPUs, with a focus on vLLM-XPU, compressed-tensors quantization, and custom
INT8 kernels for Battlemage/Xe2.

The stack: 2x Intel Arc Pro B70 (Battlemage / Xe2, 32 GB each), vLLM-XPU
(0.23.0), compressed-tensors W8A8 / W4A8 INT8 research, on a local Ubuntu 26.04
box (kernel 7.0). Since the 2026-06-23 migration we run LOCALLY on the box
(`b70s4dayz`), not over SSH -- see [MIGRATION.md](MIGRATION.md).

## Headline results

- **[BREAKTHROUGH] B70<->B70 GPU P2P is UNLOCKED on kernel 7.0 + IOMMU off.**
  `zeDeviceCanAccessPeer` = True both directions (it was False on all 12 probe
  variants on kernel 6.18). The quantified prize: a 64-layer TP=2 forward fires
  128 allreduces; today they go host-staged at ~1.16 GB/s, making 27B prefill
  **~84% collective-bound** (~2.3 s of a 2.75 s TTFT). Lifting the allreduce
  toward the ~15 GB/s Gen3 wire via P2P points to **~4x faster prefill TTFT**
  (~2.75 s -> ~0.65-0.75 s). P2P is a prefill/TTFT + concurrency win, ~1.2x on
  single-stream decode. See [docs/P2P_GPU.md](docs/P2P_GPU.md) H.10 / H.11 and
  [27b_w8a8_research.md](27b_w8a8_research.md).
- **27B serve throughput (dual B70, captured PIECEWISE):** W4A8 TP=2 single-stream
  decode **~22 t/s** (TP=1 single-card ~20.7); W8A8 27B TP=2 + captured MTP
  (spec=3) reaches **~34.82 t/s @ ~51% draft accept**, coherent -- 1.92x over
  captured-no-MTP (18.10). Single-card daily driver (27B int4-AutoRound, captured)
  decodes ~30.8 t/s. See [FINDINGS.md](FINDINGS.md) and [JOURNAL.md](JOURNAL.md).

Start with:

- [FINDINGS.md](FINDINGS.md): current working results and dead ends.
- [RESEARCH_TODO.md](RESEARCH_TODO.md): active research order.
- [rdy_to_serve/README.md](rdy_to_serve/README.md): verified serve shelf.
- [JOURNAL.md](JOURNAL.md): full experiment log.
- [AGENTS.md](AGENTS.md): standing rules for agents working in this repo.

## Current Focus

The research direction is **compressed-tensors first**. We want one comparable
artifact format across W8A8, W4A8, W4A16, TP=2, PP=2, and custom kernel work.

Primary tracks:

- **W8A8 INT8:** main path for B70 INT8 XMX research. The 14B baseline serves
  through our `XPUInt8ScaledMMLinearKernel`; 27B W8A8 serves via TP=2.
- **W4A8 INT8-activation:** next memory/perf research path. This is where custom
  int4-weight x int8-act GEMM/GEMV work should converge.
- **W4A16:** capacity baseline. AutoRound/INC int4 is the proven daily-driver
  serve path today; compressed-tensors W4A16 for 27B is intentionally deferred
  to a focused padding/ignore-list/kernel session.
- **W4A4:** later frontier research. Interesting, but not the near-term kernel
  priority.

Calibration policy today:

- Use **GPTQ** as the default producer for compressed-tensors quants.
- AutoRound remains a useful comparison and a proven int4 serving route.
- GPTQ slightly beat AutoRound on 14B W8A8 HumanEval+; verify on harder evals
  before making broad claims, especially for W4A8.

## Headline Kernel Work

The repo contains a working INT8 W8A8 path for B70:

- `contrib/vllm_int8_xpu/`: custom vLLM XPU int8 linear kernel integration.
- [docs/kernel/02_int8_w8a8_status.md](docs/kernel/02_int8_w8a8_status.md):
  kernel status and usage.
- `vllm-xpu-env:int8g`: image used by the current 14B W8A8 serve baseline.

The key runtime signal is:

```text
Selected XPUInt8ScaledMMLinearKernel for CompressedTensorsW8A8Int8
```

## Environment

- Host: local Ubuntu 26.04 box `b70s4dayz` (kernel 7.0) at `192.168.10.5`. We
  run LOCALLY on the box as user `hotschmoe`, not over SSH (retired in the
  2026-06-23 migration; see [MIGRATION.md](MIGRATION.md)).
- GPU root: `/mnt/vm_8tb/b70/`
- Models/quants: `/mnt/vm_8tb/b70/models/`
- GPUs: dual Intel Arc Pro B70 cards (Battlemage / Xe2), 32 GB each.
- Runtime: Docker with `/dev/dri` passed through.
- Default vLLM image: `vllm-xpu-env:v0230`, unless a serve recipe specifies
  `:int8g` or another image.

All serious GPU runs should go through `gpu-run`; see [AGENTS.md](AGENTS.md).

## Layout

| Path | Purpose |
|---|---|
| `AGENTS.md` / `CLAUDE.md` | Agent rules; `CLAUDE.md` is a symlink. |
| `FINDINGS.md` | High-signal current results. |
| `RESEARCH_TODO.md` | Active research plan and priorities. |
| `JOURNAL.md` | Append-only experiment log. |
| `rdy_to_serve/` | Verified serve recipes. |
| `contrib/` | Kernel and vLLM patches. |
| `docs/` | Kernel notes, hardware notes, method registry, literature. |
| `scripts/` | Numbered experiment scripts. |
| `bin/` | Stable tools shared by serve/eval workflows. |
