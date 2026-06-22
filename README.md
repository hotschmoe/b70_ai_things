# B70 AI Inference Optimization

Research and engineering notes for local LLM inference on Intel Arc Pro B70
GPUs, with a focus on vLLM-XPU, compressed-tensors quantization, and custom
INT8 kernels for Battlemage/Xe2.

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

- Host: `root@192.168.10.5`
- GPU root: `/mnt/vm_8tb/b70/`
- Models/quants: `/mnt/vm_8tb/b70/models/`
- GPUs: dual Intel Arc Pro B70 cards, 32 GB each.
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
