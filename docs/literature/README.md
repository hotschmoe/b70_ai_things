# Literature base — Intel Arc Pro B70 / Qwen3.6-27B inference

Cited research compiled 2026-06-17. Treat enthusiast tok/s figures as anchors to re-measure,
not ground truth (most public B70 numbers come from a handful of hobbyist testers).

- **[01_backends.md](01_backends.md)** — backend landscape on Battlemage: llama.cpp SYCL vs Vulkan,
  vLLM-XPU / Intel LLM-Scaler, IPEX-LLM (archived), OpenVINO/SGLang. Quant reality (Q8_0 regression,
  XMX INT8 only in prefill), required env flags, version pinning, contribution opportunities.
- **[02_multigpu.md](02_multigpu.md)** — dual/quad B70: no Arc P2P, TP vs PP on PCIe3, oneCCL, expected
  scaling (~1.0-1.3x dense single-stream; capacity + MoE concurrency wins), top configs to test.
- **[03_offload_mtp_sweep.md](03_offload_mtp_sweep.md)** — CPU/RAM offload (when it helps = rarely for
  dense 27B), MTP/speculative decoding on XPU (llama.cpp native MTP PR #22673; vLLM XPU can't on DeltaNet),
  and the rigorous benchmark sweep methodology + first sweep matrices.
- **[07_w8a8_int8_recovery.md](07_w8a8_int8_recovery.md)** — W8A8 INT8 *accuracy recovery* survey (2026-06-19):
  our fast path is INT8 W8A8 (Xe2 has no native FP8); the fidelity cost is activation quant, not weights;
  ranked recovery (selective SmoothQuant, OS+, down_proj@W8A16, RTN); the DeltaNet/SSM frontier (Quamba2,
  Q-Mamba) + the `in_proj_qkvz` silent-zeroing gotcha; skip rotation/QAT at W8A8.

Plan synthesis lives in [`../../STRATEGY.md`](../../STRATEGY.md); experiment log in
[`../../JOURNAL.md`](../../JOURNAL.md).
