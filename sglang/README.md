# sglang backend

sglang is the primary backend after the 2026-08-26 cleanup.

Retained live work:

- w8a8/: Qwen3.8 and Ornith W8A8 quantization/serving research.
- nvfp4/: retained NVFP4 source and build notes needed for the refreshed stack.

Old images, graph experiments, W4 campaigns, logs, generated kernels, and
backend-specific patches were moved to
archive/to-delete-20260826/repo/sglang/.

Refresh procedure:

1. Pin and record the new sglang, PyTorch, oneAPI, and driver identities.
2. Rebuild extensions from tracked source.
3. Run one-card import and model smoke tests through bin/gpu-run.
4. Run compiled TP=2 collective health.
5. Requalify Qwen3.8 W8A8 and Ornith W8A8 under concurrent load.
6. Add a shelf entry only after the sweep gate passes.

Do not copy old shared libraries or Python site patches from quarantine into
the refreshed environment.
