# Serving index

Updated 2026-08-26 after the clean-slate cleanup.

## Retained pre-refresh shelf

- rdy_to_serve/vllm/qwen36-27b-nvfp4
- rdy_to_serve/vllm/qwen38-27b-nvfp4

No daily-driver server was active during cleanup.

These are measured configuration controls, not currently runnable production
entries. The cleanup quarantined their ABI-specific native libraries. Rebuild
against the refreshed stack and pass the full gate below before serving or
calling either entry live.

Qwen3.8 W8A8 and Ornith W8A8 are retained research artifacts, but neither has
a newly sweep-qualified sglang shelf entry. Do not reconstruct a production
serve from old journal entries.

## Host paths

- Repository: /mnt/vm_8tb/github/b70_ai_things
- Runtime root: /mnt/vm_8tb/b70
- Model root: models/files
- Container model mount: /models
- Default API port: 18080

## GPU lease

Every GPU operation uses bin/gpu-run.

One card:

  bin/gpu-run --card 0 bash -c '...'

Both cards:

  bin/gpu-run bash -c '...'

A detached server does not retain the shell's lease. Wrap start, wait, test,
and stop in one gpu-run invocation for an exclusive session.

## Required serve gate

1. Run bin/xpu-health.
2. For TP=2, run bin/xpu-collective-health.
3. Start from a rebuilt shelf control or a current backend research script.
4. Query /v1/models and verify the method/scheme-qualified ID.
5. Run deterministic semantic probes.
6. Run concurrent prefill plus decode coherence.
7. Measure c1 and c4 performance.
8. Stop gracefully.
9. Run per-card and compiled collective post-health.

Do not promote settings that are only faster in a single-request microbench.

## TP=2 warning

Do not enable arbitrary direct P2P in vLLM serving. A failed TP=2 worker init
or graph capture can poison later oneCCL state even with P2P disabled. Follow
AGENTS.md and docs/P2P_GPU.md, and use bin/xe-reset after crash-prone failures.

## Refresh boundary

The old backend images and ABI-specific binaries are quarantined. After any
driver, PyTorch, sglang, or vLLM update, rebuild extensions and repeat the full
serve gate. The quarantine is not a runtime dependency.
