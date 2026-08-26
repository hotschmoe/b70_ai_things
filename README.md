# Intel Arc Pro B70 local inference lab

This repository is the clean working set for serving and kernel research on
two Intel Arc Pro B70 cards.

## Current scope

Primary backend: sglang.

Retained vLLM work: measured NVFP4 baselines and the recent Steve
reproduction/transfer controls.

Retained models:

- Qwen3.8-27B BF16
- Qwen3.8-27B RadixArk NVFP4
- Qwen3.8-27B compressed-tensors W8A8 GPTQ
- NVIDIA Qwen3.6-27B NVFP4
- Ornith-1.5-35B-A3B BF16+Shisa MTP, W8A8 RTN+Shisa MTP, NVFP4, and local
  GPTQ INT4 MixedCal-v2

ZML, llama.cpp, old model families, old W4 campaigns, raw historical results,
runtime build trees, and retired Docker images were moved out of the live tree
on 2026-08-26.

## Start here

- AGENTS.md: standing safety, scope, and workflow rules.
- RESEARCH_TODO.md: current clean-stack work order.
- FINDINGS.md: short current evidence ledger.
- JOURNAL.md: newest experiment window.
- docs/P2P_GPU.md: multi-GPU failure and recovery evidence.
- docs/20260825_steve_stack_component_ledger.md: exact Steve transfer ledger.
- docs/20260826_qwen36_graph_runtime_profile.md: graph/runtime boundary profile.
- docs/quant_methods.md: quantization method registry.
- docs/SERVING.md: live shelf and serving entry points.

## Host

- Host: b70s4dayz, local Ubuntu.
- Repository: /mnt/vm_8tb/github/b70_ai_things
- Runtime root: /mnt/vm_8tb/b70
- Model root: models/files
- Kernel baseline: 7.1.0-070100
- GPU access: bin/gpu-run only

The host is headless and neither B70 is display-held. Kernel 7.1 fixed the
old GuC/BCS wedge. vLLM TP>1 queue-handoff and oneCCL failure modes remain a
separate software risk; read AGENTS.md and docs/P2P_GPU.md before TP=2 work.

## Retained serving controls

The pre-refresh measured shelf contains:

- rdy_to_serve/vllm/qwen36-27b-nvfp4
- rdy_to_serve/vllm/qwen38-27b-nvfp4

These wrappers are retained configuration controls, not currently runnable
production entries: their old ABI-specific native libraries were quarantined.
Rebuild and requalify them after the software refresh. Qwen3.8 W8A8 and Ornith
W8A8 remain research targets without a newly sweep-qualified shelf entry;
sglang is the target for new promotion work.

## Clean-slate quarantine

The gitignored directory archive/to-delete-20260826 contains the reversible
cleanup batch. It includes old weights, docs, backend code, raw results,
runtime clones/builds/caches, and a restorable Docker image bundle.

The quarantine is intentionally not a runtime dependency. Review its README
and MANIFEST.tsv before permanent deletion.

Small raw evidence cited by the August 23-26 journal window is kept separately
at `archive/research-evidence-20260823-26/` and exposed through ignored symlinks
under `results/`. It is not part of the deletion batch.
