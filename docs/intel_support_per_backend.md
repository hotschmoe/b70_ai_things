# Intel Arc backend support

Updated 2026-08-26 for the clean-stack refresh.

| backend | live status | retained purpose |
|---|---|---|
| sglang | primary | New W8A8 and Ornith serving work |
| vLLM | paused baseline | NVFP4 shelf and Steve transfer controls |
| llama.cpp | removed from live tree | Historical code is quarantined |
| ZML | removed from live tree | Historical code is quarantined |

## sglang

Target for refreshed PyTorch/XPU integration and new shelf promotion. The live
tree retains W8A8 and NVFP4 work only. Old images, W4 paths, logs, and generated
extensions were quarantined.

Required requalification:

- Qwen3.8 W8A8 GPTQ
- Ornith-1.5 W8A8 RTN+Shisa MTP
- TP=2 collective and graph behavior
- concurrent prefill plus decode coherence

## vLLM

Retained as a measured baseline and exact-control workspace, not the primary
new serving backend.

Live shelf:

- Qwen3.6 NVIDIA NVFP4
- Qwen3.8 RadixArk NVFP4

Retained research:

- exact Qwen3.6/Steve controls;
- graph boundary and collective profiling;
- push-all-reduce integration oracle;
- shared custom-op and patch source.

The known vLLM concurrent prefill/decode and TP=2 queue-handoff failure modes
remain reasons not to make it the primary backend without fresh evidence.

## Removed backends

llama.cpp and ZML were useful historical probes but are outside the retained
model/backend scope. Their tracked code and runtime clones are recoverable
under archive/to-delete-20260826 until permanent purge.
