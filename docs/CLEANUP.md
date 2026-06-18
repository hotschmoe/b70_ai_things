# Cleanup / Disk Hygiene Log

Policy (user-approved 2026-06-17): prune unused backends/images/models freely to save
space, BUT log every deletion here with date + reason, and never delete a fallback until
its replacement is validated. Never touch the user's production images/containers
(nextcloud, mariadb, syncthing, clamav, specula-*).

## Keep / remove decisions (live inventory)

### Docker images
| Image | Size | Status | Reason |
|-------|------|--------|--------|
| ghcr.io/ggml-org/llama.cpp:full-intel | 11.8 GB | KEEP | standard-model baseline (7B ~90 t/s) + DeltaNet-SYCL contribution (task #10) |
| python:3.11 | 1.1 GB | KEEP | HF downloads |
| intel/llm-scaler-vllm:0.14.0-b8.3 | 33.6 GB | REMOVE after b8.3.1 validated | superseded by b8.3.1 (had DeltaNet+FP8 init bug) |
| intel/llm-scaler-vllm:0.14.0-b8.3.1 | ~33 GB | KEEP (active) | Intel-recommended for Qwen3.6-27B |
| python:3.10 | 1.1 GB | DELETED | unused (kept 3.11) |

### Models (/mnt/vm_8tb/b70/models)
| Model | Size | Status | Reason |
|-------|------|--------|--------|
| Qwen_Qwen3.6-27B-FP8 | 29 GB | KEEP | active 8-bit target (vLLM-XPU) |
| bartowski_Qwen2.5-7B-Instruct-GGUF | 4.4 GB | KEEP | known-good reference; draft-model candidate; small |
| unsloth_Qwen3.6-27B-GGUF (Q4_K_M) | 16 GB | KEEP (provisional) | crashes on llama.cpp SYCL (DeltaNet); works CPU; test artifact for DeltaNet-SYCL contribution. Remove if we abandon that path. |

## Deletion log
- 2026-06-17: removed `python:3.10` image (~1.1 GB) — unused, kept 3.11.
- 2026-06-17: `docker builder prune -f` — build cache (~734 MB).
- 2026-06-17: removed `intel/llm-scaler-vllm:0.14.0-b8.3` (33.6 GB) — superseded by b8.3.1; both shared
  the same DeltaNet+FP8 bugs so b8.3 was no longer a useful fallback. docker.img 103G->69G used.
