# Cleanup / Disk Hygiene Log

Policy (user-approved 2026-06-17): prune unused backends/images/models freely to save
space, BUT log every deletion here with date + reason, and never delete a fallback until
its replacement is validated. Never touch the user's production images/containers
(nextcloud, mariadb, syncthing, clamav, specula-*).

## 2026-06-22 -- archival batch (user-requested; `mv` to models/archive/, reversible, ~243G)
Moved out of the live `models/` top level (archive/ 184G -> 427G):
- **All Qwable** (user doesn't need it): `DJLougen_Qwable-5-27B-Coder` (60G, bf16 base), `...-W8A8-sqgptq` (33G),
  `...-W4A8-sqgptq` (33G), `...-W4A8-sqgptq-prepacked` (25G), `...-int4-AutoRound` (25G, the BROKEN XPU-calib quant).
- **Non-prepacked twins** (kept the prepacked, which is the serve target): `Qwen3.6-27B-W4A8-sqgptq` (33G; kept
  `...-prepacked` 25G), `Qwen3-14B-W4A8-gptq` (16G; kept `...-prepacked` 9.3G).
- **GGUFs**: `unsloth_Qwen3.6-27B-GGUF` (16G), `bartowski_Qwen2.5-7B-Instruct-GGUF` (4.4G).
Daily driver (Lorbus 27B int4) + the 35B int8/int4 MoE serves untouched. NOTE: the older "KEEP" rows below for the
2 GGUFs are now superseded by this archival.

### 2026-06-22 (follow-up) -- archive 14B W8A8-gptq / W8A8-gptq512 / W8A16 (user call: done chasing W8A8)
User: autoround supersedes the gptq W8A8 variants; not chasing W8A16 optimizations. Archived (+46G -> archive/ 473G):
`Qwen3-14B-W8A8-gptq` (16G), `Qwen3-14B-W8A8-gptq512` (16G), `Qwen3-14B-W8A16` (16G). KEPT in 14B: `W8A8-autoround`,
`W4A8-gptq-prepacked`, `W4A16-gptq`. **HONESTY NOTE:** by the repo's MEASURED leaderboard, `W8A8-gptq` is the validated
W8A8 winner (HumanEval+ 0.890/0.921), while `W8A8-autoround` accuracy is unmeasured ("TBD == gptq kernel"). The measured
numbers persist in evals/results/SUMMARY.md; the gptq files are archived (reversible) if the W8A8 chase reopens.

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
