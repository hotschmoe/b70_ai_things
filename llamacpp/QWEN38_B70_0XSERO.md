# 0xSero/qwen38-b70 -- llama.cpp SYCL 1x/2x B70 recipe

Source: https://github.com/0xSero/qwen38-b70 (published 2026-08-17, 4 commits).
This is NOT a shelf entry. Weight-only GGUF, not compressed-tensors.

## What it is

One-command `docker compose up -d` for Qwen3.8-27B on 1 or 2 Arc Pro B70s:

- Engine: llama.cpp SYCL (`mndodd/llama.cpp` @ `4302fb5`) +
  steveseguin/b70-optimization-lab TP2 stack (`tp2-full-stack.patch` +
  `q4k-increment.patch`, SHA-checked against the lab).
- Weights: `ggml-org/Qwen3.8-27B-GGUF` @ `0669b986`,
  `Qwen3.8-27B-Q4_K_M.gguf` (~19 GB, SHA256 pinned).
- Optional: `mtp-Qwen3.8-27B-Q4_0.gguf` (`ENABLE_MTP=1`), mmproj Q8_0
  (`ENABLE_VISION=1`, encoder stays on CPU -- GPU offload hangs xe).
- Build: oneAPI 2025.3.3 SYCL JIT. They say AOT `bmg_g31` needs oneAPI
  2026.1.1 whose UR adapter does not enumerate devices on public stacks.
- Port 8010. `GPU_COUNT=1|2`. Default ctx 262144 (2 GPU) / 131072 (1 GPU).

Q4K reorder-family fused kernels are OFF: those assume the lab AOT tensor
layout and corrupt stock ggml-org weights under JIT.

## Their measured numbers (Q4_K_M, f16 KV, temp=0)

2x B70 TP2 decode is nearly flat with context (hybrid GDN: 16/64 layers
are full attention):

| ctx | 2x prefill | 2x decode | 1x prefill | 1x decode |
|---|---|---|---|---|
| ~2.5k | 955 | 51.1 | 1164 | 33.3 |
| ~10k | 881 | 50.9 | 1018 | 33.4 |
| ~40k | 694 | 49.7 | 785 | 33.4 |
| ~65k | 563 | 46.9 | 625 | 31.9 |
| ~128k | -- | -- | 426 | 27.5 |
| ~160k | 365 | 42.0 | -- | -- |
| ~245k | 275 | 30.8 | OOM >=192k | -- |

Single-GPU prefill beats TP2 at every length (all-reduce tax). TP2 wins
decode and KV capacity.

MTP (`--spec-type mtp --draft-max 8`):

| config | easy (count) | hard (random) | baseline |
|---|---|---|---|
| 2x TP2 | 84.3 (97.2% accept) | 49.0 (37.5%) | 51.1 |
| 1x @64k | 61.7 (97.2%) | 28.2 (29.6%) | 33.3 |

Hard-task MTP is a net loss. 1x + MTP + 131k OOMs; 65536 is their
validated single-card MTP ctx.

## Engine comparison they published (2x B70)

| path | decode | ctx | note |
|---|---|---|---|
| llama.cpp SYCL TP2 | **51** | 262k | coherent, stable |
| llama.cpp + MTP | 84 easy | 262k | quality-neutral |
| vLLM XPU 0.27.2 FP8 | **21.7** | 262k | 865k-token FP8 KV |
| Intel llm-scaler 0.21 eager | -- | -- | crashed on first request |

That 21.7 vLLM FP8 number matches our 2026-08-17e rmacy DSpark first try
(20-22 tok/s). Their llm-scaler 0.21 crash is the same lineage as
`qwen38-fp8-dspark:v10-slim` before the oneCCL 2021.17 swap.

## vs our llama.cpp tree

`llamacpp/` is 3.6-era: DP=2 Q4_K_M and TP=2 Q8_0 on
`/mnt/vm_8tb/b70/llama.cpp` (older HEAD), built inside `sglang-xpu:mtp`.
We do not have a 3.8 Q4_K_M GGUF or the lab TP2 + Q4K increment patches
applied. 0xSero is a packaged 3.8 bring-up of the same lab stack, not a
new kernel family.

Trying it means taking the current DSpark serve down, building their
image (SYCL compile, long), downloading ~19 GB Q4_K_M, then a coherence
+ decode gate against 51 tok/s / 262k. Do not shelf until measured here.

## How to run it

```
# after stopping the current GPU serve
# clone + image + GGUF already at /mnt/vm_8tb/b70/qwen38-b70
cd /mnt/vm_8tb/github/b70_ai_things
./bin/gpu-run bash llamacpp/serve_qwen38_b70_0xsero.sh start
bash llamacpp/serve_qwen38_b70_0xsero.sh stop
```

Served id: `/models/Qwen3.8-27B-Q4_K_M.gguf` on :8010. n_ctx 262144.

## Measured here (2026-08-17j, ENABLE_MTP=0, GPU_COUNT=2)

xpu-health both cards OK. Load 211s. Cmdline really is TP=2
(`--device SYCL0,SYCL1 --split-mode tensor --tensor-split 1,1`).
Both BMG_G31_B70 devices enumerated. Q4K reorder-family OFF.

| workload | result |
|---|---|
| Paris | exact ("The capital of France is **Paris**.") |
| fib | coherent iterative |
| code c1 out=256 | **32.8 avg / 32.8 best** (wall 7.8s) |
| llama.cpp predicted_tokens_seconds | 32.8 bench / 34.1 after HE+ |
| HE+ 164 thinking-off | **0.970 base / 0.927 plus** (gen 1458s) |
| prefill (server gauge) | 440-508 tok/s |

vs claimed 51 / RadixArk MTP3 41.2 / DSpark 34.4 / 3.6 MTP5 48.9.
32.8 matches their published **1x B70** row (33.3), not the 2x TP2
row (51). Quality is the best 3.8 HE+ on this box (RadixArk 0.933/
0.890, Inferact 0.939/0.915, 3.6 0.988/0.945).

Long think is decode-bound. 4k think tokens is ~122s here vs ~97s
on MTP3 vs ~82s on 3.6 MTP5. Do not flip ENABLE_MTP=1: their hard-
task MTP is a net loss, and long think is a hard task.

Not a shelf. Next GPU work is why TP=2 decode equals 1x -- if 51
reproduces here this path beats MTP3 on speed and quality.
