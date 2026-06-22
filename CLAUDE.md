# CLAUDE.md — standing rules for the b70 project

Working notes for any agent on this repo. Keep it short; details live in `JOURNAL.md` / `docs/`.

## Style
- NO EMOJI. ASCII only in all files, commits, code, and terminal output. No emoji, no
  typographic Unicode (use `->` not an arrow, `...` not an ellipsis).
- ASCII diagrams and drawings are encouraged (boxes, arrows, tables built from `-|+/\`).

## Workflow
- Maintain a running **`JOURNAL.md`** (newest entry at the bottom): every experiment as
  config -> command -> result -> verdict. **Commit and push often.**
- Plans: `STRATEGY.md` / `MTP_TODO.md`. Findings: `FINDINGS.md`. Literature: `docs/literature/`.

## [!] ALWAYS verify which model/checkpoint is actually being tested
RTN vs GPTQ (and the quant scheme) get mixed up silently and have already corrupted a result -- the
Tier-1 HumanEval+ `w8a8` run served the **RTN** dup, not SmoothQuant+GPTQ. Before trusting any eval/bench:
1. Query the live server: `curl -s http://192.168.10.5:18080/v1/models | python3 -m json.tool`.
2. Cross-check the served id against `evals/configs/models.yaml` -> the exact model path.
3. `served_model_id` must encode the calibration method (`...-gptq` / `...-rtn`), never a bare `qwen3-14b-w8a8`.
4. Quant output dirs are method-tagged (`scripts/40,49,54` -> `...-${SCHEME}-${rtn|gptq}`). Less-performant
   dups are parked in `models/archive/` -- don't serve them.

## [!] Serialize B70 GPU runs through `scripts/gpu-run`
One B70, possibly several agents. Two overlapping GPU runs corrupt perf timings (noisy neighbor) and
a buggy WIP kernel can wedge the device under another run. So **gate every real GPU touch** (serve,
bench, perf_probe, on-GPU quant) behind the shared flock lease:
- `scripts/gpu-run <cmd>` -- runs `<cmd>` only while holding `/mnt/vm_8tb/b70/gpu.lock` (waits if held).
- `scripts/gpu-run --status` -- who holds it now (or `free`). Check before launching anything on the GPU.
- Editing/compiling stay fully parallel; only the GPU run is serialized. Zero cost when uncontended.
- A long-lived serve holds the lease for its lifetime (correct -- only one model fits the card anyway);
  stop the server (`docker stop vllm_*`) to release. Don't bypass with a bare `docker run --device /dev/dri`.

## [!!!] DEFAULT TO vLLM 0.23 -- image `vllm-xpu-env:v0230`. DO NOT use llm-scaler 0.14.x.
**START HERE for any XPU serve/quant/bench.** `vllm-xpu-env:v0230` = **vLLM 0.23.0+xpu** is our newest,
most-capable B70 stack: Triton fused-MoE on XPU, current Qwen3.6 / `Qwen3_5Moe` + Quark int8/int4 dispatch,
graph capture. **Never default to `intel/llm-scaler-vllm:0.14.x`** -- its vLLM is an ANCIENT 0.14 fork with
NO `_moe_C` MoE op suite (int8 MoE hard-fails on `topk_softmax`), and it has burned multiple agent-days as a
dead end (docs/kernel/20 sec 6-9). Newest-first preference: **v0230 (0.23.0) > :tf (0.20.2rc1) > 0.14.x**.
If a vLLM-XPU image NEWER than 0.23.0 exists, prefer it AND update this line + `rdy_to_serve/README.md`.
Copy-paste serves: `rdy_to_serve/` (self-contained per model) and `docs/SERVING.md`.

## [!] Serving a model? Use `docs/SERVING.md` -- the canonical recipe doc
It has copy-paste, verified serve commands (27B / 35B-A3B MoE / 14B), the `30_serve_w4a8_graph.sh` env
knobs, image picks, and the concurrency-sweep recipe. **Read it FIRST** -- do not reconstruct a serve
command from JOURNAL/scripts. When you find a working or changed recipe (new model, image, flag, gotcha),
**update `docs/SERVING.md`** (date it); don't leave the next agent to re-derive it.

## Where things live
- Models + quants: GPU host **Unraid @ 192.168.10.5**, under `/mnt/vm_8tb/b70/models/`
  (reachable via `ssh root@192.168.10.5`; NOT mounted on this dev box). Repo is synced to the host at
  `/mnt/vm_8tb/b70/` with a FLAT layout (serve/bench/gpu-run scripts at that root, not under `scripts/`).
- Serving fast path: our custom **INT8 W8A8 oneDNN kernel** (`contrib/vllm_int8_xpu`) in image
  `vllm-xpu-env:int8`. INT8 W8A8 is the real low-precision compute path on the B70 (Xe2 has no native FP8).
