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

## Where things live
- Models + quants: GPU host **Unraid @ 192.168.10.5**, under `/mnt/vm_8tb/b70/models/`
  (reachable via `ssh root@192.168.10.5`; NOT mounted on this dev box).
- Serving fast path: our custom **INT8 W8A8 oneDNN kernel** (`contrib/vllm_int8_xpu`) in image
  `vllm-xpu-env:int8`. INT8 W8A8 is the real low-precision compute path on the B70 (Xe2 has no native FP8).
