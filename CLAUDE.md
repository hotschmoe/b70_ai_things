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

## [!!!] Repo layout + anti-clobbering contract (full detail: `ORGANIZATION.md`)
Four tiers, each with a mutability contract -- so testing new stuff never clobbers a working serve:
- **`scripts/NN_*.sh` = lab notebook (APPEND-ONLY).** Never edit a committed `NN` to test a new idea --
  copy to the next number. Re-run-constantly scripts are tools, not lab entries -> they live in `bin/`.
- **`bin/` = shared tools (STABLE).** `gpu-run`, the serve engine, sweep harness, DP wrapper. Depended on
  by the lab notebook + daily driver + golden path. Mirrors the host flat root by filename.
- **`images/` = image recipes.** Docker tags are **IMMUTABLE**: never `docker commit`/rebuild onto an
  existing tag -- new build -> new dated/sha tag + a Dockerfile. The base image is **never** patched.
  A behavior patch is either **bind-mounted per-container (pure-Python)** or **baked on a separate leaf
  tag (compiled)** -- never baked into the shared base (that is how a MoE patch can't break dense models).
- **`rdy_to_serve/<model>/` = golden path (VERIFIED, self-contained).** serve.sh + local patches/ + README.
  A model lands here only when it is current-best AND verified to serve. The shelf is curated, not a mirror.
- **SWEEP GATE:** any change to `bin/` or `rdy_to_serve/_common/` requires `bin/serve-sweep --smoke` GREEN
  across all shelf models before commit (`--bench` if it could move perf).

## [!] ALWAYS verify which model/checkpoint is actually being tested
RTN vs GPTQ (and the quant scheme) get mixed up silently and have already corrupted a result -- the
Tier-1 HumanEval+ `w8a8` run served the **RTN** dup, not SmoothQuant+GPTQ. Before trusting any eval/bench:
1. Query the live server: `curl -s http://192.168.10.5:18080/v1/models | python3 -m json.tool`.
2. Cross-check the served id against `evals/configs/models.yaml` -> the exact model path.
3. `served_model_id` must encode the calibration method (`...-gptq` / `...-rtn`), never a bare `qwen3-14b-w8a8`.
4. Quant output dirs are method-tagged (`scripts/40,49,54` -> `...-${SCHEME}-${rtn|gptq}`). Less-performant
   dups are parked in `models/archive/` -- don't serve them.

## [!] Serialize B70 GPU runs through `gpu-run` (repo: `bin/gpu-run`; host: `/mnt/vm_8tb/b70/gpu-run`)
One B70, possibly several agents. Two overlapping GPU runs corrupt perf timings (noisy neighbor) and
a buggy WIP kernel can wedge the device under another run. So **gate every real GPU touch** (serve,
bench, perf_probe, on-GPU quant) behind the shared flock lease:
- `gpu-run <cmd>` -- runs `<cmd>` only while holding `/mnt/vm_8tb/b70/gpu.lock` (waits if held).
- `gpu-run --status` -- who holds it now (or `free`). Check before launching anything on the GPU.
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

## [!] Serving a model? `rdy_to_serve/<model>/` first, then `docs/SERVING.md`
For a shelved model, `cd rdy_to_serve/<model> && bash serve.sh` is the authoritative, self-contained
serve -- do not reconstruct it. `docs/SERVING.md` is the index + the cross-cutting recipes (the generic
`bin/30_serve_w4a8_graph.sh` engine knobs, image picks, DP/TP/PP, tool-calling, concurrency sweep) +
the recipes not yet shelved. **Read one of these FIRST** -- never reconstruct a serve from JOURNAL/scripts.
When you find a working or changed recipe, update the golden dir (if shelved) or `docs/SERVING.md` (date it).

## Where things live
- Models + quants: GPU host **Unraid @ 192.168.10.5**, under `/mnt/vm_8tb/b70/models/`
  (reachable via `ssh root@192.168.10.5`; NOT mounted on this dev box). Repo is hand-synced to the host
  at `/mnt/vm_8tb/b70/` with a FLAT layout (the `bin/` tools live at that root, NOT under `bin/`; the host
  is not a git repo). Sync a tool: `tar czf - -C bin <f> | ssh root@192.168.10.5 'tar xzf - -C /mnt/vm_8tb/b70'`.
- Repo tools live in **`bin/`** (filenames match the host flat names). Lab notebook: `scripts/NN_*.sh`.
- Serving fast path: our custom **INT8 W8A8 oneDNN kernel** (`contrib/vllm_int8_xpu`) in image
  `vllm-xpu-env:int8`. INT8 W8A8 is the real low-precision compute path on the B70 (Xe2 has no native FP8).
