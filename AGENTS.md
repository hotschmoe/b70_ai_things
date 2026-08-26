# AGENTS.md -- standing rules for the b70 project

Working notes for any agent on this repo. Keep this file short; details live in
`FINDINGS.md`, `JOURNAL.md`, `RESEARCH_TODO.md`, and `docs/`.

## Style

- ASCII only. No emoji, typographic arrows, or smart punctuation in files, commits,
  code, or terminal output.
- Keep status docs factual: config -> command -> result -> verdict.

## Current Research Focus

- **W8A8 INT8 of ANY model is a standing target.** 8-bit weights are preferred over
  4-bit (clearly better quality) and B70 has INT8 XMX fast paths to leverage. Every
  headline model should get a W8A8 int8 serve path; prioritize building/keeping it.
- **sglang is the primary serving backend; vLLM is paused.** vLLM batches concurrent
  prefill+decode together and emits "!!!!" garbage under load; sglang does not. Keep the
  vLLM shelf as a maintained paused baseline, but new serve work targets sglang.
- **Use compressed-tensors as the research artifact format across models and schemes.**
  This keeps W8A8, W4A8, W4A16, TP=2, PP=2, and custom kernel work comparable.
- **W8A8 and W4A8 are the main kernel research paths.** They exercise int8
  activations and are the paths that can use B70 INT8 XMX fast paths.
- **GPTQ is the default calibration method for compressed-tensors runs today.**
  It beat AutoRound on 14B W8A8 HumanEval+ by a small margin. Treat that as a
  current working choice, not a final law; verify on harder evals, especially
  before making W4A8 conclusions.
- **AutoRound/INC int4 remains the proven W4A16 serve baseline.** Do not confuse
  that with the research direction. Compressed-tensors W4A16 for 27B is still
  worth fixing, but in a focused kernel/loader session.
- **W4A4 is later frontier research.** Keep notes, but do not start W4A4 kernel
  work until W8A8/W4A8 are robust.
- **Carry the 2026-08-25 TP=2 lesson into every 27B dense experiment.** Steve's
  clone-safe custom-op contract alone was insufficient on this runtime: the
  cloned profile input also needed completion before oneCCL consumed it from
  another queue. The exact Qwen35 control fixes only large profile tensors and
  leaves graph capture/decode unfenced. For a 27B model, measure its real
  profile shape and collective count; do not blindly copy the 8192-row
  threshold. Require matched per-rank entry/return evidence plus per-card and
  compiled collective post-health. This is especially valuable on dense 27B,
  where MoE dispatch cannot confound the collective result. See JOURNAL
  2026-08-25p and `docs/P2P_GPU.md` J.23.
- **Steve timing and native-source closure is the active transfer map.** The
  exact `e190923b` source plus scratch-aware MoE interface passed at 48.5315
  tok/s. Synchronized pure-decode timing then proved the same 41-piece graph
  topology as Steve, but rank-0 model-forward is 22.675 ms versus his 5.695 ms
  (3.982x). The prior "June native" package is specifically a June-9 minimal
  patch over `28e1f5e`, not Steve's later native tree. Exact checkpoint
  `122b698b` is recoverable and adds real output-buffer variants for per-token
  quantization and fused SiLU/multiply/quantization; the
  current scratch-aware dispatcher probes for these ops but the June-9 binary
  lacks them and falls back to allocate/copy. On the accepted path, fused
  SiLU+quant is explicitly unset; the relevant delta is two scratch-targeted
  per-token quantizations across each of 40 MoE layers, or 80 calls/step. Treat
  Exact `122b698b` native binaries are now measured: 50.3706 tok/s versus the
  matched June-9 unsynchronized control at 48.5315 tok/s, a coherent +3.79
  percent with both 16/16 canaries and post-teardown health green. This proves
  native scratch-targeted quant output is useful but not Steve's missing 1.7x;
  synchronized timing on this binary is the next localization arm. Never
  compare the 50.3706 unsynchronized endpoint with the deliberately synchronized
  35.4699 diagnostic. For dense 27B, repeat the graph-piece and synchronized
  timing census, then transfer proven reusable quant/output-buffer primitives
  separately; do not copy MoE-only layerlet/sidecar code, Qwen35's mixed MoE
  workspace, or its 8192-row fence threshold. See JOURNAL 2026-08-26 and
  `docs/20260825_steve_stack_component_ledger.md`.

## Workflow

- Maintain `JOURNAL.md` newest entry at the bottom. Every experiment needs:
  config -> command -> result -> verdict.
- Use `RESEARCH_TODO.md` for active research ordering. Use `docs/quant_methods.md`
  for the method/scheme registry. Use `MTP_TODO.md` for all MTP planning.
- Commit and push often when working on the host. Do not rewrite old numbered
  experiment scripts; copy to a new number.

## Repo Layout Contract

The repo is split by serving backend. Backend-specific code lives under its backend
root; shared, backend-agnostic tooling stays at the repo root.

- `sglang/`, `vllm/`, `llamacpp/`, `zml/`: backend roots. Each holds that backend's
  serve/bench/probe scripts, patches/shims, build recipes, images, and scheme research
  (e.g. `sglang/w8a8/`, `vllm/w4a8/`). sglang = primary; vllm = paused baseline;
  llamacpp (SYCL/GGML, weight-only GGUF) + zml (oneAPI PJRT, bf16/f16) added 2026-06-30 --
  upstream sources cloned git-ignored under `/mnt/vm_8tb/b70/{llama.cpp,zml}`. See
  `docs/intel_support_per_backend.md` for the per-backend Intel-Arc support comparison.
- `kernels/`: SHARED custom-kernel SOURCE -- the oneDNN int8/int4 gemm ops, the
  `int8_gemm_kernel.patch`, and the op headers. ONE source of truth, applied to
  `vllm-xpu-kernels` and compiled PER BACKEND (ABI-incompatible: vLLM-torch ->
  `:int8g` image; sglang torch 2.12 -> runtime `_xpu_C.abi3.so` + shim). The built
  `.so` binaries are git-ignored runtime artifacts under `/mnt/vm_8tb/b70`, not repo content.
- `rdy_to_serve/<backend>/<model-quant>/`: the verified shelf. `_common/lib.sh` is shared.
- `bin/`: stable shared, backend-agnostic tools (gpu-run, serve-sweep, xpu-health, xe-reset).
- `models/`: model registry (manifest + fetch + reorg); weights in `models/files/` (git-ignored).
- `docs/`, `evals/`, `agentic-eval/`, `migration/`: shared.
- `scripts/NN_*.sh`: append-only lab notebook (historical; do NOT rewrite -- copy to a
  new number). NEW experiment scripts go under the relevant backend root.

### Shelf rules (`rdy_to_serve/<backend>/<model-quant>/`)

- EXACTLY ONE self-contained best config per (backend, model, quant). NO variations --
  no `-mtp`/`-graph`/`-sqgptq` sibling dirs. The most performant + coherent options
  (MTP, fused kernels, graph capture) are baked in as settings, not separate entries.
- "Best" = best behavior under CONCURRENT/serving load: coherent first, then fast. The
  failure mode that matters is concurrent prefill+decode (vLLM's "!!!!").
- NEVER update a shelf entry until the new settings are MEASURED both faster-or-equal
  AND coherent (sweep-gated). An untested "improvement" does not land.
- Any change to `bin/` or `rdy_to_serve/_common/` needs `bin/serve-sweep --smoke` green
  across shelf models before commit.

## Model Identity

RTN, GPTQ, AutoRound, and quant scheme mixups have already corrupted results.
Before trusting any eval or bench:

1. Query the live server: `curl -s http://192.168.10.5:18080/v1/models | python3 -m json.tool`.
2. Cross-check the served id against `evals/configs/models.yaml`.
3. Served ids and output dirs must encode method and scheme, for example
   `...-W8A8-gptq`, `...-W8A8-autoround`, or `...-W4A8-sqgptq`.
4. Do not serve a bare ambiguous id such as `qwen3-14b-w8a8`.

## GPU Discipline

Use the shared lease for every real GPU touch:

- `gpu-run <cmd>` locks both cards. Use this for TP=2, PP=2, data parallel, or
  anything that might touch both cards.
- `gpu-run --card N <cmd>` locks one card. Pair with the workload's card pin.
- `gpu-run --status` shows current holders.

Editing and compiling can run in parallel. Serving, benchmarking, perf probes,
and on-GPU quantization must not bypass the lease.

### STATUS 2026-07-02: kernel 7.1 CURED the TP=2 BCS/GuC hardware wedge

The box is now on **kernel 7.1.0-070100 + Intel Compute Runtime 26.22.38646.4** (was 7.0 + 26.05; runbook
`docs/20260702_kernel71_upgrade_plan.md`, JOURNAL 2026-07-02). This CURED the TP=2 "device_lost" BCS
copy-engine / GuC-firmware-skew wedge -- 7.1's KMD wants GuC 70.58.0 so there is no skew, and the 70.54.0
pin is RETIRED (do NOT re-add it on 7.1). **w8a8 TP=2 (and DP=2) is the production daily driver and is STABLE:**
a 12h+ run, millions of tokens, heavy cache-hit load ran clean even on 7.0+pin, then 5/5 back-to-back TP=2
serve cycles ran clean on 7.1. **The old "w8a8 TP=2 = attended-only" caveat is RETIRED -- unattended TP=2
serving is fine.**

The P2P-in-vLLM-serve / chained-TP>1-worker-crash oneCCL wedge documented next is a SEPARATE software
mechanism (oneCCL <-> vLLM-multiproc collective state), NOT the GuC hardware wedge. An early guarded exact
Qwen TP2 retest on 2026-08-25 failed at a compiled `vllm::all_reduce`; later exact-stack repairs reached a
coherent direct-P2P endpoint on the same boot series. Direct P2P therefore remains guarded experimental work,
not a production setting. Use both card-level and compiled two-rank collective health around each attempt.

### DANGER: P2P in vLLM serve wedges the multi-GPU state

Do NOT run arbitrary `CCL_TOPO_P2P_ACCESS=1` vLLM TP>1 serves. Stock and earlier
custom paths crash at worker init or compiled profile all-reduce and can corrupt
the cross-GPU oneCCL / Level-Zero state. The one scoped exception is
`vllm/w8a8/run_qwen36_s2b_clone_exact_control.sh`: its direct-P2P arm defaults
to a clone-completion fence only for profile tensors with at least 8192 rows,
requires `I_KNOW_P2P_WEDGES=1`, and passed its exact metric/canaries plus both
post-health layers on 2026-08-25. It remains experimental, not a shelf setting.
The raw mp.spawn allreduce microbench also works with P2P=1; this is a vLLM
multiprocess/queue-handoff issue, not a peer-DMA hardware failure. See JOURNAL
2026-08-25p and P2P_GPU.md J.23.

- GUARD (2026-06-24; reset correction 2026-08-25, P2P_GPU.md J.17/J.22): a layered wedge guard wraps the TP>1 serve path
  (TP=1 unchanged). `bin/xpu-health` detects a wedged box (per-card matmul, timeout-wrapped);
  `bin/xpu-collective-health` detects collective-only failure; `bin/xe-reset` runs rebind -> xe reload ->
  endpoint FLR as a non-reboot recovery ladder. lib.sh runs a pre-flight
  probe, graceful `docker stop` teardown, a stall-aware health wait, and a post-teardown verdict.
  Set `B70_AUTO_RESET=1` to auto-recover. `CCL_TOPO_P2P_ACCESS=1` in a TP>1 serve is now refused
  unless `I_KNOW_P2P_WEDGES=1`. xe-reset needs the scoped sudoers in `bin/xe-reset.sudoers`.
- Recovery (CORRECTED 2026-08-25, P2P_GPU.md J.22): **the cards are not display-held.** All connectors are
  disconnected/disabled, `/proc/fb` is empty, the VT uses the dummy console, and no process holds `/dev/dri`.
  The old unload failed because both B70 PCI functions and their four xe auxiliary children were still bound.
  Unbind both first: clean-state rebind, `xe` unload/reload, and endpoint FLR were all validated without reboot,
  with the same boot ID and green per-card plus compiled two-rank health after each. Use `bin/xe-reset`; reboot
  only if unbind hangs or the full ladder fails. Clearance of a future naturally occurring deep wedge remains
  to be recorded, so retain the final reboot fallback.
- If you must experiment with P2P-in-serve, run `bin/xe-reset` BETWEEN every attempt;
  never chain two `P2PACCESS=1` serve tries without a reset in between.
- ALSO (CONFIRMED 2026-06-24, P2P_GPU.md J.15): it is NOT only `P2PACCESS=1` that wedges
  this state. A string of TP>1 WORKER-INIT CRASHES (e.g. repeated GRAPH=1 model-load
  failures, or serves killed mid-init) corrupts the same cross-GPU oneCCL/L0 state, so
  every later TP=2 serve then `UR_RESULT_ERROR_DEVICE_LOST`s at oneCCL warmup EVEN at
  `P2PACCESS=0`. Do not chain crash-prone TP=2 starts; reset xe
  (modprobe -r/-add or reboot) after a TP=2 worker-init crash before retrying.
- CORRECTION (CONFIRMED 2026-06-24, P2P_GPU.md J.16): the wedge is NOT always spared on
  single-GPU. A TP>1 serve whose workers are killed MID-GRAPH-CAPTURE (e.g. by the
  `b70_wait_healthy` 15-min timeout) can degrade BOTH cards at the xe/driver level so that
  even a TP=1 single-card op fails -- presenting as `UR_RESULT_ERROR_OUT_OF_RESOURCES`
  (err 40, OOM-class) or a multi-minute hang, NOT only `DEVICE_LOST` (err 20). So "single-GPU
  stays fine" from H.13/J.15 does NOT always hold. After ANY TP>1 teardown that threw
  DEVICE_LOST in shutdown, verify health with a single-card matmul probe BEFORE the next
  TP>1 start; if it hangs or OOMs, reset xe first.

## Images And Serving

- Default vLLM image: `vllm-xpu-env:v0230` unless a specific recipe says
  otherwise.
- INT8 W8A8 research image: `vllm-xpu-env:int8g`, which includes the custom
  `XPUInt8ScaledMMLinearKernel` path and graph-capture fake registrations.
- For shelved models, start from `rdy_to_serve/<backend>/<model>/serve.sh`. Do not
  reconstruct a serve command from old journal entries.

## Host Paths

- **We run LOCALLY on the box now (since the 2026-06-23 migration), NOT over SSH.** The GPU host is a
  local Ubuntu 26.04 machine (hostname `b70s4dayz`, kernel 7.1 since the 2026-07-02 upgrade that cured the TP=2
wedge -- see `docs/20260702_kernel71_upgrade_plan.md`) and we act as user `hotschmoe` (uid 1000),
  not root. The old `ssh root@192.168.10.5` remote-driver workflow is RETIRED -- run commands on the box
  itself. Rollback/migration context lives in `MIGRATION.md` (section 13).
- GPU host LAN address: `192.168.10.5` (still its IP; just no longer SSH'd into from a laptop).
- **The git repo is now a SINGLE clone on the 8TB SSD: `/mnt/vm_8tb/github/b70_ai_things`**
  (consolidated 2026-06-24; the old two-repo split -- `~/github` checkout vs a non-git flat
  copy at `/mnt/vm_8tb/b70` -- is RETIRED). Work, commit, and push from there. `~/github/b70_ai_things`
  no longer exists.
- **Runtime data root: `/mnt/vm_8tb/b70/`** -- caches, kernel build artifacts, `gpu.lock*`. NOT a repo.
  Recipes default `ROOT=/mnt/vm_8tb/b70` (lib.sh) to find caches and `35_sweep_bench.sh`.
  `gpu-run` and `35_sweep_bench.sh` are kept at this root as symlinks into the clone's `bin/`.
- **Model weights live in the repo (since the 2026-06-29 reorg): `models/files/<family>/<scheme>/`**
  (git-ignored, de-rooted, no symlinks, complete with vision+MTP). lib.sh mounts
  `models/files` -> `/models` in the container. The old `/mnt/vm_8tb/b70/models/` is RETIRED.
  Reprovision a fresh box via `bash models/fetch.sh` (see `models/manifest.yaml`).
- Run pattern from the clone: `cd /mnt/vm_8tb/github/b70_ai_things && ./bin/gpu-run bash scripts/NN_*.sh`
  (or `/mnt/vm_8tb/b70/gpu-run`, the symlink). Tools live under `bin/`, experiments under `scripts/`.
