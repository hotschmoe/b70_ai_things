# AGENTS.md -- standing rules for the b70 project

Working notes for any agent on this repo. Keep this file short; details live in
`FINDINGS.md`, `JOURNAL.md`, `RESEARCH_TODO.md`, and `docs/`.

## Style

- ASCII only. No emoji, typographic arrows, or smart punctuation in files, commits,
  code, or terminal output.
- Keep status docs factual: config -> command -> result -> verdict.

## Current Research Focus

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

## Workflow

- Maintain `JOURNAL.md` newest entry at the bottom. Every experiment needs:
  config -> command -> result -> verdict.
- Use `RESEARCH_TODO.md` for active research ordering. Use `docs/quant_methods.md`
  for the method/scheme registry. Use `MTP_TODO.md` for all MTP planning.
- Commit and push often when working on the host. Do not rewrite old numbered
  experiment scripts; copy to a new number.

## Repo Layout Contract

- `scripts/NN_*.sh`: lab notebook, append-only. New experiment -> new number.
- `bin/`: stable shared tools used by serve/eval paths.
- `rdy_to_serve/<model>/`: verified shelf only. A model lands here only after
  serving is known-good and the recipe is self-contained.
- `contrib/`: patches and prototype kernels.
- `docs/`: stable explanations, kernel notes, and literature synthesis.
- `w4a8/`: W4A8 branch area. Coordinate before editing broadly.

Any change to `bin/` or `rdy_to_serve/_common/` needs `bin/serve-sweep --smoke`
green across shelf models before commit.

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

### DANGER: P2P in vLLM serve wedges the multi-GPU state

Do NOT run `CCL_TOPO_P2P_ACCESS=1` inside a vLLM TP>1 serve. It crashes at
worker init (`UR_RESULT_ERROR_DEVICE_LOST` in `xpu_worker.py` warmup all_reduce)
AND does not clean up: it corrupts the cross-GPU oneCCL / Level-Zero collective
state so that EVERY subsequent TP>1 serve -- even a known-good P2P-OFF one --
also fails with the same `DEVICE_LOST` until the GPU state is reset. Single-GPU
(TP=1) serves are unaffected. The raw mp.spawn allreduce microbench works fine
with P2P=1 (it is the oneCCL<->vLLM-multiproc-worker path that breaks, not the
hardware). See JOURNAL Lever A + P2P_GPU.md H.13.

- Recovery (CONFIRMED 2026-06-24): a reboot clears the wedge. Lighter option is
  reloading the driver `sudo modprobe -r xe && sudo modprobe xe` (needs no
  `/dev/dri` in use -- stop all containers first).
- If you must experiment with P2P-in-serve, do a GPU reset BETWEEN every attempt;
  never chain two `P2PACCESS=1` serve tries without a reset in between.

## Images And Serving

- Default vLLM image: `vllm-xpu-env:v0230` unless a specific recipe says
  otherwise.
- INT8 W8A8 research image: `vllm-xpu-env:int8g`, which includes the custom
  `XPUInt8ScaledMMLinearKernel` path and graph-capture fake registrations.
- For shelved models, start from `rdy_to_serve/<model>/serve.sh`. Do not
  reconstruct a serve command from old journal entries.

## Host Paths

- **We run LOCALLY on the box now (since the 2026-06-23 migration), NOT over SSH.** The GPU host is a
  local Ubuntu 26.04 machine (hostname `b70s4dayz`, kernel 7.0) and we act as user `hotschmoe` (uid 1000),
  not root. The old `ssh root@192.168.10.5` remote-driver workflow is RETIRED -- run commands on the box
  itself. Rollback/migration context lives in `MIGRATION.md` (section 13).
- GPU host LAN address: `192.168.10.5` (still its IP; just no longer SSH'd into from a laptop).
- Host project/model root: `/mnt/vm_8tb/b70/`.
- Models and quants: `/mnt/vm_8tb/b70/models/`.
- Repo tools live under `bin/` locally but are mirrored flat at the host root.
