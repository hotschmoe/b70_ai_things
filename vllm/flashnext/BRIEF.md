# Flash-Next on vLLM -- campaign brief (2026-10-19)

Goal: migrate Flash-Next serving (today: Strata night-final-v3, the daily driver
hotschmoe-dd) to the NEWEST upstream vLLM, carrying over the Strata features that
make it fast. Prefer, in order: (1) upstream vLLM features as-is, (2) out-of-tree
plugins / custom ops / runtime patches (vllm general_plugins, platform plugin,
our sitecustomize-style overlays under vllm/patches), (3) a thin fork branch that
stays rebase-clean on upstream main (small, isolated commits; no reformatting;
new files over edits to upstream files). No long-lived divergent fork.

Targets (AGENTS.md "Flash-Next goals"): prefill 2000+ tok/s, decode 150+ tok/s
per stream, 200k context per stream at concurrency 2, KV eviction to system RAM
restored without recompute. Current Strata DD: C1 ~141, C2 ~92/stream, prefill
~2200 tok/s, 2x200K, 32 GiB/4-slot RAM parking, image input.

Key references (read what you need):
- AGENTS.md (standing rules: ASCII only, gpu-run leases, health checks, identity)
- strata/flash-next/dd/README.md and hotschmoe-dd.json (DD config + env flags)
- docs/20261016_night_lab.md, docs/20261016_next_investigations.md,
  docs/20261013_flashnext_goals.md, docs/20261014_flashnext_strata_optimization.md
- Strata source: /mnt/vm_8tb/b70/build/strata-ab/src-ep (branch night-20261016)
- Model: /mnt/cache/b70/qwen3.8-flash-next/ista-gsq-rco/IQ3_XXS/*.gguf
  (+ MTP bf16 at /mnt/vm_8tb/b70/build/strata-ab/mtp-bf16/rt, pack at
  /mnt/vm_8tb/b70/build/strata-ab/packs/ista-gsq-rco-iq3-xxs); look in
  /mnt/cache/b70 and /mnt/vm_8tb/b70/models/files for other Flash-Next formats.
- Prior vLLM-XPU work in this repo: vllm/ (patches, contrib, int4, fp8, nvfp4),
  memory notes in JOURNAL.md (search "vLLM").

Locations:
- Scratch clones/builds: /mnt/vm_8tb/b70/build/vllm-flashnext/ (NOT /tmp; tmpfs)
- Reports: vllm/flashnext/reports/<NN>_<topic>.md (ASCII, concise, cite file:line)
- Code for the campaign: vllm/flashnext/ (plugins, patches, scripts)

Rules for agents:
- No GPU use unless your task explicitly says so; then ONLY via bin/gpu-run and
  after confirming the DD is stopped by the orchestrator. Heavy CPU (builds,
  big copies) via bin/cpu-run. Builds: MAX_JOBS<=4, RAM < 90 GB.
- Do not stop/start systemd units, do not commit or push, do not touch
  hotschmoe-dd files. Do not edit other agents' report files.
- Be evidence based: say what you verified vs inferred.

Mindset (user, 2026-10-19): there is no software moat. Anything can be made to
work, and work well, on this hardware -- there is a lot of performance left on
the table. If a kernel, format, or feature is missing on XPU, the answer is
"we build it", not "unsupported". Report missing pieces as work items with a
size estimate, never as blockers. Aim to lead, not to match.

Metric priority (user, 2026-10-19): when benchmarking/profiling, LATENCY is the
number we chase (per-token decode latency / ITL at C1 and C2, TTFT), ahead of
aggregate throughput.

Dev-model track (user, 2026-10-19): develop kernels, quant formats (W3A16,
W2A16, mixed int4/int3/int2 MoE), vLLM patches and deployment on a SMALL MoE
first, then scale to Flash-Next. Prefer a small model with the same building
blocks as Flash-Next (qwen4_exp: GDN/linear-attn + full attn hybrid, routed +
shared experts, PLE, MTP head).

## Wave 2 (2026-10-19 night) -- ownership map
DD is STOPPED by the orchestrator; GPU leases are free. Use bin/gpu-run (it
queues on the lease), bin/xpu-health before/after, bin/xe-reset after a crash.
Card plan: card 0 = kernel agents (A2, B2), card 1 = A3 smoke/harness and B1
AutoRound. Both cards (TP/EP=2) only A3/A1 and only with bin/xpu-collective-health.
Dev models: Ornith-1.5-35B-A3B (= Qwen3.6-35B-A3B finetune, qwen3_5_moe, 256
experts top-8 + shared, GDN hybrid, MTP) on disk at models/files/ornith-1.5-35b-a3b
(bf16-mtp-shisa, gptq-int4-mixedcal-v2, ...); official Qwen3.6-35B-A3B BF16 and
an IQ3_XXS GGUF of it get downloaded to /mnt/cache/b70/devmodel/ by A3.
Ownership (edit only your own paths; ask others via your report's "Requests"):
- A1 vLLM thin fork: /mnt/vm_8tb/b70/build/vllm-flashnext/vllm branch
  b70/flashnext (off origin/main). Small isolated commits, rebase-clean.
- A2 GGUF/IQ XPU kernels: vllm/flashnext/plugins/b70_gguf_xpu/
- A3 runtime + harness + downloads: vllm/flashnext/harness/ (xpu-run.sh,
  serve scripts, latency bench, coherence check)
- B1 quantization: vllm/flashnext/quant/
- B2 low-bit WNA16 XPU kernels: vllm/flashnext/plugins/b70_lowbit_xpu/
Plugins load through vLLM's general_plugins entry point (no core edits) and
register torch custom ops + quant methods. Reports: reports/1x_<agent>.md,
updated as you go (CONFIG -> COMMAND -> RESULT -> VERDICT).

Journal: append your milestone entries to vllm/flashnext/JOURNAL.md (format at
its top). Do NOT git commit/push yourself; the orchestrator commits at
milestones. Keep large artifacts (weights, builds, raw traces) out of the repo.
