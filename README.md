# B70 AI Inference Optimization

Research and engineering notes for local LLM inference on Intel Arc Pro B70
GPUs, with a focus on vLLM-XPU, compressed-tensors quantization, and custom
INT8 kernels for Battlemage/Xe2.

The stack: 2x Intel Arc Pro B70 (Battlemage / Xe2, 32 GB each), vLLM-XPU
(0.23.0), compressed-tensors W8A8 / W4A8 INT8 research, on a local Ubuntu 26.04
box (kernel 7.0). Since the 2026-06-23 migration we run LOCALLY on the box
(`b70s4dayz`), not over SSH -- see [MIGRATION.md](MIGRATION.md).

## Headline results

- **[BREAKTHROUGH] B70<->B70 GPU P2P is UNLOCKED on kernel 7.0 + IOMMU off.**
  `zeDeviceCanAccessPeer` = True both directions (it was False on all 12 probe
  variants on kernel 6.18). The quantified prize: a 64-layer TP=2 forward fires
  128 allreduces; today they go host-staged at ~1.16 GB/s, making 27B prefill
  **~84% collective-bound** (~2.3 s of a 2.75 s TTFT). Lifting the allreduce
  toward the ~15 GB/s Gen3 wire via P2P points to **~4x faster prefill TTFT**
  (~2.75 s -> ~0.65-0.75 s). P2P is a prefill/TTFT + concurrency win, ~1.2x on
  single-stream decode. See [docs/P2P_GPU.md](docs/P2P_GPU.md) H.10 / H.11 and
  [27b_w8a8_research.md](27b_w8a8_research.md).
- **27B serve throughput (dual B70, captured PIECEWISE):** W4A8 TP=2 single-stream
  decode **~22 t/s** (TP=1 single-card ~20.7); W8A8 27B TP=2 + captured MTP
  (spec=3) reaches **~34.82 t/s @ ~51% draft accept**, coherent -- 1.92x over
  captured-no-MTP (18.10). Single-card daily driver (27B int4-AutoRound, captured)
  decodes ~30.8 t/s. See [FINDINGS.md](FINDINGS.md) and [JOURNAL.md](JOURNAL.md).

### Serve shelf benchmarks (2026-06-24, new Ubuntu / kernel-7.0 install)

Validation sweep of the `rdy_to_serve/` Qwen3.6 shelf on the migrated box -- every
model served coherently with the images recovered from the old Unraid `docker.img`.
`vllm bench serve`, random dataset, IN=2048 / OUT=128, captured (GRAPH=1). **PP** =
prefill throughput (2048 / TTFT); **TG** = decode tok/s. TP=1 models were swept
two-up (one per card; decode verified identical to solo). NB: the TP=2 rows ran
**host-staged** (the serve recipes default `CCL_TOPO_P2P_ACCESS=0`); enabling the
kernel-7.0 P2P path (`P2PACCESS=1`) is the unrealized TTFT lever, see below.

| model | img | TP | TTFT c1 | PP c1 (tok/s) | TG c1 (t/s) | TTFT c4 | agg c4 (tok/s) |
|---|---|---|---|---|---|---|---|
| qwen36-35b-a3b-int4 (MoE) | v0230moe | 1 | 441 ms | 4641 | 68.5 | 1239 ms | 123.7 |
| qwen36-27b-w4a8 | int8g | 1 | 853 ms | 2400 | 20.7 | 2201 ms | 51.2 |
| qwen36-27b-w4a16 | v0230 | 1 | 1224 ms | 1673 | 21.2 | 3213 ms | 45.5 |
| qwen36-27b-int4 | v0230 | 1 | 1326 ms | 1545 | 30.5 | 3438 ms | 51.3 |
| qwen36-27b-w8a8-sqgptq-mtp | int8g | 2 | 2961 ms | 692 | 25.2 (MTP) | 6837 ms | 19.0 |
| qwen36-35b-a3b-quark-w8a8 (GRAPH=1) | v0230 | 2 | 1512 ms | 1354 | 43.1 | 3866 ms | 53.2 |

The 35B-A3B MoE (~3B active) is fastest end to end. Caveats: (1) the TP=2 rows are
**host-staged**. P2P would cut their prefill-bound TTFT ~3-4x (allreduce A/B, P2P_GPU
H.12, 8.4x) -- but `P2PACCESS=1` currently **crashes the vLLM TP=2 serve** at worker
init (`UR_RESULT_ERROR_DEVICE_LOST`), so P2P is microbench-only for now: real at the
allreduce layer, not yet reachable end-to-end (P2P_GPU H.13). (2) the bench
uses `--dataset-name random`, which depresses MTP acceptance (random tokens are
undraftable) -- so the W8A8 27B `25.2 (MTP)` understates it; on coherent NL the
captured-MTP spec=3 ceiling is ~35 t/s @51% accept (JOURNAL 2026-06-24). (3) the
35B quark-W8A8 row is shown CAPTURED (`GRAPH=1`); the `68_shelf_bench_par.sh` sweep
ran it EAGER (serve.sh default) at 4.6 t/s, which understated it 8.7x -- PIECEWISE
capture removes the per-token op-launch overhead that dominates this ~3B-active MoE
in eager mode (Lever B, below). Sweep harness: `68_shelf_bench_par.sh` (TP=1 two-up,
TP=2 solo); the 35B-quark captured numbers are from `69_lever_tests.sh` arm B.

#### Optimization levers tested (2026-06-24, `69_lever_tests.sh`)

- **P2P in serve (A):** unlocked at the allreduce layer (8.4x, H.12) but
  `P2PACCESS=1` **crashes the vLLM TP=2 serve** at worker-init and **wedges the
  multi-GPU state** (even P2P-off TP=2 then fails until an `xe` reload/reboot;
  reboot CONFIRMED to clear it 2026-06-24). Never chain two `P2PACCESS=1` serve
  attempts without a GPU reset between them. Not reachable end-to-end yet -- see
  P2P_GPU H.13 and the AGENTS.md GPU Discipline danger note.
- **MoE int4 MTP (C):** works no-graft (mtp head intact), ~**1.11x** single-stream
  decode (66 -> 74 t/s, random-data floor) but **net-negative at c>1** -- the MoE's
  ~3B active params make decode already fast, so MTP's verify overhead doesn't pay
  off like it does on the 27B dense (1.9x).
- **35B quark-W8A8 eager vs captured (B):** PIECEWISE graph capture (`GRAPH=1`) is
  **8.7x** single-stream decode over eager (4.96 -> **43.1 t/s**, TPOT 202 -> 23ms)
  and **3.7x** at c=4 (14.4 -> 53.2 t/s agg) -- both coherent. Eager's per-token
  op-launch overhead dominates this ~3B-active MoE; capture removes it. Makes it the
  fastest single-stream 35B we serve. Recommend flipping the serve.sh default to
  `GRAPH=1` (P2PACCESS=1 arm skipped -- it re-wedges; see A). JOURNAL 2026-06-24.

Start with:

- [FINDINGS.md](FINDINGS.md): current working results and dead ends.
- [RESEARCH_TODO.md](RESEARCH_TODO.md): active research order.
- [rdy_to_serve/README.md](rdy_to_serve/README.md): verified serve shelf.
- [JOURNAL.md](JOURNAL.md): full experiment log.
- [AGENTS.md](AGENTS.md): standing rules for agents working in this repo.

## Current Focus

The research direction is **compressed-tensors first**. We want one comparable
artifact format across W8A8, W4A8, W4A16, TP=2, PP=2, and custom kernel work.

Primary tracks:

- **W8A8 INT8:** main path for B70 INT8 XMX research. The 14B baseline serves
  through our `XPUInt8ScaledMMLinearKernel`; 27B W8A8 serves via TP=2.
- **W4A8 INT8-activation:** next memory/perf research path. This is where custom
  int4-weight x int8-act GEMM/GEMV work should converge.
- **W4A16:** capacity baseline. AutoRound/INC int4 is the proven daily-driver
  serve path today; compressed-tensors W4A16 for 27B is intentionally deferred
  to a focused padding/ignore-list/kernel session.
- **W4A4:** later frontier research. Interesting, but not the near-term kernel
  priority.

Calibration policy today:

- Use **GPTQ** as the default producer for compressed-tensors quants.
- AutoRound remains a useful comparison and a proven int4 serving route.
- GPTQ slightly beat AutoRound on 14B W8A8 HumanEval+; verify on harder evals
  before making broad claims, especially for W4A8.

## Headline Kernel Work

The repo contains a working INT8 W8A8 path for B70:

- `contrib/vllm_int8_xpu/`: custom vLLM XPU int8 linear kernel integration.
- [docs/kernel/02_int8_w8a8_status.md](docs/kernel/02_int8_w8a8_status.md):
  kernel status and usage.
- `vllm-xpu-env:int8g`: image used by the current 14B W8A8 serve baseline.

The key runtime signal is:

```text
Selected XPUInt8ScaledMMLinearKernel for CompressedTensorsW8A8Int8
```

## Environment

- Host: local Ubuntu 26.04 box `b70s4dayz` (kernel 7.0) at `192.168.10.5`. We
  run LOCALLY on the box as user `hotschmoe`, not over SSH (retired in the
  2026-06-23 migration; see [MIGRATION.md](MIGRATION.md)).
- GPU root: `/mnt/vm_8tb/b70/`
- Models/quants: `/mnt/vm_8tb/b70/models/`
- GPUs: dual Intel Arc Pro B70 cards (Battlemage / Xe2), 32 GB each.
- Runtime: Docker with `/dev/dri` passed through.
- Default vLLM image: `vllm-xpu-env:v0230`, unless a serve recipe specifies
  `:int8g` or another image.

All serious GPU runs should go through `gpu-run`; see [AGENTS.md](AGENTS.md).

## Layout

| Path | Purpose |
|---|---|
| `AGENTS.md` / `CLAUDE.md` | Agent rules; `CLAUDE.md` is a symlink. |
| `FINDINGS.md` | High-signal current results. |
| `RESEARCH_TODO.md` | Active research plan and priorities. |
| `JOURNAL.md` | Append-only experiment log. |
| `rdy_to_serve/` | Verified serve recipes. |
| `contrib/` | Kernel and vLLM patches. |
| `docs/` | Kernel notes, hardware notes, method registry, literature. |
| `scripts/` | Numbered experiment scripts. |
| `bin/` | Stable tools shared by serve/eval workflows. |
