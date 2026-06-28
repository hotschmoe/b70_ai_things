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
two-up (one per card; decode verified identical to solo). NB: the 27B-W8A8 TP=2 row is the
custom **push-allreduce** path, now the **shelf default** (`PUSH_AR=1 PUSH_AR_GRAPH=1`; opt out with
`PUSH_AR=0`): **3.8x prefill TTFT / +80-126% agg / +8-10% decode** vs oneCCL, decode all-reduce also
graph-captured on the push transport (see caveat (1)). That P2P/TTFT lever is realized via push-ar at
`P2PACCESS=0` -- NOT via `P2PACCESS=1`, which wedges the box (reboot-only recovery). See caveat (1).

| model | img | TP | TTFT c1 | PP c1 (tok/s) | TG c1 (t/s) | TTFT c4 | agg c4 (tok/s) |
|---|---|---|---|---|---|---|---|
| qwen36-35b-a3b-int4 (MoE) | v0230moe | 1 | 441 ms | 4641 | 68.5 | 1239 ms | 123.7 |
| qwen36-27b-w4a8 | int8g | 1 | 853 ms | 2400 | 20.7 | 2201 ms | 51.2 |
| qwen36-27b-w4a16 | v0230 | 1 | 1224 ms | 1673 | 21.2 | 3213 ms | 45.5 |
| qwen36-27b-int4 | v0230 | 1 | 1326 ms | 1545 | 30.5 | 3438 ms | 51.3 |
| qwen36-27b-w8a8-sqgptq-mtp (push-ar, **cudagraph=NONE**) | int8g | 2 | 787 ms | 2604 | 23.4 MTP (25.6 coherent) | 1493 ms | 51.5 |
| qwen36-35b-a3b-quark-w8a8 (GRAPH=1) | v0230 | 2 | 1512 ms | 1354 | 43.1 | 3866 ms | 53.2 |

**sglang W8A8 -- the target backend (NEW 2026-06-28):** built fused int8 oneDNN ops (`int8_gemm_w8a16` decode /
`int8_gemm_w8a8` prefill) -> the int8 path now HANDILY beats bf16/fp8 on PP, TTFT, AND TG with vision retained.
`sglang.bench_serving`, IN=2048/OUT=128, warm c1, TP=2:

| sglang driver | TP | TTFT c1 | PP c1 (tok/s) | TG c1 (t/s) | notes |
|---|---|---|---|---|---|
| qwen36-27b-w8a8 fused + NEXTN MTP (steps=10) | 2 | **471 ms** | **4344** | **25.2** | int8 kernels + MTP; vision; greedy; PP/TTFT champ |
| qwen36-27b-w8a8 fused eager | 2 | **448 ms** | **4570** | 8.1 | int8 kernels; vision; max PP / lowest TTFT |
| bf16 TP=2 (reference) | 2 | 661 ms | 3098 | 9.03 | the W8A8 target |

vs bf16: W8A8 fused+MTP = **PP +40%, TTFT -29%, TG +180% (2.8x)**; beats int4+MTP on decode (25.2 vs 15.3).
FP8 emulated on B70 (~1.0x bf16 prefill) -> W8A8 wins PP vs fp8 too. Recipe: sglang/README.md, scripts/123-125,
w8a8/ (kernels: w8a8/W8A8_BUILD.md; campaign: w8a8/W8A8_SGLANG_PLAN.md).

#### 2026-06-26 update -- `cudagraph=NONE` (the STABLE config) + GuC 70.54.0 firmware fix

The MTP+graph-capture campaign (`docs/20260625_w8a8_27b_mtp_graph_campaign.md`) found PIECEWISE capture
**crashes under sustained load** (torch-xpu graph-replay command-stream accumulation; ~20-32k tokens). The fix is
`cudagraph_mode=NONE` (keep torch.compile, drop graph replay) -- stable, no accumulation. SEPARATELY, the TP=2
"device_lost" hardware wedge was root-caused to a kernel **BCS copy-engine job timeout on xe/GuC** and FIXED by
pinning **GuC firmware 70.54.0** (matching the KMD; `docs/20260625_bcs_wedge_rootcause.md`). The w8a8 row above is
now the SHIPPED `cudagraph=NONE` config (re-benched 2026-06-26). Captured/PIECEWISE numbers (faster, but they
crash) are kept only for reference.

Decode @ 2048 ctx, c1 (single-stream): `cudagraph=NONE` (stable) vs captured (fast-but-crashes):

| config | TTFT | PP (tok/s) | **TG NONE (stable)** | TG captured (crashes) |
|---|---:|---:|---:|---:|
| qwen36-27b-w8a8 (TP=2, MTP) | 787 ms | 2604 | **23.4 / 25.6 coherent** | 33.9 (PIECEWISE) |
| qwen36-27b-int4 (TP=1) | 1285 ms | 1593 | **23.3** | 30.5 (PIECEWISE) |
| qwen36-27b-w4a16 (TP=1) | 1199 ms | 1709 | **21.5** | 21.6 (NONE ~= captured: free) |

Takeaways: (a) on the **stable NONE route, w8a8 TP=2 is FASTER than int4** -- 1.6x prefill (2604 vs 1593, TTFT 787
vs 1285 ms) and a slight decode edge on coherent text (25.6 vs 23.3, via MTP), plus 8-bit quality. (b) NONE costs
int4 ~23% decode (30.5->23.3) but is **free for w4a16** (compute-bound). (c) For an unattended multi-day serve:
**int4 single-card DP=2 is wedge-proof** (no TP=2), while **w8a8 NONE TP=2 is faster + higher quality** but its
longevity rides on the GuC fix holding. eager floors: int4 8.2, w8a8 12.7.

#### TP=2 sweep + KV-cache budget + 4-bit MTP (2026-06-26)

Curiosity check: int4/w4a8/w4a16 on **TP=2** (they fit one card, so normally single-card / DP=2). Result: forcing
a fits-one-card model onto TP=2 **HURTS prefill** (oneCCL all-reduce tax, ~675 vs 1593 single-card) and **triples
TTFT**, and decode regresses on NONE -- because these 4-bit recipes have **no push-ar** (the fast L0-IPC all-reduce
is a w8a8-only overlay). w8a8 TP=2 keeps high prefill (2604) only because it natively needs sharding AND has push-ar.
So a fits-one-card 4-bit model should serve **single-card DP=2**, not TP=2. The one TP=2 upside is a much bigger
**KV budget** (the model shards across cards, freeing VRAM for KV) -- but harness tests don't need 444k ctx.

decode/TTFT/prefill @ c1, 2048 ctx (decode=1000/TPOT, prefill=2048000/TTFT):

| config | TP | decode | TTFT (ms) | prefill | model GiB/card | KV GiB | KV tok | conc@8k |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| int4 NONE | 1 | 23.3 | 1285 | 1593 | ~14 (whole) | -- | -- | -- |
| int4 captured | 1 | 30.5 | 1325 | 1545 | ~14 | -- | -- | -- |
| int4 NONE | 2 | 14.5 | 3033 | 675 | 8.42 | 17.98 | 444,888 | 54x |
| int4 captured | 2 | 27.5 | 3028 | 676 | 8.42 | 17.98 | 444,888 | 54x |
| w4a8 NONE | 2 | 13.1 | 2834 | 723 | 12.25 | 13.67 | 338,392 | 41x |
| w4a8 captured | 2 | 23.0 | 2841 | 721 | 12.25 | 13.67 | 338,392 | 41x |
| **w4a16 +MTP NONE** | 2 | **22.5 (~26 coh)** | 3150 | 650 | ~12 | (KV ample) | -- | -- |
| **w8a8 +MTP NONE** | 2 | **25.6 coh** | **787** | **2604** | 16.92 | 10.0 | 266,465* | 4.1x* |

KV budget ~= VRAM(~30.5 GiB/card usable at UTIL) - (model shard + ~1.7 graph). int4 shards smallest (8.4/card)
-> biggest KV (444k / 54x). *w8a8 KV measured at MAXLEN=65536; the rest at 8192.

**MTP on 4-bit + TP=2 confirmed (w4a16+MTP serve path built: `rdy_to_serve/qwen36-27b-w4a16-mtp/`):** the merged
shim (text-only arch-reg + MTP-drafter-unquant + csag-off-on-NONE) serves coherently with real acceptance
(accept_len up to 3.15). MTP nearly DOUBLED the no-push-ar TP=2 NONE decode (w4a8 NONE 13.1 -> w4a16+MTP 22.5) by
amortizing the all-reduce ~accept_len-fold -- the TP=2 MTP hypothesis. BUT MTP can't fix prefill: w4a16+MTP TP=2
prefill is 650 (no push-ar), the worst of the lot. So w4a16+MTP TP=2 is a scientific win, not a serving win.

**Weekend serve decision (all NONE, 2048 ctx) -- SHIPPED: `int4 NONE DP=2`:**
- **int4 NONE DP=2 (SHIPPED 2026-06-26)** -- the WEDGE-PROOF pick: decode 23.3, TTFT 1285, prefill 1593, 4-bit,
  two single-card TP=1 replicas (cannot BCS/DEVICE_LOST-wedge -- no cross-card collective), 2x replica aggregate,
  API-key enforced behind Traefik. Chosen for an **unattended traveling weekend**: the box cannot be rebooted
  remotely, so wedge-immunity outranks raw speed.
- **w8a8 +MTP TP=2 -- REJECTED for unattended use.** It is faster (decode 25.6, TTFT 787, prefill 2604, 8-bit)
  but it **WEDGED card 1 under load during final verification** (`gdn_attn.py spec_state_indices_tensor ->
  UR_RESULT_ERROR_DEVICE_LOST`, reboot-only recovery) EVEN WITH the GuC 70.54.0 fix. The firmware fix made the
  TP=2 wedge rare, not impossible; sustained MTP + concurrent decode can still trip it. Fine for attended use,
  NOT for a weekend you can't babysit. (JOURNAL 2026-06-26.)
- **w4a16 +MTP TP=2** -- ruled out for serving (prefill 650 / TTFT 3150 -- the fits-one-card-without-push-ar penalty).

The 35B-A3B MoE (~3B active) is fastest end to end. Caveats: (1) the **27B-W8A8 row is now the
push-ar+graph DEFAULT** -- the shelf serve.sh defaults to `PUSH_AR=1 PUSH_AR_GRAPH=1` (custom L0-IPC
push transport, P2PACCESS=0; **opt out to plain oneCCL with `PUSH_AR=0`**). It beats the oneCCL baseline
by **3.8x prefill TTFT (767 vs 2916 ms c1) and +80-126% aggregate throughput** (c4 agg 48.6 vs 26.9,
c8 73.9 vs 32.7) **plus +8-10% per-stream decode** (c1 TG 27.9 vs 25.3). With `PUSH_AR_GRAPH=1` (default)
the **DECODE** all-reduce is graph-captured too (cross-device L0-event sync recorded into torch's XPUGraph,
P2P_GPU K.6/K.8) -> every all-reduce uses the push fabric, no oneCCL fallback; `PUSH_AR_GRAPH=0` is the
older capture-gated prefill-only push (decode on oneCCL, J.17/J.21). The 35B-quark TP=2 row stays
host-staged oneCCL (push-ar is dense-only; MoE reduce_scatter/all_gather are unaffected). The other lever, `P2PACCESS=1`, is a
DEAD END -- it crashes the vLLM TP=2 serve at worker init (`UR_RESULT_ERROR_DEVICE_LOST`) and wedges
both cards, recoverable only by reboot on this display-attached box (P2P_GPU H.13/J.19/J.20; root
cause in docs/literature/p2p_access_devicelost.md). (2) the bench
uses `--dataset-name random`, which depresses MTP acceptance (random tokens are
undraftable) -- so the W8A8 27B `25.2 (MTP)` understates it; on coherent NL the
captured-MTP spec=3 ceiling is ~35 t/s @51% accept (JOURNAL 2026-06-24). (3) the
35B quark-W8A8 row is shown CAPTURED (`GRAPH=1`); the `68_shelf_bench_par.sh` sweep
ran it EAGER (serve.sh default) at 4.6 t/s, which understated it 8.7x -- PIECEWISE
capture removes the per-token op-launch overhead that dominates this ~3B-active MoE
in eager mode (Lever B, below). Sweep harness: `68_shelf_bench_par.sh` (TP=1 two-up,
TP=2 solo); the 35B-quark captured numbers are from `69_lever_tests.sh` arm B.

#### Optimization levers tested (2026-06-24, `69_lever_tests.sh`)

- **P2P in serve (A):** the oneCCL `P2PACCESS=1` path is a dead end on this cross-die
  box -- it **crashes the vLLM TP=2 serve** at worker-init and **wedges the multi-GPU
  state** (even P2P-off TP=2 then fails until an `xe` reload/reboot; reboot CONFIRMED to
  clear it 2026-06-24). Root cause: oneCCL issues a direct PCIe peer copy across our
  cross-root-complex boundary -> `xe` copy-engine reset -> Level-Zero `DEVICE_LOST`
  (docs/literature/p2p_access_devicelost.md). BUT the lever IS realized another way: the
  custom **push-allreduce** transport (L0-IPC, `P2PACCESS=0`) beats oneCCL end-to-end in a
  live TP=2 serve (+15-55% throughput / 2.3-2.5x TTFT, J.14 eager + J.17 GRAPH=1 A/B). A
  wedge guard (`bin/xpu-health`, `bin/xe-reset`, lib.sh layers) now pre-flights, tears down
  gracefully, and can auto-reset, so the TP=2 research loop is safe (P2P_GPU J.17). Never
  chain two `P2PACCESS=1` attempts without a GPU reset between them.
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
- [docs/literature/11_int4_fp4_landscape_w4a8_roadmap.md](docs/literature/11_int4_fp4_landscape_w4a8_roadmap.md):
  INT4/FP4 landscape (why FP4 is silicon-locked off B70) + the W4A8 (next target) / W4A4 roadmap.
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
