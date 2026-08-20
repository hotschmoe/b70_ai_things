# Overnight postmortem -- 2026-08-20 LocalMaxxing loop

Closed: 2026-08-20 ~17:10Z. Scheduler `01a01dff8593` cancelled.
`qwen38_w8a8_dspark` stopped. Both cards free. xpu-health GO.
DD stayed PARKED. vLLM P2PACCESS stayed 0 after LOOP 4.

Window: LOOP 0 07:05Z through LOOP 22 16:39Z (~10 h, 24 ledger
blocks). Standing holds going in: Ornith GRAPH no-MTP **34.9**,
W8A8 k1bar **31.9**, GPTQ-Int4 S1 **47.58**.

## Headline numbers that survived

| hold | metric | start | end | note |
|---|---|---:|---:|---|
| Ornith NVFP4 GRAPH no-MTP | bench_code c1 | 34.9 | **34.9** | O1 rematch 34.8; LMX-shaped phase_bench **45.56** |
| W8A8 3.8 DSpark k1bar | bench_code c1 | 31.9 | **31.9** | rematch 28.8 / 28.1; soak 24.8. Not demoted |
| 3.8 GPTQ-Int4 MTP4 | post-first p512/g128 | 47.58 | **65.08** | W1 draft-INT4 overlay. Board 112.7 still ceiling |
| Pliny OBLITERATED Q8_0 | ignore-eos g128 2x | none | **32.03** | new. 1x 17.93. G1 always GO |

Do not mix those columns. 65.08 is vLLM GPTQ+MTP. 32.03 is
llama.cpp Q8_0 weight-only. 34.9 is vLLM NVFP4 MoE. 31.9 is
vLLM W8A8+DSpark.

## How Pliny Q8 went

Model: `OBLITERATUS/Qwen3.8-27B-OBLITERATED`
`Qwen3.8-27B-OBLITERATED-Q8_0.gguf` **27.7 GiB** (complete).
Card: temp 0, repeat_penalty 1.15, thinking off, empty system.
Engine: `qwen38-b70:latest` llama.cpp SYCL, Q4K doors OFF,
MMVQ pair/triple/quad ON, COMM_DIRECT=2, P2PACCESS=0.

It **ran**. First boot died on the image ENV Q4_K_M SHA pin
(cleared). After that: G1 Paris/391 every arm. 29 GB Q8_0
fits **1x B70**. 2x TP2 is the decode config.

Profile (ignore-eos g128 n=5, Paris/391):

| arm | tok/s | vs 32.03 |
|---|---:|---|
| first phase_bench (early EOS) | 30.89 | warmup 32.14 |
| Q8_DOORS=1 | 31.78 | |
| Q8_DOORS=0 | 31.56 | +0.7% for doors |
| COMM_DIRECT=2 | 31.99 | |
| COMM_DIRECT=0 | 31.83 | +0.5% for direct |
| **2x hold** | **32.03** | |
| 1x | 17.93 | 1.79x for TP2 |
| MMVQ_SG32=1 | 30.11 | -6% NO-GO |

What that means:

- Coherent uncensored 3.8 on this box at ~32 tok/s 2x, ~18 tok/s 1x.
- Q8_0 is **weight-only** (~W8A16). Not our W8A8-INT8 XMX path.
- vs 0xSero Q4_K_M lab doors **43.8**: **0.73x**. Same ratio as
  lab Q8/Q4 (36.8/49.7 = 0.74). The gap is the quant, not a
  missing DMMV (reorder MMVQ is on, PRIORITIZE_DMMV=0).
- vs W1 GPTQ+draft-INT4 **65.08**: different engine and MTP.
  Q8 has no speculative decode here.
- vs k1bar W8A8 **31.9**: similar tok/s, worse quality scheme
  for the INT8-XMX research target, better for "Pliny uncensored
  local."
- In-image fused Q8 doors and COMM_DIRECT are noise. Lab
  DP4A2 x GDN-quad SG24 (the 36.8 stack) is **not** in the
  0xSero JIT `.so` (`258f4729` vs `e75b9603`). AOT 2026.1.1
  still does not enumerate on public stacks. Closing ~13% to
  lab Q8 needs that binary, not another env door.
- Never `COMM_DIRECT_Q8=3` (lab DEVICE_LOST). Never vLLM P2P
  on this path (llama.cpp TP2 is already a device-0 USM sum).

Serve recipe kept: `llamacpp/serve_qwen38_obliterated_q8.sh`.
Sweep: `llamacpp/sweep_obliterated_q8.sh`. Weights stay
`models/files/qwen3.8-27b/obliterated-q8/` (git-ignored).

## What else we learned

**Draft-INT4 on dense GPTQ is real.** W1: 47.58 -> **65.08**
(+37%) on f01e24f6 with cookbook 2026.08.19 overlay (LM head
+ 5 MTP linears). G1 GO. Board 112.7 still out of reach
(their image/patches/acceptance). This is the overnight's
biggest number move. Do not publish 65.08 as W8A8.

**Ornith GRAPH INT4 boots and loses.** Opaque
`b70::int4_gemm_w4a16_cast` fixed L65 Half!=BF16. Wipe stale
`eagle_head` cache. MTP3 GRAPH G1 GO. Dense-only **21.7**,
routed-expert pack **21.1**, vs no-MTP GRAPH **34.9**. Slot
INT4 is a VRAM win (experts 1688->435 MB), not a decode win.
Keep GRAPH no-MTP as the 35B recipe.

**NVFP4 M=1 GEMV proto is real, e2e is not.** ESIMD 1D
`block_load` along K, AOT `intel_gpu_bmg_g31`, 0 dpas.
Isolated up **2.36x** oneDNN (cos 0.999996). Occupancy WG/SLM
NO-GO. Fused up+SiLU+down **1.04x** sequential GEMV (8.7x vs
eager oneDNN -- that is "not launching oneDNN," not a layerlet
win). Sidecar `b70_nvfp4_m1.gemv` XPUGraph PASS. e2e
M1_KERNEL GRAPH **32.2** vs hold 34.9. Default OFF.

**Methodology:** bench_code c1 34.8 recovers 34.9. Same serve,
LMX ignore-eos p512/g128 is **45.56**. Do not swap those.

**P2P on 7.1:** `can_device_access_peer` True/True. vLLM TP=2
`CCL_TOPO_P2P_ACCESS=1` still hangs 900s at warmup
(shm_broadcast), PUSH_AR on or off. After two hangs, P2P=0
TP=2 HEALTHY **147s** G1 GO -- that follow-up wedged 7.0 until
reboot. Wedge cured. P2P-in-serve still dead. Keep
P2PACCESS=0. TP=2 comm stays PUSH_AR L0-IPC.

**W8A8 k1bar did not move.** Rematch 28.8 (PREFIXCACHE=0/MRV2
on), 28.1 (pc1/MRV2 off), soak 24.8. Hold 31.9 stays. This
window was not a k1bar kernel session.

## Process

What worked: one-verdict loops, G1 before speed, leave-up
attach, dead-end packets, not demoting holds on a worse
rematch.

What hurt: 30m fires stealing cards from in-flight GPU
(LOOP 3 vs P2P, nested gpu-run gaps). Image ENV SHA pin.
Park loops 20-22 still committed attach noise. Scheduler
should have been deleted at LOOP 19 park.

## What not to do next

- Do not retry vLLM P2P=1, COMM_DIRECT=3, FATTN_MMA on JIT,
  emul NVFP4 G1, Python sticky/M1, M1_KERNEL as default,
  SG32=1, demote 31.9 on a 28.x rematch.
- Do not treat 45.56, 65.08, 32.03, 34.9 as one leaderboard.
- Do not fetch the whole OBLITERATUS repo (mlx + every quant).

## Optional next (not started)

1. Keep Pliny Q8_0 2x as the uncensored 3.8 llama.cpp recipe
   at 32 tok/s. Quality eval (HE+) if we want it as a daily
   talker. Lab DP4A2xSG24 only if a BMG JIT/AOT binary exists.
2. W1 65.08 vs board 112.7 is a vLLM 0.27 patch/acceptance
   chase, not a Q8 problem.
3. Ornith stays GRAPH no-MTP 34.9 until a grouped NVFP4 GEMM
   (not M=1 GEMV) beats it e2e.
4. W8A8 31.9 remains the INT8-XMX research hold.
