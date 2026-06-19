# W4A8-INT8 for single-card B70 -- dedicated workstream

Goal: a **W4A8-INT8** Qwen3.6-27B (and Gemma-4-31B) worth shipping for single-B70 users via
the int8-XMX datapath (`XPUW4A8IntLinearKernel`, oneDNN `int4_gemm_w4a8`). 14B is the test bed.

Status 2026-06-20: **B70 free.** Corrected after reading the background eval commits (see below) --
the original premise of this doc (packing = the single-card fit gate) was WRONG. Decisions:
recoverability = **AutoRound**; but first confirm w4a8 has a real niche (see "Strategic reality").

---

## [!] Corrected facts (verified from serve logs, commit 0f4e7ee)

- **w4a8 VRAM = 9.3 GiB** ("Model loading took 9.3 GiB"; Available KV 15.6 GiB corroborates).
  The int4 weights are stored UNPACKED on disk (16 GB safetensors, ~1 byte/int4-weight), but
  **vLLM repacks to 4-bit ON LOAD** -> 9.3 GiB resident (also the slowest load, 39 s vs ~23 s).
- So **fit is NOT the blocker.** w4a8 already fits one card with room (9.3 GiB, same as w4a16).
  A 27B W4A8 would be ~18 GiB VRAM -> fits 1x (cf. 27B int4-AutoRound = 17.6 GiB, already proven).
- **Packing (the on-disk 16 GB) only costs DISK + LOAD TIME, not VRAM.** It's a distribution/UX
  nicety (smaller download, faster load), NOT a fit unlock. De-prioritized accordingly.
- What actually makes w4a8 lose today: **decode kernel + accuracy**, not memory.

## The bar to beat: packed W4A16-gptq (single-B70 leaderboard, 2026-06-20)

| 14B quant (calib) | HumanEval+ b / + | decode t/s | prefill t/s | VRAM | activations |
|---|---|---|---|---|---|
| fp8 (online) | 0.915 / 0.890 | **32.1** | 3525 | ~15 GB | fp8 |
| w8a8 (gptq) | 0.921 / 0.890 | 23.5 | **5740** | ~15 GB | int8 dyn (int8-XMX) |
| **w4a16 (gptq)** | **0.872 / 0.848** | **29.0** | 2920 | **9.3 GB** | fp16 (weight-only) |
| **w4a8 (rtn) <- ours** | **0.860 / 0.817** | **16.5** | 4403 | 9.3 VRAM / 16 disk | int8 dyn (int8-XMX) |

Repo verdict: **"w4a8 is dominated -- skip for coding."** w4a16-gptq beats it on accuracy
(0.848 vs 0.817), decode (29 vs 16.5), and disk (9.3 vs 16) at the **same 9.3 GiB VRAM**.

## Strategic reality -- does w4a8 even have a niche?

Since w4a16-gptq already gives 0.848 / 29 t/s / 9.3 GiB on one card, w4a8's ONLY possible edge
is **int8 activations -> int8-XMX throughput under concurrency** (multi-user single-card serving),
where w4a16's fp16 activations can't use the systolic int8 path. But:
- w4a8 prefill is 4403 t/s vs w8a8's 5740 -- the int4-unpack overhead already hampers its int8-XMX
  even in the compute-bound regime. So the concurrency win is NOT guaranteed.
- Its only structural advantage over w8a8 (also int8-XMX) is lower VRAM (9.3 vs ~15 GiB).

=> **Before investing in AutoRound, MEASURE the concurrency niche** (w4a8 vs w4a16 vs w8a8
aggregate tok/s at C1/8/16/32/64). If w4a8 doesn't win throughput anywhere, the whole scheme is
dominated and we pivot to w4a16-gptq for single-card. If it does, accuracy + kernel become worth it.

---

## The three wins (re-ranked by what actually matters now)

### Win A -- KERNEL (decode) [the real single-stream gap, hardest]
`int4_gemm_w4a8` decode = 16.5 t/s vs w4a16's 29 at the SAME 9.3 GiB VRAM -> purely kernel-bound
(per-token int8 act-quant overhead + unoptimized oneDNN int4 GEMV/decode path; the int4-unpack
also caps prefill at 4403 < w8a8's 5740). Profile vs `int8_gemm_w8a8`/`wNa16` in
`contrib/vllm_int8_xpu` / `vllm-xpu-kernels`; fuse the act-quant; optimize the int4 decode path.
Success: decode -> ~29 t/s AND/OR a clear concurrency-throughput win over w8a8/w4a16.

### Win B -- ACCURACY (recoverability) [chosen: AutoRound]
Current w4a8 is data-free RTN -> 0.817 plus (lowest). AutoRound (int4 weights) + int8 dynamic
activations. Target: **>= 0.848 plus** (beat w4a16-gptq) on HumanEval+ Tier-1. Note: AutoRound
fixes accuracy ONLY -- it does nothing for Win A. Script `10_quant_autoround_w4a8.sh DEVICE=xpu`
(B70-accelerated; retest the old "XPU calibration unreliable" caveat, which was for SmoothQuant).

### Win C -- PACKING (disk/load) [nicety, low priority]
Re-export pack-quantized so the on-disk drops 16 GB -> ~9 GB and load 39 s -> ~23 s. Does NOT
change VRAM/fit. AutoRound's compressed-tensors export may pack for free. Probe:
`11_test_packed_export.sh` (CPU). Verify XPUW4A8IntLinearKernel still loads the packed layout.

---

## Execution order (B70 now free; route GPU runs through `scripts/gpu-run`)
0. Tear down the idle `vllm_qwen3` (w4a16-gptq, idle since 06-19 21:03) to free the B70.
1. **[B70] Concurrency-niche check FIRST** (decision-gating, cheap): serve the existing
   w4a8-RTN, w4a16-gptq, w8a8-gptq; sweep C1/8/16/32/64 aggregate tok/s. Does w4a8 win anywhere?
2. If NO -> stop; w4a8 is dominated, pivot single-card to w4a16-gptq. Record and close.
3. If YES -> [B70] AutoRound smoke (`ITERS=50`) -> validate toolchain/flags + export; then full
   run -> serve + HumanEval+ Tier-1 -> compare to the 0.848 bar. (Win C packing probe in parallel.)
4. Then Win A (kernel) -- the longer effort -- if the niche justifies it.
5. Only if 14B clears the bar: replicate on Qwen3.6-27B + Gemma-4-31B. Note the 27B already hits
   the `XPUwNa16` ÷32-dim issue (4304 dim) for w4a16; the w4a8 path may need similar pad/ignore.

## Experiment log (newest at bottom; config -> command -> result -> verdict)
- 2026-06-20 -- workstream opened. Found `Qwen3-14B-W4A8-INT` = 16 GB on disk (unpacked I8).
- 2026-06-20 -- **CORRECTED** (commit 0f4e7ee): VRAM is **9.3 GiB** (vLLM repacks on load), so fit
  is fine; the 16 GB is disk-only. w4a8 is dominated by w4a16-gptq on accuracy+decode+disk at equal
  VRAM. Re-ranked wins: kernel + accuracy are the levers; packing is a nicety. Added a
  concurrency-niche check as the decision gate before spending GPU on AutoRound.
