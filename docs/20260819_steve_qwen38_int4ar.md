# Steve lab Qwen3.8 INT4-AR 100+ tok/s -- loop digest

**Ingested:** 2026-08-19 (operator; X post + commits minutes old).
**Repo:** https://github.com/steveseguin/b70-optimization-lab
**HEAD at ingest:** `924b518` (2026-08-19T03:04Z)

Operator 2026-08-19 YOLO: S2 may use Steve's stack, his
graph-safe FA source, public nightly `f01e24f6` (already on
disk), or a NEW vLLM/sglang image. Do not stay on int8g-v0260
out of habit. Weights are ON DISK. Next pick is S2b.

## What he published

2x B70, vLLM/XPU, **AutoRound INT4 W4A16**, speculative MTP.

| row | metric | LocalMaxxing |
|---|---|---|
| MTP5 | **101.922** tok/s all-25 (median of 100.896 / 102.042 / 101.922) | `cmszbkxco0e11ms01l2rixxbt` |
| MTP4 | **100.497** tok/s all-25; **better** on the 12 historical prompts (96.627 vs MTP5 95.167) | `cmszarna10e0nms0103hv0tve` |

Metric is his conventional after-TTFT 99-interval rate on a **25-prompt**
cold suite, cache-zero, 25/25 token-identical under a **pinned
torch.compile cache**. Selection-12 has not crossed 100.

Depth sweep: MTP3 96.6 / MTP4 100.5 / MTP5 101.9 / MTP6 99.5. Config
space on flags is closed (lean-flag retest 9f90e2c: dropping oneDNN
barriers is slower and less deterministic). Next 105 needs a **code**
change: persistent GDN scratch zero-init (`vllm-xpu-kernels`
`fix/gdn-scratch-zero-init` 0ab8205) -- not built/measured on his
15 GiB second host.

Repro: `repro/qwen38-27b-autoround-int4-b70/README.md`
Diagnosis: `notes/2026-08-18-qwen38-int4-100tps-uninitialized-gdn-scratch.md`

## What we have not done

We have **not** served Qwen3.8 AutoRound INT4. On this box:

- 3.6 AutoRound W4A16 is a shelf / proven W4A16 baseline
- 3.8 Q4_K_M llama.cpp = 43.8 c1 (lab doors) / HE+ 0.970/0.927
- 3.8 GPTQ-Int4 (SergiioB) 1x B70 S1 = 47.58 post-first vs their 83.7
- 3.8 W8A8-gptq + DSpark k=4 GRAPH=1 = **29.4** `bench_code` c1

Steve's 101.9 is a different scheme, harness, and compile-cache
identity. Do not photocopy it onto W8A8.

He also notes SergiioB GPTQ-Int4 failed a code canary (30 vs 14) so
he does not promote that checkpoint. Our S1 was a speed smoke, not
that canary.

## S2 protocol (operator 2026-08-19)

Checkpoint: `devan-carlin/Qwen3.8-27B-int4-AutoRound`
rev `bce40cacab0a4535b92fb3d57615c2bea9adf3d1` (~19.02 GB).
Dir: `models/files/qwen3.8-27b/int4-autoround`.
Download log/pid: `/mnt/vm_8tb/b70/qwen38-w8a8-dspark/s2_hf_download.{log,pid}`.

Served ids:
- `qwen3.8-27b-W4A16-autoround-mtp5` (speed)
- same id for HE+ (do not score as W8A8)

**S2a DONE 2026-08-19** -- 19016936446 bytes, `quant_method: auto-round`,
bits 4 G128. Log showed `Downloaded`.

**S2b SPEED (YOLO)** -- stop W8A8 AGASYNC. Serve with
`vllm/w4a16/serve_qwen38_27b_int4ar.sh` (default IMG =
`vllm/vllm-openai-xpu@sha256:f01e24f6...` 0.27.2rc1, TP=2
MTPTOK=5 GRAPH=1 DTYPE=float16 MAXLEN=16384). Allowed:
Steve lab at `/mnt/vm_8tb/b70/b70-optimization-lab-main`
(`924b518`), his `experiments/qwen27_graphsafe_flash_attention`
build, overlay `XPU_C_SO`/`GDN_LIB`, or build a new
vLLM/sglang image. G1 first. Then `bench_code` c1 AND
`phase_bench` after-TTFT. Chase 101.922. If G1 dies, try
the next stack (0.27 -> Steve kernels -> new image), then
packet only if all three fail. Do not quietly fall back
to int8g-v0260.

S2b status LOOP 27+36-39: SYCL-9 nightlies D10/D11.
intel/vllm 0.21 TP=2 loads. D13 Python fallback G1 fib
bangs. 44fc8fde0 overlay enables FORCE_GRAPH PIECEWISE
then dies: image _xpu_C int4_gemm 7-arg vs 8-arg
input_dependency. Kernel rebuild loop39_kbuild RUNNING.
Gated cell remains f01e24f6 TP=1 GRAPH=0 c1 12.8 / 16.66.

**S2c quality** -- HE+ 164 thinking-off greedy seed=1234 on the
same served id. Compare to W8A8 **0.957 / 0.927** and Q4_K_M
**0.970 / 0.927**. Fail lists matter (base misses, not just plus).
Steve has no 3.8 INT4-AR HE+ yet -- this is new. Then restore
W8A8 AGASYNC unless Next pick still needs INT4.

Do not overwrite w8a8-gptq. Do not start DD. One fire = one arm.

## Steal later (not this fire)

- Pinned compile-cache as part of run identity (LOOP 26 landed
  SPECTOK+SO on the 0.26 DSpark path; b3f7e9e010 was the hole)
- oneDNN barriers-on as a decode lever
- GDN scratch zero-init if we ever hit residue / nondeterminism
- After-TTFT 25-prompt suite if we want a number he will accept
- INT4-AR 3.8 A/B (S2) only after W8A8 P4.1 and the living-header
  speed window say so. Calibrate CPU/CUDA never XPU.

Do not enter Phase 2 just to chase 101.9.
