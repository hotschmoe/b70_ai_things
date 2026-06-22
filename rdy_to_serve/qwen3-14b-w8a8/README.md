# Qwen3-14B W8A8 (true INT8) -- the int8-kernel baseline

Serves `Qwen3-14B-W8A8-autoround` (compressed-tensors int8) on ONE Intel Arc Pro B70 via our custom
INT8 W8A8 oneDNN GEMM. This is the WORKING BASELINE for int8 compute on the B70 (Xe2 has no native FP8) --
the foundation for the int8 GEMM/GEMV optimization research (RESEARCH_TODO / docs/kernel/).

## Run (on the GPU host)
```bash
ssh root@192.168.10.5 && cd /mnt/vm_8tb/b70/rdy_to_serve/qwen3-14b-w8a8
/mnt/vm_8tb/b70/gpu-run bash serve.sh          # start (PIECEWISE capture), wait healthy, gen-probe
bash serve.sh stop                              # stop + release the GPU
GRAPH=1 /mnt/vm_8tb/b70/gpu-run bash serve.sh run   # serve + concurrency sweep + stop, one lease
```
Endpoint: `http://<host>:8000/v1`. Served id: `qwen3-14b-w8a8-autoround`.

## [!] Image: `vllm-xpu-env:int8g` (build: `../../images/int8g/`)
`:int8g` = `:int8` (our `contrib/vllm_int8_xpu` oneDNN INT8 W8A8 GEMM, registered as
`XPUInt8ScaledMMLinearKernel` in vLLM's `_POSSIBLE_INT8_KERNELS[XPU]`) + `register_fake` on the custom int8
ops so XPU graph capture can trace them. vLLM auto-detects the compressed-tensors W8A8 int8 scheme from the
checkpoint config -> log line `Selected XPUInt8ScaledMMLinearKernel for CompressedTensorsW8A8Int8`
(no `--quantization` flag needed). Dense 14B (Qwen3, not Qwen3.6) -> no GDN, no prepack, no vision tower.

## Notes
- W8A8 = int8 weights AND int8 (dynamic-per-token) activations -> lights the systolic int8 path. ~16 GiB.
- AutoRound vs GPTQ at W8A8 (Track 3b, evals/results/SUMMARY.md): GPTQ marginally wins (0.921/0.890 vs
  0.909/0.872 HumanEval+); the gap is ~CI-noise. int8 weights survive even XPU calibration (unlike int4).

verified: _common smoke=GREEN (eager, :int8g rebuilt 2026-06-23, 62s): HEALTHY, id ok, gen
"Paris... Washington, D.C.". Re-verify: `bin/serve-sweep --smoke`. (GRAPH=1 capture: w8a8 +16.7% historically.)
