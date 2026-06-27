# sglang on B70 XPU -- GDN NaN campaign (2026-06-27)

**Result: GO (correctness proven).** SGLang serves Qwen3.6-27B (`qwen3_5` Gated-DeltaNet) on the
dual Arc B70 (Battlemage/Xe2) **without** the vLLM-0.23 mixed prefill+decode NaN ("!!!!"). Full
config -> command -> result -> verdict in JOURNAL.md (2026-06-27 entry). Repro pass/fail defined in
`../contrib/gdn_nan_repro/README.md`.

## Why it works (vLLM didn't)
vLLM-0.23 XPU NaNs in the GDN kernels under mixed prefill+decode batching (open upstream #38994 /
vllm-xpu-kernels #172). SGLang sidesteps it architecturally: `enable_mixed_chunk` defaults **False**
(prefill and decode run as separate forward passes), SSM state is **fp32**, and the linear-attention
path uses **Triton FLA** kernels (with a real Intel-XPU port; confirmed to compute correctly on Arc).

## Build the image (GPU-free; ~50 min, mostly the SYCL AOT compile for `bmg`)
```
bash images/sglang-xpu/build.sh            # -> sglang-xpu:bmg
bash sglang/verify_image.sh cpu            # torch+xpu / qwen3_5 registry sanity (no GPU)
./bin/gpu-run bash sglang/verify_image.sh gpu   # both B70 visible to torch.xpu
```

## Serve (bf16, TP=2 == BOTH cards; cannot co-exist with the int4 DP=2 daily driver)
```
# free the cards first (stops the daily driver): docker stop vllm_daily_proxy vllm_daily_dp0 vllm_daily_dp1
# health gate before any multi-card start:
IMG=vllm-xpu-env:v0230 ./bin/gpu-run bash bin/xpu-health

# serve under a held lease (gpu-run holds it for the container lifetime via docker wait):
nohup ./bin/gpu-run bash -c 'MEMFRAC=0.93 CTX=8192 TP=2 bash sglang/serve_sglang.sh start \
  && docker wait sglang_test' > sglang/serve_lease.log 2>&1 &

bash sglang/serve_sglang.sh status      # /health + served id
bash sglang/serve_sglang.sh gen         # one coherent generation
bash sglang/serve_sglang.sh stop        # stop -> releases the lease
```
Served id: `qwen36-27b-bf16-sglang` on port 30000. SGLang ignores the request `model` field for a
single loaded model, so the gdn_nan_repro scripts (which send `qwen36-27b-int4`) run verbatim.

## Validate (the whole point)
```
cd contrib/gdn_nan_repro
python3 dd_rawtokens.py 30000 8 800           # expect: valid tokens, NO "HTTP 400 ...nan"
python3 dd_loadprobe.py 30000 8 12 500        # expect: 12/12 OK, 0 DEGEN
python3 dd_mixload.py  30000 6 6 20 3 350 1200 # sustained: expect TOTALS ... GARBAGE=0 DEGEN_EMPTY=0
python3 dd_rawtokens.py 30000 8 800           # re-confirm after load: still clean (no global poison)
```
Measured 2026-06-27: all clean (incl. post-sustained re-confirm). TTFT ~0.19 s, single-stream decode
~9.5 tok/s (bf16 TP=2), ~7 tok/s/stream under 12-concurrent. ~3x slower than the int4 daily driver
(30.8 t/s graph) -- the trade is correctness for throughput.

## Key gotchas
- `--disable-radix-cache` is REQUIRED on XPU: the "auto" mamba radix cache picks an `extra_buffer`
  strategy that asserts CUDA/MUSA/NPU and refuses to start (`extra_buffer needs CUDA/MUSA/NPU (FLA)`).
- int4 AutoRound does NOT load: `packing_format auto_round:auto_gptq` -> Marlin GEMM is CUDA-gated on
  XPU. FP8 is open-bugged for this model (sglang #23687 / #19603). bf16 is the proven precision today.
- bf16 27B = 25.6 GiB/card -> needs TP=2 (both cards). No DP=2 alongside on this box.
