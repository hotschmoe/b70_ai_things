# sglang on B70 XPU -- Qwen3.6-27B daily driver (2026-06-27)

**Result: GO (correctness proven).** SGLang serves Qwen3.6-27B (`qwen3_5` Gated-DeltaNet) on the
dual Arc B70 (Battlemage/Xe2) **without** the vLLM-0.23 mixed prefill+decode NaN ("!!!!"). Full
config -> command -> result -> verdict in JOURNAL.md + the perf campaign in `sglang/PERF.md`.

## DAILY DRIVER GUIDE (perf-campaign outcome) -- see sglang/PERF.md for the full scoreboard
The campaign goal was a more performant AND correct daily driver. Outcome: CORRECTNESS achieved; STABLE
single-stream decode sits at ~9.2-9.4 t/s warm (the sglang-XPU eager ceiling). Two verified-correct,
vision-retaining drivers (pick by use):
  1. **woq int4 DP=2 (UNATTENDED / wedge-proof):**  `./sglang/serve_dp2.sh start`  -> :18080
     Two single-card int4 replicas (sglang-xpu:woq, auto_round_kernel.woqgemm) + nginx round-robin.
     ~9.4 t/s/replica, vision, int4 = big KV, NO cross-card collective (cannot BCS-wedge). VERIFIED clean
     under the agentic mixed-load that makes vLLM emit "!!!!".
  2. **bf16 TP=2 (ATTENDED / best aggregate):**  the serve command below.  ~9.2 c1 / 23.4 c4-aggregate.
LEVERS EXPLORED (sglang/PERF.md has the data): quant-for-speed (woqgemm wired in, but 4-bit doesn't beat
bf16 XMX warm without a graph), TP=2/PP=2 (PP & woq-TP=2 hang), GDN num_warps (cold-bench artifact),
cudagraph (WIRED + runs -- a first -- but the torch-xpu graph-replay accumulation degrades it; not stable),
MTP (works up to 2 unimplemented XPU tree kernels -> the documented NEXT lever, est. ~15 t/s STABLE if finished).

## GDN-NaN correctness campaign (the foundation)
Repro pass/fail defined in `../contrib/gdn_nan_repro/README.md`.

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
- int4 AutoRound: Marlin GEMM is CUDA-gated on XPU, BUT we wired `auto_round_kernel.woqgemm` (auto-round-lib)
  into sglang via `sglang/patches/woq_shim.py` (image `sglang-xpu:woq`) -> int4 serves single-card, coherent,
  vision-retaining (see serve_dp2.sh + PERF.md). FP8 is open-bugged for this model (sglang #23687 / #19603).
- bf16 27B = ~55 GiB -> needs TP=2 (both cards). int4 (~18 GiB) fits one card -> enables DP=2 (serve_dp2.sh).
- XPU cudagraph: torch.xpu.XPUGraph capture WORKS on this hybrid GDN model (wired via woq_shim B70_XPU_CUDAGRAPH=1
  + a model_runner xpu patch -- an engineering first), but the torch-xpu graph-replay command-stream accumulation
  degrades it (same dead-end as vLLM PIECEWISE); OFF by default until an upstream torch-xpu fix lands.
