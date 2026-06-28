# sglang on B70 XPU -- Qwen3.6-27B daily driver (2026-06-27)

**Result: GO (correctness proven).** SGLang serves Qwen3.6-27B (`qwen3_5` Gated-DeltaNet) on the
dual Arc B70 (Battlemage/Xe2) **without** the vLLM-0.23 mixed prefill+decode NaN ("!!!!"). Full
config -> command -> result -> verdict in JOURNAL.md + the perf campaign in `sglang/PERF.md`.

## DAILY DRIVER GUIDE (perf-campaign outcome, UPDATED 2026-06-28) -- see sglang/PERF.md + JOURNAL
The campaign goal was a more performant AND correct daily driver. **ACHIEVED + EXCEEDED:** the ~9.4 t/s eager
ceiling is BROKEN. Headline = **25.15 t/s single-stream** (W4A8/W4A16 hybrid + XPUGraph), via the FIRST
sglang-XPU decode cuda-graph (torch.xpu.XPUGraph / SYCL-Graph over Level-Zero) stacked on the oneDNN int4_gemm
ops. All drivers correct + vision-retaining; pick by use:
  1. **W4A8/W4A16 hybrid + XPUGraph (FASTEST single-stream, SAMPLING) -- the recommended single-user driver:**
     `rdy_to_serve/qwen36-27b-w4a8-graph/serve.sh` -> **25.15 t/s** warm / 24.8 soak (> the int4-woqgemm 23.5),
     **lower TTFT (~970-1110 ms)**, sampling-capable, vision, SAME int4 weights. Decode = int4_gemm_w4a16 (fp16
     act, captured); prefill = int4_gemm_w4a8 (int8 act). **ACCURACY-GATED: HumanEval+ == int4 (0.921/0.896,
     delta 0.000 -- int8-act prefill = zero code loss; see evals/results/SUMMARY.md + W4A8_PLAN.md).** Runtime
     mounts (NOT baked): the built `_xpu_C.abi3.so` + woq_shim.py + oneAPI LD_LIBRARY_PATH (see W4A8_BUILD.md).
     Pin card 0 (card 1 downclocked -> ~15). 2 users: `./sglang/serve_dp2_w4a8.sh` (~25+15, wedge-proof).
  2. **int4 + XPUGraph (prior champion, baked image, SAMPLING):**
     `rdy_to_serve/qwen36-27b-int4-graph/serve.sh` -> **23.5 t/s** = 2.5x eager, vision, ZERO mounts (simplest).
     2 users: `./sglang/serve_dp2_graph.sh` (~23.5+15.3). Use this if you want a no-mounts baked image.
  3. **int4 + NEXTN MTP (greedy latency):**  `rdy_to_serve/qwen36-27b-int4-mtp/serve.sh` -> 15.3 t/s (1.62x),
     greedy-only, vision. Superseded by the graph drivers for single-user (graph is faster AND samples).
  4. **woq int4 DP=2 (>2 users / unattended, wedge-proof):**  `./sglang/serve_dp2.sh` -> ~9.4/replica, sampling.
  5. **bf16 TP=2 (best c4 aggregate, attended):**  the serve command below. ~9.2 c1 / 23.4 c4-aggregate.
LEVERS (sglang/PERF.md + JOURNAL have the data): **XPUGraph decode capture = the WIN** -- torch.xpu.XPUGraph is
STABLE on B70 (the old "torch-xpu graph degrades" was a different/older mechanism); wired into sglang via
`patches/xpu_cudagraph.py` (B70_XPU_CUDAGRAPH=1, needs ATTN=triton to clear the SYCL-Graph work_group_scratch
wall). MTP/NEXTN done (15.3, greedy). DEAD-ENDS (don't re-try): torch.compile (no-op w/o cuda-graph), cheap
scheduler flags, per-card MAXREQ>4 (spec mamba cache), multi-bucket graph (halves single-stream), graph+MTP
stack (spec-decode crashes under capture -- walled, scripts/143). TP=2/PP=2 hang; GDN num_warps = cold-bench artifact.

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
