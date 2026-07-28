# qwen36-27b-w8a8-mtp -- W8A8 (int8) + NEXTN MTP, the int8 all-rounder

The W8A8 path that **handily beats bf16/fp8 on prefill, TTFT, AND decode**, with vision retained and
higher code accuracy than int4. Built on our fused int8 oneDNN kernels + NEXTN chain-MTP.

## Numbers (IN2048/OUT128, warm c1, TP=2 == both cards)

| metric | this (W8A8 fused+MTP) | bf16 TP=2 | vs bf16 |
|---|---|---|---|
| decode (TG) | **25.2 t/s** | 9.03 | **+180% (2.8x)** |
| prefill (PP) | **4344 tok/s** | 3098 | **+40%** |
| TTFT | **471 ms** | 661 | **-29%** |

- DECODE (25.2) is rock-solid run-to-run. PREFILL/TTFT vary more under MTP (the spec draft adds prefill-side
  work): warm c1 seen ~471-691 ms TTFT / ~2960-4344 PP. The table shows the warm-best (run2); for a CONSISTENT
  prefill/TTFT champion (and sampling) use the eager sibling (PP 4570 / TTFT 448, `scripts/123`).
- Also beats the int4+MTP driver on decode (25.2 vs 15.3) -- the MTP verify (M>1) rides the int8-XMX
  `int8_gemm_w8a8` (2.0x bf16) instead of int4 woqgemm.
- FP8 has no native B70 path (oneDNN emulates `fp8_gemm_w8a16` at ~1.0x bf16 prefill) -> W8A8 wins PP vs fp8 too.
- **Accuracy: HumanEval+ 0.970 / 0.933 (base/plus)**, sandboxed -- HIGHER than int4 same-stack (0.933/0.896).
  int8 weights are more accurate than int4; the fused kernels add zero loss; MTP is greedy-lossless.
  Result: `../../evals/results/20260628T233713Z__qwen36-27b-w8a8-vision-mtp__w8a8-fused-vision`.

## Large-prefill qualification (2026-07-28)

The shelf now uses the custom Level Zero IPC push all-reduce only at
`PUSH_AR_MIN_NUMEL=1048576`. A matched shelf-wrapper run in the same thermal/session state measured:

| metric | push off | push on, 1M gate | delta |
|---|---:|---:|---:|
| c1 decode | 21.39 t/s | 21.12 t/s | -1.3% |
| c1 TTFT | 1,725 ms | 582 ms | 2.96x faster |
| c4 aggregate decode | 13.85 t/s | 19.92 t/s | +43.8% |
| c4 TTFT | 4,310 ms | 2,373 ms | 1.82x faster |
| 2K-token soak | 16.25 t/s | 16.16 t/s | -0.6% |

Both arms were coherent and the soak stayed stable. These absolute decode values were collected
after hours of continuous GPU work and are lower than the cooler historical shelf numbers, so use
this table only as the matched on/off comparison. The separate 0.5.6 prefill A/B measured 2.09x to
3.12x c1 cold-prefill gains from 512 through 32K tokens and 3.12x to 3.17x at c4.

The version-compatible patch also covers sglang 0.5.15's newer in-place collective route. In the
matched one-request 200K/BF16-KV A/B, exact 190,048-token retrieval remained correct and cold wall
time fell from 525.00s to 333.73s (1.57x); warm reuse stayed about 3.5s with 99.93% cache hit.
The final 1M gate retained 1,735 tok/s at 2K cold prefill and 1,555 tok/s at 32K, passed the
qualifier, and left both cards healthy.

## How it works

- **Decode (M==1):** `int8_gemm_w8a16` -- s8 weight x fp16 act, per-channel dequant fused in the oneDNN
  epilogue (1 launch). At M=1 the GEMV is weight-BW-bound so int8 activations buy nothing; fp16-act is leaner.
- **Prefill / MTP-verify (M>1):** `int8_gemm_w8a8` -- s8 x s8 on the XMX/DPAS systolic array (2.0x bf16),
  with `dynamic_per_token_int8_quant` (fused single-launch act-quant).
- **Large TP prefill all-reduces:** the custom Level Zero IPC push transport is enabled only for tensors
  with at least 1,048,576 elements. This keeps batched MTP verify/decode on oneCCL while routing 512+
  token prefills to push. During the discovery A/B at the lower 65,536-element diagnostic threshold, it
  raised unique cold prefill 2.68x at 2K, 3.09x at 8K, and 2.81x at c4 2K on the confirmation run;
  code c1 stayed neutral. The 1,048,576 production cutoff is deliberately conservative: only large
  EXTEND tensors use push, while decode and batched MTP verification remain on the proven oneCCL route.
- **MTP:** NEXTN chain spec-decode, steps=10 (W8A8 peak -- the cheap int8-XMX verify rewards deeper drafts
  than int4's steps=7: 7->23.8, 10->25.25, 12->24.35). Greedy-only on XPU.
- Both ops built from vllm-xpu-kernels source vs sglang torch 2.12 (`../../../research/w8a8/W8A8_BUILD.md`).

## Dependencies (runtime mounts, NOT a baked image)

- image `sglang-xpu:mtp` (baked XPU NEXTN gates + compressed_tensors W8A8 scheme)
- built kernel `.so` at `/mnt/vm_8tb/b70/w8a8_kernel/_xpu_C.abi3.so` (sha bc643c3f8a61; build: W8A8_BUILD.md)
- the FUSED `w8a8_shim.py` (`../../sglang/patches/w8a8_shim.py`, `B70_XPU_W8A8_FUSED=1`)
- custom push-allreduce `.so` under `vllm/contrib/vllm_push_allreduce/prebuilt/`, with canonical
  `woq_shim.py` and `push_ar_xpu.py` runtime mounts
- materialized checkpoint `models/files/qwen3.6-27b/w8a8-sqgptq` (vision + W8A8 language model +
  BF16 MTP head), mounted as `/models/qwen3.6-27b/w8a8-sqgptq`

## Use

```
/mnt/vm_8tb/b70/gpu-run bash serve.sh start    # serve TP=2 (both cards), coherence-gated, stay up
bash serve.sh gen "your prompt"                # quick greedy chat probe
bash serve.sh stop                             # stop + release + health check
/mnt/vm_8tb/b70/gpu-run bash serve.sh run      # start + warm c1 bench + stop in one lease
```

- TP=2 holds BOTH cards. cudagraph is DISABLED (W8A8 TP=2+MTP is stable that way; XPUGraph capture is a
  CEILING at TP=2 -- decode is all-reduce-bound, not launch-bound).
- For **prefill-heavy or sampling** loads, use the eager sibling (no MTP, samples):
  `../../scripts/123_w8a8_fused_ab.sh` (FUSED=1 GRAPH=0) -> PP 4570 / TTFT 448 / decode 8.1.
- Greedy-only: MTP verify runs greedily on XPU (temperature/top_p/top_k ignored), like all XPU NEXTN.

## Agentic / daily-driver settings (pi.dev / omp.sh / hermes)

The daily driver runs this entry at its agentic config. Knobs (env, defaults in serve.sh):

- **`CTX`/`MAXLEN` -> 128K.** `CTX="${CTX:-${MAXLEN:-8192}}"`: the backend-agnostic `MAXLEN` knob is honored, so
  `daily_driver_serve.sh` (DD_MAXLEN=131072) serves the full 128K. Bare shelf use still defaults to 8192.
  KV is bf16 (fp8 KV is NOT supported on the XPU attention backend) and CHEAP -- this is a hybrid model, only
  16/64 layers are full-attention -> ~64 KB/token. The KV pool holds ~182k tokens: a full 128K session fits,
  and two concurrent sessions share the pool (combined < 182k; rare both-maxed -> graceful preempt).
- **`TOOLCALL=1` / `TOOLPARSER=qwen3_coder`** -- Qwen3.6 emits XML `<tool_call>` (NOT hermes JSON); returns
  structured OpenAI `tool_calls`. **`REASONPARSER=qwen3`** splits `<think>` into `reasoning_content`.
- **`THINKCAP=4096`** -> `SGLANG_MAX_THINK_TOKENS` (graceful `</think>` cap). `THINKCAP=` for unlimited.
  Lowered from 8192 on 2026-06-29: caps the worst-case thinking dead-air (~3min at 25t/s) before the first
  tool-call token, which a fronting reverse-proxy idle timeout can cut on long agentic tool calls.
- **`RADIX=0` by default; `RADIX=1` is now a qualified option.** The mounted XPU patch un-gates the
  `extra_buffer` strategy with INT8 mamba checkpoints and page size 128 while retaining Intel XPU
  attention. Repeated 12K prompts measured 4.35x reuse, and the 0.5.15 one-request 200K mode passed
  exact 190,048-token cold/warm retrieval with 99.93% cache hit. Use the 0.5.15 research serve for
  the measured 200K/BF16-KV mode: `MAXREQ=1 MAMBA_CACHE=4 sglang/serve_w8a8_0515.sh start`.
- **`METRICS=1` (Prometheus `/metrics` ON).** Adds `--enable-metrics`; exposes input/output token counters,
  TTFT (prefill), gen throughput (decode), `cache_hit_rate`, and queue depth on the serve port (same `:$PORT`,
  NOT api-key-gated). Dashboard = sglang `examples/monitoring/` Prometheus+Grafana (Grafana on a port other
  than :3000; the WebUI owns :3000). `cache_hit_rate` reads ~0 while `RADIX=0`. `METRICS=0` to disable.
- Concurrency stays `--max-running-requests 4` (mamba/spec cache bound; covers the c<=4 daily-driver load).

Campaign: `../../../research/w8a8/W8A8_SGLANG_PLAN.md`. JOURNAL 2026-06-28/29.
