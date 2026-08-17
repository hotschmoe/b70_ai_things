# rmacy Qwen3.8-27B FP8 + DSpark on dual B70

Research notes for `ghcr.io/rmacy/qwen38-fp8-dspark` (latest `v10-slim`).
This is NOT a shelf entry. Serve via `vllm/dflash/serve_qwen38_fp8_dspark.sh`.

## What it is

DSpark (DeepSeek) is DFlash-style block-diffusion drafting plus a confidence /
Markov head that picks how many draft tokens to emit. The published B70 recipe
is:

- Target: official `Qwen/Qwen3.8-27B-FP8` (block FP8 e4m3, [128,128])
- Drafter: `rwmacy/qwen3.8-27b-dflash-drafter-fp8-b70` (1.36B, BF16 weights,
  trained on 2x B70 with SpecForge XPU, warm-start from RadixArk DSpark)
- Engine: Intel llm-scaler XPU vLLM (`qwen36-b70-vllm:b3-maxperf-final-v7`
  plus patched dflash files). Published as `ghcr.io/rmacy/qwen38-fp8-dspark`.
- Spec: `--speculative-config method=dflash, k=4`
- Isolated C1 claim: **72.2 tok/s median / 85.9 peak** vs 32.4 no-spec and
  54.67 MTP2. Acceptance 62-74% pos-0, mean accept length 2.5-3.5.

Sources:

- https://github.com/users/rmacy/packages/container/package/qwen38-fp8-dspark
- https://github.com/intel/llm-scaler/pull/620
- https://github.com/sgl-project/SpecForge/pull/769
- https://github.com/rmacy/vllm (kernel fix, based on vllm-project/vllm)

## The kernel fix (required)

SpecForge DSpark trains output j to predict token `anchor+j+1` (LM-style).
Stock vLLM dflash sampled offsets 1..k (BERT-style), so every draft was off
by one and SpecForge drafters capped at ~24% acceptance.

rmacy/vllm commits (2026-08-16):

- `72f353c` fix(dflash): correct SpecForge DSpark draft-token readout offset
- `d7c7222` fix(dflash): sample all k draft slots at offsets 0..k-1
- `77242f1` fix(worker): handle list-type draft_token_ids in async scatter

Three-line core:

- `dflash.py`: `num_query_per_req = k` (was `k+1`)
- `utils.py`: `is_sample = is_query` (was `is_query & (query_off > 0)`)
- `utils.py`: `sample_out_idx = req*k + off` (was `req*k + off - 1`)

Effect on released RadixArk drafter: 24% -> 66% pos-0. Their fine-tune is
the 62-74% number above.

## Training (SpecForge#769)

XPU-specific constraints they hit:

- `num_anchors <= 64`, `objective_chunk_blocks <= 16` (else silent OOM -> NaN)
- `VLLM_XPU_ENABLE_XPU_GRAPH=0` (graphs hang xe engines)
- `dataloader_num_workers=0`
- eager attention (flex_attention is CUDA-only)
- CPU optimizer offload, no pin_memory
- 440 steps, lr 5e-5, 761 ShareGPT + 200 C1-domain samples

## How this differs from our prior DFlash spike

`vllm/DFLASH_XPU.md` (2026-07-03) used `z-lab/Qwen3.6-27B-DFlash` against
our W8A8 3.6 target on v0.24/v0.26. That path is coherent. This stack is
a newer vLLM (0.21.1.dev0 llm-scaler lineage, despite the version string),
a 3.8 FP8 target, a B70-tuned DSpark drafter, and the readout fix.

Their published serve is **not** a daily-driver config: maxlen 8192,
max-num-seqs 1, graphs off, isolated C1. Our DD is 262k + concurrent
prefill+decode. Compare numbers with that caveat.

## P2P

Their `serve.sh` sets `CCL_TOPO_P2P_ACCESS=1`. That is the documented
wedge trigger on THIS box for vLLM TP>1 (P2P_GPU.md H.13). First try
here is P2P=0. Do not flip it without `I_KNOW_P2P_WEDGES=1` and a reboot
plan.

## oneCCL 2021.15 is broken on this box

v10-slim ships oneCCL Gold-2021.15.9. TP=2 dies at the
xpu_worker warmup all_reduce with
`ze_handle_manager mem_to_ipc_handle: device_fd is invalid`
for drmfd (their recipe) AND pidfd. Same bug we patched in
the v0.25.1 int8g bake. The serve script bind-mounts
`/mnt/vm_8tb/b70/ccl_2021.17/2021.17` over
`/opt/intel/oneapi/ccl/2021.15` and sets CCL_ROOT +
CCL_CONFIGURATION=cpu_gpu_dpcpp.

## First-try command

```
./bin/gpu-run bash vllm/dflash/serve_qwen38_fp8_dspark.sh start
```

## First-try numbers (2026-08-17e, P2P=0, 2021.17 swap)

Served `qwen3.8-27b-fp8-dspark` on :8078. Paris exact. fib
coherent. KV 135,735 tok advertised (maxlen 8192).

| workload | result |
|---|---|
| isolated C1 greeting n~10 wall | median 20.58 tok/s |
| code c1 out=256 wall | **20.0 tok/s** |
| streaming LRU thinking-off out=128 wall | 22.3-22.6 tok/s |
| bench_2048 thinking-on IN~2080 | TTFT 2.9s, PP ~700, TG 8.4 (chunk-count lies) |
| code accept | pos0 63.2%, accept_len 2.42, rate 0.354 |

Acceptance matches their 62-74% / 2.5-3.5. Speed does not
(claimed 72.2). Not a DD. Details: JOURNAL 2026-08-17e.

## Long-decode collapse (2026-08-17f)

Author-side claim that this path "eventually collapses in
decode" was tested on the live :8078 serve
(`vllm/dflash/probe_decode_collapse.py`):

- 2 x 2048 thinking-off: 16.8 tok/s, bangs=0, NO_COLLAPSE
- 1024 thinking-on: 15.6 tok/s, bangs=0, NO_COLLAPSE
- 4096 thinking-off: 18.2 tok/s, bangs=0, NO_COLLAPSE
  (last window still coherent; speed flat/slightly up)

Independent check: 0xSero/qwen38-b70 lists vLLM XPU 0.27.2
FP8 at 21.7 tok/s and llm-scaler 0.21 as crashing. That
21.7 matches our first try, not the 72.2 isolated-C1
claim. See `llamacpp/QWEN38_B70_0XSERO.md`.
