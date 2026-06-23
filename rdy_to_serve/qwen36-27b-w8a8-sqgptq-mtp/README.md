# qwen36-27b-w8a8-sqgptq-mtp

Qwen3.6-27B **W8A8** (compressed-tensors INT8 weights x INT8 activations, SmoothQuant+GPTQ) + a **BF16 MTP
graft**, served **TP=2 + MTP spec=5** on 2x B70. This is the **fastest single-stream 27B config on the rig.**

## Result (config -> command -> result -> verdict)

- **Config:** `vllm-xpu-env:int8g`, TP=2, PIECEWISE graph capture, MTP spec=5, the `splitting_ops` fix, GDN kernel
  mounted, text-only (NOMM). Both cards via `gpu-run`.
- **Command:** `/mnt/vm_8tb/b70/gpu-run bash serve.sh`
- **Result (scripts/93,94 benches, temp=0 greedy, 512-tok gen):**
  | spec | decode tok/s | accept_len | accept% |
  |---|---|---|---|
  | off | 18.74 | - | - |
  | 3 | 50.37 | 3.96 | 98.6 |
  | 4 | 57.24 | 4.93 | 98.3 |
  | **5** | **63-64** | 5.90 | 98.0 |
  | 6 | 28.67 (collapse) | 3.00 | 33.3 |
- **Verdict:** **~3.4x decode vs MTP-off** (63.11 vs the best MTP-off 18.74). spec=5 is the ceiling -- spec=6
  collapses because the MTP module is 1-layer (`mtp_num_hidden_layers=1`), useful horizon ~5 tokens. Accept ~98%
  is a **temp=0 greedy best case** (near-lossless W8A8 body -> the BF16 head drafts almost perfectly); expect
  lower at production temperature.

## Why TP=2 MTP works here (no Battlemage P2P, PCIe-staged oneCCL)

TP=2 batch-1 decode pays ~2 all-reduces/layer (~128/token) of tiny, **latency-bound** messages over CPU-staged
oneCCL (no P2P). That latency is why MTP-OFF TP=2 is only ~0.87x single-card. **MTP fires one collective round per
~6 verified tokens instead of per token**, so it amortizes the interconnect latency ~accept_len x -- which is
exactly why MTP is a *bigger* win at TP=2 than single-card (single-card 2.0x vs TP=2 ~3.4x).

## The three non-obvious ingredients (all wired in serve.sh)

1. **GDN kernel mount** -- :int8g bakes GDN OFF; mount the GDN-enabled `_xpu_C.abi3.so` (+ sibling lib).
2. **MTP-BF16 shim** (`patches/sitecustomize.py` on PYTHONPATH) -- forces ONLY the `Qwen3_5MultiTokenPredictor`
   drafter unquantized/BF16, else it loads through the W8A8 path -> 0% accept.
3. **`splitting_ops` THE FIX** (`SPLITOPS` knob in ../_common/lib.sh) -- TP+MTP records the spec `vllm::all_gather`
   into the SYCL graph, but oneCCL 2021.17's `sched` allgather has no graph-recordable impl -> capture crash. Listing
   the 3 collectives (`all_gather`/`all_reduce`/`reduce_scatter`) makes inductor partition at them so they run EAGER
   while decode stays captured. This is what overturned the old "TP=2 MTP DEAD" (M4). No custom communicator needed.

## Host dependency (not in repo)

- **Model:** `/mnt/vm_8tb/b70/models/Qwen3.6-27B-W8A8-sqgptq-mtp-graft` -- the W8A8-sqgptq quant (35 GiB) with 15
  BF16 `mtp.*` tensors grafted from the bf16 base as `model-mtp-graft.safetensors` (loaded via the no-index glob).
- **GDN kernel:** `/mnt/vm_8tb/b70/vllm-xpu-kernels/vllm_xpu_kernels/{_xpu_C.abi3.so,libgdn_attn_kernels_xe_2.so}`.

## Known issue (benchmark-only)

`vllm bench serve --dataset-name random` (random token IDs) **deadlocks** this config -- the request prefills then
decode stalls at ~0 t/s (out-of-distribution gibberish hits a degenerate MTP/rejection-sampler path). **Real
prompts are unaffected** -- verified OK at 2048-ctx with 128/256-token outputs and back-to-back requests
(scripts/97,98). So: benchmark with a real dataset (sharegpt) or real prompts, NOT `--random`. This does not
affect production serving.

## Knobs

- `MTPTOK=5` (default; the winner). `MTPTOK=0` is NOT MTP-off here -- to serve **MTP-off**, also drop SPLITOPS
  (set `SPLITOPS=` and `CAPSIZES=1,2,4,8`) so the collectives capture (18.74 t/s, faster no-MTP baseline).
- `CAPSIZES=1,2,4,6,8` includes the spec-verify batch (1+spec). `COMPILESZ=` MUST stay empty for spec-decode.
- `MAXLEN` defaults to 4096 (snappy interactive). **The full 262144 model max fits with MTP on** (scripts/100:
  fp16 KV pool 372,809 tokens at `UTIL=0.95`, 1.42x concurrency at 262K -- NOT VRAM-limited). Raise `MAXLEN` per
  long-doc session; the limit is SPEED, not memory (eager prefill ~390 tok/s flat with length -> 262K TTFT ~12 min;
  decode also slows on the 16 full-attn layers). `KVDTYPE=fp8_e4m3` ~2x the KV pool for multi-session long context.
