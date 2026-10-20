# C7b -- first coherent Flash-Next IQ3_XXS serve on vLLM

Script: `/mnt/vm_8tb/b70/build/vllm-flashnext/logs/c7b-fn-graph-window.sh`

That script is the serve recipe. It ran from 16:09:43 to 16:16:18 UTC
on 2026-10-20. `WINDOW_RC:0`, `SERVE_RC:0`, `GREEDY_RC:0`, `BENCH_RC:0`.
Container `vllm-fn-flashnext-gguf-graph`, port 18104. `hotschmoe-dd`
was not started. Both GPU locks were empty after the post-health
below. No `DEVICE_LOST`. No xe reset.

The eager gate that passed first is
`/mnt/vm_8tb/b70/build/vllm-flashnext/logs/c7b-fn-eager98.sh`
(ready 15:09:33, same greedy strings). The graph recipe is the one
to reuse.

## Recipe

- Model dir: `/mnt/vm_8tb/b70/build/backend-decision/hf-flashnext`
- Weights: IQ3_XXS shard 1 plus the IQ3_S PLE shard, via
  `B70_GGUF_PATH` inside the script.
- `--load-format b70_gguf --quantization b70_gguf --dtype bfloat16`
- `--language-model-only --trust-remote-code`
- `--tensor-parallel-size 1 --data-parallel-size 2`
- `--enable-expert-parallel --api-server-count 1`
- `--max-model-len 2048 --max-num-batched-tokens 2048`
- `--max-num-seqs 4 --gpu-memory-utilization 0.98`
- `--attention-backend FLASH_ATTN`
- `--compilation-config` mode 0, `cudagraph_mode FULL_DECODE_ONLY`,
  capture sizes 1,2,3,4,8
- `B70_XPU_CG_RECLAIM=500`
- `B70_GGUF_SO`: `vllm/flashnext/plugins/b70_gguf_xpu/b70_gguf_xpu/_C.c12.so`
  (sha256 `b3c15e14ba5811d90dfe8a65e4ccc1353fd67b3f44e4180810fac22b4ed49979`).
  O3's `_C.graph.so` was left in place.
- FlashAttention: private overlay
  `/mnt/vm_8tb/b70/build/vllm-flashnext/c7b-fa-overlay`, prepended to
  `PYTHONPATH`. It supplies `_vllm_fa2_C.abi3.so` and
  `libattn_kernels_xe_2.so` (11944232 bytes) built by
  `logs/c7b-fa-build.sh` for paged decode
  `16,256,64,false,false,false`
  (`decode_policy_q16_h256_p64`, causal/local/sink all false).
  The shared vllm-xpu-kernels 0.1.15.4 wheel was not modified
  (mtime 2026-10-19 22:31, 306868136 bytes).
- Served names, in order: `hotschmoe-dev`,
  `qwen3.8-flash-next-iq3-xxs`
- Engine: `v0.32.1rc1.dev94+g5fd62688f`
- Image: `vllm-flashnext-xpu:py312`
- No P2P. `CCL_TOPO_P2P_ACCESS` is not set.

FULL capture completed (2/2, then 4/4). The log has no
"kernel not compiled" line and no Level Zero error 44.
Model load was 24.9 GiB and about 155 s per rank.

## Identity

`/v1/models` via `harness/bench_latency.py`:

- ids: `hotschmoe-dev`, `qwen3.8-flash-next-iq3-xxs`
- `first_ok=true`, `alias_ok=true`

## Coherence

Temperature 0, `enable_thinking` false.

| prompt | text | match |
| --- | --- | --- |
| Return only the integer: 17 * 19. | 323 | true |
| Return only the integer: 23 * 29. | 667 | true |

Prompt 25 tokens, completion 4. Log
`logs/c7b-graph-greedy.json`.

Bench coherence (same file as the ITL run) was `ok=true`, `same=true`,
both streams: ` Paris, which is also the country's most populous city, with an estimated `.

## ITL

`harness/bench_latency.py` under `bin/cpu-run --quiet`,
`--decode-tokens 64 --prefill 1024 --tag graph-2048`.
Output: `logs/c7b-bench-graph.json`.

C1, 24 prompt tokens, 64 completion tokens:

- ITL p50 81.75 ms, p90 106.38 ms, p99 140.09 ms, mean 67.20 ms
- decode 14.88 tok/s
- measured TTFT 609 ms (warmup TTFT 2220 ms)

C2, two streams, 29 prompt tokens, 64 completion tokens:

- stream 0: ITL p50 33.20 ms, p90 94.01 ms, p99 105.06 ms, mean 53.10 ms, decode 18.83 tok/s
- stream 1: ITL p50 33.14 ms, p90 94.22 ms, p99 105.06 ms, mean 53.10 ms, decode 18.83 tok/s

Prefill probe, 1017 tokens, 1 completion token: TTFT 16269 ms,
62.5 tok/s. Prefill stays eager under `FULL_DECODE_ONLY`.

The bench file's Strata night-final-v3 reference is C1 141 tok/s
(ITL 7.09 ms), C2 92 tok/s per stream (ITL 10.87 ms), prefill about
2200 tok/s. This serve is coherent and below those rates.

## Host RSS

Ready snapshot 16:15:28, `logs/c7b-rss-graph-ready.txt`:

- MemAvailable 48392432 kB
- AnonPages 14085920 kB (13.43 GiB)
- Worker pid 3398537: RSS 4930408 kB, PSS 4754501 kB, Anonymous 4614396 kB
- Worker pid 3398538: RSS 4938344 kB, PSS 4762187 kB, Anonymous 4620928 kB

Strata held this model in about 16 GiB after the weights were on the
GPU. Host anonymous at ready is 13.43 GiB, under that figure. The
two worker RSS values are about 4.7 GiB each and are mostly anonymous;
summing RSS double-counts the shared file maps (about 0.30 GiB
Shared_Clean each).

The large excess was transient and was measured on the eager load,
not as a steady resident set. Eager placement peaked at AnonPages
53938904 kB (15:08:08) while both ranks materialized packed tensors,
then fell to 12847624 kB (12.26 GiB) at ready 15:09:33. Steady host
anonymous on both serves is under Strata's 16 GiB.

## KV and 32k

FLASH_ATTN asked for page size 64, then the hybrid cache aligned the
attention block to 1600.

- Available KV cache memory: 0.64 GiB (DP0 log line)
- DP0 KV pool: 4352 tokens, 2.12x concurrency at 2048
- DP1 KV pool: 4608 tokens, 2.25x concurrency at 2048

32k was not launched. 32768 tokens does not fit in a 4352 or 4608
token pool. The 0.98 cap left 0.64 GiB of KV. Raising utilization
past 0.98 was not attempted. Eager at the same 0.98 cap, without
graphs, had 0.89 GiB and 6144 tokens, which is also short of 32k.
H1d and G4 can use this 2048 graph recipe. A 32k serve needs a
larger KV pool than this profile has.

## Health

Before, 16:09:43 window, collective probe finished 16:10, log
`logs/c7b-coll-health-graph.txt`:

- `HEALTH_RC:0`
- `MEM_AVAILABLE_KB:117876416`
- `COLLECTIVE_HEALTH_OK world_size=2 shape=4x5120 compiled_iterations=10 p2p=0`
- `xpu-collective-health: HEALTHY`

After, 16:17, log `logs/c7b-coll-health-graph-after.txt`:

- `HEALTH_RC:0`
- same `COLLECTIVE_HEALTH_OK` line, `p2p=0`
- `xpu-collective-health: HEALTHY`
- gpu-run done in 33s, exit 0
- both owner files were empty when that command returned

MemAvailable after teardown was 117663820 kB. AnonPages 1557196 kB.
Port 18104 was closed. Container `vllm-fn-flashnext-gguf-graph` was gone.
