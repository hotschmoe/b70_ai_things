# C7 -- GGUF FULL decode, expert parallel, Flash-Next load

Fork: `/mnt/vm_8tb/b70/build/vllm-flashnext/vllm`, branch `b70/flashnext`.
Tip `0a6170da8c`. Ahead of `origin/main` by 18. Not pushed. The campaign
repo is not committed. `_C.so` and `_C.graph.so` were not rebuilt.
`_C.graph.so` is 6987192 bytes. Image `vllm-flashnext-xpu:py312`, engine
`v0.32.1rc1.dev94+g5fd62688f`. Port 18104. Client name `hotschmoe-dev`
stays first. No `CCL_TOPO_P2P_ACCESS`. No `hotschmoe-dd` start. Claude
Fable 5.1 and Claude 5.5 consults returned unrecognized_model / not
logged in and were not a gate.

C7 commits on this branch, tip first. Other agents' commits sit between
them and were left in place (`c38d0a6197`, `e442b4b53f`, `9761b7b48a`,
`765993c197`):

```
0a6170da8c [XPU] View the Flash-Next PLE table instead of copying it
35da272b61 [XPU] Accept split indexer Q and K on one linear
2fec939431 [XPU] All-gather the MoE padding mask with naive DP dispatch
1fcc7bbd1d [XPU] All-gather equal XPU inputs into one tensor
2a50cfe2a7 [XPU] Copy a quantized GGUF payload once
4d73572e14 [XPU] Open split GGUF shards and install the PLE hash
2583e27239 [XPU] Decode GGUF MoE through fused moe_layer
```

Parent of `2583e27239` is the C4 tip `e3f784ebb2`.

## Step 1 -- one card, FULL_DECODE_ONLY

CONFIG: card 0. Qwen3.6-35B-A3B UD-IQ3_XXS,
`--load-format b70_gguf --quantization b70_gguf`,
`B70_GGUF_SO=_C.graph.so`. `FULL_DECODE_ONLY`, `FLASH_ATTN`,
compilation mode 0, capture sizes 1,2,3,4,8, `B70_XPU_CG_RECLAIM=500`,
`max_num_seqs` 4, `max_model_len` 33792, `gpu_memory_utilization` 0.90.
Container `vllm-fn-qwen36-gguf-c0`. Fork at serve: `2583e27239`.
Decode `T<=8` goes through one `moe_layer`. Prefill stays on grouped
gemm. The eager baseline is C4: C1 ITL p50 87.62 ms.

COMMAND: `harness/serve_qwen36_gguf.sh`, then
`harness/bench_latency.py --model hotschmoe-dev --alias
qwen3.6-35b-a3b-ud-iq3-xxs --decode-tokens 64 --prefill
1024,8192,32768`. Logs:
`/mnt/vm_8tb/b70/build/vllm-flashnext/logs/c7-serve-graph.log`,
`c7-bench-graph.json`, `c7-graph-probe.log`.

RESULT: Pre-health card 0 HEALTHY. Graph probe replay `max_abs=0`
(`GRAPH_PROBE_OK`; an earlier NaN probe used non-finite scales and is
not evidence). Load placed 733, skipped MTP 20, weights 16.96 GiB.
V2 runner captured 4 FULL graphs (size 8 dropped because
`max_num_seqs` is 4). Capture about 1 s, pool 0.02 GiB. KV 8.92 GiB,
381493 tokens, 11.29x at 33792
(`c7-serve-graph.log` kv_cache_utils / gpu_worker lines). Identity
`hotschmoe-dev`, `qwen3.6-35b-a3b-ud-iq3-xxs`. Greedy temperature 0,
16 tokens, prompt "The capital of France is", both calls:

```
 Paris.
The capital of France is Paris.
The capital of France is
```

Same as C4. Bench exit 0.

| measurement | ITL p50 | ITL p90 | ITL p99 | TTFT | tok/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| C1 (64 chunks) | 11.114 ms | 12.407 ms | 13.457 ms | 174.36 ms | 90.16 |
| C2 stream 0 (46 tokens) | 16.514 ms | 18.992 ms | 29.712 ms | 830.65 ms | 61.34 |
| C2 stream 1 (40 tokens) | 16.654 ms | 18.899 ms | 19.755 ms | 867.12 ms | 62.01 |

Warmup TTFT 211.21 ms. Prompt tokens on C1 were 24. Prefill (completion
1, so the rate is the prefill): 1017 tokens 177.005 tok/s, TTFT 5745.60
ms; 8185 tokens 175.581 tok/s, TTFT 46616.60 ms; 32762 tokens 173.230
tok/s, TTFT 189124.23 ms. Serve gpu-run exit 0. Post-health card 0
HEALTHY. No xe-reset.

VERDICT: the 35B GGUF decode path is graph-capturable on one card and
matches the eager greedy string. C1 ITL p50 is 11.11 ms on this serve.
Prefill stays near 175 tok/s through 32K. This is the 35B UD-IQ3_XXS
GGUF, not the Strata Flash-Next daily driver.

## Step 2 -- two cards, 35B expert parallel

Both-card runs used `bin/gpu-run` without `--card` only from a shell
that already saw both owner files empty. Each window started with
`bin/xpu-collective-health`. No P2P.

### DP=2 path to a serve

Three eager DP=2 profiles died before the measured one. Logs are under
`/mnt/vm_8tb/b70/build/vllm-flashnext/logs/`.

CONFIG (all three): TP=1, DP=2, `--enable-expert-parallel`,
`--api-server-count 1`, `--enforce-eager`, `max_model_len` 2048,
`max_num_seqs` 2, `B70_COLL_LOG` on, same 35B GGUF. `ZE_AFFINITY_MASK`
unset.

1. `08:19`, `c7-ep-dp-serve` (not preserved under an -ok name). Equal
   all-gather went through `dist.all_gather([output], input)` and
   oneCCL `allgatherv_ll_ring` segfaulted both ranks during the memory
   profile. SERVE_RC 1. No xe-reset: another agent held card 1 within
   seconds. Fix: `1fcc7bbd1d` calls `dist.all_gather_into_tensor` for
   equal sizes (`xpu_communicator.py`).
2. `08:22` to `08:24`, preserved as `c7-ep-dp-serve-into.log`.
   `all_gather_into_tensor` returned and both ranks still segfaulted in
   `allgatherv_ll_ring` inside `PersistentDeviceCodeCache::getItemFromDisc`.
   Init all-reduce had already completed. SERVE_RC 1.
3. `08:26` to `08:28`, preserved as `c7-ep-dp-serve-nocache.log`.
   `SYCL_CACHE_PERSISTENT=0` and a fresh `SYCL_CACHE_DIR` of
   `cache/sycl-c7-dp`. No segfault. Both workers raised
   `RuntimeError: is_padding size mismatch, expected: 4096` in
   `torch.ops._moe_C.topk_softmax` during the 2048-token profile
   (local mask 2048, gathered gating 4096). SERVE_RC 1. No xe-reset.
   Fix: `2fec939431`. `moe_runner._borrow_local_padding` sends the
   local mask as `extra_tensors` on naive DP dispatch and installs the
   gathered mask for top-k.

### DP=2 measured

CONFIG: fork `e442b4b53f`, which contains `2fec939431`. G4's
`--park-finished-tail` is registered and default off; the flag was not
passed. Same DP shape as above, plus the fresh SYCL cache.
`B70_COLL_LOG` stayed on, so every collective is synchronized. These
times include that sync.

COMMAND: `bash /tmp/c7-dp-window.sh` from 08:58:00. Preserved
`c7-ep-dp-serve-ok.log`, `c7-coll-ok.log`, `c7-dp-window-ok.out`,
`c7-bench-ep-dp.json`, `c7-ep-dp-greedy.json`.

RESULT: HEALTH_RC 0. Placed 733, skipped MTP 20, 11.64 GiB, linear EP
128/256. READY 09:00:32. Both greedy texts match the C4 Paris string
(`c7-ep-dp-greedy.json` `match_c4` true). Identity unchanged. Bench
exit 0. SERVE_RC 0. docker stop 09:00:52. No crash. Post-health was
not run: C9 held card 0 and G5 held card 1 at 09:01:05. No xe-reset.

| measurement | ITL p50 | ITL p90 | ITL p99 | TTFT | tok/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| C1 (n=32) | 121.404 ms | 143.799 ms | 153.086 ms | 462.42 ms | 7.86 |
| C2 stream 0 | 120.729 ms | 127.713 ms | 151.260 ms | 523.45 ms | 8.11 |
| C2 stream 1 | 120.788 ms | 128.253 ms | 151.326 ms | 522.66 ms | 8.11 |

Warmup TTFT 590.92 ms. Eager plus the counter sync, so this ITL is a
different measurement from the 11.11 ms graph serve.

Collectives, from `c7-coll-ok.log`. The counter writes
`name=count/sum_ms` (`harness/patches/c7_collectives.py`). Every
`tokens=1` execute step on both ranks is `coll=160`:
`all_gather_into_tensor=120` and `reduce_scatter_tensor=40`. No
`all_reduce` on the decode step (TP=1). That is 40 MoE layers times
3 all-gathers (hidden, router logits, padding mask) plus 1
reduce-scatter.

Steady steps drop the one warmup whose `coll_ms` is 643.273 (pid 423,
wall 743.662 ms; the single per-call dump is that warmup and includes
a 410 ms reduce-scatter). Remaining:

| rank | steady steps | wall median (min..max) | coll sum median (min..max) | 120 all-gathers, sum median | 40 reduce-scatters, sum median |
| --- | ---: | --- | --- | ---: | ---: |
| pid 423 | 67 | 118.776 ms (116.148..151.346) | 20.498 ms (18.980..29.344) | 14.897 ms | 5.607 ms |
| pid 424 | 92 | 119.501 ms (116.018..151.948) | 20.859 ms (18.467..49.680) | 15.201 ms | 5.649 ms |

Per call after warmup that is about 0.13 ms for an all-gather and
about 0.14 ms for a reduce-scatter. The 09:01 journal line that says
"median 15.01 ms" is this sum, not one collective. Collective time is
about 21 ms of a 119 ms eager step. The rest is the gathered-batch
GGUF MoE plus the counter's own sync. Profile dummy at 2048 tokens was
also 160 collectives (`all_gather_into_tensor` 120 + `reduce_scatter_tensor`
40) before any decode.

VERDICT: DP=2 expert parallel serves, matches the one-card greedy
string, and issues 160 collectives per decode step. ITL p50 is 121 ms
on this eager, sync-instrumented serve.

### TP=2

Packed attention, shared-expert, and vocab sharding are not implemented.
`serve_loader.py` raises when tensor-parallel size is not 1
(around line 425 at tip `0a6170da8c`). That raise is the measurement.
Sharding was not added.

1. `09:06`. Collective health passed, then card 0 was taken before
   `gpu-run` acquired it. SERVE_RC 124 (flock timeout). RAISE 0.
   Preserved `c7-ep-tp-serve-race.log`, `c7-tp-window-race.out`. Not a
   loader result.
2. `10:41` health HEALTHY, serve `10:42:05`. Both workers at 10:43:23
   raised `RuntimeError: tensor parallel size is 2. Packed attention,
   shared expert, and vocab sharding are not implemented.`
   (TP0 pid 259, TP1 pid 260). SERVE_RC 137 is docker stop after
   RAISE_SEEN, not a wedge. Preserved `c7-ep-tp-serve-raise.log`,
   `c7-tp-window-raise.out`. The following Flash-Next window's
   collective health passed, so no xe-reset.

VERDICT: TP=2 does not serve this packed GGUF. The working expert-parallel
shape is DP=2, TP=1, `--enable-expert-parallel`.

## Step 3 -- Flash-Next IQ3_XXS, DP=2

Recorded Strata DD strings (`harness/strata_reference.json`, stack
`ista-iq3-dd`, temperature 0, `enable_thinking` false, model
`hotschmoe-dd`). The short math prompts are 25 tokens, under the 2051
KV length where dense GQA and QSA agree, so the text is a real
coherence check once the text model loads. `B70_QSA` stays unset.
Indexer weights still have to load so the GGUF audit passes.

| prompt | text | sha256 |
| --- | --- | --- |
| Return only the integer: 17 * 19. | 323 | 3949ac1596ec77106a709a618bf5adcb19b77537ce8bcbdf54ff830169cdd084 |
| Return only the integer: 23 * 29. | 667 | 8b5551ea922dd24625c45051c64adb50fdff91fecdf5327a02c7b0be3933965e |

Weights: shard 1
`/mnt/cache/b70/qwen3.8-flash-next/ista-gsq-rco/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf`
(arch `qwen4exp`). Shard 2 is `per_layer_token_embd.weight`, ggml type
20, ne `(160, 320001536)`, uint8 shape `(320001536, 90)`, 28800138240
bytes. HF config is `/mnt/vm_8tb/b70/build/backend-decision/hf-flashnext`.
The 102 GB BF16 `ple_table_qwen4exp.pt` was not opened.

### Load attempt 1 -- indexer K

CONFIG: fork before `35da272b61`. DP=2 TP=1 EP, enforce-eager,
`max_model_len` 2048, docker `--memory 100g`,
`SYCL_CACHE_PERSISTENT=0`, container `vllm-fn-flashnext-gguf-ep`.
Served names `hotschmoe-dev`, `qwen3.8-flash-next-iq3-xxs`.

COMMAND: `/tmp/c7-fn-window.sh` after health at 10:48:48. Serve
10:49:16. Preserved `c7-fn-serve-index.log`, `c7-fn-window-index.out`.

RESULT: HEALTH_RC 0. Both workers placed 100 tensors, then
`IndexError` in `_check_linear` at `sizes[_SHARD_INDEX["k"]]`.
`index_qk_proj` is one `ReplicatedLinear`. GGUF stores indexer Q
`(rows 512)` and K `(rows 128)` as separate tensors onto that one
partition. SERVE_RC 1. Stop 10:51:11. No greedy. No xe-reset.
Fix: `35da272b61`. A one-partition module accepts shards `q` and `k`
(and the other single-shard names) and `_check_split_shards` requires
the row sum to equal the module output. `py_compile` passed.

### Load attempt 2 -- host OOM

CONFIG: fork `35da272b61`. Same serve shape.

COMMAND: waiter `/tmp/c7-wait-fn.sh` appended the 11:12 journal line
and exec'd `/tmp/c7-fn-window.sh`. Preserved `c7-fn-serve-oom.log`,
`c7-fn-window-oom.out`.

RESULT: HEALTH_RC 0 at 11:12:39. Serve 11:13:08. Engine config DP=2
TP=1 EP, `max_seq_len` 2048, enforce-eager. Both workers placed 100,
200, ... 1200 tensors, last line 11:17:22. No Python traceback. Kernel
11:17:41:

```
oom-kill:constraint=CONSTRAINT_NONE, global_oom,
task=VLLM::Worker_DP,pid=3258155,uid=1000
Out of memory: Killed process 3258155 (VLLM::Worker_DP)
total-vm:129572252kB, anon-rss:46915752kB, file-rss:12kB
```

Mem-Info at the kill: active_anon 70711248 kB, inactive_anon 26628912
kB, active_file 0, gpu_active 22211584 kB, all_unreclaimable, Normal
free about 195 MB. Reaped 11:17:44. Engine core initialization failed.
Stop 11:18:06, SERVE_RC 1. The window script's exit 0 is the script
finishing, not a ready server. No xe-reset. Cards at 11:18 were C12
on card 0 and E2 on card 1.

The placed-1200 line is `blk.47` territory. The following tensor is
the 27 GiB PLE table. `_owning_uint8` copied it off the memmap into
anon. Two workers plus about 21 GiB `gpu_active` do not fit in the
121 GiB host. File cache was already 0, so the copy could not land
in reclaimable cache.

### PLE view, not yet served

CONFIG: fork `0a6170da8c`. `serve_loader._host_table_payload`
(`serve_loader.py` around line 249) builds a uint8 tensor with
`torch.from_numpy` on the C-contiguous memmap. A strided or
non-memmap source raises. `HostPackedTable.owner`
(`ple.py` around line 34) holds that numpy object so the mapping
survives `del readers` at the end of `load_weights`. Quantized expert
and linear payloads still go through `_owning_uint8`.

COMMAND: `B70_AGENT=C7 bin/cpu-run` on a CPU script that calls
`test_host_table_payload_views_memmap` and opens both shard-2 headers.
No GPU.

RESULT: cpu-run exit 0 in 12 s. `HOST_TABLE_TEST_OK`. `PY_COMPILE_OK`.
Both the IQ3_XXS and the IQ3_S shard-2 tensors are C-contiguous uint8
memmaps of 28800138240 bytes, so the serve path will take the view and
will not allocate the anon copy. `torch.from_numpy` warns once that the
read-only memmap is not writable. Gather reads rows. It does not write
the table.

At 11:32 card 0 was held by H1 (`xpu-health` since 11:22) with C12 and
C13 queued on that card, and card 1 was held by E2
(`vllm-fn-lowbit-e2` since 11:18). MemAvailable was about 108 GiB.
No C7 container, no C7 gpu-run, no C7 waiter. A both-card `gpu-run`
was not queued, because queuing while one card is held blocks the
other card for everyone else.

VERDICT: Flash-Next DP=2 has not produced a greedy string. The indexer
check is in the fork. The PLE copy that killed the host is replaced by
a memmap view and checked on CPU. The coherence compare is the next
measurement, and it waits on both leases plus a quiet host.

## Next step

When both `/mnt/vm_8tb/b70/gpu.lock.0.owner` and
`gpu.lock.1.owner` are empty, and `MemAvailable` is at least 100 GiB,
append a JOURNAL window line and, in that same shell, run:

```
bash /mnt/vm_8tb/b70/build/vllm-flashnext/logs/c7-fn-window.sh
```

The script (also at `/tmp/c7-fn-window.sh`) refuses if either owner
file is non-empty, runs `bin/xpu-collective-health`, refuses if
`MemAvailable` is under 100 GiB, serves eager DP=2 TP=1 EP for at most
16 minutes, and `docker stop`s `vllm-fn-flashnext-gguf-ep` if
`MemAvailable` falls under 8 GiB. Chat greedy, temperature 0,
`enable_thinking` false, prompts `17 * 19` and `23 * 29`, compared to
`323` and `667`. Logs: `c7-fn-serve.log`, `c7-fn-greedy.json`.

Do not raise docker memory above host RAM. Do not load the BF16 PLE
`.pt`. Do not set `B70_QSA`. Do not set `CCL_TOPO_P2P_ACCESS`. Do not
implement packed TP sharding. If the view still drives a global OOM,
the watchdog should stop the container first; if it does not, stop and
record the new anon watermark instead of retrying the copy.

## What was measured

- One-card 35B GGUF FULL_DECODE_ONLY: identity, exact greedy match with
  C4, C1 ITL p50 11.114 ms, C2 ITL p50 16.514 ms and 16.654 ms, prefill
  about 175 tok/s through 32762 tokens, clean teardown, card 0 HEALTHY.
- DP=2 EP eager: same greedy string, 160 collectives per decode step,
  C1 ITL p50 121.404 ms with the collective counter synchronizing each
  call.
- TP=2: loader refusal on both ranks. One earlier attempt was a lock
  timeout (exit 124), not a model result.
- Flash-Next IQ3_XXS: two load failures (indexer partition, then host
  global OOM). No greedy text and no ITL. The PLE memmap view is
  committed and CPU-checked, and the serve has not been repeated.
