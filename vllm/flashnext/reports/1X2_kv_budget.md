# X2: Flash-Next VRAM and KV budget

Agent X2, 2026-10-20. Work in progress until the measurement section is
completed. No commits, systemd changes, or daily-driver startup.

## Header accounting (exact bytes)

CONFIG -> ISTA GSQ-RCO IQ3_XXS shard 1 and its shared IQ3_S shard 2;
48 layers, 512 experts/layer, 36 GDN layers, 12 full-attention layers,
24 Q heads, 2 KV heads, dimension 256. No MTP or vision weights in this table.

COMMAND -> `python3 vllm/flashnext/memory/gguf_budget.py <shard1> <shard2>`.
The script memory-maps headers and never copies/dequantizes payloads.
Raw inventory: `/mnt/vm_8tb/b70/build/vllm-flashnext/x2/headers.json`.

RESULT -> DP=2 here means DP=2 **with expert parallelism**, TP=1. Each
rank owns 256 experts/layer and a complete trunk. It is not two complete
model replicas. Sidecar means world size 1, trunk/KV on card 0 and the
profile-ranked 138/374 expert split. All experts within one layer have
the same shape/type, so expert identity does not affect these byte sums.

| Category | GGUF bytes | Each DP+EP card | Sidecar card 0 | Sidecar card 1 |
|---|---:|---:|---:|---:|
| Routed experts | 42,912,972,800 | 21,456,486,400 | 11,566,387,200 | 31,346,585,600 |
| Attention, GDN, norms | 1,770,450,432 | 1,770,450,432 | 1,770,450,432 | 0 |
| Hyper-connections | 1,283,235,840 | 1,283,235,840 | 1,283,235,840 | 0 |
| Token embeddings | 273,152,000 | 273,152,000 | 273,152,000 | 0 |
| lm_head | 437,043,200 | 437,043,200 | 437,043,200 | 0 |
| Routers | 125,829,120 | 125,829,120 | 125,829,120 | 0 |
| Shared experts and gate | 121,077,760 | 121,077,760 | 121,077,760 | 0 |
| PLE projections/state weights | 65,740,800 | 65,740,800 | 65,740,800 | 0 |
| QSA indexer weights | 39,333,888 | 39,333,888 | 39,333,888 | 0 |
| Device weight subtotal | 47,028,835,840 | 25,572,349,440 | 15,682,250,240 | 31,346,585,600 |
| PLE host mmap | 28,800,138,240 | 0 | 0 | 0 |

These are stored GGUF bytes, not allocator measurements: loading can
cast F32 norms/conv weights, create buffers, or duplicate tensors. The
trunk is 4,115,863,040 bytes. DP+EP duplicates that trunk across cards;
the sidecar removes the second trunk by changing placement and execution,
not by remotely aliasing independently executing DP ranks' parameters.
The host PLE pages are shared file-backed pages across DP workers.

VERDICT -> Do not use the H1d shorthand IQ3_XXS GU/Q2_0 down estimate of
11,361,484,800 / 30,791,270,400 bytes. The actual mixed-type expert totals
are 11,566,387,200 / 31,346,585,600, before the trunk.

## KV and fixed state

Only the 12 full-attention layers grow with context:

| Format | Main K/V bytes/token, all 12 layers | Extra pooled indexer bytes/token |
|---|---:|---:|
| BF16 | 24,576 | 0 for the current dense fallback |
| FP8, per-tensor scale | 12,288 | 0 for the dense fallback |
| Strata / B70 QSA INT8, FP16 scale per 64 | 12,672 | 1,536 (FP32, one 128-wide row per 4 tokens/layer) |

BF16 formula: `12 * 2(K,V) * 2(KV heads) * 256 * 2(bytes)`.
INT8 formula: `12 * (1024 codes + 32 scale bytes)`. QSA total is
14,208 bytes/token. FP8 scale tensors and alignment are additional fixed
costs. An unused hypothetical BF16 indexer would cost 768 bytes/token;
the implemented Strata/B70 pooled rows are FP32, so budget 1,536.

GDN is fixed per live sequence, not 36 more token-growing KV layers:
`36 * (48*128*128*4 + 10240*3*2) = 115,458,048` bytes/slot before
alignment, PLE history, checkpoint copies, and speculative state. vLLM
align mode needs two state blocks/group; speculative/prefill checkpoints
can add more. This is not an allocation of a full GDN state at every token.

C7b's 0.89 GiB / 6,144-token line is **group-aware capacity at a 2,048
request length**, not a universal 155 KiB/token storage cost. Attention
manager block size is 1,600; the cache pool also supplies three GDN groups.
At short request lengths the two aligned GDN blocks/group dominate.
Long requests amortize the fixed state. Sources: fork
`vllm/v1/kv_cache_interface.py:1057`,
`vllm/v1/core/kv_cache_utils.py:1015`, and C7b eager log lines 140-175.

## Strata comparison

Read `docs/20261016_night_lab.md`, DD configuration (read only), and
`/mnt/vm_8tb/b70/build/strata-ab/src-ep/src/core/{session,layer}.cpp`.
The DD uses INT8 KV, `--max-context 200000 --batch 2`, host token-embedding
lookup, a shared host PLE table, and the asymmetric expert split. Its
session planner allocates KV/indexer pools per session, fixed GDN state,
and one reusable set of operation buffers, rather than token-growing GDN.

The retained DD log `/mnt/vm_8tb/b70/results/hotschmoe-dd/engine.log:5`
onwards reports: host token embedding 260 MiB; two slot sessions 2.82 GiB
each; experts 10.77/29.19 GiB; card-1 prefill exchange 1,260 MiB;
final free card-0 VRAM 1,569 MiB (first startup). These are rounded log
measurements, not byte-exact reconstructions. It also carries MTP weights,
private draft buffers, HC Q8 copies, and later vision, absent from C7b.
The 141 tok/s Strata number includes batch MTP; X2 makes no speed claim.

For 400,000 resident tokens the QSA K/V+indexer payload is 5,683,200,000
bytes on card 0. BF16 dense K/V alone would be 9,830,400,000 bytes there.
The sidecar's much smaller card-0 expert allocation makes either memory
layout plausible; actual activation/graph/workspace peaks must still fit.
DP+EP instead needs 200,000 tokens/card: 4,915,200,000 BF16 bytes/card or
2,841,600,000 QSA bytes/card, plus fixed/checkpoint state and all overhead.

## Isolated implementation

New general plugin: `plugins/b70_memory_xpu/`. Default is inert. Set
`B70_MEMORY_FIX=1` to enable storage fixes, and
`B70_MEMORY_REPORT=/absolute/output/directory` for evidence. Installed
editable in the campaign runtime venv; no existing plugin/fork edits.

1. Full dense GGUF linears alias their shard and registered weight instead
   of retaining both device copies. Merged b/a and indexer q/k linears
   discard their unused dense placeholder after loading. Header estimate:
   129,024,000 + 57,016,320 = 186,040,320 bytes/rank. HC/PLE use plain
   `_WeightContainer` parameters and are **not** duplicate linear shards.
2. Dequant prefill uses one grow-only arena per device/dtype/stream instead
   of one allocation per quant type/shape. Existing callers consume the
   view before requesting another; no cross-stream alias is introduced.
   Header-derived dense shape cache: 2,083,389,440 -> at most 1,271,398,400
   bytes if all shapes, including a large-batch head, are exercised. This
   is a potential saving, not part of C7b's baseline: its legacy gemv path
   does not invoke this dequant prefill cache. X3 owns prefill dispatch.
3. Extend the worker's scoped 20 MiB allocator split limit to XPU through
   the device-neutral setter, restoring the previous configuration on
   exit. Upstream `gpu_worker.py:417` currently skips XPU. This tests
   whether small persistent allocations pin freed large profile segments.
   The profiler arithmetic is left intact; actual transient peaks and
   oneDNN/SYCL allocations must not be subtracted as guessed overhead.

For the EP sidecar, set `B70_EP_MAX_T=2048` with a matched
`--max-num-batched-tokens 2048`. The current default is 8192; capping it
reduces the persistent exchange payload on both cards. The cap must cover
every prefill/decode batch. Do not simply lower it under an 8192-token
scheduler limit.

The sidecar's raw allocations can be counted directly from
`plugins/b70_ep_xpu/csrc/moe_ep.cpp:203` and `region.hpp:58`:

| Explicit exchange/map bytes | Default max T=8192 | Matched max T=2048 |
|---|---:|---:|
| Card 0 | 168,698,052 | 42,475,716 |
| Card 1 | 170,074,308 | 42,672,324 |

These include dedicated region tail pages, flags/counters, and the owned
map copies, but exclude driver allocation rounding, host flags, torch
map tensors, and kernel scratch. Three kernel arenas have a decode
ceiling of 24,663,552 bytes/card: Q8 floor 1,572,864; two legacy float
buffers 8,388,608; scheduler arena 14,702,080. Q8 input scratch can grow
during larger prefill and retains old addresses for captured graphs;
do not interpret that ceiling as the measured total SYCL memory usage.

CPU evidence: `x2/cpu-test.log` prints
`CPU_ALIAS_FORWARD_OK STREAM_ARENA_OWNERSHIP_OK`; no devices mounted.
`x2/register.log` prints `REGISTER_OK`. Claude consultations with
`fable-5.1` and `claude-5.5` both returned `unrecognized_model`/404;
no responding model ID or advice was obtained.

## Measurement status

The detached X2 window controller waits outside GPU leases for C7b then
H1e. Prepared arms: current eager DP+EP BF16 at batch 2048; fixes at the
same batch; fixes plus FP8/capped batch 512. All use model identity checks,
two deterministic greedy probes, an 8 GiB host-memory abort, health,
compiled collective health, and teardown. No speed/stability/FP8 quality
promotion follows from these short probes.

Exact runtime current/fixed bytes and maximum-context tables will replace
this paragraph after those leased runs finish.
