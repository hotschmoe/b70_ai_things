# X2: Flash-Next VRAM and KV budget

Agent X2, 2026-10-20. No commits, systemd changes, or daily-driver startup.

Both fixed layouts have pool space for 2x200K with BF16. DP needs the
512-token batch cap and has little headroom for graphs/MTP. The EP
sidecar has measured KV budget
15,311,619,195 bytes on card 0 versus 10,341,580,800 required. Short
identity, deterministic coherence and concurrent-prefill gates pass.
This is a memory result; long-context quality and latency are not qualified.

## Header accounting (exact bytes)

CONFIG -> ISTA GSQ-RCO IQ3_XXS shard 1 and its shared IQ3_S shard 2;
48 layers, 512 experts/layer, 36 GDN layers, 12 full-attention layers,
24 Q heads, 2 KV heads, dimension 256. No MTP or vision weights in this table.

COMMAND -> `python3 vllm/flashnext/memory/gguf_budget.py <shard1> <shard2>`.
The script memory-maps headers and never copies/dequantizes payloads.
Final inventory: `/mnt/vm_8tb/b70/build/vllm-flashnext/x2/headers-final.json`;
the original `headers.json` preserves the initial mapping upper bounds.

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
| FP8, dynamic per-token/head FP32 scale | 12,480 | 0 for the dense fallback |
| Strata / B70 QSA INT8, FP16 scale per 64 | 12,672 | 1,536 (FP32, one 128-wide row per 4 tokens/layer) |

BF16 formula: `12 * 2(K,V) * 2(KV heads) * 256 * 2(bytes)`.
INT8 formula: `12 * (1024 codes + 32 scale bytes)`. QSA total is
14,208 bytes/token. Per-tensor FP8 scales are fixed; dynamic token/head
scales are included above. Backend alignment adds padding. An unused
hypothetical BF16 indexer would cost 768 bytes/token;
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

`memory/capacity.py` calculates a conservative bound for that four-group
layout, retaining a null block and six GDN blocks per live stream. The
rounded 0.89 GiB baseline implies 24 pool blocks: one long sequence can
hold about 27,200 tokens/card, rather than a universal 6,144-token limit.
The exact measured budget/specs must replace this rounded-log inference.
For 200K per live stream (no MTP/checkpoint extras):

| Layout / format | Expected manager block | Required pool bytes |
|---|---:|---:|
| DP+EP, one 200K stream/card, BF16 | 1,600 | 5,190,451,200/card |
| DP+EP, one 200K stream/card, FP8 | 3,136 | 2,735,996,928/card |
| DP+EP, one 200K stream/card, FP8 token/head | 3,136 | 2,778,746,880/card |
| DP+EP, one 200K stream/card, QSA INT8 | 2,752 | 3,128,033,280/card |
| Sidecar, two 200K streams/card 0, BF16 | 1,600 | 10,341,580,800 on card 0 |
| Sidecar, two 200K streams/card 0, FP8 | 3,136 | 5,433,458,688 on card 0 |
| Sidecar, two 200K streams/card 0, FP8 token/head | 3,136 | 5,518,356,480 on card 0 |
| Sidecar, two 200K streams/card 0, QSA INT8 | 2,752 | 6,216,966,144 on card 0 |

Expected blocks use the observed 64-token backend alignment. Different
backend metadata, MTP state, prefix checkpoints or grouping change them.

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
editable in the campaign runtime venv. The one-line fork constructor fix
is preserved in `patches/x2-packed-embedding.patch`: pass `quant_config`
to `embed_tokens`. Existing plugins and other agents' fork edits were
preserved. No commit.

1. Keep token embeddings packed. The existing constructor materializes
   1,271,398,400 BF16 bytes instead of 273,152,000 packed IQ3_S bytes:
   998,246,400 bytes/card. The fixed arm matches all 20,480 sampled BF16
   lookup values against CPU GGUF dequantization on both cards.
2. Full dense GGUF linears alias their shard and registered weight instead
   of retaining both device copies. Merged b/a and indexer q/k linears
   discard their unused dense placeholder after loading. Measured saving:
   57,016,320 bytes/rank (17,694,720 GDN b/a + 39,321,600 indexer q/k).
   Router, conv, HC and PLE bypass this method. The initial header-only
   129,024,000-byte full-dense upper bound is not an actual duplicate
   in this model and is excluded from the final savings.
3. Dequant prefill uses one grow-only arena per device/dtype/stream instead
   of one allocation per quant type/shape. Existing callers consume the
   view before requesting another; no cross-stream alias is introduced.
   Header-derived dense shape cache: 2,083,389,440 -> at most 1,271,398,400
   bytes if all shapes, including a large-batch head, are exercised. This
   is a potential saving, not part of C7b's baseline: its legacy gemv path
   does not invoke this dequant prefill cache. X3 owns prefill dispatch.
4. Use `torch.xpu.mem_get_info` in place of the vLLM kernel plugin's
   override of `torch.accelerator.get_memory_info`. The leased comparison
   below finds a constant 1,712,115,712-byte reduction in both total and
   free memory from the override.
5. Calculate extra transient headroom as `max(0, peak_reserved -
   retained_reserved)`. The driver-consumed delta already contains
   retained allocator segments. Reservation peaks include fragmentation;
   subtracting all inactive bytes is unsafe. Measured non-Torch
   consumption remains fully included.

The initial scoped split-limit candidate was tested and removed. The XPU
setter accepts/restores `max_split_size_mb:20` but still retains a 64 MiB
segment for a 2 MiB live tensor. `UR_L0_DISABLE_USM_ALLOCATOR=1` did not
change that result. `expandable_segments:True` reduced the synthetic reservation
to 20 MiB; it is not enabled in the delivered fixes, since graph pointer
stability and full-model behavior were not measured with it.

For the EP sidecar, set `B70_EP_MAX_T=2048` with a matched
`--max-num-batched-tokens 2048`. The current default is 8192; capping it
reduces the persistent exchange payload on both cards. The cap must cover
every prefill/decode batch. Do not simply lower it under an 8192-token
scheduler limit.

The sidecar's raw allocations can be counted directly from
`plugins/b70_ep_xpu/csrc/moe_ep.cpp:203` and `region.hpp:58`:

| Explicit exchange/map bytes | Default max T=8192 | Matched max T=2048 |
|---|---:|---:|
| Card 0 | 168,763,652 | 42,541,316 |
| Card 1 | 170,139,908 | 42,737,924 |

The method reserves 64 layers even though this checkpoint uses 48.
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
The research alias is registered as
`x2-flashnext-ista-gsq-rco-iq3xxs-memory` in
`evals/configs/models.yaml`; live `/v1/models` identity and the exact
GGUF model/method/scheme are saved in each arm's JSON evidence.

## Measurement status

CONFIG -> matched eager DP=2+EP, TP=1, batch 2048, max length 2048,
two sequences/rank, utilization 0.98, `_C.c12.so`, QSA/PREFILL off.
Current workers 493/card 0 and 484/card 1; fixed workers 492/card 0
and 493/card 1 (different containers/output directories).

COMMAND -> `memory/run_window.sh`; JSON summarized by `memory/summarize.py`.
Raw directories `x2/current/` and `x2/aliases/` under the campaign build.

RESULT -> exact bytes, before KV initialization:

| Item | Current card 0 | Current card 1 | Fixed card 0 | Fixed card 1 |
|---|---:|---:|---:|---:|
| Initial device use, before model (excluded from delta) | 585,863,168 | 227,766,272 | 585,871,360 | 227,799,040 |
| Model-owned Torch storage | 26,656,237,720 | 26,656,237,720 | 25,600,975,000 | 25,600,975,000 |
| All live Torch allocations after load | 26,741,253,120 | 26,741,253,120 | 25,681,431,552 | 25,681,431,552 |
| Retained Torch reservations after profile | 28,087,156,736 | 28,087,156,736 | 27,101,495,296 | 27,101,495,296 |
| Non-Torch increase since baseline | 1,493,999,616 | 1,472,102,400 | 1,508,294,656 | 1,486,389,248 |
| Driver-consumed delta | 29,581,156,352 | 29,559,259,136 | 28,609,789,952 | 28,587,884,544 |
| Extra transient headroom | 1,337,489,408 | 1,337,489,408 | 1,260,388,352 | 1,260,388,352 |
| Total non-KV budget | 30,918,645,760 | 30,896,748,544 | 29,870,178,304 | 29,848,272,896 |
| Requested budget | 31,879,578,502 | 31,879,578,502 | 33,557,451,899 | 33,557,451,899 |
| Available KV budget | 960,932,742 | 982,829,958 | 3,687,273,595 | 3,709,179,003 |
| Max BF16 tokens, one long stream/card | 27,200 | 27,200 | 137,600 | 139,200 |

Current requested bytes reconstruct the worker's `ceil(total * .98)`;
fixed requested/available values are also emitted by the worker wrapper.
Maximum tokens reserve one null block and six aligned GDN blocks per
active stream; the configured 2048 cap was kept for the matched serve.
The max-token rows are budget calculations, not completed long prefills.

Owned storage falls by exactly 1,055,262,720 bytes/card: the packed
embedding plus the two merged placeholder categories. The remaining
live Torch allocation is runner/allocator-managed storage. RoPE is a
single shared 33,554,432-byte BF16 buffer, not 12 duplicate tables.

The census below assigns each unique storage once to its first owning
module. Every DP rank has the same tensor bytes; allocator overhead above
is separate. This explains the difference between GGUF payload and live
storage without counting the same shared tensor twice.

| Model-owned category | Current, each DP card | Fixed, each DP card |
|---|---:|---:|
| Routed experts | 21,456,486,400 | 21,456,486,400 |
| Attention/GDN including shared RoPE | 1,818,725,696 | 1,801,030,976 |
| Hyper-connections | 1,281,249,280 | 1,281,249,280 |
| Embedding | 1,271,398,400 | 273,152,000 |
| lm_head | 437,043,200 | 437,043,200 |
| Routers | 125,829,120 | 125,829,120 |
| Shared experts | 121,077,760 | 121,077,760 |
| PLE projections | 65,679,640 | 65,679,640 |
| QSA indexer | 78,649,344 | 39,327,744 |
| Other buffers | 98,880 | 98,880 |
| Total | 26,656,237,720 | 25,600,975,000 |

Subtracting routed experts, the fixed DP trunk/buffers are
4,144,488,600 bytes/rank. Both DP ranks need their own copy. The sidecar
owns 47,057,559,704 model bytes across both cards versus 51,201,950,000
for fixed DP, saving 4,144,390,296 total bytes after its extra map buffer.

Current profile peak reserved is 29,345,447,936 bytes/card. Its physical
headroom over retained reservations is 1,258,291,200, so the original
allocation-based formula overcounts 79,198,208 bytes. In the fixed layout
the analogous correction is 77,101,056. Inactive slack is 1,345,903,616
bytes in the current layout, but its largest free block is 835,928,064;
fragmentation prevents treating all of it as interchangeable workspace.

The non-Torch row includes SYCL/native kernel arenas, communications,
driver caching and oneDNN allocations. It is a measured combined delta;
it is not all oneDNN scratch. Largest first-profile raw growth occurs
around the first shared-expert projection (about 510/534 MB across ranks)
and first GDN qkv projection (about 128/126 MB). These callbacks can
overlap communication, so they do not establish exclusive ownership.
Source-counted kernel arenas are listed above; the remainder is retained
in the budget rather than guessed away.

The legacy DP MoE profile gathers 4096 rows across two ranks (2048/rank),
top-10, hidden 2560, FFN 640. Dominant explicit buffers are:

| Buffer | Shape | Bytes |
|---|---|---:|
| Expanded BF16 input | 40960 x 2560 | 209,715,200 |
| FP16 copy for grouped GEMM | 40960 x 2560 | 209,715,200 |
| Gate, up, or hidden FP32 (each) | 40960 x 640 | 104,857,600 |
| Down output FP32 | 40960 x 2560 | 419,430,400 |
| Weighted down output FP32 | 40960 x 2560 | 419,430,400 |

These lifetimes overlap, not all at once; the measured allocated-byte
peak increase is 1,337,489,408. A 512-token/rank chunk cuts these variable
MoE shapes fourfold. This cap changes prefill scheduling and is not a
claim that prefill latency/throughput improves.

Both matched arms are eager: graph-pool bytes are zero. Independently,
C7b's graph serve reports 0.26/0.25 GiB estimated and 0.03/0.02 GiB in
the later real capture (`c7b-graph-serve.log:196`). The difference cannot
automatically be reclaimed: first-capture initialization may stay live
after the throwaway pool is released. The observer records that retained
delta when graph profiling is enabled.

VERDICT -> current and fixed reach readiness, identity `hotschmoe-dd`
then the detailed IQ3_XXS alias, and exact greedy `323`/`667`. Fixed also
passes two concurrent 2047-token prefills plus one generated token each,
exercising workspace reuse with the allocated KV pool. No speed claim.
Current server exited cleanly at 16:36:41. Fixed server exited cleanly;
post-health is HEALTHY on both cards at 16:56:58. No reset.

CONFIG -> one-card allocator/API probes, selected using `xpu-run --card N`
and its ZE affinity mask. About 15-30 seconds each after lease acquisition.
First probe queued behind C7b's pair serve; H1e retains pair priority.

COMMAND -> `memory/allocator_probe.py`, `memory/info_probe.py`.
Logs `x2/allocator-probe.log`, `allocator-ur-off.log`, `info-probe.log`.
Health before/after: HEALTHY; no crash/reset.

RESULT -> same settled allocation lifecycle, native vs vLLM API:

| Stage (card 1) | Native free | vLLM free | Native total | vLLM total |
|---|---:|---:|---:|---:|
| Baseline | 33,929,830,400 | 32,217,714,688 | 34,242,297,856 | 32,530,182,144 |
| 64 MiB temporary | 33,862,721,536 | 32,150,605,824 | 34,242,297,856 | 32,530,182,144 |
| 2 MiB pin, 64 MiB reserved | 33,862,721,536 | 32,150,605,824 | 34,242,297,856 | 32,530,182,144 |
| All torch storage released | 33,862,721,536 | 32,150,605,824 | 34,242,297,856 | 32,530,182,144 |

VERDICT -> the APIs track the same device-used bytes but expose totals
and free values differing by 1,712,115,712 bytes (about 5%). At utilization
0.98 this changes requested memory by 1,677,873,397 bytes/card, without
changing the measured profile consumption delta. Native accounting is an
opt-in candidate; the model runs below must validate allocation behavior.
Released torch storage also remains visible in the driver count, so that
residual is not automatically attributable to live oneDNN workspaces.

The detached X2 window controller waited outside GPU leases for C7b then
H1e. Measured arms: current eager DP+EP BF16 at batch 2048; fixes at the
same batch; fixes plus FP8/capped batch 512 at maximum length 200K; fixed
EP sidecar BF16 at maximum length 200K with a 2048-token exchange cap.
All use model identity checks,
two deterministic greedy probes, an 8 GiB host-memory abort, health,
compiled collective health, and teardown. No speed/stability/FP8 quality
promotion follows from these short probes.

CONFIG -> target DP+EP arm: fixed storage/accounting, eager, batch 512,
max length 200000, two sequences/rank, utilization 0.98, Triton attention,
`--kv-cache-dtype fp8_per_token_head`. This is a different backend/batch
from the matched BF16 comparison above. A prior attempt only exited in
CLI parsing because `--calculate-kv-scales` was removed upstream; it did
not load a model. The corrected command omits that obsolete option.

COMMAND -> `memory/run_window.sh`, raw `x2/target-fp8_per_token_head/`.

RESULT -> READY at 17:20:04. Exact KV budgets: 5,365,499,003 / 5,387,396,219
bytes. Actual block size 3136; page 3,261,440 bytes/layer; 137 shared pool
blocks/rank, or 5,361,807,360 pool bytes. Maximum one-stream capacity is
407,680 tokens/card after null/fixed-state reservation. Configured maximum
is 200000. Owned storage grows by exactly the pool bytes; total live
Torch allocations grow by 5,363,248,640, leaving 1,441,280 bytes of runner
state/rounding beyond the pool. This run includes X1's later compact PLE
ring implementation, so it must not be used to estimate old PLE tables.
Dynamic scale payload is included in each 520-byte/head token slot.

Both embedding sample checks pass. Concurrent 2047-token prefills plus one
output token pass. Greedy 23*29 returns 667, but 17*19 returns corrupt
text instead of 323. Native free memory immediately after KV initialization
is only 103,051,264 / 483,037,184 bytes, despite readiness: utilization
0.98 is an aggressive capacity experiment. Driver/context allocations
already present at the initial snapshot are outside vLLM's consumed delta.

VERDICT -> reject this FP8 mode for serving. A capacity fit does not
establish numerical correctness; the failure's root cause is unassigned.
Next diagnostic is a short BF16/Triton
control and layerwise K/V scale/readback comparison against BF16, followed
by fixing the first divergent operation. Estimate: one CPU
inspection stage and one short leased numeric probe before a new serve.
No claim of 200K quality, completed 200K prefill, or latency improvement.
Controlled teardown completed; post-health HEALTHY on both cards at
17:24:47, with no reset.

## EP sidecar: measured 200K configuration

CONFIG -> world size 1; card 0 trunk and 138 experts/layer, card 1
374 experts/layer. BF16 KV, eager, max length 200000, two sequences,
batch/exchange cap 2048, utilization 0.98. C12 MoE SHA256
`b3c15e14ba5811d90dfe8a65e4ccc1353fd67b3f44e4180810fac22b4ed49979`;
H1e EP binary `2e22f57ea997a5c28d7256a79adf9ea00d28efdf923a001a6928ff7fad7e21a0`.
Kernel 7.1.0-070100-generic. Full identity in `x2/final-identity.json`.

COMMAND -> `memory/run_window.sh` sidecar; `x2/sidecar/107-*.json`.

RESULT -> exact bytes:

| Item | Fixed card 0 | Fixed card 1 |
|---|---:|---:|
| Model-owned storage | 15,710,974,104 | 31,346,585,600 |
| All live Torch allocations after load | 15,764,337,664 | 31,392,000,000 |
| Retained Torch reservations after profile | 17,454,596,096 | 31,516,000,256 |
| Native non-Torch use after profile | 1,195,732,992 | 661,000,192 |
| Extra transient reservation headroom | 0 | N/A (no independent profile) |
| vLLM non-KV budget (card 0 worker) | 18,245,832,704 | N/A |
| Available KV budget | 15,311,619,195 | 0, by placement |
| Allocated KV pool | 15,296,102,400 | 0 |
| Native free immediately after KV init | 294,227,968 | 2,065,235,968 |
| Pool ceiling, two equally long streams | 300,800 tokens/stream | N/A |

Card 1's non-Torch row is `total - free - reserved` from the same
post-profile snapshot; it contains sidecar/native context costs, not KV.
There is one card-0 worker and no separate card-1 profiler budget. Graph
pools are zero in this eager run. Original allocated-peak headroom would
be 287,415,296 bytes on card 0; all of it fits inside retained reservations.

Actual BF16 manager block is 1600, page 3,276,800 bytes/layer, 389 pool
blocks, or 39,321,600 bytes/block. Two 200K requests need 263 blocks,
including null/fixed state: 10,341,580,800 bytes. Available budget exceeds
that by 4,970,038,395 bytes. Configured length remains 200000; 300800 is
the pool-only arithmetic ceiling, not a completed context-quality test.
The engine logs 593,893 tokens and 2.97x at 200000. That line is
`389 / (125+6) * 200000`: capacity normalized to the configured request
length. The 300800/stream ceiling instead solves
`1 + 2*(ceil(length/1600)+6) <= 389`. Both include GDN amortization;
the latter additionally retains a null block.

The old logical storage layout with this same split is:

| Model-owned bytes | Current layout, counterfactual | Fixed, measured |
|---|---:|---:|
| Sidecar card 0 | 16,766,236,824 | 15,710,974,104 |
| Sidecar card 1 | 31,346,585,600 | 31,346,585,600 |

Add the dense embedding and merged placeholders back to card 0. These
are exact counterfactual tensor bytes, not an unfixed sidecar allocator
measurement. H1e's separate graph serve
reports 12.06 GiB available KV with the uncapped 8192 exchange and original
memory API; that is a different configuration and only a rounded reference.
It already exceeds the 10,341,580,800-byte BF16 2x200K requirement.

Readiness 17:29:39; identity and greedy 323/667 pass. Embedding sample
comparison is bit exact. Two concurrent 2047-token prefills plus one
generated token each pass. Owned-storage growth is exactly the pool
bytes; total live Torch growth has another 615,936 bytes of runner state
and rounding. Server exited cleanly. Post-health is HEALTHY on both cards
at 17:37:17; no reset. Compiled collective post-check passes at 17:40:05.

For a later graph/MTP qualification, cap card-0 KV explicitly near
11,000,000,000 bytes rather than allocating the entire measured budget.
That still supplies at least two 200K streams in this grouping and leaves
about 4.3 GB more runtime headroom than this capacity experiment. Recheck
group/state requirements when MTP or checkpoint settings change.

## DP BF16: cap the profile batch instead of using FP8

CONFIG -> fixed DP=2+EP, TP=1, BF16/default FlashAttention, eager,
max length 200000, two sequences/rank, batch 512, utilization 0.98.
Same C12 binary; compact PLE ring; no QSA/MTP or graph capture.

COMMAND -> target-only copy `x2/run-bf16-target.sh`. Reproduce with
`X2_START_ARM=target X2_STOP_ARM=target X2_TARGET_DTYPE=auto bash
vllm/flashnext/memory/run_window.sh`. Raw `x2/target-auto/`; workers
423/card 0 and 405/card 1.

RESULT -> exact bytes:

| Item | Card 0 | Card 1 |
|---|---:|---:|
| Model-owned storage | 25,600,975,000 | 25,600,975,000 |
| All live Torch allocations after load/profile | 25,681,411,584 | 25,681,411,584 |
| Retained Torch reservations after profile | 27,103,592,448 | 27,103,592,448 |
| Non-Torch increase since baseline | 1,086,271,488 | 1,064,378,368 |
| Driver-consumed delta | 28,189,863,936 | 28,167,970,816 |
| Extra transient reservation headroom | 2,097,152 | 2,097,152 |
| Total non-KV budget | 28,191,961,088 | 28,170,067,968 |
| Requested budget | 33,557,451,899 | 33,557,451,899 |
| Available KV budget | 5,365,490,811 | 5,387,383,931 |
| Actual pool blocks, 39,321,600 bytes/block | 136 | 137 |
| Allocated pool bytes | 5,347,737,600 | 5,387,059,200 |
| Native free immediately after KV initialization | 117,747,712 | 457,842,688 |
| Pool ceiling, one long stream/card | 206,400 tokens | 208,000 tokens |

The profile's original allocated-peak increase is 336,241,664 bytes;
only 2,097,152 bytes require additional reservations. This correction
is 334,144,512 bytes, distinct from the smaller matched batch-2048
correction. The batch cap also reduces retained native/runtime use.
Compared with fixed batch 2048, KV budget increases by
1,678,217,216 / 1,678,204,928 bytes. Do not label all of that a profiler
bug: the capped workload has a genuinely smaller workspace requirement.

Owned storage grows by exactly each pool's bytes at KV initialization.
The required 132 blocks for one 200K stream/card leave only 4/5 blocks
(157,286,400 / 196,608,000 bytes) of allocated pool margin. The configured
maximum stays 200000. Two simultaneous 200K requests on the same DP rank
do not fit; concurrency 2 here means one stream on each rank.

VERDICT -> READY 17:50:28, correct identity and greedy 323/667, sampled
embedding lookups bit exact on both ranks. Two concurrent 2047-token
prefills plus one generated token each pass. Server exited cleanly;
post-health HEALTHY on both cards at 17:53:11. Final compiled collective
passes, and the controller closes at 17:58:02. Raw final check is preserved
at `x2/target-auto/collective-after.txt`.
BF16 can meet the pool budget without FP8, but graph
estimates, MTP state/weights, and long-context execution must be requalified.

## Capacity comparison

All ceilings below invert the measured grouped pool, reserve a null block
and six GDN blocks/live stream, and assume no MTP/checkpoint extras. They
are memory ceilings, not completed long-prefill or quality results.

| Layout | KV format / batch | Card 0 KV budget | Card 1 KV budget | Pool ceiling at total concurrency 2 |
|---|---|---:|---:|---|
| Current DP+EP | BF16 / 2048 | 960,932,742 | 982,829,958 | 27,200/card |
| Fixed DP+EP | BF16 / 2048 | 3,687,273,595 | 3,709,179,003 | 137,600 / 139,200 |
| Fixed DP+EP | BF16 / 512 | 5,365,490,811 | 5,387,383,931 | 206,400 / 208,000 |
| Rejected DP+EP candidate | FP8 token/head / 512 | 5,365,499,003 | 5,387,396,219 | 407,680/card; incoherent |
| Fixed sidecar 138/374 | BF16 / 2048 | 15,311,619,195 | 0 (no KV) | 300,800/stream on card 0 |

For the unfixed sidecar, exact counterfactual model bytes are given above;
no matched unfixed allocator/KV measurement was taken. H1e's rounded
12.06 GiB graph result is deliberately kept separate from these exact
eager byte measurements. Its graph/exchange/binary configuration differs.

Evidence limitation: the target-only controller reused
`x2/collective-after.txt`, replacing the earlier sidecar collective log.
The sidecar pass and 17:40:05 close are independently recorded in
`JOURNAL.md` and `reports/1G2_audit.md` F176. Per-card sidecar health and
all model/profile/KV/request JSON remain separate and intact. The harness
now stores future final collective logs under each arm's directory.

The dequant arena change has CPU forward/ownership tests; the measured
GPU serves use the legacy dispatch and do not exercise that arena. A new
dequant-prefill dispatch needs its own GPU numeric/peak check. Native
allocations are counted exactly as a combined measured residual; exclusive
oneDNN versus communication/driver ownership needs allocation-callsite
tracing. No part of that residual was optimistically removed.

Final close: all X2 controllers, watchdogs and containers have exited.
No host-RAM abort or reset occurred. Pre-readiness samples across the five
measured arms stayed above 40,678,368 KiB MemAvailable; the supplemental
watchdog covered request verification in the last arms. Syntax, ASCII,
model-registry cross-check, patch consistency and whitespace checks pass.
No X2 git commit/push or DD service action was performed.
