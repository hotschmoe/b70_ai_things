# Strata night-final-v3 feature inventory

Task 03. CPU-only read of the daily driver and the SYCL tree. No GPU, no
server change.

Source of the line numbers: `/mnt/vm_8tb/b70/build/strata-ab/src-ep`, branch
`night-20261016`, worktree HEAD `e753e9cb` (read 2026-10-19). The daily
driver binary is `releases/strata-night-20261016`. The night lab pins the
v3 engine to commit `1c360526`. This worktree is four commits later. Those
commits add default-off kernels (`STRATA_NIGHT_IQ3S_LO`, `STRATA_NIGHT_GDN2`,
`STRATA_NIGHT_MTP_SELECT`, `STRATA_NIGHT_MTP_HCQ8` / `STRATA_NIGHT_MTP_ROUTE`).
None of them is set in `strata/flash-next/dd/hotschmoe-dd.json`. Line numbers
below are the worktree that was read.

Config: `strata/flash-next/dd/hotschmoe-dd.json` (night-final-v3). Evidence:
`docs/20261016_night_lab.md`, `docs/20261014_flashnext_strata_optimization.md`,
`docs/20261013_flashnext_goals.md`, and the stage notes cited inline.
vLLM analogues are architectural (expert parallel, FusedMoE, MTP/EAGLE
spec decode, KV connector, CUDA/XPU graph, multimodal preprocessor). This
task did not check out upstream vLLM, so the analogue and the S/M/L rating
are inferred from that architecture plus the local XPU notes in `AGENTS.md`.

Difficulty: S = existing vLLM feature or a thin wrapper. M = out-of-tree
plugin or custom op, no worker-model rewrite. L = core model runner,
scheduler, cache engine, or the two-process execution model.

## Scoreboard

Fixed-prompt night lab (512-token coding replies) unless noted.

| Target | Night start (EP + batch MTP depth 2) | night-final-v3 | Status |
|---|---:|---:|---|
| Prefill 2000+ tok/s | 4K 1138, 32K 1568 | 4K 2204/2202, 32K median 2151 | met at 4K and 32K |
| Decode 150+ tok/s per stream | C1 91.1, C2 51.2 (engine 53.0) | C1 141.4, C2 92.1 (engine 93.1) | C1 at 94% of 150; C2 at ~92 |
| 200k x 2 | configured | requalified, fresh 200k prefill 1563-1636 tok/s | met |
| RAM restore, no recompute | S30 on an earlier binary | restores 410.7-422.3 ms, 3.16-3.17 GB, replay 0 | met |

Deploy note in `strata/flash-next/dd/README.md`: a 2048-token code reply
measured 152.6 tok/s on C1. That is a different prompt than the 512-token
table. Quality gates in the night lab stayed inside the rounding band
(decode NLL 3.1974 reference vs 3.1920 final; prefill-path NLL 2.5142 vs
2.4771).

Layer split, same morning binary, is the topology control: C1 81.3 and
32K prefill 2044, and batch MTP refuses a layer split. EP is the decode
topology. The prefill goal was met later, on EP, by the reduction plus
the night prefill kernels.

## Ranked by contribution to the four targets

Deltas are the isolated A/B when the lab printed one. Stacked stages are
marked. A feature can be large because it is the floor the later gains
sit on.

| Rank | Feature | What it moves | Best cited delta | Port | Where |
|---:|---|---|---|---|---|
| 1 | Two-card expert parallel, all experts resident | decode floor, and the only topology that also runs batch MTP | S3 coding decode 91.9; night EP C1 91.1 vs layer-split C1 81.3 | L, core | both |
| 2 | EP prefill reduce + push | prefill, about 2x before any night kernel | 32K 575.6 to 1106; 4K 569.7 to 1037 | L runtime, M kernels | both |
| 3 | KREG small-batch MMVQ | largest single decode jump | C1 107.2 to 131.7; C2 63.9 to 73.1 | M, custom op | OOT |
| 4 | ISA IQ expert decode, 2-row lanes | next decode jump; experts were arithmetic-bound | to C1 139.3, C2 76.9 (stacked on KREG) | M, custom op | OOT |
| 5 | Grouped XMX expert prefill | prefill, host launch and expert GEMM | 4K 1342 to 1507, 32K 1708 to 1738; with XMX attn 1804 / 2002-2047 | M kernel, L overlap | OOT op, core schedule |
| 6 | XMX union QSA prefill attention | the step that crosses 2000 at 32K | after grouped: 32K 1738 to 1959-2009; mode 2 later ~2120 to ~2245 | M/L, custom attn | OOT if the runner can call it |
| 7 | INT8 KV | 200k x 2 fits; XMX prefill attn requires it | S3: 2.82 GiB per 200k slot on the primary. No tok/s A/B | L cache + M kernel | core cache, OOT quant op |
| 8 | MTP spec 4, batch depth 2, pair head | C2 decode multiplier | spec4 vs spec3 coding 91.9 vs 85.5; depth2 +6.3% aggregate; pair head +4.5% aggregate | L for this batching | core spec worker; head GEMV is OOT |
| 9 | Conversation cache, park move, D2D admit | the RAM-restore target | replay 0; restore ~411-422 ms; admission ~380 ms to ~70 ms | L | KV connector plugin if hybrid state is exposed, else core |
| 10 | HC v2 Q8 read | dense decode bytes | same-binary C1 101.4 to 106.4, C2 56.6 to 60.6 | M | OOT op, once the model has hyper-connections |
| 11 | Batched verify QSA attention | C2 attention | 268 to 61 us/layer at 1K; 743 to 228 us at 64K; serving C2 60.6 to 64.7 | M | OOT attn op |
| 12 | Prefill overlap and dequant (halves, early, f16 sum, DQ) | prefill margin above 2000 | halves alone 4K 1352 to 1449; DQ stack 4K ~2070 to ~2180 | M kernels, L schedule | both |
| 13 | Shared-expert placement in the verify graph | launch tax and C2 | SH_LATE C1 93.3 to 96.4; fused shared expert C2 85.7 to 88.7 | M | core graph if the second queue is inside vLLM; op is OOT |
| 14 | EP decode psum, card0 plan, 2-row lanes | small C2, and the lane shape IQ needs | C2 55.7 to 57.4; C1 inside noise | L | core EP |
| 15 | GDN commit, slot fusion, GRF | about 1% and fewer launches | commit layers +1% C2; slot fusion C2 76.9 to 77.3 | M | OOT, if GDN state layout matches |
| 16 | Draft-vocab subset head | small C1 | C1 93.3 to 94.0-97.2 (40525 tokens, 68 MiB) | M | OOT head |
| 17 | Retained ~1% opt-ins and VRAM accounting | prefill residency and a point of C2 | see the env appendix | S to M | mostly flags on top of 1-8 |
| 18 | SYCL graph capture | how every window runs | eager tiny kernel 3.3 us, replay 0.55 us; captures once per shape | M/L | vLLM piecewise/XPU graph, not a new op |
| 19 | CPU vision | image input | on since 2026-10-16; no tok/s claim | S/M | OOT preprocessor |

## 1. Expert parallelism across two cards

Card 0 runs every layer (attention, dense, its own experts). Card 1 holds
the experts card 0 does not, and computes those rows of the same window.
A device flag carries the epoch, because a host value would freeze inside
a captured graph. Contract: `sycl/include/strata/kernels/ep_kernels.hpp:1`
and `:14` (`ep_push`), `:21` (`ep_send_rows`), `:18` (`ep_wait`). CLI:
`sycl/src/program/generate.cpp:552` and the parser at `:1973`. The flag
requires `--stream-experts` and every expert resident on one of the two
cards (`generate.cpp:915`, guard at `:2277`).

Placement measured at S3 (`docs/20261014_flashnext_strata_optimization.md`):
190 experts/layer on card 0 (9120 experts, 14.83 GiB) and 322/layer on
card 1 (15456 experts, 25.13 GiB). All 24576 experts resident. Coding
decode 91.9 tok/s, 4K and 32K prefill both ~578 tok/s. The night-start
EP baseline is C1 91.1 / C2 51.2 with prefill already partly recovered
(4K 1138, 32K 1568). Layer split on that same binary: C1 81.3, 32K
prefill 2044, and `--pipeline-windows 2` crashes on SYCL at the first
verify (`UR_RESULT_ERROR_UNINITIALIZED`).

Night decode additions on this path, all default-off in source and set
in the DD:

- `STRATA_NIGHT_EP_PSUM=1`. Card 1 returns one weighted row per token
  instead of one row per expert entry. `ep_kernels.hpp:35`, `ep_send_psum`
  at `:39`, `ep_combine_psum` at `:46`. Latch `sycl/src/core/verify.cpp:84`.
  FP32 reorder, MoE output agrees to rounding. C2 55.7 to 57.4; C1 97.6
  is inside the noise band (night lab N7).
- `STRATA_NIGHT_EP_PLAN1=1`. Card 0 builds card 1's expert plan and pushes
  it with the rows. `ep_kernels.hpp:99` (`resident_plan_ep_pair`), latch
  `verify.cpp:94`. Bitwise. Same N7 stack.
- `STRATA_NIGHT_EP_ROWS=2`. Two rows per expert lane. Bench string in
  `sycl/src/kernels/ep_night_bench.cpp:221`. Required for the IQ win:
  the night IQ comment says IQ2_XS / IQ3_XXS / IQ3_S / IQ2_S gate-up are
  faster only at 2 rows (`iq_kernels.dp.cpp:5013`).
- `STRATA_NIGHT_SH_LATE=1`. Shared expert moves onto the main queue after
  card 0's own experts and before the peer wait, so the verify graph stays
  linear. `verify.cpp:888`. Profiled event resets were ~2226 per replay
  (4.3 ms tail per C1 window) with the forked second queue. Result: C1
  93.3 to 96.4/95.6, C2 53.6 to 55.0/54.9 (N5). Later profile: event
  resets per C1 window 1934 to 153.
- `STRATA_NIGHT_EP_SHARE0` is not set. The hot-expert share knob had no
  gain (N19).

vLLM analogue: `--enable-expert-parallel` plus an all-to-all MoE backend
(naive, DeepEP, FlashInfer). That analogue splits experts but still runs
attention and dense on every rank. Strata's card 1 is an expert engine
with a P2P inbox, and the combine is a custom reduction inside one
captured graph. That execution shape is a core worker change (L). The
push/reduce kernels themselves can be a custom collective (M) once two
ranks exist. A stock TP=2 all-reduce of full hidden states is a different
algorithm and, on this host, the older vLLM TP path has a separate
oneCCL queue-handoff failure (`AGENTS.md`).

## 2. Expert streaming and the VRAM cache

`--stream-experts` (`generate.cpp:947`, parser `:1987`): no host arena.
Experts are read from the GGUF through the native pack into a VRAM cache
that is required to hold every expert. `--expert-cache auto`
(`generate.cpp:590`) sizes that cache from free VRAM after weights, KV
and the reserve. `--expert-profile` (`generate.cpp:564`, parser `:1979`)
is `data/expert-profile-sycl.bin`, a frequency ranking from
`tools/make_profile.py`. `--vram-reserve-mib 1024` leaves 1024 MiB for
graphs and scratch (`generate.cpp:591`).

On this model the cache is not a hit-rate optimizer. S3, S38 and the
night lab all record all 24576 experts resident. The stream path is what
EP requires, and it avoids a host-mapped expert arena (the help text
cites a 34 GB mmap that measured 71.97 vs 34.78 ms/token cold vs warm
on another machine). No DD miss-rate number exists, because a full cache
does not miss.

`STRATA_OWNED_PRICE=exact` (`generate.cpp:4470`) prices the prompt
buffers at the real allocation. The 0.1.39 formula is `160 + chunk *
680 / 1024`, about 16 GiB at a 24576 chunk against about 2.4 GiB really
allocated. Exact pricing returns that gap to the expert cache. With
`--no-prefill-borrow` and `--prefill 12288` this is what keeps the
12288-token workspace and a full expert cache at the same time.

vLLM analogue: weight loading, not a runtime cache. If the port loads a
sharded non-GGUF checkpoint, this whole mechanism stays behind. Porting
the GGUF streamer is L and does not buy decode once experts fit. S, and
only as a loader, if the campaign converts weights first.

## 3. EP prefill push, reduce, and the night prefill stack

Decode EP and prefill EP are different kernels.

- `STRATA_EP_PREFILL_REDUCE=1` (`sycl/src/prefill/prefill.cpp:2031`).
  Card 1 returns one FP32 weighted sum per token. `ep_prefill_reduce` /
  `ep_prefill_finish` at `ep_kernels.hpp:59` and `:63`. S6
  (`docs/20261014_flashnext_strata_optimization.md`): 26 numerical cases,
  max abs error 9.54e-7 vs router-order FP32. Serving medians, three
  fresh prompts: 4K 569.7 to 1037, 32K 575.6 to 1106. At 4K the peer
  return is about 40 MiB per layer instead of about 244 MiB, and the
  combine bucket falls from about 3698 ms to 622 ms. Decode unchanged
  to within text noise. This is the single largest prefill measurement
  in the campaign.
- `STRATA_EP_PREFILL_PUSH=1` (`prefill.cpp:2033`). The reduced row is
  pushed into the primary's buffer. On whenever reduce is on.
- `STRATA_EP_PREFILL_GATHER_ALIAS=1` (`prefill.cpp:2564`). One gather of
  peer inputs into the FP16 view of the existing FP32 output buffer,
  experts run in reverse offset order. S38
  (`docs/20261015_strata_peer_gather_serving.md`): hashes match; 4K
  prefill 1303.0 to 1326.2 (+1.78%), 32K 1546.1 to 1556.1 (+0.65%), C2
  aggregate 104.516 to 104.017. The night stack keeps it on.

Night prefill flags, all in `prefill.cpp` unless noted:

| Flag | Latch | Measured |
|---|---|---|
| `STRATA_NIGHT_PREFILL_EARLY=1` | `prefill.cpp:312` | Bitwise. Card 1 starts on a device flag; the host waits only for routing. With F16SUM below. |
| `STRATA_NIGHT_PREFILL_F16SUM=1` | `prefill.cpp:316`, kernel `ep_kernels.hpp:72` | FP16 per-token scaled return, MoE rel RMS 1.6e-4. Control 4K 1175/1288 and 32K 1640/1605; EARLY+F16SUM 1213/1342 and 1746/1711 (N11). |
| `STRATA_NIGHT_PREFILL_GROUPED=1` | `prefill.cpp:330`, kernel `sycl/include/strata/kernels/night_grouped.hpp:1` | One launch per projection for small experts. IQ blocks decode straight into 8x16x16 FP16 `joint_matrix`. 4K 1342 to 1507, 32K 1708 to 1738 (N17, second rep). |
| `STRATA_NIGHT_PREFILL_HALVES=1` | `prefill.cpp:334` | Chunk split into two halves, interleaved per layer so card 1's experts hide under card 0's attention. Alone: 4K 1352 to 1449, 32K ~1724 to ~1834. |
| `STRATA_NIGHT_PREFILL_DQ32=1` | `night_grouped.hpp:26` | Expert dequant, one work-item per 32 values. Bitwise vs `iq_dequant_expert_f16`. |
| `STRATA_NIGHT_PREFILL_DDQ=1` | `sycl/src/kernels/cuda/dequant_bf16.dp.cpp:220` | Dense FP16 dequant with coalesced stores. Same values. |
| `STRATA_NIGHT_PREFILL_DQCACHE=1` | `prefill.cpp:340` | Dense weights dequantized once for both halves. |
| DQ32+DDQ+DQCACHE together | N19 | Residual dump bitwise at 151 positions. 4K prefill ~2070 to ~2180. |

In-tree grouped XMX expert kernels and oneMKL `gemm_batch` did not pay
(N11, and the "what did not work" list). The grouped kernel that did pay
is the night one that decodes IQ blocks into `joint_matrix` with no FP16
weight copy in global memory.

vLLM analogue: a custom MoE prefill kernel (grouped GEMM) plus a custom
dispatch that overlaps expert compute on rank 1 with attention on rank 0.
The GEMM is an OOT op (M). The half-chunk pipeline and the doorbell are
core scheduling (L). Stock vLLM chunked prefill does not split one chunk
across two cards this way.

## 4. XMX prefill attention

`STRATA_NIGHT_PATTN_XMX=2`. Implementation:
`sycl/include/strata/kernels/qsa_prompt_attn_union.hpp:1`. Eight
neighbouring queries share one pass over the union of their selected
cells. K and V are staged in SLM as FP16 from INT8 code times the
64-value scale. Scores and the value product run on `joint_matrix`
(FP16 x FP16 to FP32, 8x16x16). A cell the query did not select is -inf,
so each query still attends only its own selection. Mode 2 is
`S^T = K Q^T` in 64-cell chunks (night lab N19). INT8 KV only; any other
KV layout returns false and the caller keeps the vector path.

`STRATA_PROMPT_ATTN_XMX=0` is a different, older kernel
(`sycl/include/strata/kernels/qsa_prompt_attn_xmx.hpp:8`): one query per
work-group, left off because it was slower than the fallback. The DD
uses the night union path, not this one.

Measured (N17, second rep, on top of the then-current prefill): base
4K/32K 1342/1708; grouped alone 1507/1738; XMX after grouped
1567/1959-2009; grouped+XMX **1804 / 2002-2047**. Mode 2 later: 32K
~2120 to ~2245. Kernel note: 1.3-2.8x per layer vs the vector union
path. Quality: XMX attention +0.002 nats/token on the fixed long-prefill
NLL gate. Not bitwise (FP16 products).

The vector path still has two older opt-ins that stay on:

- `STRATA_QSA_PREFILL_TILED=1` (`qsa_select.dp.cpp:1703`). Eight-query
  scorer tiles, only for a full long prefill (nq=256, active blocks
  4097..50001). S89: 32K native 1547.5 to 1559.2 tok/s, prompt time
  -0.55%. Short 4K stays on the old path.
- `STRATA_ATTN_VALUE_PIPELINE=1` (`qsa_decode_attn.dp.cpp:1013`). Staged
  prefill only, and only the exact shape n_q=32, cap=2051, 24/2/256,
  page 4. S93
  (`docs/20261015_strata_attention_value_serving.md`) failed to activate
  it in serving. The DD still sets it. Treat the speed contribution as
  unverified on the night-final binary.

vLLM analogue: a custom sparse-attention op, not FlashAttention. Qwen3-Next
QSA (indexer + top cells + gated delta-net on the other layers) has to
exist in the model runner before this kernel has a caller. The math is
an OOT op (M). Wiring the indexer, the cell ids, and the INT8 pool is L
if upstream vLLM's Qwen3-Next attention does not already expose that
selection.

## 5. KREG and wide MMVQ

Dense projections and the output head are small-batch GEMVs over GGUF
blocks. Two generations are both on.

Wide kernels (pre-night), `sycl/src/kernels/cuda/native_mmvq.dp.cpp`:

| Flag | Value | Line | Role |
|---|---|---|---|
| `STRATA_MMVQ_WIDE` | 1 | `:3674` | Q6_K wide kernel. Default on if unset. |
| `STRATA_MMVQ_WIDE_MIN` | 0 | `:5207` | Minimum rows before the wide kernel. 0 with A2. |
| `STRATA_MMVQ_WIDE_MAXC` | 8 | `:5208` | Maximum columns. 8 with A2. |
| `STRATA_MMVQ_WIDE_K` | 1 | `:2216` | Q4_K / Q5_K / IQ4_XS take the wide kernel. |
| `STRATA_MMVQ_WIDE_K_RPW` | 1 | `:2081` | Rows per sub-group of that kernel. |
| `STRATA_MMVQ_WIDE_RPW` | 1 | `:1840` | Q6_K rows per sub-group. |
| `STRATA_MMVQ_A2` | 1 | `:1539` | 16-byte aligned Q6_K loads. Default on. |
| `STRATA_Q6_CENTER` | 1 | `:1544` | Centered Q6_K on two shapes (6144x2560, 2560x10240). |
| `STRATA_MMVQ_PF` | 0 | `:1835` | Software prefetch distance. Off. |
| `STRATA_MMVQ_WIDE_32` | unset | `:3644` | Q8_0 / IQ4_NL wide path stays at its default (on). |

S87 (`docs/20261013_flashnext_goals.md`): Q6 center C2 aggregate 103.088
to 104.279 (+1.16%), one fixed-order sequence. The wide family is the
floor KREG replaced for small batches; there is no separate "wide vs
scalar" serving number on the night binary.

KREG, `native_mmvq.dp.cpp:2236`. K is split across the work-group,
activations sit in registers, R-row tiles, sub-group butterfly, SLM sum.
The wide kernels re-read every column's activations per row and were
issue-bound (night lab: ~850 instructions per 4-block step, 468 mov;
82 us L2-resident vs 94 us DRAM).

- `STRATA_NIGHT_MMVQ=1` (`:2247`). Q4_K / Q5_K / Q6_K / IQ4_XS, 1-8
  columns, head included. Microbench: head nc1 1062 to 760 us, nc6 2662
  to 1166 us; Q4_K/Q5_K/Q6_K at 76-94% of roofline for nc<=3. Serving,
  same binary: **C1 107.2 to 131.7, C2 63.9 to 73.1** (engine 66.7 to
  76.6). Largest decode delta in the night lab (N12). FP32 order
  changes. NLL: MMVQ alone +0.013 nats (better) on the small gate.
- `STRATA_NIGHT_MMVQ2=1` with `STRATA_NIGHT_MMVQ2_MASK=2` (`:2256`,
  mask at `:2274`). Mask bits: bit0 pipelined KREG and 9-16 columns,
  bit1 IQ3_S, bit2 IQ4_NL. Mask 2 keeps only the IQ3_S kernel, and only
  at >= 5 columns (`STRATA_NIGHT_IQ3S_MINC` in the lab text). Printed
  A/B (C1 / C2): round 1 only 139.0 / 78.9; pipelined+9-16 col 139.1 /
  76.6; IQ3_S 134.1 / 79.8; IQ4_NL 136.8 / 78.9; all 137.5 / 76.5.
  Microbench wins did not survive as a blanket serving win. The DD
  still sets mask 2. Do not treat mask 2 as a measured C1 win.
- `STRATA_NIGHT_BF16=1`. KREG BF16 router and indexer GEMVs
  (`sycl/src/kernels/cuda/native_bf16.dp.cpp`, called from
  `verify.cpp:1181`). Bitwise. Serving +0.4% C1, called neutral (N19).
- `STRATA_NIGHT_IQ3S_LO` is absent. The later commit's 1-4 column IQ3_S
  kernel was bitwise in the bench and not bitwise in the engine, C1
  141.9 vs 141.6, C2 flat. Left off (night lab, next-session item 1).

Left off, with their own serving notes:

- `STRATA_Q6_PACKED=0` (`native_mmvq.dp.cpp:4459`). Packed Q6_K head.
  Not promoted.
- `STRATA_Q6_A4=0` (`:1750`). S6-A4 serving
  (`docs/20261016_strata_q6_a4_serving.md`): aggregate 102.89 to 103.33
  (+0.43%), kept off.
- `STRATA_HEAD_Q5K_RPW2=0` (`:5464`). S55
  (`docs/20261015_strata_head_rpw_serving.md`): wall difference inside
  the control range. Off.
- `STRATA_Q5_HEAD_RUNTIME_ONES=0` and `_C6=0` (`:2149`, `:2175`). S130
  (`docs/20261016_strata_q5_runtime_ones_serving.md`): +0.66% aggregate
  on an earlier full-vocab head. The DD uses the English draft-vocab
  subset head and sets both to 0. No v3 rerun is in the night lab.

vLLM analogue: a quantized GEMV custom op (the Marlin / Machete / Cutlass
role, for GGUF block types on XPU). Strong OOT candidate (M). It is worth
porting only if the served weights stay in these GGUF types. A repack to
INT4/FP8 groupwise GEMM would replace this family instead of wrapping it.

## 6. ISA IQ decode

`STRATA_NIGHT_IQ=1`, `sycl/src/kernels/cuda/iq_kernels.dp.cpp:2409` and
the latch at `:5011`. Same integers out of every 32-weight part as the
previous grouped expert kernels, so outputs are bitwise. Changes: sign
masks from a 2-4 KB table instead of popcount/SWAR, Q2_0 fields by
shifts instead of the DPCT byte-permute emulation (476 instructions down
to 142-203 on the Q2_0 down projection), IQ4_NL by a position table,
pointer-walk loops, IQ2_XXS multi-entry. Default formats
`16,17,18,20,21,22,42` (IQ2_XXS, IQ2_XS, IQ2_S, IQ4_NL, IQ3_S, IQ3_XXS,
Q2_0). `STRATA_NIGHT_IQ_ALU` defaults on (ALU sign masks).
`STRATA_NIGHT_IQ4NL_TAB` defaults to the position table.

Production kernels before this were 280-480 instructions per 32-weight
step with DP4A at 2-3% of the ISA. N7's finding: routed experts are
limited by IQ decode arithmetic per entry, so every extra verify row
pays a full decode of its experts.

Serving (N13, best3 = KREG + IQ + everything earlier): **C1 139.3, C2
76.9/stream (engine 80.7), C4 40.2**. Prefill unchanged at that stage
(4K 1219, 32K 1745). Coherence passed. The prefill grouped kernel later
reuses the same decode style inside `joint_matrix` (rank 5).

`STRATA_IQ2XS_MULTI=1` (`iq_kernels.dp.cpp:5138`) is an older cross-entry
reuse for IQ2_XS. S100
(`docs/20261015_strata_routed_iq2_reuse.md`) measured a small difference
and kept the flag off. The night N2 stack turned it on together with
several other flags (C2 51.2 to 53.6, +5% for the whole stack). No
isolated v3 A/B. Leave it as "on in the DD, isolated contribution
unmeasured".

`STRATA_GU_LANES=8` (`iq_kernels.dp.cpp:4942`) sets lanes per row of the
grouped expert kernels. `STRATA_EXPERT_SPLIT=0` (`:3656`) keeps the
port's grouped kernels; `1` would launch upstream's multi kernels.
No isolated tok/s number for `GU_LANES=8` was found in the cited docs.

vLLM analogue: the dequant inside a fused MoE kernel. OOT custom op (M),
same caveat as MMVQ: the layout is GGUF IQ blocks. Calling it from
vLLM's FusedMoE modular kernel is M if the modular-kernel ABI accepts
an XPU op. Teaching the weight loader to keep IQ3_XXS blocks is L.

## 7. Hyper-connection Q8 (HC v2)

Qwen3-Next hyper-connections are BF16 in the ISTA GGUF. Per forward,
dense traffic is 3.48 GiB and the HC projections are 1.17 GiB of that
(night lab byte budget).

- `STRATA_HC_Q8=1` (`sycl/src/core/native_dense.cpp:34`) quantizes the
  BF16 down/up projections and the final mixer to Q8_0 at load, ggml
  reference rounding, 0.63 GiB. N1 reported C1 91.1 to 93.0. The lab
  then corrected that: the SYCL HC kernels did not read those copies
  (`fused_gr.dp.cpp` comment "Q8_0 reads are not built"). The +2% was
  an expert re-split (139/373) or noise. The flag stays on because the
  night reader consumes its buffers.
- `STRATA_NIGHT_HC=q8` (`native_dense.cpp:351`, reader
  `sycl/src/kernels/cuda/fused_gr_night.inc:1`). Two kernels replace
  four: norm+down+reduce fused, then up. Q8 weights, split layout.
  Same-binary A/B (N8): HC off C1 101.4 / C2 56.6; v2 bf16 105.4 / 58.1;
  **v2 q8 106.4 / 60.6 (engine 63.0)**. Synthetic rel RMS 7e-4. NLL for
  bf16 v2 +0.011 nats; the full best3 stack (including q8) was 3.178 vs
  reference 3.197.

A persistent single HC kernel was slower: the grid barrier cost more
than the kernel boundary, and cooperative launch fails inside SYCL
graphs (night lab, what did not work). Drafter HC Q8
(`STRATA_NIGHT_MTP_HCQ8`) doubled TTFT and cost 5% at C2. Off, and not
in the DD env.

vLLM analogue: a custom GEMV on the hyper-connection projections inside
the Qwen3-Next block. OOT op (M) once the model definition has those
tensors. The fusion (norm, down, reduce, up) is the part that paid;
a generic Q8 linear that still launches four kernels does not match
the measurement.

## 8. MTP: spec 4, batch depth 2, pair head, draft vocab

CLI, all in `sycl/src/program/generate.cpp` unless noted:

- `--spec 4` (`:620`). Verify window of 4 tokens: 1 accepted + up to 3
  drafts. `--mtp-max-t 4` caps the MTP window at 4 even if spec is
  larger. `--spec-min-p 0` disables the confidence cutoff. S4 used 0.5;
  the DD uses 0. No v3 A/B of min-p was found.
- `--mtp` points at the BF16 drafter runtime
  `.../mtp-bf16/rt`. Drafter class `sycl/src/core/mtp.cpp`, capture at
  `:1357` and `:1389`.
- `--suffix-draft 0` and `--lookup-chain 0` leave n-gram speculation
  off. MTP is the only drafter.
- `STRATA_BATCH_MTP=1` (`generate.cpp:4023`). Each batch slot verifies
  one MTP proposal per window. Needs `--batch` >= 2, which `--parallel 2`
  supplies.
- `STRATA_BATCH_MTP_DEPTH=2` (`generate.cpp:4027`,
  `include/strata/core/batch_depth.hpp`). Up to 2 chained proposals per
  slot. S21: matched C2 coding aggregate 93.064 to 98.901 (+6.27%).
  Night lab N20: depth 3 at C2 was worse (88.1). Depth 2 stays.
- `STRATA_BATCH_MTP_PAIR_HEAD=1` (`generate.cpp:4136`, class comment
  `sycl/include/strata/core/mtp.hpp:306`). The two slots' heads at one
  depth are one two-column GEMV. S29: C2 aggregate 97.881 to 102.279
  (+4.49%), all six outputs match. N6 on the night binary: C1 98.8, C2
  55.7, acceptance slightly lower (308/404 vs 317/388 per stream).
- `STRATA_BATCH_MTP_PAIR_GRAPH=1` (`mtp.cpp:2144`). Records the depth-2
  pair. Included in the N2 stack (+5% C2 for the whole stack). No
  isolated A/B in the night lab.
- `--mtp-draft-vocab data/draft_vocab_en.bin` (`generate.cpp:729`,
  `mtp.hpp:69`). 40525-token subset, 68 MiB head, instead of the full
  vocab head. Solo C1 94.0-97.2 vs 93.3 (N4). N6 is what made the pair
  path accept the subset. Small.
- `STRATA_MTP_PARALLEL_ARGMAX=1` (`mtp.cpp:71`). S114: aggregate
  103.074 to 103.674 (+0.58%) on an earlier binary.
- `STRATA_MTP_SHARED_BRANCH=0` (`mtp.cpp:1121`). Drafter shared expert
  runs inline, so the MTP graph stays linear. N20: C1 141.4, C2
  92.1/stream (engine 93.1), C1 text identical. This is the step from
  night-final-v2 to v3.
- `STRATA_NIGHT_MTP_SELECT` and `STRATA_NIGHT_MTP_ROUTE` are not in the
  DD. Select was bitwise and inside noise. Route fusion regressed.

S4, earlier EP binary, min-p 0.5: EP spec 3 coding 85.5 tok/s, EP spec 4
coding 91.9 (the S3 screen, acceptance 83.0% / 67.5% on that row). N20
on v3: `--parallel 1 --spec 6` reached C1 144.9 vs spec4 142.4; in the
2-slot config spec6 was +0.5% and spec7 was worse. The production
contract is spec 4 and batch depth 2. A single-user box can run spec 6.

Verify width is `kVerifyMaxT = 8` rows. C4 with depth 2 needs 4 slots x
3 rows = 12, so slots rotate two per window (N3, N10). A 12-row window
(`STRATA_NIGHT_VERIFY_T12`) worked and was slower (49.6 vs 51.2 per
stream). Not in the DD. C4 on the v3 config is a separate json (depth 1,
64k context): 51.3/stream.

vLLM analogue: speculative decoding, method `mtp` or `eagle`,
`num_speculative_tokens=4`. Upstream Qwen3-Next MTP covers the single
sequence case if that model is in the version you build (inferred, not
checked here). Batch depth 2 and the paired two-column head are not the
stock speculator. Those are L: the spec worker, the bonus-token sampler,
and the hybrid KV/GDN append all have to agree on a 2-slot window.
The draft-vocab GEMV is an OOT head op (M). `STRATA_BATCH_MTP_OWNER_QUEUES=0`
(`mtp.cpp:2151`) was -0.33% aggregate in S131
(`docs/20261016_strata_mtp_owner_queues_serving.md`). Leave it off.

## 9. INT8 KV, 200k context, conversation cache

INT8 KV. `--kv int8` (`generate.cpp:757`, applied at `:2338`,
`sycl/src/core/layer.cpp:182`). Codes are int8 with an fp16 scale per
64 values, half of fp16. Kernel: `sycl/src/kernels/cuda/kv_q8.dp.cpp:36`
(`kv_append_q8_kernel`). Group size constant `KV_Q8_GROUP` is 64
(`:26`). S3 startup: each 200k slot session is 2.82 GiB on the primary,
with spec 4. The XMX prefill attention refuses anything else. No
matched fp16-vs-int8 tok/s A/B is in the cited docs; the contribution
is capacity. `STRATA_KV_ROT=0` (`generate.cpp:2340`) would put those
INT8 codes through the Hadamard used by `--kv q4_0`. Off. The lab did
not promote q4 or k8v4 for this model.

200k. `--max-context 200000` (`generate.cpp:796`). `--no-prefill-borrow`
(`:611`) keeps the prefill workspace reserved for the session so a long
prompt does not borrow expert-cache slots. `--prefill 12288` is the
chunk. Together with INT8 KV, full expert residency, and exact owned
pricing, this is the capacity configuration. Fresh 200k prefill on v3
is 1563-1636 tok/s, below the 4K/32K rate and still the qualified
full-context path.

Conversation cache. `--conversation-cache-mib 32768` and
`--conversation-cache-slots 4` (`generate.cpp:684`, constructed at
`:7772`). Class: `include/strata/core/conversation_cache.hpp:226`.
A parked entry is the running state: QSA KV plus GDN recurrence state,
not just the KV pool. Snapshot copy:
`sycl/src/core/conversation_snapshot.cpp`. Park of a batch slot:
`generate.cpp:9402` and the call before an overwrite at `:12264`.

S30 (`docs/20261013_flashnext_goals.md`), earlier binary, 32 GiB / 4
entries: four distinct 199501-token prompts, four forced RAM returns
reuse 199839 held tokens, 31 new, replay 0. Restores 663-685 ms. Peak
native RSS 16.3 GiB, swap 0.

Night-final requal (`bench_context_four.py`, night lab): 
`correctness_passed`, `four_distinct`, `forced_ram_exchange_passed`.
Native full restores **410.7-422.3 ms**, payloads 3.16-3.17 GB, replay
0. Cached-return TTFT p95 2.99 s resident, 4.67 s after RAM restore.
The 2 s figure is a diagnostic, not a gate (`AGENTS.md`).

Two night flags cut admission, which is the wall-clock cost around
that restore:

- `STRATA_NIGHT_ADMIT_D2D=1` (`generate.cpp:9189`, copies in
  `conversation_snapshot.cpp:253` and `conversation_state.cpp:208`).
  Slot copies go device-to-device. The old path round-tripped ~120 MB
  of GDN state and per-layer KV through host memory with a sync per
  piece. Admission copy 230 ms to 1.0 ms (N15). Unit test: 4640 byte
  checks pass.
- `STRATA_NIGHT_PARK_MOVE=1` (`generate.cpp:7956`). Checkpoints move
  into the image instead of being copied. Each checkpoint is ~118 MB
  of GDN state. Park 190 ms to 49 ms (N18). The lab stacked this with
  `GLIBC_TUNABLES=glibc.malloc.hugetlb=1`. The two were not separated.
  Combined admission (park + copy) went from ~380 ms at night start to
  ~70 ms in the final table.

vLLM analogue: `kv_cache_dtype` for the INT8 pool (today the stock
dtypes are fp8 family; int8 group-64 is a custom cache layout, L in
the cache engine, M as the quant kernel). RAM parking is a KV
connector (`OffloadingConnector` / LMCache / NixlConnector). Those
connectors move KV blocks. This cache also moves GDN recurrent state
and the QSA indexer pool, and restore must skip recompute. That is L
unless the connector API already accepts arbitrary per-layer state
tensors, in which case an OOT connector can do the copy (M) on top of
a core change that exposes the state (still L for GDN).

## 10. GDN commit and the verify recurrence

Gated delta-net state is updated on the linear-attention layers.
`STRATA_GDN_COMMIT_LAYERS=1` (`verify.cpp:128`, call at `:3424`, kernel
`sycl/src/kernels/cuda/verify_kernels.dp.cpp:1857`). One launch commits
up to 3 rows across 36 layers. The kernel refuses to run unless
`STRATA_GDN_COMMIT_SPLIT=1`, `STRATA_GDN_SPLIT=0`, and `STRATA_DBG_GDN=0`
(the DD sets exactly that). N19: output text identical, C2 draft counts
shift by 1, **+1% C2**. The earlier S148 hold was this rounding, later
accepted as rounding.

Other GDN flags in the DD:

- `STRATA_NIGHT_GDN_SLOTS=1` (`verify_kernels.dp.cpp:2052`). One verify
  recurrence launch for every slot group. Bitwise. C2 76.9 to 77.3 (N14).
- `STRATA_NIGHT_GDN_AB=1` (`:226` and latch `:1909`). `gdn_ab_multi` uses
  one sub-group per (row, token). Bitwise. Part of the N18 stack that
  reached C2 85.7; not isolated.
- `STRATA_NIGHT_GDN_GRF=1` (`:2068`). 256-GRF recurrence, no spills, C1
  shape. Bitwise. Same N18 stack, not isolated.
- `STRATA_NIGHT_GDN2` is not in the DD. The later commit-from-deltas
  kernel (1553 to 485 instructions, at the 3 MB/layer state bandwidth
  floor) was bitwise and inside noise (C1 142.2 vs 141.6, C2 91.7 vs
  92.3). The lab's next lever, deferring the commit into the next
  window, is estimated at -0.25 ms per C1 window and is not built.

`STRATA_NIGHT_GDN_CHUNK` (chunkwise prefill in FP32) was 4.4x slower
than the scan. Off, not in the DD.

vLLM analogue: the GDN / linear-attention kernel inside Qwen3-Next.
Replacing the state update with this fused commit is an OOT op (M) if
the state tensor layout matches, and L if vLLM's GDN keeps a different
split of conv state vs recurrent state. CUDA-graph capture of the
recurrence is the same piecewise-graph problem as the rest of the layer.

## 11. Decode attention batching and the router

Night attention flags, all bitwise vs the original (75 hashes, N9):

| Flag | Line | What |
|---|---|---|
| `STRATA_NIGHT_ATTN_BATCH=1` | `verify.cpp:129` | A C2 verify window runs QSA attention per slot, not per row. |
| `STRATA_NIGHT_ATTN_VPIPE=1` | `qsa_decode_attn.dp.cpp:920` | Value-pipelined chunk kernel. |
| `STRATA_NIGHT_ATTN_MERGE_ACTIVE=1` | `qsa_decode_attn.dp.cpp:912` | Merge loops only over chunks the row actually used. |
| `STRATA_NIGHT_TOPK_SCAN=1` | `qsa_select.dp.cpp:2190` | Wide top-k digit search as a work-group scan. Same ids. |
| `STRATA_NIGHT_QSA_SCORES_T=1` | `qsa_select.dp.cpp:1727` | Block scores with transposed warp reductions. |

Kernel: C2 attention 268 to 61 us/layer at 1K context, 743 to 228 us at
64K. Serving with HC q8 already on: C1 107.6, C2 64.7 (engine 67.4).
The C1 move is about 1 tok/s; the C2 move is about 4 tok/s. Prefill
unchanged at that stage.

Router: `STRATA_NIGHT_TOPK10=1` (`sycl/src/kernels/cuda/native_router.dp.cpp:390`,
header `sycl/include/strata/kernels/native_router.hpp:8`). Top-10 without
dynamic indexing. Bitwise. Part of N18 (C1 140.1, C2 85.7 together with
GDN_AB, GDN_GRF, park move, and the prefill flags). A fused
router-GEMV-plus-top-10 kernel was slower and stays off.

vLLM analogue: custom decode attention op, plus the existing grouped
topk router. OOT (M) for the kernels. The "one launch per slot instead
of per row" part is how the spec-decode batch is laid out, so it rides
on the MTP work (rank 8).

## 12. Graph capture

`sycl/src/core/graph.cpp:56`. `CapturedGraph::begin` records the SYCL
queue (`sycl::ext::oneapi::experimental::command_graph`), finalizes an
executable graph, and replays it. Verify windows:
`verify.cpp:1973` (`capture_segments`), `:2157` (the main capture),
`:2277` (`capture_commit`), `:3256` (`capture_batch` for multi-slot
shapes). MTP graphs: `mtp.cpp:1357`, `:1375`, `:1389`. The night lab
says captures happen once per window shape, at load or first use, not
in steady state. Microbench: graph-replayed tiny kernel 0.55 us vs
eager 3.3 us. A forked second queue inside the verify graph is what
SH_LATE removed (rank 13); the runtime was appending thousands of event
resets per replay.

`STRATA_TRACE=1` (`generate.cpp:1292` and the per-step trace at `:8861`)
prints startup VRAM and one line per request step. It is observability.
The README says engine.log grows a few MB per hour of generation. No
speed claim.

vLLM analogue: CUDA graph / XPU graph, piecewise vs full. Local history
(`AGENTS.md`): FlashAttention FULL capture fails because SYCL
work-group scratch is not visible to the graph extension, and a
Qwen3.6 piecewise control was 41 graph pieces per token. Strata's win
is one graph per verify window, which is the FULL-graph end of that
spectrum, achieved by keeping the queue linear (SH_LATE,
`MTP_SHARED_BRANCH=0`). Porting the kernels as eager custom ops and
letting vLLM piecewise-graph them is M. Getting one XPU graph per
token, with `joint_matrix` and SLM, is L and may be impossible on the
current graph extension. Measure that before depending on it.

## 13. PLE

PLE is the per-layer n-gram embedding block, not a speculative n-gram
drafter. One token gathers sixteen rows and applies a residual stack
(key, query, value, signed-sqrt gate, conv over the normalized history).
Contract: `include/strata/kernels/ple.hpp:1`. GPU half:
`sycl/src/kernels/cuda/ple.dp.cpp:1`. The table is the second GGUF shard,
passed with `--ple-gguf`. Without it, layer 1's PLE is skipped
(`generate.cpp` help text near `:849`).

`--ple-io direct` (`generate.cpp:750`) is unbuffered reads of that
table. The comment at `:561` says the gather measured 2.10-2.61 ms/token
when sixteen row reads were sixteen page faults into a 26.8 GB mapping.
S32, on a later serving candidate, recorded PLE gather median 0.829 ms
inside the host boundary trace. That is a latency slice, not a tok/s
A/B.

`STRATA_BATCH_MTP_PLE_PREFETCH=0` and `STRATA_BATCH_MTP_PLE_PREFETCH_DIAG=1`
(`generate.cpp:4144` and the diag at `:12431`). The prefetch is
experimental CPU-side only; the slot's GPU `PleRun` stays unbound.
S47 (`docs/20261015_strata_ple_prefetch_readiness.md`) validated the CPU
reader and found batch sessions do not receive the table metadata the
prefetch needs. No serving gain. Diag is logging.

vLLM analogue: a custom embedding submodule plus a host-side table
reader. The GEMV is an OOT op (M). The 26.8 GB direct-IO gather is a
runtime service (M as a sidecar, L if it has to sit on the critical
path inside the model runner and stay graph-friendly). If upstream
Qwen3-Next already folds this table into the checkpoint as a normal
embedding, use that and do not port the GGUF shard reader.

## 14. Vision on CPU

`hotschmoe-dd.json` `vision` block: exe
`build-vision-cpu/bin/strata-vision`, mmproj BF16, 8 threads, plus
CLI `--vision` (`generate.cpp:820`). The encoder is a separate process.
`serve/server.py:2208`: llama.cpp mtmd reads `ENC <image> <out>` and
writes embeddings; the engine receives them as `GENI` lines. Results
are cached by image hash. It does not use the B70s. Text-only fallback
is `hotschmoe-dd-text.json`.

This is a capability (image input on the daily driver since 2026-10-16),
not a tok/s feature. No speed measurement is cited.

vLLM analogue: a multimodal preprocessor that runs outside the GPU
worker and returns image embeddings. OOT (S/M). The model runner still
has to accept precomputed vision embeddings in the prompt (M if the
Qwen3-Next vision path already exists, L if the embedding injection
points differ).

## 15. Flags set to zero, and what the measurement said

These are in the DD env so the binary cannot pick up a different
default. The value is the measurement.

| Flag | Set | Evidence |
|---|---|---|
| `STRATA_BATCH_PREFILL_QUANTUM` | 0 | S33: a 4096 quantum cut the worst content gap and made incoming TTFT worse (29.5 s to 37.6 s). Keep 0. `generate.cpp:10924`. |
| `STRATA_PREFILL_FIRST` | 0 | Explicit chunk policy. `prefill.cpp:2528`. No isolated win cited. |
| `STRATA_BATCH_MTP_PLE_PREFETCH` | 0 | S47, no serving path. Diag stays 1. |
| `STRATA_Q6_PACKED` | 0 | Not promoted. |
| `STRATA_HEAD_Q5K_RPW2` | 0 | S55, inside control noise. |
| `STRATA_MMVQ_PF` | 0 | Prefetch distance 0. |
| `STRATA_PROMPT_ATTN_XMX` | 0 | Older single-query XMX, slower than fallback. The night union path is the one that is on. |
| `STRATA_KV_ROT` | 0 | Opt-in Hadamard on INT8 KV. Changes first-token KL. |
| `STRATA_EXPERT_SPLIT` | 0 | Keep the port grouped expert kernels. |
| `STRATA_Q6_A4` | 0 | S6-A4 serving +0.43%, kept off. |
| `STRATA_BATCH_MTP_PAIR_SKIP_PROB` | 0 | Probability skip off. No v3 win cited. |
| `STRATA_BATCH_MTP_OWNER_QUEUES` | 0 | S131, -0.33% aggregate. |
| `STRATA_Q5_HEAD_RUNTIME_ONES` | 0 | S130 +0.66% on a full-vocab head. DD uses the subset head and leaves it 0. |
| `STRATA_Q5_HEAD_RUNTIME_ONES_C6` | 0 | Same family, C6 shape. |
| `STRATA_QSA_COMMIT_GROUP_STEPS` | 0 | `verify.cpp:127`. Grouped QSA commit was a Codex candidate. Not in the night-final win list. |
| `STRATA_ROPE_TABLE` | 0 | `native_rope.dp.cpp:208`. Table angles differ in the last bits. Off keeps the fast-math rope. |
| `STRATA_GDN_SPLIT` | 0 | Required off for `gdn_commit_layers` to run. |
| `STRATA_DBG_GDN` | 0 | Debug prints off; also required for the commit kernel. |
| `STRATA_MTP_SHARED_BRANCH` | 0 | The v3 win. See rank 8. |

Also measured and kept out of the env entirely: pipeline-windows 2
(crash), persistent HC kernel, fused router+top-10, oneMKL grouped
GEMM, GDN chunkwise prefill, pipelined persistent MMVQ (C2 -4% with
`CAP_PCT 0`), EP hot-expert share, C4 depth 2 at 12 rows, spec 5/6
inside the 2-slot config, drafter HC Q8.

## SYCL kernels that can be standalone torch custom ops

The hot kernels live under `sycl/src/kernels/cuda/*.dp.cpp`. They are
SYCL (DPCT), launched on `sycl::queue`, and several use
`sycl::ext::oneapi::experimental::joint_matrix` or work-group SLM.
They are not CUDA. A torch custom op means: compile the kernel with
icpx against the same oneAPI as the XPU PyTorch, export a function
that takes raw pointers plus a queue, and register it through an
out-of-tree vLLM plugin (`vllm/plugins` or a platform plugin). They
include Strata headers and `strata::q_of(stream)` today, so each lift
is a small extraction, not a drop-in file.

Good lifts, in the same order as the ranking:

| Source | Entry | Why it lifts cleanly | vLLM call site |
|---|---|---|---|
| `sycl/src/kernels/cuda/native_mmvq.dp.cpp` | `native_mmvq` around `:4808`, KREG path `:2236` | Shape-gated, returns by launching or falling through. Biggest decode win. | Quantized linear / logits head |
| `sycl/src/kernels/cuda/iq_kernels.dp.cpp` + `night_tables.inc` | night IQ at `:2409`, `native_gu_kernel_n` / `native_down_kernel_n` | Bitwise with the current kernels. Pure weight-block in, fp out. | Fused MoE expert |
| `sycl/include/strata/kernels/night_grouped.hpp` + `night_grouped_expert.inc` | `night_grouped_f16` `:24` | Returns false on a bad type. No global FP16 weight copy. The prefill expert win. | MoE prefill grouped GEMM |
| `sycl/include/strata/kernels/xmx_gemm.hpp` | `xmx_gemm_iq` `:9` | Same fallback contract. IQ type in, FP16 activations, FP32 out. | Dense or expert prefill GEMM |
| `sycl/src/kernels/cuda/qsa_prompt_attn_union.dp.cpp` | `qsa_prompt_attn_union` | Returns false unless INT8 KV and 24/2/256. The 2000 tok/s prefill step. | QSA prefill attention |
| `sycl/src/kernels/cuda/qsa_decode_attn.dp.cpp` | value-pipeline and night merge, `:912` | Fixed head geometry. C2 attention 4x at 1K. | QSA decode attention |
| `sycl/src/kernels/cuda/qsa_select.dp.cpp` | block scores `:1690`, top-k scan `:2190` | Indexer scores and cell ids. Needed before either attention op. | QSA indexer |
| `sycl/src/kernels/cuda/fused_gr_night.inc` | `night_hc::read`, included at `fused_gr.dp.cpp:1755` | Two kernels, Q8 or BF16. The HC delta. | Hyper-connection block |
| `sycl/src/kernels/cuda/kv_q8.dp.cpp` | `kv_append_q8_kernel` `:36` | Group-64 quant. Small. | KV append, only with a matching pool |
| `sycl/src/kernels/cuda/verify_kernels.dp.cpp` | `gdn_commit_layers` `:1857`, slot recurrence `:2052` | Fused state update. Geometry is hard-coded (36 layers, specific strides). | GDN layer, after layout check |
| `sycl/src/kernels/cuda/dequant_bf16.dp.cpp` | `night_ddq_on` `:223` | Coalesced dense dequant. Useful only while dense weights stay GGUF. | Weight prep |
| `sycl/src/kernels/cuda/native_router.dp.cpp` | top-10 at `:390` | Small, bitwise. Fusion with the GEMV was slower. Low priority. | MoE router |
| `sycl/src/kernels/cuda/ple.dp.cpp` | PLE value / postops | The math lifts. The 26.8 GB gather does not. | PLE block |

Poor fits for a single-device custom op:

- `sycl/src/kernels/cuda/ep_kernels.dp.cpp` (`ep_push`, `ep_send_psum`,
  `ep_prefill_reduce_h`). These take another card's pointers and a
  device flag, and they are recorded into the window graph. They belong
  in a custom all-to-all / P2P helper shared by two workers, not in
  `torch.ops` called from one rank. L, next to the EP runtime.
- `sycl/src/core/graph.cpp`. Capture runtime. Use vLLM's graph manager.
- `include/strata/core/conversation_cache.hpp` and
  `conversation_snapshot.cpp`. Host policy plus copies. A KV connector,
  not a kernel.
- `serve/server.py` `Vision`. A subprocess. A preprocessor plugin.

Capture risk, from this repo's own XPU record: work-group SLM and
`joint_matrix` scratch often cannot be replayed inside a full XPU
graph. The KREG, grouped-expert, and XMX attention kernels all use
SLM. Plan them as eager custom ops first. Piecewise graph is the
fallback, and it is the path that already worked for the older vLLM
XPU stack. A full-graph replay of these kernels is an experiment, not
a port assumption.

## What to port first, if the target is the same four numbers

1. Model skeleton that already has Qwen3-Next blocks: QSA layers, GDN
   layers, hyper-connections, MTP head, PLE. Without that, every kernel
   above has no caller. This is the core-vLLM question (L), and it is
   the gate.
2. Two-rank expert split with card 1 idle during attention, plus the
   prefill reduction. This is the decode floor and the first prefill
   doubling. Core (L). Do not start from TP=2 all-reduce.
3. KREG MMVQ and the IQ expert kernel as XPU custom ops, if weights stay
   IQ3_XXS GGUF. These two are the decode path from ~107 to ~139 tok/s
   on C1. OOT (M).
4. Grouped `joint_matrix` expert GEMM and the XMX union attention.
   These two cross 2000 tok/s prefill. OOT kernels (M), schedule in
   core (L).
5. INT8 group-64 KV and a connector that also parks GDN state. That is
   the 200k and the RAM-restore target. Core (L).
6. Batch MTP depth 2 and the paired head, after single-sequence MTP
   already matches spec 4 acceptance. Core (L).
7. Leave the ~1% flags, the vision process, and the GGUF expert cache
   until the five items above are measured on vLLM.

## Evidence limits

- Line numbers are from worktree `e753e9cb`, not from a disassembly of
  `releases/strata-night-20261016`. The four later commits are
  default-off and are not in the DD env.
- Night-lab deltas are fixed-prompt, usually two reps, often one
  process order. The lab says a non-bitwise kernel can move greedy
  tok/s by about +/-3% through MTP acceptance. Stacked stages (N8+N9,
  N17, N18) are not fully crossed.
- S87, S89, S38, S100, S114, S130, S131 are earlier binaries. They
  support the small flags. They are not night-final-v3 A/Bs.
- vLLM difficulty ratings were not checked against a vLLM tree in this
  task.
