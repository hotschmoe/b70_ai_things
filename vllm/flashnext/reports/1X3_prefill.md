# X3: Flash-Next real-shape prefill budget and GGUF serving dispatch

Work in progress, 2026-10-20. No commit, no DD changes. Raw evidence:
`/mnt/vm_8tb/b70/build/vllm-flashnext/x3/`.

## CONFIG

12288-token chunk. Flash-Next ISTA GSQ-RCO IQ3_XXS, 48 layers:
36 GDN, 12 full attention, hidden 2560, 512 routed experts, top-10,
expert intermediate 640, shared expert, HC=4, PLE at GGUF blk.1.
One-card measured probes use card 1 with ZE_AFFINITY_MASK=1 via xpu-run.sh;
GPU execution leased, timed windows under cpu-run --quiet. Priority
C7b/H1e pair serves go first. Host torch 2.14.0+xpu, existing campaign
container and host UMD. No driver/runtime change.

C13's 3775.74 ms table mixes measured expert time with scaled projection
estimates and a projected full-chunk head. It omits GDN, attention, HC
pointwise work, PLE, indexer selection and communication. It is not a
measured model prefill rate. In a normal serve the logits processor selects
sampling positions before lm_head; a 12288-row head is not a standard
single-request prefill cost.

## COMMAND

- `harness/x3_prefill_ops.py`: event and wall timings of individual real
  shapes, with actual GGUF projection/expert/HC/PLE weights where specified.
- `scripts/x3_wide_build.sh`: additive C13 wide dequant registration, leaving
  the installed decode library intact. Build under cpu-run, MAX_JOBS=4.
- `harness/x3_serve_probe.py`: matched baseline/new prefill on one loaded
  35B proxy server, diagnostic event spans, identity and coherence.
- CPU `tests/x3_hash_check.py`: 100 ragged hash/history trials.

Claude consultation was attempted concurrently through `claude --bare -p`
with requested models fable-5.1 and claude-5.5. Both returned exit 1,
unrecognized_model and Not logged in. No actual responding model ID or
model advice exists for these calls.

## RESULT

CPU audit proves the GGUF serve bypassed every optimized prefill wrapper:
`gguf/ops_interface.py:linear` unconditionally called `gemv`, whose token
axis is decode-sized, and `moe` used one singleton grouped_gemm entry per
expanded token slot. It did not group tokens by expert for GEMM reuse.
This explains why C12's prefill rate stayed roughly constant at
177/175/173 tok/s as prompt length increased. E2's 1961 tok/s used unpack
plus oneDNN linear, with its remaining cost the per-expert launch loop.
The checkpoints and quant formats differ; that is not a matched speedup.

Added opt-in B70_PREFILL=1 routing to existing prefill.py wrappers. Decode
T<=8 is unchanged. Mixed gate/up types are decoded independently. The
35B Q2_K/Q3_K expert types must use dequant, since the fused prefill kernel
has no Q2_K/Q3_K instantiations. Dense and expert dispatch support that.
C13 smoke .so cannot serve on its own; the additive _C.x3wide.so supplies
only wide dequant after loading a full decode library.

The padded scratch cache previously retained a separate buffer for every
observed expert capacity, which varies per layer. Keys now reuse one
buffer per expert/activation geometry. A Triton cast/multiply/top-k sum
candidate avoids materializing two fp32 slot cubes. PLE host hashing has
an opt-in NumPy uint64 implementation; its 100 ragged CPU trials pass,
including EOS reset, overflow, initial history, empty requests and slot
permutation. GPU measurements and serving qualification follow below.

## VERDICT

No end-to-end 2000 tok/s claim yet. The remaining work is measurement and
serving validation of the opt-in dispatch and the complete budget table.

## Measured missing operations (card 1, 12288 tokens)

Event median in milliseconds, 3 warmups and 5 samples unless noted.
Norm dispatch is explicit compilation mode 0, `forward_xpu`, matching
the serving recipe. `ops-card1.json` used a default standalone config
that dispatched norms to native Torch; those norm rows are superseded.

| Operation | ms/call | model count |
| --- | ---: | ---: |
| gdn_conv_chunk | 6.725 | 36 |
| gdn_norm_gate | 0.989 | 36 |
| full_attention | 24.155 | 12 |
| q_norm | 0.684 | 12 |
| k_norm | 0.174 | 12 |
| rope | 0.364 | 12 |
| attention_gate | 1.464 | 12 |
| ple_hash | 902.715 | 1 |
| ple_hash_numpy | 2.909 | 1 |
| ple_whole | 983.972 | 1 |
| ple_whole_numpy | 79.466 | 1 |
| router_softmax_top10 | 1.269 | 48 |
| reduce_torch | 11.418 | 48 |
| reduce_triton | 1.577 | 48 |

PLE hash/whole are diagnostic parent/child spans; never add them together.
Whole PLE falls 983.972 -> 79.466 ms; hash falls 902.715 -> 2.909 ms.
Measured on the actual packed host table and actual PLE weights, with
warm pages. Cold paging is outside that warm budget.

Fused routing reduction is 1.577 vs 11.418 ms, max_abs 1.91e-6 on
the same bf16 slots, passing atol=rtol=1e-5. On real blk.0/3/36
expert weights with uniform synthetic routing, whole layers are
47.911/47.557/47.080 ms versus 57.908/57.285/56.892 ms. Real routing
skew and the exact per-type census are measured below.

HC with actual blk.0 weights: mix 12.187 ms (97 calls), combine
2.679 ms (96 calls). Those include HC GEMMs and pointwise work.

The first 35B serving attempt loaded weights then failed its zero-input
profile: a concentrated router padded all 256 experts to the full
capacity, and bmm requested a 6 GiB output beside 27.52 GiB already
allocated. This was a torch OOM, not DEVICE_LOST. Post-health passed.
Both dequant layer variants now fall back to unpadded per-expert
GEMMs for pathological padding. The configured padding ratio cap is
`B70_PREFILL_PAD_RATIO` (default 2). This also covers real router skew.


## Exact 12288-token dense census and pointwise budget

The census reads every real GGUF layer's type and shape, groups equal
operations, times a representative with that shape/type, and multiplies
by the actual occurrence count. These are measured operator budgets,
not whole-model TTFT. Random activations are used except the full PLE
path, which uses actual table lookup. Parent and child spans do not add.
Source: x3/census.json, ops-rest.json, checks-tile.json; census-r2.log
contains numeric results and pre/post HEALTHY with GPU exit 0.

| Operation | model calls | ms/chunk |
| --- | ---: | ---: |
| GDN packed QKV projection, all types | 36 | 227.385 |
| GDN gate projection, all types | 36 | 142.325 |
| GDN alpha + beta projections | 72 | 17.059 |
| GDN output projection | 36 | 131.316 |
| GDN conv + chunked delta | 36 | 242.100 |
| GDN RMS norm and gate | 36 | 35.604 |
| GDN QKVZ / BA concatenation | 72 | 87.450 |
| GDN value-axis retiling, direct | 36 | 22.519 |
| Full-attention Q / K / V projections | 36 | 104.280 |
| Full-attention output projection | 12 | 51.555 |
| Full causal FlashAttention, 24/2/256 | 12 | 289.860 |
| Attention Q / K norm | 24 | 10.296 |
| Text M-RoPE, actual model config | 12 | 7.255 |
| Attention sigmoid gate / multiply | 12 | 17.568 |
| Shared-expert gate / up / down projections | 144 | 127.650 |
| Shared-expert silu / multiply | 48 | 10.347 |
| Shared-expert scalar gate projection | 48 | 9.245 |
| Shared-expert sigmoid / scale | 48 | 27.210 |
| Routed router projection | 48 | 17.870 |
| Actual fused softmax/top-10 router | 48 | 13.885 |
| HC mix, fused pointwise + original oneDNN | 97 | 476.204 |
| HC combine, fused | 96 | 145.590 |
| Initial HC expand | 1 | 0.634 |
| Whole PLE, NumPy hash and actual host table | 1 | 79.466 |
| Token embedding gather/dequant | 1 | 1.345 |
| Sampled-position LM head | 1 | 0.809 |
| Routed expert layers | 48 | see routing table below |
| DP/EP dispatch + combine, native C1 | 48 | 4859.7 |
| Indexer K / Q projections, QSA=1 only | 24 | 13.959 |
| QSA append / query prep / selection+union, QSA=1 helper candidate | 12 | 105556.6 |

GDN split: convolution 2.206 ms and chunked delta 4.633 ms per layer;
the independently timed parent was 6.747 ms. The table uses the matched
ops-rest parent 6.725 ms and does not add the split again. Full attention
is exact first-chunk causal attention, not a 200K-context continuation.
KV cache writes, scheduler, logits processing and any unhooked glue will
be reconciled against real model-runner and client spans. No 2000 tok/s
claim is made from adding this operator table.

HC matched reference/new max_abs=0 for both mix and combine on the tested
real-weight input. Mix 12.149 -> 4.909 ms and combine 3.560 -> 1.517 ms
save 898.374 ms in the operator budget. GDN direct retiling is bit-exact,
2.018 -> 0.626 ms, saving another 50.117 ms. PLE whole saves 904.506 ms.

Expert routing matters. Uniform routes at layer 36: old reduce 56.892 ms,
Triton slot reduce 47.080 ms; grouped direct pad/reduce 41.801 ms with
7.018 GiB peak incremental allocation. Expert tile 128/64/32 measured
48.213/53.769/61.171 ms and 2.168/1.318/0.894 GiB. These are all 512
experts on one card; a DP2 EP rank owns 256. A measured real learned
router on random activations has much more skew and triggers the bounded
unpadded fallback. Across ten actual quant-type pairs it costs
110.369-121.780 ms/layer (5705.818 ms after actual count weighting).
Natural text can route differently; this is a stress diagnostic, not a
measured natural-text distribution. Power-of-two row bucketing is under
numeric/timing test to avoid the pathological padding and the 512 loops.

Direct grouped reduction passes the original BF16 reference tolerance
and the tight unchanged-geometry comparison (atol=1e-7, rtol=1e-5).
Tiling changes oneDNN GEMM geometry for mixed gate/up types: max_abs
0.000617 with cosine 1.0000001 in census-r2. It passes the original BF16
reference gate (atol=rtol=0.05). It is not bit-identical.

35B one-layer crossover with actual packs and uniform routing:

| tokens | fused ms | wide dequant+bmm ms |
| ---: | ---: | ---: |
| 1024 | 10.741 | 12.771 |
| 8192 | 29.818 | 13.776 |
| 12288 | 44.018 | 16.254 |

The proxy serving candidate therefore uses B70_PREFILL_DQ_T=8192;
unsupported Q2_K/Q3_K/Q8_0 types still require dequant at smaller T.

## EP sidecar traffic distinction

The sidecar maps peer device VRAM, not system RAM. At T=12288, H=2560,
top-10, an expert layer transfers X BF16 62,914,560 bytes, IDs int32
491,520 bytes, weights fp32 491,520 bytes, and one FP32 partial result
125,829,120 bytes: 189,726,720 peer bytes/layer or 9.107 GB/48 layers.
Local inbox/stage copies add VRAM traffic, not a second PCIe transfer.
H1's 64 KiB micro result (5.834 us, 11.233 GB/s) would project about
16.9 ms/layer, but that is not a measured 190 MB prefill transfer.
The stock DP2 Ag/Rs candidate is a different communication path and
requires its own event measurements. New Python prefill expert dispatch
is not wired into the raw EP sidecar expert worker; forwarding switches
in serve_flashnext_ep.sh accelerates trunk/PLE only.


## Skew fix: measured expert row buckets

Power-of-two row buckets avoid padding every expert to the most popular
expert and avoid the fallback's 512 separate GEMMs. Experts with zero
local assignments are skipped; optional tiles bound dequant workspace.
The scratch arena no longer retains separate C13 buffers for every
bucket expert count. Grouping no longer performs ok.all() on the host
and converts its one small starts copy with tolist().

| real layer | old fallback ms | buckets ms | bucket tile64 ms | tile64 peak delta GiB |
| ---: | ---: | ---: | ---: | ---: |
| 0, types17/17/20 | 113.132 | 60.707 | 63.583 | 1.411 |
| 3, types16/16/42 | 116.275 | 58.890 | 67.339 | 1.442 |
| 36, types18/18/42 | 118.323 | 60.462 | 64.852 | 1.473 |

These are complete 512-expert layers on one card at T12288, with actual
learned routers on random inputs. Old peak delta is6.438 GiB. Untiled
buckets peak delta2.007/3.347/2.886 GiB. Source bucket.json/bucket.log,
GPU exit0, post-health HEALTHY. Same/mixed type and -1 ID reference
checks pass the original .05 absolute/relative gate. Full-shape bucket
max_abs differences .000959/.000456/.047396, cosine1/1/.999719.
Layer36's larger difference is retained as a limitation; no bit-exact
or model-quality equivalence claim follows from this numerical gate.

A separate synchronized stage diagnostic on layer36/tile64 sums:
group5.235, dequant gate8.135/up7.197/down7.200, direct pad3.356,
gate-up bmm12.003, silu2.505, down bmm6.449, direct reduce4.633 ms.
The 14 bucket calls and their synchronizations make this diagnostic
different from the 64.852 ms unsynchronized median. Packed weight
gathers, remapping and output accumulation are the outer residual.


## Reproducible candidate switches

The fork changes are small opt-in dispatches in gguf/ops_interface.py,
gguf/linear.py and xpu/model.py. Main implementation lives in plugins.
C12 supplies the complete decode operator set; X3wide registers only
prefill_dequant_wide. Never replace a decode library with C13 smoke-only.

- B70_PREFILL=1: packed dense/expert prefill dispatch at T>8.
- B70_PREFILL_DQ_T=8192: measured proxy crossover.
- B70_PREFILL_REDUCE=grouped: direct token pad and slot-order combine.
- B70_PREFILL_BUCKET=1: expert row buckets for dequant paths.
- B70_PREFILL_EXPERT_TILE=64: bound expert workspace for full serving.
- B70_PREFILL_PLE_HASH=numpy: wrapped uint64 CPU hash.
- B70_PREFILL_HC=triton: fused HC pointwise work, original GEMMs.
- B70_PREFILL_DQ_OVERLAP=0: single queue for this budget/candidate.
- B70_PREFILL_QSA=1: optional QSA helper candidate, QSA1 only.

Decoder T<=8 keeps the existing path. These flags have not qualified
MTP verification batches above8 or general full-graph prefill capture.
The candidate uses MTP0, FULL_DECODE_ONLY captures<=8, BF16 KV,
no prefix cache, dense GQA (QSA0), and stock DP2 EP (not sidecar).
serve_flashnext_prefill.sh takes both leases through xpu-run, includes
the private C7b head256 FA overlay and the memory plugin path, and
keeps hotschmoe-dd first with a registered detailed research alias.
The harness contains an 8 GiB MemAvailable abort watchdog. It never
starts a systemd service.


## QSA: full-size timings and wrapper fix

QSA uses the real 24/2/256 geometry,4 indexer heads of128, compression4,
budget2048, actual blk.3 Q/K norm gamma and 3-axis text positions.
Inputs are random, not natural text; the sparse selection/union distribution
is therefore a diagnostic. Single event/wall sample at T12288 after small
kernel warmup, existing QSA binary; timed under CPUquiet on card1.

| QSA operation, one attention layer | old ms | candidate ms |
| --- | ---: | ---: |
| KV/indexer append core,3 input copies/token | 2060.655 | included below |
| actual model_hook append, including side state and unused queries | 22042.110 | 3455.148 |
| whole query norm + text M-RoPE | 4719.806 | 0.795 |
| pooled score/topk + union attention + gate | 5340.442 | 5340.442 |
| combined helper budget per layer | 32102.358 | 8796.385 |
| combined helper budget for12 attention layers | 385228.3 | 105556.6 |

Do not add the core append row to its parent. Candidate append skips query,
gate and indexer-query preparation that append never reads; side state is
loaded/stored per request, and M-RoPE positions are staged in a vector.
Query RoPE uses one vector operation instead of a Python loop over tokens.
The probe conservatively splits append into1024-token helper calls for
progress reporting. Exact cache pages/tail/dead/block_pos/length match;
query max_abs=0 at T12288. The mixed-expert arena checks also passed.
GPU exit0, pre/post-health HEALTHY; qsa.json and qsa.log hold evidence.

QSA1 cannot meet2k tok/s with these kernels. The8-query selection/union
loop alone is64.1s over12 layers; append is still token-at-a-time. It needs
batched prefix/pool append and many-query selection/attention kernels.
B70_PREFILL_QSA=1 remains opt-in and has no full QSA serving/C2 quality
qualification. All real-server candidate numbers below explicitly use
QSA0/dense GQA, matching the current campaign's serving controls.


## Matched35B serving result

Same loaded Qwen3.6-35B-A3B UD-IQ3_XXS, one physical card1,
C12+wide, FULL_DECODE_ONLY, FA, SYCL GDN, maxlen16384/batch12288,
seqs2, util.75, prefixoff. B70_PREFILL_CONTROL switches only prefill
flag0/flag1 between requests. Candidate uses bucket1, tile0, grouped
reduce and DQ_T8192. Timed under CPUquiet, with diagnostic event hooks
on both arms. Model identity hotschmoe-dd first plus registered detailed
alias. Both deterministic Paris gates pass, two concurrent Paris
completions match, and32 generated tokens after a1024-token prompt are
identical between old/new. Teardown exit0 and post-health HEALTHY.
Raw serve-ab.json, proxy-spans.jsonl, serve-probe.log, serve-window-r2.log.

| actual input tokens | old TTFT ms | new TTFT ms | old tok/s | new tok/s |
| ---: | ---: | ---: | ---: | ---: |
| 1024 | 6025.851 | 675.427 | 169.93 | 1516.08 |
| 8181 | 45445.784 | 2388.216 | 180.02 | 3425.57 |
| 12277, first natural routing | not run | 52254.904 | n/a | 234.94 |
| 12277, repeated with prefix cache OFF | not run | 2456.050 | n/a | 4998.68 |

This is a matched19.03x speedup at8181 tokens. E2's1961 tok/s used a
different BF16-checkpoint lowbit artifact; the new GGUF rate is higher
but that comparison is not a matched quantization/quality speedup.
At8181 tokens, supported expert types are just below the8192 dequant
threshold and still use fused prefill; unsupported types use dequant.
At12277, all types use the wide dequant path.

Actual scheduled8181-token spans, parent rows only:

| parent operation | old ms | new ms |
| --- | ---: | ---: |
| 30 GDN layers, including projections and core | 8806.609 | 389.844 |
| 10 full-attention layers, including projections/cache writes | 2343.418 | 156.961 |
| MoE experts incl shared/dispatch framework | 34013.166 | 1571.518 |
| routed expert apply, child of preceding row | 31835.677 | 1494.564 |
| shared expert, child of MoE row | 2162.586 | 61.973 |
| complete model forward event | 45183.150 | 2138.313 |
| runner wall | 45409.583 | 2352.234 |
| client TTFT | 45445.784 | 2388.216 |

Do not sum parent/child rows. Full-attention core itself stays76.624 ->
76.914 ms and KV writes .601 ->.613 ms; the large improvement is packed
projection/expert dispatch, not faster attention. GDN in_proj_qkvz
6409.572 ->215.171 ms, out_proj2272.534 ->51.996 ms. This directly
explains the C12 roughly175 tok/s plateau: decode GEMV and singleton
expert entries were used for every long input, not the prefill kernels.

The first natural12K request exposed an unacceptable52.25s cold spike:
its routed-expert span was50.305s versus1.387s on repeat. vLLM logged
inference JIT of _pad_tokens. CAP was constexpr in padding/inverse and
changed with routing, causing many compiled variants. CAP is now a
non-specialized runtime scalar in both kernels. Exact dynamic-capacity
checks at16/24/40 run before the Flash-Next budget window. The original
cold result is retained, not averaged into or hidden behind warm rates.


## Measured DP/EP communication at the real chunk size

R2 uses one xccl process group per worker for the whole experiment.
Per-card pre-health and eager/compiled10-iteration before/after checks
pass. All transfer outputs match rank-labelled input values and SUM=3.
No topology env override.3 warm calls then5 samples, CPUquiet, both
cards leased. Each rank sees both physical devices, like the stock DP2
serve. Source collective-budget.log and collectives-rank0/1.json.

| global chunk / op | rank0 event ms | rank1 event ms |
| --- | ---: | ---: |
| C1 sizes12288/1,3-gather dispatch | 7.666 | 7.747 |
| C1 sizes12288/1, native uneven reduce-scatter | 93.251 | 93.496 |
| balanced sizes6144/6144,3-gather dispatch | 3.210 | 3.215 |
| balanced sizes6144/6144, equal reduce-scatter | 3.083 | 3.067 |

The C1 native route alone projects4.86s over48 layers including dispatch.
The balanced test has12288 tokens globally, not12288 per stream. It is
not a C2 throughput claim. Native uneven RS is a clear top cost; the fork
now reuses its existing padded equal-size route for B70_PREFILL=1,
max(sizes)>=512 and unequal sizes. B70_PREFILL_EP_RS=native disables it;
pad enables it. Decode and balanced transfers keep their current route.
The full-model comparison below will measure this path in loaded context.

R1 has no transfer result: it destroyed the health process group and
reinitialized one, then timed out. Its immutable source/log copy remains
under x3/collective-budget-r1.py and r1-collective-budget.log. An X3 live
shell edit also caused an EOF error and skipped its intended post-health.
That harness mistake was corrected with an immutable R2 shell. X1 paused
its next load, X2's fresh compiled health passed, and R2's native and
compiled before/after health passed. No health evidence required xe-reset,
and none was attempted. No full X3 model ran during the failed window.
