# X3: Flash-Next 12288-token prefill budget

2026-10-20. The matched 35B GGUF dispatch fix is measured: 180 -> 3426
input tok/s at8181 tokens, with coherence and teardown/health evidence.
Flash-Next real-server qualification is still running. No end-to-end
2000 tok/s claim yet. Strata DD enables QSA; the stock vLLM campaign
controls use dense GQA (QSA0). These are different attention configurations.

## CONFIG

- Hardware:2 B70s, kernel7.1.0-070100-generic; unchanged campaign host UMD,
  Level Zero1.28.2, torch2.14.0+xpu and vllm-flashnext-xpu:py312.
- Model: ISTA GSQ-RCO IQ3_XXS shard1 under
  `/mnt/cache/b70/qwen3.8-flash-next/ista-gsq-rco/IQ3_XXS/`;
 48 layers,36 GDN/12 full attention, H2560,512 experts/top10, F640,
  HC4/lowrank320, PLE at GGUF blk.1. The host PLE table is mmap-backed.
- One-card probes: physical card1, ZE_AFFINITY_MASK=1 through xpu-run
  --card1, leased; all timed bodies under cpu-run --quiet.
- Full candidate: TP1/DP2/EP, BF16 KV, MTP0, QSA0, prefixoff,
  maxlen12800/batch12288/seqs2/util.98, FULL_DECODE_ONLY captures1..8,
  SYCL GDN, C7b private head256 FA overlay, memoryfix1. Primary name
  hotschmoe-dd plus the registered detailed X3 research alias.
- Decode library `_C.c12.so` sha256
  b3c15e14ba5811d90dfe8a65e4ccc1353fd67b3f44e4180810fac22b4ed49979.
  Additive `_C.x3wide.so` sha256
  98c87510c7e713c08b312f5527ba4cdb9868e16ea975e8eeb8a70d9ae76aad4c.
  C13 smoke-only is not a complete serving library.
- QSA native library `/mnt/vm_8tb/b70/build/vllm-flashnext/b70_qsa_xpu/libb70_qsa_xpu.so`,
  sha2562ee734458f9ed0ac0ebd827c409c3771ba315fab15dfdf004bd19da24f67bf71,
  mtime10:57:30 UTC, before these probes.

Raw artifacts are under `/mnt/vm_8tb/b70/build/vllm-flashnext/x3/`.
The existing fork has other agents' changes; X3 made no commit/push and
never started DD. User changes and other agents' work remain preserved.

## COMMAND

- `harness/x3_prefill_ops.py`: actual GGUF shape/type census, real packed
  weights, GDN/attention/HC/PLE/router/expert/QSA event and wall timings.
- `scripts/x3_wide_build.sh`: one additive native build, cpu-run,
  MAX_JOBS4,36s. No repeated full vLLM build.
- `harness/x3_serve_probe.py`: same-loaded35B flag0/flag1 comparison.
- `harness/x3_collective_budget.py`: both-card transfer correctness,
  events and eager/compiled10-iteration health using one process group.
- `harness/serve_flashnext_prefill.sh`, `x3_flashnext_probe.py`: exact
 12288 input IDs, native/padded RS, plain TTFT, C1/C2 and long math.
- CPU hash checks:100 ragged/history/EOS/overflow cases pass. Direct GDN
  value retiling is bit-exact in CPU and GPU checks.

Claude CLI consultation requested fable-5.1 and claude-5.5 independently.
Both returned unrecognized_model / Not logged in, exit1. There was no
responding model ID or advice; no consultation result is invented.

## RESULT: exact operator census

All values below are measured events at T12288. Equal GGUF type/shape
operations are represented by one real weight tensor and multiplied by
actual layer occurrence counts. Ordinary rows use3 warmups/5 samples.
Activations are synthetic unless explicitly identified otherwise.
Events enclose each operation and include host-launch/data-readiness
gaps on the GPU timeline; wall samples are also stored. This is a
component budget, not model TTFT. Parent/child rows do not add.

| operation | model calls | ms/chunk |
| --- | ---: | ---: |
| GDN QKV projection, all quant types | 36 | 227.385 |
| GDN gate projection | 36 | 142.325 |
| GDN alpha/beta projections | 72 | 17.059 |
| GDN output projection | 36 | 131.316 |
| GDN convolution + chunked delta | 36 | 242.100 |
| GDN RMS norm + gate | 36 | 35.604 |
| GDN QKVZ/BA concatenation | 72 | 87.450 |
| GDN value-axis retile, direct | 36 | 22.519 |
| Full-attention Q/K/V projections | 36 | 104.280 |
| Full-attention output projection | 12 | 51.555 |
| Native paged causal FA,24Q/2KV/256, block1600 | 12 | 294.777 |
| Attention Q/K norms | 24 | 10.296 |
| Actual text M-RoPE | 12 | 7.255 |
| Attention sigmoid gate/multiply | 12 | 17.568 |
| Shared expert gate/up/down projections | 144 | 127.650 |
| Shared silu/multiply | 48 | 10.347 |
| Shared scalar gate projection | 48 | 9.245 |
| Shared sigmoid/scale | 48 | 27.210 |
| Router projection | 48 | 17.870 |
| Actual fused softmax/top10 | 48 | 13.885 |
| HC mix, fused pointwise + original GEMMs | 97 | 476.204 |
| HC combine, fused | 96 | 145.590 |
| Initial HC expand | 1 | 0.634 |
| Per-layer wide residual mean | 48 | 40.228 |
| FA KV cache write, BF16 | 12 | 2.378 |
| Whole PLE, NumPy hash + real host table | 1 | 79.466 |
| Token embedding gather/dequant | 1 | 1.345 |
| One sampled-position LM head | 1 | 0.809 |
| Routed MoE including dequant/pad/GEMMs/reduce | 48 | routing-dependent, below |
| DP/EP communication | 48 | path-dependent, below |
| Indexer Q/K projections, QSA1 only | 24 | 13.959 |
| QSA append/query prep/selection, QSA1 only | 12 | about3357, below |

The numeric dense-GQA rows sum to2344.349ms before routed experts,
communication and remaining runtime glue. A12288-token TTFT at2000
tok/s must be below6144ms. These rows are not an end-to-end prediction.

Source: census.json, ops-rest.json, checks-tile.json, reuse-r2.json,
paged-fa-r2.json. Norm dispatch is
explicit compilation mode0/forward_xpu. Earlier ops-card1.json native
Torch norm rows are superseded. GDN split is2.206ms convolution and
4.633ms delta per layer; the independent parent was6.747ms. The table
uses its matched6.725ms parent, not the split sum. Full attention uses
the actual C7b private serving library and aligned1600-token cache
pages:24.564740ms/layer, source paged-fa-r2.json. Block64 is25.047500ms,
and complete outputs are bit-exact across the two cache layouts.
Explicit FP32 score/softmax/value reference slices at first/middle/last64
queries pass unchanged atol.002/rtol.02 (RMS.000494/.0000265/.0000188).
The initial Torch SDPA reference flagged32 elements and is retained in
paged-fa.log; the kernel/tolerance were unchanged for the FP32 check.
This is first-chunk causal attention, not a200K-context continuation. HC final
mix has no inject GEMM; using the97-call injected representative is a
small conservative overestimate. KV-write byte checks pass. Real model spans must still account for
final sampling, concatenation details and unhooked glue.

C13's3775.74ms projection omitted these major costs and projected a
full12288-row head. vLLM prunes to sampling positions before lm_head;
the real one-position head is.809ms. That projection was not model TTFT.

## RESULT: fixed costs and routed experts

| operation, one call | reference ms | candidate ms | check |
| --- | ---: | ---: | --- |
| PLE hash | 902.715 | 2.909 |100 CPU hash/history cases |
| Whole PLE | 983.972 | 79.466 |actual warm host table/weights |
| HC mix | 12.149 | 4.909 |tested max_abs0 |
| HC combine | 3.560 | 1.517 |tested max_abs0 |
| GDN value retile | 2.018 | .626 |bit-exact |
| Slot cast/weight/reduce | 11.418 | 1.577 |max_abs1.91e-6 |

PLE hash is part of whole PLE; never add both. Cold host paging is not
covered by the warm table budget. HC mix/combine save898.374ms per model
in this operator census. Original BF16 reference tolerances are retained.

For real blk.36 weights with uniform routing, complete512-expert layers:

| path | ms/layer | peak incremental GiB |
| --- | ---: | ---: |
| Original torch slot reduction | 56.892 | not retained |
| Triton slot reduction | 47.080 | not retained |
| Direct grouped pad/reduce | 41.801 | 7.018 |
| Direct + expert tile128 | 48.213 | 2.168 |
| Direct + tile64 | 53.769 | 1.318 |
| Direct + tile32 | 61.171 | .894 |

Uniform routes hide skew. Actual learned routers on random activations
triggered unpadded fallback110-122ms/layer across10 quant type pairs,
5705.8ms after actual count weighting. Power-of-two expert row buckets
avoid both giant padding and512 individual GEMMs:

| real layer/types | old fallback ms | buckets ms | bucket/tile64 ms | tile64 peak GiB |
| --- | ---: | ---: | ---: | ---: |
| 0,17/17/20 | 113.132 | 60.707 | 63.583 | 1.411 |
| 3,16/16/42 | 116.275 | 58.890 | 67.339 | 1.442 |
| 36,18/18/42 | 118.323 | 60.462 | 64.852 | 1.473 |

These are512 experts/card, not256/EP rank. Do not divide blindly or
claim a natural-text route distribution. Source bucket.json/bucket.log,
GPU exit0/post-health HEALTHY. Same/mixed gate-up, invalid IDs and skew
pass original atol=rtol=.05. Full-shape bucket max_abs differences
.000959/.000456/.047396, cosines1/1/.999719; the last difference remains
a limitation, not a bit-exact or model-quality equivalence claim.
Direct unchanged-geometry reduction also passes1e-7 absolute/1e-5 relative.

One synchronized tile64 diagnostic totals: grouping5.235, dequant gate
8.135/up7.197/down7.200, pad3.356, gate-up bmm12.003, silu2.505,
down bmm6.449, reduce4.633ms. These14 bucket calls synchronize for
attribution; their sum differs from the unsynchronized64.852ms median.
Packed-weight gather, remapping and accumulation are the outer residual.
Scratch no longer retains a buffer for every observed bucket capacity.

After R4, buckets now reuse outer grouping/counts instead of repeating a
native histogram and D2H count wait for every subset. Exact E5/M245760
slot membership, mixed-type/skew/invalid-ID checks and learned-router
fixtures pass (reuse-r2.log/json, body exit0/post-health HEALTHY).
Strided packed GU/down gathers are byte-exact for both implementations;
torch is faster (.144/.183/.163ms vs Triton.208/.204/.210ms) and stays
default. These isolated passes do not prove loaded EP readiness.

## RESULT: why35B GGUF was about175 tok/s, and the matched fix

The fork unconditionally sent dense long inputs to decode GEMV, chunking
M by8. Expert prefill built singleton grouped_gemm entries per expanded
token slot instead of grouping tokens by expert. Optimized prefill.py
was never called. E2's1961 tok/s used unpack + vendor linear on another
artifact; it was not a matched quantization comparison.

B70_PREFILL now dispatches long dense/expert inputs to the measured paths.
Mixed gate/up types are decoded independently; Q2_K/Q3_K/Q8_0 types
force dequant where fused prefill has no instantiation. C13 wide dequant
is added to C12 rather than replacing its decode implementation.

Same loaded35B, physical card1, prefixoff, identical prompts:

| actual tokens | old TTFT ms | new TTFT ms | old tok/s | new tok/s |
| ---: | ---: | ---: | ---: | ---: |
| 1024 | 6025.851 | 675.427 | 169.93 | 1516.08 |
| 8181 | 45445.784 | 2388.216 | 180.02 | 3425.57 |
| 12277, first natural routing | not run | 52254.904 | n/a | 234.94 |
| 12277, warm repeat, prefixoff | not run | 2456.050 | n/a | 4998.68 |

Identity, deterministic Paris in both modes, two concurrent Paris replies,
and equal32-token output after1024 input tokens pass. Teardown0 and
post-health HEALTHY. Source serve-ab.json/proxy-spans.jsonl/serve-probe.log.
The matched8K speedup is19.03x, including diagnostic hooks in both arms.

At8181 tokens, GDN parent8806.609 ->389.844ms; attention parent2343.418
->156.961ms; MoE incl shared34013.166 ->1571.518ms. Routed expert child
31835.677 ->1494.564ms. Attention core itself76.624 ->76.914ms and KV
writes.601 ->.613ms: the improvement is dispatch, not faster attention.
Complete forward45183.150 ->2138.313ms, runner45409.583 ->2352.234ms.

The52s cold12K result is retained. Triton specialized CAP on changing
expert counts, generating many inference compilations. Padding/inverse
now take a non-specialized runtime CAP; exact16/24/40-capacity checks
pass on both cards. The proxy's measured warm rate predates this fix;
no improved cold rate is invented. One-layer crossover measurements
1024 fused10.741 vs DQ12.771ms;8192 fused29.818 vs DQ13.776ms;
12288 fused44.018 vs DQ16.254ms. DQ_T8192 retains short fused inputs.

## RESULT: communication and loaded-context readiness

R2 oracle:3 warmups/5 samples, one xccl group/worker, CPUquiet. All
rank-labelled payloads and SUM=3 pass; eager +10 compiled iterations
pass before/after. No topology env override.

| route/op | rank0 ms | rank1 ms |
| --- | ---: | ---: |
| sizes12288/1,3-input gather | 7.666 | 7.747 |
| sizes12288/1,native uneven RS | 93.251 | 93.496 |
| sizes6144/6144,3-input gather | 3.210 | 3.215 |
| sizes6144/6144,equal RS | 3.083 | 3.067 |

Source collectives-rank0/1.json. Balanced means12288 tokens globally,
not per stream. The3-input micro uses X/weights/IDs. Actual loaded vLLM
selects router-logit dispatch (X plus512 logits), so its communication
must be measured separately. Native C1 RS alone projects4.49s/48layers.
Long B70 prefill now reuses the existing padded equal-size RS route;
B70_PREFILL_EP_RS=native/pad selects the comparison. Decode/balanced
transfers keep their route.

R3 startup exposed the oneCCL handoff problem: rank0 waits at grouping
starts.cpu(), rank1 spins inside a later all_gather_into_tensor. Stacks
are r3-stack-rank0/1.txt. A prefill-only Gloo CPU readiness fence before
AG/RS was tested in R4. Both ranks complete the first three layers, but
rank0 then stalls in a redundant five-expert bucket grouping on layer3
(M245760), while rank1 waits at the CPU RS fence. The outer256-expert
grouping has completed. The CPU fence alone does not fix this stall.
Detailed stacks are r4-stack-rank0/1.txt; native post-health passes.
A short oracle now tests reusing the outer ordering/counts and byte-exact
strided packed-weight gathers before another full weight load.

The loaded-memory oracle then passed:22GiB/rank ballast plus actual
layer3 type16/16/42, AG(X,512 logits)->five-expert buckets->FP32 RS,
four iterations each at12288/12288 and12288/1. Both completion-wait
modes pass, expected rank partial sums match, eager/10compiled health
passes before/after, and native/compiled teardown exits0. Residency
was23,958,388,736B/rank; this approximates, rather than duplicates, the
full model's23.92GiB. At C1/async,3 post-first samples give:

| rank | AG ms | expert ms | padded FP32 RS ms | wall ms |
| --- | ---: | ---: | ---: | ---: |
|0|10.115|9.086|12.607|33.993|
|1|10.153|9.126|12.317|33.015|

Source loaded-ep.log and parsed clean lines loaded-ep-parsed.json.
The five fixed routes are startup-like, not natural-text routing.
Actual GGUF dispatch casts routed outputs back to BF16 before model RS;
the FP32 oracle stresses twice that transfer size. It is a loaded
operator check, not a full-model timing or C1 stability qualification.

The raw EP sidecar is a separate path. At T12288/H2560/top10 it would
transfer X BF16 62,914,560B, IDs491,520B, weights491,520B, partial FP32
125,829,120B:189,726,720B/layer,9.107GB/48layers. This is device-peer
VRAM traffic; local stage copies are not counted twice as PCIe. H1's
64KiB result projects16.9ms/layer at11.233GB/s but does not measure this
large transfer. Python prefill experts are not wired into the raw sidecar.

## RESULT: QSA missing budget and built fast path

Actual geometry, real GGUF indexer gamma, random Q/K/V/indexer inputs.
Reference large loops are single samples after small warmup; new paths
have3 samples. Append/cache equality includes all page bytes, tail,
dead, block_pos, length, pos and step at17/33/81/12288, first/continuation
chunks and page boundaries. Queries and batched attention are bit-exact on these fixtures.

| QSA operation, one layer | old ms | first helper ms | batched ms |
| --- | ---: | ---: | ---: |
| Model append incl unused queries/side copies | 22042.110 | 3455.148 |not timed as a complete parent |
| Core append loop, block1600 | 1991.643 | included above |26.698 |
| Query norm/text M-RoPE | 4719.806 | .795 | .795 |
| Selection/union/gate, block1600 | 5337.157 | 5337.157 |252.253 |
| Approximate12-layer measured helper budget |385 seconds |105 seconds |3.36 seconds + parent glue |

Do not add core append to its parent. Source qsa.json, qsa-batch.json,
qsa-append.json, qsa-aligned.json and logs; all GPU exit0/post-health
HEALTHY. Block64 append is4.318ms; block1600 uses bounded scratch tiles
and26.698ms. Private pooled scratch pages avoid native sentinel races;
only completed pooled rows are scattered back. No new native build.
Contiguous text/base0 is supported; other layouts retain the old path.

Query batches256/1024/4096 at block64 measure397.849/293.954/269.644ms,
all exact. Default1024 bounds peak delta.492GiB;4096 uses.781GiB.
Native score/union already accepted many8-query tiles; Python forced8.
At block1600, instrumented stages: score47.702, topk48.057, union build
.596, XMX union151.964, gate3.976, step1.420ms; whole283.858 including
30.143ms glue/event overhead, distinct from bare252.253ms.

These inputs have synthetic query overlap. A capture-gated one-card
probe is prepared for real first-attention activations after3 real GDN
layers. One-layer extrapolation still is not QSA-enabled model TTFT.
Strata DD explicitly enables QSA (DD config QSA_PREFILL_TILED=1).
Dense-GQA serving results do not close native-QSA parity or200K serving.

## RESULT: real Flash-Next serving qualification

R4 failed readiness during startup profiling, before any API or TTFT
result. R5, after grouping reuse and post-collective GPU waits, completes
AG/RS1..74 but reports DEVICE_LOST at the completion wait for AG75
on rank1. Both fresh native and compiled post-health pass; no reset was
needed. This is a real loaded-runtime failure and no2ktok/s result.
The observer's final allocated/reserved/free values are
26,700,187,136 /32,359,055,360 /461,168,640B, peak29,548,236,288B;
free bottoms at461,045,760B. Cached allocator slack and native
workspace headroom are plausible additional factors, not a proven cause.

The next short oracle tests pending main/auxiliary GPU producers across
48 collective rounds, with a GPU drain BEFORE the CPU readiness fence.
A separate policy guard uses local DP token counts and graph mode so
gathered8+8 decode rows16 never enter CPU-counting prefill paths. CPU
policy cases pass on both ranks. The first producer fixture failed its
byte-backed BF16 router reshape before the loop; corrected retry queued.

The prepared real-model probe compares native/padded RS on
identical12288 IDs, then3 unprofiled TTFT requests, C1/C2 math and a
>12288-token math prompt. Profiled capture I/O is excluded from plain
rates. Identity must be primary hotschmoe-dd plus its detailed X3 alias.

## VERDICT and reproducible switches

The35B bottleneck is fixed and measured. The complete Flash-Next budget
exposes costs absent from the C13 GEMM projection, and the largest CPU,
HC, routing and QSA orchestration costs are substantially reduced.
Real Flash-Next2000+ tok/s remains unclaimed until serving gates finish.

Fast prefill switches: B70_PREFILL=1, DQ_T8192, REDUCE=grouped,
BUCKET=1, EXPERT_TILE64, PLE_HASH=numpy, HC=triton, DQ_OVERLAP0,
PREFILL_WIDE_SO=X3wide. Optional QSA helpers use B70_PREFILL_QSA=1,
B70_PREFILL_QSA_BATCH1024, B70_PREFILL_QSA_APPEND=batch. Local decode T<=8, including gathered DP=2 rows16, stays on the
decode path and graph forwards skip prefill; MTP verification>8/general prefill capture are unqualified.
The candidate harness accepts B70_QSA=1 to attach those helpers, with
the QSA plugin import path present; that full-model mode remains unqualified.
The serving harness preflights the same-checkpoint B70_EMBED_REF required
by memoryfix1 and watches the8GiB MemAvailable floor. No shelf promotion.

Failure/setup evidence is retained: initial35B concentrated-router OOM
(fixed by pad guard/buckets), R1 process-group reinit timeout and live-shell
EOF cleanup mistake, first full load's missing embedding reference, R3
startup handoff stall, R5 AG75 device loss (post-health passes), and
producer-fixture router decode assertion. Every retry is now an immutable shell script and
health-gated. All completed fresh native/compiled health checks pass;
no xe-reset/reboot or DD action occurred. Full weight load is about155s,
so remaining comparisons share one load and reuse just-completed health.
