# S81 QSA prefill select source audit (2026-10-15)

CONFIG

Active scope remains Strata, small measured changes, C1/C2-C4, prefill2000+
and decode150+/stream. True RAM recovery is qualified in its retained scope;
concurrency/decode/MTP outrank chasing a hard2s prefix target. Re-read AGENTS.md,
goals and checkpoint8f680d8: S79 overlap parked on unmatched work; S80 grouped
GU passed the empirical full reference but exact mapping was much slower.
No backend reconsideration. This stage reads small retained summaries/source;
no full trace parse, model/payload, build, GPU, model call or production edit.

COMMAND

Read S30 prefill-phase-diagnostic.json, S10 retained4K operations.csv, S22
steady-summary/operations.csv and S38 report. Inspect actual prefill selection,
SYCL scorer/top-k dispatch, existing multi scorer and paged attention value
loads. Exact input paths/full small-file hashes are in source-pins.json;
selected line-numbered excerpts are in source-excerpts.txt.

RESULT: measured support and limits

S30 has two ordinal fresh199501-token prompt sequences. Equal12288 timed-token
calls have select224/225ms at ordinal1 and6278/6284ms at ordinal16; attention
2291/2303 ->2630/2636ms. Select totals51946/52045ms and attention41497/41649ms
across distinct calls within each sequence; these are separately attributable
primary event-timeline buckets, NOT removable kernel budgets or additive with
native wall/host/peer times. The final2886-token call stays separate. Timed work
totals199494, seven below prompt count: ordinals are not exact context offsets
or proof of A/B labels. PfTimer includes ordered-stream gaps/waits. The select
bucket combines scorer/top-k/debug work, not scorer-only service.

S10 actual4044-token main-prefill shapes: scorer192calls/24.052207ms across16
batch shapes, top-k192/7.364605ms, attention chunk1524/613.721646ms plus
merge1524/50.617593ms. These are traced operation sums, not request critical
path; tracing inflated wall. This demonstrates the original scorer is actually
used, while also bounding the short-context priority: it is small there.
S22 eight-cycle DECODE census uses multi scorer576calls/1.483884ms and wide
top-k576/0.618860ms (median2.395/1.145us). Those tiny decode-context results
cannot substitute for prefill256-query/long-context data or forecast a gain.
S38 alias retained OFF: 4K1303->1326.2,32K1546.1->1556.1tok/s; decode104.516
->104.017 aggregate. It changed peer MoE activation gather/IPC, not QSA select.

RESULT: exact source path

Every fourth layer (12/48) uses qsa_real_shapes:24Q heads,2KV heads,HD256;
indexer4heads x128FP32, blockR4, top_k2048. Width=min(n_kv,2051). Indexer
pooled rows are FP32[block,128], dead is FP32[128], query FP32[nq,4,128],
step records contain each query's n_kv/n_bid/width. Scores remain row-major
[nq,max_blocks], max_blocks=max_cells/4+2; ids[nq,2051] ascending. At configured
max_cells200000, max_blocks50002. No score/ID storage format changes proposed.

Prefill finishes the entire chunk indexer append before selection. It loops
sel_batch256, host-known active=last_query.n_bid+1, calls scores_tc then the
warp fallback and top-k on the same queue. SYCL scores_tc explicitly returns
false. Existing fallback uses256threads/WG,SG32,8warps; grid
(ceil(active/8),nq). Each warp computes one(query,block), reading the SAME
128-float pooled key again for each query and q's fourfloat4s again per block.
Per head: exact four-product expression, XOR16/8/4/2/1, positive-part, then
sum heads0..3. A query's n_bid uses dead, plus1e9 iff its tail is nonempty.
These boundary rules are essential even after the full chunk is appended.

An existing multi scorer already shares each warp-held key across up toMQ8
queries and stages8*4*128 floats in SLM. Its dispatch is restricted to nq<=8
AND active_blocks<=0, so ordinary positive-active prefill bypasses it.
The fixed grid is256WGs of256threads; eight warps stride key blocks by2048.
Declared staging is16384B query floats +128B of two8-entry int64 arrays,
before any compiler padding. No measured occupancy/resource claim follows.

The QSA attention fallback is distinct:32-query batches,33 chunks of64 cells
at cap2051,2KV heads,WG256/SG32, followed by24-head merge. Selected IDs resolve
paged rows once in local storage; INT8 K/V use per64 half scales; values fold
ascending cells with explicit fma per12 shared-query heads. Native prompt XMX
is opt-in and returns false when unset. This proposal changes none of that.

VERDICT: choose ONE source candidate

GO_SOURCE_PRIVATE_PREPARATION only: a PREFILL-ONLY eight-query tiled dispatch
of the EXISTING multi scorer arithmetic. Keep one host launch per current
256-query scoring call: add a query-tile grid dimension ceil(nq/8), retain
256key WGs/tile,256threads/WG/SG32/root_sync and the original key-stride loop.
Inside each tile offset q byq0*512, steps byq0*kStepCount, scores byq0*stride;
pass min(8,nq-q0). Existing multi body then keeps each query's own step bounds.
Limit reach to legal active/context; tile-uniform top/early return precedes
SLM staging, with converged existing barriers. Preserve original public decode
and nq<=8/active<=0 dispatch. Do not increaseMQ, add queues/allocations/events,
change top-k, concatenate matrices or port TF32/XMX math.

For every(query,block), load identical key/dead operands; retain the exact
four-product statement, XOR order, relu/head sum and tail bonus. Existing
max_cells/4+1 pooled rows include the spare last row; top<=max_cells/4 must
remain explicit before any speculative pooled read. Write only b<=n_bid with
unchanged max_blocks stride; inactive score bytes must not become readers.
For ragged tiles, q0=8*group_id_y and n=min(8,nq-q0): q-local i maps
EXACTLY to global q0+i in q/steps/scores; loops only visit i<n, never padded
queries. Require each n_bid+1<=active<=max_blocks, retaining the per-query
b>n_bid skip, including empty-tail score whose weight remains0. Do not clamp
step fields or replace dead by the later completed pooled tail. Query tiles
are independent, using read-only completed indexer state and separate rows. Top-k remains queued after all score tiles, same IDs and
same attention inputs. Frontend/compiler equivalence is not proved by source:
a future private original/public bit anchor, every active score and exact ID
comparison must precede timing. No implementation/build/GPU requested here.

This targets repeated key/query readiness and load issue, not fewer FLOPs:
a full tile can reuse one pooled-key load across8queries rather than8loads.
Actual cache traffic, register/SLM residency, serialization and net latency
are UNKNOWN. One warp serializes eight independent query computations, and
SLM/barrier cost may lose. The source mechanism is concrete, not a speed claim.

Applicability:4K fresh prefill has little measured scorer budget; no promise
of a useful short TTFT gain.32K and long200K fresh/remaining uncached prefill
execute this active positive-bound scorer at rising block counts; it can
address that growth. Fully cached RAM restore and steady decode are unaffected.
Long-context per-stage attribution still needs future evidence; do not call
all52s scorer service, forecast8xspeedup, or re-run full200K merely to explore.

Not repeats: S11 enlarged attention batch32/128/256 was neutral; S12 lane-cell
changed score sharing inside ATTENTION and was not broadly faster; S13 value
transpose/four-load ATTENTION pipeline was strict-pass neutral. S38 aliases
peer MoE activation gather. Here the INDEXER scorer reuses pooled FP32 keys
across independent queries without changing per-score reduction or selections.
No matching prefill key-reuse test was found in inspected history; this is
bounded search coverage. No new knob/format sweep is selected.

RESULT: supported prelink availability (source/table audit only)

S60 common8f5a7e wrapper does NOT include QSA scorer kernels. Its frozen
common-inputs.sha256 and exact build-container-v2.sh enumerate only native_mmvq,
iq_kernels and six core objects. S59 final symbols have no qsa_block_scores or
top-k symbol. No wrapper bytes were rehashed or images/ISA reparsed here.
Current qsa_select.dp.cpp.o exists154712B, SHA
4cac1286456dbcfbf3618e0f0734bd05aceff6d82624d88770bb5b5277da2c3d.
nm -C --defined-only shows public scores/topk/ref/tc/cluster functions;
ar t libstrata_kernels.a lists qsa_select.dp.cpp.o. A public-anchor oracle
therefore needs a SMALL separate supported device-image prelink of this
existing QSA object plus a private adapter image, then ordinary host link
selecting the matching archive member; cached S60 common may supply existing
core images if needed. Exact source/object/compiler/archive binding and
nonduplicate target registrations need root build audit. This is a concrete
input-availability recipe, not a build success or exact-source recompilation
proof. No main QSA TU recompile/common regeneration or huge ISA parse needed
for source preparation; actual link dependencies remain untested.
