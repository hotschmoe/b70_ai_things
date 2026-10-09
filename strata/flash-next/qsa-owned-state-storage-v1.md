# Conditional owned QSA history/selection storage model v1

CONFIG -> New bounded CPU source lane; frozen original_math_scalar.py and its
qsa_selected_attention/indexer functions remain unchanged. No actual original
weight payload, GPU/runtime, SDK, commit or JOURNAL work during parent controls.
Supplied per-layer inputs are explicitly conditional, not whole-model residual
or48-layer composition. Header inventory only supplies role/type/shape receipts.

COMMAND ->

```
python3 strata/flash-next/test_qsa_owned_state_storage_v1.py
python3 -m py_compile strata/flash-next/qsa_owned_state_storage_v1.py
```

RESULT -> Fifteen CPU controls PASS. Indexer4-cell pooling/spare/dead/tail,
causal finite tail bias, ascending lower-ID ties, per-head ReLU, actual
2048+3 selection-width boundary, FP16 KV, NEOX partial RoPE, head24/2 division,
order/history/reset and replayed main/slot mathematical checkpoints are tested.
At64/65 cells there are16/17 four-cell KV pages:64 is a tested history boundary,
NOT a source pooling granule. Header role inspection reads no weight payload.

VERDICT -> Owned local mathematical/storage state and selection implementation;
not a native arithmetic, checkpoint or full-model qualification. No new numeric
threshold/gate is inferred. Native exp/rsqrt/RoPE/score/attention/GEMM/FMA/top-k
rounding remain unsupported; actual runs and comparisons are UNOBSERVED.

## Authoritative source and original header contracts

Source hashes and inventory fingerprint are pinned in the source plan.
include/strata/kernels/qsa.hpp:96-138 defines24 query heads,2 KV heads,256dims,
64 rotated dimensions,4 indexer heads of128dims, indexerblock4, topk2048 and KV
page4. Selection width is min(nkv,2051), not2048 and not64. Rope base is1e7.
Original GGUF headers give dimension sections[11,11,10,0], dimension_count64 and
no required scaling in this local lane; text axes all share the same position.
Multimodal axis positions and YaRN remain unsupported.

Header-only source metadata proves the selected original QSA roles:

| Role suffix | Type | GGML shape |
| --- | --- | --- |
| attn_q.weight | Q8_0 | [2560,12288] |
| attn_k.weight / attn_v.weight | Q8_0 | [2560,512] |
| attn_output.weight | Q8_0 | [6144,2560] |
| attn_q_norm.weight / attn_k_norm.weight | F32 | [256] |
| indexer.k_proj.weight | BF16 | [2560,128] |
| indexer.q_proj.weight | BF16 | [2560,512] |
| indexer.q_norm.weight / indexer.k_norm.weight | F32 | [128] |

The actual inventory shows all twelve attn_q weights Q8_0. Stale source comments
about Q2_0/Q3_K/IQ4_XS from another artifact are not admitted as current identity.
QsaConditionalProjection admits only exact role shapes/types for QSA layer indices
3,7,...47; each requested layer is validated independently before rows/project.
It accepts supplied HC attention mixed input, clearly conditional. Native decode
Q8_0 consumes own Q8_1 image; prefill uses original Q8_0 dequantizedF16 weights and
F16 inputs. Original BF16 indexer weights consume F32 decode or BF16 prefill
inputs. Projections are FP64 mathematical dots with F32 stores, not exact native
reductions. Full residual/HC and other-layer generation are outside this helper.

## Owned resident FP16 state and selection

sycl/src/core/layer.cpp:1251-1315 projects RAW indexer key before normalization,
normalizes/RoPEs K, splits per-query-head q/gate halves, normalizes/RoPEs q and
indexer queries, and appends normalized K plus unnormalized V to residentFP16
KV. NEOX pairs are(i,i+rot/2), with tail dimensions unchanged. Query-to-KV map is
head//12, not head%2. Native attention gating stays F32 before output Q8_1
projection; the frozen scalar helper's declared-storage F16 gate output is not
silently substituted for this native source path.

native_qsa_indexer.dp.cpp:79-168 rounds raw key through F16 then widens to F32.
Positions0..2 update three chronological tail slots. Atpos0, four copies of cell0
initialize normalized/RoPE spare and dead at zero angle even with pos_base.
At4b+3, oldest-first four raw F16 keys are added with materialized F32 sums and
scaled by.25, then RMS+gamma and RoPE atpos_base+4b. pooled[b] receives the result,
pooled[b+1] receives fixed dead, and block_pos receives that anchor. Incomplete
blocks use dead, not a pool of future tokens. The new state owns all of these.

qsa_select.dp.cpp:40-86 / qsa.dp.cpp:323-405 score each pooled key by
sum_head(ReLU(dot(query_head,key))) with no softmax/attention scaling. Incomplete
tail cells get finite1e9 bias; completed blocks do not. Scores are stored F32 in
this model. Selection owns min(nkv,2051) best cells with low-ID ties, returning
ascending cell IDs. Attention uses only those causal selected rows with scaled
softmax over FP16 KV, division head map and sigmoid gate. Exact native score
ordering/reduction/intrinsic equivalence remains false, including near ties.
No captured selected-ID list enters advance; its local attention method can
validate a candidate roster for diagnostics, not claim independent selection.

## Checkpoint/main-slot reconstruction and actual gap

QsaOwnedState records token labels and F32 conditional input history, owns FP16
KV, tail/dead/pooled/block_pos and derives step counts from owned history length.
checkpoint binds geometry/pos_base/key-gamma and SHA of token/order/input bytes.
restore requires caller's expected committed token prefix, replays inputs from
zero, and verifies complete reconstructed KV/tail/dead/pooled/block_pos against
the snapshot. It never restores selected IDs or relies on stale score scratch.
A resumed independent slot and uninterrupted main produce identical synthetic
local outputs. Altered prefix/order/tail/dead/blockpos are rejected.

Actual conversation_state.cpp:168-197 saves/loads indexer tail/dead/block_pos;
restoration reconstructs spare pooled[ids.size()/4] from dead. Prefix completed
pooled rows and KV require their retained backing snapshots, not just that small
checkpoint. All stage/slot owners must retain the correct token boundary and
physical resident page mapping. The CPU replay model does not prove actual GPU
snapshot bytes or main/slot copies. Those remain an explicit runtime gate.

Actual compiled layer.cpp:1038-1040 qsa_state_zero clears tail/dead/pooled but
omits idx_block_pos and step/position metadata. Independent reset sets logical
block_pos0/history empty. Frozen compiled behavior is preserved and this known
source gap is not masked by the reference. Separate root reset source work must
be built/measured; no full-state reset/checkpoint claim follows from these tests.

This lane remains conditional on supplied layer inputs, despite independently
owning selection/history. GDN/PLE/HC/FFN/upstream residuals, all48 compositions,
cache/concurrency/full-model math and natural-language/latency/stability/shelf
qualification remain outstanding. Any actual discrepancy must be preserved;
no threshold, selection rule or format is weakened to manufacture PASS.
