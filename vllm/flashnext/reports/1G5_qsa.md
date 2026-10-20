# G5 -- QSA / indexer sparse attention for qwen4_exp on XPU

Date: 2026-10-20. Owner: G5. No commit.

Flash-Next full-attention layers are QSA, not dense GQA. The XPU decoder
from PR #61607 builds `Qwen3NextAttention` and does not run the indexer.
Dense GQA and QSA select the same cells only while
`n_kv <= idx_top_k + idx_block - 1`. On this artifact that bound is
`min(n_kv, 2048 + 4 - 1) = 2051` tokens. Past that, the selection is a
real top-k over cells.

This note is the Strata contract, the CPU reference that pins it, and the
XPU plugin that implements the decode path. Line numbers are the trees
read on 2026-10-20.

## What was verified

- CPU: `python3 -m flashnext_ref.test_qsa_layer` printed `QSA_LAYER_OK`
  (2026-10-20 08:30). Chunk prefill matches token-at-a-time. Radix top-k
  matches `np.lexsort` through n=2100, width=2051.
- Source reading: Strata `night-20261016` under
  `/mnt/vm_8tb/b70/build/strata-ab/src-ep`, and vLLM
  `vllm/models/qwen4_exp/` in the fork.
- XPU: card 1, 2026-10-20, `tests/run_gpu.sh` printed `QSA_XPU_OK`
  (exit 0, 16s, health after HEALTHY). Details in "XPU measurement".
  Greedy agreement past 4K tokens is still unmeasured. No speed claim.

## Geometry

From `flashnext_ref/geometry.py` and `qwen4exp_text_config.json`:

| item | value |
| --- | --- |
| layers | 48, 12 full-attention (blocks 3, 7, ..., 47) |
| hidden | 2560 |
| Q heads / KV heads / head dim | 24 / 2 / 256 |
| RoPE | dim 64, base 1e7, sections [11, 11, 10, 0] |
| RMS eps | 1e-6 |
| indexer | 4 query heads, dim 128, 1 key head |
| idx_block / idx_top_k / page | 4 / 2048 / 4 |
| selection cap | 2051 cells |

Text mRoPE has t=h=w, so 1D NeoX equals interleaved mRoPE
(`mrope.hpp` pair % 3; a null table returns the 1D position). A split
(t, h, w) triple is a different rotation. Position 0 is the identity
when the table is unscaled (sin row 0 is 0, cos row 0 is 1).

## Pipeline, one token

Strata `qsa.hpp` and `qsa.cu`. The CPU reference is
`flashnext_ref/qsa_layer.py`. `qsa.py` (G3) is unchanged.

1. Project Q, K, V, indexer Q, indexer K. Main Q is packed
   `[query | gate]` per head. Gate is the raw second half: not normed,
   not rotated (`qsa.hpp` `qsa_gate_apply`, `qsa.cu` line 518).
2. RMSNorm on Q, K, and indexer Q. The kernel multiplies by gamma.
   GGUF stores `(1+w)`. The fork loader's `TRANSFORM_DEVIATION` subtracts
   1 and `GemmaRMSNorm` adds it back (`layernorm.py` line 170,
   `weight + 1`). Pass the stored GGUF gamma, or `(1 + module.weight)`
   after that load. Indexer K is raw: no norm, no RoPE.
3. RoPE on Q, K, and indexer Q. K is already rotated before the KV store.
   V is never rotated (`qsa.hpp` line 180).
4. `kv_append` quantizes K and V into the int8 page (below).
5. `indexer_key_append` updates the tail and, on a block boundary, the
   pooled key (`qsa.cu` lines 155-243).
6. `qsa_index` scores every visible cell. ReLU is per indexer head, then
   summed. No `1/sqrt(d)`, no softmax (`qsa.hpp` lines 252-265).
7. `topk` keeps `width = min(n_kv, top_k + block - 1)` cells. Ties break
   to the smaller index. Ids come out ascending.
8. `kv_gather` dequantizes those cells to fp16.
9. `qsa_attend` is softmax GQA over the gathered cells only.
   KV head = `q_head // (n_q / n_kv)`, not modulo (`qsa.hpp` lines 287-294,
   `qsa.cu` line 458).
10. Gate: `sigmoid(raw second half) * attn` in double, then one store to
    fp16 (`qsa.cu` lines 513-520). SIGMOID, not SiLU.

There is no separate sliding dense window. The "dense window" is the
identity selection: while `n_kv <= 2051` every cell is kept, and the
open tail is forced in by `+1e9` even after that.

### Indexer keys

`tokens_per_state` in vLLM is `compress_ratio` (`common/qsa_cache.py`
line 616). Strata's block is the same 4.

Per cell `pos`, `slot = pos % 4`:

- `slot < 3`: store the raw f32 key in `tail[slot]`. The completing cell
  is not stored in the tail (`qsa.cu` lines 166-168).
- `pos == 0`: `dead = rope(rms(raw), angle 0)`. Also written to
  `pooled[0]`. Angle 0 even when `pos_base` or mRoPE is nonzero
  (`qsa.cu` lines 189-199). The kernel multiplies by cos of row 0
  because sin(0) is 0. The CPU reference runs the full pair rotation at
  position 0; those agree on an unscaled table.
- `slot == 3`: mean of the 3 tail rows plus the new raw, in double and
  in cell order (`qsa.cu` lines 208-211). Then
  `pooled[b] = rope(rms(mean), position of cell b*4)`.
  `pooled[b+1] = dead`. `block_pos = pos_base + b*4` (the position, not
  the cell index, and not the mRoPE t component).

RMS is `y * gamma` with the sum of squares in double (`qsa.cu` lines
183-188). Not `(1+w)` inside the kernel.

Pooled row `b` is the completed block. After a quiescent append the spare
sits at row `n_bid` and equals `dead`. A chunk append can already have
written a later block into `pooled[n_bid]`. Scoring must read `dead` for
the open block (`qsa_select.hpp` lines 8-11). At a decode boundary the
two aliases match, which is why token-at-a-time and chunk prefill produce
the same ids (CPU test, bit-exact on T=16).

RoPE of a completed row is in-place NeoX (`rope.hpp` `rope_neox_pair`):

```
oa = a*c - b*s
ob = a*s + b*c
```

Pair `d` pairs with `d + n_rot/2`. Read both partners before writing
either. With an mRoPE table, pair `d` uses
`table[(pos_base + b*4)*3 + (d % 3)]` (`mrope.hpp` line 51). The CPU
reference indexes the table by cell, which equals that absolute position
when `pos_base` is 0. The XPU kernel follows Strata.

### Scores and top-k

For query at cell `i`, `n_kv = i + 1` (prefill and decode). Cell `j` maps
to block `min(j / 4, n_bid)`.

```
score[j] = bias[block] + sum_h relu(dot(key[block], q_idx[h]))
score[j] += 1e9   if j is in the open tail (n_kv % 4 != 0 and j >= n_bid*4)
```

`key[block]` is `pooled[block]` for a finished block and `dead` for the
open one. The tail bonus is what keeps the current cell in the set once
the prompt is longer than 2051. Completed blocks do not get it.

Top-k (`qsa.cu` `topk_kernel`, lines 297-401): order the f32 by the
unsigned total order (sign bit flipped; `-0` canonicalized to `+0` in
Strata; NaN maps to 0 and is never selected). Ties break to the smaller
index. Strata does this with a 32-bit threshold plus a chunk walk. The
CPU reference and the XPU kernel pack `ordered_f32` in the high 32 bits
and `~index` in the low 32, then do a 64-bit MSB radix select and scan
`key >= marginal` ascending. Same ids on finite scores. Checked against
`np.lexsort((arange, -scores))` including planted ties and `+1e9`.

Strata's single-block top-k refuses `n_kv > 32768` (`qsa.hpp` line 282,
`kTopkMaxCells`). The long-context path is `qsa_block_topk` in
`qsa_select.hpp` (radix over blocks, then cells). The XPU kernel is one
work-group of 256 and scans every cell. It is correct past 32768. The
attend scratch caps the useful width at 4096; Flash-Next's width is 2051.

### Causal rule

A query at cell `i` sees a block only when that block's last cell is
`<= i`, plus tail tokens `<= i`. Implemented by scoring exactly `n_kv`
cells and letting top-k drop the rest. No extra mask.

### Strata cells vs vLLM blocks

vLLM `expand_qsa_block_indices` (`nvidia/ops/qsa_indexer.py` lines
443-454) keeps `token_topk / compress_ratio` whole compressed blocks
(512 blocks when top-k is 2048 and the ratio is 4) plus the causal tail
of the open group (0..3 tokens). Strata keeps 2051 cells and can cut the
marginal block.

CPU counterexample, block 4, top-k 4, width 7, T=16:

- Strata: `[4, 5, 6, 7, 12, 13, 14]`
- vLLM expand: `[4, 5, 6, 7]`

The XPU plugin follows Strata. `qsa_layer.vllm_block_expand_ids` is only
the contrast helper.

### int8 KV

`kv_q8.cu` lines 29-74. Group 64 along the head.

```
scale = fp16(max(abs(x)) / 127)     # stored, then widened back to f32
code  = clamp(rint(x / scale), -127, 127)
```

A zero group stores scale 0 and codes 0. `rint` is ties-to-even
(CPU: 0.5->0, 1.5->2, -1.5->-2). Quantize against the stored scale, not
the pre-round f32. Dequant for attention is `fp16(code * scale)`
(`kv_q8.cu` lines 95-106).

Layout, contiguous, same as the host pools:

```
codes  [page, kv_head, page_size, head_dim]  int8
scales [page, kv_head, page_size, head_dim/64]  fp16 bits in Strata,
                                                 f32 of that fp16 in the
                                                 CPU reference and the plugin
```

`page = page_table[pos / page_size]`, `slot = pos % page_size`. The page
table is not assumed to be the identity (`qsa.hpp` line 179). A negative
physical page means "not resident" in Strata's streaming copy; the plugin
always writes the physical page the table names.

Bytes per cell per layer: `2 * 256` int8 codes for K and for V, plus
`2 * 2 * (256/64) * 2` bytes of fp16 scales = 1024 + 32 = 1056.
Two KV heads, head dim 256.

vLLM's CUDA QSA cache is bf16 or fp8-e4m3 (`nvidia/qsa.py` lines 154-158).
It rejects a kv-quant dtype. The indexer cache there is a circular raw
ring plus a compressed MLA cache (`common/qsa_cache.py`), not Strata's
fp32 pooled keys. The daily driver flag is `--kv int8`. The plugin
implements that quant, not vLLM's fp8 cache.

### Decode vs prefill

Decode is one cell. `n_kv = pos + 1`. The step buffer is four int32s on
the device (`qsa.hpp` lines 156-161):

```
step[0] = pos
step[1] = pos + 1          # n_kv
step[2] = n_kv / block     # n_bid, completed blocks
step[3] = min(n_kv, top_k + block - 1)
```

A captured graph bakes host scalars and launch grids. Every per-token
count is read from `step`, and every grid is sized to a capacity
(`qsa.hpp` lines 326-340). `pos_base` is constant for the sequence and
may be a kernel argument. `pos` must not be.

Prefill in Strata can score many queries after the chunk is appended.
Each query uses its own `n_kv`. The CPU `prefill_chunk` matches
`prefill` because the open block reads `dead`. The XPU plugin's measured
path is one token per call (decode, and a Python loop for a short
prefill). The multi-query prefill kernel is not ported.

### Gate and attention numerics

Attend scale is `1/sqrt(head_dim)`. Softmax is fp32 over the fp16 K/V
scratch. Empty width writes zeros (`qsa.cu` lines 449-456).

The fp16 gate is what Strata stores before the output GEMV. There is also
`qsa_gate_apply_f32` for a caller that quantizes to Q8_K (`qsa.hpp` lines
314-324). The plugin stores the fp16 form. The output projection in the
hook casts that fp16 up to the residual dtype.

## Per-request state (for G4)

One sequence, one full-attention layer. This is what parking has to keep
if the next token must not recompute. G4's byte model in
`reports/2O2_kv_parking_spec.md` lines 81-88 matches these counts.

Fixed per layer (does not grow with the prompt):

| tensor | shape | bytes at Flash-Next |
| --- | --- | --- |
| tail | (3, 128) f32 raw of the open block | 1536 |
| dead | (128,) f32 spare key | 512 |
| block_pos | (1,) i32 | 4 |
| step | (4,) i32 | 16 |

`dead` aliases `pooled[n_bid]` only at a quiescent decode boundary. Park
it anyway. A restored chunk that continues mid-block, or a replay that
is not on a boundary, needs the spare even if the pooled row was
overwritten. `block_pos` is the last completion's first-cell position
(`pos_base + b*4`).

Grows with the prompt:

| tensor | shape | bytes |
| --- | --- | --- |
| pooled | (n_kv/4 + 1, 128) f32 | 128 B/token, plus one spare row |
| K codes | int8, 2 * 256 per token | 512 |
| V codes | int8, 2 * 256 per token | 512 |
| K scales | fp16, 2 * 4 per token | 16 |
| V scales | fp16, 2 * 4 per token | 16 |

12 full-attention layers: `12 * (1056 + 128) = 14208` B/token, plus the
MTP draft layer's int8 KV with no pooled rows (`1056`) =
`15264` B/token. The CPU test checks that slope. Fixed indexer state is
about 2 KB/layer (tail + dead + block_pos), times 12.

Also required, not inside the tensors above:

- `page_table`: int32 logical page -> physical page. `page_size = 4`.
  Identity is legal. A permutation is legal (CPU test gathers through a
  swapped table).
- `pos_base`: int32, constant per sequence. Cell `j` rotates at
  `pos_base + j` unless mRoPE overrides the pair.
- mRoPE table, if any image tokens exist: int32 `[cells, 3]` of
  `(t, h, w)`, indexed by absolute position. Text does not need it.
- cos/sin tables are weights of the rope, not request state. They can be
  rebuilt. The pooled keys cannot: they already include the rotation.

Do not drop `tail` or `dead` because `pooled[n_bid] == dead` happened to
hold at the park point. The next cell's slot is `pos % 4`, and the mean
on the next boundary reads the tail.

## CPU reference

`flashnext_ref/qsa_layer.py`.

- `QsaGeom`, `QsaState`, `QsaWeights`.
- `append_indexer`, `append_kv`, `scores`, `select`, `gather`.
- `decode_token`, `prefill` (token oracle), `prefill_chunk`.
- `quantize_q8` / `dequant_q8`, `topk_radix`, `vllm_block_expand_ids`.

Gamma contract is documented on `QsaWeights`: the multiply weight, not
`(1+w)` a second time. Rope tables match `rope_neox`. The apply helper
copies both halves before writing; an in-place first-half write aliases
the partner and breaks pair 0.

`qsa.py` was not edited, so G3's `compare_layers.stage_self` still holds.

## XPU plugin

`vllm/flashnext/plugins/b70_qsa_xpu/`.

SYCL source: `csrc/qsa_kernels.cpp`. Built to
`/mnt/vm_8tb/b70/build/vllm-flashnext/b70_qsa_xpu/libb70_qsa_xpu.so`
(`B70_QSA_SO`). Ops are `b70_qsa::{write_step, kv_append, indexer_append,
index_scores, topk, kv_gather, attend, gate}`. All are in-place into
caller storage. Grids are capacities. `step` is device memory.

The formulas are attributed in the cpp header to the Strata files above.
This is a SIMT port of the decode math, not a paste of the CUDA, and not
the XMX union prefill (`sycl/src/kernels/cuda/qsa_prompt_attn_union.dp.cpp`,
joint_matrix 8x16x16, fixed head dim 256, 8-query tiles). That kernel is
the 2000 tok/s prefill path. Porting it is size L. It returns false unless
the KV is int8 and the shape is 24/2/256, so it belongs on top of this
quant, not instead of it.

Intel GPU fp64 is emulated. The indexer RMS and the gate sigmoid use
double so the reduction order matches `qsa.cu`. Indexer dots in
`index_scores` are fp32, matching `qsa.py` `indexer_scores` (Strata's
per-token index kernel accumulates the dot in double; the long-context
`qsa_block_scores` is already fp32).

Graph capture: `write_step` is one thread and reads `pos[0]` from a
device tensor. Later kernels read `step`. Host arguments are geometry
and `pos_base`. Attend local memory is sized to the ids capacity, checked
`<= 4096`. A work-group past `n_bid` returns before any barrier.

The int8 scale is `f16_to_f32(f32_to_f16(amax / 127))` in
`csrc/qsa_kernels.cpp`. `float(sycl::half(float))` was compiled back to
the original f32 on icpx 2026.1, so the first card-1 run quantized
against the unrounded scale. The explicit RTNE bit round trip matches
numpy float16. Codes then use `sycl::rint` and clamp to [-127, 127].

### Fork hook (C7)

Uncommitted edit on fork `b70/flashnext`,
`vllm/models/qwen4_exp/xpu/model.py`:

- `B70_QSA=1` imports `b70_qsa_xpu.model_hook.attach_qsa`.
- Default off. The dense `Qwen3NextAttention` path is unchanged.
- The runner is attached only when the layer built indexer weights
  (`b70_gguf`). W4A16 checkpoints have none; `attach_qsa` returns None.
- Forward calls the runner instead of `self.self_attn`. vLLM's paged KV
  is not updated. The runner's own int8 pool is the QSA cache.
- TP=1 only. One sequence. Positions are the cell indices.
- `B70_QSA_MAX_CELLS` defaults to 8192, not 262144.
- First forward allocates. Do not capture that forward.
- Indexer K gamma is `(1 + k_layernorm.weight)`.
- Main Q/K come from `_project_qkv_gate` (already normed and rotated).
  Set `B70_EAGER_QK=1` when comparing numerics; the fused Triton
  qk-norm-rope path is the wrong debug path.
- Image mRoPE: the runner writes the token triple into an `[cells, 3]`
  table and ropes the indexer query with pair `% 3`. Not in the GPU
  unit test. Text 1D is the tested rotation.
- MTP draft (`xpu/mtp.py`) does not construct this attention. Draft QSA
  is a separate work item. Park the draft's 1056 B/token anyway; it has
  no pooled rows.

C7: this is one default-off edit plus the new plugin. Please keep it as
its own commit when you rebase. I did not commit.

## XPU measurement

Card 1, 2026-10-20. Library
`/mnt/vm_8tb/b70/build/vllm-flashnext/b70_qsa_xpu/libb70_qsa_xpu.so`
is 369040 bytes, built 09:07 with icpx 2026.1 for `intel_gpu_bmg_g31`.
Image `vllm-flashnext-xpu:py312`. Host UMD via `harness/xpu-run.sh`
(libze 1.28.2, IGC 2.36.3). `xpu-run.sh` already takes the `bin/gpu-run`
lease. Do not wrap this script in a second `gpu-run`.

```
CARD=1 B70_AGENT=G5 B70_GPU_LOCK_TIMEOUT=7200 \
  vllm/flashnext/plugins/b70_qsa_xpu/tests/run_gpu.sh
```

CONFIG: one card, `tests/test_qsa_xpu.py` against `QsaState`, seed 2.
Small geom is n_head=4, n_kv=2, head_dim=64, n_rot=8, indexer 2x32,
block=4, top_k=4. Wide geom is head_dim=256, n_rot=64, indexer 4x128,
top_k=4, T=10. Tolerances: ids, codes, scales, tail, block_pos exact;
dead/pooled 1e-5; scores 1e-4; attn/gated 2e-3; gathered K/V fp16 1e-3.

An earlier card-1 run of the `sycl::half` scale path exited 1 in 23s.
K input matched (max_abs 0). Device scales were the raw f32
`amax/127` (bits 1015215198, 1018736225 =
0.01598566398, 0.02254408784). CPU numpy float16 scales were bits
1015218176, 1018732544 = 0.0159912109375, 0.0225372314453125.
`k_codes[0,0,0,20]` was -103 against the reference -102. One fp16 step
down of that scale also flips index 24, so the device value sat between
the two bins. Health after that fail was HEALTHY. The rebuild stores
the scale with the explicit RTNE bit round trip above.

RESULT of the rebuilt library, same command, card 1 acquired with no
wait: `QSA_XPU_OK`, `gpu-run` done in 16s, exit 0. `xpu-health --card 1`
after: HEALTHY, rc 0. No wedge, no `xe-reset`.

Passed, all ids exact:

- top-k, including n=2100 width=2051
- zero group (codes 0, scales 0)
- small T=8, including the all-zero K/V token
- reversed page table, `pos_base=100`, split mRoPE triples
- head_dim 256, T=10
- graph captured while executing pos 4, replayed at pos 4, pos 5, and
  pos 7 (slot 3, a completing cell). Replay ids, tail, dead, pooled,
  block_pos, codes, and scales exact versus the eager GPU run. Replay
  attn and gated max_abs 0.

Codes and scales max_abs 0 on every checked position, including
head_dim 256 pos 9. Gathered fp16 K/V max_abs 0. Largest residuals
inside the stated atols, head_dim 256: pooled 2.384e-07, scores
9.537e-07 (pos 8), attn 7.153e-07 (pos 9), gated 2.441e-04 (pos 8).
This is a correctness run. It is not a latency number.

Greedy agreement with `harness/strata_reference.json` on prompts longer
than 4K is blocked until C7/H1 can serve Flash-Next. The short greedy
rows already collected (17*19->323, 23*29->667, thinking 23*17->391)
are below the 2051 bound, so dense GQA would match them. They do not
exercise QSA. Do not start `hotschmoe-dd` for this. The hook in
`vllm/models/qwen4_exp/xpu/model.py` (docstring line 17, `qsa_runner`
at 566-572, forward at 613-614) is uncommitted and was not exercised
by this unit test.

## Work left

| item | size | note |
| --- | --- | --- |
| XMX union prefill (`qsa_prompt_attn_union.dp.cpp`) | L | on top of this int8 KV, HD=256, 24/2 |
| multi-query prefill index (`qsa_block_scores` / `qsa_block_topk`) | M | decode scan is enough under 8K cells |
| serve wiring: vLLM block table, slots, batch>1 | L | hook is one sequence, own pool |
| MTP draft QSA | M | no indexer module on the draft layer yet |
| greedy >4K vs `strata_reference.json` | S once a serve exists | not a kernel gap |
| park/restore of tail, dead, pooled, block_pos, int8 pages | M | G4, layout above |

## 2026-10-20 10:38 G5b -- page spec, serve hook, fork commit

The 09:07 note that the fork hook is uncommitted is stale. At 10:10 C7
had not committed it. G5b committed only that file:

    9761b7b48af63922e8be4599aebda548c8ba77e4
    [XPU] Attach the default-off QSA plugin when B70_QSA=1

Not pushed. G4's later 765993c197 sits on top of it. The fork worktree
is clean. No further fork edits in this step.

SYCL library rebuilt 10:16 under bin/cpu-run, rc 0:

    /mnt/vm_8tb/b70/build/vllm-flashnext/b70_qsa_xpu/libb70_qsa_xpu.so
    817928 bytes, icpx 2026.1, qsa_kernels.cpp + qsa_batch.cpp

qsa_kernels.cpp signatures are unchanged, including the explicit RTNE
fp16 scale round trip. The new translation unit is the batch SIMT path
(kv append, indexer, scores, top-k, gather, attend, gate, union build,
union attend). It is compiled. It is not yet compared to the CPU
reference on a GPU.

### KV spec (QSA_SPEC_OK)

plugins/b70_qsa_xpu/b70_qsa_xpu/kv_spec.py. One uint8 state per vLLM
block, state_content_bytes = page_bytes from layout.py. Pooled Flash-Next
block 16 is 18944 bytes. Draft (no pooled rows) is 16896. The registry
manager is FullAttentionManager and the uniform base is FullAttentionSpec,
so merge keeps the page size. B70QsaBackend.get_supported_kernel_block_sizes
returns the manager block size, and select_common_block_size keeps 16
and 64 whole. A backend that advertises only kernel block 16 does split
a 64-token manager block; that is the split the custom backend avoids.
create_kv_cache_views of a uint8 page is [num_blocks, 1, 1, page_bytes]
and a byte write lands at block_stride = page_bytes.

The metadata builder copies block_table, slot_mapping, query_start_loc,
and seq_lens. Cudagraph support is UNIFORM_SINGLE_TOKEN_DECODE, so a
one-token-per-request decode can be captured and a chunked prefill stays
eager. get_impl_cls still raises. Attention runs in the plugin.

### Serve hook (QSA_SERVE_OK, CPU)

model_hook.py patches get_kv_cache_spec and get_attn_backend on the
attention instance inside attach_qsa. The fork diff does not grow.
When attn.kv_cache is a contiguous uint8 buffer whose last dimension
is the page, the runner binds it and writes through the metadata block
table. A split kernel block is refused and the private one-sequence
pool remains, with one stderr line. The first bind allocates. Do not
capture that forward.

Indexer tail, dead, block_pos, and the token count are keyed by the
physical id of logical block 0, so a request can change batch rows
between steps. pos < 0 is inactive. Each inactive row scatters to its
own sink slot, because index_copy_ is undefined on a repeated index.
Position 0 clears the gathered open block before the kernel runs.
Import writes only the logical blocks named in the blob. It does not
walk padding physical id 0, which may be another request's page.
pos_base on this path is 0: pooled rows were rotated at absolute
positions. export_state(seq) / import_state(seq, blob) use the live
block-table row. export_request(phys) / import_request(dest, blob)
take an explicit physical list for G4.

Decode (max_query_len == 1) is one run_decode over the batch. Chunked
prefill loops tokens on the host. That loop is the correctness path.
The SIMT union kernel is the fast path and is not wired into the hook
until the GPU test prints QSA_BATCH_OK.

Draft: attach_draft(layer) and mark_draft(layer) select the no-pool
page. set_draft_ids supplies the target selection. The existing fork
hook calls attach_qsa, which is pooled. C7 owns the one-line MTP call.
G5b did not edit mtp.py.

QSA_SERVE_OK does not launch a kernel. It checks the page view, the
external storage alias, gather/scatter, a physical remap, and the spec
patch. GPU numeric agreement of the batch kernels is still the queued
job.

### GPU job

CARD=0, B70_AGENT=G5b, timeout 7200 s, log
/mnt/vm_8tb/b70/build/vllm-flashnext/logs/g5b-qsa-batch.log.
xpu-health before card 0: HEALTHY, rc 0, gpu-run done in 26 s, lease
released. xpu-run is waiting on card 0, held by H1 pid 3218685 since
10:22:06 (bin/xpu-health). No QSA_XPU_OK or QSA_BATCH_OK in this log
yet. No xe-reset. The .so was not rebuilt after 10:16.

Greedy agreement with harness/strata_reference.json on prompts longer
than 4K is still blocked until Flash-Next serves. Do not start
hotschmoe-dd for it.

### Work left at 10:38

| item | size | note |
| --- | --- | --- |
| batch SIMT vs CPU, including graph replay | M, running | queued on card 0 behind H1 |
| XMX union prefill, mode 1 only | L | after SIMT is green; else report SIMT |
| hook uses union for chunked prefill | S once SIMT is green | host loop is the current prefill |
| MTP draft call from C7 plus a GPU draft check | S + the queued test | mark_draft and set_draft_ids exist |
| greedy >4K vs strata_reference.json | S once a serve exists | not a kernel gap |
| park GPU round trip | S | CPU remap passed; GPU is in test_qsa_batch.py |

## 2026-10-20 10:55 G5b -- SIMT batch measured

The 10:38 line that xpu-run was still waiting is stale. Card 0 ran
the suite at 10:40. QSA_XPU_OK. Every eager batch check passed,
including chunked prefill, SIMT union attend (words exact, attn
max_abs 7.007e-4), draft pages, and export/import. Graph replay of
the page, tail, dead, block_pos, and length was exact. The run then
failed "FAIL graph ids mismatch" because the compare included the
unwritten id tail after fill_(-1). Health after was HEALTHY. No
xe-reset. Log logs/g5b-qsa-batch.log. RUN_EXIT:1.

The compare is now the live prefix, ids[:, :width], matching the
single-sequence graph test. Card 1 rerun, launcher 10:46:43, log
logs/g5b-qsa-batch-r2.log:

- QSA_XPU_OK. Single-seq graph ids exact at widths 5, 6, and 7.
- QSA_BATCH_OK. Graph pos 4 and pos 5 exact on pages, tail, dead,
  block_pos, length, attn, gated, and the id prefix.
- Inactive seq 1 during a slot-3 completion kept its pages, tail,
  and dead. Seq 0 ids exact, attn and gated max_abs 0.
- Health before and after HEALTHY. gpu-run 24s, RUN_EXIT:0.

Codes, scales, and ids were exact. Pooled stayed inside 1e-5,
scores inside 1e-4, attn inside 2e-3. Not a speed claim.

XMX mode 1 (Strata union_attn, page-local addressing, no dpct) is
compiling to libb70_qsa_xpu_next.so. It is not measured. The 817928
byte library is unchanged. The hook prefill is still the host token
loop. Greedy prompts past 4K stay blocked until Flash-Next serves.

### Work left at 10:55

| item | size | note |
| --- | --- | --- |
| XMX union vs SIMT, 24/2/256, atol 2e-2 | M, compiling | mode 1 only; drop the TU if it will not match |
| hook prefill uses score + union | S | host loop is still the prefill |
| MTP mark_draft from C7 | S | API and GPU draft check passed |
| greedy >4K vs strata_reference.json | S once a serve exists | not a kernel gap |

## 2026-10-20 11:17 G5b -- XMX mode 1 measured

Card 0, log logs/g5b-qsa-xmx.log. Library 962040 bytes (the SIMT
817928-byte file is libb70_qsa_xpu.so.simt-817928). Health before
and after HEALTHY. Container 32s, exit 0. RUN_EXIT:0.

- QSA_XPU_OK and QSA_BATCH_OK still printed on this library.
- PASS xmx refuse: a 4/64 shape returns ok=0 and does not launch.
- 24/2/256, union of 58 cells, ok=1.
- vs SIMT max_abs 9.710e-04 (atol 2e-2).
- vs CPU attend_ids max_abs 5.405e-04 (atol 5e-2).
- QSA_XMX_OK. Not bitwise. No speed claim. No xe-reset.

The 10:55 sentence that XMX was still compiling is stale. Mode 1
stays. Mode 2 is not ported.

Hook prefill for pooled pages now appends one token at a time
(append_cells) and then union-attends that request in tiles of 8.
XMX runs when ok is 1. SIMT covers every other shape. Decode stays
the graph path and does not call the union. Draft prefill still
uses the caller-supplied ids. The composition test is queued
(logs/g5b-qsa-prefill.log) behind C7's both-card serve. The hook
class itself has not been executed on a GPU.

Greedy prompts past 4K remain unmeasured. C7's current window is
max_model_len 2048, which is under the 2051 selection bound.
G5b will not start hotschmoe-dd.

### Work left at 11:17

| item | size | note |
| --- | --- | --- |
| prefill_chunk GPU vs CPU, tiles 8+2 | M, queued | card 0 behind C7 |
| hook forward on a live model | S once a QSA serve exists | composition test does not call QsaRunner |
| MTP mark_draft from C7 | S | API and the earlier GPU draft check passed |
| greedy >4K vs strata_reference.json | S once max_model_len > 2051 | not a kernel gap |

## 2026-10-20 11:21 G5b -- prefill tiles measured

The 11:17 line that the composition test was still queued is stale.
Card 0, log logs/g5b-qsa-prefill.log. Health before and after
HEALTHY. Container 26s, exit 0. RUN_EXIT:0.

- QSA_XPU_OK, QSA_BATCH_OK, QSA_XMX_OK on the same run.
- Small geom, 10 queries, tiles of 8 and 2: append_cells matched
  the CPU page, the other request's pages stayed empty, ids exact,
  SIMT gated worst max_abs 1.953e-3 (atol 2e-3). XMX did not launch.
- 24/2/256, 16 cells, 10 queries: XMX launched. Gated vs CPU
  max_abs 9.766e-4 (atol 5e-2). The hand-built 58-cell union
  repeated at 9.710e-4 vs SIMT and 5.405e-4 vs CPU.
- Not bitwise. No speed claim. No xe-reset.

Pooled hook prefill calls this path. Decode stays on run_decode
inside the graph. Draft prefill still uses caller ids. The runner
class was not executed on the device.

### Work left at 11:21

| item | size | note |
| --- | --- | --- |
| QsaRunner.forward on a live model | S once a QSA serve exists | the GPU test calls QsaBatch |
| MTP mark_draft from C7 | S | API and the GPU draft check already passed |
| greedy >4K vs strata_reference.json | S once max_model_len > 2051 | not a kernel gap |
