# A1 -- vLLM thin fork, Qwen4Exp GGUF port

Branch: `b70/flashnext` in `/mnt/vm_8tb/b70/build/vllm-flashnext/vllm`.
Tracks `origin/main`. Ahead 4. Not pushed.

Commit list (`git log --oneline origin/main..HEAD`), tip first:

```
7b40eb79c6 [XPU] Load Qwen4Exp PLE host table, indexer weights, and MTP eh_proj
cadd70b850 [XPU] Add b70 GGUF quant hook and qwen4_exp weight map
fd045ff3a0 [XPU] Pin vllm_xpu_kernels 0.1.16.1 for Qwen4Exp paged decode
4e70b2d300 [XPU] Add Qwen4Exp (Qwen3.8-Flash-Next) model architecture support
```

`4e70b2d300` is the cherry-pick of open PR vllm-project/vllm#61607 (the
patch commit, author zhaohb kept). It was `0ad6bf6338` before the rebase.
Nothing else from that PR's dependency list was required; the pick applied
on main by itself.

## CONFIG

- Clone: `/mnt/vm_8tb/b70/build/vllm-flashnext/vllm`
- Base after rebase: `origin/main` `2296c12e47`
  (`[Doc] Document DeepSeek-V4.1-Flash B300 Dynamo disaggregated profile (#62561)`)
- HEAD: `7b40eb79c648f3d904fa65ec9393b6f1f28e9bce`
- `git merge-base --is-ancestor origin/main HEAD` succeeded. Rebase of the
  four commits onto the fetched main was clean (no conflicts). The five
  commits that arrived on main (`53172fa72e..2296c12e47`) do not touch
  `vllm/models/qwen4_exp`, `requirements/xpu.txt`, `pyproject.toml`, or
  the fused-MoE / linear / quantization paths.
- Interpreter: `/home/hotschmoe/.local/share/uv/python/cpython-3.12.13-linux-x86_64-gnu/bin/python3.12`
- `PYTHONPATH`: clone, then
  `/mnt/vm_8tb/b70/build/vllm-flashnext/venv/lib/python3.12/site-packages`,
  then `/mnt/vm_8tb/b70/build/strata-ab/llama-pinned/gguf-py`
- Env: `ONEAPI_DEVICE_SELECTOR=opencl:cpu`, `ZE_AFFINITY_MASK=""`,
  `VLLM_TARGET_DEVICE=cpu`, `VLLM_NO_USAGE_STATS=1`
- pytest is not installed. Each `test_*` function was called directly.
- No GPU. No `bin/gpu-run`. No systemd change. The 26.8 GiB PLE payload
  was not copied; `GGUFReader` supplied the header (`n_bytes`, `shape`).
- Git identity in the clone was already `hotschmoe` / `stronggarner66@gmail.com`.
  It was not changed.

Files the fork calls:

- Quant config: `vllm/models/qwen4_exp/gguf/config.py:46` (`get_quant_method`)
- Ops: `vllm/models/qwen4_exp/gguf/ops_interface.py:57` (`linear` / `gemv`),
  `:140` (`moe`), `:256` (`dequant`), `:275` (`gather_rows`)
- Types: `vllm/models/qwen4_exp/gguf/types.py:23` (Q2_K, Q3_K) and `:36` (Q2_0)
- Names: `vllm/models/qwen4_exp/gguf/mapping.py:424` (`_under_mtp`),
  `:489` (`consume_names`)
- PLE host table: `vllm/models/qwen4_exp/gguf/ple.py:60` (`gather_host_rows`),
  `:88` (`assign_ple_hash`)
- Loader: `vllm/models/qwen4_exp/gguf/loader.py:71` (`load_mapped_tensor`)
- XPU hooks: `vllm/models/qwen4_exp/xpu/model.py:428` (skip the BF16
  `PLE_TABLE_PATH` when `host_table` is set), `:556` (indexer weights),
  `xpu/mtp.py:156` (`attach_gguf_eh_proj`),
  `xpu/indexer_weights.py:19`
- Entry point: `pyproject.toml` `b70_gguf = vllm.models.qwen4_exp.gguf:register`
  Tests call `register()` directly. A reinstall is required before the
  entry point is visible to a serve process.
- Kernel pin: `requirements/xpu.txt` `vllm_xpu_kernels==0.1.16.1`
  (was `0.1.15.4`). The package was not installed into the shared venv
  by this agent.

A2 op definitions the fork binds to, unchanged by A1:
`vllm/flashnext/plugins/b70_gguf_xpu/csrc/bind.cpp:94-105`.

## COMMAND

```
cd /mnt/vm_8tb/b70/build/vllm-flashnext/vllm
git fetch origin && git rebase origin/main
PY=/home/hotschmoe/.local/share/uv/python/cpython-3.12.13-linux-x86_64-gnu/bin/python3.12
SP=/mnt/vm_8tb/b70/build/vllm-flashnext/venv/lib/python3.12/site-packages
GGUF_PY=/mnt/vm_8tb/b70/build/strata-ab/llama-pinned/gguf-py
PYTHONPATH="$PWD:$SP:$GGUF_PY" \
  ONEAPI_DEVICE_SELECTOR=opencl:cpu ZE_AFFINITY_MASK="" \
  VLLM_TARGET_DEVICE=cpu VLLM_NO_USAGE_STATS=1 \
  "$PY" -c 'import tests.models.qwen4_exp.test_b70_gguf_mapping as t; ...'
```

The same 14 functions were run once before the rebase and once after.
Both runs exited 0.

A separate header walk (no vLLM import, no payload copy) printed expert
shapes from
`/mnt/cache/b70/devmodel/unsloth/Qwen3.6-35B-A3B-MTP-GGUF/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf`.

## RESULT

CPU suite after the rebase, 14/14 PASS, exit 0, about 87 s:

- `test_cpu_dequant_matches_gguf_py_one_row`
- `test_dense_transforms_and_gdn_rows`
- `test_ep_slice_two_ranks`
- `test_gather_rows_keeps_table_on_cpu`
- `test_iq3_xxs_every_tensor_once`
- `test_linear_and_moe_call_ops`
- `test_meta_qwen4exp_constructs`
- `test_mtp_sidecar_every_tensor_once`
- `test_ple_metadata_array_lengths`
- `test_ple_shard_is_one_host_table`
- `test_q2_0_and_histogram_types`
- `test_quant_config_registers`
- `test_qwen35_moe_schema_consumed_once`
- `test_qwen36_iq3_xxs_file_every_tensor_once`

Import noise, expected on this CPU launch: `Failed to import from vllm._C_AVX2`,
and `XPU device count is zero` under `opencl:cpu`.

Name coverage (every tensor consumed once):

| file | tensors | arch | check |
| --- | --- | --- | --- |
| IQ3_XXS shard 1 | 1223 | qwen4exp | Q2_0 count 38, `ple_layer_ids` `[2]`, no MTP block |
| IQ3_S shard 2 | 1 | qwen4exp | `per_layer_token_embd.weight`, IQ4_NL, raw ne `(160, 320001536)`, `n_bytes` 28800138240 |
| MTP sidecar | 34 | qwen4exp | block 48 maps to `model.layers.0` and `model.eh_proj.weight` |
| Qwen3.6 UD-IQ3_XXS | 753 | qwen35moe | block 40 maps under `mtp.`, `blk.0` stays `model.layers.0` |

Qwen3.6 type histogram, measured from that header:
`{0:368, 8:8, 10:2, 11:1, 14:252, 21:39, 22:78, 23:3, 30:2}`.
Q2_0 count is 0. The two Q2_K tensors are
`blk.40.ffn_gate_exps.weight` and `blk.40.ffn_up_exps.weight`,
raw ne `(2048, 512, 256)`. The one Q3_K tensor is
`blk.40.ffn_down_exps.weight`, raw ne `(512, 2048, 256)`.
`packed_row_bytes(2048, 10) == 672` and `packed_row_bytes(512, 11) == 220`.
Sizes match llama.cpp `GGML_QUANT_SIZES` with `QK_K = 256`:
Q2_K `(256, 84)`, Q3_K `(256, 110)`.
The file contains no `ffn_gate_up_exps` tensor (fused count 0).
Sampled base experts: blk.0 gate/up ggml 22 (IQ2_S), down ggml 21 (IQ3_S);
blk.39 gate ggml 21, down ggml 23 (IQ4_XS). Same ne layout, E=256.

PLE metadata from the IQ3_XXS shard-1 header, copied onto a CPU
`PLEEmbedding(3, 16)` and read back:

- `qwen4exp.ple.layers == [1]`. vLLM attaches PLE when
  `layer_idx in {v-1}`, so the loader's `ple_layer_ids` value is `[2]`
  (`mapping.py:83`). Do not copy `[1]` straight into the HF config.
- multipliers length 3, offsets length 16, vocab sizes length 16.
- offsets are the prefix sums of the vocab sizes. The sum is 320001446.
- The IQ4_NL table has 320001536 rows. The hash covers 320001446 of them
  (90 rows past the last head). The reason for those 90 rows was not
  traced in llama.cpp this session.
- A second `assign_ple_hash` kept the same `data_ptr` (in-place copy).

Gather: a 4-row toy IQ4_NL table stayed on CPU (`data_ptr` unchanged).
The dequant call saw shape `(2, 90)`, the selected rows, not the table.

EP slice, CPU, toy uint8 tensor: 512 experts, rank 0 is `[0, 256)`,
rank 1 is `[256, 512)`. Byte identity on the toy rows passed.
Meta construct is world size 1, so `b70_n_local_experts` equals
`text.num_experts` (4 in the tiny config). That is the upstream
`n_local = n_physical // ep_size` path with `ep_size == 1`. A 2-rank
GPU EP run was not done.

Meta construct (`Qwen4ExpForCausalLM` and `Qwen4ExpMTP`, device meta,
gloo TP=PP=1, real `CompilationConfig(mode=NONE)`):

- layer 0 has a PLE module and `host_table is None`
- embed weight device is meta
- `mlp.experts.routed_experts.b70_w_gate` exists, device meta
- layer 1 indexer `index_n_heads == 2` and the placeholder weight is bf16
- MTP `eh_proj` weight device is meta, and the draft layer's indexer
  also has `index_n_heads == 2`

Fake CPU ops recorded the calls from `linear` and `moe`:

- two `gemv` calls, ggml types 42 then 20, N 3 then 5, K 4
- decode MoE: three `moe_gemv` calls, types 17, 16, 42; expert id -1
  arrived at the op as 0
- prefill MoE (T=9, N=K=64): three `grouped_gemm` calls after the w13
  split, types 12, 12, 42; no `moe_gemv`

One real IQ4_NL row and one real Q2_0 row through `libdequant_cpu.so`
matched gguf-py / the Q2_0 formula after the bf16 round trip
(`torch.equal`).

`vllm_xpu_kernels==0.1.16.1` is the pin in `requirements/xpu.txt`.
The wheel was not installed here. Report 04 is why the pin moved:
0.1.16.1 has the Qwen4Exp TP=2 paged-decode config (qgroup 16, head 256).

## VERDICT

The branch builds the GGUF load path the campaign asked for, on CPU,
and it rebases cleanly onto current `origin/main`. Every tensor name in
the Flash-Next IQ3_XXS text shard, the PLE shard, the MTP sidecar, and
the Qwen3.6 UD-IQ3_XXS file is mapped exactly once. Meta construction of
the causal LM and the MTP module succeeds under `b70_gguf`. The quant
methods call A2's real op names.

What this does not measure:

- No weight payload was copied into a module. Coverage is names, ggml
  type ids, and row-byte sizes.
- No GPU matmul, no serve, no 2-rank expert parallel.
- Q2_K and Q3_K have sizes and no kernel. See Requests.
- QSA selection is not computed. The indexer projection weights exist so
  the GGUF tensors have a home (`xpu/indexer_weights.py:5`).
- The order of q, k, v inside a fused `attn_qkv` blob is inferred from
  llama.cpp sizes. It was not checked against a numeric payload.
- The fused `ffn_gate_up_exps` split follows llama.cpp's gate-then-up
  view. The Qwen3.6 file has no such tensor. The split is covered by the
  fake-op prefill test only.
- PLE `_compute_rows` was not executed on a token sequence. The hash
  constants round-trip. A forward still needs attn metadata and a GPU
  (or a CPU stand-in) before the row ids can be checked against llama.cpp.
- `grouped_gemm` is invoked with one entry per expanded row. That matches
  A2's entry format and re-reads the expert. A2 already flagged the
  prefill bandwidth. The fork does not fuse gate and up into one call.

## Work items

| item | owner | size | next step |
| --- | --- | --- | --- |
| Q2_K and Q3_K dequant / gemv / moe_gemv / grouped_gemm | A2 | medium | three MTP expert tensors on the dev model; block sizes above |
| grouped_gemm entry packing (one row per entry) | A2 | medium | prefill bandwidth; fork contract already allows `nrows > 1` |
| QSA / indexer compute | A1+kernels | large | weights load; dense attention still runs the token |
| GPU EP=2 serve of the dev model, then Flash-Next | A1 with A3 | large | `bin/gpu-run` on both cards, `bin/xpu-collective-health` around it, after the plugin `.so` is on the process. Do not start `hotschmoe-dd`. |
| PLE row-id check against llama.cpp on a short CPU sequence | A1 | small | `assign_ple_hash` is in place; call `_compute_rows` with the real multipliers |
| Install `vllm_xpu_kernels==0.1.16.1` into a serve venv | A3 | small | pin is committed; the shared venv was not modified |
| Transformers 5.20 vs the installed 5.19 | A3 | small | meta construct passed on 5.19.0. Main's CI pin is 5.20.0. Do not upgrade the shared venv until a serve needs it. |

## Requests

A2: add ggml ids 10 and 11 to `dequant_bf16`, `gemv`, `moe_gemv`, and
`grouped_gemm`. Layout is llama.cpp Q2_K `(256, 84)` and Q3_K `(256, 110)`.
No new op name. No gather op. The fork will call the existing four ops
once those ids stop raising.

## Verified vs inferred

Verified this session: rebase result, the 14 CPU tests, the Qwen3.6
histogram and the three Q2_K/Q3_K names, the absence of fused
`ffn_gate_up_exps` in that file, blk.0 and blk.39 expert shapes, PLE
header shape and `n_bytes`, PLE hash prefix sums and the 320001446 used
rows, in-place hash assign, toy gather `data_ptr`, toy EP slice, meta
construct assertions listed above.

Inferred, not re-checked against a payload: q/k/v order inside fused
qkv; that the 90 extra PLE rows are unused padding; that every Q6_K
tensor in the Qwen3.6 file is a non-expert linear (only the type counts
were asserted, not a per-tensor class for id 14).
