# C4 -- first GGUF serve on vLLM-XPU

Branch: `b70/flashnext` in `/mnt/vm_8tb/b70/build/vllm-flashnext/vllm`.
Tracks `origin/main` `2296c12e47`. Ahead 7. Not pushed.

Commit list (`git log --oneline origin/main..HEAD`), tip first:

```
e3f784ebb2 [XPU] Log the XPU GDN prefill backend once
0fb5b0e134 [XPU] Gate fused QK-norm with B70_EAGER_QK
c46e1521bb [XPU] Serve Qwen3.6 GGUF and invert the GDN out_proj K map
7b40eb79c6 [XPU] Load Qwen4Exp PLE host table, indexer weights, and MTP eh_proj
cadd70b850 [XPU] Add b70 GGUF quant hook and qwen4_exp weight map
fd045ff3a0 [XPU] Pin vllm_xpu_kernels 0.1.16.1 for Qwen4Exp paged decode
4e70b2d300 [XPU] Add Qwen4Exp (Qwen3.8-Flash-Next) model architecture support
```

`git merge-base --is-ancestor origin/main HEAD` succeeded. Working tree
clean after these three new commits. The campaign repo is not committed
here.

## Step 1 verdict

Load, identity, deterministic coherence, teardown, and post-health
pass on one card. The served string for "The capital of France is"
(temperature 0, 16 tokens) is:

```
 Paris.
The capital of France is Paris.
The capital of France is
```

Two calls matched. `finish_reason` is `length`. First `/v1/models` id
is `hotschmoe-dev`.

Latency (eager gemv, max_model_len 2048, max_num_seqs 1), from
`/mnt/vm_8tb/b70/build/vllm-flashnext/logs/c4-bench-outfix.json`:

- C1 ITL p50 87.62 ms, p90 91.03 ms, p99 97.03 ms, 11.72 tok/s.
  Short-prompt TTFT 265.45 ms. 32 decode tokens.
- Prefill 1017 tokens, TTFT 5754.67 ms, 176.73 tok/s.
- C2 ITL p50 is about 86 ms per stream, but `max_num_seqs` is 1. One
  stream's TTFT was 3198 ms while the other was 302 ms. That is queueing,
  not concurrency.

Strata DD (different model, two cards, MTP) is 141 tok/s C1 and about
2200 tok/s prefill. This number is the eager one-card GGUF path.

## CONFIG

- HF config and tokenizer: `/mnt/cache/b70/devmodel/Qwen/Qwen3.6-35B-A3B`
- Weights: `/mnt/cache/b70/devmodel/unsloth/Qwen3.6-35B-A3B-MTP-GGUF/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf`
- Arch `qwen35moe`. 40 layers. MTP block tensors skipped (20). Q8_0
  dequantized to bf16 at load (7 tensors). Placed 733.
- Serve: TP=1, dtype bfloat16, max_model_len 2048, max_num_seqs 1,
  gpu_memory_utilization 0.90, language-model-only, enforce-eager,
  `--load-format b70_gguf --quantization b70_gguf`.
- Names: `hotschmoe-dev` then `qwen3.6-35b-a3b-ud-iq3-xxs`.
- Port 18104. Container `vllm-fn-qwen36-gguf-c0`. Card 0 only.
- `B70_GGUF=1`, `B70_GGUF_REF=0`, `B70_EAGER_QK=0`. No `GDN_BACKEND`.
  Log line: `Using sycl GDN prefill kernel (requested=auto, head_k_dim=128)`.
- Ops: installed `b70_gguf_xpu` (`torch.ops.b70_gguf.gemv` / `moe_gemv` /
  `dequant_bf16`). The shared venv was not reinstalled. Banner version
  stays the wheel `vllm-0.32.1rc1.dev94+g5fd62688f.xpu`. The fork is
  first on `PYTHONPATH`.
- Health image for `bin/xpu-health`: `vllm-xpu-env:int8g-v0251`.

## COMMAND

```
bin/xpu-health --card 0
CARD=0 PORT=18104 B70_AGENT=C4 B70_GGUF_REF=0 B70_EAGER_QK=0 \
  vllm/flashnext/harness/serve_qwen36_gguf.sh
# log: /mnt/vm_8tb/b70/logs/grok-flashnext/C4_serve_c0_outfix.log
python3 vllm/flashnext/harness/bench_latency.py \
  --base-url http://127.0.0.1:18104 \
  --model hotschmoe-dev \
  --alias qwen3.6-35b-a3b-ud-iq3-xxs \
  --prefill 1024 --decode-tokens 32 \
  --tag qwen36-iq3-xxs-eager-tp1-outfix \
  --out /mnt/vm_8tb/b70/build/vllm-flashnext/logs/c4-bench-outfix.json
docker stop -t 25 vllm-fn-qwen36-gguf-c0
bin/xpu-health --card 0
```

CPU mapping tests (pytest is not installed; each `test_*` called
directly), fork first on `PYTHONPATH`, `VLLM_TARGET_DEVICE=cpu`:
15 passed, 0 failed, 86 s under `bin/cpu-run`.

## RESULT

- Acquire 05:53:25 UTC. Load log 05:54:02. `b70 GGUF load ... root=Qwen3_5MoeForCausalLM ep=0/1 qk_rows=4096`.
- `placed 733 skipped_mtp 20 q8_dequant 7 kinds {14: 245, 0: 361, 21: 39, 22: 78, 23: 3, 8: 7}`.
- Model loading 16.96 GiB, 29.266 s. KV 8.93 GiB. 110,080 tokens.
  Max concurrency at 2048 is 53.75x in the KV math; the server flag
  is still `max_num_seqs 1`.
- Application startup complete. `/v1/models` HTTP 200.
- Greedy calls HTTP 200 in 2 s and 1 s. Strings matched. Paris is in
  the text. Saved `/tmp/c4_outfix1.json` and `/tmp/c4_outfix2.json`.
- `docker stop` then `gpu-run: [C4] done in 178s (exit 0)`.
  `xpu-health: HEALTHY (cards 0)`. No crash, so no `xe-reset`.

### What was wrong

`gdn_out` leaves the packed `ssm_out` matrix in GGUF K order
(`mapping.py` `apply_packed_layout` returns it unchanged). vLLM's GDN
writes the activation in `(n_groups, heads_per_group, head_dim)` order.
`restore_rows` turns GGUF `(heads_per_group, n_groups, head_dim)` into
that vLLM order. The old `restore_last_dim` called `restore_rows` on
K. That second swap is an involution only when the two counts are
equal. Qwen3.6 uses 2 and 16 (K=4096, head_dim 128). The old unit test
used 8 rows and head_dim 2, so both counts were 2 and the bug was
hidden.

`ValueHeadLayout.restore_last_dim` now runs `_vllm_rows_to_gguf` on K
(`mapping.py`). `linear.py` `apply` calls it when `b70_act_layout` is
`gdn_out`. The wide-layout test checks the inverse and the matmul
identity (`tests/models/qwen4_exp/test_b70_gguf_mapping.py`,
`test_dense_transforms_and_gdn_rows`).

On real `blk.0.ssm_out` before the edit, the old direction's matmul
error was max abs 11.62. The inverse matched the HF-order matmul at
max abs 3.8e-6. That hits every GDN out projection (30 of 40 layers),
which is why three earlier serves were deterministic mixed-script
garbage and still shared fragments (Cyrillic, CJK, and `ityEngine`).

Those three serves were not a one-variable bisect:

- Kernel, default GDN, fused QK, older `_C.so`: repeating `ityEngine`.
  Log `C4_serve_c0.log`. Torn down 04:26, health OK.
- `B70_GGUF_REF=1` and `B70_EAGER_QK=1`: different garbage. Log
  `C4_serve_c0_ref.log`. Profile was about 15 min because host dequant
  is lazy and vLLM had set OMP threads to 1. Torn down 04:53, health OK.
- `GDN_BACKEND=triton`, gemv, fused QK on: different garbage again.
  Log `C4_serve_c0_triton.log`. A first attempt overlapped C3 because
  `gpu-run` treated a failed `flock` during `TIMEOUT=0` as success.
  EngineCore died on free memory 14.71/30.3 GiB. That is not a model
  failure. `bin/gpu-run` now exits 1 on a failed flock. The requeue
  at 04:59 served cleanly and was torn down 05:27, health OK.

### What was ruled out

XPU oracle, card 0, 35 s, exit 0, log
`/mnt/vm_8tb/b70/logs/grok-flashnext/C4_kernel_oracle.log`. Installed
ops vs gguf-py. dequant max is bf16 rounding. gemv max is bf16 matmul
noise.

- token_embd Q6_K, 4 rows: dequant max 0.000122; gemv N=4 max 0.004091
- blk.3.attn_q Q6_K, 64 rows: dequant max 0.000244; gemv max 0.014824
- blk.0.attn_qkv Q6_K, 64 rows: dequant max 0.000244; gemv max 0.009895
- expert 0 gate IQ2_S full: dequant max 0.000232; gemv max 0.007313
- expert 0 down IQ3_S full: dequant max 0.000222; gemv max 0.003598
- blk.0.ssm_out Q6_K, 8 rows: dequant max 0.000244; gemv max 0.010596

`gather_rows` always calls XPU `dequant_bf16`. That is not the
coherence bug. CPU weight compares (earlier in this run) already
matched HF within quant noise for norms (+1 on Gemma RMSNorm only),
GDN qkv/conv/gate value rows, router, embed, lm_head, and experts.
Do not flip those permutations.

Ornith GPTQ on the same architecture is coherent with SYCL GDN, so a
shared XPU GDN bug was not the explanation.

## Step 1b, step 2, step 3

Not measured.

At 06:05 card 1 was held by E1 (`vllm-fn-lowbit-e1`, pid 2960791,
since 05:59:23, port 18084, lowbit FULL_DECODE_ONLY). Both-card work
was not started. No collective health this run. No `xe-reset`.

Step 1b next command, once both cards are free. Announce in
`vllm/flashnext/JOURNAL.md` first. Run `bin/xpu-collective-health`
(P2P off) first. Do not set `CCL_TOPO_P2P_ACCESS=1`. Keep the serve
short.

```
B70_AGENT=C4 B70_GPU_LOCK_TIMEOUT=180 \
  vllm/flashnext/harness/xpu-run.sh --card both \
  --name vllm-fn-qwen36-gguf-ep --port 18104 \
  --env B70_GGUF=1 \
  --env B70_GGUF_PATH=/mnt/cache/b70/devmodel/unsloth/Qwen3.6-35B-A3B-MTP-GGUF/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf \
  --env PYTHONPATH=/mnt/vm_8tb/b70/build/vllm-flashnext/vllm:/mnt/vm_8tb/b70/build/strata-ab/llama-pinned/gguf-py:/mnt/vm_8tb/github/b70_ai_things/vllm/flashnext/plugins/b70_gguf_xpu \
  --env B70_GGUF_REF=0 --env B70_EAGER_QK=0 \
  -- vllm serve /mnt/cache/b70/devmodel/Qwen/Qwen3.6-35B-A3B \
     --host 0.0.0.0 --port 18104 \
     --served-model-name hotschmoe-dev qwen3.6-35b-a3b-ud-iq3-xxs \
     --tensor-parallel-size 1 --data-parallel-size 2 \
     --enable-expert-parallel \
     --dtype bfloat16 --max-model-len 2048 \
     --gpu-memory-utilization 0.90 --max-num-seqs 1 \
     --language-model-only --enforce-eager \
     --load-format b70_gguf --quantization b70_gguf \
     --trust-remote-code
```

Why DP=2 and TP=1: `serve_loader.py` raises if the tensor-parallel
world size is not 1. With `--data-parallel-size 2` and
`--enable-expert-parallel`, global TP stays 1 and MoE `ep_size`
becomes 2 (`fused_moe/config.py`, `ep_size = flattened tp`). The
loader already slices the expert axis (`slice_expert_tensor`,
`ep_rank` passed into `load_mapped_tensor`). Attention, embeddings,
and `lm_head` stay replicated, which is the path that just produced
Paris. `B70GGUFMoEMethod` maps global ids through `expert_map` and
zeros id -1. `supports_internal_mk` is false (`moe_kernel` is None),
so `dp_size > 1` takes the runner's naive dispatch/combine
(`moe_runner.py` `do_naive_dispatch_combine`). Compare the greedy
string to the one-card text above.

If that process errors or wedges: `docker stop`, `bin/xe-reset` only
after a crash, then `bin/xpu-health` and `bin/xpu-collective-health`.

TP=2 is the fallback and is still blocked in the loader on purpose.
`--tensor-parallel-size 2 --enable-expert-parallel` sets global TP to
2, so attention, GDN, shared experts, and the vocab are sharded even
though expert weights are EP-sliced. Packed sharding is not
implemented. The pieces that are not a contiguous packed-row half:

- `gdn_out` K. Head dim 128 is half a Q6_K block (256), and GGUF order
  is `(heads_per_group, n_groups, head_dim)` with 2 and 16. A
  contiguous byte slice is the wrong heads. Dequant, apply
  `restore_rows` on K, slice the vLLM-order K, store dense bf16, and
  do not call `restore_last_dim` on that activation.
- `in_proj_qkvz` is four output sizes `[key, key, value, value]`
  (`qwen_gdn_linear_attn.py` `create_qkvz_proj`). Value rows must be
  permuted before the row slice. The GGUF loader currently returns the
  full cat from `apply`.
- `A_log` and `dt_bias` are sharded on heads. `conv1d` uses
  `mamba_v2_sharded_weight_loader`.
- Full attention `QKVParallelLinear` includes the per-head output gate.
  `o_proj` K is 4096 and 256-aligned, so a packed K slice is legal
  there if the order is already vLLM order.
- Vocab 248320 / 2 = 124160. The embedding method indexes global ids
  into the packed table. A sliced table needs the vocab-parallel
  offset. Do not ship that untested.

Step 2 (Flash-Next IQ3_XXS, `qwen4exp`, EP=2, PLE in host RAM) waits
on a coherent 2-card path. The PLE host-table load is already in
`7b40eb79c6`. Eager first. Coherence target is Strata DD greedy tokens
on fixed prompts.

Step 3 (MTP, graph) waits on two things. The GGUF MTP block is Q2_K /
Q3_K and the loader skips it (`serve_loader.py`, types 10 and 11).
The installed `.so` was not shown to run those types; base-model
IQ2_S, IQ3_S, and Q6_K gemv were. Graph capture stays after eager
coherence. `vllm_xpu_kernels` was not upgraded to 0.1.16.1.

## Campaign tree left uncommitted

These are not fork commits. `bin/` still needs the live-shelf smoke
the AGENTS note requires before the orchestrator commits it.

- `bin/gpu-run`: a failed `flock` during `TIMEOUT=0` now exits 1
  instead of running the command without the lock.
- `vllm/flashnext/harness/serve_qwen36_gguf.sh`: passes
  `B70_GGUF_REF`, `B70_EAGER_QK`, and optional `GDN_BACKEND`.
- `vllm/flashnext/harness/patches/sitecustomize.py`: skips plugin
  register when `import torch` fails, so the entrypoint's
  `/usr/bin/python3.12` loader check does not traceback.

## Requests

- C1: a Q2_K / Q3_K gemv proof on the installed `.so` before MTP load.
- Whoever has card 1: step 1b needs both leases for a short DP=2 EP
  serve after collective health. It was not queued on top of E1.
