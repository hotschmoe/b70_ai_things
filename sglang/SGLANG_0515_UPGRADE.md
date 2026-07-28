# SGLang XPU 0.5.6 -> 0.5.15.post1 upgrade + re-graft

Config -> command -> result -> verdict for bumping the sglang XPU serving stack from the
0.5.6-era floating `main` (SHA 09ca4fc = 0.5.6.post3.dev6841) to the pinned release
**v0.5.15.post1** (tag confirmed via `git ls-remote`, released 2026-07-14), and re-grafting all
custom B70/XPU work so the W8A8 Qwen3.6-27B serve can be benched on 0.5.15.

The initial re-graft session was compile-only. The 2026-07-28 GPU campaign below now covers device
visibility, load, mixed-load coherence, performance, physical KV capacity, and exact long-context
retrieval with prefix reuse.

## Images / tags

| layer | 0.5.6 tag | 0.5.15 tag | recipe |
| --- | --- | --- | --- |
| base (sglang+torch+sgl-kernel) | `sglang-xpu:bmg` | `sglang-xpu:bmg-0515` | `images/sglang-xpu/build_0515.sh` |
| + woqgemm int4 + shims | `sglang-xpu:woq` | `sglang-xpu:woq-0515` | `images/build_layers_0515.sh` |
| + MTP gates + memory_pool | `sglang-xpu:mtp` | `sglang-xpu:mtp-0515` | `images/build_layers_0515.sh` |

Build order (all GPU-free):
```
cd sglang/images/sglang-xpu && bash build_0515.sh      # -> sglang-xpu:bmg-0515
cd sglang/images            && bash build_layers_0515.sh # -> sglang-xpu:woq-0515, sglang-xpu:mtp-0515
```
`build.sh` now takes `SG_LANG_BRANCH` (default `main`, preserves the old build); `build_0515.sh`
pins it to `v0.5.15.post1` and tags `sglang-xpu:bmg-0515`. The W8A8 serve only needs
`sglang-xpu:mtp-0515`.

Confirm the built version: `docker run --rm sglang-xpu:bmg-0515 bash -lc 'python -c "import sglang; print(sglang.__version__)"'`
-> expect `0.5.15.post1` (the in-build wheel logged `sglang-0.5.15.post1-py3-none-any.whl`).

## Pin conflicts 0.5.6 -> 0.5.15 and resolutions

`pyproject_xpu.toml` drift is small and self-resolving (the Dockerfile copies it to `pyproject.toml`
and `pip install .` honors the in-tree pins):

| pin | 0.5.6 | 0.5.15 | action |
| --- | --- | --- | --- |
| torch / torchao / torchvision / torchaudio (`+xpu`) | 2.12.0 / 0.17.0 / 0.27.0 / 2.11.0 | **unchanged** | none |
| xgrammar (Dockerfile `--no-deps`) | 0.1.33 | **unchanged** | none |
| transformers | 5.8.1 | 5.12.1 | auto (pyproject) |
| mistral_common | >=1.11.0 | >=1.11.5 | auto (pyproject) |
| sgl-kernel | `git+sgl-kernel-xpu.git` (main) | same (main, unpinned) | rebuilt from source in-image |

No manual pin surgery was required. The only behavioral bump is transformers 5.8.1 -> 5.12.1; the
Qwen3.6 model classes (`qwen3_5.py`, `qwen3_5_mtp.py`, `qwen3_next.py`) are **upstream in sglang**
(HTTP 200 at the tag) and load through sglang's own registry, not transformers' auto-model, so the
bump is low-risk. GPU load is the final gate.

## Shim drift audit (validated against a v0.5.15.post1 source clone)

All custom shims are canonical `sglang/patches/` files. Every monkeypatch target module/symbol was
grep-verified present in the 0.5.15 source. Result per shim:

### Still-needed, still-applies (no change)
- **w8a8_shim.py** -- targets all present and semantics intact:
  `compressed_tensors.schemes.compressed_tensors_w8a8_int8.CompressedTensorsW8A8Int8`
  (`process_weights_after_loading` + `apply_weights`), and
  `compressed_tensors.compressed_tensors.CompressedTensorsConfig._check_scheme_supported`. The CHANNEL
  strategy still does `layer.weight = Parameter(weight.t(), ...)` ([N,K]->[K,N]), which is exactly the
  layout the fused hybrid's `_pw_fused` relies on (and it self-checks the `B_nt` NT stride at runtime).
  `get_min_capability` is now 80; the shim's emulated cap 90 still passes.
- **woq_shim.py** -- `gptq.schemes.gptq_linear.GPTQLinearScheme._init_kernel` present (this is the
  install-time import gate that also unblocks W8A8+MTP -- so the woq layer's `auto-round-lib` is
  required even for the W8A8 serve). `marlin_utils.check_marlin_supported` signature unchanged
  (`device_capability` is the 4th positional; the guard reads `a[3]`). `ModelRunner.load_model`,
  `LogitsProcessor._compute_lm_head` (+ `use_fp32_lm_head`), `platforms.current_platform` all present.
- **mtp_tree_xpu.py** -- `eagle_utils.{build_tree_kernel_efficient,verify_tree_greedy_func,TreeMaskMode,
  eagle_sample}`, `triton_ops.cache_locs.assign_extend_cache_locs_func`,
  `mamba_state_scatter_triton.{fused_mamba_state_scatter_with_mask,fused_conv_window_scatter_with_mask}`,
  `hybrid_linear_attn_backend`, `server_args.ServerArgs._validate_mamba_extra_buffer` (needle
  `is_cuda() or is_musa() or is_npu()` still exact) all present.
- **qwen3_coder_detector.py** -- upstream `function_call/qwen3_coder_detector.py` is **byte-identical**
  between 0.5.6 and 0.5.15, so our incremental-streaming patch (derived from that same 477-line base)
  applies unchanged. Still mounted over the baked copy by the serve script.
- **fused_recurrent.py / fused_gdn_gating.py** -- upstream **byte-identical** 0.5.6 vs 0.5.15; our
  baked copies are upstream + an inert `B70_GDN_DECODE_WARPS` env knob (PERF.md: no warm speedup).
  Zero drift; baked for continuity.

### Drifted -- FIXED
- **memory_pool.py** -- upstream grew **2959 -> 3749 lines** (+790: dedup sliding-window conv-window
  layout for spec-decode, etc). The old baked file was the 2959-line 0.5.6 copy + a 3-site
  `device="cuda" -> device=device` fix (spec-decode mamba state cache). Baking the stale file onto
  0.5.15 would **revert those 790 upstream lines** -> dangerous. **FIX:** re-derived
  `images/sglang-xpu-mtp-0515/memory_pool.py` = the v0.5.15.post1 upstream file with the identical
  3-site fix re-applied (the file still has exactly 3 `device="cuda"` allocations, all the mamba
  spec-state/conv-window caches, with `device` in local scope). Verified 3/3 sites rewritten.

### Drifted -- now NATIVE upstream (our patch is a benign no-op)
- **eagle_sample greedy branch** -- 0.5.15 upstream ships
  `if sampling_info.is_all_greedy or _is_npu or _is_hip or _is_xpu:` (with `_is_xpu = is_xpu()` at
  module top). Our DOMINO 4 re-exec looks for the OLD string `... or _is_hip:`, no longer matches, so
  it prints `SKIPPED (condition not found -- upstream changed)` and does nothing -- which is CORRECT,
  because upstream now does exactly our fix. No action needed; the print is expected.
- **assign_extend_cache_locs_func** -- 0.5.15 upstream added `_is_xpu` to the branch that allocates
  `out_cache_loc` (our DOMINO 1 cause). Our shim still replaces the function with the draft-slot
  gather (takes precedence, validated on 0.5.6); it is now redundant but harmless.

### Not used by the W8A8 serve (baked but inert)
- **xpu_cudagraph.py**, **w4a8_shim.py**, **push_ar_xpu.py** -- all env-gated OFF in the W8A8 serve
  (`B70_XPU_CUDAGRAPH`/`B70_XPU_W4A8`/`B70_XPU_PUSH_AR` unset). Baked/available for other drivers;
  not exercised here. `sglang.srt.platforms` and `sglang.srt.utils` are both packages in 0.5.15
  (`utils/__init__` does `from .common import *`, so `from sglang.srt.utils import is_xpu` still works;
  `platforms.current_platform` is a lazy `__getattr__` singleton).

## W8A8 Qwen3.6-27B serve for 0.5.15

`sglang/serve_w8a8_0515.sh` -- a copy of the shelf `rdy_to_serve/sglang/qwen36-27b-w8a8/serve.sh`
with only these deltas (proven settings otherwise identical: `--skip-server-warmup`, vision graft
ckpt, TP=2, NEXTN MTP steps=10, `--disable-cuda-graph`, page/mamba-extra-buffer, tool/reason parsers):
- `IMG=sglang-xpu:mtp-0515`
- `NAME=sglang_w8a8_mtp_0515`
- `RADIX=1` by default (prefix/radix caching ON -- user request; extra_buffer strategy + int8 mamba
  checkpoint pool + page 128, keeping the intel_xpu XMX attention backend).
- optional `MAMBA_CACHE=<slots>` maps to `--max-mamba-cache-size`. The default stays auto-sized.
  For one-request 200K BF16-KV mode, `MAXREQ=1 MAMBA_CACHE=4` is the measured configuration.

It is deliberately NOT a shelf sibling (shelf rule: exactly one best config per backend/model/quant;
no promotion until GPU-bench-gated faster-or-equal AND coherent). `bash -n` clean. All runtime mounts
are the canonical `sglang/patches/` files (validated above).

## DONE
- v0.5.15.post1 tag confirmed; `build.sh` parametrized (`SG_LANG_BRANCH`) + `build_0515.sh` added.
- Base build recipe -> `sglang-xpu:bmg-0515` (sglang 0.5.15.post1 wheel built in-image).
- Pin-conflict audit: only transformers/mistral_common bumped, auto-resolved; torch/xgrammar unchanged.
- Full static shim-drift audit against a 0.5.15 source clone (table above).
- Re-derived memory_pool.py (0.5.15 upstream + 3-site device fix) -> `sglang-xpu-mtp-0515/`.
- 0.5.15 layer Dockerfiles + `build_layers_0515.sh` -> `sglang-xpu:{woq,mtp}-0515`.
- `sglang/serve_w8a8_0515.sh` (caching on), `bash -n` clean.
- `sglang/qualify_w8a8_0515.sh`: identity, physical token-pool, coherence, native performance,
  unique-prompt prefill, exact long-context, cache metrics, fatal-log, cleanup, and card-health gates.
- GPU runtime: torch 2.12.0+xpu, sglang 0.5.15.post1, and both XPU devices confirmed.
- Coherence: 18/18 mixed prefill+decode at 131K and again at 200K with auto-sized pools.
- One-request 200K BF16 KV: `MAXREQ=1 MAMBA_CACHE=4` allocates 220,288 tokens and passes exact
  190,048-token retrieval cold/warm in 334.58s/3.54s, with 189,952 cached tokens and 99.93% hit rate.
- `sglang/ab_w8a8_0515_vs_0506.sh`: matched short-context, radix-off A/B with runtime identity,
  coherence, native performance, unique cold prefill, usage-based code throughput, fatal-log,
  cleanup, and card-health gates.
- `sglang/profile_w8a8_0515_vs_0506.sh`: matched EXTEND and DECODE CPU+XPU traces, shapes,
  per-rank artifacts, parser output, graceful cleanup, and card-health gates.
- `sglang/ab_w8a8_0515_spec_compile.sh`: controlled default-off/on test of the two XPU
  speculative metadata helpers disabled by upstream commit 4fffc6448.

## Regression profile

`sglang/profile_w8a8_0515_vs_0506.sh` profiled one cold approximately 2K-token EXTEND and five
decode batches per version with radix disabled. Both versions used torch 2.12.0+xpu and the same
custom W8A8 kernel. The images differ in sglang, transformers, auto-round-lib, and sgl-kernel:
the 0.5.6 image uses sglang 09ca4fc96 and sgl-kernel 6cd2a07; the 0.5.15 image uses sglang
0b3bb0 and the descendant sgl-kernel 3bc8de6.

| EXTEND device time | 0.5.6 rank 0/1 | 0.5.15 rank 0/1 |
| --- | ---: | ---: |
| total kernels | 961 / 1,276 ms | 1,217 / 1,557 ms |
| collectives | 299 / 664 ms | 615 / 970 ms |
| INT8 GEMM | 175 / 158 ms | 165 / 162 ms |
| BF16 GEMM | 122 / 114 ms | 119 / 114 ms |
| GDN recurrent | 116 / 106 ms | 76 / 72 ms |
| copy/reshape | 106 / 98 ms | 103 / 101 ms |
| activation quant | 36 / 33 ms | 34 / 34 ms |
| raw FMHA | 11.5 / 11.1 ms | 10.8 / 10.4 ms |

The all-reduce count is unchanged at 264/rank. Normal GPU memcpy is about 0.1 ms/rank, so this is
not a host/device-copy regression. The newer trace adds `sglang::inplace_all_reduce` and registered
all-gather CPU annotations around the same oneCCL work, and its XPU driver span grows sharply. The
math kernels and attention are flat or faster. The matched decode traces likewise have essentially
identical summed device time across ranks, but 0.5.15 adds 225 device kernels over five batches and
about 8% CPU-op duration. Therefore the shelf loss is in collective scheduling/driver and eager
metadata overhead above the XMX kernels, not in the W8A8 GEMM or the newer FMHA implementation.

Traces:
`/mnt/vm_8tb/b70/sgl_cache/profile_sglang_w8a8_ab_20260728_064224`.
Run log:
`results/logs/sglang_w8a8_0515_vs_0506_profile_retry2_20260728.log`.

## Speculative metadata compile A/B

Upstream commit 4fffc6448, "Speculative decoding support on XPU", changed two decorators to disable
compile on XPU. `B70_XPU_SPEC_COMPILE=1` in `mtp_tree_xpu.py` is a default-off diagnostic that
rewraps both helpers. It removed the five extra eager metadata kernels per draft iteration, but
the controlled A/B rejected it:

| metric | compile off | compile on | delta |
| --- | ---: | ---: | ---: |
| native c1 | 24.32 t/s | 24.28 t/s | -0.2% |
| native c4 aggregate | 35.47 t/s | 33.73 t/s | -4.9% |
| 2K-token soak | 19.40 t/s | 19.34 t/s | -0.3% |
| unique cold prefill | 1,868 t/s | 1,820 t/s | -2.6% |
| code c1 | 25.9 t/s | 25.3 t/s | -2.3% |
| code c4 aggregate | 91.3 t/s | 89.7 t/s | -1.8% |

Both arms passed 18/18 mixed-load coherence and both cards passed health. Fewer device kernels do
not repay the Dynamo/runtime overhead on this stack. Keep the override default-off.

## Shelf promotion verdict

- NO-GO. The controlled 8K, radix-off A/B passed 18/18 coherence on both versions, but 0.5.15 was
  slower in every matched measure: native c1 24.04 versus 25.60 t/s (-6.1%), native c4 aggregate
  35.26 versus 36.31 t/s (-2.9%), code c1 24.9 versus 25.6 t/s (-2.7%), code c4 aggregate 89.4
  versus 91.7 t/s (-2.5%), and unique cold prefill 1,719 versus 1,988 t/s (-13.5%).
- Keep 0.5.6 as the shelf entry. Retain 0.5.15 as the research and one-request 200K path; profile
  data localizes the regression to collective/runtime handling, but no measured fix has recovered
  it. Investigate a large-prefill collective transport A/B before attempting another promotion.

## Qualification commands

Full 200K qualification under one dual-card lease:
```
cd /mnt/vm_8tb/github/b70_ai_things
MAXREQ=1 MAMBA_CACHE=4 MIN_POOL_TOKENS=190128 \
  ./bin/gpu-run bash sglang/qualify_w8a8_0515.sh
```

Focused capacity/retrieval rerun:
```
MAXREQ=1 MAMBA_CACHE=4 MIN_POOL_TOKENS=190128 \
  RUN_COHERENCE=0 RUN_PERF=0 RUN_PREFILL=0 \
  ./bin/gpu-run bash sglang/qualify_w8a8_0515.sh
```

## BLOCKER ROOT CAUSE CORRECTED (2026-07-28): WOQ layer upgraded torch -> no XPU

The W8A8 serve died at startup because the final `sglang-xpu:mtp-0515` image carried CUDA-only
`torch 2.13.0+cu130`. Fresh inspection corrected the earlier attribution:

- `sglang-xpu:bmg-0515` is healthy and still carries `torch 2.12.0+xpu` plus sglang
  `0.5.15.post1`; its `pip install .` and `torchcodec==0.12.0` did not replace torch.
- `sglang-xpu:woq-0515` is the first bad layer. Its unconstrained
  `pip install auto-round-lib` sees auto-round-lib 0.14.2's unbounded `Requires: torch` and
  re-resolves the newest PyPI CUDA wheel, replacing the installed XPU build.
- `sglang-xpu:mtp-0515` inherits that bad WOQ environment. sgl-kernel was compiled in the
  correct base image, so a full base/kernel rebuild is unnecessary.

FIX -> the WOQ layer now installs the known `auto-round-lib==0.14.2` with `--no-deps` and asserts
`torch.__version__ == "2.12.0+xpu"` while building. `build_layers_0515.sh` also verifies both final
layer images. Rebuild only the two lightweight layers, then run the GPU image gate before serving.

RESULT (2026-07-28) -> lightweight rebuild passed:

- base `sglang-xpu:bmg-0515` image id `8602d8966f1b`
- corrected `sglang-xpu:woq-0515` image id `8f30adeddeae`
- corrected `sglang-xpu:mtp-0515` image id `c99cf2b7a8e2`
- both final layers report `torch 2.12.0+xpu` and sglang `0.5.15.post1`

VERDICT -> packaging blocker fixed and GPU-qualified. The 0.5.15 image is a valid research serve and
the BF16-KV one-request 200K path is proven. Shelf promotion still waits on the controlled
short-context, radix-off performance A/B.
