# 04 -- Ecosystem survey (vLLM XPU, hybrid MoE, MTP, KV RAM)

Date: 2026-10-19. No GPU. Line cites are the local clone
`/mnt/vm_8tb/b70/build/vllm-flashnext/vllm` at `5fd62688` (2026-10-19 14:37
-0700, `origin/main`, 0 commits behind that remote at log time). Sibling
clones read the same way: `vllm-xpu-kernels` `19c6ddc` (2026-10-16, contains
tag `v0.1.16.1`) and `vllm-gguf-plugin-ro` `e2b8ad5` (2026-09-24). PR state
is from the GitHub API the same day. The clone was absent at the start of
this survey; task 01 had populated it before `git log`.

Upstream release in that tree's history: **vLLM v0.32.0** (2026-10-18,
https://github.com/vllm-project/vllm/releases/tag/v0.32.0). XPU wheel index
for that tag exists. Phoronix on 2026-10-16 still said "0.31"; that was the
previous release.

## Recommendation

Ship on unmodified upstream main (v0.32.0 / `5fd62688`). Put the kernel pin,
GGUF, and MoE tune JSON in plugins and requirement pins. Keep one thin
rebase branch whose model commit is https://github.com/vllm-project/vllm/pull/61607
until that PR merges. Delete the branch when it does.

That matches the brief order. The Qwen4Exp architecture is already on main
for CUDA and ROCm (`vllm/models/qwen4_exp/` has `amd/`, `common/`, `nvidia/`;
there is no `xpu/` directory). Intel's packaged stack is the wrong base:
llm-scaler-vllm 1.5 (2026-10-16) is vLLM **0.26.0** plus patches, kernels
**0.1.11.2**, torch 2.12.0+xpu, Triton 3.7.1, oneAPI 2025.3.3, UMD
**26.14.37833.4**. This host's standing baseline is kernel 7.1 and Compute
Runtime 26.22. Pulling the 1.5 image would pin an older UMD.

What belongs in a plugin or a pin, with no fork:

- `vllm_xpu_kernels` version (today `requirements/xpu.txt:23` pins
  `0.1.15.4`; release `0.1.16.1` is the one that can start Qwen4Exp at TP=2).
- `vllm-gguf-plugin` if we try the ISTA IQ3_XXS GGUF before requantizing.
- A B70 fused-MoE JSON for the Flash-Next shape, produced by the in-tree
  tuner from https://github.com/vllm-project/vllm/pull/53065. The loader is
  already platform-agnostic.
- `OffloadingConnector` and automatic prefix caching, which are flags.
- LMCache only as an external process, the way llm-scaler runs it, and only
  after the native connector fails a hybrid restore.

What belongs on the thin branch, as new files plus the smallest edits:

- `vllm/models/qwen4_exp/xpu/` from PR 61607 (+1614/-5, 7 files, open).
  Author validation is `--enforce-eager` on a W4A16 conversion, TP=4 with
  `--enable-expert-parallel`. No QSA/indexer path.
- https://github.com/vllm-project/vllm/pull/62111 (+208/-5, open, updated
  2026-10-19) only if multi-layer MTP module selection fails at serve time.
- https://github.com/vllm-project/vllm/pull/61956 is a one-line kernels bump.
  Prefer changing the pin in our requirements over forking vLLM for it.

Leave the scheduler, graph runtime, and collectives unforked until a
measurement on this host fails. intel/llm-scaler#660 ("Enabe qwen3.8 flash
next", open since 2026-09-02, +38365/-495) is a tree dump. Skip it.

## Ranked adopt / cherry-pick

1. **Base = vLLM main, XPU requirements as written, kernels bumped to
   0.1.16.1.** `requirements/xpu.txt` pins torch==2.14.0, triton==3.8.0+xpu,
   auto_round_lib==0.16.0, vllm_xpu_kernels==0.1.15.4. Kernels `v0.1.16.1`
   (commit `8b95ea6`, 2026-10-14) adds the Qwen4Exp TP=2 paged-decode config
   (qgroup 16, head 256, page 64). At TP=1 the GQA ratio stays in the qgroup=8
   table; at TP=2 it becomes 12 and the 0.1.15.4 build has no specialization,
   so serve fails at startup. Same release adds
   `gated_delta_rule_spec_recoverssm` (`78799fe`, kernels PR 618). Open vLLM
   PR 61956 is that pin bump. Applicability: direct for two B70s.

2. **Cherry-pick PR 61607, the XPU Qwen4Exp (Qwen3.8-Flash-Next) model.**
   Open, head `support_qwen4exp`, author zhaohb, updated 2026-10-15.
   https://github.com/vllm-project/vllm/pull/61607
   Reuses `Qwen3NextAttention`, `QwenGatedDeltaNetAttention`, and
   `Qwen3NextSparseMoeBlock`. Also accepts MTP hidden width under both
   `hc_mult` and `hc_count`, and guards an empty `qzeros` copy on symmetric
   group quant (the 512-expert W4A16 load crash they hit). Known limits in
   the PR body: no QSA if the checkpoint actually has indexer tensors
   (`indexer_n_heads=4` on their W4A16 file, but no indexer weights);
   `moe_intermediate_size=640` makes TP=4 illegal for group_size 128
   (640/4=160) unless `--enable-expert-parallel`; TP=2 is 320 and divides
   by 128; `transformers>=5.16.1`; eager-only validation. The CUDA/ROCm
   recipe is already in-tree at `recipes/models/Qwen/Qwen3.8-Flash-Next.yaml`
   and on https://recipes.vllm.ai/Qwen/Qwen3.8-Flash-Next (vLLM >=0.30,
   verified on nightly, NVIDIA/AMD only): `--enable-expert-parallel`,
   prefix caching, fp8 KV, `speculative-config method=mtp` with 3 tokens.

3. **Serve symmetric int4 (AutoRound or GPTQ W4A16), and retune MoE for
   this shape.** On main, INC routes every `RoutedExperts` layer through
   `scheme.get_moe_method` (`inc/inc.py:356-357`). The XPU branch accepts
   GPTQ int4 symmetric and prefers ARK MoE when
   `VLLM_XPU_INC_WNA16_BACKEND` is not `w4a16`/`w4a8` and ARK is installed;
   otherwise it falls through to the default WNA16 MoE path
   (`inc_wna16_scheme.py:161-220`). Linear INC on XPU allows bits {2, 4}
   and requires `sym` (`inc_wna16_scheme.py:21` and `:72`). The MoE oracle
   on XPU returns only `WNA16MoEBackend.XPU`
   (`int_wna16.py:130-131`), which calls
   `vllm_xpu_kernels` `xpu_fused_moe(is_int4=True)` and passes no qzeros
   (`int_wna16.py:1726` area: "sym int4 on XPU has none"). Our old
   `vllm/contrib/vllm_moe_xpu` 16-line patch was for vLLM 0.23.0, where the
   XPU INC path returned None for `RoutedExperts` and the experts inflated
   to bf16. That branch is gone on this main. Re-test a load before
   carrying the patch. Merged tuner: PR 53065 (2026-10-07) added B70 bf16
   configs and an XPU search space; geometric means about 1.9x on the
   kernel for OLMoE / Qwen1.5 shapes, and only 1.04x end-to-end on
   Qwen3-Coder-30B-A3B (E=128, N=192, TP=4). Open PR 56788 is a B70
   int4_w4a16 JSON for **Qwen3-30B-A3B** (E=128, N=768), kernel geomean
   ~1.82x versus the Triton heuristic. Copy the method. The JSON itself
   is the wrong shape for Flash-Next (N=640 on the W4A16 file in PR 61607).

4. **IQ3_XXS GGUF is a plugin experiment, then a requantize.** The published
   matrix marks GGUF unsupported on Intel GPU
   (`docs/features/quantization/README.md:77`) and points at
   https://github.com/vllm-project/vllm-gguf-plugin (`gguf.md:7-12`). The
   plugin clone does contain Triton GEMM and fused-MoE kernels for
   `GGML_TYPE_IQ3_XXS` (`vllm_gguf_plugin/triton/gemm/iq_quant/iq3_xxs.py`,
   `vllm_gguf_plugin/triton/fused_moe/iq_quant/iq3_xxs.py`). It has no XPU
   source of its own. Whether Triton-XPU 3.8 runs those kernels, and
   whether Qwen4Exp GGUF tensor names map, is unverified here. llama.cpp
   SYCL is why Strata can serve this pack today; that does not make the
   ggml kernels a vLLM backend. If the plugin fails, cut a symmetric int4
   checkpoint and use item 3. Building an IQ SYCL dequant inside vLLM is
   a new kernel project, behind the int4 path.

5. **CPU KV tier: upstream `OffloadingConnector`, align-mode Mamba, then
   measure.** Docs: CUDA, ROCm, and XPU
   (`docs/features/kv_offloading_usage.md:6`). `XPUPlatform` implements
   `swap_out_blocks_to_host` as a synchronous `.cpu()` copy
   (`vllm/platforms/xpu.py:602-611`). The guide's `cudaMemcpyAsync` sentence
   is the CUDA description; this survey did not trace an XPU DMA path.
   Hybrid state is in the connector: a `MambaSpec` group offloads one
   chunk, and align mode rounds the hit window to the chunk
   (`offloading/scheduler.py:148-174`). `support_hybrid_kv_cache` returns
   True (`xpu.py:512-513`). Merged PR 52771 (2026-09-07) stops the
   connector from storing KV and then reporting zero reusable tokens under
   MTP/EAGLE. The reporter confirmed that fix on one Arc Pro B70, vLLM
   `3756bf1e`, Qwen3.8-27B W4A16 (hybrid GDN, not Flash-Next), 16 GiB host
   pool, fp8 KV, MTP on, about 179k GPU tokens. Issue
   https://github.com/vllm-project/vllm/issues/52773 (open, 2026-08-18)
   still describes the hybrid failure we care about: full-attention blocks
   can outlive the recurrent boundary state, and the connector does not
   backfill that boundary, so a RAM-resident prefix is recomputed. Open
   follow-ups, all unmerged: PR 58413 (align-mode Mamba state at
   hash-block granularity under MTP, +203, updated 2026-09-24), PR 61019
   (partial tails with EAGLE/MTP groups, +561, updated 2026-10-16), PR
   61282 (keep the Mamba prompt-boundary state until the request ends,
   +424, updated 2026-10-12). PR 38261 (hybrid offload planner, open since
   2026-03) is stale. llm-scaler PR 707 merged 2026-10-14 packages LMCache
   KVWeave for **Qwen3.8-27B**: host L1 for attention, conv, and SSM,
   `--mamba-cache-mode align`, `--enforce-eager`, `--enable-prefix-caching`,
   block size 1024, `max-model-len` 18432. Useful as a recipe shape. It is
   the dense hybrid, two orders of magnitude under 200k, and eager-only.
   Native connector first. LMCache second. A fork of the scheduler third,
   and only for a failure we reproduce.

6. **Prefix caching: turn on the upstream hybrid path; fork nothing yet.**
   Hits resume only on the Mamba block grid
   (`docs/features/automatic_prefix_caching.md:16-18`).
   `--enable-mamba-shared-prefix-checkpoint` stores the shared-prefix
   junction and takes effect only with prefix caching, EAGLE/MTP on the
   Mamba group, `--prefix-match-unit` smaller than the Mamba block, and
   no multi-module MTP (`automatic_prefix_caching.md:20-31`). XPU forces
   the GDN attention block size to a multiple of 64 (`xpu.py:462-494`),
   so `--prefix-match-unit 64` matches the documented example. Recent
   merged fixes on main: PR 60533 (2026-10-08, inconsistent hybrid Mamba
   boundaries), PR 51108 (hidden-state extraction), PR 57127 (hit-rate
   coverage). XPU CI skip PR 57301 (merged 2026-09-17,
   `test_hybrid_prefix_cache_hit_rate`) is a warning, not a disable of the
   feature. llm-scaler 0.26.0-b2 (2026-09) fixed a prefix-caching + MTP
   miscompile in their fork; that fix is not a patch we can see as a clean
   upstream commit from here. Re-measure on 0.32.

7. **MTP after the model serves, RecoverSSM later.** Flash-Next recipe
   string is `method=mtp`, not the older Qwen3-Next-80B `qwen3_next_mtp`
   (https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3-Next.html
   still shows `qwen3_next_mtp` and `--no-enable-chunked-prefill` for the
   80B). In-tree docs for MTP are generic (`docs/features/speculative_decoding/mtp.md`)
   and do not mention XPU. PR 61607's `hc_count` fix is part of making the
   MTP buffer the right width on this arch. PR 62111 (open) fixes
   multi-layer MTP module selection for Qwen3.5 and Qwen3-Next. RecoverSSM
   wiring is open vLLM PR 61309 (+3142/-309, 38 files, updated 2026-10-13,
   https://github.com/vllm-project/vllm/pull/61309); the SYCL kernel it
   needs is already in kernels 0.1.16.1. Closed and unmerged PR 42382
   (2026-05, spec-decode SYCL GDN dispatcher behind
   `VLLM_XPU_USE_SYCL_SPEC_GDN`) lost to the Triton/SYCL toggle that did
   merge. GDN on XPU today: if `torch.ops._xpu_C.gdn_attention` exists,
   `auto` selects the fused SYCL op, which replaces conv1d and decode as
   well as prefill; otherwise Triton
   (`qwen_gdn_linear_attn.py:110-112`, `:132-140`, `:170-179`). That
   toggle is commit `38168d5b82`, PR 56835, 2026-10-19. Kernels ship Xe2
   chunk GDN sources under `csrc/xpu/gdn_attn/xe_2/`. Initial XPU GDN
   enablement was PR 33657 (merged 2026-04-03): Triton attention, block
   size multiple of 64. CUDA-only GDN speedups (vLLM PR 31722, merged
   2026-01-06, H200 +16.8% decode throughput) do not apply.

8. **Graphs: first serve uses `--enforce-eager`.** `support_static_graph_mode`
   returns True (`xpu.py:516-517`). PR 51600 (merged 2026-09-24) turned XPU
   graph on by default and removed `VLLM_XPU_ENABLE_XPU_GRAPH`.
   `check_and_update_config` still forces `CUDAGraphMode.NONE` when
   `supports_xpu_graph()` is false or sleep mode is on (`xpu.py:362-377`),
   and it disables fusion passes (allreduce+RMS, attn+quant, RoPE+KV)
   (`xpu.py:379-397`). PR 60784 and PR 60529 (merged 2026-10-12 and
   2026-10-08) only made graph tests able to run. PR 50038 (open, updated
   2026-09-01) still gates FlashAttention-inside-graph on oneAPI 2026.0+.
   llm-scaler 1.5 release notes call XPU graph experimental. PR 61607 did
   not validate a captured graph. Historical FULL-capture stalls on this
   host (SYCL work-group scratch versus the graph extension) stay in
   force until a new capture succeeds. PIECEWISE is the second step.
   Keep `vllm/patches/xpugraph_submit_without_event.patch` off the branch
   until a capture stalls again.

9. **Expert parallel: use the upstream flag when a measured imbalance or
   a group-size reject says so.** Negative expert-id padding on XPU EP is
   fixed in kernels (`89bbfcc` PR 578, `c1fd11d` PR 572) and in vLLM
   (`0cc0460870`, PR 57787, merged 2026-10-09, DP+EP `topk_ids=-1`).
   llm-scaler 1.3 (vLLM 0.11.1, Jan 2026) already advertised TP+EP and
   DP+EP; that image is too old to build on. For the PR 61607 W4A16
   shape, TP=2 does not need EP to satisfy group_size 128. TP=4 does.

## Skip

| Item | State | Why it stays off the branch |
| --- | --- | --- |
| IPEX | Removed. RFC https://github.com/vllm-project/vllm/issues/33214 closed. PR 33379 merged 2026-02-03, PR 34111 merged 2026-02-11. `vllm/platforms/xpu.py` has no `ipex` string. | Kernels live in vllm-xpu-kernels. An IPEX build will not track main. |
| intel/llm-scaler as the runtime | Image `intel/llm-scaler-vllm:1.5` == `0.26.0-b2`, tag `vllm-1.5` (2026-10-16). https://github.com/intel/llm-scaler/releases/tag/vllm-1.5 | Six releases behind 0.32. UMD 26.14. Models called out are Qwen3.6-27B, Qwen3.8-27B, Qwen3.6-35B-A3B, Gemma 4. MTP/LoRA/FP8 KV there are for those models. Qwen3-Next-80B was added back in their 1.3 (vLLM 0.11.1, Jan 2026), then left behind by the 0.26 jump. |
| llm-scaler PR 660 | Open, +38365, stale since 2026-09-23. https://github.com/intel/llm-scaler/pull/660 | Not a reviewable port. PR 61607 is the same model at 1.6k lines. |
| PR 28855 CPU KV offload | Closed unmerged 2025-11-24. https://github.com/vllm-project/vllm/pull/28855 | Superseded by `OffloadingConnector`, which now lists XPU. |
| PR 42382 spec GDN dispatcher | Closed unmerged 2026-05-26. https://github.com/vllm-project/vllm/pull/42382 | Superseded by PR 56835 plus kernels RecoverSSM. |
| int3 INC / asymmetric GPTQ | int3 is outside `XPU_WNA16_SUPPORTED_BITS`. Asymmetric GPTQ linears are open PR 61731 (updated 2026-10-15). | Symmetric int4 is the merged XPU MoE contract. |
| ARK W4A8 MoE PR 56412 | Open, updated 2026-10-19. Tested on Qwen3-30B AutoRound, PP=2, eager. https://github.com/vllm-project/vllm/pull/56412 | Optional backend behind `VLLM_XPU_INC_WNA16_BACKEND=ark`. Default path already runs. Revisit after a sym-int4 baseline. |
| MXFP4 / online NVFP4 | PR 33679 merged 2026-03-09 (XPU MXFP4 MoE). PR 54006 open (AutoRound MXFP4 Hadamard, +647). PR 61602 open (online NVFP4 to AutoRound MXFP4, +891, updated 2026-10-19). | Real XPU quant work, wrong checkpoint. Use if we requantize to MXFP4. |
| AWQ | Matrix says Intel GPU yes (`README.md:69`). No B70 Flash-Next AWQ PR found. | Fine for a symmetric-layout checkpoint that the GPTQ/WNA16 path can express. Marlin is marked no on Intel (`README.md:71`). |
| CUDA/ROCm GDN and MoE fusions | sglang PRs 17094, 17981, 17983, 19434, 19775 (Blackwell/H200). vLLM PR 61325 (MI355X mono decode for this model, open). | Different ISA. sglang's useful XPU fact is below. |
| Our `vllm/contrib/vllm_wna16_xpu` dequant-to-bf16 kernel | Local, unwired, unverified on a real workload. | Unpacks int4 at load. That is the OOM the MoE patch was written to avoid. |
| Mooncake XPU PR 62546 | Open, 2026-10. https://github.com/vllm-project/vllm/pull/62546 | Store connector, not the RAM tier. |
| Level Zero IPC allreduce PR 54768 | Open, +249, updated 2026-09-14. https://github.com/vllm-project/vllm/pull/54768 | Watch once TP=2 collectives show up in a profile. Standing rule still forbids arbitrary `CCL_TOPO_P2P_ACCESS=1` vLLM TP>1. |

## Item notes

### intel/llm-scaler and other Intel vLLM trees

Repo: https://github.com/intel/llm-scaler (active, release 1.5 on
2026-10-16). Patches actually carried in-tree are small and visible at
https://github.com/intel/llm-scaler/tree/main/vllm/patches :
`vllm_for_multi_arc.patch`, `vllm_xpu_kernels.patch`, oneCCL align,
LMCache KVWeave. The image is a patched vLLM, not a clean fork we can
rebase. 0.26.0-b2 fixed three correctness bugs they had shipped: prefix
caching with MTP, symmetric INT4 output, and block-FP8 under concurrent
decode. Those symptoms are worth re-checking on 0.32; the patches
themselves are not a starting point.

No other Intel org fork showed a newer Qwen4Exp XPU port than PR 61607
and the abandoned-looking PR 55068 (open, +5045/-3, empty body, author
xiaolong-intel, updated 2026-09-23,
https://github.com/vllm-project/vllm/pull/55068). 61607 is newer, smaller,
and describes the model. 55068 is the one to ignore if the two conflict.

### vllm-xpu-kernels and IPEX

https://github.com/vllm-project/vllm-xpu-kernels. Latest release v0.1.16.1
(2026-10-16). README on that tree requires Python 3.12, torch 2.14.0+xpu,
oneAPI 2026.1. IPEX migration finished before vLLM 0.16 (RFC 33214). vLLM
main imports kernels through `torch.ops._xpu_C` and does not import
`vllm._C` on XPU (`xpu.py:134-137`). Pin gap is the action in rank 1.

v0.1.12 era notes that still matter: Xe 2D-block surface-height overflow
on large paged KV was fixed in that line. 200k context will stress that
path again; treat a startup failure at large `num_gpu_blocks` as a kernels
bug first, not a fork of the scheduler.

### GGUF, int4, int3, MXFP4, AWQ, GPTQ

| Format | Where | Status | For this campaign |
| --- | --- | --- | --- |
| GGUF, including IQ3_XXS kernels | vllm-gguf-plugin `e2b8ad5`; docs matrix line 77 | Plugin current as of the clone (2026-09-24). Intel GPU marked unsupported. IQ3_XXS Triton GEMM and MoE exist. Remote id `IQ3_XXS` is accepted (`tests/test_gguf_utils.py`). | Try as a plugin on a tiny shape before any SYCL port. Hybrid Qwen4Exp weight mapping is unverified. |
| GPTQ sym int4 MoE | main, `XPUExpertsWNA16` | Merged (PR 41426, 2026-05-28, then the oracle rewrite). | Primary quant if we leave GGUF. |
| AutoRound sym int4 | main INC, `auto_round_lib==0.16.0` | In the XPU requirements. Routed experts no longer fall through to bf16. | Same path as GPTQ when `is_gptq` and `sym` (`inc_wna16_scheme.py:163-165`). |
| AutoRound / GPTQ int3 | INC bits set is {2, 4} on XPU | Unsupported in that scheme. CUDA humming covers 2/3/5/6/7 (`inc_wna16_scheme.py:211-216`) and is CUDA-only. | IQ3 is not this format. |
| Asymmetric GPTQ | PR 61731 | Open. | Skip unless the checkpoint has qzeros. |
| AWQ | docs matrix line 69 | Intel GPU marked yes. No Flash-Next XPU PR. | Usable if a checkpoint is AWQ; Marlin is marked no. |
| MXFP4 MoE | PR 33679 merged; 54006 and 61602 open | Maintained. | Requantize option, not the DD pack. |
| llm-compressor W8A8 / FP8 | docs matrix lines 72-74 | Intel GPU marked no for those rows. `XPUPlatform.supports_fp8` is still True (`xpu.py:551-552`). Online FP8 is what llm-scaler ships. | Separate from this GGUF/int4 migration. |

### KV offload, prefix cache, MTP, graph

Covered in the ranked list. The 200k "evict to RAM and restore without
recompute" goal is not a merged, measured XPU feature for a hybrid MoE.
What is merged is: XPU listed on `OffloadingConnector`, Mamba align-mode
accounting, block swap to host, and a B70 proof on the dense Qwen3.8-27B
hybrid with MTP. What is open is the boundary-state backfill that makes
a divergent hybrid hit cheap (issue 52773 and PRs 58413, 61019, 61282).

### Other projects, only where they change a port

SGLang XPU is real and current. Docs: https://docs.sglang.io/docs/hardware-platforms/xpu
(Arc Pro B-series, `--device xpu`, `--attention-backend intel_xpu`, page
size 32/64/128). Kernels: https://github.com/sgl-project/sgl-kernel-xpu
with build flags `USE_GDN`, `USE_QSA`, `USE_MOE`. `docker/xpu.Dockerfile`
on sglang main pins oneAPI essentials 2026.1.2 and Compute Runtime
**26.27.39122.11**, newer than this host's 26.22. Q2 2026 roadmap
(https://github.com/sgl-project/sglang/issues/24922, closed) listed FLA
Triton-to-SYCL, experimental Qwen3.5 MTP, FP8, MXFP4, and EP readiness,
aimed at small and medium Qwen3/3.5, bf16 first. Porting lesson: vLLM
already has the Xe2 GDN SYCL op, so we do not need sglang's GDN. If a
Flash-Next checkpoint actually contains QSA indexer tensors, sgl-kernel-xpu
is the public SYCL QSA reference; PR 61607 refuses that path on XPU.
sglang's Blackwell GDN fusions do not port. SGLang site listed v0.5.22 on
2026-10-19; that version was not checked out here.

llama.cpp SYCL is the current DD engine (Strata night-final-v3) and the
reason IQ3_XXS, GDN, MTP, and RAM parking already run on these cards.
Nothing in vLLM's XPU tree replaces that combination. The port is the
serving features in the ranked list, with GGUF IQ kept behind the plugin
boundary so a failure falls back to symmetric int4 instead of a ggml fork
inside vLLM.

## Verified versus inferred

Verified by reading the clone or the PR/release text:

- SHAs, the missing `qwen4_exp/xpu` directory, requirements pins, INC MoE
  routing, WNA16 XPU backend selection, GDN SYCL-or-Triton selection,
  hybrid KV and graph platform flags, GDN block-size rounding, the quant
  matrix, prefix-caching docs, OffloadingConnector's XPU sentence and
  Mamba align helpers, kernels GDN/RecoverSSM/Qwen4Exp TP=2 commits, and
  the GGUF plugin's IQ3_XXS Triton files.

Inferred, and not run:

- That Triton-XPU will compile the GGUF plugin's IQ kernels.
- That our ISTA IQ3_XXS pack is the same tensor layout as the W4A16 file
  PR 61607 loaded (`devan-carlin/Qwen3.8-Flash-Next-W4A16`, 512 routed
  experts, `moe_intermediate_size=640`). Expert count of the local GGUF
  is task 02's job.
- That `OffloadingConnector` on XPU uses DMA rather than the synchronous
  `.cpu()` helper.
- End-to-end speed. No numbers in this report are from this host.
