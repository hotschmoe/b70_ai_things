# SergiioB Qwen3.8-27B vLLM XPU cookbook -- loop digest

**Ingested:** 2026-08-18v (operator). Not a GPU loop.
**Source page:** https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook/blob/master/docs/qwen38-27/QWEN38-VLLM-XPU.md
**Repo hub:** https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook
**Older on-box work:** `docs/COOKBOOK_CAMPAIGN.md` (2026-08-10) is the **3.6** family
on digest `2c427ef...` / vLLM 0.26.1rc1. Do not mix those numbers or that
image with this 3.8 page.

Loops read this after the campaign living header when the pick is S1 or
any Phase 2 / 0.27 item. Do not photocopy the 83.7 cell onto W8A8 TP=2.

## What the page actually is

1x B70, C1, client post-first, GPTQ-Int4 (W4A16-class) + BF16-preserved
MTP. Kernel `XPUwNa16LinearKernel`. Not W8A8. Not DSpark. Not TP=2.

| pin | value |
|---|---|
| Weights | `SergiioB/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16` rev `9d189a60e4c0ad7f9f47cd94bfa393ca10b3924e` |
| Quant | gptq 4-bit, sym, G128, desc_act=false; `mtp.*` excluded (stay BF16) |
| Image | `vllm/vllm-openai-xpu@sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f` |
| vLLM / kernels | `0.27.2rc1.dev77+gac7509e2b` / `vllm-xpu-kernels 0.1.12.3` |
| Patches (order) | `patch_mtp_nightly.py` then `patch_mtp_boundary.py` (same pair we already ported for 3.6) |
| Env | `B70_MTP_BF16_DRAFT=1` `VLLM_XPU_ENABLE_XPU_GRAPH=1` `ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE` |
| Serve | `--quantization gptq --dtype float16 --max-model-len 131072 --kv-cache-dtype fp8 --language-model-only --no-enable-prefix-caching` |
| UTIL | 0.90 no-spec; **0.88** MTP |
| Scheduler | `--max-num-seqs 64 --max-num-batched-tokens 8192` |
| Power | write 230 W cap; driver rejects 300 W; no xe clock knobs |
| Agent | `docs/qwen38-27/PI-AGENT-BACKEND.md` -- tool parser `qwen3_xml` (3.6 used `qwen3_coder`) |

Headline they published (LocalMaxxing `cmsur82fz06svms01ga1f0z83` APPROVED):
MTP4 p512/g128 **83.7** decode, cold input **1774**, ctx 131072.

## Their 4-mode table (do not mix with our bench_code)

Client post-first, n=5, cache off, 230 W, C1, 131072.

Decode p512: no-spec 32.9 / MTP1 52.0 / MTP2 65.8 / **MTP4 83.7** (g128).
Decode p8192 MTP4: 77.1 (g128), 52.1 (g512).
Full ctx p130944/g128 MTP4: 56.3. p130560/g512 MTP4: 36.2.
MTP accept: MTP1 ~100%, MTP2 96-99%, MTP4 93-96%.
One corrupted MTP4 p8192/g32 rep (41587 tok/s) -- they discarded the mean.

Their 3.6 dense GPTQ-Int4 MTP4 on the *older* digest was 69.3 at the same
cell. 3.8 is the new dense ceiling they claim.

## What is useful to THIS campaign

- Written 0.27-only feature list (campaign Phase 2 gate): nightly
  `0.27.2rc1` + kernels `0.1.12.3`, `B70_MTP_BF16_DRAFT`, 131k MTP
  boundary, fp8 KV on 3.8 dense, XPU graph on, language-model-only.
- MTP4 on a BF16-preserved GPTQ draft still looks like a 2.5x vs
  no-spec on 1 card. Our W8A8 MTP3 GRAPH=1 is 26.62 on TP=2
  `bench_code`; different harness and scheme.
- fp8 KV is how they fit 131k on 1x 32 GiB. Our W8A8 3.8 serve has
  **no KV_FP8 hook** (P0.2); 262k MTP-off used bf16 KV; DSpark GRAPH=1
  fit is 122880. Steal the hook later, do not assume it exists on 0.26.
- Same two MTP patches we already have under `vllm/patches/cookbook/`.
  Re-hash before applying to 0.27 -- anchors may have moved.
- Image digest is a **third** generation. Do not use 3.6 `2c427ef` or
  Nemotron `1da0a954` for 3.8. Do not apply Nemotron grouped-topk / SSU.
- Tool parser for 3.8 agents is `qwen3_xml`, not `qwen3_coder`.

## What is wrong or not transferable

- "NVFP4 unsupported on Intel" -- false on this box (3.6 NVFP4 DD).
- "FP8 block has no XPU scaling kernel" -- true enough for *linear*
  FP8 (we already refuse Xe2 FP8 GEMMs). Not a reason to drop W8A8.
- "AWQ / compressed-tensors not proven" -- false here. Our 3.8 W8A8
  is compressed-tensors GPTQ + `XPUInt8ScaledMMLinearKernel`.
- 83.7 is not a W8A8 number and not a TP=2 number and not
  `bench_code` c1. Item-4 policy in `docs/COOKBOOK_CAMPAIGN.md` still
  holds: INT4+MTP is a C1 ceiling reference. It does not demote W8A8.
- Their speed recipe disables prefix cache. Our campaign G5 / resident
  load cares about cache. Do not copy `--no-enable-prefix-caching` into
  the W8A8 shelf.
- `--language-model-only` drops vision. Our graft keeps the tower.
- Power cap 230 W is host-level. Do not change it mid-loop without an
  A/B; we did not set it in the 08-10 campaign.

## Related repo pages worth stealing later

| page | take |
|---|---|
| `docs/IMAGE-AND-PATCH-MATRIX.md` | three-family digest law |
| `docs/nemotron35-30a3/NEMOTRON-DFLASH-B70.md` | official `method=dflash` n=7 on 0.26.1.dev668; **186** C1. Our 0.26.0 still has method=dflash unregistered. Phase 3/2, not this speed window. |
| `docs/qwen36-27/DENSE-FP8-GAP.md` | no FP8 linear kernel on XPU -- matches Xe2 rule |
| `docs/BENCHMARK-FORMAT.md` | client post-first contract (we already have `phase_bench.py`) |
| `docs/qwen38-27/PI-AGENT-BACKEND.md` | thinking vs non-thinking sampling + `qwen3_xml` |

## Queued pick (do not jump the speed window)

**S1** -- after P1.6 fusedq has a verdict: 1-card smoke of this exact
3.8 GPTQ-Int4 + MTP4 pin (`gpu-run --card 0`). G1 Paris first. One
`phase_bench` p512/g128 cell. Compare to their 83.7 and to our 08-10
3.6 dense 52.1 (this box, shorter ctx). Then restore the W8A8 DSpark
serve. Do not start a 0.27 W8A8 kernel rebuild in that fire.

Do not enter Phase 2 as "pip install 0.27" just because this page
exists. Phase 2 is still a dedicated ABI rewrite.
