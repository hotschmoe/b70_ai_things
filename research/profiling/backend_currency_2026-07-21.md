# Backend currency check -- 2026-07-21

Research-only (no GPU, no builds). Question: is any backend/toolchain upgrade worth
doing now, given our torch==2.12.0+xpu ABI lock on the custom int8/nvfp4 oneDNN .so?

Current baseline (from context / docs):
- vLLM 0.25.1 on torch 2.12.0+xpu. Images vllm-xpu-env:{v0251,int8g-v0251}. Four XPU
  regressions already fixed in 0.25.1 (device-env, int8g anchor, oneCCL 2021.17,
  hpc_rope splitting_op).
- sglang 0.5.6 working (torch 2.12.0+xpu); 0.5.15 images built but W8A8 serve BLOCKED --
  pip pulls torchcodec which upgrades torch to 2.13.0+cu130 (CUDA), clobbering the xpu
  wheel. See docs/SGLANG_0515_UPGRADE.md, sglang/NVFP4_PORT.md.

## Summary table

| item | newest version | XPU-relevant? | torch-ABI impact | effort | go/no-go |
|------|----------------|---------------|------------------|--------|----------|
| vLLM | 0.25.1 (newest; = what we run) | n/a (already current) | none -- stays 2.12.0+xpu | none | NO-OP: already newest |
| sglang | 0.5.15.post1 | yes (XPU + PP-on-XPU landed 0.5.11) | keeps 2.12.0+xpu IF torchcodec/CUDA-triton pin held; else 2.13-CUDA break | medium (constraints pin + bounded rebuild) | DEFER |
| torch-xpu-ops #2992 (XCCL SYCL-graph guard) | merged to main 2026-05-07, no release backport | yes (in-graph allreduce record/replay = our captured-decode leak) | needs torch >= 2.13+xpu -> ABI break -> rebuild ALL custom .so | high | NO-GO this session |
| oneCCL SYCL-graph allreduce | needs oneCCL >= 2021.17.2 (maybe 2022.0); we ship 2021.17 | yes | tied to a torch bump | high | NO-GO this session |
| vLLM-XPU PP=2 | supported on 0.25.1 (beta, mp, single-node) | yes (TP=2 vs PP=2 A/B) | none -- current stack | low (config-only) | GO (only actionable item) |

## Item 1 -- vLLM newer than 0.25.1?

No. 0.25.1 is the newest tag on vllm-project/vllm as of 2026-07-21; it is a 2-commit
bugfix on top of 0.25.0. There is no 0.25.2 / 0.26.x / 0.27.x. We are already current,
torch stays 2.12.0+xpu, and our ABI-locked .so keeps loading -- nothing to do.

Context for the last two real releases (in case a future bump matters):
- 0.25.0 (2026-07-11, 558 commits): torch stable-ABI migration, universal spec-decode
  for heterogeneous vocabularies (TLI) + new DSpark/DFlash drafters, MTP for Bailing
  hybrid models.
- 0.24.0 (2026-06-29): XPU W8A8 FP8 linear kernel, XPU **pipeline-parallel accuracy
  fix**, DeepSeek-V4 attn/MoE. (This is the release that made XPU PP viable -- see item 4.)
- vllm-xpu-kernels track (v0.1.7) carries block_fp8_moe, block-scaled W8A8 FP8, WNA16
  GPTQ sym-int4 oracle, rms_norm/act-quant fusions, GDN-attention MTP -- relevant if we
  ever move off our hand-built oneDNN ops, but not tied to a core vLLM version bump.

Sources: https://github.com/vllm-project/vllm/releases ,
https://docs.vllm.ai/en/latest/models/hardware_supported_models/xpu/

## Item 2 -- sglang newest + torchcodec pin problem

Newest is 0.5.15.post1 (post1 is a patch over the 0.5.15 we already built; it does not
change the dependency structure, so the torchcodec block persists in the newest too).

The torchcodec -> torch-2.13.0+cu130 clobber is a dependency-resolution problem, not an
sglang-XPU code gap. The official sglang XPU install path already pins the xpu wheels
(`torch==2.12.0+xpu torchao==0.17.0+xpu torchvision==0.27.0+xpu torchaudio==2.11.0+xpu`
via `--index-url https://download.pytorch.org/whl/xpu`) and documents the SAME class of
fix we need: it calls out that "xgrammar will introduce CUDA-enabled triton which might
conflict with XPU" and installs it `--no-deps`. torchcodec is not named explicitly, but
the documented remedy pattern applies directly -- install torchcodec `--no-deps` (or drop
it; it is audio/video decode, not needed for our text+vision serve) and/or add a pip
constraints file pinning `torch==2.12.0+xpu`, then a bounded rebuild. That is the
approach docs/SGLANG_0515_UPGRADE.md already scopes.

Verdict: fixable at medium effort with no torch bump, but 0.5.6 serves today, so there
is no urgency. DEFER to a dedicated sglang session; do not spend GPU time this session.

Sources: https://github.com/sgl-project/sglang/releases ,
https://lmsysorg.mintlify.app/docs/hardware-platforms/xpu (sglang XPU install doc)

## Item 3 -- torch-xpu-ops #2992 / oneCCL SYCL-graph allreduce

torch-xpu-ops PR #2992 ("XPU-graph-aware XCCL guard") is MERGED, but to `main` only on
2026-05-07 -- no milestone, no release/2.12 or release/2.13 backport visible. It adds
graph-capture awareness to XCCL collectives (skips watchdog/enqueue paths under graph
record) and a version gate: oneCCL SYCL-graph record/replay is only supported on
**oneCCL >= 2021.17.2 (maybe 2022.0)**. Tests cover allreduce/allgather/reduce_scatter;
broadcast and P2P are explicitly NOT yet supported.

Impact on our captured-decode all-reduce leak: our shipped stack is torch 2.12.0+xpu with
oneCCL 2021.17 -- BELOW the 2021.17.2 threshold AND without the #2992 guard (main-only,
not in any 2.12 wheel). So in-graph oneCCL allreduce record/replay is genuinely
unavailable to us today; that is exactly why the graph-reclaim workaround
([[mtp-graph-neo-abort-drafter-eager-fix]]) and the push-AR (L0-IPC) path exist. Getting
the native in-graph allreduce would require torch >= 2.13+xpu (whenever main is cut) with
oneCCL >= 2021.17.2 -- which is an ABI break that forces a rebuild of ALL our custom .so
(int8/int4/nvfp4 oneDNN ops are ABI-locked to torch 2.12). High effort, and the
reclaim/push-AR workarounds already give crash-free captured+MTP at full speed, so the
payoff is marginal.

Verdict: NO-GO this session. Keep watching for a torch 2.13+xpu release that ships #2992
+ oneCCL >= 2021.17.2; revisit as a batched "torch 2.13 rebuild" effort, not piecemeal.

Sources: https://github.com/intel/torch-xpu-ops/pull/2992 ,
https://github.com/intel/torch-xpu-ops (branches/releases)

## Item 4 -- PP (pipeline parallel) on vLLM-XPU

SUPPORTED on our current 0.25.1 stack. vLLM XPU docs list pipeline parallel as a BETA
feature for online serving, single-node, multiprocessing (mp) backend, invoked with
`--pipeline-parallel-size=2` (tensor parallel is fully supported for both offline and
online). 0.24.0 landed an XPU-specific "pipeline-parallel accuracy fix", so the path is
maintained, not vestigial.

This means the TP=2-vs-PP=2 A/B for the dense 27B (to cut the per-layer all-reduce that
we currently attack with push-AR) can be run on the CURRENT stack with NO version change,
NO rebuild -- config-only. Caveats to plan for: PP is single-node/mp/beta, and PP splits
by layer so the 27B weight+KV must fit the 2-way layer split per card; and vLLM's
multiproc + our TP-wedge history means the P2P/oneCCL cautions in AGENTS.md still apply.

Verdict: GO -- this is the one actionable item and it needs no upgrade. Slot a PP=2 A/B
into the next GPU session against the NVFP4/W8A8 TP=2 daily driver.

Sources: https://docs.vllm.ai/en/latest/getting_started/installation/gpu.html (XPU tab) ,
https://docs.vllm.ai/en/latest/models/hardware_supported_models/xpu/

## Recommendation (this session vs deferred)

1. Do NOTHING on versions this session: vLLM 0.25.1 is already the newest release, and
   both sglang 0.5.15 and torch 2.13 (for #2992's in-graph allreduce) are net-negative
   right now -- they either force a torch bump that breaks our ABI-locked .so or give no
   win over the reclaim/push-AR workarounds we already have.
2. The ONLY worthwhile near-term action is config-only: A/B TP=2 vs PP=2 for the dense
   27B on the current 0.25.1 stack (PP is supported, beta, mp, single-node) -- no rebuild.
3. sglang 0.5.15 unblock (torchcodec --no-deps / torch==2.12.0+xpu constraints pin) is a
   clean medium-effort task but non-urgent (0.5.6 works); DEFER to a dedicated session.
4. torch 2.13+xpu (for #2992 + oneCCL >= 2021.17.2 native SYCL-graph allreduce) is a
   future BATCHED rebuild-everything effort, not piecemeal; watch for the release.
5. Net: no upgrade this session; queue the PP=2 A/B as the actionable follow-up.
