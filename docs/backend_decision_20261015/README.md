# Backend decision experiments E1-E4 (2026-10-15)

2026-10-15 source-audit note: the [interpretation follow-up](source_audit_followup.md)
qualifies strong original E2/E3/E4 claims, including E3's replay fixed-point
control limitation. Original scripts, measured rows and verdict text below
remain preserved as historical evidence.

Follow-up to [the backend investment review](../20261015_backend_investment_review.md).
Four small experiments chosen because each could change where backend-locked
work (scheduler, cache/state ownership, speculative rollback) is invested.
Run from a separate session alongside active Strata work, coordinated through
bin/gpu-run, bin/cpu-run and /mnt/vm_8tb/b70/agent-coordination.log.
Scripts are in scripts/; raw results in /mnt/vm_8tb/b70/build/backend-decision/.

User priorities for this decision (2026-10-15): C1 and C2-C4 weighted
equally; future formats GGUF and safetensors low-bit both possible; speed now
with hedged risk.

Consultation: before the experiments, independent critiques came from
claude-opus-5-5 and claude-fable-5-1 (self-reported model IDs; Agent tool
subagents, not CLI metadata). Both converged on "no new locked investment
until cheap discriminators run". The E1-E4 set follows their proposals.

## Pinned identities

| Item | Identity |
|---|---|
| llama.cpp | e467e5b5e5ba43071d6151b3d881deb9300d000e (build 11644) |
| SGLang | 9ea5472833 (2026-10-15), torch 2.14.0+xpu, sglang-kernel-xpu 0.4.0, triton-xpu 3.8.0 |
| Model | ISTA GSQ-RCO IQ3_XXS GGUF (models/files/qwen3.8-flash-next/ista-gsq-rco-iq3-xxs) |
| MTP head (llama.cpp) | Unsloth mtp-Qwen3.8-Flash-Next-Q4_K_M.gguf sidecar (ISTA GGUF has no nextn layers) |
| Runtime | oneAPI 2026.1, compute runtime 26.22.38646.4, AOT bmg-g31, kernel 7.1.0-070100 |

## E1 llama.cpp in-RAM hybrid-state restore (CPU only)

CONFIG: CPU llama-server, -np 2, -c 8192, --cache-ram 8192, -ctxcp 8, 16
threads. Four ~870-token chats, 32-token replies; chats C,D force A,B out of
the slots; returns add a 17-token follow-up. Token arrays end to end.

COMMAND: scripts/e1_restore.py (evict and control phases on fresh servers),
scripts/e1-mtp-arm.sh (same with --spec-type draft-mtp, n-max 3),
scripts/e1_conc.py (two slots decoding concurrently).

RESULT:

| Arm | Return reuse (held/reused/evaluated) | Return vs control |
|---|---|---|
| no MTP | 903/902/18 and 902/901/18 | both exact |
| draft-mtp | same | both exact; acceptance identical (14/24 each) |
| draft-mtp, two slots concurrent | same | first replies exact; returns diverge mid-reply, coherent |

The one extra evaluated token is llama.cpp re-evaluating the last held token
for logits; zero retained-history replay. MTP versus no-MTP differs at token
16 of 24 on return A (numerical near-tie under batched verify), not a
restore effect.

VERDICT: upstream llama-server restores complete qwen4exp hybrid state
(attention KV, GDN, checkpoints) from host RAM without replay for this
model on CPU. The "hybrid restore is broken" objection (issue 28194 is the
disk-slot path) does not apply to the in-RAM cache. Concurrent-slot
divergence is consistent with batch numerics; see E2b leak check.

## E2 llama.cpp SYCL on the exact IQ3_XXS (both cards)

CONFIG: SYCL F16, GGML_SYCL_GRAPH on, AOT bmg-g31, -ngl 99, -fa on, all
experts in VRAM. Pre/post bin/xpu-health and post xpu-collective-health.

COMMAND: scripts/e2-bench.sh (llama-bench pp4096/tg128, r=2);
scripts/e2b-run.sh + e2b_serve.py (llama-server -sm layer, draft-mtp n-max 3,
Strata layer-mtp4 coding template, 1500 max tokens, plus a sysadmin prompt;
then both concurrently).

RESULT:

| Arm | pp4096 tok/s | decode tok/s |
|---|---|---|
| llama.cpp -sm layer, no MTP | 606.09 +- 1.65 | 26.24 +- 0.01 |
| llama.cpp -sm tensor, no MTP | 400.67 +- 1.04 | 15.47 +- 0.08 |
| llama.cpp layer + draft-mtp, C1 coding | - | 37.27 (TTFT 2.31 s, acceptance 0.78, mean len 3.35) |
| llama.cpp layer + draft-mtp, C1 sysadmin | - | 33.87 (acceptance 0.69) |
| llama.cpp layer + draft-mtp, C2 | - | 25.57 / 23.80 per stream |
| Strata reference, same coding template (layer mtp4) | 1260-1337 at 4K (goals doc) | 84.37 |

C2 outputs: no cross-topic leak; they equal C1 for the first 1391/1042
characters, then diverge coherently. All health checks pass.

VERDICT: llama.cpp SYCL runs this model at about 0.44x Strata C1 decode,
about 0.5x C2 per stream and 0.45-0.5x prefill, with MTP acceptance
comparable to Strata. The gap is kernels and execution, not speculation or
server state. Tensor split is slower than layer split on this host. Under
the pre-agreed rule (<=0.5x) llama.cpp is a kernel-contribution target, not
the next primary serving backend.

## E3 small-message all-reduce latency (both cards)

CONFIG: torchrun 2 ranks, xccl, CCL_TOPO_P2P_ACCESS=0, [rows,2560] bf16,
96 all-reduces per "token" with a dependent op between them, 200 timed
tokens. Eager, and one token's chain captured in torch.xpu.XPUGraph.
Values verified after replay (chain preserves 1.0 and 3.0 only if every
reduction executes).

COMMAND: scripts/e3-run.sh (container needs SYS_PTRACE + seccomp unconfined
for oneCCL IPC handles; without them oneCCL fails "device_fd is invalid").

RESULT:

| Arm | per sync p50 | per 96-sync token p50 | value check |
|---|---|---|---|
| eager, 1 row (5 KB) | 94.06 us | 9.03 ms | pass |
| eager, 4 rows | 97.94 us | 9.40 ms | pass |
| XPUGraph, 1 row | 12.64 us | 1.21 ms | pass |
| XPUGraph, 4 rows | 21.81 us | 2.09 ms | pass |

Post xpu-health and collective health pass.

VERDICT: eager xccl alone caps a 96-sync TP decode near 110 tok/s, but
captured xccl all-reduce is correct and costs about 1.2 ms per token at C1.
TP is not ruled out by synchronization on this PCIe3 host, provided the
engine captures the whole decode step including collectives. llama.cpp's
tensor-split slowdown is its own reduction path, not a hardware limit.

## E4 SGLang head XPU host overhead (card 0)

CONFIG: SGLang head in an isolated venv (scripts/e4-run.sh), single card,
4-layer dummy-weight Flash-Next (3 GDN + 1 QSA, 512 experts, PLE layer kept,
ngram table shrunk to 200k; scripts/e4-model-config.json), bf16,
--disable-radix-cache, attention backend triton (default) and intel_xpu.
Arms: eager and --cuda-graph-backend-decode full.

Setup findings (upstream head as of 2026-10-15):
- qwen4_exp.py imports layers/moe/qwen4_decode unconditionally since #42175
  (Blackwell decode optimization, 2026-10-09); that pulls in tvm_ffi, which
  pyproject_xpu does not install, so Qwen4Exp cannot be constructed on a
  clean XPU install. Installing apache-tvm-ffi segfaults on import here. A
  local try/except guard in the venv copy (prepare_qwen4_decode_comm already
  returns None off CUDA) was used; this is a small upstream fix candidate.
- xgrammar must be installed --no-deps as in docker/xpu.Dockerfile.

RESULT:

| Arm | C1 tok/s | C2 per stream | C4 per stream | C1 ITL p50 |
|---|---|---|---|---|
| eager, triton attention | 29.93 / 29.59 | 28.74 | 27.23 | 33.4 ms |
| eager, intel_xpu attention | 30.18 / 30.16 | 28.93 | 27.05 | 33.1 ms |
| decode graph (both backends) | capture FAILED | - | - | - |

Graph capture fails in qsa/kernel.py qsa_fast_topk ("selected index k out
of range"), then "Failed to instantiate native UR executable graph".
Card-0 health passes after every arm.

VERDICT: SGLang eager costs about 33 ms per decode step for only 4 of 48
layers, nearly flat from C1 to C4 (per-step overhead, not per-row compute).
Scaling linearly to 48 layers would be far beyond the 6.7 ms budget for
150 tok/s (assumption: linear scaling; not profiled per op). Decode graph
capture, the mechanism E3 shows TP needs, does not work for Qwen4Exp on XPU
at head. Under the pre-agreed rule SGLang is out as the C1 primary for
Flash-Next today. It stays the candidate home for safetensors low-bit
breadth, contingent on the QSA capture fix and the import regression.
Caveat: dummy weights and a 4-layer model; real-weight numerics untested.

## Decision

- Strata remains the home for backend-locked Flash-Next work. It is the
  only measured engine near the targets (84 tok/s C1, ~1300 prefill at 4K,
  full RAM recovery).
- Portable work (SYCL quant kernels, transfer, NUMA, profiling) should be
  written to be upstreamable to ggml-sycl. llama.cpp already has correct
  hybrid RAM restore, per-slot batched MTP and model breadth; its gap is the
  kernel/execution path. Closing that gap upstream is the strongest hedge
  against Strata's single-maintainer risk.
- SGLang: watch and contribute small fixes (the #42175 XPU import guard,
  the QSA top-k capture failure). Re-run E4 when decode graph capture works
  for Qwen4Exp on XPU.
- Revisit triggers: Strata upstream rejecting our state/EP PRs, ggml-sycl
  reaching >=0.75x Strata C1 decode on this model, or SGLang XPU capture
  working with eager overhead <3 ms/step on the reduced model.

No daily-shelf change, no endpoint migration, and no AGENTS.md backend
change follow from these experiments.

## Upstream contribution: SGLang qwen4_exp XPU import (2026-10-15)

CONFIG: upstream SGLang bba535e (2026-10-15) still imports
layers/moe/qwen4_decode unconditionally; the chain is qwen4_decode ->
custom_all_reduce_v2 -> kernels/ops/communication/all_reduce -> tvm_ffi
(reproduced in the E4 venv with no GPU device: ModuleNotFoundError).
Fix: gate the import and prepare_qwen4_decode_comm on is_cuda().

COMMAND: scripts/e4-prcheck.sh eager under bin/gpu-run --card 0 (E4 eager
arm with the PR file bind-mounted over the venv copy; upstream file is
otherwise byte-identical to the E4 venv's 9ea5472833 copy).

RESULT: server starts and serves C1 ~30.3, C2 ~28.9, C4 ~27.1 tok/s per
stream (same as E4 eager; not a timing claim), no tvm_ffi error, card-0
xpu-health HEALTHY. ruff check, ruff format and isort pass.

VERDICT: submitted as https://github.com/sgl-project/sglang/pull/44561
(branch hotschmoe/sglang:fix-qwen4-exp-xpu-import). The QSA top-k capture
failure is not addressed by it.

## E5/E6 matched harness, llama.cpp SYCL correctness bug, decode census (2026-10-15 evening)

Run between Strata S98-S100 windows; every window announced and released in
/mnt/vm_8tb/b70/agent-coordination.log. Raw results:
/mnt/vm_8tb/b70/build/backend-decision/e5e6-results/.

### llama.cpp SYCL returns garbage on short prompts (reorder path)

CONFIG: same llama.cpp e467e5b5 SYCL build, ISTA IQ3_XXS, -sm layer, -fa on,
-np 2; prompt "Return only the integer: 17 * 19." (Strata's harness
coherence gate) and padded variants.

COMMAND: scripts/e5b-run.sh (MTP on/off, serial/concurrent),
e5c-run.sh (pad sweep), e5d-run.sh (exact E2b server command, E2b and
Strata request formats), e5e-cpu.sh (CPU build), e5f-run.sh (SYCL switches).

RESULT:

| Arm | arithmetic | E2b coding prompt |
|---|---|---|
| SYCL, MTP on or off, serial or concurrent | garbage ("333333 Mountfonoon...") | coherent |
| SYCL, prompts 25-441 tokens | garbage at every length | - |
| SYCL, E2b-identical command, either request format | garbage | coherent |
| CPU build, same GGUF | "323" | coherent |
| SYCL GGML_SYCL_ENABLE_GRAPH=0 / FUSION=0 / DNN=0 | garbage | coherent |
| SYCL GGML_SYCL_ENABLE_OPT=0 | "323" (both formats) | coherent |

The flash-attention-off arm OOMed at load (UR_RESULT_ERROR_OUT_OF_RESOURCES)
and was stopped. All post xpu-health and collective health passed.

VERDICT: a prompt-dependent correctness bug in the ggml-sycl reorder
optimization (ENABLE_OPT; reordered IQ3/Q6_K mat-vec kernels are in the E6
trace) on this model. E2's llama.cpp speed rows were measured with this
path on and are superseded by the OPT=0 rows below.

Follow-up: the cause is upstream #29107 (IQ3_S/IQ3_XXS reorder, merged
2026-10-14) meeting the plain fused-dequant GEMM; open PR #30604 fixes it.
e467e5b5 + PR 30604 (scripts/pr30604-build.sh, 7 min incremental under
cpu-run; scripts/pr30604-check.sh) with default OPT: arithmetic 14/14 correct
at 25-441 tokens, coding coherent, post health PASS. Reported as a
confirmation comment on the PR instead of a duplicate issue:
https://github.com/ggml-org/llama.cpp/pull/30604#issuecomment-6203833524

### E5 matched serving (Strata's own harness), llama.cpp OPT=0

CONFIG: GGML_SYCL_ENABLE_OPT=0, -sm layer, -fa on, -np 2, draft-mtp n-max 3
(Unsloth MTP sidecar). Harness scripts/e5_matched.py = bench_serving.py coding
case (3 fresh repeats, 1500 tokens) + bench_concurrency.py logic (warm pair +
3 fresh pairs), same prompts/caps/metrics; only the identity alias differs
(hotschmoe-dd + ...-llamacpp-sycl). cpu-run --quiet around the window.

COMMAND: scripts/e5-opt0-run.sh

RESULT:

| Metric | llama.cpp OPT=0 | Strata (same harness) | ratio |
|---|---|---|---|
| llama-bench pp4096 | 604.69 | ~1300 at 4K (goals doc) | ~0.47x |
| llama-bench tg128, no MTP | 21.65 (was 26.24 with buggy OPT) | - | - |
| C1 coding client decode, median of 3 | 32.37 (acceptance ~0.80) | 83.8 (goals doc) | 0.39x |
| C2 aggregate output tok/s, median of 3 | 46.38 (23.7 per stream) | 103.0-104.3 (S87/S100 today) | 0.45x |

Coherence gate (concurrent arithmetic) PASS; all C2 outputs are on-topic
1500-token asyncio answers; cache_n 0 on fresh prompts; post health PASS.
Strata rows are its own same-day/same-harness runs, not a back-to-back arm
in this window.

### E6 llama.cpp decode kernel census (unitrace)

CONFIG: llama-bench -sm layer, no MTP, OPT on (before the bug was found);
unitrace -d; per-token = (tg80 - tg16)/64 so load/warmup cancel.

COMMAND: scripts/e5e6-run.sh (E6 part), scripts/e6_diff.py

RESULT: 35.7 ms device time per token (both cards summed; layer split is
serial) against ~38 ms wall at 26 tok/s, so decode is device-bound, not
host-bound. Families: bf16 GEMV + f32->f16 convert 13.3 ms (37%, 484 GEMV
calls/token), routed MoE mmvq + top-k 9.0 ms (25%), dense quant GEMV 6.6 ms
(18%), elementwise/norm 4.2 ms (12%), flash attention 0.3 ms, GDN 0.3 ms.
The bf16 tensors are the per-layer hyper-connection hc_attn/hc_ffn
up/down (4 x 6.25 MiB per layer), ffn_gate_inp, ssm_alpha/beta, QSA indexer
q/k and inject weights: ~1.4 GB/token read at ~110 GB/s effective.

VERDICT: the llama.cpp gap is device kernels, led by an untuned bf16 GEMV for
this model's hyper-connection tensors (a clear, portable upstream target),
then routed-expert GEMV. Unitrace ran with the buggy OPT path; the family
shares may shift slightly with OPT=0.

### Updated decision

Strata stays the Flash-Next backend. The matched harness puts correct
llama.cpp SYCL at 0.39x C1 and 0.45x C2, and today's llama.cpp SYCL needs a
non-default switch to be correct on this model. The hedge remains
upstreamable work in ggml-sycl: the reorder bug already has a fix
(PR 30604, confirmed here); the open kernel item is a bf16 GEMV for hc
tensors.
