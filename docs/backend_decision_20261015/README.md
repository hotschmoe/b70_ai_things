# Backend decision experiments E1-E4 (2026-10-15)

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
