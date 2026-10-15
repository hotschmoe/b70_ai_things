# Backend investment review: two B70s, C1 through C4

Date: 2026-10-15. Decision review, not a server migration or qualification.
Current runnable Flash-Next control remains Strata. Existing user changes
and backend stacks are preserved.

## Workload contract

The user's platform requirements are strong prefill and decode, usually one
active stream and at most four. Four active streams must actually make
progress; two decoding and two parked is a different workload. The existing
numerical targets remain 2000+ prefill tok/s, 150+ decode tok/s per stream
and two 200k contexts. No four-by-200k capacity claim is implied.

For each selected model, require a useful dual-GPU execution mode (TP, PP,
EP or independent replicas as appropriate), prefix caching, true RAM
eviction/recovery without replaying retained history, speculative decoding
such as MTP or DFlash, and correlated host/device data-movement profiling.
TP, PP and DP are alternatives to evaluate, not a promise that every
architecture supports every combination. Quantization and XPU support can
be contributed; their current absence is engineering scope, not exclusion.

User-specified hardware is Threadripper 1950X with PCIe 3.0 x16 GPU links
and peer communication. NUMA placement, small-message synchronization and
host launch overhead belong in every measurement. A generic multi-GPU or
Intel-support checkbox does not establish performance on these links.

The RAM contract includes all mutable state: attention KV, GDN/Mamba
recurrent and convolution state, QSA indexer state, PLE history, token and
position metadata, and speculative commit/rollback state. A cache-hit count
alone cannot establish correct restoration. Cached-prefix 2s TTFT remains
a diagnostic reference, not a hard target.

## Recommendation

Prioritize a bounded SGLang XPU contribution spike for the long-term C1-C4
serving platform. Keep Strata development moving on correctness and useful
measured optimizations while evaluating that choice. Upstream llama.cpp
SYCL is the strongest challenger if SGLang's remaining state, low-bit or
dual-GPU seams require a broad private fork.

This recommends where to spend the next research effort, not which engine
has already won a throughput comparison. No challenger has qualified this
complete Flash-Next workload on this host. Strata remains the only locally
measured full-length RAM-recovery path. Do not migrate the daily endpoint
or change AGENTS.md's active backend solely from this review.

SGLang offers scheduler, radix/hybrid cache and speculation infrastructure
that fits the requested serving problem, plus an existing Intel contribution
path. Flash-Next XPU functionality was merged in
[PR37213](https://github.com/sgl-project/sglang/pull/37213), with performance
work explicitly left for later. That reduced-model evidence is not a
full-size IQ3, TP2 or RAM-recovery qualification. The exact low-bit path
and complete-state offload are the first things to prove, before a full
model port or broad tuning campaign.

The strongest objection is practical: native C++ in llama.cpp already has
the exact GGUF formats and Flash-Next graph. Python host work and generic
collectives on a 1950X could consume SGLang's advantage at C1. A thin
upstream llama.cpp kernel/state fix may deliver more useful progress than
several new SGLang subsystems. The spike must be allowed to reject SGLang.

## Independent agent findings, recommendations and reasoning

These are summaries of the agents' final recommendations under the expanded
platform contract. Both were consulted through the actual authenticated
Claude CLI, read primary upstream sources, and were instructed not to read
one another's reports. They disagree on investment order. Neither report
qualifies an untested backend.

| Reviewer | First investment | Challenger | Strata role |
|---|---|---|---|
| Claude Opus5.5 (claude-opus-5-5) | SGLang XPU contribution spike | llama.cpp SYCL | Working control and correctness fixes |
| Claude Fable5.1 (claude-fable-5-1) | llama.cpp SYCL qualification/contributions | SGLang XPU contribution spike | Working control and EP investigation |
| Root/Codex | SGLang state/low-bit/TP feasibility first, conditional | llama.cpp if that scope is broad | Continue useful authorized Strata work during discussion |

### Claude Opus5.5

Findings: SGLang already has a serving scheduler, radix cache, hybrid-cache
modules, speculation and Intel graph infrastructure. Flash-Next XPU
functionality is merged. It does not yet provide proven execution of our
IQ3 artifact or complete Flash-Next hybrid-state RAM recovery. Its generic
XPU QSA path, low-bit MoE support, graph coverage and actual collective cost
are the important seams. XPU two-batch overlap remains unsupported.

Recommendation: prioritize SGLang as the long-term serving contribution
home, subject to one bounded layer/state spike. Keep llama.cpp as the first
challenger and Strata as the working Flash-Next control. Opus initially
favored the smallest upstream unit of work in llama.cpp; the C4 active,
prefill-fairness and complete-state platform requirements changed its
ranking toward SGLang.

Reasoning: owning the GPU kernels is acceptable, but rewriting scheduler,
cache ownership and speculative state handling together may consume more
time than contributing to a serving engine with those abstractions.
Existing Intel enablement gives a plausible upstream integration path.
This is an architectural judgment, not evidence that all those features
compose on B70 today.

Strongest objection in Opus's review: a Python scheduler and generic
collectives on Zen1/PCIe3 may lose at the usual C1 workload; the missing
low-bit, fusion and full-state paths could turn into several substantial
projects. If a measured layer/state spike shows those costs are diffuse
or cannot be fixed within upstream interfaces, prefer llama.cpp.

Proposed discriminator: real QSA and GDN layers including HC/PLE/MoE and
TP2 transfer; host eviction/restoration and speculative rollback; measure
C1/C2/C4 and prefill interference. Prove the low-bit kernel seam separately
from a BF16 plumbing test. Root rejects using whole-window time divided
by48 as a matched per-layer performance threshold, and rejects treating
the current vLLM XPU guard alone as a reason not to contribute there.

Latest full report: backend-investment-20261015/claude-opus-5-5/
update-platform.md in the raw root given below; earlier decisions are
superseded there.

### Claude Fable5.1

Findings: llama.cpp has the exact GGUF architecture and relevant SYCL
quant/operator paths upstream, native C++ host code, and both layer split
and source-allowed tensor split for qwen4exp. Its SYCL source contains a
two-card all-reduce implementation. Actual C4 drafting independence,
TP/MTP combinations, hybrid RAM restoration and throughput on this host
remain unqualified. The open hybrid slot-file checkpoint issue is a
concrete state-management contribution opportunity, with in-RAM impact
still unknown.

Recommendation: prioritize llama.cpp SYCL as the long-term investment;
retain SGLang as the platform challenger. Keep Strata for the measured
serving control and publication investigation. Fable retained this order
after receiving the same C1-C4, custom-quant and dual-GPU requirements.

Reasoning: this workload usually runs at C1 and never aims beyond C4.
Exact low-bit kernels and native dual-card execution are already present
in llama.cpp source, so work can concentrate on measured kernel, transfer
and state gaps. Wider GGUF model coverage and an active SYCL contribution
path may make those changes more reusable than growing Strata's overlay.
Fable considers server/state improvements a more bounded starting point
than adding low-bit kernels and hybrid-host offload to a larger runtime.
That scope judgment is still an estimate.

Strongest objection in Fable's review: Strata already passes the central
RAM-recovery oracle while llama.cpp does not have a current local
qualification. Generic routed-expert execution may lose to Strata EP;
server fairness and independent drafting at C4 are unknown. A familiar
format and upstream code do not establish a fast or correct runtime.

Proposed discriminator: qualify a pinned native SYCL candidate's layer
and tensor modes, compare C1/C4 decode plus prefill and speculation, then
prove forced RAM recovery and rollback. Advance only for competitive
critical-path behavior or one bounded upstreamable bottleneck. Root does
not adopt arbitrary patch-line limits, promised review deadlines, exact
CPU/GPU token-hash equality or a full-context run before a cheap state
oracle. Fable's corrected AOT target is BMG-G31.

Latest full report: backend-investment-20261015/claude-fable-5-1/
decision-update.md in the raw root given below.

### Root/Codex synthesis and unresolved disagreement

The disagreement is about the cheapest sustainable route to the same
platform: improve a native GGUF engine's serving/state behavior, or add
low-bit and exact-state XPU support to a serving engine. Both routes
require real two-card measurements. Source feature counts and guesses
about Python versus C++ are insufficient.

Root favors first proving SGLang's complete-state and low-bit integration
scope, because a wrong cache/scheduler foundation is costly to discover
after kernel tuning. This is deliberately a small feasibility investment,
not authorization for a wholesale migration. Fable's native-engine case
is strong enough that a broad SGLang spike should hand off to llama.cpp
rather than become a private backend fork. Continue useful Strata work in
parallel as requested; its result remains the serving reference.

## All six options

| Engine | Useful foundation | Dual-B70 and remaining work | Investment role |
|---|---|---|---|
| Strata | Exact IQ3_XXS model, native SYCL kernels, measured MTP and full-state RAM recovery | Local EP/PUSH overlay; C2 measured, C4 active not qualified; publication oracle unresolved; Intel changes span several open PRs | Working Flash-Next control and ongoing correctness work |
| SGLang | Serving scheduler, radix/hybrid cache, speculation, XPU infrastructure and merged Flash-Next enablement | TP documented; exact low-bit loader/kernels and complete QSA/PLE/GDN RAM recovery need proof; XPU two-batch overlap unsupported | Primary platform contribution candidate, conditional on bounded spike |
| llama.cpp | Exact qwen4exp GGUF graph and SYCL quant kernels, native host, model breadth | Layer and tensor split in SYCL source; qwen4exp source-allowed for TP; real TP/MTP/C4/state performance unqualified | First challenger, especially for C1 and new GGUF models |
| vLLM | Serving ecosystem, scheduler, speculation and XPU kernels; local 27B fallback | Current qwen4exp dispatch rejects XPU; exact model/low-bit/state port required, plus topology/collective qualification | Viable contribution alternative if a concrete port is smaller or better supported than SGLang's |
| ZML | Zig/MLIR, documented Intel oneAPI platform | oneAPI currently documents one GPU and no sharding; exact Flash-Next graph/quant/state serving would be additional work | Compiler/runtime research option, not the first serving investment |
| Own Rust/Zig | Direct control of host scheduling, ownership and profiling | Reuse SYCL kernels through a C ABI; dual-GPU protocol, model state, speculation and cache ownership still need implementation | Reserve for a measured limitation existing engines cannot accommodate |

Evidence for this table is source-level unless explicitly described as a
local measurement. Missing support can be added. The decision is about
the amount and reuse of that work, not about forbidding new ports.

## Source checks that affect the decision

Pinned upstream heads inspected through GitHub on 2026-10-15:

| Project | Commit |
|---|---|
| Strata | 296b59d2f73c8f04a805df29c2fa97655adace78 |
| SGLang | 0cf683ea8d27041da981f7875e1755b9f7772799 |
| llama.cpp | a0e6dc7384edb8f5d927155e1d26c10b4a310e6d |
| vLLM | f46021a7faa62015016f2862d02f2a64c24bb960 |
| ZML | cddc9cf60743e3c4e7e0d5400d5ab6df909630df |

SGLang's current [QSA indexer](https://github.com/sgl-project/sglang/blob/0cf683ea8d27041da981f7875e1755b9f7772799/python/sglang/srt/layers/attention/qsa/qsa_indexer.py#L762)
has forward_xpu and gates CUDA fused preparation on tensor.is_cuda. A
CUDA-only optional helper does not mean the whole indexer needs porting.
Its [GGUF dispatch](https://github.com/sgl-project/sglang/blob/0cf683ea8d27041da981f7875e1755b9f7772799/python/sglang/srt/layers/quantization/gguf.py)
has no XPU kernel import branch. The [XPU guide](https://github.com/sgl-project/sglang/blob/0cf683ea8d27041da981f7875e1755b9f7772799/docs/docs/hardware-platforms/xpu.mdx)
documents speculative decoding and graph options, but not XPU two-batch
overlap; EP is unvalidated on its W4A16 MoE path. Neither native MTP nor
DFlash support elsewhere proves this exact Intel/model combination.

llama.cpp's [SYCL guide](https://github.com/ggml-org/llama.cpp/blob/a0e6dc7384edb8f5d927155e1d26c10b4a310e6d/docs/backend/SYCL.md)
documents tensor split optimized for two GPUs and requiring flash attention.
The [architecture gate](https://github.com/ggml-org/llama.cpp/blob/a0e6dc7384edb8f5d927155e1d26c10b4a310e6d/src/llama-arch.cpp#L1230)
allows qwen4exp but excludes Nemotron-H variants. Source eligibility is
not proof that the exact quantization, graph and shapes run correctly.
[Issue28194](https://github.com/ggml-org/llama.cpp/issues/28194) reports
hybrid slot-file restoration losing checkpoint reuse on two B70s. That
disk-slot issue is not proof that the in-memory RAM cache has the same bug;
test the actual path. An attached community patch is not a qualified fix.

vLLM's [current dispatch](https://github.com/vllm-project/vllm/blob/f46021a7faa62015016f2862d02f2a64c24bb960/vllm/models/qwen4_exp/__init__.py#L27)
explicitly rejects XPU for qwen4exp. Removing the guard alone is not a port. An open Intel
[PR61607](https://github.com/vllm-project/vllm/pull/61607) adds an XPU
sibling for a W4A16 checkpoint, with eager TP4 validation and no QSA/indexer
path. This is a real contribution base worth watching, not a qualification
of our indexed-attention IQ3 model. SGLang also has an open
[HiCache XPU effort, PR32503](https://github.com/sgl-project/sglang/pull/32503);
inspect its scope before duplicating RAM-tier work. Its current head
5fc2f401 explicitly refuses Mamba host pools on XPU because their transfer
kernels are CUDA JIT kernels. Thus generic HiCache support is insufficient
for the required hybrid-state recovery: this is known port work, not just
a missing benchmark.
ZML's [oneAPI guide](https://github.com/zml/zml/blob/cddc9cf60743e3c4e7e0d5400d5ab6df909630df/platforms/oneapi/README.md)
limits that platform to one GPU, without sharding.

## The bounded contribution spike

Start with a source/shape/ownership inventory in an isolated current
SGLang tree. Reuse supported XPU paths rather than rewriting them. Track
each missing operation, quant layout and mutable-state field to its owning
module. Inspect existing Intel work before duplicating it. No driver or
framework upgrade is required merely to prepare this inventory.

Exercise one real QSA layer and one GDN layer, including HC, PLE, router,
MoE and the actual two-rank transfer. A BF16 dequantization of those layers
can test scheduler/state/collective plumbing; it does not demonstrate the
whole low-bit model fits or performs well. Follow it with a representative
real IQ3 expert tile using the intended reusable SYCL kernel seam. Avoid
reimplementing every GGUF quant type before testing that seam.

For those layers, compare uninterrupted continuation with RAM eviction and
restoration, then speculative rejection/rollback. Verify state/logit
agreement under a declared numerical tolerance and no retained-history
recompute. Keep prefix-only KV handling separate from full hybrid state.

Measure the same shapes at C1, C2 and C4, with speculation both off and on,
plus one incoming prefill. Report per-stream decode, aggregate throughput,
acceptance, time to first token, p95/max token stalls and fairness. Join
host enqueue/copy/completion spans with per-device events; count transfers
and collectives and attribute GPU wait time. Use actual layer timings, not
an average derived by dividing a whole MTP window by 48 layers. Do not
interpret a BF16 plumbing result as an IQ3 performance comparison.

Continue if correctness holds and the measured critical path is competitive
or exposes one bounded, reusable fix. Reconsider if the solution needs a
private scheduler fork, scattered unsupported kernels/state paths, or
unexplained state divergence. A numerical failure pauses optimization;
it does not ban contributing a focused correctness fix. No arbitrary
unsupported schedule or percentage gain is promised.

Only a useful candidate advances to whole-model matched serving, C4 active
progress and prefill interference, then small forced-RAM exchange, then the
existing full-context qualification. The established C2x200k goal and
C4's measured context capacity must be reported separately. No repeated
full-context tests, six-engine bakeoff or routine daily-server restoration.

If SGLang's spike fails, apply the same questions to one pinned llama.cpp
SYCL tensor-split candidate. Do not assume layer split is its only mode.
Keep all numerical probes, quant kernels, NUMA findings and client harnesses
usable across either integration.

## Nemotron and a custom host runtime

NVIDIA's [announcement](https://blogs.nvidia.com/blog/local-ai-rtx-spark-laptops-software-updates/)
names Nemotron3.5Bolt74B-A9B24GB, 2-bit quantization with QAT, and October15
availability, listing llama.cpp and vLLM among launch runtimes. Exact
released files, packing, architecture and B70 kernels remain unverified.
GGUF 2-bit, NVFP4 and packed INT2 are different implementation contracts.

If weights plus full target-context state and activations fit one B70,
independent replicas per card deserve an early comparison. A 24GB model
advertisement alone does not prove that fit. Replicas can avoid cross-card
decode synchronization; Flash-Next's current all-resident artifact requires
both cards and does not offer that same option.

Rust or Zig is a reasonable host-runtime choice if traces establish an
unfixable scheduling/ownership bottleneck in existing engines. It can
reuse an existing loader and raw-pointer SYCL kernels through a narrow
C ABI. Changing the host language alone does not reduce GEMM, dequant,
attention or peer-wait service time. Building a compiler and replacing all
kernels is a separate investment, not a prerequisite for such a runtime.

## Local evidence and review provenance

S20 proves four bounded full-length RAM returns at depth1, each reusing
199839 retained tokens with31 new and zero retained replay. S21 separately
measures depth2 concurrent coding at98.901 aggregate tok/s and52.737 median
per stream. Depth2 full-cache and C4 active serving remain unqualified.
S26 finds standalone EP publication freshness failures without proving
production output corruption. Failed-arm timings are not speed evidence.
See [measured goals](20261013_flashnext_goals.md) and
[stage evidence](20261014_flashnext_strata_optimization.md).

Actual independent reviewers are claude-opus-5-5 and claude-fable-5-1,
verified from CLI response metadata. Root rechecked important source claims
and retained corrections, rather than treating agent conclusions as proof.
In particular, the paused UD-Q4_K_XL campaign was patched Strata, not a
failed vLLM port; CPU/GPU greedy hashes are not a universal cross-kernel
correctness contract; B70 AOT here is BMG-G31.

Raw source snapshots/hashes, consultation streams, original and updated
reviews, and root dispositions live under
/mnt/vm_8tb/b70/results/backend-investment-20261015/.

## Follow-up experiments (2026-10-15)

E1-E4 ran after this review; see
[backend_decision_20261015/README.md](backend_decision_20261015/README.md).
Summary: llama.cpp restores full qwen4exp hybrid state from RAM without
replay but runs this model at about 0.44x Strata C1 decode on SYCL;
captured xccl all-reduce costs about 12.6 us per sync, so TP is not ruled
out by synchronization; SGLang head cannot capture Qwen4Exp decode on XPU
and its eager step is about 33 ms for 4 layers. Strata keeps backend-locked
Flash-Next work; portable kernel work should be upstreamable to ggml-sycl.
