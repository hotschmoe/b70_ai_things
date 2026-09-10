# Bounded FP16 graph/MTP plans on14ee, not launched

CONFIG -> Read-policy image14ee retains the current-main2f7393f0 backend and
native extensions. With KV auto resolving FP16, the narrow FP8 read-policy
branch is inactive. No quantization-param-path is passed, so the calibrated
FP8 loader is not activated. Fixed GPTQ INT4 target, Triton target/draft
attention, FP32 SSM, prefixOFF, overlap disabled, context8192/chunk512/c4.
Text-only skip-server-warmup avoids deliberately excluded native vision FMHA.
Stable primary model remains hotschmoe-dd; research identity is in manifests.

COMMAND -> feature_plans.json freezes four opt-in plans for card1 with separate
fresh caches/outputs/jobs. feature_gate.py prints by default; --run checks14ee
pair health, its complete nine-case numerical pass, source hashes and prior
26-request viability/lifecycle success before exec of the existing leased
TP1 lifecycle. No GPU action was performed by this preparation.

- graph-only port18231: target decode FULL, prefill disabled, capture sizes1/2/4.
- mtp1-eager port18232: NEXTN steps1, topk1, draft tokens2, graphs disabled.
- graph-mtp1 port18233: combine only after both preceding viability arms pass.
- graph-mtp3 port18234: steps3/draft tokens4, after combined one-step viability.

All jobs use the same26 real temperature0/seed42 text checks, HTTP300s,
job1500s and startup1200s. The existing current-main FP16 eager baseline
finished lifecycle0/job0. Its result is a prerequisite for the first two arms,
not a speed control for these new-image/new-cache experiments.

RESULT -> CPU lifecycle dry-run command generation and argv checks pass for
all four arms. Source review confirms XPU -> EAGLEDraftCudaGraphRunner and
EAGLEDraftExtendCudaGraphRunner mappings. The graph runners use the device
module. NEXTN steps1 skips iterative draft-decode capture (steps>1 guard);
steps3 therefore remains a distinct qualification arm. Draft-extend capture
has an explicit XPU route. Existing source/native identity checks establish
that these graph/worker modules are unchanged in14ee.

Prerequisite unit tests confirm failed numerical gates, changed source and
failed prior lifecycle prevent launch. Actual capture/replay, acceptance,
draft weight loading, memory fit and concurrent coherence remain unmeasured.
Inspect capture/replay and acceptance evidence before promoting a combined
arm; a CLI or generic coherence pass alone is insufficient.

VERDICT -> Concrete, bounded greedy FP16 feature tests prepared for scheduling
only after numeric/health gates. No graph, MTP or performance qualification is
claimed. Prefix and calibrated FP8 remain later separate additions. The pinned
XPU NEXTN verifier still takes argmax for non-greedy requests; these plans do
not qualify Pi temperature0.7 or seeded sampling parity. No sampling overlay
is activated. Native vision support and full feature parity remain unproven.

Raw plans: /mnt/vm_8tb/b70/results/bang_isolation_20260910/sglang14ee-fp16-feature-plans
