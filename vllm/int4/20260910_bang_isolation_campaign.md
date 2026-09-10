# Bang isolation campaign, 2026-09-10

Goal: identify and repair corruption while retaining prefix caching, graphs,
MTP and calibrated FP8 KV. User authorizes endpoint downtime, custom kernels,
custom ops, transformations and alternative vLLM images. TP1 versus TP2 is
now the priority contrast based on the user's unconfirmed TP1 observation.
SGLang with the same features is a parallel research direction, not an
assumed fix. No model output is allowed to execute tools in these probes.

Raw root: /mnt/vm_8tb/b70/results/bang_isolation_20260910/.
Earlier incident analysis: 20260910_live_bang_investigation.md.

## Baseline and shutdown

CONFIG -> R276 INT4, MTP3, calibrated FP8 E4M3 KV, prefix on, full decode
graphs, TP2, 200K/c4. Exact image521eb277..., lifecycle20260909T224703Z.
COMMAND -> Snapshot container, unit, host and endpoint identity/metrics.
SIGTERM verified owning trial launcher10892; leave the parent lease held
through backend teardown and post-health. No system configuration change.
RESULT -> Frontdoor and backend stopped. Lifecycle exit0; both per-card
checks and compiled P2P-off collective post-health pass. Unit Restart=no.
VERDICT -> Endpoint offline for authorized investigation. Original startup
recipe remains recoverable; no replacement promoted.

## SGLang C1: topology recognition alone

CONFIG -> Same adc915d266 image, synthetic2048x5120 FP16 subgroup reductions,
P2P0; change CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK from0 to1 versus C0.
COMMAND -> Existing collective_control.py under collective_lifecycle.py and
run_arm.py; named unit b70-bang-sglang-c1,240-second bound, both GPU leases.
RESULT -> Small4x5120 reduction passes both ranks. Rank0 enters first20MiB
reduction without returning. Rank1 completes that reduction numerically,
then enters the next and stalls. Timeout124. Container removal verified,
both devices rebound, per-card and compiled P2P-off post-health pass.
VERDICT -> Topology recognition alone is insufficient. Same-root-cause as
loaded SGLang is not proved. No full-model unchanged retry. Next candidate
for this collective branch is CCL_ENABLE_SYCL_KERNELS=0, holding C1's other
environment fixed; do not introduce it into the vLLM TP comparison silently.

## CPU/source preparation

CONFIG -> Three affected Pi histories, first corrupted response excluded.
No exact Pi system prompt/tool schemas/upstream payload in original bundle.
COMMAND -> replay_pi_history.py prepare; preserve payload hashes. Validate
recorded tool-result pairings, first-corruption cutoff, prior-thinking
omission and absence of tool execution. Seven lifecycle CPU tests include
single-card lease/device pin, health scope and deferred pair-wide recovery.
RESULT -> All six reconstructed histories have no orphan or pending tool
calls. They use a clearly reconstructed system prompt/minimal tool schemas,
manifest seeds,temperature0.7,medium thinking,max output8192. Successful prior
transcript/tool results are used as context without executing tools.
VERDICT -> Controlled reconstruction, not exact historical wire replay.
Requested actual upstream payload and gateway source from the other team.
Comparison retains actual source/payload hashes and SSE/partial failures.

## TP1 initial capacity control

CONFIG -> Exact same image, weights and scale artifact, MTP3/prefix/graphs/
FP8 KV on,200K/c4/batch32768. Single-card lease0 and matching device pin.
COMMAND -> vllm-tp1-fp8kv-mtp3-prefix-graph plan, fresh compiler cache.
RESULT -> Weight load17.48GiB. Startup rejects200K: needs7.26GiB KV but
4.25GiB available, estimated maximum105600 tokens. No inference occurred.
Per-card post-health passes. Pair-wide recovery was correctly deferred
until the single-card lease ended; separate two-card rebind and per-card/
compiled collective health passed before retry.
VERDICT -> Capacity rejection, not evidence about bangs. New matched TP1
and TP2 plans use100K context, keeping other features and batch budget.
This covers the selected first-failure histories.200K remains a later
qualification gate; a lower prefill batch budget is a possible memory
tradeoff to measure, not a demonstrated repair.

## Reviewed source candidates

Fetched source refs without moving either checkout:
- Steve: steveseguin/b70-optimization-lab origin/main ac2f723e.
- Sergio: SergiioB/intel-arc-pro-b70-inference-cookbook origin/master966c593.

Steve's134322d9 validates a GDN short-prefill call-site correction on his
stack: treat short target prefills as prefills when is_prefilling metadata
exists, preserving legacy behavior for metadata-less capture/draft callers.
Our exact installed R276 call also omits this flag. His actual-source CPU
regression passes8 stock cases and8 in-memory candidate cases in our exact
image, without GPU devices/network. This confirms the classification defect,
not its causal role in Pi. A one-call-site Python-only candidate is prepared
under gdn-phase-candidate-v2; original preparation's ASCII encoding error
is retained under gdn-phase-candidate and made no runtime change. Native
libraries must remain byte-identical before any candidate GPU test.

Sergio'sfb860bb ports three separate hybrid-cache defects: accepted-count
D2H/double permutation, ignored eagle boundary, and backward state copies.
Our installed V1 runner retains the late wait plus prev_positions gather;
input-batch swaps/condense mutate those same counters. Installed Mamba copy
sites also lack a backward-copy guard. These are source leads requiring
actual-path and causal tests. Do not apply all changes at once and claim
which one fixed the symptom.

Sergio'sd9c95e5 disables his draft INT4 patch for TP>1. Our R276 path is a
different implementation: a shallow draft-only ParallelLMHead copy with
private packed buffers and unchanged verifier head. TP>1 drift is still a
reason to test draft-head INT4 off, but his guard is not proof of the same
bug in this implementation.

No kernel, driver, PyTorch or native library has been updated. SGLang source
inspection still shows incomplete FP8 extend support in its intel_xpu
attention backend; Triton is a candidate route, not feature qualification.

CONFIG -> Python-only GDN phase candidate built atop exact R276 base.
COMMAND -> Build one COPY layer; compare five native/routing file hashes;
run eight actual-source cases in a CPU-only network-disabled2CPU/2GiB
container. Verify tracked patches/r276-gdn-prefill-phase.patch reproduces
exact candidate source bytes.
RESULT -> Image0328900cf1f8f29f5a8e76ed21a3eff71ef54d0abf495a88fdb86e93e880b077;
source a646a6750f16b961d0fa8eb6d7e00b46b4ea4e6992b1cf620cb5eb01b02d6b38.
Native hashes equal, eight CPU cases pass. A separate controlled evaluation
of the actual installed accepted-count gather reproduces double permutation:
already-reordered counts[4,1] become[1,4]; proposed direct copy keeps[4,1].
VERDICT -> Candidate prepared, no GPU repair claim. Counter test is not an
actual asynchronous D2H race reproduction. The phase patch is a deliberate
port of Steve's134322d9 finding; no upstream checkout was moved.
