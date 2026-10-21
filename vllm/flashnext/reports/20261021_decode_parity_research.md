# Strata/vLLM Flash-Next decode parity research, 2026-10-21

Owner: Codex research agent. CPU-only source/evidence review. No serving,
GPU work, build, runtime change, service change, commit or push.

CONFIG -> retained ISTA GSQ-RCO IQ3_XXS on two B70s; fresh matched C4
20K+512 comparison; current Strata night-final-v3 and O4/F1/O5c vLLM.
COMMAND -> read the fresh raw configs/logs, retained reports, current fork,
plugins and Strata source; consult current primary upstream documentation.
RESULT -> most large Strata optimizations are already present in vLLM.
The immediate questions are actual C4 verify/draft cost, attention's
repeated KV reads, and batch-width limits. Native QSA verification remains
a separate missing integration, with substantial correctness work.
VERDICT -> run the bounded shape/critical-path experiments below before
porting more kernels. No defensible evidence yet attributes the entire
55.3 percent steady-TG gap to one subsystem or promises parity.

## What the fresh comparison establishes

| Recipe | Effective cold PP tok/s | Steady C4 TG tok/s | Cold batch wall s |
|---|---:|---:|---:|
| Strata Flash-Next, depth1 | 1739.5 | 188.487 | 52.74 |
| vLLM Flash-Next, MTP1 | 2262.3 | 121.408 | 51.92 |
| vLLM 27B INT4, MTP3 | 722.4 | 161.4 | 122.29 |

The fixture is four 19798-token rendered prompts and 512 greedy outputs
each, thinking off, one excluded warmup and two measured batches. PP is
input divided by time until all four first content arrivals, including
scheduling and overlapping decode. TG counts client token arrivals only
after all four streams begin and before any finishes. Both are useful
serving metrics; neither is an isolated device timer.

Strata's cold burst never has all four streams decoding together. Its TG
number comes from a separate cached replay at the same approximate decode
context lengths; its cold PP/wall numbers remain cold. vLLM's steady TG
comes from the cold runs. This supports a four-way decode-state comparison
but differs in admission, cache history and potentially completion text.
Do not convert 188/121 into a matched kernel speedup or claim token-ID
parity. The output retokenization/count checks are not token-ID equality.

The 51.92 versus 52.74 second cold walls are nearly tied despite the TG gap.
For actual repository work, fixed-length TG is only one part of task time:
tool calls, incremental/cached prompts, reasoning length, retries and task
success can dominate. Preserve cold batch wall, each stream's wall and
TTFT/ITL distributions alongside aggregate TG in future tests.

Evidence: [fresh report](20261021_c4_compare.md:13),
[raw root](/mnt/vm_8tb/b70/results/c4_compare_20261021), and
[comparison JSON](20261021_c4_compare.json).

## Already enabled or ported: avoid repeating this work

The actual vLLM recipe uses TP1/world-size1 plus a card-1 EP sidecar,
maxlen131072, chunk12288, maxseq4, prefix caching off, FULL_DECODE_ONLY,
capture sizes1/2/3/4/8, MTP1 and a 40525-row draft vocabulary. Actual
server-info proves async_scheduling=true, not merely an intended flag.
Strata uses four active slots, INT8 KV, maxctx65536 for this test and batch
MTP depth1. Its --spec4 is a capacity/C1 setting, not three drafts per
stream in the C4 batch. Its logs show eight-row four-slot verify graphs.

| Strata idea | Current vLLM status | Evidence |
|---|---|---|
| Single trunk plus expert-only peer, GPU graph handshake | Present | plugins/b70_ep_xpu/b70_ep_xpu/hooks.py:103; csrc/moe_ep.cpp:494 |
| Resident IQ expert kernels, aligned loads, plan/activation fusion | Present; historical small-T component comparison already favors vLLM | reports/2O3_moe_decode.md; not a loaded C4/T8 proof |
| KREG multi-column reuse | Present in pinned O5c native library | reports/2O5_mtp.md:22; O5 source csrc/kreg.hpp:503 |
| Q8 drafter dense GEMV | Present, B70_KREG_Q8=1 | O5 source csrc/kernels.cpp:290 |
| Draft subset head / batched head | Present; get_top_tokens accepts a batch | fork xpu/mtp.py:362 |
| Shared expert inside EP decode graph | Present, fused into own-expert call | plugins/b70_ep_xpu/b70_ep_xpu/method.py:235 |
| Router/top-k/dispatch fusion | Present, B70_EP_ROUTE=1 | method.py:53; csrc/moe_ep.cpp:560 |
| Q8 HC down/activation and up/collapse fusion | Present, B70_DECODE_HC_Q8F=1 | fork xpu/hc_decode.py:183 |
| Device PLE history/gather and fused small-T tail | Present | fork gguf/ple_device.py:330; reports/2O4_flashnext_decode.md |
| Async CPU scheduling | Already true | raw vLLM server-info.json:219 |
| Native sparse QSA + INT8 KV + batched speculative verification | Component/non-MTP integration only; absent from fresh fast recipe | next section |

Raw evidence paths are the flashnext-prepared/{recipe,command}.json and
vllm-qwen38-flashnext-ista-iq3xxs-mtp1-r1/server-info.json directories
under the raw root. Strata config/engine.log are in its warm-r1 directory.
Read source is Strata HEAD e753e9cb and vLLM HEAD0a6170da8c plus existing
dirty/untracked files; HEAD alone does not pin either effective runtime.
The fresh report pins actual libraries. Preserve those dirty files.

The existing route+Q8F bundle's earlier 3.28 percent no-spec canonical
gain is already in the fresh C4 recipe. It cannot be counted again as a
new improvement. Likewise, "port pair head", "enable graphs", or "enable
async scheduling" is not an unclaimed win.

## The eight-row fast-path boundary is central

Let C be active streams and k draft tokens per stream. Uniform target
verification has T=C*(1+k) rows. Today the fast decode stack is specialized
for T<=8:

- fork gguf/linear.py:241 limits gemv_out to eight rows; :288 limits BF16
  GEMV to eight. Larger shapes take different dense implementations.
- fork xpu/model.py:148 defaults HC maxT to8; xpu/hc_decode.py:182 and
  gguf/ple_device.py:330 limit Q8 HC and fused PLE to eight.
- EP method.py:75 limits route fusion to8; :229 sends T>8 to the prefill
  sidecar path; :235 limits shared-expert fusion to8.

Thus C4/MTP1 is exactly T8. C4/MTP2 is T12 and changes the kernel family,
not just speculation depth. KREG alone also chunks M>8 into separate
eight-column calls (O5 csrc/kernels.cpp:295). A blind k2/k3 C4 sweep can
misdiagnose the resulting fallback as an intrinsically poor drafter.

The supported dynamic schedule [1,1,3], [2,2,2], [3,4,1] keeps target
widths4/6/8/8. Add capture6 explicitly; the fresh recipe has no6 capture.
Dynamic depth can improve declining-concurrency tails and variable-load
operation. It does not increase k or change width during steady C4.
The older measured d32v schedule was [1,1,3],[2,4,2]; it measured only C1/C2
and should not be copied unchanged to C4 because that selects T12 there.
Historical C1/C2 dynamic timings were9.28/12.90ms per output token, on
shorter prompts and a different configuration; no additive gain follows.

The local V2 graph manager supports per-tier decode shapes
(vllm/v1/worker/gpu/cudagraph_utils.py:273); config/vllm.py:1064 and:1086
check dynamic-SD runner/method and DP compatibility. This is DP1, so the
DP>1 prohibition does not apply. Upstream's current
[dynamic-SD documentation](https://docs.vllm.ai/en/latest/features/speculative_decoding/dynamic_speculative_decoding/)
describes the batch-size schedule and V2 FULL-graph constraint. Its tested
methods list is narrower than this fork's local MTP evidence; the local
measurement is the evidence for this MTP port, not a general upstream
support guarantee.

## Ranked missing pieces and hypotheses

1. **Measure C4's actual verify/draft critical path before implementing.**
   The raw serve log records draft acceptance in ten-second intervals,
   mixed with prefill, coherence and warmup. Rates around76-84 percent in
   measured-generation intervals do not reveal exact four-way acceptance,
   step cost or target/draft overlap. Strata warm logs report whole-batch
   windows around35.5ms including commit/emit, but these include changing
   active widths/admission; they are not matched vLLM target-graph timers.
   Older O5 events showed verify T6 at26.5ms, propose about3.5ms and host
   gap about1ms, at a different short-context recipe. Reuse its event
   observer pattern, not those numbers. Cost S: instrumentation and a
   bounded C4 test, no native rewrite. Gate: observer-on/off perturbation
   check; no per-operation host sync in timed serving; report target,
   draft, sampling/commit, CPU gap and both-card readiness separately.

2. **Attention reuse across a stream's verify rows is a concrete gap.**
   The pinned O5 FA overlay, flash_attn_interface.py:26-47 and:66-143,
   flattens each (request,query-position) into a separate one-token
   pseudo-sequence. It repeats block-table rows and scans the same KV
   range per query position, adjusted for causality. C4/MTP1 therefore
   becomes eight one-query rows over four approximately20K histories.
   This fixes the slow small-Q prefill path and gets split-K occupancy,
   but gives up shared KV reads across each width2 query group.
   Strata verify.cpp:1372-1422 groups rows per slot and runs multi-pool
   batched attention. A width2, split-K dense kernel that reuses loaded KV
   tiles is a specific vLLM-side candidate without adopting sparse QSA.
   It matters only if attention/readiness is material on the T8 critical
   path; no such C4 profile exists yet. Cost S for split-count experiment,
   M/L for new XPU kernel/backend; prefer OOT op. Preserve per-row causal
   bounds, FP32 accumulator order, fixed split decomposition, ragged/
   padded rows and MTP-on/off parity. No bandwidth-only speed claim.

3. **HC normalization/down fusion is only partly ported.**
   Strata fused_gr_night.inc:655-740,905-968 performs combined residual
   application, normalization and down read, then up/collapse in two
   kernels. vLLM hc_decode.py:214 combine_mix has a separate combine/norm
   Triton kernel followed by down_act and up_collapse. Q8F already handles
   the latter two; the remaining opportunity is the producer/normalizer
   boundary, not another copy of Q8 HC. Cost M OOT kernel with thin hook.
   Layout/rounding differs (vLLM materializes BF16 intermediates), so a
   literal Strata transplant is not bit-exact by construction. The recent
   M1 subgroup HC candidate failed strict signed-zero comparison and was
   never timed (3D1_decode.md:689). Do not describe it as a ready speedup.
   Profile M8 first; old M1 trace budgets do not rank today's C4 costs.

4. **Tune routed traffic if the peer is actually late.**
   Fresh Strata places186:326 experts/card; vLLM138:374. Placement uses a
   ranked snake to balance traffic, not equal expert counts. Card0 also
   runs trunk/shared/draft. Expert count and free VRAM do not prove compute
   imbalance. Existing B70_EP_SHARE0 changes hot traffic at fixed quota
   (placement.py:85-126,:174); no new scheduler is needed. Card1 prefill
   was once critical, which does not establish a C4 decode bottleneck.
   Capture per-layer active expert entries/rows, quant mix, peer start/
   finish, card0 own/shared finish and peer-wait exposed time. Only if
   those show peer lateness should a fixed-quota traffic-share treatment
   follow. Cost S config, with a fresh model load. Risks: different
   accumulation order, route ulps, model text, weight residency. Lowering
   q0 to100 already OOMed historically; increasing q0 to150 did not help
   an earlier short no-spec control. Neither is a reason for a blind
   quota sweep.

5. **Scheduling helps mixed work and task time, not pure C4 compute.**
   Chunk12288, long_prefill_threshold0 and prefill_schedule_interval1
   allow long prefill work between decode updates. The saved QSA C2 32K
   trace has13.6s/6.16s early arrival gaps while another request awaits
   first content; exact overlap/causality remains unproved. Smaller
   chunks or a nonzero long-prefill cap are existing configuration
   controls that can bound interference, potentially sacrificing PP.
   The upstream [chunked-prefill tuning guide](https://docs.vllm.ai/en/v0.22.1/configuration/optimization/)
   explains the decode/TTFT tradeoff. Local scheduler.py contains the
   applicable controls, so no custom admission machinery is warranted.
   Keep this out of a claimed steady-TG kernel gain. Stream_interval1
   could be increased only as a separately labeled client-overhead test;
   changed delivery batching can otherwise distort arrival-based TG.

## Native QSA/INT8 KV: fidelity and 200K track, not a cheap C4 flag

The fresh recipe explicitly sets B70_QSA=0 and uses dense BF16 FA. Sparse
selection differs after roughly2051 tokens. Current QSA R2 selector is a
real advance over its native-QSA predecessor, but its39/59 percent C1
median-ITL reduction at16K/32K compares two QSA implementations. It is not
a measured advantage over today's dense-FA C4/MTP1 recipe.

There is a concrete integration barrier: QSA kv_spec.py:106 declares
UNIFORM_SINGLE_TOKEN_DECODE. model_hook.py:637 dispatches max_query_len>1
to _prefill; :601 rejects captured pooled prefill. MTP1 verify has width2.
The non-pooled draft runner also requires caller-supplied selection IDs
(:510). A native-QSA MTP path needs multi-row per-slot transactional KV/
index append, rejection/commit, draft selection/state alignment and graph
support. It is not enough to make T8 buffers larger or toggle B70_QSA.

The existing Q1 serving pair remains strictly unqualified: repeat and
Strata-text gates fail, short decode regresses, and long C2 tails persist.
Dense-FA references also fail some long Strata/repeat gates, so those
failures cannot uniquely be blamed on QSA. N1 deterministic projection
is not a qualified fix; its model attempts had liveness/startup failures.
Isolate the first differing operation on fixed saved inputs before
promoting either attention path. See3Q1_qsa.md:905-1015 and3N1_determinism.md.

Similarly, RAM parking components and Ornith restore evidence must not be
substituted for current Flash-Next200K full-state recovery qualification.
PLE token/conv history, GDN, target attention, indexer and draft state all
belong to the saved conversation. reports/2O2_kv_parking_spec.md:321-443
records these obligations. The fresh131072 maxlen/prefix-off recipe did
not test200K or RAM restore. This is an evidence gap, not a claim that
the architecture cannot support4x200K with two histories parked.

## Next three bounded experiments

1. **Establish the C4 MTP multiplier and budget, T4 versus T8.** Keep the
   frozen fresh vLLM recipe, same fixture, maxlen131072/chunk12288/maxseq4,
   Q0=138, QSA0, route/Q8F/O5c, fixed16 FA splits, prefix-off and same
   capture list. Control disables speculation; treatment is the existing
   k1 config. Retain B70_GDN_SPEC_DECODE=1 in both. One warmup plus two
   measured batches per arm; no full128K sweep. Collect exact four-way
   acceptance and targetT8/draftT4 events in a separate short diagnostic
   slice, both-card stage readiness/traffic and graph keys, then measure
   normal arrival TG/PP/walls without per-op instrumentation. This tests
   whether k1's output multiplier exceeds its added target/draft cost.

2. **If attention is material: fixed FA splits16 versus8, same C4/k1.**
   Keep every other recipe field, library and fixture fixed. First reuse
   the existing FA graph gate with four histories19798..20310, width2,
   padded/inactive rows and causal per-position lengths; compare each
   split setting's verify rows against its own sequential rows. A smaller
   fixed split count reduces partial/reduction overhead but may reduce
   occupancy; the direction is unmeasured. Cross-setting exact bits need
   not match because reduction changes, so preserve observed token/
   coherence changes and no strict promotion on failure. Only a passing
   component gate earns one full treatment arm. The prior short-context
   fixed8 trial did not fix the canonical latency slope; do not rerun it
   as that old hypothesis. If attention is minor, skip this experiment
   and direct the same bounded slot at the measured HC or peer critical
   path instead of paying another blind model load.

3. **Optimize tails using upstream dynamic depth, without crossing T8.**
   Compare statick1 versus maximumk3 with tiers[[1,1,3],[2,2,2],[3,4,1]],
   capture sizes[1,2,3,4,6,8] in both arms, same maxseq4 and all other
   fresh settings. Verify live config and actual target4/6/8-row/draft
   graph use atC1/C2/C3/C4, including transitions and rejected drafts.
   Replay the fixed four-request burst and one staged-completion workload
   to expose tails. Report steady C4 separately from whole-batch finish
   time: this treatment should not be credited with a different steady
   C4 kernel. Historical dynamic C1/C2 evidence makes this a low-source-
   cost candidate, not proof at20K or at changing concurrency.

All are proposed, not executed. Reuse disk caches, prepared fixtures and
existing component gates. The original full vLLM arm including load/JIT/
health cost522s; setup can exceed measurement time. Budget fresh loads
deliberately, do not repeatedly restore Strata, and avoid redundant long
serves or correctness sweeps once unchanged gates pass. GPU scheduling,
bin/gpu-run leases, quiet CPU windows, live model identity, coherence,
normal teardown, per-card/compiled-collective health and source/SO freeze
remain required. Baseline nondeterminism is an explicit diagnostic limit;
successful finite throughput samples do not establish strict promotion.

## Source map

- [Strata inventory](03_strata_inventory.md), [O5 MTP](2O5_mtp.md),
  [O4 decode](2O4_flashnext_decode.md), [O3 MoE](2O3_moe_decode.md),
  [D1 current decode](3D1_decode.md), [Q1 QSA](3Q1_qsa.md),
  [N1 determinism](3N1_determinism.md), [P1 prefill](3P1_prefill.md).
- [Strata verify source](/mnt/vm_8tb/b70/build/strata-ab/src-ep/sycl/src/core/verify.cpp:1372),
  [Strata HC](/mnt/vm_8tb/b70/build/strata-ab/src-ep/sycl/src/kernels/cuda/fused_gr_night.inc:655),
  [Strata pair head](/mnt/vm_8tb/b70/build/strata-ab/src-ep/sycl/src/core/mtp.cpp:2139).
- [Pinned O5 FA overlay](/mnt/vm_8tb/b70/build/vllm-flashnext/o5/fa-overlay/vllm_xpu_kernels/flash_attn_interface.py:26),
  [vLLM linear](/mnt/vm_8tb/b70/build/vllm-flashnext/vllm/vllm/models/qwen4_exp/gguf/linear.py:241),
  [vLLM HC](/mnt/vm_8tb/b70/build/vllm-flashnext/vllm/vllm/models/qwen4_exp/xpu/hc_decode.py:182),
  [vLLM MTP head](/mnt/vm_8tb/b70/build/vllm-flashnext/vllm/vllm/models/qwen4_exp/xpu/mtp.py:362).
- [EP method](../plugins/b70_ep_xpu/b70_ep_xpu/method.py:53),
  [EP placement](../plugins/b70_ep_xpu/b70_ep_xpu/placement.py:85),
  [QSA graph support](../plugins/b70_qsa_xpu/b70_qsa_xpu/kv_spec.py:106),
  [QSA dispatch](../plugins/b70_qsa_xpu/b70_qsa_xpu/model_hook.py:637).

External documentation was read on2026-10-21; local source and runtime
evidence control applicability. No responding Claude model ID was
obtained by this research agent, and none is attributed. The preceding
matched campaign recorded two requested Claude consultations timing out.
