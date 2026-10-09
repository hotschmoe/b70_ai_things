# Actual consumed0013..0017 observer audit

CONFIG -> actual built source at
/mnt/vm_8tb/b70/build/strata-native-hc-engine-20261009T080647Z-6_ja3xd5/source.
Native HC enabled, ordinary serial GEN, no MTP/pipeline/batch observer coverage.
The source audit records consumed verifier, prefill, generation and observer
header hashes. No frozen patch/source tree or collector was modified.

COMMAND -> python3 strata/flash-next/audit_fidelity_observer_coverage.py
--output strata/flash-next/fidelity-observer-consumed-source-audit.json
and python3 strata/flash-next/test_fidelity_observer_coverage_cpu.py.

RESULT -> static source checks and coverage negative controls pass.
Verifier copies248320 raw F32 logits after native head matvec, before the
captured argmax/greedy sampler. Host penalty/stochastic sampling occurs later
in run(). The copy uses the same stage compute queue; the copied snapshot is
independent of later transformations to head_logits_. Runtime execution and
graph dependency correctness still require the paired GPU equivalence gate.

The first generation window consumes ids[n-1] at position n-1 with T=1;
the observer row contract additionally requires last-prompt position and row0.
Each native verifier post(l) completes the FFN F32 HC write before snapshot,
and the next layer pre(l+1) follows it. Native mode materializes pending writes.
Generic non-native fused/pending routes are not certified by this audit.
Prefill R is token-major T*10240 storage as independently consumed by the
native row read/write descriptors. Last-row copy is R+(T-1)*10240 after
half1 write, labeled pos0+chunk_offset+T-1 and its actual GEN token ID.
Thus T>1 captures one complete4x2560 residual, not the first stream of another
row. Stage ranges bound snapshot slots and the layer loops cover [lb,le).

ARM must exist at begin() and output directory must exist. Request ordinal
advances only for armed requests; only first6 supported requests can activate.
Each Prefill::Impl has a separate request ordinal and first2 chunk quota,
across multiple read() calls in that request. Each verifier stage uses its
own snapshot; only its final head stage reserves/copies logits. At most
41349120 raw output bytes are needed for6 requests, two complete prefill
positions plus first-window all48 layers and one head row per request, below
64MiB. Beyond quotas captures are explicitly skipped. With flag off no snapshot
allocations or diagnostic graph copy nodes are added; normal graph selection
remains original. Host branch/traversal overhead is not a runtime timing claim.

VERDICT -> no additional source observer blocker found for the selected native
serial route. The prior collector has a coverage defect: empty or partial traces
can validate, and observations absent in both compared arms are invisible.
NEW audit_fidelity_observer_coverage.py requires positive expected request count,
explicit stage ranges partitioning all48 layers, one correctly placed head row,
and exactly48 first-window layers for activation arms. Present prefill positions
also require all48 layers; absent prefill remains unobserved (short-window or
fully reused prompts can legitimately have none). It rejects missing/duplicated
layers/head, wrong stages, reused files, unknown phases and nonfinite vectors.
Cancelled request ordinals are an explicit unobserved exemption; they are never
claimed as complete activation coverage. Logits-only scope remains explicit.

Example pair activation capture coverage:
python3 strata/flash-next/audit_fidelity_observer_coverage.py --log ENGINE.log
--expected-requests 6 --stage 0:0:32 --stage 1:32:48 --activations --output AUDIT.json
Supply actual engine device IDs/ranges, using spaces between flags and values.
For a cancelled request add --cancelled-request N. Parent V3 harness owns
cancellation classification and actual generated-output off/on comparisons.

Persistent GDN, PLE, QSA/indexer/KV and pending-state internals remain unobserved;
all-token/state fidelity, graph execution, teardown and timing remain GPU gates.
