# Strata clean C2 protocol audit, 2026-10-15

S67 supports a cheaper future matched serving protocol using existing admission
logging. It does not explain the S29/S55 throughput difference or justify an
S55 retry. Native audit author: `/root/strata_ep_s27`; root read and accepted
the source findings. No Claude review or new measurement was performed.

CONFIG -> Current SYCL source, frozen S29/S51/S55 configs and harnesses.
S55's 256-output pair took about 4.77 s; retained S29 used 3000 outputs and
about 102 aggregate tok/s. These are different protocols and histories.

COMMAND -> Read flag consumers, timestamp helpers, S51 admission/comparison
logic, S29 concurrency client and one representative saved row per stage.
No GPU, build, source change, HTTP, benchmark or whole-log scan.

RESULT -> Neither `STRATA_TRACE` nor `STRATA_HOST_BOUNDARY` adds GPU profiling
operations. S55 configs contain both, but contain neither GPU profiling flag.
S29 configs contain `STRATA_PREFILL_TIMING=1`.

| Flag | Source behavior | Clean future setting |
| --- | --- | --- |
| STRATA_TRACE | Presence enables CPU fprintf/fflush, including prompt-start. No GPU events/barriers. | Keep 1 in both arms for admission. |
| STRATA_HOST_BOUNDARY | First character 1 enables steady-clock timestamps, bounded 8192-record CPU vector and idle-batch dump. | Omit. |
| STRATA_VERIFY_PROFILE | Presence enables device stamp tasks, copy/wait/accounting; also disables some shared-expert fork branches. | Must be absent. |
| STRATA_PREFILL_TIMING | Presence enables primary/peer phase events and dpct sync barriers. | Must be absent. |
| STRATA_PREFILL_SYNC | Presence waits/logs at each prefill mark, even without phase timing. | Must be absent. |
| STRATA_PREFILL_HOST_BOUNDARY | First character 1 enables separate CPU prefill markers. | Omit. |

For presence-controlled flags, values `0` and empty still enable the feature.
Inspect the effective native child environment, including inherited variables;
also omit VERIFY_TRACE, VERIFY_EAGER, unitrace/preload and other debug overrides.
No normal queue/backend change is proposed.

The inspected S55 repeat1/slot0 and S29 repeat1/slot0 both have 107-token fresh
prompts. Their messages and every request field except max_tokens match exactly.
Do not cite prompt differences for that sample. Output length, admission,
binary, paths, graph/cache history, acceptance/tail behavior and accounting
version remain confounded. Source does not support a causal 2x instrumentation
regression. S29's invalidated draft totals remain separate from its retained
throughput/hash evidence.

VERDICT -> GO for protocol preparation at the next meaningful candidate;
no new run, hook, build or promotion from S67 alone. Reuse the S51 gate: save
a fresh engine-log offset, submit only request0, observe its complete flushed
`strata trace: prompt start <n-1> -1` line, reject if request0 already finished,
then release request1. Save request/marker/release timestamps and prove actual
client overlap and nonzero paired-head activation. This proves native main-loop
entry before release, not GPU prefill execution or fixed slot assignment.
The polling interval, an SSE role chunk and fixed sleep are not admission proof.

For a future stage-local harness clone, use the retained 107-token coding
bodies, cap1500, one warm pair and three fresh finite pairs per arm. Preserve
bench_serving.send request/SSE data, full text/hash, usage, cache/prompt/predicted
counts and current carried draft totals. Require exact per-request work and
native window/pair/solo totals across arms; unmatched work stays unmatched,
without sorting or retrospective slot normalization. SSE events are not tokens.
Keep candidate-independent settings, localhost exposure, quiet CPU conditions,
identity/residency, functionals, normal QUIT/removal and post-health fixed.

Removing the boundary ring requires dropping its frame-dependent parser gates.
Unconditional native batch/pair summaries remain available. Source/build/config
binding, captured-path eligibility, successful warm/use and no capture errors
remain the capture checks. Existing clean logs do not expose every graph cache
hit or per-window row layout: equal aggregate work does not reproduce the old
exact sequence gate. Do not claim capture-free timing, equal graph identities
or independent dispatch proof. Three finite pairs are a bounded latency
observation, not stability qualification.

Skip redundant format probes, an instrumentation-off baseline campaign, extra
diagnostic servers, full200k requalification and routine daily restore. A prior
exact health closure can serve as prehealth only when root verifies no
intervening GPU use or stack change; per-arm teardown/post-health remain required.

Frozen evidence:
`/mnt/vm_8tb/b70/results/flashnext_ista_gsq_rco_iq3xxs_strata_cachefix_20261014/clean-serving-loop-s67-prep/`.
Manifest `manifest.json` SHA-256:
`ae7ecdb8aaf17bdc2192e76c4a555b247ce1b6888520bf4a016af8fee8818c86`.
It binds findings, sampled request comparison and `source-and-config-pins.json`
(SHA-256 `28613942b2b4abfd7a474c34e546a243cff70f7aa9d2cc6e0ac459dd173326ff`).
Key audited source pins: generate.cpp
`6433d750e414aeb9c71135257ae40a8f9e38ea661537c251b50cabdbc2217d83`,
verify.cpp `e97ce93a3434fa1d4e9a3a3dce99a3fd41d86aac8662fbd3c959a3c78ce0d4b2`.
Exact remaining source/config/client hashes and line references live in that
frozen evidence; these pins describe the audit snapshot, not a new runtime.
