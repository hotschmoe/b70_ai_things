# S113 attention-value fixed-policy serving, 2026-10-16

CONFIG -> Same existing S107 executable in both fresh arms:
`064b63ce75d7568db541af5cf24eee07f41d0e54ea49813af9958a816d75d5ca`.
Only `STRATA_ATTN_VALUE_PIPELINE=0/1` differs algorithmically. Native argv
fixes suffix-draft0/lookup-chain0/spec4/mtp-max-t4/spec-min-p0. The explicit
HTTP override is only `strata_tune.spec_min_p=0`. EP149/363 resident experts,
200k INT8 contexts, 32GiB/four parking slots, Direct PLE, prefill12288 and
headpair/depth2 stay common; unrelated candidates stay OFF.

COMMAND -> No build/source change. Each arm runs one new unmeasured 4K/256
warm, six original ordered timed requests (3x4K then3x32K/256 outputs),
functionals, normal QUIT and both-card/compiled collective post-health.
The actual root_run_ready_arm.py retains try/finally cleanup and stops
phases after any client failure. GPU leases span servers; quiet CPU leases
cover timed clients. Prehealth continuity is explicitly contextual reuse
of the prior full S107 closure plus later S112 card0 closure, same boot;
see serving/root-preflight-go.json and prehealth-context/.

RESULT -> PASS_MATCHED_DISCRIMINATOR. Both warm requests actually have 4060
prompt tokens/256 outputs. All six timed requests per arm have the original
4059/32728 prompt counts and cache 0. Full request/text/SSE content/usage,
native offered/accepted work, part bounds and observed capture/window rows
match. Functionals, literal HOST marker checks, startup identity, actual
native argv/env/library provenance, normal QUIT, fullCID/name absence,
wrapper 0, observed prestop Running/notOOM and post-health 0 all pass.

| Request | Native prompt tok/s OFF -> ON | Change | Native prompt ms OFF -> ON |
| --- | ---: | ---: | ---: |
| 4K | 1294.9 -> 1299.2 | +0.3321% | 3134.5 -> 3124.3 |
| 32K | 1548.9 -> 1590.6 | +2.6922% | 21129.3 -> 20576.2 |

These are medians of three nested requests per shape, one fresh process per
arm in fixed OFF-then-ON order. Exact ranges/client TTFT/wall are retained
in descriptive-metrics.json and serving/comparison.json. SSE content
matching does not establish event-chunk-boundary or raw token-ID equality.
Removed containers provide no final inspect OOM state; the observed state
is the recorded prestop Running/notOOM observation.

VERDICT -> Retain default OFF; descriptive matched opt-in result only.
No immediate repeat, full-cache campaign, shelf/default/stability promotion
or S94 cause-fixed claim. S94 remains FAILED_NATIVE_WORK_MATCH with its
cause unresolved. This new fixed policy/warm protocol does not rewrite it.
S114 source work starts after S113 closure and changes no S113 identity.

Actual post reviews claude-opus-5-5 (55.236s) and claude-fable-5-1 (64.737s)
completed exit0 in their own resumed sessions; root read both full answers.
Both keep default OFF/no immediate repeat/cache campaign. Root rejects
Opus' shelf/stability interpretation of qualified candidate, shorthand
implying five explicit HTTP fields, and universal source/negative-zero
claims in the separate HC proposal. The HC next step is an offline audit
of existing selected UP/reduce ISA, before any private implementation.
See postreviews/root-post-decision.json for each correction.

Exact evidence organization:

- serving/: all 155 closed prep/runtime files, including six controllers,
  frozen source-only records, root driver/phase logs, raw request/response
  and SSE-content JSONs, configs, identity/maps/library evidence, native
  segments/final logs, functionals and every teardown/health record.
- source/: exact S94 resident-caller patch and pinned prefill/attention
  sources plus bench_serving.py. S107 full executable remains external;
  unchanged attention-object identity 06eb664c is retained from its actual
  receipt, not newly inferred from a source hash/stat.
- prior-build/: small original S94/S107 build receipts and hashes; no
  binary/object/archive copies or new large-artifact hash.
- independent-source-review/: historical frozen audit unchanged. Its
  client-only cleanup limitation is superseded operationally by the exact
  later root driver with finally cleanup, which is retained in serving/.
- postreviews/: exact packet, answers, commands/prompts, root decision and
  filtered completion metadata with actual IDs/session/elapsed/exit. Raw
  completion/event streams stay external; cumulative usage is omitted.
- record-copies.json: every exact copy's original absolute path/bytes/SHA.
  Derived metrics/filtered metadata are identified separately. Historical
  source-only statuses remain historical, not rewritten runtime verdicts.

Original result root:
`/mnt/vm_8tb/b70/results/flashnext_ista_gsq_rco_iq3xxs_strata_cachefix_20261014/`.
These retained controllers preserve their external paths and are evidence,
not runnable in-place recipes or permission to overwrite a completed run.
