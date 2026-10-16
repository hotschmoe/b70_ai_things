# Strata attention-value fixed-policy serving, 2026-10-16

S113 passes the new matched serving discriminator. Native prefill medians
are 1294.9 -> 1299.2tok/s at 4K and 1548.9 -> 1590.6tok/s at 32K. Keep the
attention-value pipeline default OFF: these are three nested fixed-order
samples per shape, not a stability or shelf/default qualification.

CONFIG -> Both fresh arms use the same existing S107 executable
`064b63ce75d7568db541af5cf24eee07f41d0e54ea49813af9958a816d75d5ca`.
`STRATA_ATTN_VALUE_PIPELINE=0/1` is the sole algorithm flag difference.
Native argv fixes suffix-draft0, lookup-chain0, spec4, mtp-max-t4 and
spec-min-p0. HTTP requests explicitly override only spec_min_p0.
Headpair/depth2, EP149/363 with all experts resident, two 200k INT8 slots,
32GiB/four RAM parking entries, Direct PLE and prefill12288 stay common.
QSA, A4/CENTER/PACKED, IQ2 multi, whole-pair graphs, headRPW2, commit
overlap, XMX and KV rotation stay OFF; LANECELL is absent.

The existing caller accepts complete staged or resident mode0/no-elastic
identity pages, INT8, 32 full-width queries and cap 2051. The wrapper further
checks 24 query heads/two KV heads, head_dim 256/page 4, nonnull INT8 pool
pointers/scales and absence of Q4 pools.
Ineligible calls retain the original entrypoint. The older staged-only
comment is not the current caller predicate. S107's actual build receipt
retains unchanged attention object 06eb664c and prefill object ae8493fd.
S113 performs no build or source edit, and makes no new ISA/arithmetic claim.

COMMAND -> Root verifies frozen source/controller/fixture pins, existing
binary identity and model-stat continuity, then runs OFF followed by ON.
Prehealth uses explicitly recorded continuity from the prior full S107
both-card/compiled closure and later S112 card0 closure on the same boot;
it is not a newly run full S113 prehealth. Servers hold coordinated pair
GPU leases. CPU leases cover startup/client phases/health; timed clients
use cpu-run --quiet. No profiler or new trace/parse is introduced.

Each arm receives one new unmeasured 4K/256 warm followed by the six exact
original S38 timed texts: three 4K, then three 32K,256 output tokens each.
The warm changes only the initial repeat 0 label to warmup 0; equal character
length does not prove tokenization. Its actual count is 4060 in both arms.
Timed prompts retain 4059/32728 tokens with cache 0. Every ON response must
match its OFF request, text, concatenated SSE content, usage and native
prompt/cache/generated/offered/accepted counts before another request.
Native completion counts, part bounds and observed capture/window/pair/solo
reports are also reconciled. Missing evidence or a first mismatch ends
generation without retry or normalization. The actual root stage driver
uses finally cleanup after failed or successful client phases.

RESULT -> All response/work/activation/identity gates pass. The exact
attention-value HOST selection literal is absent OFF, appears once ON by
the new warm and remains once through functionals/teardown. QSA 0 and absent
unrelated ON markers match their expected scope. This establishes observed
host eligibility, not independent GPU dispatch counts or all-call coverage.
Actual native argv/env, loaded libstdc++ path/hash and process maps are
retained. Both arms pass concurrent arithmetic, reasoning 391 and the actual
tool roundtrip. Normal TERM -> native QUIT, final diagnostic, fullCID AND
name gone, wrapper 0 and per-card/compiled collective post-health 0 all pass.
The observed prestop container state is Running/notOOM; removed containers
provide no later final inspect OOM observation.

| Request | Native prompt tok/s OFF -> ON | Change | Native prompt ms OFF -> ON | Client TTFT seconds OFF -> ON |
| --- | ---: | ---: | ---: | ---: |
| 4K | 1294.9 -> 1299.2 | +0.3321% | 3134.5 -> 3124.3 | 3.16453 -> 3.15428 |
| 32K | 1548.9 -> 1590.6 | +2.6922% | 21129.3 -> 20576.2 | 21.19172 -> 20.63773 |

One fresh process per arm in fixed OFF-then-ON order supplies these three
nested requests per shape. Ranges and full response records remain retained;
there is no independent process replication, significance, p95 or stable-gain
claim. SSE content equality does not establish identical event chunk
boundaries or raw token IDs. No new C2 decode throughput, full-context/cache
qualification or upstream/default promotion follows.

VERDICT -> Keep default OFF, retain the descriptive matched opt-in result,
and do not immediately repeat or run a full-cache campaign. S94 remains
FAILED_NATIVE_WORK_MATCH and its cause remains unresolved. S106/S107 fixed
policy matching motivated this new discriminator; it does not prove that
S94's probability/policy cause was fixed. New S114 source work begins after
S113 closure and does not retroactively change its source/binary identity.

Actual post-result consultation resumed the existing independent sessions:
claude-opus-5-5 completed 55.236s/exit0 and claude-fable-5-1 completed 64.737s/
exit0. Root read both full answers. Both recommend default OFF and no
immediate repeat/cache campaign. Root does not adopt Opus' qualified-candidate
wording as shelf or stability qualification. For the separately proposed HC
producer-activation change, both require an offline audit of existing
selected UP/reduce ISA first. Root preserves extra+0/division, signed-zero,
subnormal/nonfinite and native-exp obligations, and rejects universal
negative-zero/source-bit-equivalence claims. Absent resource fields remain
unknown; raw byte equality and ordered FP dataflow are distinct evidence.
No HC implementation/build/GPU result is established by this consultation.

Dev-loop cost: no compilation, daily-server restore, cache campaign or
profiler setup. The bounded six-request timed phases took 95.203s OFF and
93.468s ON; each warm took about 7.07s, functionals about 6.0s and normal
stop/post-health about 58s. These are phase wall costs, not device timings.
The next work is the separately reviewed S114 MTP integration and a source/
existing-image HC audit, with no immediate S113 repeat.

Exact controllers, source/build receipts, root phase driver/logs, all raw
responses, identity/maps/library evidence, normal closures/health, comparison,
actual model answers and root corrections are retained in
[attention_value_fixed_policy_s113](../strata/flash-next/lowlevel_probe/attention_value_fixed_policy_s113/README.md).
The per-file index preserves original paths/bytes/SHA; filtered model
completion metadata retains actual responding IDs/local elapsed/exit and
omits cumulative usage. Large binaries/objects/archives remain external.
