# S108 retained head-readiness evidence

CONFIG -> Historical instrumented S95 primary-GPU trace, reviewed cycles5..16,
with source-informed ordinal roles. No CPU/peer clock join or independently
logged owner/buffer binding. Timestamp arithmetic retains trace float precision.

COMMAND -> One CPU-only bin/cpu-run extraction, terminal0 in6.76s. Root read
source, metadata, note and execution record, independently verified17 indexed
payload hashes and all12 interval calculations/medians. Curation copies those
ASCII payloads, index/root verification and prior source-coverage audit exactly.
The269853744-byte raw trace remains external; curation did not reread/hash it.

RESULT -> Exactly132 events,11 per interval, resolve the chain:

C2 head -> sampler -> top probability -> sampler -> top probability ->
select -> select -> mapped step/position copy -> token embedding -> RMS -> Q8.

| Observed metric | Median us | Scope |
| --- | ---: | --- |
| C2 head | 1202.2395 | Kernel service |
| Four sampler/probability operations | 243.175 | Per-interval service sum |
| Two samplers | 179.1135 | Per-interval service sum |
| Two probability kernels | 64.009 | Per-interval service sum |
| Two selectors | 33.6975 | Per-interval service sum |
| Second-selector end to mapped-copy start | 21.875 | Gap in collected operation coverage |

The254.625us head-end-to-first-selector span contains substantial measured
sampler/probability work and EXCLUDES selector service. No additional/crossing
primary-GPU event occurs within these12 bounded extracts. Source ordering makes
the mapped copy the next graph's start and embedding its selected-token consumer.

Only the64.009us probability-service pool is relevant to a separate paired-only
omission candidate; the243.175us four-operation pool also includes samplers.
No omission, attainable saving or serving benefit is measured. Medians and
service/interval scopes must not be added into a latency budget.

Separate current-source audit S110 finds global, slot-solo and paired heads
ALL use full vocabulary in the observed configuration. There is no current
subset/full asymmetry. N2560/V248320 Q5_K encoded weights total437043200B;
this static extent is not measured physical traffic or bandwidth. A future
subset-head proposal would change draft configuration and needs its own gates.

VERDICT -> Coverage resolved without another GPU profile. The21.875us gap is
not demonstrated physical idle, data starvation, a removable host wait or
queue-readiness causality. Twelve cycles are nested within one historical trace,
not independent runs. No S106 timing/stability/serving gain or new candidate
qualification follows. Retain these small outputs to avoid broad raw reparsing.

[source-copy-index.json](source-copy-index.json) records exact source/destination
hashes. [index.json](index.json) indexes17 original payloads;
[root-verification.json](root-verification.json) records independent root checks.
See [note.md](note.md), [findings.json](findings.json),
[ordered-timeline.csv](ordered-timeline.csv), [bounded-intervals.json](bounded-intervals.json),
[execution.json](execution.json) and [coverage/note.md](coverage/note.md).

[Actual Opus/Fable review decision](postreview/root-post-decision.json),
[review copy index](postreview-copy-index.json),
[source audit copy index](source-audits-copy-index.json) and
[root copy verification](root-copy-verification.json) retain the decisions and
exact copies. Both actual models accepted bounded coverage; root corrected the
first-select span, probability-only pool and vocabulary-row distinction.
No raw trace belongs in this directory.
