# MTP head-readiness coverage, 2026-10-15

CONFIG -> Historical instrumented S95 primary-GPU trace, reviewed cycles5..16,
with source-informed ordinal roles. No CPU/peer clock join or independently
logged owner/buffer binding. Timestamp arithmetic retains trace float precision.

COMMAND -> One CPU-only bin/cpu-run extraction, exit 0 in 6.76 s. Root read
source, metadata, note and execution record, independently verified 17 indexed
payload hashes and all 12 interval calculations/medians. Curation copies those
ASCII payloads, index/root verification and prior source-coverage audit exactly.
The 269853744-byte raw trace remains external; curation did not reread/hash it.

RESULT -> Exactly 132 events, 11 per interval, resolve the chain:

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

The 254.625 us head-end-to-first-selector span contains substantial measured
sampler/probability work and EXCLUDES selector service. No additional/crossing
primary-GPU event occurs within these 12 bounded extracts. Source ordering makes
the mapped copy the next graph's start and embedding its selected-token consumer.

Only the 64.009 us probability-service pool is relevant to a separate paired-only
omission candidate; the 243.175 us four-operation pool also includes samplers.
No omission, attainable saving or serving benefit is measured. Medians and
service/interval scopes must not be added into a latency budget.

Separate current-source audit S110 finds global, slot-solo and paired heads
ALL use full vocabulary in the observed configuration. There is no current
subset/full asymmetry. N2560/V248320 Q5_K encoded weights total 437043200 B;
this static extent is not measured physical traffic or bandwidth. A future
subset-head proposal would change draft configuration and needs its own gates.

VERDICT -> Coverage resolved without another GPU profile. The 21.875 us gap is
not demonstrated physical idle, data starvation, a removable host wait or
queue-readiness causality. Twelve cycles are nested within one historical trace,
not independent runs. No S106 timing/stability/serving gain or new candidate
qualification follows. Retain these small outputs to avoid broad raw reparsing.

[Indexed evidence, source copies and verification](../strata/flash-next/lowlevel_probe/head_readiness_s108/README.md).

Actual postreviews by `claude-opus-5-5` (24.523 s) and
`claude-fable-5-1` (16.287 s) accepted bounded coverage and source audits.
Root corrected the selector span, the probability-only pool and the vocabulary
row/column distinction; neither review establishes a speed improvement.
See the [review decision](../strata/flash-next/lowlevel_probe/head_readiness_s108/postreview/root-post-decision.json)
and [full/subset source audit](../strata/flash-next/lowlevel_probe/head_readiness_s108/source-audits/mtp-subvocab-source-s110/contract.txt).
