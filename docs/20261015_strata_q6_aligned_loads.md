# Q6 aligned-load findings, 2026-10-15

The private A4 load candidate passes strict checks on two real Q6 C6
projection shapes. Four balanced warm rounds in one process show lower
median paired completed latency on both. This is a primitive result;
production decode and prefill benefit remain unmeasured.

| Shape, six columns | Original median paired mean | Candidate | Median paired change |
| --- | ---: | ---: | ---: |
| 6144 -> 2560 | 63.097 us | 60.053 us | -4.853% |
| 2560 -> 10240 | 73.828 us | 72.207 us | -2.205% |

A4 loads a four-byte-aligned vector plus a fifth dword and retains the
halfword funnel. The compiled offset-selection chain is simpler: static
selects38->8, moves471->439, compares43->37; UGM sends remain41. Signed
centering,48 DP4As, per-column FP products/MAD/ADD and XOR reduction remain.
SIMD32/GRF128 are unchanged; absent spill/scratch fields are unknown. None
of these counts predicts physical traffic or end-to-end savings.

All114 snapshots pass against public, saved S44 and private-original
anchors: both inputs/restored input, full outputs, canonical Q8, weights
and inputs readonly, guards/tails/finiteness, single and20-call graphs.
All correctness checks on both shapes precede timing. Only two pinned
weight slices totaling34.4MB were read; no model-wide hash or copy.

Actual claude-opus-5-5 source41.562s/compiled38.755s and
claude-fable-5-1 source53.569s/compiled62.821s accept the bounded tests,
with root corrections to universal FP claims, absent fields and static
count interpretation. Both public and private-original computational
prefixes match the retained S85 original16384 bytes; raw full image/text
identity is false. The candidate has its own distinct host/image binding.

The loop used one25-second CPU adapter build, one10-second selected decoder
and one40-second card0/quietCPU window including health. All normal exits,
identity/analysis and pre/posthealth pass, exact created CID/name gone;
no reset/retry. The retained --rm runner does not observe a final Docker
OOM field. This does not qualify production stability, a default change or
a serving gain. Keep the candidate separate while testing whole-pair graphs.

[Exact sources, actual model reviews, compiled evidence and runtime records](../strata/flash-next/lowlevel_probe/q6_aligned_load_s104/README.md).

Post-result consultation: actual claude-opus-5-5 (23.074s) and
claude-fable-5-1 (18.456s), resumed in their own sessions, both recommend
preparing a narrow default-OFF main proposal after S106. Root keeps A4
private while that proposal is reviewed. No main application, build or
serving measurement follows automatically. Prefer one single-flag arm in
the next planned matched serving window over another primitive campaign.

Observed resource agreement covers SIMD32/GRF128/eight threads; absent
spill/scratch fields remain unknown. Consistent nested timings do not exclude
systematic measurement artifacts or establish a general non-slower result.
The 25-second CPU adapter build excludes queue time; the immutable review
packet's earlier including-wait wording is corrected by this additive record.
Actual answers and root corrections are in the post-review/ record bundle.

S106 subsequently closes with effectively flat serving throughput. A guarded
Q6 main-source proposal is now prepared outside the live checkout, with the
original kernel unchanged, exact shape/configuration and alignment guards,
and original fallbacks. It remains unreviewed, unapplied and unmeasured.
