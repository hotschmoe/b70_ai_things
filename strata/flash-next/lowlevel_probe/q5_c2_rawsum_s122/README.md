# S122 Q5_K C2 selected-image audit

The original selector HOLD and one explicitly authorized corrected continuation
are separate. V2 changes only the host Info/cache/guard exact-symbol predicate;
its source, identity and selected-image gates are unchanged. The first failure
is an audit predicate error, not a kernel or model failure.

- prep/: original/corrected audit source, exact selector diff and diagnosis,
  source/memory contract, pins and one-image offline decoder.
- v1/: original failure/command/log, preserved without retry.
- v2/: corrected command/exit/log, actual binding facts, source-conditioned
  host review, exact critical-line excerpts and selected metadata.
- decode/: actual one-image offline command/exit/tool/input/output receipts.
- source-copy-index.json: exact copy hashes and explicitly derived excerpts.
- external-identities.json: excluded binary/full-ASM identities. These are
  known retained identities, not new large-artifact hashes.

The scripts preserve original external path assumptions and are evidence
copies. Run only from an explicitly prepared external directory with the
pinned dependencies; no repository script is an execution authorization.
Source review decisions and actual Opus/Fable answers are already curated in
[the S121 bundle](../dense_mmvq_source_s121/reviews/root-decision.json).

The complete selected-ISA review is retained in decode/isa-findings.md with
exact operand windows and pins. Four distinct raw16 input sums survive as
expanded byte extraction/sign extension/add trees (40 static vector
instructions each); all16 native DP4As are weighted dots. Static binding is
not observed runtime dispatch. A runtime-uniform correction operand is the
next source-only discriminator, subject to actual compiled ABI/FP review and
strict private tests. No GPU, prototype, numerical, speed or bandwidth claim.
