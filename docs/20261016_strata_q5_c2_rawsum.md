# Q5_K C2 input half-sum audit (2026-10-16)

CONFIG -> S107 frozen native binary 064b63ce, full Q5_K head N=2560,
M=248320, C2, wideK=1, headRPW2=0, generic RPW1. This follows the
[source-only S121 lead](20261016_strata_dense_mmvq_attribution.md).

COMMAND -> One bounded selected-image/host binding audit, one explicit
exact-symbol selector correction after the first HOLD, then one offline
ocloc BMG-G31 decode under cpu-run. No GPU or kernel build.

RESULT -> V1 exited 1 in 0.341872s before selecting an image: its predicate
confused the registered-kernel name with the distinct host metadata spelling.
The exact three-symbol V2 predicate passed unchanged identity gates, exit 0
in 1.083052s. Four actual host functions and pinned source close the
source-conditioned route to the exact 30168B Q5/C2 image (SHA aa4d0202),
8320B executable text (SHA 7e4b99f3). The missing compiled try_wide body and
case13 jump-table entries remain explicitly source-only edges. The selected
image offline decoder exited 0 in 9.858363s. Original image and short-name
alias retain identical text/data/relocations. GRF128, SIMD32 and eu-thread8
are present; absent SLM/barrier/spill metadata is UNKNOWN.

Actual lowering: the compile-time ones-dot input sums were strength-reduced,
not eliminated. All 16 native DP4A sites compute weighted Q5*q8 dots. Four
distinct raw16 sums (ua/uc in each of two columns) each contain 16 byte-pack
MOVs, 16 signed-byte-to-dword MOVs, one ADD and seven ADD3 instructions:
40 static vector instructions per sum, excluding conversions/FP correction.
The distinct column inputs are loaded with d32x4 sends; ds[0] supplies scale,
and ds[1] does not substitute for these raw16 encoded-q sums. This is not
duplicate work within one half-vector. Row-independent setup repeats in
separate row-owning subgroups; cross-workgroup compiler CSE cannot share it.

The review retains exact operand anchors, FP MAD contraction/order, separate
column accumulation and subgroup XOR reduction trees. The existing source
expression alone does not prove a future candidate's compiled FP behavior.
The observed CR0 OR mask is 0x400004C0; its denorm/rounding meaning is not
established here. Full raw assembly stays external (SHA 5aae8c12).

VERDICT -> Static binding and one selected-image ISA audit PASS. The next
source-only discriminator is a runtime-uniform, kernel-captured 4-byte
ones_word=0x01010101 passed to the existing correction DP4A calls, aiming to
avoid compile-time expansion without changing weighted dots or adding a
producer sidecar. Actual candidate payload forwarding, packed correction
DP4As, removal of the four expanded chains, unchanged FP/reduction ledger
and resources must be inspected before GPU. Then strict integer/rawsum and
output checks with byte extremes and changing A/B/A replay precede timing.
Failure to obtain this lowering closes that spelling. No implementation or
numerical qualification is part of S122; encoded-q sidecar sums remain parked.

Encoded weight bytes/service do not measure physical bandwidth. Static
instruction/send/wait counts do not measure dynamic critical-path cost or
establish a serving saving. Integer sum bounds [-2048,2032] do not establish
universal compiled FP/output equivalence. Actual Opus 46.035s and Fable
66.380s accepted the bounded CPU audit in the linked S121 review record;
those reviews do not authorize a prototype or runtime test.

[Curated evidence](../strata/flash-next/lowlevel_probe/q5_c2_rawsum_s122/README.md)
preserves both audit attempts, exact correction, commands, metadata, host
windows and decoder receipts. Large binaries and full assembly stay external
with retained identities. Production source/configuration is unchanged.
