# Q5 runtime-ones with the production control: S128 (2026-10-16)

S128 passed its finite numerical, compiled-binding and lifecycle gates. The
private candidate completed the two paired round means about 25.6% faster than
the production control in this one process. This is a whole-build comparison;
it does not isolate the effect of replacing the constant ones operand.

CONFIG -> Q5_K full vocabulary projection N2560/M248320/C2/RPW1/SG32;
437043200B real output.weight, two saved S52 canonical Q8 inputs, physical
card 0 (PCI0000:0b:00.0). The candidate captures host-validated 0x01010101 as a
4B scalar. Integer correction dots, FP expression/order and output ownership
retain the reviewed proposal. No producer sidecar or main source change.

The earlier [S126 private-control HOLD](20261016_strata_q5_runtime_ones_compiled.md)
stands. Its control clone failed compiled identity before any GPU run. S128
changes the nominal `original` arm to the same verified production public
native_mmvq used by its public anchor. The candidate header/math are unchanged;
the exact host delta and evidence are
[curated](../strata/flash-next/lowlevel_probe/q5_public_control_s128/README.md).

COMMAND -> One e41/DPC++2026.1 noDRI/network-none CPU adapter build reused the
frozen S59 public/common inputs: 22.138738s, exit 0, ten pre/post pin checks
passed. One selected compiled audit passed in 2.244261s. Actual responding
claude-opus-5-5 (32.265327s) and claude-fable-5-1 (46.184550s), both exit 0,
gave conditional experiment GO; root read both and closed their binding gates.
An independent host read verified argv/scalar validation, capture at offset32,
and distinct candidate/raw-validation/public metadata and live call paths.

The public 8128B computation through EOT, metadata, options, constants and
relocations match S122. Its whole image/text are not byte-exact: text first
differs at 8140, beyond EOT. The candidate whole image and complete 6464B text
equal the already-decoded S126 candidate; its computation ends at 6272.
The existing candidate ISA analysis could therefore be reused without
another decoder run.
SG32/GRF128/EU8 are observed fields; absent SLM/barrier/spill/scratch fields
remain UNKNOWN. These static facts do not measure occupancy or dynamic cost.

RESULT -> One quiet CPU/card-0 window completed in 43s. All 49 snapshots,
656 INT64 rawsum cells (including extremes and a changed-payload negative
control), two whole-weight checks and 74 full binary artifact checks passed.
Full logits match production and saved S52 anchors for initial/changed inputs,
repeats, restores, eager execution and own single/batch graphs. Guards, tails,
inputs and canonical Q8 stayed intact. Five actual DOTs contain 43 nodes and
38 edges; both public/original graphs name the same production kernel, and
candidate graphs name RuntimeOnesQ5C2. Timed graphs contain no quantization.

| Balanced round | Production completed wall us/call | Private candidate us/call | Change |
| --- | ---: | ---: | ---: |
| OCCO | 1266.656700 | 944.261600 | -25.452445% |
| COOC | 1281.063150 | 950.900775 | -25.772529% |

All eight 20-call timing samples are retained; median paired change is
-25.612487%. These are two nested round means in one process, timed on initial input A
and one real tensor; both A and B were checked for numerical correctness. Completion includes host/graph boundaries and is
not native kernel duration. No exclusion, repeated-process population claim,
serving forecast or physical bandwidth inference is made.

Fresh pinned card-0 health before and after passed. Native/start/wait/rm,
driver and tee exited 0; final OOMKilled was false, full CID and name were
absent, and identities matched. The raw closure's `analysis.json` pointer is
inaccurate; a separate clarification points to actual `comparison.json`.

VERDICT -> Keep this as a successful bounded primitive result. Private-TU
context and changed scale/min/load scheduling prevent attributing the timing
change solely to runtime ones. S126 remains HOLD; S128 supplies no main-TU
compiled result, serving qualification, Q4 generalization or automatic next
GPU authorization. Post-result decisions, when closed by root, belong in a
separate indexed supplement. Root removed three obsolete S126V3 build files
after this replacement passed; their identities and failed-control evidence
remain historical records. No old S126 executable is needed to reuse its
retained selected candidate image/ISA.

Post-result decision -> Actual claude-opus-5-5 (43.736249s) and
claude-fable-5-1 (63.623681s), both exit0, reviewed this result together with
the minimal production source. Both returned CPU_BUILD_GO_WITH_CONDITIONS.
Root selected one main integration build with separate default-OFF flags;
compiled and serving gates remain pending. Both also noted that an explicit
C2 head can be a two-token verifier window, not exclusively an MTP pair.
The indexed decision retains response hashes and the whole-build timing limit.
