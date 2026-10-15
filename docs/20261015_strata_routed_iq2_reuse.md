# S97 routed IQ2_XS decode reuse audit, 2026-10-15

Authors: /root/mtp_q4_share (linked-image/ISA audit), /root/ple_upstream
(source arithmetic/ownership ledger and curation), and /root (review and
experiment decisions).

CONFIG -> Inspect the frozen S94 d94e96ca executable and its unchanged
IQ object a61c1dca, source acaa7da6. S95 observed the routed GU
`native_gu_port<17,8,true>`: IQ2_XS, LN8, SG32, 256 threads, H2560/FF640.
This is the routed-expert family, disjoint from the nonvocabulary MMVQ
name predicate. The earlier 7.504 ms routed-family service diagnostic is
not an additive latency budget or an isolated IQ2_XS GU measurement.

COMMAND -> Read the exact Fmt17, lane/entry loops, generic multi-entry
machinery, retained rejection history and existing fixture descriptors.
Bind the actual host launch through object section/addend relocations,
final metadata/cache/guard and registered-name pointer to one embedded
image. Decode only that selected image under cpu-run in pinned e41 with
no DRI/network. No build, GPU, model payload read or new rawtrace parse.
Exact evidence and commands: [curated records](../strata/flash-next/lowlevel_probe/routed_iq2_existing_isa_s97/README.md).

RESULT -> The reuse gap survives compilation. The entry loop
L1824..7152 contains the K loop L2304..6632; each entry resets its K and FP
accumulators. Four weight-code loads, four codebook gathers, sign expansion
and scale conversion/product remain inside that nested K loop. Decoded
weights were not hoisted across entries. All eight signed SIMD32 DP4A
instructions are already present: a missing hardware dot intrinsic is not
the mechanism to pursue.

The exact 43,272-byte image hashes to
`37cb80cd3034716da7cb4916cbe2379a35abc375169c171568f8525fe780cd78`;
its 7,488-byte target text hashes to
`bfbb475964f034bba8063a2e746a44a1dbef26d97f68ce4c79a99d2a374db2e1`.
The only other kernel is Intel_Symbol_Table_Void_Program. Metadata reports
GRF128, SIMD32 and eu_thread_count8. SLM, spill and scratch fields are
absent; absence does not establish runtime zero or occupancy. The source
and S95 launch use local size256. Decoder aliasing changed only section
names on a copy; selected code bytes and original executable were retained.

Source eligibility is narrow: Multi17 is absent, so the current lane port
uses Fmt17 per entry. A private specialization can prepare the same eight
signed packed weight words once for natural E4/E2/singleton groups while
keeping each entry's two ordered four-DP4A chains. Preserve exactly
`(s0*ls0+s1*ls1+(s0+s1)/2)/4`, signed truncation, explicit `float d`
statement split, ascending per-lane K additions and XOR4/2/1 reduction.
The source ledger bounds the integer numerator below INT32_MAX and the
last IQ2_XS reads within its 74-byte block. Repeated ent_tok indices alias
read-only activations; output ownership remains entry-major, separate
640-row gate/up planes. Public phase1 scratch provides a byte anchor.

VERDICT -> Retain one private QT17 reuse source candidate. S98 source
preparation is frozen but unbuilt and unqualified; no speed or serving
claim follows. Its mixed, singleton-only and zero layouts must distinguish
reuse benefit from an E1 resource cost. Public/original/candidate full-byte
parity, real-weight routing and actual host/device bindings remain gates.
No lane/format sweep, slower EXPERT_SPLIT/grid staging, alias workaround,
new reduction or tolerance relaxation is proposed. Existing S42 exported
GU slices suffice; no model copy is needed.

The loop evidence does not measure actual reuse frequency, cache misses,
transactions, bandwidth, occupancy or service causality. Parser preflights
stopped before the single decoder run; reusable helpers needed real
section/addend support, target-section enumeration and explicit demangler
normalization. Those corrections are evidence limits, not extra GPU runs.
