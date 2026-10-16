# Q5 runtime-ones compiled audit: S126 (2026-10-16)

S126 closed CPU-only on HOLD. The candidate realized the intended packed
correction arithmetic, but the private control clone failed its required
compiled identity check. No GPU numerical oracle, graph replay or timing ran.

CONFIG -> One Q5_K full-head shape, N=2560/M=248320/C2/RPW1/SG32. Candidate
changes only the rawsum ones operand to a host-parsed, kernel-captured int
0x01010101. Source weights, canonical Q8, FP expression, K order and reduction
tree are retained. No sidecar, model change or production source edit.

V1/V2/V3 retain identical kernel header a8045fa3, build scripts and fixture
pins. V2 removes 47 redundant full-weight readbacks from the planned oracle:
20541030400 bytes (about 20.54GB), while preserving 49 output/input/Q8/guard
snapshots and two full 437043200B weight checks. This is a planned testing-cost
trim, not measured runtime savings. V3 ports balanced capture/error drain and
checks the validation output tail before its host INT64 comparison. Neither
host change supplies compiled or numerical proof.

COMMAND -> Actual responding claude-opus-5-5 (53.480798s) and
claude-fable-5-1 (58.192092s), both exit 0, approved only the CPU build. Root
read both answers and the small V3 readiness delta. One e41/DPC++2026.1,
precise/SG32/per-kernel/BMG-G31, noDRI/network-none adapter build reused the
frozen S59 common8f/kernelsf2b/core34e9 artifacts. No main AOT rebuild or model
payload read. Both audit attempts and the selected decode ran under cpu-run.

RESULT -> The build exited 0 in 23.125537s, with compile 10.375s, private AOT
0.526s and host link 0.386s. All ten source/reused-input pre/post checks passed.
Built executable SHA66fae0f1, host object8e75c111, images e5968f2f; full
identities and actual commands are [curated](../strata/flash-next/lowlevel_probe/q5_runtime_ones_s126/README.md).

| Completed CPU step | Result |
| --- | --- |
| Initial selected audit, 2.275797s | HOLD: SDK raw-wrapper ABI assumption wrong |
| Explicit corrected audit, 2.670346s | Wrapper fixed; private-original prefix HOLD |
| Public frozen-f2b control | All 8128B computational-prefix/metadata/options/data/relocation gates PASS |
| Private OriginalQ5C2 | Text 8704B versus 8320B; first difference byte 167 |
| One early candidate decode, 10.153557s | Exit 0; input unchanged |

The wrapper error was an audit bug, not a kernel/runtime failure. Its actual
ABI has two by-value aggregates: range8B and closure24B, with pointer-valued
fragments marked is_ptr=true. It has no separate arg_bypointer signature.
The narrow correction retains exact source offsets/fragments and leaves all
projection/control gates unchanged. Both attempts and HOLDs are preserved.

The corrected audit binds five LIVE forms: public, private-original,
candidate, direct Raw16Sums and the SDK's wrapped Raw16Sums, with ten distinct
cache/guard cells. Public host provenance uses the actual frozen f2b archive
member. Public complete text differs first at 8140, beyond the equal 8128B
computational prefix; full-image equality is not claimed. Private-original
metadata/options/constants/raw and decoded relocations match, but its body
does not. No cause, numerical failure or identity waiver is inferred.

The early decoder input and corrected-audit candidate are byte-identical:
27816B, SHA6069468cf1b16aa42e7a868b0e928b8c512784485f3b7c553bf4df78dd6612a7.
The independent review read all 510 candidate ASM lines. Its computation ends
before trailer byte 6272: 32 native DP4As comprise 16 weighted dots and 16
correction dots consuming the extra uniform payload r4.0. Four expanded
40-instruction sum chains disappear; no ADD3 remains. The tracked weighted
operands, signed integer dots, FP conversions/MUL-MAD boundaries, per-column
accumulation, K order, XOR tree and output ownership match the original
semantic ledger despite register/load/branch scheduling changes.

That is a bounded local compiled-mechanism PASS. SG32/GRF128/EU8 match;
absent SLM/barrier/spill/scratch fields remain UNKNOWN. The raw CR0 mask matches
without assigning denorm semantics. It is not whole-body identity, universal
FP equivalence, numerical correctness, dynamic cycles or serving speed.

VERDICT -> Root retains overall S126 HOLD and issues no GPU_GO. The prepared
runner remains unexecuted; source review gates and expected future snapshot
counts are not test results. A separate S128 harness is being prepared to use
the verified public production kernel directly as control, avoiding an
unnecessary private clone. Candidate runtime arithmetic remains the same
proposal. S128 must establish its own source/build/compiled/runtime closure;
it cannot turn the failed S126 control gate into a pass.
