# S96 routed IQ2 source reuse gap, 2026-10-15

The source audit identifies a distinct candidate direction: routed IQ2_XXS
(type16) and IQ2_XS (type17) GU currently have no Multi/KqSplit specialization,
so their source loops repeat weight decoding per routed entry. Existing
Multi18/21/22 GU and20/42 DOWN already share preparation within E2/E4 groups.
This is source evidence, not a new GPU result or compiled redundancy proof.

Root read the final findings and actual Fmt16/17, row_entries, row_dot_lanes,
Multi dispatch and dp4a implementation. All ten pinned source/evidence hashes
match. Conservative integer bounds are INT32-safe, but any future candidate
must preserve truncation, per-lane k order, FP conversion/scale/accumulation,
XOR reduction and output ownership. Keep IQ2_XS's explicit uint32 shift
extraction; the source warns that a non-char type-punning alternative produced
incorrect values. Integer algebra alone does not certify a new implementation.

Next: inspect ONLY the existing native_gu_port<17,8,true> image from frozen
main d94e96ca. Bind its own actual registered kernel name and image; the
attention13-tag and S66 dense-IQ3 proofs do not identify this routed kernel.
Check emitted dot instructions, repeated decode/load placement, and resources
before proposing code. No new kernel, build, ISA decode, GPU test, model payload
read, model consultation, numerical proof or speed claim occurred in S96.
Actual grouped-entry reuse frequency remains unmeasured. A uniform launch-grid
capacity is not observed work or hardware traffic.

The two files here are byte-exact copies from external
`routed-dot-lowering-s96-source/`, under the results root referenced in the
[S95 report](../../../../docs/20261015_strata_current_c2_profile.md).
All experimental candidates remain OFF; S30 recovery and throughput goals
are unchanged. Do not run an old fixed-path fixture without revalidation.
