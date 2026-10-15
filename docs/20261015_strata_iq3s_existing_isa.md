# S66: existing IQ3_S C6 machine-code audit (2026-10-15)

CONFIG -> Narrow the next candidate to SWAR sign decoding only. Existing
compiled code already folds the original narrow source loads; an aligned-load
rewrite has no six-to-three emitted-message premise.

COMMAND -> Inspect one public IQ3_S NC6/EXACT_N kernel from frozen S59 executable
56e551af and its common wrapper8f5a7efe. Offline extraction and pinned e41d59c0
ocloc disassembly run under cpu-run without DRI/network. No new compilation,
GPU workload, model payload, source edit, archive write or candidate timing.
Native audit author: /root/strata_ep_s27; root read source binding, address
construction, actual assembly excerpts and the offline alias proof.

RESULT -> Index1086 images/read5,838,269 metadata bytes; select image84 at
executable offset8170112,49384bytes. Exact compiled name identifies public
launch_mmvq_multi_nc<21,false>, NC6/EXACT_N. This identifies the image present,
not an independent observation of runtime dispatch. Canonical image SHA256:
`bad2ccded9f4136e6809825844815652eba954f059b807774c45d7916d92beb4`.

The original main text section's264-byte name plus .asm exceeds the filesystem
filename limit. ocloc exited0 but did not produce the expected main assembly;
checking only exit status would have missed that. One copied image received a
short section-name alias for offline disassembly only. The changed range is
[47479,47737), entirely in .shstrtab; all other sections and16,128bytes of main
machine code are identical. Neither the copied image nor alias enters a link
or runtime. The canonical image remains unchanged. Main text SHA256:
`84928699e5a62acaf245b7df75dbc40681492961a491da76599ff8a57099866f`.

The recovered main assembly is101,331bytes/1152lines. At lines164/167/168,
three load.ugm.d32.a64 messages reconstruct qs+0/4 and signs+0. Address tracing
binds them to row+110*kbx+4*iqs+2/+6 and row+110*kbx+2*iqs+74. The separate
qh byte load is not one of those three. Six source halfword loads have already
become three d32 messages; a source-level load count is not an ISA count.

Byte compare/subtract expansion survives:32 static byte nonzero predicates,
with byte extraction, select, repack and word subtract sequences in the loop.
Assembly excerpts220..260/302..338/390..411 retain that dataflow. Native dp4a
instructions also appear in the dot path; source helper spelling is not proof
of emulated integer multiplication. Static instruction occurrences are not
dynamic counts, cycles, traffic, utilization, occupancy or latency attribution.

VERDICT -> S65 source preparation now keeps the exact original load expressions
and changes only sign compare/subtract to existing packed SWAR helpers. This
updates the earlier S63 proposal before any candidate build or model review;
frozen S63 evidence remains unchanged. No wider-load variant or GPU retry is
needed. Compiled numerical equality and useful latency improvement remain open.
Public/private strict anchors and actual Claude reviews precede the next build.

Raw root:
`/mnt/vm_8tb/b70/results/flashnext_ista_gsq_rco_iq3xxs_strata_cachefix_20261014/iq3s-existing-isa-s66-prep/`.
`s66-report.json`, `selection.json`, `assembly-excerpts.json`, original/aliased
zebin, `alias-proof.json`, exact tool logs and hidden assembly files preserve
all evidence. The36-file manifest excludes itself; SHA256:
`d2dd8a6c3423d40afb3b47f4313813b8bbce89feb59468b8718dde59212724da`.
Exact extraction/alias/summary sources are retained in
[lowlevel_probe/iq3s_existing_isa_s66](../strata/flash-next/lowlevel_probe/iq3s_existing_isa_s66/README.md).
