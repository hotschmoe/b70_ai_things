# S97 existing routed IQ2_XS evidence

Date: 2026-10-15. Linked-image/ISA audit: /root/mtp_q4_share; source ledger and
curation: /root/ple_upstream; review and experiment decisions: /root. See the [dated report](../../../../docs/20261015_strata_routed_iq2_reuse.md).

CONFIG -> Frozen S94 d94e96ca, IQ source acaa7da6/object a61c1dca,
S95-observed routed native_gu_port<17,8,true>. No candidate build/GPU here.
COMMAND -> tools/extract.py and tools/decode.sh record the exact executed
absolute-path recipe; use bin/cpu-run. decode.sh intentionally refuses
existing outputs. Helpers require the original external paths recorded
in record-copies.json; this is an evidence copy, not a portable framework.
RESULT -> audit/report.json, selection.json and qt17.ze_info bind the
actual public host launch to one selected image. assembly-excerpts.txt
shows per-entry repeated decoding and existing hardware DP4A. The bounded
host-metadata-excerpt.txt supplies original host callsite lines.
VERDICT -> One QT17 E4/E2/tail reuse source lead, no speed/serving claim.

record-copies.json binds 22 byte-exact ASCII copies to their raw paths and
hashes. completed-isa-manifest.json is the unchanged upstream audit inventory:
60 files/387757 bytes, SHA256
62ccb5119de28d5a90cdf88b111576f74c67db860200370ef2316ede29ebc675.
It inventories omitted zebins/full ISA/host disassembly and stopped helper
preflights externally. No binary payload is copied here. The derived host
excerpt has its own original-source hash and line range in the copy index.

The source/findings.md ledger is the earlier source-only record: its final
compiler-hoisting question is now answered by audit/report.json. Its early
layout proposal predates S98's final singleton-only layout addition. S98
source preparation is frozen separately (17 files/69521 bytes; manifest
8c1211066e78e5d607b16ee474c81fd5a70a0d76d9307d94ed1e4e1501fcf2a8)
and remains unbuilt/unqualified. No S98 candidate source is copied here.

Absent resource fields do not mean zero; static loop instructions do not
measure reuse frequency, cache behavior, physical traffic or latency cause.
The routed-expert and nonvocabulary MMVQ operation families are disjoint.
