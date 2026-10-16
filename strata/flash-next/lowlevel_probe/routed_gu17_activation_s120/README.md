# S120: existing routed GU17 activation audit

CONFIG -> Frozen S107 main064b63ce, unchanged IQ object09447b7e and kernel
archivecf710; TG17/LN8/KMtrue. Existing compiled image only, no new kernel.

COMMAND -> Root's first CPU audit saved the selected image/metadata then
failed on literal whitespace in the metadata parser. One authorized V2
continuation fixed exact-name whitespace matching, reused the saved image,
and retained all strict comparison/source/receipt gates. No new large-main
image scan, archive walk, decoder, build, GPU, model or numerical fixture.

RESULT -> V2 exit0/0.244829s, CPU wrapper rounded1s. Actual S107 host metadata
binding and7296-byte native prefix/metadata/options/codebook/data/relocation
comparison passed against S97. The compiler already combines eight activation
words into TWO d32x4 load sites. Their dynamic traffic, latency and removable
critical-path cost remain UNKNOWN. Whole image/text equality stays false;
missing resource fields stay UNKNOWN. The selected executable image offset
was not saved before V1 failed, so V2 records null rather than rescanning.

VERDICT -> Close this cheap calibration; no new TG17 kernel/fixture/GPU is
selected. TG17 is23.18% of whole-collection primary GU service; dominant
TG21/IQ3_S already has weight reuse across E4/E2/tail entries. This narrow
audit does not establish a larger-family or serving benefit.

`original-proposal/` preserves the incorrect TG17=IQ2_XXS/helper attribution
verbatim. `audit-prep-v1/correction-and-ranking.md` explicitly corrects it to
TG17=IQ2_XS (TG16=IQ2_XXS) using the type-name switch and Fmt mapping; it is
not silently rewritten. `partial-v1/` preserves the saved metadata and an
external binary identity; the actual initial failure is reported by root in
V2 context, with no invented original raw log/handle.

`audit-prep-v1/` and `continuation-v2/` contain exact scripts/pins/status
records and the exact correction diff. `actual-v2/` retains strict comparisons,
raw inventories, selected metadata, findings, actual command/log/exit and
closure manifest. The optional post-audit instruction-offset annotation had
a syntax-only failure before execution, was not rerun, and produced no
derived addresses; source line/loop-label anchors are used instead.

`compiled-operand-excerpts.txt` is a bounded derived excerpt from the retained
S97 selected K-loop window, with original ASM line numbers and provenance.
Full native ASM, main/object/archive and selected binary images remain
external. `record-copies.json` indexes absolute sources/bytes/SHA256 and the
derived excerpt; `external-artifacts.json` reuses known identities without a
new large-artifact read/hash. See the
[dated document](../../../../docs/20261016_strata_routed_gu17_activation_audit.md).
