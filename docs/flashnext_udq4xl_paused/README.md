# Flash-Next UD-Q4_K_XL RAM-spill campaign (PAUSED 2026-10-13)

Evidence from the 2026-10-08 to 2026-10-13 campaign that served Unsloth
UD-Q4_K_XL Flash-Next on two B70s with experts spilled to host RAM, on a
patched Strata fork. Last measured state: about 255 tok/s prefill and
20-22 tok/s decode at ctx 4096, one request, no cache.

Paused by user decision on 2026-10-13 in favour of ~3-bit quants with all
experts in VRAM on stock upstream Strata (65-81 tok/s decode, 1.3-2.0k tok/s
prefill, 2 x 131072 context). See docs/20261013_flashnext_iq3_vs_27b_ab_brief.md
and docs/20261013_flashnext_goals.md.

These files were moved here unchanged from docs/ on 2026-10-14. JOURNAL.md
entries and strata/flash-next/ files that cite docs/2026100x_flashnext_* or
docs/2026101x_flashnext_* now resolve under this directory. Start with:

- 20261008_flashnext_udq4xl_campaign.md (plan and scope)
- 20261011_flashnext_claude_handoff.md
- 20261011_flashnext_steve_sergio_strata_performance_audit.md
- 20261012_flashnext_campaign_completion_checkpoint.md
