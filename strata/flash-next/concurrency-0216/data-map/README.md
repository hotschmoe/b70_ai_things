# Strata 0.2.16 data map (in progress)

Open index.html locally. The original 2026-10-15 map remains untouched in ../../data-map/. This copy imports the captured C1 operations from ../ledger/b1/summary.json; it does not yet contain a complete census or audited 0.2.16 stage bindings. Per-window figures are total service divided by 82 head events, with edge effects and overlapping work. See ../LEDGER.md and ../JOURNAL.md.

Run `python3 -I build_map.py --import-ops` to re-import C1, then `python3 -I build_map.py --check` to validate the embedded JSON and ASCII.
