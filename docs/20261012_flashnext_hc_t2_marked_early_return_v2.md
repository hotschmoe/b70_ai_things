# HC T2 marker V2: successful early stage handoff

CONFIG -> Preserve marked V1 and actual controls65345 FAILED. Raw native
normalexit0, six restored stage frames/three steps passed; GPU1 captured48/48
and submitted/completed3, GPU0 had no capture row, final proof failed1. This
is a marker closure failure, not a rewritten model/hardware/qualification pass.

COMMAND -> Root fresh source preparation:

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=strata/flash-next python3 strata/flash-next/prepare_hc_t2_route_marked_v2.py --output /mnt/vm_8tb/b70/build/hc-t2-paired-mtp-marked-compile-inputs-v2

After preparing, hash its actual compile-recipe.json; root invokes new
build_hc_t2_paired_mtp_v3.py with --compile-root that directory and
--recipe-sha256 ACTUAL_SHA under bin/gpu-run --card0 (spell --card 0).
No author compiler/GPU/model actions; actual fresh receipt/ELF required.

RESULT -> record_window's earlier-stage successful handoff returns at native
verify.cpp1690 before the final-head path's hc_record.finish. V2 adds exactly
hc_record.finish immediately before THAT successful return, after original
handoff copies and optional old batch stamp. The later stage keeps its old
finish. Earlier failed paths do not publish completion. Nothing numerical,
ABI, queue, graph selection, original return/error or GPU command changes.

VERDICT -> One focused CPU control transforms actual source, proves V1 early
return lacks finish, checks V2 both successful paths, and removes the single
new line to recover V1 bytes. Runtime V2/new compiler marker-plan pins select
this corrected assembly; recorded eligibility PLUS actual submitted/completed
counts remain mandatory. Existing original V6 512 history bcaf3f5d stays a
historical comparison target only. MTP ON both natural HC0/HC1 arms; health,
full5, source, real outputs, normal owned EOF/teardown remain unchanged.
