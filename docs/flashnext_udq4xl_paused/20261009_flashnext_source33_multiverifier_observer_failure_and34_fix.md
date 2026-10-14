# Source33 multi-window observer failure and source34 repair

CONFIG -> Exact source33 SDK0b32d570/C111 completed onecard baseline,
P30V2 onecard2048/prefill64, SFD activations and hostinput33 ON in both arms,
P30 OFF first. Current PLE32 semantic gather enabled. Original model/source
identities fixed. This is an instrumented diagnostic, not a latency run.

COMMAND -> qualify_prefix_residual30_v2.py with genuine C111/P30V2 plan and
root-owned pair lease. Inspect actual merged producer log and terminal state;
retain wrapper-owned posthealth and new four-shard publisher hash scan.

RESULT -> FAIL551s. Prefix1 request1 published final pos0/T1 and completed.
Prefix2 request2 ran an earlier verifier prompt window pos0/T1, then final
pos1/T1. Source33 published both using request2 epoch4294967298, and threw
PLE_INPUT33 duplicate currentordinal after the second publication. Engine
exit139, no OOM; child removed. No second P30 arm ran. No successful input33
or P30 numerical/lifecycle qualification. Parent retained failure artifacts,
owned terminal, strict/compiled posthealth PASS, kernel-fault gate PASS and
a NEW complete four-shard publisher hash PASS. The source remains unchanged.

SOURCE -> Input33 collector/checker already require T1 and pos=prefix_length-1.
Source33's publication call did not enforce that final-window selection. Short
prompts can run earlier windows through Verifier.run, so one-frame-per-request
bookkeeping correctly rejected the wrong publication roster. No model math,
ngram hash, table decoder or source32 gather change is justified by this failure.

REPAIR -> New patch0034 observes only active final accepted prompt windows
(pos0+T equals current accepted IDs size). Earlier windows still execute their
original source32 gather/fences and graph launch; empty publication handles
suppress observer submit markers. Frozen helper33 preflight/quota and defaultOFF
path are unchanged. All source33/C111/P30V2 files and receipts are preserved.

VERDICT -> Diagnostic source/runtime bug localized and repaired in new source34
prototype.11 CPU controls plus pristine34patch reconstruction63files27headers
PASS. New C112 source admission85CPU PASS, all prior source/lifecycle gates
plus final_window_PLE_observer_fix34=true and EAGER-absent graph route. Fresh
source34 eight-target SDK is now compiling under the pair lease (session9860).
New SDK/upload/C112/P30V3 actual replay remains mandatory. C111 source33 baseline
still has bounded observerOFF service evidence; it is not original fullmath
proof and cannot qualify changed source34 runtime. Full campaign remains active.

Failure evidence directory:
/mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f14-source33-20261009/p30v2-onecard-run/.
Actual producer log: child/p30_off/engine.combined.log.
Terminal parent receipt: parent-qualification.json.
Source34 recipe: strata/flash-next/current-ple-lastwindow34-engine-build-plan-v1.json,
SHA89ba4019dd5b9b05ed6cc2f61fcf776e35fcd97824743582aac1a42f1d419338.

Follow-up -> source34 SDKPASS309s/all8 targets and actualC112sourcegatePASS;
newlinked source390 oraclePASS47s. Newupload43045 currentlylive underpairlease.
Actual source34 GPU/input replay remains unqualified until newupload/C112/P30V3
full completion. Source34 run evidence root is f15-source34-20261009 under
/mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/.
