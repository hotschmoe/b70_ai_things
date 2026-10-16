# S140: runtime-source limits and adapter discovery HOLD

CONFIG -> Existing S135 mode0 oracle on card0, pinned e41 oneAPI2026.1 runtime,
in_order-only two queues, no enable_profiling property. No new compilation,
adapter force, driver-list knob change or performance measurement. Main MTP
and verifier DPCT profiling-enabled construction is a separate source finding.
The installed June package and current official upstream source are not claimed
byte-identical. Source references and original frozen manifest are retained.

COMMAND -> One finite owned-container run reused the checksum-bound S135 leaf,
its strict full-output/guard/read-only analyzer and DOT checks. Set only discovery
logging: SYCL_UR_TRACE1 and loader/LevelZero INFO loggers. Adapter override and
UR_L0_USE_DRIVER_INORDER_LISTS were explicitly unset. Root attested same boot
and no intervening GPU work after closed S139 post-health, avoiding duplicate
preflight; the exact reuse receipt is preserved. Native lifecycle had no timeout,
forced cleanup, reset or retry. The checker required surviving-adapter load text
in its selected stderr stream, not merely a library mapping.

RESULT -> The native leaf exited0 and passed all three A/B/A phases, eight
depths, complete intermediates/guards/read-only checks and three DOTs. However,
the original discovery checker failed with exit1: stderr contained exactly four
lines, namely default-IP V1 rejection and three queue INFO lines, with no
required surviving-adapter loaded line. Container exit was1; driver exit2 and tee exit0.
That HOLD and traceback remain intact; no comparison.json PASS was synthesized.

Saved stopped-container diagnostics showed non-OOM, no Error and normal wait
and removal; full CID plus NAME were absent. Automatic post-health was correctly
withheld at the failed qualification gate. Root read diagnostics and then
explicitly authorized manual card0 post-health, which exited0. This is a healthy
closed diagnostic HOLD, not an unexpected native crash or a rerun.

Additive posthoc two-stream analysis -> The saved stdout leaf.log contained
loader lines for BOTH adapters and an explicit SYCL_UR_TRACE `Selected device`
followed immediately by platform `Intel(R) oneAPI Unified Runtime over Level-Zero V2`
and B70 device. Root fully read both streams. Combined with V1 default-IP skip,
unforced environment and actual native LevelZero strict PASS, root accepts
selectedV2 for this PRIVATE process. This is stronger than a loaded-DSO inference.
It is new saved-evidence analysis, not retrospective checker/container PASS.
Exact source line numbers and both complete small streams are preserved.

Source result -> DPCT's profiling macro is defined before includes in both MTP
and verifier and can add enable_profiling independently of VERIFY_PROFILE.
Removing it globally would also change the barrier/dummy-kernel branch. Current
upstream describes V2 selection and driver-inorder behavior, while an installed
legacy helper has conditional semantics making unset versus0 potentially equal.
Strings in both adapter DSOs do not prove the V2 queue consumes that knob. The
inspected current upstream SHA is b29c1834fc5795c9748173a1b3ab48e3d75abd28,
later than the installed June compiler package. Primary URLs in the payload are
interpretation references, not an exact installed-source/callgraph certificate.

VERDICT -> Preserve original strict-discovery HOLD and independently record
posthoc selectedV2 evidence for this one process. Main queue properties, actual
main adapter behavior, event/reset causes, overlap and speed remain unmeasured.
No source/setting change, adapter force, compiler experiment or automatic retry
follows. The queue-property test is parked; subsequent source proposals have
no result here. Larger upstream source files and full phase buffers remain
external with their existing manifest/analyzer identities. No new GPU/build/
model invocation or large artifact hash was performed by this curation.

[Compact runtime and discovery evidence](../strata/flash-next/lowlevel_probe/ur_runtime_s140/posthoc-two-stream-proof.json).
