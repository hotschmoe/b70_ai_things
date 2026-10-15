S100 v2 finite concurrent decode serving harness - SOURCE PREPARATION ONLY

CONFIG -> New same-binary OFF/ON STRATA_IQ2XS_MULTI0/1 only. Native executable
and SHA are deliberately UNBOUND: no mode/config/serving-input files have been
created. Root must review/apply/build/audit source and bind the new frozen
binary before running make_configs.py. Existing S87 ordered-pair protocol is
reused; actual S98 primitive correctness does not establish a serving gain.
Source marker supplied by routed-iq2-multi-s100-source is ON ONLY:
strata IQ2_XS multi: enabled=1 GU17 LN8 SG32 WG256 N2560 FF640 E4E2tail
Client checks exactly once after the warm pair/capture, never as readiness;
OFF requires no marker. Host selection is not independent GPU dispatch/count.

Controls: VALUE/QSA/Q6CENTER/XMX/ROT0; depth2/headpairON/spec4; C12288/Q0/
FIRST0; HCcommon/globalRPW1/A2/PF0/headRPW2OFF/S49prefetchOFF; explicit LN8
and EXPERT_SPLIT0 select the public routed GU port. Two200k INT8 slots,
all24576 experts resident,32GiB/four-slot RAM parking, exact newNVMe weights
and sibling, DirectPLE, TRACE1 CPU prompt admission, CPU rings/profile ABSENT.
All main/kernel/backend/Python/image identities remain root build/run gates.

COMMAND -> Future ROOT only after source review, build/binding audit, shared
lease coordination and explicit GPU GO. Port18085 localhost, unique containers
strata-s100-mode0/1 and services strata-routed-iq2-multi-s100-mode0/1.
Replace FROZEN_S100_EXECUTABLE/ACTUAL_SHA below with root's actual build result:

cd /mnt/vm_8tb/github/b70_ai_things
P=/mnt/vm_8tb/b70/results/flashnext_ista_gsq_rco_iq3xxs_strata_cachefix_20261014/routed-iq2-multi-s100-prep-v2
bin/cpu-run python3 "$P/harness-prep/make_configs.py" --exe FROZEN_S100_EXECUTABLE --sha256 ACTUAL_SHA

For arm0 first, then arm1, STOP at any failed command; do not blindly run the
next line. Cleanup remains separately available after a client mismatch:

bin/cpu-run python3 "$P/harness-prep/start_arm.py" 0
python3 "$P/harness-prep/client.py" 0 warm
bin/cpu-run --quiet python3 "$P/harness-prep/client.py" 0 timed
python3 "$P/harness-prep/client.py" 0 functional
bin/cpu-run --quiet python3 "$P/harness-prep/stop_and_health.py" 0

Repeat these five commands with arm1 only after arm0 normalQUIT/CID+NAMEgone/
leasewrapper0/bothcard+compiled10 posthealth PASS. After both complete:
bin/cpu-run python3 "$P/harness-prep/compare.py"

No outer gpu-run around clients/stop helper: server launcher holds pair GPU
leases for entire lifetime; stop helper acquires new health leases only AFTER
confirmed fullCID+NAME disappearance and wrapper0. Startup holds normal heavy
CPU lease only until ready. Detached systemd process closes fd7, so startup
CPU lease can release without losing the server's GPU leases. Quiet CPU is
used only for timed clients and stopping/health; unmeasured warm/functionals
are separate. Source tasks do not acquire a GPU waiter or execute any command
above. Root handles applicable prehealth continuity; no redundant automatic
prehealth/daily restore is added. Another session currently owns quiet leases.

RESULT -> PREPARED_NOT_EXECUTED. Per arm: one unmeasured256+256 warm pair,
then three fresh107prompt/1500completion ordered coding pairs, then existing
concurrent math/reasoning/tool roundtrip. Request1 is released only after the
flushed request0 native prompt-start at a fresh log offset while request0 is
still outstanding. Full request/text/SSE-content/usage/nativecache/prompt/
predicted/draft accepted/offered, windows/pair/solo/printedrows/newcapture
layouts match slot-for-slot. No sorting/normalization/retries. Each ON pair
checks its control immediately before sending the next pair; warm also gates
startup capture layouts, so fail/unmatched stops remaining arm workload and
root performs normal cleanup/health. Final comparator independently requires
all four pairs, functionals, final diagnostics, image/effectiveenv and full
normal teardown/health before calculating any three-pair medians.

Before warm/timing, existing native identity probe additionally records actual
/proc/PID/maps + its SHA, whitelisted LD_LIBRARY_PATH/LD_PRELOAD/LD_AUDIT, and
the selected mapped libstdc++ path/realpath/SHA. This is read-only loaded
process inspection, not ldd/preloading, a new helper, engine execution, or a
retrospective S95 clock join. Full maps contain ASLR differences and are not
cross-arm byte-equality gates; selected library identity/loaderenv must agree.
No profiling, six-prefill screen, fullcache/200k, mixed request or daily restore.

VERDICT -> Await actual root build pin + source/controller review + execution
GO. No serving/timing evidence exists yet. Three nested matched pairs are a
bounded screen, not stability qualification. SSE events can contain multiple
tokens and are not token ITL. TRACE CPU cost is common; no GPU-only latency,
physical traffic/occupancy, per-window kernel dispatch or promotion claim.
S87 controllers are retained unchanged at their external historical paths;
this directory changes no runtime source and has no build or payload copies.

V2 DEV-LOOP DELTA -> Warm only uses max_new_tokens256 instead of1500 per
stream. Timed107prompt/1500completion requests, ordering, functionals and health
are unchanged. Generated warm text is checked against the FULL retained S16
text via startswith and actual completion_tokens256; no reference is truncated
by characters and no raw token-ID equality is claimed. Native work/capture
counts are still compared exactly per slot/arm, without fitted relaxation.
256 warm tokens retain positive paired activation and actual capture/use gates;
if these gates fail, stop rather than silently extending/retrying warmup.
The frozen v1 manifestadea6715 remains unchanged. No v2 run evidence exists.
