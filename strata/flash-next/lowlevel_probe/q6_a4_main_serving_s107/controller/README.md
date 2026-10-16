# S107 Q6 A4 clean serving controller preparation

CONFIG -> Frozen S106 clean six-controller derivative, same newly audited future
binary in both arms. Only STRATA_Q6_A4=0/1 differs besides arm-local paths/names.
Binary path and full SHA are mandatory future make_configs.py arguments; no
unknown binary identity is invented or accepted by this preparation.

Both arms: pair graph0, IQ2 multi0, Q6 CENTER0/PACKED0, QSA0/value0, head RPW2=0,
HC common S51, S49 OFF, global RPW1/A2/PF0. Depth2/head-pair ON, spec4/mtp-max-t4,
suffix0/lookup0/spec-min-p0 in native argv and every HTTP request. EP149/363 all
resident, parallel2/max-context200000, four parked conversations/32768MiB,
Direct PLE and prefill12288 are inherited unchanged. TRACE1 only supplies the
prompt-start admission marker; CPU ring/profile/unitrace remain absent.

COMMAND -> SOURCE PREPARATION ONLY. Root full-reads this delta and actual
source/build/image closure before binding or execution. Future order:

1. Fresh pinned per-card and compiled collective pre-health as root determines.
2. bin/cpu-run python3 harness-prep/make_configs.py --exe AUDITED_EXE --sha256 FULL_SHA
3. Arm0 then arm1: bin/cpu-run python3 harness-prep/start_arm.py ARM.
   Release CPU lease at readiness; detached systemd unit keeps pair GPU lease.
4. bin/cpu-run --quiet python3 harness-prep/client.py ARM warm
   One ordered107-prompt/256-output C2 pair, unmeasured for final medians.
5. bin/cpu-run --quiet python3 harness-prep/client.py ARM timed
   Three fresh ordered107-prompt/1500-output C2 pairs; quiet covers only this
   bounded timed phase. After EACH ON pair, exact control parity must pass
   before any next pair or functional request. No normalization or retry.
6. bin/cpu-run python3 harness-prep/client.py ARM functional
7. Always normal-stop after client failure/success using stop_and_health.py:
   owned ready fullCID TERM -> native QUIT, wrapper0, fullCID AND NAME gone,
   final aggregate diagnostic, pinned per-card and compiled post-health.
8. compare.py only after both normal teardown/health closures. Any mismatch
   yields NULL timed summary, no aggregate speed comparison or promotion.

Normal teardown is byte-identical S106 logic. It requires an owned ready CID;
failed startup/abnormal TP termination remains root-owned recovery rather than
an automatic force kill/restart or blind health/collective run. No cleanup
framework or extra generation has been added.

RESULT -> SOURCE ONLY. Configs not generated, scripts not imported/executed,
no client/network/CLI/model/build/GPU work. Only Python AST (including embedded
/proc probe), bash -n, source hash and exact source delta checks performed.
original-inputs.json pins original six controllers plus source/dependency
provenance; controller-diff.patch records all changes from the frozen S106 six.

After warm and each later pair, cumulative A4 markers must equal exactly one
full literal in ON and none in OFF. Final whole log is checked again. This is
first eligible HOST selection only: not both-shape coverage, execution counts,
GPU dispatch or GPU timing. Graph marker must be absent BOTH arms. Effective
native env/argv, executable, image, loader environment/maps and libstdc++ hashes
are retained, as are fresh admission ordering, actual client lifetime overlap,
S16 warm-prefix/fulltext/SSE/usage, cache bounds, per-request draft/accepted work,
paired/solo/window/reported-row/capture parity and existing functionals.

VERDICT -> Frozen for root review and later actual binary binding only. A clean
one-process-per-arm fixed-order serving discriminator remains nested evidence;
no isolated load improvement predicts full-model gain. No new profiling,
counters, oracle, builds, cache campaign, daily restore or automatic rerun.
