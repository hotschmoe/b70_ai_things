# Current PLE staging before device-planned verifier graphs

CONFIG -> Fresh29 serial source path, native originalPLE, owning stage mirror,
STRATA_VERIFY_NO_HOST=1, actual V9 prefix1 expert tiers[2,2,1,2,1,2,2,2,2,2].
COMMAND -> python3 strata/flash-next/test_ple_nohost_staging32_cpu_v1.py
RESULT -> Nine CPU source/ordering controls pass. Fresh source29+30+31+32 applies
with62 source fingerprints,26 added-header payloads, six runtime Python sources
and eight inherited build targets. No compiler/GPU/model payload was invoked.
VERDICT -> Concrete missing host staging is corrected in new source only. Actual
native h_ple bytes and contribution to the first layer1 difference are unobserved.

The immutable failed semantic source is preserved. In old Verifier::run,
verify.cpp2056-2066 computes PLE rowIDs, then2090 submits the graph. Host gather
only occurs at2122 when ar_on(), or2215 in the loop guarded by !no_host. ar_on()
requires all_resident_&&!ar_off_ (consumed SYCL verify.hpp360), while owning mirror
planning enables device_plan_&&!all_resident_ (verify.cpp698). Actual V9 config
no_host=1 and its tier2 frames establish this route: neither gather executes.
Layer1 pre copies m_ple_->ple_ at900 without an AR flag wait. Its actual bytes
could be old/uninitialized; zero contents have not been observed.

The related device_plan_&&!ar_on()/no_host=0 route may race as well: layer0 uses
wait_flag_ge_or(...skip_) at1522/1531/1543 and can bypass all host waits while
its CPU loop still gathers later. The minimal corrected predicate is therefore

    do_ple && !ar_on() && (device_plan_ || no_host)

Current rows are gathered into the existing private h_ple_ before graph submission,
with host seq_cst and sfence publication. A source boolean ple_precollected prevents
double gather in the legacy host loop. Warm graph replay still uses the current
mutable host image. Failure propagates before graph launch. All-resident AR keeps
its existing later gather plus flag wait; ordinary doorbell/no_device_plan keeps
its existing later gather. Stages excluding layer1 never gather. No new runtime
flag, weight/math/reset/cache policy, GPU allocation or graph geometry is added.

CPU controls cover all eight actual window sizes, doPLE/noHost/AR/mirror branch
matrix, configured two-stage ownership, prelaunch errors, warm image replacement,
legacy order preservation and old absent-gather negative. They are source/host
schedule controls, not compiled SYCL, GPU producer or inference qualification.
Root's independent audit finds stage_batch already gathers before batch launch;
this patch changes only ordinary Verifier::run, not the separate batch route.

Metadata clarification supersedes the uncertainty in the earlier PLE audit:
qwen4exp.ple.eos_token_id is248044 explicitly; tokenizer EOS248046 is a different
metadata role. Original GGML uses the PLE-specific value. Both native and frozen
owned PLE use248044, so no padding sentinel correction is warranted. Root's
separate readonly original/norm-pack receipt establishes CPU norm bytes match
original GGUF; actual GPU norm upload remains unobserved.

Required gates before actual corrected-model claims: new matched SDK/390/C1 and
source receipts; actual PLE staging/current input witness; off/on diagnostic
output equivalence; own original prefix residual/head comparison; correct-model
serial-vs-batch and complete-state/session/concurrent fidelity. Existing old29
numerical/API source/lifecycle positives do not transfer to corrected32 math.
