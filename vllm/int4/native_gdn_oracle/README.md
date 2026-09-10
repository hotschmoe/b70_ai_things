Measured finding: see RESULT.md and result_summary.json. The shipped native
uses per-prefix three-row conv state; the worker copy contract expects a
rolling six-row window. Publication/read probes confirm the mismatch. Earlier
preparation notes below are preserved; no corrected backend is qualified.

# Native GDN arithmetic and stride oracle

CONFIG -> Exact phase+MRV1 image7b107d, installed _xpu_C SHA271db0d4 (full values
in oracle.py), FP16 projected inputs/conv, FP32SSM, global K16/V48,dim128.
Default TP2 LOCAL shard K8/V24 on one physical card; no collective.
Spec q4/MTP3 six-history conv state. Default128 calls with accepted1,2,3,4.

COMMAND -> `python3 oracle.py` prints the plan without importing Torch.
Inner GPU argv for a reviewed selected-card lifecycle, NOT a standalone launch:
`python3 /candidate/oracle.py --run-xpu --tp-size 2 --kv-layout fp8 --steps 128 --accepted-pattern 1,2,3,4 --output /out/result.json`
The outer lifecycle must enforce the selected lease, ZE_AFFINITY_MASK,
image/source identity,8GiB/420s limit, strictpre/post health and ownedcleanup.
No graph capture, model loading or backend mutation is part of this runner.

RESULT -> CPU-only exact-image test_cpu.py passes8reference cases and4Torch
interleaved-view layouts; the native operator's29argument ABI matches the
actual call. Installed full gdn_attention exists; separate spec_decode does
not. Native library/schema query and owned CPUcontainer cleanup are recorded
under bang_recurrence_20260910T172335Z/native-gdn-oracle. NoGPUresult yet.

The independent CPUreference uses explicit chronological width4 convolution,
SiLU and FP16 postconv rounding, followed by FP32 Q/K normalization and delta
recurrence. The rolling six-history conv lives ONLY in column0. Before a call,
acceptedN selects conv rows[N-1:N+2], and SSM tablecolumnN-1. Afterq4, conv
becomes old selected window's last2rows followed by the4new projectedQKVrows;
SSM publishes one prefix per tablecolumn. This contract agrees with actual
worker get_conv_copy_spec and retained compatible kernel source a397c58e:
causal_conv1d.hpp:674,794; gated_delta_rule.hpp:391,509. That source is NOT
claimed to be the source of the installed native binary. Operator execution
will test its contract directly. Do not use old Steve separate-spec scripts.

Every call compares independent long-running reference and local one-step
reference seeded from the native pre-call snapshot. Packed and hybrid-layout
calls have identical logicalinputs and shapes. Layout arithmetic must be
byte-exact for out/z/conv/SSM. Untouchedstate/pages and convrollingwindow must
be exact. Output/state numerical gates are frozen maxabs0.005 ANDrelativeL2
0.03, both finite. These are proposed bounded random-fixture gates, not a
claim of previously measured native precision; failures preserve metrics and
must not be hidden by loosening tolerances. z is exact unpacked input.
The run stops at firstfailed step and writes fullresultJSON with nonzeroexit.

VERDICT -> CPUchecked direct-native runner ready for external lifecycle/review,
not yet GPUqualified. It tests arithmetic, acceptedprefix selection and state
strides. It does NOT execute actual Mamba precopy/postcopy kernels, scheduler
publication, cancellation, GEMM, model generation, graph or TPcommunication.
Use the existing Mamba-copy oracle for actual bytecopy first; compose its
metadata/kernel with this operator only as a separately reviewed next test.

The new boundary hypothesis is separately CPUchecked in
state-lifecycle-review/boundary_contract.py: actual Pythonpreprocess and actual
Triton scalarpostprocess across128steps,4acceptpatterns and boundaries1600,
4800,9600,24000 preserve symbolicacceptedstate and boundarycheckpoint. This
rules out no physicalkernel, aliasing or ordering failure; no new sourcebug
was proved by those tests.

Owned TP1 lifecycle prepared after copy-oracle results

The primary frozen plan is plan.json / raw native-gdn-tp1-plan-card0/plan.json.
Its launch field is the exact command for parent allocation; no launch here.
The lifecycle requires PRE allocator-adapter OUTCOME.passed and targetedPOST
TP1 OUTCOME.passed, exact image/native/source/health identities, selectedlease,
productionallocator, freshcache/output,8GiB420s, strictpre/posthealth and owned
cleanup. It preserves completed-step JSON on numerical failure and keeps
failure status despite healthy teardown. No pair reset underonelease.

Actual parameters use FP16dt_bias, FP32A_log, FP16convweights/projections and
FP32SSM. A prelaunch review corrected fixture dt_bias from FP32 to FP16; actual
compatible kernel source explicitly requires it to match input/outputdtype.
Exact-image CPUtestv2 passes actual parameter-builder dtypes, referencecases,
Torchview shapes and29argument native ABI. Both CPUcontainers were removed.

Nonzero/permuted physical table is [7,3,9,2], included in result and lifecycle
gate. TP1 packed state stride3268608 versus actual FP8hybrid3276800bytes;
conv and SSM each retain their own fullpage stride. All12checks are frozen in
plan. Scalar .005absolute/.03relative gates are distinct from exact
packed-versus-hybrid comparison; numerical tolerances never relax stride,
z, convwindow, padding or inactive-state byte checks.

Compatible retained source a397c58e explicitly passes ssm_state.stride(0)
(gated_delta_rule.hpp:698) to specinitialpointer(:397) and finalpointer(:514)
with int64physical index multiply. Conv similarly reads stride(0)(:1032) and
uses it for specstate column0(:684). This source does not itself demonstrate
a remaining stride bug, and is not proven to match the installed native SHA.
The numerical oracle can test the actual binary independently.
