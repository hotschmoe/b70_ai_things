# Synthetic source35 HC composition arithmetic observations

CONFIG -> fresh leaf links current source35 SDK/archive/eight ABI ELFs, exact
compiler image/options, in-order LevelZero physicalcard0, parent pair health.
COMMAND -> new prepare/compile/qualify_hc_composition35_v1 wrappers only.
RESULT -> CPU controls only. No compile/device/model execution by the agent.
VERDICT -> source-only fixture proposal. Model arithmetic, normal model graph,
universal device intrinsic, quality and speed qualification remain false.

The leaf calls the EXISTING compiled hc_native_read_f32 and
hc_native_write_f32; it does not recreate them as a replacement reference.
Four synthetic cases cover analytic/heldout T1/T2, including cancellation,
subnormal residuals and original-format signed Q8_0 half scales. Each case runs
three applications (none, pending nonalias, pending alias), each directly and
then twice from the SAME captured command graph:36 frames total.
Alias input is reset from the same original synthetic bytes before every run.
Pending and standalone writes use independent copies of identical R/bo/inject.
Graphs retire before owning USM. No model parameters or activation captures are
inputs. Input-only packages are mounted; no expected outputs enter the device.

Eight fields per frame are actual compiled composition buffers:rs,xn,postSiLU
low,gate,inject,mixed,residual after pending and standalone write. PreSiLU is a
SECOND actual compiled projection of the actual synthetic normalized input.
It is explicitly secondary, not a newly invented capture from the existing
composition. Remaining fields are separate shadow kernels for RMS sum/arg/rs,
normalized input, ordinary sycl::exp outputs, SiLU and separate/FMA mix variants.
They never claim to be model-consumed buffers. Source HC uses ordinary
sycl::exp(float); GDN native intrinsics are not substituted.

The collector requires all19 fields/36 frames/3802788 F32 values, actual device
FP capability flags, and current canonical source/input hashes. It compares
actual buffers with shadows, repeated graph versus direct queue execution,
alias versus nonalias, and pending versus standalone writes. Mix candidates
are reported individually; neither is selected as a policy or accuracy gate.
The two negative controls mutate already observed rs/write bytes to prove the
self-consistency checker rejects them. They are reader controls, not extra
device cases. No arbitrary tolerance is assigned.

These observations can locate current source/device exp, rsqrt and contraction
rounding on this corpus. They cannot prove universal intrinsics or normal full
model graph arithmetic: the leaf graph contains HC only, without GDN/FFN/QSA,
model state, model weights or serving context. Independent original-weight HC
control V2 remains separate and receives no captured operand substitutions.

The fresh compile recipe pins all63 source files/27 headers/35 patches/six
Python sources/eight ABI ELFs and current static archives via the existing
source35 SDK gate. The original SDK receipt did not record archive hashes;
fresh archive identity is recorded now, with that historical limit explicit.
Root compiler version is preserved in fresh compile.log. Current qualified host
libm/FMA identity is bound for later host intrinsic-candidate analysis; that
host proof cannot qualify device exp/rsqrt. Device FP flags are observed
capabilities, not evidence that a compiler chose a particular lowering.
An additional separate FMA shadow uses runtime USM operands for gradual output
(minnormal*0.5) and gradual input (minsubnormal*2^24). The two F32 observations
per token cannot be constant folded. They are reported as observed behavior of
this shadow, without inferring a universal device FP mode or accuracy gate.

Root lifecycle wraps strict per-card and compiled P2P0 pair pre/post health,
owned compile/run terminal/removal, actual USM logical-free recollection,
kernel journals, current source/archive/ELF/corpus admission and a NEW
postterminal/posthealth all4 publisher hash bracketed by known pages. Pure
plan/count/source admission precedes leases/health so bad metadata cannot start
device work. Root alone compiles and executes. Old projection/HC/own receipts
and sources are unchanged; no prior success receipt is transferred.
