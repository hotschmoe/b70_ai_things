CONFIG -> Isolate actual unchanged HC normalization RS using the identity-bound
saved independently decoded original embedding/residual from firstHC report
076c1e1069de311ee3effa39988f3a78875d8d618804bedc404d31196cec8081.
No native operand or fresh weight payload read enters fixture preparation.
Norm=1 and zero Q8_0 down/up are explicitly synthetic downstream weights.
Actual HcNativeArgs.rs output is written by unchanged SDK37 HC. There is no
injection, pending residual write, GDN, model decode or wholeHC/model claim.

COMMAND -> Root owns fresh compile and every actual GPU touch/lease. Python
native_rms_rsqrt37_proposal_v1.prepare(NEW_INPUTS) validates original saved
report/post4/stat/field SHA and preregisters two argument families and three RS
hypotheses without seeing native RS. It writes a bounded 40960-byte residual
and exact raw hypothesis vectors. Root calls compile_argv() in pinned39992
image with SDK37 mounted /sdk:ro, tracked sources /leaf:ro and NEW build /out.
No model mount is needed for leaf compile or execution. Root runtime uses:

    bin/gpu-run --card 0 ... native-rms-rsqrt37 --residual OWNED_INPUT/residual.f32 --output NEW_OUTPUT

Actual environment ZE_AFFINITY_MASK=0, ONEAPI_DEVICE_SELECTOR=level_zero:gpu;
EAGER absent including0. Parent must separately bind fresh helper ELF, compile
command/stdout/owner rc0/removal, current SDK source/header/object/archive/ELFs,
actual image/compiler/libs, kernel/device and pre/post percard+compiledhealth,
current model/source/pages and NEW postterminal full4. This proposal provides
leaf/source/fixture/math contracts; a qualified runtime parent is still needed.

RESULT -> Source/CPU only, actual native RS unobserved. Source FMA32 lanes with
XOR16/8/4/2/1, F32 mean/epsilon versus frozen legacy F32 square/FP64 mean/F32
mean/epsilon are separate preregistered arguments. For EACH, compare host
sqrtf reciprocal; mathematically rounded F32 reciprocal sqrt using exact
rational midpoint decisions; and legacy FP64 sqrt reciprocal rounded to F32.
No hypothesis is selected by fitting captured normalization or packet bytes.
Native direct queue, graph replay1 and same graph replay2 write actual RS and
normones normalized vectors plus independently compiled sum/argument shadow.
Shadow is NOT a witness of actual HC's unexported internal argument.

Actual build.ninja HC object uses -O3 -DNDEBUG -std=c++20 -fsycl,
-fsycl-default-sub-group-size=32 -fsycl-device-code-split=per_kernel and
-fp-model=precise with SDK defines. Actual rules.ninja uses icpx2026.1; actual
production strata link stanza1136/1138 includes correctly-rounded-divide-sqrt
backend flag separately from HC object compile flags. Fresh leaf combines
compile and link, using that same production device-link flag; this is not a
claim that the HC object FLAGS contain it. The GGML
C-only compile_commands.json is not HC compile authority. Same pinned static
archives supply native HC; the leaf does not rebuild/change its math kernel.

VERDICT -> Direct/replay raw equality and known-negative checks are necessary,
but grant no model arithmetic, normal-model graph or speed qualification.
CPP uses fresh owned USM, poison outputs, explicit queue waits, graph retirement
before USM frees. Native RS compares to ALL preregistered hypotheses with exact
bytes/ULP descriptions, no inferred tolerance or backend math modification.
