# S77: preregistered grouped prefill contract proposal (2026-10-15)

CONFIG -> COMMAND -> RESULT -> VERDICT

CONFIG -> Source-only decision preparation. S42's strict cross-algorithm GU
failure remains FAILED; S62 remains 36 selected normal-scale, CPU-reconstructed
operand observations without GPU half-weight byte identity. Neither is renamed
PASS. No payload/array read, implementation, build, model review, heavy CPU
calculation, GPU or shared repository edit was performed here.

COMMAND -> Read ACTUAL S42 v2 source (f69603f2), its build/strict report, S45
format equivalence, S62 method/class metadata, current Gemm/DPCT/SiLU source,
and official Intel/Khronos contracts. Inspect tool-environment Python capability only, superseded by root target
preflight below; no dot-product trial. Exact small-source pins and references are in source-pins.json.

RESULT: API/precision boundary

Gemm::f16 at gemm.dp.cpp:833-868 uses actual FP16 input/weight storage and
FP32 output, alpha1/beta0. Its DPCT f32 plus mm_default selects unset
(blas_utils.hpp:865-889). S42 v2:237-240 calls the direct variable-shape
oneMKL grouped half/half/float/float API with unset; GU is m1280/n=ne/k2560,
down m2560/n=ne/k640, transA/T and transB/N, per-group size1. Descriptor
arrays must remain distinct/immutable in context through returned completion
events and scratch reuse. The in-order queue plus dependent barriers precede
SiLU and down/scratch reuse. Vendor calls need not map to one GPU kernel.

Intel documents this overload and no alternate modes by default, but does not
supply a per-FMA RNE/denormal theorem for these internal implementations.
The application's precise AOT flags do not constrain already compiled vendor
kernels. unset must be recorded with MKL_BLAS_COMPUTE_MODE absent and actual
library identity fixed; do not call it an explicit standard-mode assertion.
Intel also allows reduction-order/run-to-run changes absent separate CNR.
Strict within-algorithm repeat checks therefore measure a required property;
they are not assumed or repaired by silently enabling CNR. [Intel grouped GEMM](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-dpcpp/2025-0/gemm-batch.html),
[compute modes](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-dpcpp/2024-1/compute-modes.html),
[numerical reproducibility](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-dpcpp/2025-2/numerical-reproducibility.html).

One concrete contract: FULL GU target, complete-wave coherence

candidate-gate.json is the ONE proposed contract, pending root/model review.
Reuse the exact S42 16-expert/10-format wave: 1748 rows, maxne641, capacity4096.
Precompute all CPU RN-half GU/down planes with pinned ggml routines, and FULL
GU FP64 references from S42's actual saved gathered half input. Changed input
flips each stored half sign bit exactly; its ideal GU is the negative initial
reference and has the same absolute-sum bound. No new random-input reproducer,
FP32 requantizer or quantized-GEMM CPU surrogate is needed.

Prefer one full 157,286,400B (100+50MiB) GPU GU/down plane export and exact
CPU/GPU byte comparison over a new selected-row machinery layer. This includes
all 577 known raw-subnormal GU scale blocks and closes operand identity for
this fixture, including previously unobserved rows. It is a new prerequisite,
not existing evidence. All stored GU operands must be finite normal/zero for
the first target; unexpected half subnormals STOP rather than assume vendor
FTZ behavior. S62's reconstructed GU and saved X were normal, but GPU bytes
are unmeasured. Full plane identity does not prove vendor internal arithmetic.

For each of 2,237,440 GU outputs from EACH algorithm, define T64=X64*W64^T,
Ahat=abs(X64)*abs(W64)^T, and an outward Aupper=Ahat/(1-gamma2560,64).
Propose the conventional source-derived FP32 envelope, frozen before collection:
abs(float64(y)-T64) <= (gamma2560,32+gamma2560,64)*Aupper, with all bound/residual
arithmetic enclosed outward. gamma(n,u)=n*u/(1-n*u); u32=2^-24, u64=2^-53.
The reference uncertainty is E64=gamma2560,64*Aupper; keep it explicit
separately from B32=gamma2560,32*Aupper. Aupper0 requires numerical zero. Reuse S62's exact36 dyadic dots as
independent reference sanity; do not choose a threshold from their errors.

Under normal IEEE RNE additions, half products are exact in FP32 (at most22
significant bits), nonzero products lie on2^-48, and K*65504^2 <1.10e13, so
FP32 underflow/overflow does not arise in an ordinary dot. gamma2560,32 is
0.00015261117724262125; gamma2560,64 is about2.8422e-13. These are conventional
arithmetic-model constants, NOT a vendor-promised bound. Full measured outputs
can satisfy this preregistered acceptance target even if the opaque library
operation order is unknown. Do not promote success into a theorem covering
other inputs, hidden precision or all denormal cases. The standard error model
and componentwise |error| <= gammaK*|X|*|W|^T are described in
[Higham/Mary sections3-4](https://eprints.maths.manchester.ac.uk/2841/1/paper_eprint.pdf).

Full reference is affordable in scope, not yet timed

GU is 5,727,846,400 MAC (11.456GFLOP) per reference, not trillions of operations.
Reference plus absolute-sum costs twice that. Reconstruct/stream one expert:
25MiB FP64 GU weights, max12.52MiB FP64 X, max6.26MiB reference, matching
absolute/bound temporaries; full RN planes total150MiB on disk/mmap. A bounded
working-set plan is comfortably below1GiB, without a full model map/copy.
Root actual TARGET host and pinned e41 Python both lack NumPy. The tool-side
Python import observed during this audit is not target capability and must not
be used as an execution dependency. Root preflight confirms existing
/opt/intel/oneapi/mkl/latest/include/mkl_cblas.h, lib/libmkl_rt.so(.3), and icpx.
Use a small CPU-only C++ helper linking CBLAS dgemm and abs-product dgemm,
MKL_NUM_THREADS=4/OMP_NUM_THREADS=4 under cpu-run BEFORE GPU. No installation,
SYCL/LevelZero link or new math is needed. Budget seconds to roughly a minute,
unmeasured; CPU decoding/storage can dominate. No billions of scalar Python
operations or whole-model map is proposed.

Why this is not an ideal full-pipeline accuracy claim

kernels.dp.cpp:1777-1786 uses hf_sat(g/(1+sycl::native::exp(-g))*u).
Khronos makes native-exp domain/accuracy implementation-defined, so a universal
ideal SiLU64 interval cannot be asserted from these headers. [Khronos native math](https://github.khronos.org/SYCL_Reference/iface/native-prec-math-func.html).
The minimal contract keeps this EXACT production producer, verifies identical
H for the SAME saved GU under segmented/packed launches per algorithm, and
requires each algorithm's own eager/repeat/changed captured GU/H/down bytes to
match itself. Inputs, quant blobs, plane contents, IDs, metadata, guards/tails
remain strict. Cross-algorithm H/down differences are retained descriptively.
This establishes a full-GU accuracy target and complete-path implementation
coherence; it does NOT claim ideal SiLU/down error or model qualification.
A full down reference would add 4,474,880 outputs x640 =2,863,923,200 MAC per
algorithm/input because the actual H operands differ; it cannot reuse one
algorithm's H as the other's truth. It is outside this minimal contract.

Performance sequence and decision

Do heavy CPU precomputation first. In one later root-owned quiet/card1 window,
confirm full operand bytes, both initial/changed full GU targets, own complete
pipeline snapshots/changed graphs and immutable metadata before accepting any
timing. S42 never established grouped oneMKL graph capture; unsupported capture
or failed strict own replay is a real STOP, not a gate to remove. Retain the
exact original scalar initial GU/H/down anchor from S42; declare source/library
identity changes before interpreting a failed old-anchor match.

Only then time the COMPLETE one-wave path, including gather, real dequant,
GU, unchanged SiLU/F16 store, down, explicit dependency barriers and actual
per-wave metadata fill. Preallocate slots outside the window, restore canonical
alias input/poison and drain outside samples. Three warm runs/arm followed by
four paired observations AB/BA/BA/AB is a small balanced proposal. Record each
host submit/completed wall/metadata duration; validate own canonical final
GU/H/down after EVERY sample, including readonly weights/metadata. This is not
request prefill throughput or a graph-only GEMM speed claim. Require all four
completed-wall pair deltas favor the candidate and median benefit exceed the
control range as a descriptive prescreen; inconclusive means HOLD, no retry.
Confirmed created CID/name removal precedes post-health/next container; preserve
failure exits and health closure. No per-stage host waits/readbacks in timing.

VERDICT -> GO for review of this bounded contract/preparation only. HOLD any
assertion of vendor-guaranteed IEEE error, ideal full-pipeline accuracy, grouped
serving or progress to2000tok/s from a primitive result. No custom grouped XMX,
native-decode revival, FP16 API substitution, tolerance fitting or new framework
is proposed. Full-reference/plane proof replaces weeks of selected-cell work;
root must approve the exact contract and actual model reviews before any new
adapter, build or GPU execution.
