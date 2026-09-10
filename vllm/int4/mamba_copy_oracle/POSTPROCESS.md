# Targeted postprocess boundary oracle

CONFIG -> Immutable7b107d phase+MRV1 image, default TP2-local Mamba dimensions,
FP16conv with six-history window, FP32SSM, page1638400bytes corresponding to
FP8KV block1600. This operator performs no TP collective. Independent one-card
plans exist forcard0/card1. Production PYTORCH_ALLOC_CONF=expandable_segments:True
is required before Torch import; pointerhex and signed-int64 representability
are recorded before actual metadata initialization. No pointer casts/workarounds.

COMMAND -> Prepared only, no GPU run during authoring:

    bin/gpu-run --card 0 python3 vllm/int4/mamba_copy_oracle/postprocess_lifecycle.py /mnt/vm_8tb/b70/results/bang_recurrence_20260910T172335Z/mamba-postprocess-plans/card0/plan.json --run

The selected lifecycle uses exact source snapshots, fresh output/cache,
8GiB/4CPU/420s, devicepin, strictselected pre/posthealth and existing reviewed
owned processgroup/container cleanup. It requires the image-matched prior
7bpairPASS marker. No pair reset is attempted under one lease. A failed
numericalresult remains failure after clean teardown; fullJSON is retained.
The image/source/health/helper hashes and allocator argv are frozen inplans.

RESULT -> CPU test_postprocess_cpu.py runs the actual retained source
run_fused_postprocess helper and scalar body of postprocess_mamba_fused_kernel
across2states*16tiles for all12cases. All decision tuples and accepted outputs
match explicit fixture expectations. SourceSHA936d65... matches installed7b.
No backend import/GPU is used by that test. The actual GPU oracle imports the
installed modules and checks exactworker/modelSHA before calling their method.
Dryrun/parser/source compilation pass; no GPU correctness claim.

Important coverage added beyond the older precopy oracle:

- computed4796,q4,draft3,accepted4: sourcecol2,destinationcol2,bias3;
  conv rows3..5 shift to0..2 IN PLACE, SSM fromtablecol5 tocol2, acceptedout1.
- computed4798,q4,draft3,accepted2/3/4: sourcecol3,destinationcol2,bias1;
  conv rows1..5 to0..4, SSM fromtablecol4 tocol2, acceptedout unchanged2/3/4.
- computed4798 accepted1 and4796 accepted1/2/3: no publication copy.
- q2/q3 self-publication checks biases1/2 and acceptedreset1.
- Packed-page controls duplicate the primary self and backward cases.

The whole backing uint8 buffer is compared to a CPU snapshot-based memmove
reference. This permits source/destination overlap in selfcopy while verifying
all untargeted bytes and padding remain intact. The source-untouched claim of
old precopy fixtures does not apply to overlapping source bytes here. Conv
copy length is six minus bias, not always three. The test separately requires
exact acceptedout (including three inactive -777sentinels), unchanged accepted
input, and distinct acceptedinput/outputstorage. This catches count reset
failures even if state bytes happen to match. No alias race is presumed:
production helper first copies acceptedinput into its separate outputbuffer.

VERDICT -> Ready for parent review/allocation; tests actual postprocess decision,
copy and acceptedreset, not native GDN arithmetic, full scheduler publication,
graph ordering or cancellation. CLI fp16layout is a physical-stride comparison
with the decision block1600 kept fixed; it is NOT an actual FP16block832 runtime
configuration. Default frozenplans use the actual FP8geometry only.

The old precopy oracle failure before its first kernel was a signed pointer
metadata/allocator boundary, not numerical copy failure. This plan carries the
production allocator correction explicitly and fails closed if allocation
addresses remain unrepresentable; it does not reinterpret overflowing pointers.

PRIMARY TP1 PLAN UPDATE

Use postprocess_tp1_plan.json / raw mamba-postprocess-tp1-plan-card0/plan.json
for the current TP1 reproduction. It sets --tp-size1, FP8 page3276800,
conv_dim10240,Vheads48 and retains block1600. Its launch field explicitly
points to postprocess_lifecycle.py and the new plan. The earlier frozenTP2
plans remain unchanged: their oracle_command is POST correctly, but their
inherited advisory launch field mistakenly points to old PRE. Do not invoke
that stale field. The original explicit POST lifecycle command remains valid
for those separate TP2-local geometry controls. No old plan was GPU-run here.

Independent cache agent review reran12actual CPUdecision cases and checked
allTP1/TP2 FP16/FP8 region/page bounds and lifecyclecleanup; no oracle blocker
found. The CPUmemmove fixture now uses deterministic randombytes (instead of
a periodic byte ramp). Actual GPUexpecteddata already used randombytes.
