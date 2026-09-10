# Native convolution publication/read diagnostic

CONFIG -> Exact7b/native271db0, productionallocator, TP1FP8hybridpage3276800,
physicalspecstate columns[7,3,9,2], six-history allocated conv, FP32SSM. Uses
unchanged original oracle.py independent rolling-contract reference.

COMMAND -> publication_plan.json contains reviewed owned per-card launch.
Raw plan: bang_recurrence_20260910T172335Z/native-gdn-publication-plan-card0/plan.json.
No launch by author. Requires PRE and POSTcopy PASS, strictpre/posthealth,
selectedlease/pin,8GiB420s, exactsource/nativehash, freshoutput, ownedcleanup.

RESULT -> Original native oracle observed conv_window_exact=false and
inactive_conv_exact=false atstep0, while first-step math and all packed-versus-
hybrid exact checks passed. Originalfailure remains unchanged. It did not save
actualconv tensors, so a new five-call diagnostic is required to identify writes.

The probe saves publication-tensors.pt (CPU tensors only: before/afterconv/SSM,
inputs/weights/output) and four read-acceptedN-tensors.pt snapshots. Its JSON
maps every one of12physicalslots*6convrows to an exact earlierhistory or newQKV
row where possible, records changedbytes and samplevalues, and compares each
specslot's first3rows to the chronological history after thatprefix.

Then four independently restored copies of the actual first-call cache receive
an identical second input, with accepted1/2/3/4. Independent CPUmath compares:
A. worker rolling-column0 window[N-1:N+2], SSMcolumnN-1;
B. per-prefix convcolumnN-1 first3rows, sameSSMcolumnN-1.
It records errors for both; neither hypothesis is substituted as a new backend
qualification gate. Original rolling oracle remains the candidate fix oracle.

VERDICT -> diagnostic_completed/exit0 means all observations collected with
finite results and basic z/padding preservation; it does NOT mean the rolling
worker contract passes or a backend is qualified. Native source-provenance
agent independently reconstructed shipped1e90+r35/r50 source and found old
per-prefix publication/read semantics, with upstream fix5802a414 absent. This
probe tests those semantics in the installedbinary directly. It does not apply
a patch, build an image, alter original expectations, or touch an endpoint.
