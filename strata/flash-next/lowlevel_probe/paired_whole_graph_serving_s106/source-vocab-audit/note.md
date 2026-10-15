# S106 bounded S95 vocabulary-head source audit

CONFIG -> Read-only current source plus retained S95 facts/CSV/native log.
S95 binary is d94e96ca; no new trace parse, GPU/model/build or ISA work. Exact
source/derived paths, bytes and SHA256 are in pins.json. Current verify.cpp,
generate.cpp and native_mmvq.dp.cpp hashes equal retained S94/S95 source pins.
MTP now contains the S106 addition; its original OFF tail/calls remain intact.

RESULT -> The reviewed steady motif has THREE vocabulary heads: C2, C2, C6.
It does NOT have three C6 heads per cycle. The whole-profile discrepancy is
24 GPU C6 heads versus22 batch S6 CPU windows; it is not two extras per cycle.
No phantom repeated C6 head should be queued as an optimization on that reading.

Source-supported mapping:

- MtpPair::capture_tail(j), mtp.cpp2145-2166, quantizes the two owners' sample
  rows separately then explicitly calls native_mmvq_head(..., ncols=2). It
  runs each owner's post-head and mtp_select. The original depth2 caller launches
  tail0 and tail1 once each: two C2 heads per paired draft. The42 observed C2
  heads therefore reconcile with21 native paired calls times2 steps.
- Verifier::capture_batch at verify.cpp2980-3003 records record_window(S).
  record_window at1597-1683 records one head over T columns, including
  native_mmvq_head(..., T) at1665. Thus a six-row target batch gives one C6;
  a three-row tail gives one C3. This agrees with CPU22S6+2S3 and GPU2C3.
- native_mmvq.dp.cpp1977-2009 registers distinct native_mmvq_wide<WideQ5K,
  dpct_kernel_scalar<NCOLS>> kernels and selects NCOLS from actual ncols.
  The CSV names encode6 and2 separately. Both show SIMD32 grid62080, local
  {32;4;1}; grid depends on output rows, so equal grids do not mean equal
  columns. RPW1 gives248320 output rows (=62080*4). Input N is a caller
  geometry value, not independently encoded in this grid. Head-RPW2 remains
  OFF; its explicit2/6 branch at3810-3830 does not alias C2 into C6.

The two whole-profile C6 residuals now have a concrete aggregate accounting
candidate. The retained native segment explicitly logs TWO successful
'read6tokens(windows)' prompt-tail passes (actual spelling has spaces).
Actual S95 argv is --spec4, no --suffix-draft or --mtp-max-t override.
The default suffix_draft=3 (generate.cpp716) triggers2355-2357:
mtp_max_t=4, effective spec=6. S at8937 therefore equals6. The ordinary
read_windows loop10605-10628 calls ver.run(T=min(S,b-q)); six-token passes
thus execute one T6 verifier window each. These calls are outside
run_slot_rows' batch host-boundary markers (verify.cpp3318-3323).

Crucially, read_windows' NoHeadSampling only changes host-side resampling;
record_window still computes full vocabulary logits/argmax. The native log
reports two six-token window reads, and22 batch C6 +2 ordinary prompt T6
heads equals the observed24 C6 total. This is source/native-log/count-level
reconciliation, not a CPU/GPU timestamp join or exact event-ID attribution.
It does not establish which two GPU entries correspond to those prompt calls,
exclude all alternative callers/replays independently, or justify rewriting
historical unresolved-event evidence. Capturing/finalizing and Verifier::warm
are not shown to execute a dummy head; no dummy/capture execution is inferred.

VERDICT -> Steady three-head motif is explained by two MTP C2 heads plus one
six-row target head. Whole-profile residual is consistent with two explicitly
logged ordinary prompt T6 reads. Exact per-event ownership remains unresolved;
no new counter/framework or special profiling run is selected to chase it.

ONE critical-path question for a next already-planned bounded device profile:
after S106 coarsening, what latency remains from the FIRST C2 head completion,
through both owners' post-head/select dependencies, to the dependent j1 MTP
step, versus that C2 head's own service? That distinguishes a head-kernel
lever from a data-readiness/selector lever using one GPU's operation order.
The S95 three-head4.664113ms service sum is diagnostic, not attainable savings,
physical idle, a cross-card budget, or evidence to repeat rejected head RPW2.
