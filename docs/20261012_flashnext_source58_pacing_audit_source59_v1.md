# Source58 pacing evidence and source59 candidate

CONFIG -> Admitted source58 paired native2 diagnostic OFF/ON run; authentic
P2048 A512 plus C64 after actual A token256, default share0.5, copy0/mirror0,
STRATA_TRACE in both arms. Complete512/64 histories matched each other and
accepted52 serial histories. Both arms had63 actual rows2 events. This is one
diagnostic comparison; budget ledger, cancellation usability and clean
performance remain unqualified.

COMMAND -> Read only the completed public JSON, cohort host-monotonic events,
combined logs and pinned source58 generate.cpp. Match every cohort event to its
original log line in order. Associate each service print with the immediately
preceding contiguous batch-event/BT window(s), and group token gaps by the
service's actual stage/chunk/layer. These are local receive timings and source
reported host service durations, not calibrated GPU or idle measurements.

RESULT -> ON151 service calls: stage0 73/5098.697ms; stage1 78/4161.180ms.
147 calls served one batch window. Four calls at the first eligible layer of
larger chunks served8/2/2/6 windows, producing14 extra burst windows. Their
intra-service receive intervals total697.151ms. The current callback drains
all positive accumulated allowance with a do/while. Early setup credit can
therefore be consumed in a burst, before later router safe points become
eligible. Credit remains source-bound; no runtime ledger is reconstructed.

Of same-stage/chunk inter-service gaps,97 advanced one layer: median138.921ms,
P95 163.489ms. Another38 skipped one layer: median222.451ms, P95 248.396ms.
One three-layer gap was317.145ms. These skipped safe points are a concrete
source-backed target for smoothing, rather than a kernel-throughput finding.

The largest ON gap942.970ms was long token423->424, spanning final incoming
admission and the first rows2 window. The934.159ms gap262->263 preceded the
first eligible service for the first512-row chunk. Pacing at existing router
safe points cannot guarantee either setup/shape-transition duration. No GPU
idle or pure-prefill attribution is made for these whole gaps.

Whole-long P95 worsened148.201->224.102ms while max improved13427.507->942.970ms.
Incoming TTFT improved37111.952->28063.166ms. While incoming was active,
ON long median/P95 were135.618/238.143ms. Faster startup in one arm is not a
clean speed claim.

VERDICT -> Prepare one defaultOFF source59 knob:
STRATA_BATCH_PREFILL_SINGLE_WINDOW=1, with layer cooperation1. Serve at most
one unchanged batch_step at each existing layer ticket, carrying unused credit
to subsequent eligible safe points. Preserve share0.5, pause subtraction,
actual-service debit and original postchunk residual service. The patch changes
only generate.cpp, with no new queue waits, buffers, kernels, chunk/cut changes
or lifecycle handling. It can redistribute burst credit toward skipped points;
it does not guarantee a P95 improvement or a subsecond maximum.

Next matched control: fresh59 single-window0/1 with cooperation1 in both arms,
same A512/C64/defaultshare/copy0/mirror0 and diagnostic trace. Require full
histories/reference, current source/all8/identity/page/health/normal teardown;
report service windows, actual rows counts, P50/P95/max and incoming TTFT.
The ON service prints should have at most one batch window each. Treat subsecond
max and lower P95 as measured targets, with TTFT tradeoff reported, rather than
source guarantees. Existing58 artifacts/source remain unchanged. No compiler,
GPU or model payload execution was performed for this audit/candidate.
