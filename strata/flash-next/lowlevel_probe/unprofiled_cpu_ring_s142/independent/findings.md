# S142 independent interpretation: profiler bundle changes CPU brackets

CONFIG -> SAME S133fe57 executable, all-OFF serving flags, nativework policy,
107prompt/64output each fresh orderedC2 pair as S136. CPUring and TRACE1 remain;
S142 removes unitrace injection/control/flush and the associated capture setup.
One instrumented reference and one unprofiled diagnostic, fixed chronological
order; this is not a serving candidate or significance/stability experiment.

COMMAND -> Read existing CPU-comparison.json and both already-parsed small
CPUanalysis.json records, original source bracket windows and closure records.
No parser execution, rawtrace, binary/modelhash, build, GPU or new request.
Root's actual corrected run closed0: start30.464s, collect3.893s, stop59.166s,
analyze0.182s. Earlier nonexecutable-wrapper startup is retained separately;
no native executable launched in that failed startup. Permissionfix was root's
explicit byteidentical chmod-only setupcontinuation, not an unrecorded retry.

RESULT -> All820declared markers present, dropped0, window0=0, explicit final
window_end; full request/text/concatSSE/usage/nativework/nativewindows/paired/
solo/capturelayout gates match S136. Both parsers independently select EXACT
SAME16replaywindow IDs5..16 and19..22. Both fullnormalQUIT/fullhealth closed;
no broad functional campaign was attempted in S142.

| CPU steady-clock bracket, n16 | S136 unitrace median ms | S142 no unitrace median ms |
| --- | ---: | ---: |
| verify_submit | 3.2213425 | 0.067378 |
| pair_submit | 1.9506695 | 0.046123 |
| commit | 2.928889 | 1.585553 |
| stage_PLE | 0.709177 | 0.6686695 |
| pair_poll_and_PLE | 1.0989555 | 2.7100315 |
| pair_queue_wait | 2.9238945 | 2.617585 |
| whole_window (diagnostic only) | 57.1811135 | 46.4703525 |

The verifier-submit ranges are disjoint in this sample:3.062910..3.532445ms
versus0.062059..0.103747ms. Removing the profiler bundle accompanies a3.1539645ms
median reduction in that CPU bracket (97.9084percent). This is strong practical
reason to stop budgeting the instrumented3.221ms as ordinary baseline hostcost.
It identifies a configuration-level perturbation, not an individual LevelZero
API, eventreset implementation, device bubble or GPU3.317ms-gap cause.

Source verify.cpp L3326-3347 brackets peer graph submission, primary graph
submission/errorcheck and existing ext_oneapi_empty, AFTER stage_batch. It is
not merely one API and excludes earlier PLE staging. mtp.cpp L2406-2448 brackets
actual chosen launch path and ext_oneapi_empty; with ownerqueues/pairgraph OFF
this is the existing componentgraph path, not a changed arithmetic kernel.

Crucially, not every interval shrinks. Pairpoll+PLE rises1.611076ms while
submission returns earlier. That bracket (mtp.cpp L2459-2468) contains prefetch
callbacks and waiting for mapped first-depth outputs. It is not pure PLEcompute;
an earlier bracket start can expose more dependency waiting. Do not call it a
PLEregression or add/subtract independent medians to reconstruct a step.

A direct combined interval from each saved window's existing records gives
pair_submit_begin->pair_return median5.978394ms in S136 vs5.370319ms in S142
(ranges5.951828..5.987546 vs5.341534..5.403742ms, n16each). This is computed from
each window's endpoint timestamps before taking a median, not summed medians.
It changes much less than the submit bracket alone. End-to-end behavior absorbs
some earlier return into later polling/waiting. This is CPU-clock observation,
not a GPU execution decomposition or clean serving throughput gain.

Cross-window matched n14: pair_return->nextstage_PLE_begin0.0301475->0.022573ms;
pair_return->nextverify_submit_begin0.759122->0.703531ms. Current unprofiled PLE
staging remains about0.669ms. No CPU/peer/GPU-clock joins were made, and these
numbers cannot be used to allocate S136's LEFT or device uncovered intervals.

VERDICT -> Park queueprofiling-property builds and stop pursuing the traced
3.3ms gap as a demonstrated production host bottleneck. Keep S136 operation
names/shapes/order and source-conditioned service rankings, with explicit
profiler perturbation warning on all timing/readiness interpretation. Use
S142 as the current CPU-bracket diagnostic; prioritize the already-approved
real QSA/kernel experiments. If future tracing is needed, keep it short and
interpret against an unprofiled CPU control rather than optimizing tracing
infrastructure or repeatedly collecting huge traces. No runtime override,
synchronization removal, serving promotion or repeat campaign follows here.
