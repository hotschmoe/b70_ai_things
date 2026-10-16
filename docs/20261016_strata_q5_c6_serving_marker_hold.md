# Q5 C6 serving marker HOLD: S134 (2026-10-16)

S134 stopped at the ON warm marker gate. The intended C6 process had the
correct new executable and effective C2=0/C6=01010101 environment, but its
native log emitted the old C2 HOST marker. There is no completed OFF/ON
comparison or ON timed result. Comparison metrics are NULL, not zero.
Cause remains UNKNOWN pending a separate source/host diagnosis; this marker
assertion establishes neither a numeric failure nor actual C2 GPU execution.

CONFIG -> Fresh control and ON processes both used S133 main
fe57222eea163dd7396f2bdabb95d2c18326a729c4d7919d400a9614f42d47ee.
Only intended algorithm difference was STRATA_Q5_HEAD_RUNTIME_ONES_C6
0/01010101. C2 word0, ownerqueues0, parallelargmax0 and other experiment
flags stayed OFF; fixed policy/EP149+363/two200k/depth2/headpair1 and existing
native work gates were retained. The actual ON /proc identity confirms this
EXE hash and effective environment, not merely the requested config.

COMMAND -> Root ran the frozen six controllers. Prehealth was explicitly
reused from last healthy S131 both-card/compiled closure and S132 card0
closure, with no intervening GPU/stack change; S133 was CPU-only. The reuse
receipt pins those records. This avoids a redundant precheck, not either
arm's full normal posthealth.

The control completed its107-prompt/256-output C2 warm, three fresh107-prompt/
1500-output C2 pairs and existing concurrent math/thinking/tool functionals.
All control phases exited0 and normalstop+both-card/compiled posthealth
closed0 in59.303215s. Its raw pair records/metrics are retained as control-only
descriptions; they are not a speed comparison without a qualified ON arm.

RESULT -> ON warm returned two client response objects and an engine segment,
then client.py:80 raised:

    AssertionError: C2 runtime ones unexpectedly ON

The condition observes a log literal. At native engine log line86:

    strata Q5 head runtime ones: enabled=1 word=01010101 N=2560 M=248320 C=2 RPW=1 host-selection-only

That is the old C2 literal, not the required distinct `runtime ones C6` literal.
The assertion occurred before writing ON warm pair.json or calling immediate
control-parity checks. Therefore warm cross-arm equality is not qualified by
this stage. No ON timed pairs or functional requests were sent; no comparator
ran and no comparison.json was manufactured. Existing raw responses stay
external as failure evidence, not retroactively accepted numerical results.

| Closed event | Actual result |
| --- | --- |
|Control warm/timed/functional |All three phase exits0 |
|ON warm |Exit1,7.030741s at marker assertion |
|ON timed/functional |Not run |
|Root lifecycle helper |Terminal2, exception=null; preserves failed phase |
|ON normalstop+posthealth |Exit0,57.635691s |
|Both arm lifecycle/health |NormalQUIT, running/not-OOM prestop, wrapper0, fullCID+NAMEgone, final aggregate and per-card+compiled health0 |
|OFF/ON timing/comparison |NULL; no completed comparator output |

Cleanup remained available after the failed client. The helper performed
normalTERM->nativeQUIT, captured the final aggregate before health, and
preserved the failure rather than sending later requests. No source change,
automatic retry, reset, forced container stop or daily-server restore occurred
at this record's cutoff. Posthealth success closes teardown only; it does not
convert the failed marker/qualification gate into a pass.

VERDICT -> HOLD / FAILED_MARKER_NULL_COMPARISON. Preserve the correct whole-
EXE/effective-env identity alongside the conflicting old marker. Do not
silently accept that marker as C6, replace the expected literal, infer ISA or
FP equivalence, claim a zero-percent result or extrapolate S132 private timing.
The separate diagnosis is pending root; no cause is asserted here. User's
optional CPU-certification deferral remains distinct from the strict runtime
selection/work gates that stopped this experiment.

Additive root observation -> A narrow read of native object9dc9 reported both
C2/C6 logging blocks referencing the same C2 rodata string/length99, while
the parsers have distinct environment-string offsets/exact0-off semantics
and the dispatch guards are distinct. This is a preliminary host-logging
fact, not proof C2 ran or a completed root-cause explanation. The separate
full evidence/CPU reproducer was still pending at this curation cutoff.

The payload keeps exact six controllers/delta/manifests, configs/actual
identity projections, control warm/three pair records, ON warm traceback/
segment/admission evidence, engine prefix/marker excerpts, reused-prehealth
and all final normal closure/removal/aggregate/health receipts. Raw SSE/full
maps/binaries/model/large source remain external. A derived HOLD record is
labeled as curation, not an actual comparator result. Payload: strata/flash-next/lowlevel_probe/q5_c6_serving_s134/.

Later diagnosis is recorded separately in S137: a fresh tiny CPU compilation
reproduced the duplicated C2 marker, and a unique C6 static identifier
corrected it in that reduction. This does not qualify this failed run.
