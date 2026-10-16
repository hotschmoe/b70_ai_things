# S114 MTP parallel argmax serving controller preparation

CONFIG -> SOURCE ONLY. Minimal derivative of the exact S107 six controllers.
The same future root-audited binary runs both arms. The sole candidate arm
change is STRATA_MTP_PARALLEL_ARGMAX=0/1; only log/container/unit names otherwise
differ. make_configs.py requires --exe and its actual full --sha256 after root
source/build/compiled-image review. No binary identity is invented here.

Common OFF: Q6 A4/CENTER/PACKED, IQ2 multi, value pipeline, QSA tiled prefill,
prompt attention XMX, KV ROT, whole-pair graph, head RPW2 and deferred pair
probability-skip. Existing global WIDE/RPW1/A2/PF0, HC S51 common and S49 OFF
remain unchanged. Native fixed work is suffix0/lookup0/spec4/mtp-max-t4/min-p0;
every HTTP request also sets spec_min_p=0. Depth2/pair-head1, EP149/363 all
resident, parallel2, context200000, prefill12288, Direct PLE, 32768MiB/four
parked conversations are inherited. TRACE1 supplies native prompt-start
admission evidence; CPU ring/profile/eager tracing/commit overlap are absent.
Probability and select remain unchanged on both routes.

COMMAND -> NOT AUTHORIZED OR EXECUTED. Root first full-reads this delta and
actual source/build/image closure; an existing service/lease is preserved.
Only after explicit root execution GO, intended finite order from this folder:

1. Root confirms cards0+1 availability and required pinned per-card/compiled
   collective pre-health, announcing expected duration. No routine qualified
   daily-server restore between arms.
2. bin/cpu-run python3 harness-prep/make_configs.py --exe AUDITED_EXE --sha256 FULL_SHA
3. Arm0 then arm1: bin/cpu-run python3 harness-prep/start_arm.py ARM
   Startup is CPU leased; detached systemd service closes inherited fd7 and
   retains bin/gpu-run pair lease. Release CPU lease at readiness, not after a
   long-lived server. A 1200s observation deadline never authorizes restart.
4. bin/cpu-run --quiet python3 harness-prep/client.py ARM warm
   One ordered C2 pair: 107 prompt tokens and 256 output tokens each. Excluded
   from timed medians. Native request0 prompt-start must be observed while
   request0 remains outstanding before releasing request1; lifetimes overlap.
5. bin/cpu-run --quiet python3 harness-prep/client.py ARM timed
   Three fresh C2 pairs, each 107 prompt and 1500 output tokens per stream,
   cache_n<=64. Quiet CPU lease covers only the finite phase; service already
   owns the GPU lease. ON warm and each ON pair must match control exactly
   before the next phase/pair can send requests. First mismatch stops work.
6. bin/cpu-run python3 harness-prep/client.py ARM functional
   Existing concurrent math, thinking and tool roundtrip; no added generation.
7. bin/cpu-run python3 harness-prep/stop_and_health.py ARM
   Run after client success OR failure. Exact S107 normal teardown: only ready
   owned fullCID, prestop Running=true/OOMKilled=false, TERM to Python server
   then QUIT to native; require wrapper0 and both fullCID and NAME absent.
   Preserve one final aggregate diagnostic before pinned per-card health
   (adc915 image) and compiled collective health. No retry/force kill/reset.
8. bin/cpu-run python3 harness-prep/compare.py
   Only after both normal-stop/health closures. Requires exact card+collective
   health keys, both exits0; partial health JSON cannot close the stage.

On any failed pair, issue no later requests, do not change work/thresholds or
zero/reuse scratch as recovery; attempt the same ready-owned normal teardown.
Failed startup, crashed TP, unknown identity or failed normal exit/health is
root-owned diagnosis/recovery, never automatic force kill/restart or a blind
collective probe. stop_and_health.py requires a ready owned CID and does not
pretend it handles an ambiguous startup. No additional request on observation
timeout. A missing mandatory artifact makes compare abort without a success
summary; a completed mismatch produces FAILED_OR_UNMATCHED with NULL timing.

RESULT -> PREPARATION ONLY. No config/serving-inputs/mode directories generated;
controllers were not imported/executed. No client/network/model CLI, native
binary/archive/model payload reads, GPU access or build. Validation is Python
AST (including embedded /proc probe), bash -n, ASCII, exact original pin/diff
and byte-identical teardown checks. original-inputs.json binds controller and
small dependency/workload bytes; MANIFEST.json binds this new packet.

ON must contain exactly this one cumulative full literal by completion of warm
and in final log; OFF must contain none:

    strata MTP parallel argmax: enabled=1 rows=1 vocab=248320 host-selection-only

It MAY be emitted during startup capture. Marker checks never require emission
after warm or within a specific measured segment. This proves first eligible
HOST selection only, not all shapes, calls, GPU dispatch or device timing.
Graph/IQ2/A4/prob-skip activation markers must remain absent. The old greedy
fallback and probability/select are not inferred from marker presence alone;
root's source/compiled-image review must close that binding before execution.

Exact observable gates retain full output text, emitted SSE content, usage,
per-request cache/prompt/predicted/drafted/accepted work, windows/paired/solo,
printed rounded rows, observed new/startup capture layouts and functionals.
Raw native token IDs and unprinted per-window work are not observed. Native
executable/image/effective configured env+fixed argv, loader env/maps and
actual loaded libstdc++ are retained; this is not an all-environment audit.

Metrics are client wall, TTFT, per-stream decode/whole-request rates, aggregate
pair rates and SSE content-event gaps; SSE events may contain many tokens and
are not token ITL. Final three-pair medians only follow all gates and teardown.
One fixed-order process per arm is a bounded serving discriminator, not a
universal correctness/stability or promotion claim. The S112 private 29.7%
two-owner host-chain result predicts no model throughput improvement.

VERDICT -> Freeze for root review and future actual binary binding only.
If serving gain is tiny like earlier small observations, stop and keep the
flag OFF; no automatic rerun, wider sweep or new arbitrary numerical gate.
Any unmatched gate invalidates speed interpretation and likewise stops here.
No new profiling, probability skip, head/RPW experiment or cache campaign.
