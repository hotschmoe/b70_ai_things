# S106 source-only serving controller preparation

CONFIG -> Same future audited S106 executable in both fresh arms. Only
STRATA_BATCH_MTP_PAIR_GRAPH=0/1 differs besides arm-local logs/names. Depth2,
PAIR_HEAD=1, EP149/363, INT8 200000 context, Direct PLE, 32768MiB/four parked
conversations, loader/map provenance and stable model identity are inherited
from the accepted S100 six-controller loop. IQ2 multi=0, Q6 center/packed=0,
head RPW2=0, commit-overlap absent and other inherited candidates OFF. The
separate S104 Q6 A4 private leaf is not integrated. No Q6 A4 runtime flag exists.

BOTH arms explicitly use --suffix-draft 0 --lookup-chain 0 --spec 4
--mtp-max-t 4 --spec-min-p 0. Every request (coding, concurrent math, thinking,
tool first/final) explicitly passes strata_tune.spec_min_p=0. This is a NEW
fixed-work configuration, not an S100 speed comparison. Native process argv is
checked from /proc as well as the generated config. Old S16 text supplies only
a deterministic greedy warm-prefix correctness reference, never timing/work.

COMMAND -> Preparation only. Root binds --exe/--sha256 AFTER its actual build
and compiled/source review; no binary identity is invented here. Future order:

1. Existing per-card and compiled collective pre-health, pinned runtime image.
2. Under bin/cpu-run, run make_configs.py with audited binary path and SHA.
3. For arm0 then arm1: bin/cpu-run python3 start_arm.py ARM. Release CPU lease
   once ready; detached systemd/gpu-run keeps the serving pair lease only.
4. bin/cpu-run --quiet python3 client.py ARM warm (one 107/256 C2 pair).
5. bin/cpu-run --quiet python3 client.py ARM timed (three fresh 107/1500 C2
   pairs). Current S100 implementation deliberately holds quiet across these
   three timed pairs, not across the long-lived server.
6. Release quiet CPU lease; bin/cpu-run python3 client.py ARM functional.
7. Always normal-stop using stop_and_health.py, including after failed client
   gates; it sends TERM to owned Python -> native QUIT, waits for normal exit,
   proves full CID AND NAME gone, retains diagnostics, runs pinned full health.
8. Compare only after both normal stops and health. No automatic retries,
   restarts, force-kill, arm continuation after ON mismatch, or promotion.

The client preserves the fresh flushed native prompt-start admission barrier,
actual client lifetime overlap, fresh prompt/cache bounds, full output/SSE,
usage/native work, paired/solo/window counts, reported rounded rows and observed
verify-capture logs. ON parity is checked immediately after each pair, before
any next ON request. Thinking/tool/math checks and final QUIT/health are retained.

RESULT -> SOURCE ONLY. Controllers have not generated configs or run. Python
AST (including embedded /proc probe) and bash -n are the only executable syntax
checks; no imports of controllers, serving requests, builds, GPU or Claude CLI.

VERDICT -> Ready for root binding/review, conditional on actual built identity.
No latency, correctness, stability, L0 submission or key-coverage result exists.

The ON once-marker must exist exactly once after warm; OFF has zero. It records
ONLY the first successful eligible exact (T0,T1) host capture key, persisted in
each pair record. Conservative production bounds are narrowed to 1..3 by the
actual batch_depth2 caller proof in caller-key-evidence.txt. This does not log
all keys/cache hits, GPU dispatch, or all max_steps==2 admissions. No new native
logging was added. A source proof bounds reachable keys; actual subset remains
unknown. The separate submission-count proposal addresses actual main L0 calls
without injecting profiling into this clean throughput loop.

Files are copied/adapted directly from routed-iq2-multi-s100-prep-v2. The exact
small delta is controller-diff.patch; stop_and_health.py is byte-identical.
original-inputs.json pins six originals and bounded relevant source dependencies.
Do not modify the already-frozen paired-whole-graph-source-prep.
