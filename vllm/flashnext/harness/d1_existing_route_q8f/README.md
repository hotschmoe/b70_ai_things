# Retained-default vs existing route/Q8F: final finite diagnostic pair

CONFIG: both-card single-process EP sidecar, maxlen131072/chunk12288,
mode0/FULL_DECODE_ONLY/captures1,2,3,4,8, no prefix caching/speculation.
Originalc7bFA, plan0, N1off; noFA8, HC_FUSE, P1 or QSA/selector change.
Current retained recipe hasQ8=1 butQ8F/EP_ROUTE absent. Treatment adds only
B70_EP_ROUTE=1 andB70_DECODE_HC_Q8F=1. Both arms install IDENTICAL default-
off one-return observer via private API shim. Observer directory path is
per-run evidence metadata. No library rebuild, new FA or graph behavior.

COMMAND: preflight.py verifies current pinned F1EPc249cf94e and O4HC0377d20b
ELF/schema/symbols. ActualF1EPc exports ep_moe_route/ep_route_dispatch,
contains route schema/device metadata and retained prefill_init support.
No oldO4e substitution is required. B/d1/existing_route_q8f/compatibility.json is CPU-only
ELF/source compatibility, not proof of loaded native-op dispatch.

CPU mock verification (explicit output path required; safe after sourceport):

    python3 vllm/flashnext/harness/d1_existing_route_q8f/cpu_check.py --out /mnt/vm_8tb/b70/build/vllm-flashnext/d1/existing_route_q8f/cpu-check-fresh

RESULT: cpu-check-final/result.json PASS14 scoped checks. Covers defaultoff
withoutTorch import, realspawnAPI shim beforeTorch, kwargs/Tensorcoercion
traps, original return and exception identity, logging failure failopen,
overlapping first calls, restoreonce, capture markers, deadline cleanup,
continuecanonical afterstrictreject, and rejecting shell0 fatalengine or
incomplete/bounded requests. Not a Torch/XPU runtime qualification.

Activation: branch_hook.py observes successful ACTUAL torch.ops calls,
not flags or module entry predicates. Hooks install only after native
library load returns. They inspect NOargs/results/Tensors, add noGPU work,
and emit only internally constructed scalar metadata. Op namespace entry
is restored after its first successful call; timed FULL graph replay is
unchanged. Logging failures preserve original return; native exceptions
re-raise unchanged. A capture_model completion marker (exact real kwargs-
only signature) is startup-only and restores after non-profile capture.
Classifier requires native op_return before full capture_model returned
in SAMEworkerPID. Control requires w8_gemv and forbids fused/route returns;
treatment requires moe_ep_route,w8_down_act,w8_up_collapse. Missing evidence
fails operational qualification, even if the numerical requests finish.
No GPU synchronization is implied by a native host-op return.

Private lifecycle: make_snapshot.py changes ONLY postREADY client block
of current f1_merged_serve.sh; top-levelH/source paths remain explicit.
prepared/serve.diff is reviewable. Server args/loader/lease/startup/health/
teardown/fatal-engine/SO checks remain standard. Source helpers can be
ported byte-identically; generated outputs stay inB/d1, never campaign.
Each arm uses a fresh output with no overwrite, broad dirty-fork/campaign
snapshot and private-helper hashes, actual effectiveSO checks and actual
canonical /server_info config. outcome.py verifies maxlen/chunk/TP/maxseq,
prefixoff/specnull/mode/capture config from the live response, not flags.

Client trim: ONE promo_gate covers identity,323/667,exactrepeat,8K/32Kneedles;
short64token bench; minimalF1 length1024/repeat1/coherence1024/decode0;
canonical d1_lmx_callback ONLY, warmup+3x512reasoning/code. No separate
harnessgreedy block or d1_combined_callback secondrepeat. Client driver
has240s wholebody bound beginning ONLYAFTERREADY, not startup/load.
Each native client owns its existing CPUquiet lock; canonical callback
owns its own lock (no nestedquiet). Host-only5s phase/output/sample counters
preserve progress. A bound expiration records activephase/partialRCs and
kills child process group; it is NOT proof of a deadlock/no-progress stall.
Strict promo reject does not skip later canonical collection. Individual
phase RCs/strictcheck fields remain separate from lifecycle/source/native
closure; a shell0 or partial gate file cannot become operational success.

VERDICT: PREPARED, NOT RUN. Exactcommands proposed ONLY after root grants
each arm, expected~7-9min each including model load/240s bound/health:

    ARM=control TAG=d1-existing-control-r1 bash vllm/flashnext/harness/d1_existing_route_q8f/window.sh
    ARM=routeq8f TAG=d1-existing-routeq8f-r1 bash vllm/flashnext/harness/d1_existing_route_q8f/window.sh

No auto-chain; parent freezes all source across pair and audits control
closure before separately starting treatment. Stablehotschmoe-dd first,
registered O4researchalias retained. Dirtydiagnostic canonical payloads
remain ranking_eligiblefalse/no submission/promotion/claimedbitexactness.

Important numeric evidence: HCcomponentprobe-r2 kept unchangedSO84b64 and
all9cases. Existing fused and new subgroup outputs match each other, but
one adversarial largegate h386 differs from original ONLYsignedzero:
originalBF16=0x0000, fused/new=0x8000. Raw archive bits confirmed with
stdlib (no CPU FTZ ambiguity). Strictbitgate remainsFAIL, timingSKIPPED;
new subgroup speed is UNMEASURED. No kernel correction/rebuild is approved.
This permits only a parent-approved diagnostic serve, not a claim of
universalbitexactness. Prior route weights may differ ulps and tokens;
actual strict request evidence must remain visible in both arms.
