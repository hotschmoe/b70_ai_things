# TP2 independent diagnostics after the strict arm

CONFIG

Prepared root:
/mnt/vm_8tb/b70/results/bang_recurrence_testing_20260910/native5802-tp2-independent-diagnostics-plan

Same d55637 native6717 image, be02 FP8 scales, TP2 MTP3 FULL_DECODE_ONLY,
prefixON 100K, P2P0 and all preserved collective/profile settings. New server
name, output, cache directories and request cache namespaces. Config and mount
bytes match the strict100K arm. The strict arm's failures remain unchanged.

COMMAND

Read server-plan.json and freeze.json. Start only freeze.launch_server_only
after the strict arm has finished teardown/post-health and parent allocates
both GPUs. This is a direct kv_campaign_server.py invocation. Do not wrap it
in gpu-run and do not pass --leased: the existing server acquires both once.
Do not invoke run_arm.py for this diagnostic plan.

After READY and startup identity/profile review, copy only one prepared job
into run/jobs at a time using atomic rename. Preserve its .done status, log,
raw outputs and independent semantic/cache results before proceeding. Order:

1. Raw stop-token observation, one request.
2. Four-client concurrent16 semantic/reuse controls.
3. Tiny24.
4. Early-history3 and its separate strict quality check.
5. Deterministic32 seed42 and its separate strict quality check.
6. Sampled32 seed42 and its separate strict quality check.
7. Final paired startup host-trace review.

The stop-token payload uses a new diagnostic cache salt. Relative to that
namespace-adjusted original wire request, its only change is return_token_ids.
Its exit0 means observation completed, not that the truncated answer passed.

RESULT

CPU preparation checks pass: unchanged serving config, ten ordered manual jobs,
fresh output and source/config/payload hashes. No requests or GPU work occurred.
The only runnable server argv is in freeze.json; there is no automatic campaign
runner or qualification marker creation in this preparation.

Record an aggregate diagnostic table with each exit status, semantic result,
cache result and transport/engine status. An ordinary semantic negative can be
retained while an independent job proceeds under parent supervision. An engine
error, missing completion, unhealthy transport, teardown uncertainty or device
failure requires stopping and reviewing lifecycle/recovery before more probes.

VERDICT

Diagnostic only, not a retry that erases the strict negative, and never a
WORKLOADS_PASSED or promotion. End with STOP, verified owned-container teardown,
strict both-card and compiled P2P0 post-health. The existing pair lifecycle's
reset-on-crash path and parent recovery review remain required. Keep profile
shapes/counts and capacity evidence; no speed or full100K qualification claim.
