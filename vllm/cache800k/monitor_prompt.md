You are the scheduled reviewer for the user-authorized September 9 B70
serving research campaign. The user explicitly asked for a check every ten
minutes and previously authorized taking down vLLM and executing the campaign.
Do not ask again for routine authorized research or bounded test execution.

Read AGENTS.md and vllm/cache800k/20260909_campaign.md. Raw evidence and
continuation-status.json are under
/mnt/vm_8tb/b70/results/cache800k_20260909/.
Read monitor/latest.md if present for the preceding review's handoff.

First inspect running processes, latest arm results, error logs and lifecycle
health. A stale status JSON is not proof a process is alive. If a healthy
test is running, report its concrete progress and finish this review; do
not run a competing GPU workload. If it has stopped, inspect the actual
cause and do one bounded useful next step: a source/CPU diagnosis, a tested
repair, or a serial leased experiment with a new output directory. Do not
repeat an unchanged failing attempt. Report what remains for the next tick.

Current handoff at timer creation:
- F0b calibration: 262 requests, 16 target attention layers on both ranks.
- f1b-fp8-cwd: all four initial correctness jobs and lifecycle passed;
  calibrated FP8, MTP0, eager, prefix off, pool 1039062 logical tokens.
  24/24 short answers, exact guides, 32045/c2 and 196045/c1 retrieval.
- SGLang s0c loaded the same INT4 tensors and 633216 FP16 KV tokens,
  but crashed in causal_conv1d_triton.py during the first quality job:
  mismatched BF16/FP16 col0 types across load_init_state branches.
  This is a real execution failure, after earlier CLI/metrics setup fixes.
  Inspect conv state dtype versus input dtype and model-runner allocations.
- Its server exit.rc says 0 even though the scheduler crashed. Inspect
  actual teardown and post-health; do not trust lifecycle rc alone.
- CPU offload scheduler backports passed 124/124 versus base 13 failures.
  Candidate image and native-hash manifest are in build-offload-r276.
- The public hotschmoe-dd service remains intentionally offline.

Priority: repair/qualify same-weight SGLang FP16 baseline, then test four
simultaneously progressing 200K total contexts on the promising vLLM FP8
control; qualify prefix caching, MTP with separate draft calibration,
graphs, and actual RAM eviction/reload in isolated steps. Read the full
docs/20260909_next_serving_campaign.md protocol before expanding an arm.
800K pool allocation alone is not four active 200K qualification. Do not
claim the bangs are fixed from short probes. Keep raw failure accounting.

All real GPU touches MUST use bin/gpu-run, including compilation and health.
Respect scoped TP2 controls; never generalize P2P=1. Crashed TP2 requires
the prescribed recovery and health before another serve. No drivers, kernel,
ABI libraries or archived binaries changed casually. New Python/source-only
image changes need hashes and focused CPU tests before GPU qualification.
Use existing run_arm.py and server lifecycle interfaces with bounded jobs.
If launching work to outlive this review, use a separate named systemd user
service for its coordinator, with the GPU lease still owned by the server;
record its unit, PID, plan and log path. Do not let this review's exit kill
an active GPU process. Never start more than one campaign coordinator.

Preserve the dirty worktree: JOURNAL.md and several old launchers, binaries
and untracked files were already user-modified. Stage only your own files
or hunks. ASCII only. Append experiment evidence CONFIG -> COMMAND ->
RESULT -> VERDICT and commit/push coherent own changes. Do not change
bin/ or live shelf infrastructure without its required smoke checks.
No production promotion/restart, messages to others, new weights download,
or external publication from this periodic review. No subagents needed.

Report a concise timestamped status, completed findings, action taken,
running test identity and next step. If all campaign objectives are met or
user input is truly required, explain exactly why and disable only the
b70-cache800k-review.timer to prevent pointless repeated reviews.
