# Native5802 TP2 100K qualification preparation

CONFIG

Candidate d55637b3353e / native 6717e3ec7fe3, original repaired7b Python and
package identities unchanged. The exact paused clean100K Config.json and
Mounts.json were copied unchanged: TP2, FP8 E4M3 KV with be02 fresh calibration,
MTP3, prefix caching, FULL_DECODE_ONLY graph, context100000, c4, batch32768,
memory0.96. Runtime overrides MTP4/auto placeholders in the preserved Config
to MTP3/FP8 exactly as the original campaign did. Adapter absent/off.

Retained P2P0, TCP/loopback OFI, pidfd IPC, direct send/recv, 4GiB simple
collective thresholds and VLLM_XPU_ALLREDUCE_HOST_WAIT=1. No P2P1 probe.
hotschmoe-dd stays first, with the existing registered TP2 ctx100K detailed
alias. New image identity is explicit in every plan and runtime receipt.
Port18125 is the original private loopback port; no public endpoint change.

COMMAND

Prepared root:
/mnt/vm_8tb/b70/results/bang_recurrence_testing_20260910/native5802-tp2-100k-plan

Read-only gate:
python3 vllm/int4/native_gdn_oracle/native5802_candidate/tp2_launch.py ROOT

After parent review and allocation only, the same command with --run delegates
to run_arm.py. Do NOT wrap it in gpu-run: kv_campaign_server.py acquires both
leases itself. Its frozen strict health and recovery behavior is retained.

RESULT

CPU source/config/scale checks and read-only prerequisite validation passed.
Original 100K external helpers and hooks still match every recorded hash.
Candidate pair health and TP1 plus TP2-local native numerical outcomes match.
Full TP1 model matrix review, new TP2 runtime evidence, and qualification remain
pending. Nothing was launched and no pass marker was created by preparation.

Stages are startup host-trace review, boundary10, exact serial-session0 cold/
reuse2, four-client semantic/reuse16, then the entire existing clean100K gate:
tiny24, early-clean history3 with quality review, three deterministic32 rounds,
three sampled32 rounds (seeds42/43/44) each with strict quality review, and final
host-trace review. Failure stops the arm and initiates normal teardown.

Before trusting workload results, inspect actual cache capacity, resolved
1600-token hybrid block, loader receipts for all34 scales, graph/MTP activation,
model identity and actual max_model_len. Compare the recorded candidate startup
profile with the prior recipe's per-rank FP16[32768,5120] 138 all-reduces and
FP16[32768,2560] three all-gathers. The automatic first-stage review requires
matched shapes/counts and entries/returns on both ranks; changed real shape or
counts requires investigation. These are host boundaries, not GPU completion
or graph replay counts. Preserve raw capacity/profile logs even on failure.

The server runs strict both-card and compiled P2P0 pre-health, normal stop,
strict both-card and compiled pair post-health. Its crash branch requests
rebind reset under the pair lease, followed by candidate post-health. Parent
must verify owned container removal before any manual recovery and review
cleanup/reset logs; do not treat a workload marker alone as whole-arm success.

VERDICT

Prepared, not activated or qualified. Require all new jobs, lifecycle exit0,
strict post-health and teardown evidence. The earlier interrupted100K run has
no complete marker and contributes no fabricated pass here. A 200K successor
must be separately frozen to require this new completed predecessor; the old
long-context gate remains unchanged and blocked by its old incomplete input.
This is an instrumented correctness arm, not a speed qualification.
