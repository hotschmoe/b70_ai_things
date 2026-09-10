# Patched Pi recurrence: authorized downtime tests

User authorized endpoint shutdown and testing on September 10, after arranging
the client pause. Preserve hotschmoe-dd as the first served name. This supersedes
the earlier read-only boundary for this testing window; it does not authorize
claiming qualification from HTTP success or installing unreviewed client guards.

## Shutdown

CONFIG -> Patched day trial 20260910T072531Z, image7b107d0e, TP2/P2P0,
MTP3/FULL/prefix/freshbe02 FP8, context200000.

COMMAND -> python3 vllm/int4/day_trial/serve.py stop. This is the existing owned
STOP-marker path; no sudo permission was needed to stop the user-owned runtime.
The parent retained both GPU leases through teardown and strict post-health.

RESULT -> Stop helper and parent exit0; systemd inactive/dead, MainPID0; model
container absent. Strict health passed both cards. Compiled two-rank collective
health passed ten iterations at shape4x5120 with P2P0. Kernel remains
7.1.0-070100-generic. Captured shutdown evidence is under:

    /mnt/vm_8tb/b70/results/bang_recurrence_testing_20260910/shutdown

VERDICT -> Endpoint is down for the authorized investigation. No reset or host
stack change was required. The live trial's quality failures remain preserved.

## Initial MTP comparison

CONFIG -> Both arms use image
sha256:7b107d0e675390fedbbdbe743f104c08072b6175ce8ccbbfdad64dc82bf90ad1,
fresh scale SHA256
be02d915a8ac188341870cc9f642d77665b744235e142330a7a37b8f4c711062,
TP1/P2P0, FP8 E4M3 KV, prefix on, FULL_DECODE_ONLY, context100000,
concurrency4 and prefill batch32768. Card0/port18151 uses MTP3;
card1/port18152 uses MTP0. Both preserve the primary hotschmoe-dd and register
their detailed research aliases in evals/configs/models.yaml.

COMMAND -> The frozen *.plan.json server argv launch kv_campaign_server.py
directly, which acquires the selected bin/gpu-run lease and pins that card.
This overlaps model startup with CPU client preparation. Frozen *.jobs.json
jobs are atomically queued into the owned server's jobs directory and execute
only after READY. The coordinator retains responsibility for STOP and final
health; empty-job run_arm plans must not be launched.

Raw root:

    /mnt/vm_8tb/b70/results/bang_recurrence_testing_20260910

Plans: tp1-card0-mtp3-ctx100k.plan.json and
tp1-card1-mtp0-ctx100k.plan.json. Corresponding .launch.json, .jobs.json,
client-freeze.json and runtime-comparison-validation.json preserve exact
commands/source/payload/configuration identities. Model source/config inputs
were not changed while either process ran.

RESULT -> Startup resolved block1600 in both arms. Initial jobs are eleven
requests: reconstructed earliest agent1 warm request and two target repeats;
then exact integer-array tasks with caps64/256/1024/2048, each cold and reused.
The corpus is frozen at SHA256
7d4d112860c2db722db2423d981172a5c3e61f79dcbbd6410b2b392926579260.
Client collects all content outcomes and exits nonzero if quality gates fail.
Actual token usage/cache hits and SSE are retained. HTTP success is not a gate.

CPU client checks passed: exact-answer rejection, missing reuse, incomplete
output, and continued collection after an initial failure with failing final
status. Reconstructed Pi output still needs manual semantic review even when
structural checks pass. Its original system/tools/sampling/wire filtering are
unknown; these are explicitly controlled reconstructions, not exact replay.
The clean length tasks use thinking off, temperature0 and seed42. Requested
caps are not actual generated lengths; inspect usage and answer completeness.

VERDICT -> Initial inference results pending. Device differs with MTP, so
cross over cards if results split before attributing causality. These short
TP1 tests do not qualify the former200K TP2 service.

## Additional evidence and planned controls

The user's post-export report describes success declining with requested
output length. Its newest sessions are absent from the verified archive;
retain it as a hypothesis, not measured backend correctness evidence.
The citizen-authored observatory was downloaded without executing scripts.
One cross-check contradicts its claimed agent10 birth-only bang: the frozen
session has three completed assistant tool turns before the first bang at
line16/id17b41973. Avoid deriving causal triggers solely from citizen prose.
See raw observatory-crosscheck.json for the exact scope.

Cancellation followed by drained cached continuation is being prepared as a
separate controlled job. Actual-source CPU review found no additional proven
Python bug: prior accepted-count fixes and explicit padded graph-tail setup
remain present. Immediate block-free behavior relies on execution ordering;
that is a lifecycle test target, not a demonstrated defect. Native GDN
arithmetic with padded state strides and partial acceptance remains a separate
numerical-oracle target from the already prepared Mamba-copy checks.

No SGLang feature-parity conclusion follows from these tests. Retain the
[earlier campaign](20260910_bang_campaign_resume.md) and
[recurrence evidence](20260910_pi_bang_recurrence.md) for backend alternatives
and remaining controls. Append measured outcomes below without rewriting
failed or incomplete raw experiments.
