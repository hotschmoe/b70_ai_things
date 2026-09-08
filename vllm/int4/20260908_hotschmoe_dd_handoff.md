# hotschmoe-dd INT4 handoff - 2026-09-08

## Selected configuration

Qwen3.8-27B AutoRound INT4 W4A16 group 128, R276, TP2/MTP4, FP16 KV.
Context 200000, max sequences 4, batch 32768, utilization 0.96, 64 GiB cgroup.
Prefix caching and CPU offload are disabled. xhigh thinking remains the
default; clients can explicitly disable thinking. Language-only serving.
Public alias remains `hotschmoe-dd`, API-key frontdoor port 18080, loopback
backend 18124. The existing API key is retained and is never written here.

Image: `sha256:521eb277c0733f8c2ce47aea1bb98ed576c6f1ad63bf5baf22d38fc07abf54ad`.
Source: `54aefaf067b422d507f51f1e362efeecc58668ba`.
Model: `devan-carlin/Qwen3.8-27B-int4-AutoRound`, revision
`bce40cacab0a4535b92fb3d57615c2bea9adf3d1`.
R212 changes GPTQ kernel routing in configuration; tensor bytes remain
AutoRound. Original/relabel weights, pinned image, release assets, custom
ops and source archives are preserved locally. See [the artifact lock](neural_replica_lock.json).
No host kernel, UMD or PyTorch upgrade was performed.

## Qualification and limits

CONFIG -> Frozen published short-context recipe, then separately qualified
200K daily extension with prefix caching still disabled.
COMMAND -> Four fresh strict lifecycles, full daily campaign, a fresh
production startup/readiness/auth/coherence/stop lifecycle.
RESULT -> Strict MTP4 pair 100.53594/99.89262 tok/s, mean 100.21428 on the
author's class-balanced first 99-interval metric. Complete token arrays
match 12/12 between fresh runs and against MTP0. Full-output wall-throughput
medians 93.99733/94.15117 tok/s are separate metrics. Do not promise 100 tok/s
for every context, task or concurrent client.
Daily C1/C4, repeated 2048-token guides, 32/32 long-history tool checks,
cancellation, xhigh, 180K/199K recall, overload and post-overload coherence
pass. The four 132K+8K overload requests queued without preemption; the
slowest TTFT was 539.1 s. This is not an INT4 eviction/reload test.
Both profile ranks have matched real collective shapes/counts. All completed
benchmark and fresh production lifecycles passed teardown/post-health.

Coding: INT4 **158/164 base, 150/164 extended**, FP8 **157/164, 152/164**.
No raw symbol corruption. This meets the frozen <=2 net additional-failure
gate, exactly at the extended-test boundary. There are real edge-case
regressions; see [the paired failure review](20260908_coding_failure_review.json). This is bounded
thinking-off HumanEval+ evidence, not no-loss parity or a Terminal-Bench
ranking. The model remains fallible.

RAM offload: rejected for promotion. Both corrected matched MTP/offload
candidates produced repeated-symbol tool corruption despite faster cache
reloads. The original prefix-on FP8 baseline also corrupted later coding
outputs under stress. The clean FP8 comparator used prefix-off. These
findings do not isolate one proven root cause. Keep both CPU offload and
prefix caching disabled here; long histories still incur costly prefill.
FP8 KV remains deferred at the user's lowest priority.

VERDICT -> Select the qualified INT4 daily configuration with the above
quality and long-prefill limits; do not ship the rejected offload images.

## Boot configuration command

The repository unit is prepared and statically verified. Its real readiness
and ExecStop commands passed in the fresh production lifecycle. Actual
systemd installation and reboot behavior have not been tested: local sudo
requires the user's password. The final serving instance runs manually in
a detached session while holding both GPU leases.

Run this to install/enable the INT4 unit for startup without interrupting
the current instance (the fp8 path is retained for compatibility):

```bash
sudo bash /mnt/vm_8tb/github/b70_ai_things/vllm/fp8/install-hotschmoe-dd-systemd.sh
```

The installer backs up the previous units, reloads systemd, enables
hotschmoe-dd and disables the retired b70-daily-driver boot unit.
Do not start a second serve alongside the manual instance. To transfer
ownership to systemd immediately after installation, first drain clients:

```bash
python3 /mnt/vm_8tb/github/b70_ai_things/vllm/int4/serve_qwen38_autoround_r276_daily.py stop
sudo systemctl start hotschmoe-dd.service
```

The stop command waits for teardown health. After systemd owns the serve,
use `sudo systemctl stop hotschmoe-dd.service` for normal shutdown.
The unit intentionally has Restart=no: failed TP2 startup requires
investigation/recovery, not an automatic crash loop.

## Evidence

- Detailed chronology: [replica campaign](20260908_neural_replica_campaign.md).
- RAM findings: [RAM campaign](../fp8/20260908_kv_campaign.md).
- Exact startup pins: [qualification manifest](r276_daily_qualification.json).
- Raw replica campaign: /mnt/vm_8tb/b70/results/int4_replica_20260908.
- Raw RAM/reference campaign: /mnt/vm_8tb/b70/results/kv_campaign_20260908.
- Completed production startup/stop:
  /mnt/vm_8tb/b70/results/qwen38_int4_r276_daily/20260908T104540Z.
- Final lifecycle pointer: /mnt/vm_8tb/b70/run/hotschmoe-dd-int4-current.
- Final launch record: replica raw root/final-serving-launch.json.
