# M04 Qwen3.8 graph census and host-stall review

Date: 2026-08-29

Status: passed after a contained two-step retry. The paired-rank structural
census is repeatable and the retry met the profiling-overhead gate. The
overnight attempt remains rejected as experiment evidence because the host
stopped making observable progress while the replacement server loaded
weights.

## Matched configuration

- Model: Qwen3.8-27B compressed-tensors W8A8 GPTQ, GDN RTN overlay.
- Backend: current SGLang XPU image
  `adc915d266eaa74f7bea164d97cb7870b04dd7eb4c613952c56f4fbff1584a78`.
- Topology: TP=2, `CCL_TOPO_P2P_ACCESS=0`.
- Target and KV dtype: BF16.
- Decode: breakable graph, batch size 1, graph reclaim interval 500.
- Context: 4,096; memory fraction 0.75; max running requests 1.
- Cache and draft: radix cache off, MTP off.

## Structural result

The accepted structural evidence is the four-decode-step capture at:

`/mnt/vm_8tb/b70/results/m04_graph_census/20260829T064150Z/`

Both ranks reported the same stable signature on all four target tokens:

| Per target token | Rank 0 | Rank 1 |
| --- | ---: | ---: |
| Graph pieces (`zeFenceReset`) | 131 | 131 |
| Host waits | 131 | 131 |
| Queue submissions | 262 | 262 |
| BF16 all-reduce `[1,5120]` | 129 | 129 |
| BF16 all-gather `[1,124160]` | 1 | 1 |
| Total collectives | 130 | 130 |

The analyzer passed rank structural, traced collective, and rank collective
agreement. The census JSON SHA256 is
`1ca603b54d1ce45a4e03ec385ab9a3e24ad1a29e88256ba3dee8ae56f41f7db7`.
The two rank trace SHA256 values are
`be7927fe0e84044b8de588191f0e03dac0565b8956116be62879042265adfa45`
and
`aeb5782421c3221d45908d7cf038aef4e4119a2eb77dcae8be01cda7c49a28d1`.

This is substantially more fragmented than the exact June Qwen3.6
PIECEWISE control, which had 41 graph pieces, 41 waits, and 82 submissions per
token. The new result identifies 130 collective boundaries per Qwen3.8 token;
it does not prove that any one boundary is removable or that removing it will
improve endpoint speed.

The profiled request produced 10.1215 post-first-token tok/s versus a 14.4349
tok/s mean for the two controls. The ratio was 0.701183, or 29.9 percent loss,
so the predeclared maximum 25 percent instrumentation-loss gate failed.
Teardown, per-card health, and compiled P2P-off collective health passed.

## Passing contained retry

The accepted M04 result is:

`/mnt/vm_8tb/b70/results/m04_graph_census/20260829T182821Z/`

It ran from committed Git identity
`b6cc0362d975af031d5239383865d81fa82b8f4e` with two profiled decode steps,
the 96 GiB available-memory admission gate, at most 1 GiB preexisting swap,
and a requested 64 GiB memory-plus-swap container ceiling.

Both tokens on both ranks reproduced the exact structural signature in the
table above. The profiled request measured 12.6260 post-first-token tok/s
versus a 14.6965 tok/s mean for the two controls. The 0.859115 ratio is a 14.1
percent loss and passes the predeclared minimum 0.75 ratio.

All 48 five-second host samples recorded zero swap use. MemAvailable ranged
from 123,996,420 KiB before loading to a 61,283,424 KiB minimum while serving.
Memory PSI briefly reached 0.05 at 60 seconds and returned to zero. After
teardown, MemAvailable returned to about 123.9 million KiB with zero swap and
zero current PSI. Card health and the compiled P2P-off collective passed both
before and after serving, and the endpoint disappeared during teardown.

The census JSON SHA256 is
`41010eeb690c286b2629f2b46360b5c70d2715fa530728384e3c930c51abe144`.
The memory-monitor SHA256 is
`a68ee108d0743d4b4012d282493ea0309afd9cdc7aa56c5284ccc8fdd1c68190`.
The rank trace SHA256 values are
`8724e31a1bd84887dcd63034f0f2543b6e469451b2f656e3d555f10b54dc1ad8`
and
`94653427473c130d38e9dfacf7d9a0808e76f3a835d95c540b2e830acc65ec81`.

## Unresponsive-host incident

The third attempt used two decode steps and result directory
`/mnt/vm_8tb/b70/results/m04_graph_census/20260829T064910Z/`. It never reached
endpoint health or profiling and is not M04 evidence.

The previous boot ID was `e2d5777df6bb4d92a7180fb07ae17919`. Its journal
ends at 06:50:10 UTC, two seconds after container
`c6437c0895c60ecf185a2f5f8858db5edd328154305a96f3355c863deec28ac5`
started. The container log continued through 06:50:34 and ended after both TP
ranks entered weight loading with 31.89 GiB available on each XPU. Docker
recorded no memory limit, no swap limit, `oom_score_adj=500`, and
`OOMKilled=false`; the container received exit 255 when Docker recovered after
the reboot. The container-log SHA256 is
`02ba20e3cbbb5ab65dd33b92cc689bf50baf6696044598e48db1211cccb90df0`.

The last sysstat sample before launch showed active host pressure despite a
moderate load average:

| 06:50 UTC metric | Value |
| --- | ---: |
| MemAvailable | 68,685,040 KiB (65.5 GiB) |
| Swap used | 3,893,560 KiB (46.4 percent of 8 GiB) |
| Swap in / out | 1,040 / 2,739 pages/s |
| Major faults | 476/s |
| Direct/background reclaim scans | 7,082 / 20,143 pages/s |
| Reclaimed pages | 46,038 pages/s |
| Root NVMe queue depth | 60.91 |
| Root NVMe await | 58.87 ms |
| Root NVMe utilization | 4.03 percent |

The high queue latency with low bandwidth utilization is consistent with
blocked or reclaim-coupled root I/O, not simple throughput saturation. The
workspace/model Btrfs device was reading about 71 MiB/s with 7.65 ms await.

There is no final kernel OOM, hung-task, or GPU-fault record for this incident
because the journal stopped almost immediately. However, the same boot already
contained a directly observed event on 2026-08-28 in which global memory
pressure blocked `jbd2/nvme0n1p2`, journald, and Btrfs writeback for more than
122 and 245 seconds. That event included global OOM kills and approximately
59,102,188 KiB of `gpu_active` memory. It demonstrates a mechanism that can
leave ICMP ping responsive while SSH and other userland services stop making
progress.

The current boot is quiet: swap use is zero, memory PSI is zero, Btrfs device
error counters are zero, and sampled root-device latency is normal. SMART data
could not be read without elevated authentication, so storage hardware health
is not established.

## Classification and safeguards

The most likely classification is a recurrence of host memory-reclaim,
swap, and root-journal I/O stall during the third back-to-back TP=2 model load.
This is not classified as a GPU hardware wedge. The exact initiating component
cannot be proven from the truncated journal, and an underlying root-NVMe
latency problem is not ruled out.

The Qwen3.8 W8A8 launcher now fails before container creation when host
available memory or used swap exceed configured bounds. The M04 path requires
at least 96 GiB MemAvailable and at most 1 GiB swap used. It also starts the
container with a 64 GiB memory-plus-swap ceiling, which disables container
swap and preserves host headroom, and records five-second memory/PSI samples.
An unexpected loader peak should now fail as a container-local capacity error
instead of consuming the host.

## Verdict

M04 passes. Retain the exact 131/131/262 structural signature and the shaped
130-collective census for later topology comparisons. The accepted two-step
capture met the 0.75 throughput-ratio gate with exact rank agreement, bounded
host pressure, clean teardown, and post-health. Reject the overnight attempt
entirely as an experiment and retain the host admission, cgroup, and monitoring
safeguards for subsequent Qwen3.8 work.
