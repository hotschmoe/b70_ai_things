# W02 Qwen3.8 graph/reclaim target-divergence result

Date: 2026-08-29

Status: closed negative. Breakable decode was deterministic and reclaim500
removed severe replay-count slowdown, but both graph arms changed the native
greedy output relative to eager beginning at token index 24. W02 therefore
fails its target-exactness gate and supplies no promotable graph speed claim.

## Matched configuration

- Model: Qwen3.8-27B compressed-tensors W8A8 GPTQ with selective GDN RTN.
- Image digest:
  `adc915d266eaa74f7bea164d97cb7870b04dd7eb4c613952c56f4fbff1584a78`.
- Runtime: SGLang `0.5.19.dev443+gbede6bc37c`, PyTorch `2.13.0+xpu`.
- TP=2, P2P off, BF16 target/KV, 65,536 context, memory fraction 0.70.
- Maximum one request, MTP off, radix cache off, empty think cap.
- Arms: eager, breakable with reclaim disabled, and breakable with reclaim500.
- Per arm: one fresh server, repeat-exact eight-prompt corpus, one 768-token
  warmup, and three native greedy 2,048-token measurements.
- Containment: 96 GiB host admission floor, at most 1 GiB used swap, and a
  64 GiB no-swap container ceiling.

The accepted negative transaction ran from Git identity `7a3c2ac` and saved
evidence at:

`/mnt/vm_8tb/b70/results/w02_qwen38_w8a8/20260829T213708Z/`

## Correctness result

All three eight-prompt corpora were repeat-exact. Both graph-arm corpora matched
all eager completion hashes at the 96-token cap. Every measured 2,048-token
stream also repeated exactly within its own arm.

The stronger native-array comparison failed:

| Pair | Exact | First mismatch, zero based | Mismatched positions |
| --- | --- | ---: | ---: |
| Eager versus breakable | No | 24 | 2,011 / 2,048 |
| Eager versus reclaim500 | No | 24 | 2,011 / 2,048 |
| Breakable versus reclaim500 | Yes | none | 0 / 2,048 |

Eager's output-array SHA256 was
`c64d070e5b79138c30386367506613066d38b9c9d3759207df71c57bfc021b0f`.
Both graph arms produced
`a1856299df39da9652f45a05a9f51475cf28384db6d354756087efa49a71109b`.
This localizes output change to graph mode, not reclaim. It does not by itself
identify the first numerically different operation or prove which trajectory
has better model quality.

The persisted comparison JSON SHA256 is
`736322d04b4044e584ddc1603caea372d02b188e40fb8b861005c4f02187ef23`.

## Diagnostic performance

These rates are useful mechanism evidence but are not target-exact performance
claims because the eager and graph arrays differ.

| Arm | Repeat 1 | Repeat 2 | Repeat 3 | Median | Repeat 3 / repeat 1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Eager | 6.0379 | 6.0420 | 6.0485 | 6.0420 | 1.0018 |
| Breakable, no reclaim | 11.7182 | 10.0590 | 8.8579 | 10.0590 | 0.7559 |
| Breakable, reclaim500 | 14.7893 | 14.8028 | 14.8269 | 14.8028 | 1.0025 |

No-reclaim lost 24.4 percent from its first to third repeat. Reclaim500 stayed
within 0.25 percent and emitted 16 executable re-instantiation markers. The
no-reclaim graph trajectory and the reclaim500 graph trajectory were exactly
the same, so reclaim repaired replay performance stability without adding a
second token-identity change.

The raw medians imply breakable/eager `1.664851` and
reclaim500/no-reclaim `1.471597`, but the W02 analyzer marks performance
attribution unqualified because cross-arm target equality failed.

## Host, teardown, and health

All three servers stopped normally. Both cards and the compiled two-rank
P2P-off collective passed before the transaction, after each arm, and after
final cleanup. The kernel transaction had no configured OOM, hung-task, GPU VM
fault, dead-engine, wedge, or failed-reset marker.

Across 612 five-second host samples, minimum MemAvailable was 62,545,508 KiB
(59.648 GiB) and swap use remained zero. Memory PSI `some` and `full` totals
increased by 6,185 and 6,155, respectively; these totals are microseconds and
did not coincide with swap, service failure, or health damage. The final card
and collective artifact SHA256 values were
`e9f3293cbccc9b9d07d5f665e37f940b1ea0f23da34b50468c052d459b52eeff`
and `e50a62bb983a5be20844ab4a6355482a792b1190c3bcea8c8cacdee379e6e632`.

## Verdict

Close W02 negatively. Do not run the planned 50K no-reclaim canary because its
short target-equality prerequisite failed. W01 remains valid evidence for a
stable, deterministic graph-specific 50K trajectory, but it must not be
described as eager-target-exact. Do not advance W03-W06 on this graph route
until a source-level numerical/state audit explains or repairs the token-24
divergence.

The next requested campaign lane is a deliberate port of the Neural.Download
official-FP8 vLLM recipe. Begin with exact source/model identities and a P2P-off
MTP0 target control; do not copy its direct-P2P launch onto this host without a
separate loaded-context oracle.
