# W01 Qwen3.8 W8A8 long-output baseline

Date: 2026-08-29

Status: passed. The corrected SGLang Qwen3.8 W8A8 TP2 control completed an
exact 50,000-token native greedy stream beyond every prior graph failure
boundary, retained 96.13 percent of its initial 5K-window throughput, stopped
cleanly, and left both cards and the compiled P2P-off collective healthy.

## Accepted configuration

- Model: Qwen3.8-27B compressed-tensors W8A8 GPTQ with selective GDN RTN.
- Image: `b70-sglang-xpu-int8-runtime` digest
  `adc915d266eaa74f7bea164d97cb7870b04dd7eb4c613952c56f4fbff1584a78`.
- Runtime: SGLang `0.5.19.dev443+gbede6bc37c`, PyTorch `2.13.0+xpu`,
  PyTorch Git `cf30153c4c131c8164ee7798e5022d810682e2cb`.
- Host: kernel `7.1.0-070100-generic`, Compute Runtime
  `26.22.38646.4-0`, Level Zero loader `1.28.2-2`.
- Topology: TP=2, `CCL_TOPO_P2P_ACCESS=0`, BF16 target and KV cache.
- Decode: target only, breakable batch-size-1 graph, reclaim every 500
  replays, MTP off, radix cache off.
- Capacity: 65,536 context, memory fraction 0.70, maximum one running request.
- Host containment: 96 GiB admission floor, at most 1 GiB used swap, and a
  64 GiB container memory-plus-swap ceiling that disables container swap.

The accepted transaction ran from Git identity
`a17eb6ad9caffd46d4a853b3952fcd5fe31a3edd` and saved evidence at:

`/mnt/vm_8tb/b70/results/w01_qwen38_w8a8/20260829T195551Z/`

## Coherence and identity

The exact served ID on both fresh lifetimes was
`qwen3.8-27b-W8A8-gptq-gdn-rtn-breakable-reclaim500-tp2-bf16kv-w01`, with
`max_model_len=65536`. Both server logs reported BF16 target and KV dtypes,
the intended compressed-tensors plus GDN RTN overlay, breakable decode, and
reclaim500. Both containers used the pinned image and exact 64 GiB no-swap
limit.

Server A produced repeat-exact greedy completions for all eight prompts, with
two requests per prompt. Fresh server B was also repeat-exact and matched all
eight server-A completion hashes; the mismatch list was empty. Corpus A and B
SHA256 values are:

- `b5b01782764cc310f828e395e933471e555879cf317f85184915ae53d1fa47ff`
- `2740f737bf0e97b9900974e13f96ee69e67eb5fe75249ef9ead4ef4a9aba2163`

## Native-client oracle

Before the accepted transaction, a bounded real-endpoint oracle established
the corrected native `/generate` contract after the rejected `seed` attempt.
With temperature zero, `ignore_eos=true`, and no unsupported seed field, it
returned exactly 30 tokens and a length finish at 13.4060 tok/s. It then tore
down cleanly and passed card plus compiled collective post-health. Its result
directory is:

`/mnt/vm_8tb/b70/results/w01_native_probe/20260829T195200Z/`

The replay JSON SHA256 is
`f6225abb821a6568e6c13543b11bf6eb1431d7b7da00b48e2a58c7b747cffc36`.

## Fifty-thousand-token result

One native streaming request used temperature zero, `ignore_eos=true`, no
seed field, and an exact 50,000-token output limit. The client preserved the
SHA256 of the validated 50,000-entry output token array, reported a length
finish, and wrote durable partial evidence at every 5,000-token boundary. That
client revision did not serialize the literal array; future revisions do.

| Metric | Result |
| --- | ---: |
| Prompt tokens | 32 |
| Completion tokens | 50,000 |
| TTFT | 323.074 ms |
| Total response time | 3,435.460 s |
| Post-first-token rate | 14.5552 tok/s |
| First 5K-window rate | 14.8396 tok/s |
| Final 5K-window rate | 14.2652 tok/s |
| Final/initial ratio | 0.961298 |
| Required ratio | 0.800000 |

The ten successive window rates were 14.8396, 14.8047, 14.7181, 14.6605,
14.6035, 14.5202, 14.4467, 14.3912, 14.3266, and 14.2652 tok/s. This is a
gradual 3.87 percent decline, not a throughput collapse. It is a single-stream
result and makes no concurrent-serving claim.

The output text SHA256 is
`2bbf3233fd2fff9250848947a9cb845d999407e1455dfee79b159e7e70703682`.
The complete output-token-array SHA256 is
`01d78ddc5700922abcebc4ef5298df5c98840915eda72dcc3454c014860ca3a1`.
The final replay JSON SHA256 is
`4300568c7a2da2d731124bf65284c5f10e40b6dfeb0484011727c5122557e349`.

The server log crossed the earlier approximately 17,664-token W8A8 FULL
fault, the 19,328-token NVFP4 FULL abort, and the former 26,368-token
Terminal-Bench maximum. It contained 21 executable re-instantiation markers
and no configured fatal server marker.

## Host safety, teardown, and health

Across 775 five-second samples, MemAvailable never fell below 65,245,888 KiB
(62.223 GiB). Swap use remained zero. Memory PSI `some` and `full` totals were
unchanged at 144124 and 143620. MemAvailable returned to 123,624,824 KiB after
the final health transaction.

The kernel transaction contained only normal container-network lifecycle
messages plus one `perf_event_max_sample_rate` reduction. It contained no OOM,
hung-task, GPU VM fault, dead engine, wedge, or failed reset marker. Both
servers disappeared after graceful teardown. Card and compiled P2P-off
collective checks passed before serving, between the two fresh servers, and
after final teardown. The final durable health artifacts report both cards
healthy and
`COLLECTIVE_HEALTH_OK world_size=2 shape=4x5120 compiled_iterations=10 p2p=0`.

The pre- and inter-server card commands were fail-closed and returned success,
but their `tee` files are empty because that version of the harness captured
stdout while `bin/xpu-health` wrote diagnostics to stderr. The populated final
card artifact is SHA256
`e9f3293cbccc9b9d07d5f665e37f940b1ea0f23da34b50468c052d459b52eeff`.
The final collective artifact is SHA256
`93830e24e5201487f24df92401edb4e5054ec720ef64e6239a5d9e0325f5f614`.
The harness now captures both streams for future runs.

## Verdict and next experiment

W01 passes as the corrected target-only Qwen3.8 W8A8 long-output baseline.
It proves deterministic two-lifetime corpus coherence, one stable 50K stream,
bounded host behavior, clean teardown, and post-health for this exact
single-request configuration. It does not establish concurrency, cache-state
correctness, MTP correctness, or a graph speed attribution.

Proceed to W02 with matched prompts, outputs, cache-off state, and timing:
eager, breakable without reclaim, and breakable plus reclaim500. Use eager
versus breakable to attribute graph speed, then breakable versus reclaim500 to
attribute reclaim overhead and long-run protection. Do not promote a shelf
entry until the eventual winning recipe passes intended-concurrency coherence.
