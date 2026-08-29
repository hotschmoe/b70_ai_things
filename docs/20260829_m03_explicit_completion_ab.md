# M03 explicit collective completion A/B

Date: 2026-08-29

Status: passed as an isolated P2P-off all-reduce correctness oracle. This is
not a model-serving or endpoint-performance qualification.

## Configuration

- Host kernel: `7.1.0-070100-generic`.
- Host Compute Runtime package: `26.22.38646.4-0`.
- Image: `vllm/vllm-openai-xpu@sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f`.
- vLLM: `0.27.2rc1.dev77+gac7509e2b`.
- PyTorch: `2.13.0+xpu`.
- World size: two B70 ranks; `CCL_TOPO_P2P_ACCESS=0`.
- Dtype and shapes: BF16 `[1,5120]` and `[4,5120]`.
- Modes: blocking `dist.all_reduce` and `async_op=True` followed by
  `Work.wait()`.
- Consumer: `collective_output * 2 + 1`, issued before any post-collective
  XPU synchronization.
- Order: balanced and alternating by lifetime, stage, and round.
- Per lifetime and rank: two warmups plus eight measured rounds for both
  modes at both shapes, or 40 calls.
- Source commits: initial oracle `57e4e4f`; fingerprint correction `fddeb41`.

The tracked oracle and launcher SHA256 values are
`68338a5dcefa256f2f0101fac6e5613d3bf5a44d12b84a2f561b201147530103`
and `c1d4e6bc71ac40dc55107bd1ee3746e95faef5d104ce32bb9a1272d6c9269c4a`.

## Failed harness attempt and recovery

The first lifetime reached exact tensor equality, then the evidence-only
fingerprint path attempted to convert a nested byte list directly to
`bytes()`. Python raised `TypeError` before the validation event was written.
This was an oracle bug after collective validation, not a collective mismatch,
hang, or device loss.

The container tore down and card plus compiled P2P-off collective post-health
passed. The required `bin/xe-reset --method rebind` then completed on the same
boot ID with clean card and compiled collective health. The correction flattens
the byte view before hashing and was committed before the full rerun.

Rejected evidence directory:
`/mnt/vm_8tb/b70/results/m03_completion_wait/20260829T061039Z/`.

## Passing evidence

Three fresh process-group and container lifetimes passed. Each rank completed
40 calls per lifetime. The externally audited evidence contains:

- 240 total collective/consumer calls;
- 1,080 flushed events with strictly increasing per-rank monotonic times;
- exact outputs for every call and exact blocking/async fingerprints for each
  matched shape, stage, and round;
- identical call signatures and output fingerprints across both ranks;
- no exception or unreturned call ID; and
- clean teardown plus card and compiled P2P-off collective health before,
  between, and after lifetimes.

The blocking event contract was `entry -> collective_return ->
consumer_return -> validation`. The asynchronous contract was `entry ->
work_created -> wait_return -> consumer_return -> validation`. The consumer
was submitted before the validation synchronize, so exact output establishes
that neither completion route exposed a consumer race in this oracle.

Accepted evidence directory:
`/mnt/vm_8tb/b70/results/m03_completion_wait/20260829T061458Z/`.
The sorted 15-file evidence manifest SHA256 is
`dc19da09ffdcf2504775f574c54e1140616ae7dcc109fdebfd99b0c1c4d29210`.

## Exploratory host timing

These values describe Python-to-runtime host boundaries in the isolated
oracle. They are not synchronized device-kernel latency or endpoint speed.
Each median has 48 measured calls across three lifetimes and two ranks.

| Shape | Blocking call return median | Async launch plus wait median | Entry through consumer return, blocking | Entry through consumer return, async |
| --- | ---: | ---: | ---: | ---: |
| `[1,5120]` | 94.134 us | 158.392 us | 184.652 us | 239.362 us |
| `[4,5120]` | 96.925 us | 160.471 us | 182.628 us | 240.504 us |

The explicit async/wait route did not remove host overhead in this eager
micro-oracle; its observed medians were higher. This does not predict a
compiled model result, where completion ownership and graph boundaries differ.

## Verdict

M03 passes: blocking c10d and explicit asynchronous work plus `Work.wait()`
both provide exact completion to an immediate dependent consumer with P2P
disabled. The async/wait mechanism is correctness-qualified for a later
matched endpoint control, but this result supplies no speed motivation and
does not qualify a Steve model patch. Proceed to M04 graph-boundary census
tooling before deciding whether W05 should spend model-serving time on this
route.
