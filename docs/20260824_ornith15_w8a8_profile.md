# Ornith-1.5 W8A8 TP=2 bottleneck profile - 2026-08-24

## Bottom line

The current Sglang Ornith path is launch/scheduler bound first, collective
bound second, and expert-GEMM bound third. P2P is not the missing 4-8x lever
in the current eager serve.

The corrected MTP1 serve reached 11.64 output tok/s with 97.5% draft
acceptance. The unprofiled request therefore used about 169.7 ms per verify
step. The slow rank performed only 25.0 ms of summed device work per verify
step. About 14.1 ms of that device work was oneCCL all-reduce. Even removing
all measured collective device time would only move the current path to about
12.7 tok/s. Eliminating the complete CPU-side collective call time gives a
more generous upper bound of about 13.5 tok/s.

The primary target is graph/replay or an equivalent persistent launch path.
Once launch gaps are removed, collective work becomes the largest device
bucket and is the next target.

## Community source refresh

The local community checkouts were fetched before this profile.

| Source | Local revision | Relevant result |
| --- | --- | --- |
| Steve Seguin, `b70-optimization-lab` | `0cf5b751`, clean `upstream-main` tracking `origin/main` | Qwen3.6-35B-A3B Quark W8A8: eager 16.7-17.2 tok/s, piecewise graph about 92, graph plus clone-safe custom collectives about 95.3; TP=2 safe smoke 85.87 |
| Sergio, `intel-arc-pro-b70-inference-cookbook` | `44e97e1`, clean `master` tracking `origin/master` | Ornith MixedCal-v2 on one B70: no-spec 70.74, MTP1 BF16 96.43, MTP1 DraftINT4 106.27; MTP2 84.16 and MTP4 66.27 |
| 0xSero, `qwen38-b70` | `e873853`, clean fast-forward | Current Qwen3.8 B70 source/reference material |

Steve's local `main` was an isolated local graft. It was preserved; a clean
`upstream-main` branch was created instead of overwriting it.

Sergio's MTP-depth result is directly applicable. Ornith has one trained MTP
layer. Later speculative positions collapse to 15%, 2.5%, and 0.5%
acceptance, so MTP1 is the correct setting. Our old steps=3/draft=4 serve was
not a valid performance configuration.

## Profile configuration

- Model: `ornith-1.5-35b-a3b-W8A8-rtn-mtp-shisa`.
- Backend: Sglang 0.5.15.post1, `sglang-xpu:mtp-0515`.
- Hardware: two Arc Pro B70 cards, TP=2, both at the existing 230 W cap.
- Context: 8192 for the diagnostic.
- Quantization: INT8 W8A8 routed experts; eligible dense text weights stored
  INT8 and dequantized once to BF16 compute; GDN, vision, lm_head, and MTP BF16.
- Speculation: one trained MTP step, two draft tokens.
- Execution: eager, overlap off, radix off, one active request.
- Communication: oneCCL with `CCL_TOPO_P2P_ACCESS=0`.
- Profiler: Kineto CPU+XPU, five decode steps, semantic `record_function`
  ranges, no per-operation synchronization.

Command:

```bash
./bin/gpu-run bash sglang/w8a8/profile_ornith15_w8a8.sh
```

Runtime artifacts:

```text
results/logs/ornith15_w8a8_profile_20260824T231106Z/
/mnt/vm_8tb/b70/sgl_cache/ornith15_w8a8_profile_20260824T231106Z/
```

Both cards passed health probes before and after the run. The endpoint was
stopped at exit.

## Throughput and MTP

The p512/g128 client test used the cookbook post-first-token method. The
entropy prompt tokenized to about 1179 tokens.

| Metric | Result |
| --- | ---: |
| Median client post-first output | 11.644 tok/s |
| MTP mean accept length | 1.975 tokens/verify |
| MTP draft acceptance | 97.5% |
| Implied unprofiled verify wall | 169.6 ms |

MTP quality is not the problem. The first trained head is extremely effective.

## Five-step decode census

The decode traces contain five target/MTP verify iterations per rank.

| Item | Count per verify | Slow-rank device ms/verify | Notes |
| --- | ---: | ---: | --- |
| All device kernels/copies | 1246.8 | 24.97 total | 93.1% idle inside the instrumented trace span |
| oneCCL all-reduce | 84 | 14.06 | 56.3% of slow-rank device work |
| Dense/shared/router GEMM | 238 | 5.12 | BF16 compute, including one-time-dequant weights |
| Routed-expert fused MoE kernels | 82 | 2.79 | w13 and w2 across 40 target layers plus one MTP layer |
| MoE top-k | 41 | 0.29 | 40 target layers plus one MTP layer |
| Dynamic INT8 activation quant | 80 | 0.10 | two expert activation quantizations per target layer |
| GDN recurrent update | 30 | 0.31 | one per GDN target layer |
| Full-attention FMHA | 20 | 0.17 | two device kernels per ten attention layers |

The target has 40 MoE blocks and the MTP head adds one, matching the 41
routing/top-k calls per verify. The two fused expert GEMMs per block match the
82 fused-MoE kernel launches. The trace also contains 84 all-reduces per
verify. This is a launch-count problem before it is a compute-throughput
problem.

The profiler itself increases CPU wall time, so the 93-96% trace-span idle
figures are diagnostic rather than an unprofiled throughput measurement. The
stronger comparison is the unprofiled 169.6 ms verify wall against the slow
rank's 25.0 ms of measured device work: about 145 ms per verify is outside the
device kernels themselves.

## Rank and communication asymmetry

| Rank | Five-step device busy | Five-step all-reduce | Mean all-reduce kernel |
| --- | ---: | ---: | ---: |
| TP0 | 78.88 ms | 24.83 ms | 59.1 us |
| TP1 | 124.86 ms | 70.30 ms | 167.4 us |

Non-collective device work is close between ranks. The 2.8x all-reduce latency
asymmetry creates nearly the entire device-time difference. This deserves a
rank-map/topology follow-up, but it is not large enough to explain the current
11.64 tok/s result by itself.

At the present acceptance rate:

- ideal ceiling using all current slow-rank device work: about 79 tok/s;
- current-rate ceiling if measured collective device time were free: about
  12.7 tok/s;
- current-rate ceiling if the full CPU-side collective call time were free:
  about 13.5 tok/s.

Thus communication is the post-graph bottleneck, not the pre-graph root cause.

## Kernel-path findings

Sglang emitted both of these warnings for the B70:

```text
Using default MoE kernel config. Performance might be sub-optimal!
E=256,N=256,...,dtype=int8_w8a8,per_channel_quant=True.json not found
Using MoE kernel config with down_moe=False. Performance might be sub-optimal!
...int8_w8a8,per_channel_quant=True_down.json not found
```

The current Triton fused-MoE path is therefore untuned for this exact B70
shape. It is worth tuning, but the complete fused-expert bucket is only about
2.8 ms per verify today. Even a 2x expert-kernel win saves less than 1% of the
current unprofiled verify wall. It becomes worthwhile after graph/replay.

The profile's first semantic-range install missed only the explicit top-k and
TP-all-reduce labels because it referenced `TopK` through the wrong module.
Raw Kineto correlation still measured those operations exactly. The import is
fixed for the next capture.

## Engineering order

1. Try Sglang XPU graph/replay with MTP1, P2P off, and a narrow batch-1 capture.
   Require exact output/coherence and clean card health.
2. If Sglang graph cannot capture this hybrid MoE path, port the artifact to
   the current vLLM Quark W8A8 path used by Steve. Use piecewise graph, mixed
   INT8 MoE workspace, graph-safe clone collectives, native GDN fallbacks, and
   keep P2P off on this box.
3. Tune the two missing B70 Triton MoE configurations for `E=256,N=256` and
   the down projection. Measure only after a graph-capable baseline exists.
4. Reprofile collectives. Reverse TP rank/card mapping to determine whether
   the 2.8x asymmetry follows the rank or physical card. Then test graph-safe
   push/custom all-reduce paths. Do not enable unsafe P2P-in-serve.
5. Extend W8A8 fast paths to shared expert/full-attention/GDN projections
   after the graph and collective floors are known.

Promotion requires the same coherent serving and Pi/Terminal-Bench gates as
the other product candidates. A tok/s-only graph result is not sufficient.
