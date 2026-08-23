# TP=2 optimization campaign on 2x Intel Arc Pro B70

Started: 2026-08-23

## Objective

Increase coherent, stable TP=2 inference performance by attacking measured
bottlenecks in order. Sglang compressed-tensors W8A8 is the primary serving
and kernel research lane. The stock llama.cpp Q4_K_M daily driver remains the
production service and is restored after every experiment block.

The baseline profile is `docs/20260823_tp2_inference_profile.md`.

## Rules

- Hold both cards with `bin/gpu-run` for every serve, benchmark, or GPU probe.
- Keep `CCL_TOPO_P2P_ACCESS=0`. Do not test broad peer-output writes.
- Stop the production llama.cpp server cleanly before an experiment block and
  restore its exact shelf entry before releasing the lease.
- Record config -> command -> result -> verdict in `JOURNAL.md`.
- Use position-balanced comparisons for small performance differences.
- Promote only when identity, coherence, c1, c4, TTFT, soak, fatal-log, and
  post-teardown card-health gates pass.
- Do not treat reduced device time as a serve-speed win without end-to-end
  measurements.

## Baselines

### Sglang W8A8 TP=2

- Model: Qwen3.6-27B compressed-tensors GPTQ W8A8 with NEXTN MTP10.
- Backend: sglang 0.5.6, eager, radix off, 8192 context.
- Production push gate: `PUSH_AR_MIN_NUMEL=1048576`.
- One ordered A/B signal for push-all: c1 +5.2%, c4 aggregate +0.3%, soak
  +6.1%, TTFT 1.8% slower.
- Mechanism trace: oneCCL collective 319/878 ms per five batches versus push
  8/8 ms, but only 5-6% end-to-end gain.

### Llama.cpp Q4_K_M TP=2

- Model: stock Qwen3.8-27B Q4_K_M, F16 KV, context 262144, MTP off.
- Fresh decode baseline: 37.88 tok/s.
- Estimated ideal weight traffic: about 359 GB/s/card at that decode rate.
- Existing communication path already fuses direct-Q8 all-reduce boundaries.

## Campaign ladder

### C1 - qualify eager push-all

Run a position-balanced A-B-B-A where A is the 1M production gate and B is
push-all. Each arm must use the same shelf wrapper and run:

1. identity and context check;
2. 18-stream staggered mixed prefill/decode coherence;
3. warm c1 and c4 regime;
4. 2K-token windowed soak;
5. fatal-log scan and both-card health.

Promotion candidate thresholds, applied to the position-balanced arm means:

- c1 decode and soak: at least +3.0%;
- c4 aggregate output: no worse than -2.0%;
- c1 and c4 TTFT: no worse than +3.0%;
- every arm coherent and soak first/last ratio between 0.95x and 1.10x;
- no device-lost, out-of-resources, engine-dead, NaN, or garbage marker.

Passing C1 authorizes a longer push-all qualification and shelf-gate review.
It does not by itself promote the shelf.

### C2 - remove eager host synchronization

Map and separately time the Python/ctypes call, host rendezvous, Level Zero
submission, peer copy/reduction, and completion wait. Prototype the smallest
correct replacement that avoids one host barrier per logical collective.
Preferred order:

1. event/command-stream ordered handoff without host blocking;
2. persistent device-driven sequence/ack state;
3. batched submission across boundaries where dependencies allow.

Every prototype first passes a two-rank numerical microbench with randomized
delays and many iterations, then a short real-serve mechanism trace, and only
then the C1 serving regime. Known unsafe graph-capture spin and broad peer-write
experiments remain closed unless new evidence changes their failure mechanism.

### C3 - reduce collective boundary count

Build a target-versus-draft census of every decode all-reduce. Rank candidates
for fused GEMM-output/reduce/residual/norm operations. A boundary can be delayed
or batched only when no intervening operation consumes the globally reduced
value. Compare structural TP fusion with PP=2 as a separate throughput-oriented
topology experiment.

### C4 - optimize post-push math

Use the push-all trace as the new device baseline:

- BF16 GEMM: 46.5%;
- INT8 GEMM: 20.6%;
- copy/reshape: about 8%;
- activation quantization: 6.8%;
- GDN recurrence: 1.7%.

Start with a shape/call census of BF16 MTP/draft and remaining non-INT8
linears. Kernel work must retain the compressed-tensors W8A8 serve path.

### C5 - optimize llama.cpp production lane

Profile TP=1 safely and use TP=2 source/launch census until a safe TP=2
profiler exists. Prioritize weight streaming, MMVQ tile/occupancy efficiency,
and launch fusion. Do not transfer sglang oneCCL percentages to this lane.

## Ledger

| ID | Change | Mechanism result | Serving result | Verdict |
| --- | --- | --- | --- | --- |
| C0 | Baseline profile | oneCCL 319/878 ms -> push 8/8 ms | push-all c1 +5.2%, soak +6.1%, c4 flat | Communication sync confirmed; campaign opened |
| C1 | Push-all A-B-B-A | pending | pending | running |
| C2 | Host-sync removal | pending | pending | queued |
| C3 | Boundary reduction/fusion | pending | pending | queued |
| C4 | Post-push math | pending | pending | queued |
| C5 | Llama.cpp weight/MMVQ | pending | pending | queued |

## First command

The campaign runner is `sglang/campaign_push_ar_abba.sh`. The caller must
hold both cards for its entire lifetime:

```bash
./bin/gpu-run bash sglang/campaign_push_ar_abba.sh
```

The runner snapshots the live production identity and container, stops the
exact stock Q4_K_M shelf inside the held lease, and restores it from an exit
trap only after both cards pass health. It refuses a new TP=2 start if health
is red.
