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

Result, 2026-08-24: the core serving gate passed. Drift-balanced deltas were
c1 +7.62%, extended soak +4.42%, real-code c1 +6.36%, real-code c4 aggregate
+3.32%, and random c4 aggregate +0.34%. TTFT improved slightly and cold
prefill was flat. All 96 mixed-load streams were coherent, every long soak
was stable, no fatal marker appeared, and every card-health probe passed.

The entropy-prompt `phase_bench` diagnostic had 6.9-18.9% within-process CV
and large restart spread in both arms. It failed its own repeatability check
and is excluded from the promotion evidence rather than retroactively relaxed.
The shelf default remains unchanged pending a deterministic final shelf review.

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

Result, 2026-08-24: the existing one-host-sync implementation is a strict
no-go. Across exact 159-call BF16 sequences at rows 1/11/44, oneCCL took
133/170/373 us per call, the current two-wait push took 76/82/115 us, and the
one-host-sync safe path took 84/90/120 us. All results were numerically exact
with injected rank delay, but the proposed path was 4-10% slower than current
push. The riskier native input-event variant was not run because the safe
variant had already missed the go threshold.

### C3 - reduce collective boundary count

Build a target-versus-draft census of every decode all-reduce. Rank candidates
for fused GEMM-output/reduce/residual/norm operations. A boundary can be delayed
or batched only when no intervening operation consumes the globally reduced
value. Compare structural TP fusion with PP=2 as a separate throughput-oriented
topology experiment.

The completed census is 159 all-reduces plus 11 logits all-gathers per
scheduler decode iteration. The 159 all-reduces are target 129 (embedding 1,
attention 64, MLP 64) plus MTP10 30 (embedding, attention, and MLP once per
draft forward). Standard TP dependencies prevent batching the 128 target layer
reductions across sublayers or layers.

The first exact boundary-count prototype uses SGLang's native unsharded input
embedding for the target and shares that exact module with NEXTN. It removes
11 all-reduces without changing values for about 1.184 GiB additional storage
per card. It passed the 159 -> 148 collective census, deterministic output,
capacity, and position-balanced serving gates and is now the shelf default.
The next kernel prototype enables SGLang's existing delayed MLP all-reduce plus
residual/RMSNorm fusion contract on XPU, initially covering 63 target layer
edges.

Mechanism result, 2026-08-24: GO to serving qualification. Both candidate
ranks measured exactly 148 all-reduces plus 11 all-gathers per scheduler
iteration, versus 159 plus 11 for baseline. The removed embedding signatures
were nine `[1,5120]` and two `[11,5120]` BF16 all-reduces per iteration; no
collective metadata or logits-gather signature changed. Eight fixed greedy
responses totaling 2,048 output tokens were byte-identical. At 131072 context,
token capacity changed from 182208 to 143360 (78.68% retained), leaving 12288
tokens above the advertised context and passing the 139264-token hard gate.
The feature was promoted to the shelf default after the A-B-B-A serving gate
passed. Drift-balanced deltas were c1 +4.69%, extended soak +2.20%, real-code
c1 +3.68%, real-code c4 aggregate +3.13%, and standard c4 aggregate +1.43%.
TTFT gates were within bounds, all 96 mixed streams were coherent, all four
soaks were stable, fixed outputs were byte-identical, no fatal marker appeared,
and every card-health probe passed.

C3b contract-only result, 2026-08-24: PASS. A fail-closed XPU shim enabled the
existing delayed marker route only for dense Qwen3.5, TP=2, PP=1, BF16
contiguous `[M,5120]`, eager execution, and `M=1..128`. Both ranks reported
exactly 63 eligible, 63 consumed, and 63 generic-fallback routes on the first
target forward. The fallback remained SGLang's unchanged MoE-TP all-reduce plus
Gemma RMSNorm path. Eight fixed 128-token responses were byte-identical to the
baseline, no fatal marker appeared, and all card-health and production-restore
gates passed. This proves boundary lifecycle and group equivalence only; it
does not remove a collective, launch, or host wait and makes no performance
claim.

The active C3b kernel stage now combines push reduction, the BF16 residual add,
and Gemma RMSNorm in one post-rendezvous SYCL kernel. It must preserve two BF16
rounding points: one after the TP sum and one after adding the old residual.
The bounded prototype retains the proven push plus host rendezvous, uses a
scratch ring so an asynchronous epilogue cannot be overwritten, and targets
the same 63 delayed MLP edges before considering the 64 attention edges.

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
| C1 | Push-all A-B-B-A | 1M gate correctly fell back; push-all engaged | core gate: c1 +7.62%, soak +4.42%, code c1/c4 +6.36%/+3.32%, 96/96 coherent | GO candidate; noisy phase diagnostic excluded, shelf unchanged |
| C2 | One-host-sync immediate-list path | exact, but 4-10% slower than current push | not advanced to serve | NO-GO; native-event variant not justified |
| C3a | Native replicated target/MTP embedding | exact 159 -> 148 AR; 11 AG unchanged; byte-identical fixed corpus; 143360-token capacity | c1 +4.69%, soak +2.20%, code c1/c4 +3.68%/+3.13%, 96/96 coherent | GO; promoted to shelf default |
| C3b | Delayed MLP AR plus residual/RMSNorm | contract-only route exact on both ranks: 63 eligible/consumed/generic; fixed corpus byte-identical | performance pending | lifecycle PASS; fused SYCL primitive active next |
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
