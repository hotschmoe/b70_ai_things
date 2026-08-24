# Pi + Terminal-Bench 3.0 model-selection campaign

**Status:** active; Ornith W8A8 qualified and its three-task Pi smoke is running
**Decision:** choose the local coding product by correct tasks completed over
wall time, not by isolated tokens/second.

## 1. The three exact arms

| arm | backend and quant | target serving contract |
|---|---|---|
| `qwen36-w8a8` | Sglang, Qwen3.6-27B W8A8 SQ-GPTQ | TP=2, NEXTN MTP10, radix prefix cache, 131072 context, served id `qwen36-27b-w8a8-mtp` |
| `qwen38-xl` | llama.cpp SYCL, Qwen3.8-27B Unsloth UD-Q4_K_XL | TP=2, embedded MTP3, prompt/KV reuse, 262144 context, exact GGUF SHA already pinned, public id `hotschmoe-dd` |
| `ornith15-w8a8` | Sglang, Ornith-1.5-35B-A3B INT8 W8A8 | TP=2, trained BF16 MTP head, radix prefix cache if coherence-qualified, 262144 context, served id `ornith-1.5-35b-a3b-W8A8-rtn-mtp-shisa` |

This is a product-config comparison, not a same-base quant ablation. Each arm
uses its best coherent MTP and caching configuration. Every result directory
encodes the complete arm identity even though the llama.cpp API retains the
legacy `hotschmoe-dd` alias.

## 2. Ornith 8-bit decision

Hugging Face was searched on 2026-08-24. There is no public Sglang-compatible
INT8 W8A8 Ornith-1.5 checkpoint. The available names must not be conflated:

- Official FP8 and mixed NVFP4/FP8 are 8-bit floating-point families aimed at
  Blackwell-style kernels, not B70 INT8 XMX W8A8.
- Official GGUF Q8_0 is an honest 8-bit weight artifact, but it is W8A16 in
  llama.cpp. It does not exercise dynamic INT8 activations or the XMX W8A8 path.
- MLX 8-bit is Apple-specific.
- NInfer is a CUDA engine and its public Ornith artifact is mixed Q4/Q5/Q6/W8,
  not uniform W8A8.

The selected source is
`shisa-ai/Ornith-1.5-35B-A3B-MTP` at revision
`779a91ed5b7597bc6db383d9fffb4343b67892ea`. Its target shards preserve the
official Ornith BF16 weights while replacing the weak native MTP payload with a
code-tuned BF16 head. The source downloads to
`models/files/ornith-1.5-35b-a3b/bf16-mtp-shisa` through the pinned manifest.

The first deployable artifact is per-output-channel symmetric INT8 weights
with dynamic per-token INT8 activations. Quantization runs on XPU under a
card-0 `gpu-run` lease; a single pinned XPU performs layer-at-a-time work
because splitting one sequential quantization pipeline across two cards adds no
useful parallelism. Vision, routers, GDN/linear-attention state machinery,
`lm_head`, and the trained MTP head stay BF16. Routed experts and eligible text
linears become INT8 in the artifact. Sglang executes routed experts through its
fused INT8 W8A8 MoE path. The current dense fallback dequantizes eligible text
linears once and computes them in BF16, so this arm is true W8A8 where it matters
most for the MoE but is not yet an all-linear fused-INT8 implementation.

Sglang is the backend because it is the project's coherent concurrent-serving
backend and already has a proven B70 Quark INT8 fused-MoE route. Current Sglang
supports compressed-tensors W8A8 INT8 for dense linears but exposes fused-MoE
INT8 only on NPU. Therefore the initial Ornith artifact uses the proven
Quark-compatible layout as a documented backend exception. A later
compressed-tensors MoE loader is useful research, but it must not block this
model-selection run.

The initial artifact is explicitly `W8A8-rtn`, not mislabeled GPTQ. It was built
on Arc XPU 0 under `gpu-run` in 426 seconds. The converter produced 30,880 INT8
weights with matching scales, quantized 32,610,713,600 elements, preserved the
exact trained MTP sidecar, and measured relative L2 error 0.008452. At 8-bit the
per-channel weight error is expected to be small, but the hard coding gate will
decide whether it is acceptable. If it trails unexpectedly, the next producer
is XPU AutoRound/AWQ or GPTQ for the sensitive dense paths while preserving the
same INT8 expert serving layout.

## 3. Benchmark and harness

Primary benchmark: the 70-task locally runnable subset of Terminal-Bench 3.0,
executed in Harbor Docker environments. Pi is the sole coding agent and talks
to the current local OpenAI-compatible endpoint at
`http://192.168.10.5:18080/v1`.

TB3 contains four tasks that require one H100: `exam-pdf-eval`,
`fp8-rmsnorm-gemm`, `jax-speedrun-gpu`, and `math-eval-grader`. They are excluded
before task limiting because this host's two Intel B70s are serving the model
and cannot satisfy a CUDA H100 task contract. Counting those as ordinary model
misses would corrupt the comparison. The result is named `TB3-local-70`, not an
official all-74 TB3 score.

Pinned harness:

- Harbor 0.22.0
- Pi 0.84.3
- Terminal-Bench 3.0, pinned to package content hash
  `sha256:a32a61879ea94eb9dc16fa1fbeb398759f0c07ca633d9d1f6aec760207036da3`
- one task at a time (`n_concurrent=1`)
- identical Pi prompt, tools, thinking level, retry policy, task order, and
  verifier for every arm

The harness lives at `agentic-eval/harnesses/terminalbench_pi/`. Run setup once:

```bash
bash agentic-eval/harnesses/terminalbench_pi/setup.sh
```

Run a live arm only through the GPU lease:

```bash
./bin/gpu-run bash agentic-eval/harnesses/terminalbench_pi/run.sh qwen36-w8a8 smoke
```

Subsets are `smoke` (first 3), `gate` (first 12), and `full` (70). The exact
ordered names are frozen in `tasks-3.0.0.txt`; the package content hash pins all
task payloads. The full local 70-task run is the headline; the smaller sets are
plumbing and cost gates, not publishable scores.

## 4. Headline metric

For a fixed total of N tasks:

`correct_completion_pct(t) = verifier-passing tasks finished by t / N * 100`

The analyzer emits the whole step curve and reports:

- final correct percentage and total wall time;
- correct tasks per wall-clock hour;
- time to 10%, 25%, and 50% correct;
- normalized area under the correct-completion curve, which rewards correct
  answers that arrive earlier;
- summed Pi agent time, verifier time, input/output/cache tokens, and infra
  errors as diagnostics.

The raw score counts an unrecovered infrastructure error as incomplete. A
model-only score excludes infrastructure errors and is reported separately.
One fixed automatic retry is allowed for infrastructure failures. Wrong verifier
results are never retried.

Container/image setup is prewarmed before timed product runs. End-to-end wall
time remains the headline, while summed agent time separates model/agent work
from verifier and environment overhead.

## 5. Qualification before scoring

Each arm must pass these gates before its Terminal-Bench timer starts:

1. `/v1/models` exact identity matches `arms.sh` and `evals/configs/models.yaml`.
2. MTP is engaged and produces coherent deterministic output. Ornith uses the
   trained Shisa head; the native Ornith head is not the campaign arm.
3. Repeated multi-turn prompts prove cache hits and lower repeated-prefix TTFT.
4. A concurrent prefill+decode coherence sweep passes even though the headline
   benchmark itself is concurrency 1.
5. Native context is advertised and a long-context retrieval probe passes.
6. A failed cache or MTP mode is disabled only if the failure is documented;
   that arm is then marked noncompliant with the requested product contract.

## 6. Run order and decision

1. Finish source download and build Ornith W8A8 on XPU.
2. Qualify Ornith load, tool calls, trained MTP, radix caching, and 262144 context.
3. Re-qualify Qwen3.6 W8A8 with `RADIX=1` and Qwen3.8 XL with `ENABLE_MTP=1`
   plus its prompt-cache contract.
4. Prewarm Harbor/Pi/task containers without scoring.
5. Run the identical 3-task smoke on all arms.
6. Run the frozen 12-task hard gate on all arms.
7. If all remain viable, run the full local 70-task suite once per arm at concurrency 1.
8. Select on the correct-completion curve and total wall time. Decode tok/s,
   cache hit rate, MTP acceptance, and tokens are explanations, not the winner.

## 7. Ornith qualification result

The exact product arm (`CTX=262144 RADIX=1 MTP=1`, TP=2, P2P access off) passed:

- exact endpoint identity and coherent generation;
- native OpenAI tool-call parsing with exact function arguments;
- trained MTP engagement (three speculative steps, four draft tokens);
- 4,129-token repeated-prefix cache gate: 7.743 seconds cold, 0.241 seconds warm;
- 250,042-token retrieval: correct early needle, 370.478 seconds cold, 5.450
  seconds warm, byte-identical outputs;
- 8/8 coherent streams under staggered concurrent prefill and decode.

The research endpoint remains live on port 18080 for the Pi smoke.
