# Terminal-Bench 3 campaign relaunch plan

Status: H01-H07 harness validity passed on 2026-08-29. Relaunch remains blocked
for the NVFP4 and GPTQ arms on graph-safe runtime recipes; the stable Qwen and
Ornith W8A8 routes may proceed through the ordered one-task policy gate.

Do not start another official pilot or a full campaign until the preflight
checklist below passes. The historical jobs and README remain useful evidence,
but two labels used in that evidence were wrong.

## Corrections to historical interpretation

### The 2026-08-29 thinking-off job was not thinking-off

`harbor_pi.py` maps `off` to JSON null. Pi 0.84.3 treats a null thinking-level
mapping as unsupported, clamps `off` upward, and sends
`chat_template_kwargs.enable_thinking=true`. The transcript's final 4,096-token
thinking block confirms that behavior.

Reclassify job
`tb3-qwen-w8a8-reclaim500-20260829-bun-off-max4096` as native thinking with a
4,096-token hard response cap. It is not evidence that true thinking-off fails,
and it does not justify raising the true-off cap without first fixing the
adapter.

### The retained GPTQ results used FP16 KV, not BF16 KV

`vllm/gptq_int4/serve_qwen38_gptq_int4_v0272.sh` launches with
`--dtype float16` and leaves `kv_cache_dtype=auto`. The preserved runtime log
reports `dtype=torch.float16, kv_cache_dtype=auto`. The served ID, lifecycle
metadata, journal text, and 45 tok/s qualification therefore overstate the
cache dtype.

Treat every retained Qwen GPTQ INT4 fit, exactness, speed, and Terminal-Bench
result as an FP16-KV result. Requalify the route from target-only under
`--dtype bfloat16`; leave KV dtype on auto only if the runtime log proves
`torch.bfloat16`. Do not pass an unsupported `--kv-cache-dtype bfloat16` enum.

## Preflight work before any GPU pilot

H01-H07 passed on 2026-08-29. The exact Pi 0.84.3 payload, launcher policy,
observed identity/dtype, trajectory replay, mock lifecycle, and local-70 lock
are automated by `evals/terminalbench/phase0_preflight.sh`.

1. Repair the Pi model metadata.
   - Make `off` a supported, non-null level.
   - Null unsupported `minimal`, `low`, `medium`, `high`, and `max` levels.
   - Keep `xhigh` mapped to `xhigh`.
2. Add a Pi 0.84.3 request-payload oracle that requires:
   - `thinking=off` sends `chat_template_kwargs.enable_thinking=false`.
   - `thinking=xhigh` sends `chat_template_kwargs.enable_thinking=true`.
   - Neither route sends the unsupported OpenAI `reasoning_effort` field.
3. Make the server thinking cap policy-dependent.
   - True off must start without `THINKCAP` and strict-thinking grammar.
   - Xhigh may retain the explicitly recorded private-thinking cap.
4. Fix campaign evidence collection.
   - Record final Pi stop reason, especially `length`.
   - Record tool-call count, whether an edit occurred, and whether a post-edit
     test ran.
   - Check endpoint health and scan fatal server markers before teardown.
   - Record the observed KV dtype from the runtime, not a hard-coded claim.
   - Start total machine-occupation timing before pre-health and stop it after
     teardown and post-health.
5. Add an exact `/v1/models` identity assertion to the GPTQ launcher.
6. Append corrections to `JOURNAL.md`; do not rewrite historical entries.

No scored result is campaign-valid until these checks are automated.

## Ordered qualification and pilot plan

### 1. Qwen3.8 W8A8 policy calibration

Use the already stable runtime:

```text
backend=SGLang
tp=2
p2p=off
kv=BF16
context=65536
memfrac=0.70
max_running_requests=1
decode=target-only
graph=breakable
graph_batch=1
graph_reclaim=500
thinking=off, verified by payload oracle
max_tokens=8192
task=terminal-bench/bun-sourcemap-leak
```

Do not use the rejected FULL arm. The pilot passes the policy gate only if:

- exact served identity is observed;
- the agent edits within about ten minutes;
- at least one relevant post-edit test runs;
- Pi does not end on `length`, timeout, or connection failure;
- a verifier result is produced;
- the endpoint is healthy immediately before teardown;
- teardown, card health, and compiled P2P-off collective health pass.

A zero reward with a normal verifier result is model-quality evidence. A zero
caused by length, timeout, or endpoint death is not. If 8,192 tokens ends on
`length` before an edit, try 12,288 once. Do not keep increasing the cap against
one task.

### 2. Ornith-1.5 W8A8 policy transfer

Transfer the exact policy that passes on Qwen. Retain target-only decode,
breakable graph, reclaim500, BF16 KV, and memory fraction 0.70. Do not retry
memory fraction 0.90: that setting caused the host OOM that killed user systemd
and the tmux session. Keep MTP off until the new head is target-exact and its
long-agent runtime is separately qualified.

### 3. Qwen3.8 NVFP4 runtime repair

Never rerun the rejected FULL graph arm. Extract the quant-neutral breakable
graph and executable-reclaim support from the W8A8 overlay and load it for
NVFP4. Qualify this candidate before Harbor:

```text
backend=SGLang
tp=2
p2p=off
collective_sycl_kernels=off
kv=BF16
context=65536
memfrac=0.70
max_running_requests=1
graph=breakable
graph_batch=1
graph_reclaim=500
mtp=off
```

Require deterministic equality to the eager control, a forced replay soak that
crosses at least 50,000 output tokens, no Level Zero or kernel fault, flat-enough
late throughput, graceful teardown, and clean post-health. If breakable cannot
be qualified, use eager for score completion and record the performance cost.

### 4. Qwen3.8 GPTQ INT4 runtime repair

Start from correctness, not the failed PIECEWISE speed recipe:

```text
backend=vLLM 0.27.2
tp=1, card 0
dtype=bfloat16
kv_cache_dtype=auto, observed as torch.bfloat16
context=65536
prefill_window=16384
memory_utilization=0.90
max_sequences=1
graph=eager
mtp=off
draft_lm_head_int4=off
```

First require target-only repeat exactness, the same-stack corpus, 65K cache
fit, a 48K prefill canary, and a long forced-decode canary. Then test eager
MTP4 plus the INT4 draft LM head against that new BF16 target corpus. Use MTP
only if it stays target-exact and long-agent stable; otherwise run
Terminal-Bench target-only eager. PIECEWISE plus reclaim may be optimized only
after a score-completing eager path exists.

## Expansion gates

After the Bun policy pilot completes normally on all four arms, run the same
three-task canary on every arm:

1. `bun-sourcemap-leak` for long tool use and replay pressure.
2. `production-planning` for a nontrivial operations task.
3. `sglang-qwen-burst` for longer ML and tool use.

Proceed only when all three produce verifier results without infrastructure
loss and at least one task has nonzero reward. Keep the task order, Pi version,
prompt, thinking policy, output limit, context, concurrency, and BF16 KV policy
matched across arms.

Until cancellation propagation is proven, run one task per fresh server
lifecycle. Then use deterministic, resumable shards of roughly four to six
tasks. Never start four monolithic jobs after a single successful pilot.

## Local-70 and official-74 boundary

Four Terminal-Bench 3.0.0 tasks require an H100 task environment:

- `exam-pdf-eval`
- `fp8-rmsnorm-gemm`
- `jax-speedrun-gpu`
- `math-eval-grader`

The local B70 Docker environment cannot run those tasks unchanged. A local
campaign must contain the other 70 tasks and be labeled `TB3-local-70`. An
official 74-task result requires remote H100 Harbor workers with a reachable
model endpoint. Do not override the GPU requirement to zero and call the result
official.

The task files set the following serial timeout ceilings per arm:

- all 74 agent phases: 201.69 hours;
- all 74 agent plus verifier phases: 226.17 hours;
- all 74 agent, verifier, and environment-build phases: 254.34 hours;
- local 70 agent plus verifier phases: 201.67 hours;
- H100 four-task agent plus verifier phases: 24.50 hours.

Actual successful runs should be shorter, but these ceilings make checkpointed
shards mandatory.

## Final reporting contract

For every arm report:

- reward mean and reward sum, aggregated by task rather than shard mean;
- unique tasks accounted for and whether the result is local-70 or official-74;
- normal completions, agent timeouts, length stops, and infrastructure failures;
- summed agent, verifier, Harbor, startup, and safe-lifecycle time;
- full machine-occupation time from pre-health through post-health;
- engineering and failed-retry time separately from accepted evaluation time;
- served identity, image/runtime identity, observed KV dtype, policy, prompt,
  context, graph mode, and teardown health.

Do not rank model quality or speed using the existing four zero-score pilots.
Three ended through runtime failure and the two stable arms timed out under the
verbose native-thinking policy. Their elapsed values are failure or timeout
times, not successful completion times.
