# TerminalBench 3 with Pi and local SGLang

This directory contains the Harbor adapter used to expose local Qwen-style
reasoning models to Pi. Harbor's stock Pi adapter defines custom models without
reasoning metadata, which causes Pi to clamp `--thinking xhigh` to off.

`SglangReasoningPi` declares the Qwen chat-template thinking format and maps Pi
`xhigh` to the model's native thinking mode. It deliberately does not send an
unsupported OpenAI `reasoning_effort` field.

`run_arm.sh` is the campaign entry point. It keeps BF16 KV, one agent request at
a time, the same Pi prompt and limits, server startup, Harbor, teardown, and
post-run health inside one GPU lease. It supports these arms:

- `qwen-w8a8`: Qwen3.8 W8A8 GPTQ, SGLang TP2 FULL graph.
- `qwen-w8a8-reclaim500`: Qwen3.8 W8A8 graph-safety diagnostic using the
  previously qualified breakable backend and executable reclaim every 500
  replays.
- `qwen-nvfp4`: Qwen3.8 RadixArk NVFP4, SGLang TP2 FULL graph.
- `qwen-gptq-int4`: Qwen3.8 GPTQ INT4, vLLM 0.27.2 TP1 MTP4.
- `ornith-w8a8`: Ornith W8A8, SGLang TP2 breakable graph with reclaim500.

The vLLM arm keeps a 65,536-token model window but compiles 16,384-token
prefill chunks. Setting its batched-token limit to the full context forced a
4.25 GiB compile allocation and failed before KV-cache sizing on the 32 GiB
card. It uses memory utilization 0.90 because 0.75 left only 1.0 GiB for a
5.07 GiB BF16 KV requirement. Override `PREFILL_WINDOW` or `GPTQ_UTIL` only as
a separately recorded configuration.

Run the matched one-task pilot on each arm:

```bash
for arm in qwen-w8a8 qwen-nvfp4 qwen-gptq-int4 ornith-w8a8; do
  INCLUDE_TASK=terminal-bench/bun-sourcemap-leak N_TASKS=1 \
    evals/terminalbench/run_arm.sh "$arm"
done
```

Remove `INCLUDE_TASK` and `N_TASKS` to run all 74 tasks. Run arms sequentially;
each endpoint is qualified for one request maximum and the goal is comparable
whole-job wall time, not throughput from overlapping trials.

Each job receives `b70_lifecycle.json`, which records server startup, Harbor,
and end-to-end time through post-health teardown. Compare completed jobs with:

```bash
python3 evals/terminalbench/summarize.py \
  /mnt/vm_8tb/b70/evals/harbor-jobs/tb3-qwen-w8a8-JOB \
  /mnt/vm_8tb/b70/evals/harbor-jobs/tb3-qwen-nvfp4-JOB \
  /mnt/vm_8tb/b70/evals/harbor-jobs/tb3-qwen-gptq-int4-JOB \
  /mnt/vm_8tb/b70/evals/harbor-jobs/tb3-ornith-w8a8-JOB
```

On 2026-08-28 the reclaim500 Ornith arm crossed the old 17.7K replay failure
and remained healthy through 42,112 live tokens. It scored 0.0 on
`bun-sourcemap-leak` because Pi hit the official 1,800-second agent timeout,
not because serving failed. Harbor wall was 34m34s and full server-start through
post-health teardown was 38m39s. This qualifies the runtime fix but rejects the
current Ornith Pi/xhigh policy as an efficient recipe for that task.

The matched Qwen W8A8 pilot also scored 0.0. It completed 24 of 36 verifier
tests, but its TP2 FULL-graph endpoint suffered CCS/BCS engine resets and GPU
virtual-memory faults during the next response after a 17,309-token turn. Its
Harbor wall was 18m23s and full server-start through post-health teardown was
21m52s. Post-failure card and compiled collective health passed, as did the
mandated xe rebind recovery. Treat this configuration as campaign-unstable;
the score alone does not capture the endpoint failure.

The matched Qwen NVFP4 pilot scored 0.0 after its agent spent 10m15s planning
and never edited the task. Its TP2 FULL-graph endpoint crossed 17K tokens but
aborted in Level Zero `linear_stream.h:90` at 19,328 live tokens. Harbor wall
was 14m46s and full server-start through post-health teardown was 17m52s.
Post-failure health and xe rebind recovery passed. This arm is also unsuitable
for the full 65K-context campaign in its current graph mode.

The matched Qwen GPTQ INT4 pilot also scored 0.0. A 16,384-token prefill window
and memory utilization 0.90 were required to fit one 65,536-token BF16 KV
request; the resulting cache held 82,965 tokens. Pi emitted a 9,067-token plan,
inspected the baseline release, and made no edit before vLLM PIECEWISE aborted
in Level Zero `linear_stream.h:90`. The unchanged baseline passed 17 of 36
tests. Harbor wall was 12m25s and full server-start through post-health teardown
was 15m00s. Card and compiled collective post-health passed. Treat the qualified
short-context GPTQ recipe and this failed 65K agent recipe as separate results.

The completed matched pilot is:

| Arm | Score | Task evidence | Harbor wall | Total | Endpoint |
| --- | ---: | --- | ---: | ---: | --- |
| Qwen W8A8 | 0.0 | 24/36 passed | 18m23s | 21m52s | kernel fault at about 17K |
| Qwen NVFP4 | 0.0 | 17/36 passed | 14m46s | 17m52s | graph abort at 19,328 |
| Qwen GPTQ INT4 | 0.0 | 17/36 baseline | 12m25s | 15m00s | graph abort before edit |
| Ornith W8A8 | 0.0 | agent timeout | 34m34s | 38m39s | stable through 42,112 |

There is no score winner and no arm is ready for the 74-task campaign. The
Qwen totals are times to failed completion, not successful speed results.
Among the original four arms, Ornith is the only long-agent runtime survivor,
but its current Pi/xhigh policy does not finish this task within the official
budget.

A separate Qwen W8A8 graph-safety diagnostic replaced FULL with breakable
decode plus reclaim500. It survived the complete 1,800-second agent phase and
26,368 live tokens without an endpoint or kernel failure, then shut down with
clean card and collective health. The agent made a late, unverified edit and
timed out after its first test exposed a generated `Set.filter` bug. Official
score remained 0.0; Harbor wall was 34m32s and total server-start through
post-health time was 38m17s. This qualifies the runtime change but rejects the
Pi/xhigh policy. The next score experiment should use a matched lower-thinking
or hard-output-bound policy on the stable Qwen W8A8 and Ornith arms.

Do not raise the Ornith graph arm to `MEMFRAC=0.90`. That setting allocated
about 58 GiB of shared GPU memory and triggered a global host OOM during graph
capture, killing the user systemd/tmux session. The campaign uses 0.70, which
still provides 443,392 BF16 KV tokens per rank and leaves capture headroom.

For a manual eager diagnostic, start the server under the GPU lease:

```bash
./bin/gpu-run bash
NAME=sglang_ornith15_tb3 PORT=18080 CTX=65536 MAXREQ=1 \
  MEMFRAC=0.70 MTP=0 DENSE_NATIVE=0 DECODE_GRAPH=0 \
  TOOLPARSER=qwen3_coder THINKCAP=2048 \
  SERVED=ornith-1.5-35b-a3b-W8A8-rtn-shisa-target-eager \
  bash sglang/w8a8/serve_ornith15_w8a8_refresh.sh start
```

Run one official task with Pi 0.84.3:

```bash
PYTHONPATH=/mnt/vm_8tb/github/b70_ai_things \
OPENAI_BASE_URL=http://192.168.10.5:18080/v1 \
OPENAI_API_KEY=EMPTY \
harbor run -d terminal-bench/terminal-bench@3.0.0 \
  -i terminal-bench/bun-sourcemap-leak -l 1 \
  -a evals.terminalbench.harbor_pi:SglangReasoningPi \
  -m openai/ornith-1.5-35b-a3b-W8A8-rtn-shisa-target-eager \
  --ak model_api=openai-completions --ak thinking=xhigh \
  --ak version=0.84.3 --ak context_window=65536 --ak max_tokens=16384 \
  --ak prompt_template_path=/mnt/vm_8tb/github/b70_ai_things/evals/terminalbench/pi_concise_prompt.j2 \
  --allow-agent-host 192.168.10.5 -n 1 -k 1 \
  -o /mnt/vm_8tb/b70/evals/harbor-jobs --job-name JOB_NAME --yes
```

`SGLANG_MAX_THINK_TOKENS` is a soft limit on the private reasoning channel, not
a hard completion limit. Ornith can continue planning as visible text after
strict thinking closes. Treat the prompt template and thinking cap as explicit
agent configuration, and keep scored results separate from unprompted arms.

For the post-xhigh policy diagnostic, use the model's only other real thinking
state and a hard response bound:

```bash
THINKING=off MAX_TOKENS=4096 \
PROMPT_TEMPLATE_PATH="$PWD/evals/terminalbench/pi_concise_off_prompt.j2" \
INCLUDE_TASK=terminal-bench/bun-sourcemap-leak N_TASKS=1 \
  evals/terminalbench/run_arm.sh qwen-w8a8-reclaim500
```

This is a new agent-policy arm, not a matched xhigh result. Run Ornith with the
same three policy variables only if the Qwen pilot becomes task-effective.

The first such pilot proved that 4,096 is too small. Thinking off reduced the
first four inspection responses to 46-95 tokens, but the first implementation
response hit the hard bound before issuing an edit. The unchanged baseline
scored 0.0 at 17/36 tests in 10m51s Harbor wall and 14m19s total. Do not repeat
this cap or transfer it to Ornith; test Qwen with `MAX_TOKENS=8192` next.
