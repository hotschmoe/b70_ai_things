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
- `qwen-nvfp4`: Qwen3.8 RadixArk NVFP4, SGLang TP2 FULL graph.
- `qwen-gptq-int4`: Qwen3.8 GPTQ INT4, vLLM 0.27.2 TP1 MTP4.
- `ornith-w8a8`: Ornith W8A8, SGLang TP2 breakable graph with reclaim500.

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
