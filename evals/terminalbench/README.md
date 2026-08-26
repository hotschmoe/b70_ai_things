# TerminalBench 3 with Pi and local SGLang

This directory contains the Harbor adapter used to expose local Qwen-style
reasoning models to Pi. Harbor's stock Pi adapter defines custom models without
reasoning metadata, which causes Pi to clamp `--thinking xhigh` to off.

`SglangReasoningPi` declares the Qwen chat-template thinking format and maps Pi
`xhigh` to the model's native thinking mode. It deliberately does not send an
unsupported OpenAI `reasoning_effort` field.

For Ornith agent work, use eager decode. The refreshed breakable graph is the
short-context performance winner, but two long agent trajectories aborted in
`torch.xpu.graphs.replay` at about 17.7K live tokens. Do not use that graph path
for a full TerminalBench run until the replay failure has an isolated fix.

Start the exact eager server under the GPU lease:

```bash
./bin/gpu-run bash
NAME=sglang_ornith15_tb3 PORT=18080 CTX=65536 MAXREQ=1 \
  MEMFRAC=0.90 MTP=0 DENSE_NATIVE=0 DECODE_GRAPH=0 \
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
