# A/B coding evals for local models (Flash-Next IQ3 vs Qwen3.8-27B INT4)

Tools, venvs, datasets, images live in /mnt/vm_8tb/b70/evals/ab/ (not in git). No eval here
touches a GPU. Served model id is `hotschmoe-dd` on both arms; ARM_LABEL names result dirs.

## One Pi configuration for BOTH arms (agentic evals)
- Harness: Harbor 0.22.0 + Pi 1.2.0 (@earendil-works/pi-coding-agent, installed in each task
  container by `--ak version=1.2.0`; host copies: /mnt/vm_8tb/b70/tooling/pi-1.2.0, pi-0.84.3).
- Adapter: evals/terminalbench/harbor_pi.py (SglangReasoningPi), unchanged. Payload oracle on
  Pi 1.2.0: off -> chat_template_kwargs {enable_thinking:false,preserve_thinking:true};
  xhigh -> {enable_thinking:true,...}; no reasoning_effort. (pi_payload_oracle.py --pi ...)
- Thinking: THINKING=xhigh (== server default) or off. Calibration (cargo-flight-dispatch): xhigh
  with MAX_TOKENS=16384 died at turn 3 (length stop, Pi ends the run); xhigh+32768 finished 23/27
  verifier tests in 50 min; off finished 13/27 in 3 min. See report; recommended: xhigh, 32768.
- Context window passed to Pi: 131072 (CONTEXT_WINDOW). One request at a time (-n 1).
- Agent timeout multiplier 1.0 (official task timeouts) unless calibration says otherwise.
- Prefix caching stays on; no nonces. Endpoint must be reachable from containers (LAN IP).

## Primary: Terminal-Bench 3, 24 pre-registered tasks (tb3_ab24_manifest.json)
  BASE_URL=http://192.168.10.5:18080/v1 API_KEY=$(cat /mnt/vm_8tb/b70/secrets/dd_api_key) \
  MODEL=hotschmoe-dd CONTEXT_WINDOW=131072 ARM_LABEL=armA THINKING=xhigh \
  evals/terminalbench/run_external_arm.sh          # add --print-config to inspect only
Report: PYTHONPATH=. <harbor python> -m evals.terminalbench.ab_report --a JOB_A --b JOB_B
(per arm: resolved, median/mean agent wall, tokens, end reasons; paired per-task table).
Harbor python: /home/hotschmoe/.local/share/uv/tools/harbor/bin/python.

## Secondary: SWE-bench Verified, 30 instances, Harbor + Pi
Same script with DATASET=swe-bench/swe-bench-verified TASK_PREFIX=swe-bench JOB_PREFIX=swe30
MANIFEST=/mnt/vm_8tb/b70/evals/ab/swe30_manifest.json (task ids are swe-bench/<instance_id>).
Subset rule (no model results consulted): Verified test split, keep difficulty in
{"<15 min fix","15 min - 1 hour"} (455), sort ids, take indices 7,22,37,... -> 30.
List: ab/swebench_verified_subset30.txt. Task images pre-pulled (pull_images.sh).
Cross-check only: evals/ab/run_swebench_mini.sh (mini-swe-agent 2.4.6, official swebench scoring).

## Optional: Aider polyglot (UNVERIFIED: image build was aborted for disk contention; 225 exercises, diff format, 2 tries)
  BASE_URL=... MODEL=hotschmoe-dd ARM_LABEL=armA THREADS=1 THINKING=xhigh|off MAX_TOKENS=16384 \
  [LIMIT=45] [LANGS=python,go] [EDIT_FORMAT=whole] evals/ab/run_aider_polyglot.sh
Pins: aider 5dc9490bb35f9729ef2c95d00a19ccd30c26339c, polyglot-benchmark
7e0611e77b54e2dea774cdc0aa00cf9f7ed6144f. LIMIT uses a seeded shuffle (same subset per arm).

## Tier1 code (HumanEval+/MBPP+): evals/.venv (Python 3.12, evalplus 0.3.1)
  evals/.venv/bin/python evals/orchestrator/run_evals.py --endpoint URL --model hotschmoe-dd \
    --quant int4 --tiers 1 [--tier1-think]      # OPENAI_API_KEY for the front door
