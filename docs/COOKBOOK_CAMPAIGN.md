# Cookbook campaign (2026-08-10) -- 5 items from SergiioB B70 comparison

Source cookbook:
https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook

This document tracks the five follow-ups from the comparison of that repo
against our README/JOURNAL/patches. Results land under
`results/cookbook_campaign/<UTC>/` (gitignored logs; summary JSON may be
copied into this doc).

## Items

| # | Item | Status | Artifacts |
|---|------|--------|-----------|
| 1 | Re-open MoE MTP with BF16 draft on vLLM >=0.26 + cookbook patches | IN PROGRESS | `vllm/patches/cookbook/`, campaign runner |
| 2 | Port boundary + draft patches; exact 128k MTP cell | IN PROGRESS | `patch_mtp_boundary.py`, boundary bench cell |
| 3 | Stock public image dense MTP4 + fp8 KV baseline | IN PROGRESS | `launch.sh` default IMAGE = public digest |
| 4 | INT4+MTP is a **ceiling reference**, not a DD demotion | DONE (policy) | this section + COMMUNITY_CONFIGS |
| 5 | Phase-separated client post-first methodology harness | DONE (tool) | `vllm/cookbook_campaign/phase_bench.py` |

## Item 4 policy (do not demote W8A8 / NVFP4 DD)

The cookbook's single-card GPTQ-INT4 + MTP4 peaks (~70 t/s dense, ~150+ t/s MoE
C1 post-first) are a **single-stream ceiling reference** on stock nightly.

They are **not** a daily-driver replacement for us because:

1. **Quality path:** our research target remains W8A8 INT8 XMX (and NVFP4 where
   it wins decode). INT4 W4A16 is already a shelf entry, not the primary path.
2. **Serving shape:** cookbook cells are C1; our DD is gated on concurrent
   coherence, soak, and agentic multi-turn (prefix cache, tool call).
3. **Dual card:** TP=2 push-AR / DP=2 isolation are out of scope for the
   cookbook single-GPU recipe.
4. **Correctness:** cookbook explicitly does not claim output parity across MTP
   depths. Our HumanEval+/gate is a different bar.

**Use cookbook INT4+MTP numbers to:** set a C1 upper bound, re-open MoE MTP
research if accept_len is real, and calibrate our phase_bench harness.

**Do not:** drop W8A8 research, replace the NVFP4/W8A8 DD, or chase LocalMaxxing
peaks without concurrent gates.

## Item 5 methodology

`vllm/cookbook_campaign/phase_bench.py` implements:

- **Client post-first tok/s** = `(completion_tokens - 1) / (request_end - first_token)`
- **Prefill proxy** = `prompt_tokens / TTFT` (not isolated engine prefill)
- Unique entropy cold prefixes (no intentional cache hit)
- Warmup + n timed medians
- Optional `/metrics` accept-counter delta

This is for **external-comparable** tables. Internal DD benches remain
`bin/35_sweep_bench.sh` / serve-sweep / coding harness.

## Restore DD recipe (after campaign)

Live at campaign start (2026-08-10):

```text
image:  vllm-xpu-env:int8g-v0260
model:  /models/qwen3.6-27b/nvfp4-modelopt
name:   hotschmoe-dd
mode:   TP=2, MAXLEN=262144, MTP=5, PIECEWISE, prefix cache
KV:     bf16/auto (no --kv-cache-dtype)
push-AR graph + CG reclaim 1000
port:   18080 + API key from /mnt/vm_8tb/b70/secrets/dd_api_key
```

Restore:

```bash
DD_MODEL=vllm/qwen36-27b-nvfp4 DD_REPLICAS=1 DD_MAXLEN=262144 \
  DD_API_KEY="$(cat /mnt/vm_8tb/b70/secrets/dd_api_key)" \
  DD_ENV="TP=2 SERVED_FORCE=hotschmoe-dd KV_FP8=0" \
  ./vllm/daily_driver_serve.sh start
```

## How to run

```bash
# stop DD first (holds both cards for TP=2)
./vllm/daily_driver_serve.sh stop

# single-card campaign (card 0)
./bin/gpu-run --card 0 bash vllm/cookbook_campaign/run_campaign.sh

# or manual:
bash vllm/cookbook_campaign/launch.sh dense27-gptq mtp4 on 8000 0
bash vllm/cookbook_campaign/wait_healthy.sh 8000 b70_cb_dense27-gptq_mtp4_on
python3 vllm/cookbook_campaign/phase_bench.py --base http://127.0.0.1:8000 \
  --model auto --prompt-tokens 512 --gen-tokens 128 --n 5 --out /tmp/cell.json
```

## Checkpoints

| Path | Role |
|------|------|
| `models/files/community/qwen36-27b-gptq-mtp-preserved` | Cookbook dense GPTQ + BF16 MTP |
| `models/files/community/qwen36-35b-gptq-mtp-preserved` | Cookbook MoE GPTQ + BF16 MTP |
| `models/files/qwen3.6-27b/int4-autoround` (+ `model-mtp.safetensors`) | Local dense fallback |
| `models/files/qwen3.6-35b-a3b/int4-autoround` | Local MoE (MTP experts quantized -- weak accept) |

## Results (filled after GPU campaign)

_Pending GPU runs. See latest `results/cookbook_campaign/*/summary_table.json`._
