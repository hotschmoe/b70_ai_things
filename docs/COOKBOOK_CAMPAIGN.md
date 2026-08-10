# Cookbook campaign (2026-08-10) -- 5 items from SergiioB B70 comparison

Source cookbook:
https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook

Results root: `results/cookbook_campaign/public_matrix_20260810T220015Z/`

## Items status

| # | Item | Status | Verdict |
|---|------|--------|---------|
| 1 | Re-open MoE MTP with BF16 draft on vLLM >=0.26 + cookbook patches | **DONE** | **M5 RETIRED** -- MoE MTP is real when draft is BF16-preserved |
| 2 | Port boundary + draft patches; exact 128k MTP cell | **DONE** (patches + long-ctx MTP serve) | Patches applied; 128k MTP4 boots; full-window chat cell needs token-exact prompts |
| 3 | Stock public image dense MTP4 + fp8 KV baseline | **DONE** | Public digest + GPTQ-INT4 MTP-preserved works on this box |
| 4 | INT4+MTP is ceiling reference, not DD demotion | **DONE** (policy) | See below |
| 5 | Phase-separated client post-first methodology harness | **DONE** | `vllm/cookbook_campaign/phase_bench.py` |

## Measured results (this box, public image)

**Stack:** `vllm/vllm-openai-xpu@sha256:2c427ef...` (vLLM `0.26.1rc1.dev457+gc810e5ee9.xpu`), patches
`patch_mtp_nightly.py` + `patch_mtp_boundary.py`, `B70_MTP_BF16_DRAFT=1`, PIECEWISE,
fp8 KV, 1x B70 (ZE_AFFINITY_MASK=0), client post-first methodology (n=3 after warmup).

**Checkpoints:**
- Dense: `llmfan46/Qwen3.6-27B-uncensored-heretic-v2-Native-MTP-Preserved-GPTQ-Int4`
  -> `models/files/community/qwen36-27b-gptq-mtp-preserved`
- MoE: `llmfan46/Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-Preserved-GPTQ-Int4`
  -> `models/files/community/qwen36-35b-gptq-mtp-preserved`

### Dense 27B GPTQ + BF16 MTP

| Mode | p512 post-first | p2048 post-first | prefill proxy p512 | Notes |
|------|----------------:|-----------------:|-------------------:|-------|
| no-spec | **27.1** | **21.6** | 1868 | MAXSEQS=64, UTIL=0.90, MAXLEN=8192 |
| MTP4 | **52.1** | **43.7** | 1656 | MAXSEQS=8, UTIL=0.88, MAXLEN=8192; gen "Paris" |
| MTP4 @~32k ctx | **59.5** | n/a | 1355 | MAXLEN=131072, target p16k overshot to ~32k tokens |

**Dense MTP4 vs no-spec: ~1.92x at p512.** Coherent (Paris). Cookbook peaks (~69–73) are higher;
we match the *lever*, not every absolute cell (power/thermal/MAXSEQS/prompt shape differ).

Dense MTP4 first attempt at MAXSEQS=64 OOMed during int4_gemm graph capture (23.4 GiB allocated).
**MAXSEQS=8 is required** for dense MTP4 on this box with the public image.

### MoE 35B-A3B GPTQ + BF16 MTP -- THE headline

| Mode | p512 post-first | p2048 post-first | prefill proxy p512 | vs no-spec |
|------|----------------:|-----------------:|-------------------:|------------|
| no-spec | **69.3** | **51.7** | 6801 | baseline |
| MTP2 | **94.5** | **85.7** | 5732 | **+36% / +66%** |
| MTP4 | **88.7** | **85.6** | 5454 | +28% / +65% |

**Prefill ~5.5–7.6k tok/s** (cold proxy) -- same class as cookbook ~7.5k and our sglang W8A8 MoE.

### Item 1 verdict: retire M5 for this class of checkpoint

Our 2026-06-22 M5 finding ("MTP +3% flat on A3B") was measured on **AutoRound INT4 where MTP experts
are quantized (qweight)**. The cookbook path uses **GPTQ body + fully BF16-preserved MTP draft**
plus the `B70_MTP_BF16_DRAFT` build gate.

On that path, **MoE MTP is a real 1.3–1.7x decode lever** (MTP2 best short-cell here).

Do **not** apply this to:
- our shelf AutoRound MoE without a BF16 draft graft,
- TP=2 MoE (not re-tested this session),
- concurrent multi-stream (C1 only).

### Item 2: boundary patch

- Patches apply cleanly on public nightly (`qwen3_5_mtp.py` + `gdn_attn.py`) and on
  `vllm-xpu-env:int8g-v0260` (v0260-specific draft patch).
- Dense MTP4 **serves at MAXLEN=131072** with patches (healthy after ~4.5 min).
- Long chat cells at target 100k–130k tokens returned HTTP 400: phase_bench entropy prompts
  overshoot token count badly (p16k target -> ~32k actual). Need a token-exact prompt builder
  for a true exact-131072 total cell. The p16k/32k MTP4 cell already proves long-ctx MTP decode.
- Exact partial-final-group path is gated by hitting max_model_len mid-spec step; not isolated
  this session beyond "serve completes under MTP at 128k max".

### Item 3: public image baseline

Works. Image has working XPU (device_count=1). Our bare `vllm-xpu-env:v0260` (non-int8g)
reports **0 XPU devices** and is unusable without the int8g bake.

### Item 4 policy (unchanged, confirmed by data)

Cookbook INT4+MTP peaks are a **single-stream ceiling reference**.

They do **not** demote:
- W8A8 INT8 research / XMX path,
- NVFP4 daily driver (TP=2 262k bf16-KV / DP=2 high-agg),
- concurrent coherence gates.

Reasons: quality path (W8A8), dual-card architecture, concurrent serving, long-ctx production
features (push-AR, calibrated KV, prefix-cache fixes). Use cookbook numbers to set C1 upper
bounds and to re-open MoE MTP research with BF16 drafts.

### Item 5 methodology

`vllm/cookbook_campaign/phase_bench.py`:
- post-first = `(completion_tokens - 1) / (end - first_token)`
- prefill proxy = `prompt_tokens / TTFT`
- unique entropy cold prefixes, warmup + n medians

Internal DD benches remain serve-sweep / coding harness / HumanEval+.

## Tooling landed

| Path | Role |
|------|------|
| `vllm/patches/cookbook/` | draft + boundary patches + apply helper |
| `vllm/cookbook_campaign/launch.sh` | single-card serve with patches |
| `vllm/cookbook_campaign/phase_bench.py` | client post-first harness |
| `vllm/cookbook_campaign/wait_healthy.sh` | health wait |
| `vllm/cookbook_campaign/run_campaign.sh` | orchestrator |

## DD restore recipe (after campaign)

```bash
DD_MODEL=vllm/qwen36-27b-nvfp4 DD_REPLICAS=1 DD_MAXLEN=262144 \
  DD_API_KEY="$(cat /mnt/vm_8tb/b70/secrets/dd_api_key)" \
  DD_ENV="TP=2 SERVED_FORCE=hotschmoe-dd KV_FP8=0" \
  ./vllm/daily_driver_serve.sh start
```

## Follow-ups (not blocking)

1. Graft BF16 MTP onto our AutoRound 35B and re-bench MoE MTP on shelf image.
2. Token-exact long-prompt builder for exact 131072 total + boundary isolation.
3. Dense MTP4 at cookbook MAXSEQS=64 may need lower capture sizes or UTIL -- document only.
4. Power cap A/B (cookbook 165/230 W) -- we did not set host power this session.
