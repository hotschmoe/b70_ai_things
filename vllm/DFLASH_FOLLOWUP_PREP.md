# DFlash follow-ups 1-2 -- prep (2026-07-03)

Prep for DFLASH_XPU.md follow-ups (1) accept telemetry on a real coding workload and
(2) drafter W8A8 quant. Two runnable scripts + the metric/feasibility findings. NO GPU was
touched here (CPU-only image runs without --device, plus host-side syntax/logic validation).

## Deliverables

- `vllm/dflash_accept_probe.py` -- spec-decode accept telemetry, MTP vs DFlash A/B.
- `vllm/quant_dflash_drafter.py` -- data-free RTN W8A8 int8 of the drafter (calibrated is
  infeasible -- see verdict below).
- this note.

---

## TASK 1 -- accept telemetry probe

Drives a running OpenAI server with a cumulative 8-turn (up to 10) realistic coding
conversation (deterministic, temperature 0, 400 tok/turn), scrapes `/metrics` before/after
each turn, and prints per-turn tok/s + acceptance stats and a cumulative row incl. the
per-position accept vector.

### Metric names found (vLLM v0.24.0 image `vllm-xpu-env:int8g-v0240`)

From `vllm/v1/spec_decode/metrics.py` (prometheus_client appends `_total` to Counters):

- `vllm:spec_decode_num_drafts_total`
- `vllm:spec_decode_num_draft_tokens_total`
- `vllm:spec_decode_num_accepted_tokens_total`
- `vllm:spec_decode_num_accepted_tokens_per_pos_total{position="i"}` (per draft position)

IMPORTANT: DFlash uses this SAME counter family as MTP. It is NOT on the diffusion metric
path -- `is_diffusion` (metrics.py) comes from `model_config.is_diffusion`, which flags a
*target* dLLM, and is not set by the dflash draft method. So the probe works identically for
MTP and DFlash serves. Verified: the only `is_diffusion` writers are target-model config, not
the spec/dflash path.

Definitions the script prints:
- `acc_len` (acceptance_length) = accepted/drafts + 1 (mean tokens emitted per verify step; the
  +1 is the always-emitted target bonus token). This is the number to compare vs MTP.
- `acc/draft` = accepted/drafts (mean spec tokens accepted per step).
- `acc_rate` = accepted/draft_tokens (fraction of PROPOSED tokens accepted).

### Commands for the orchestrator

First get the served id:

    curl -s http://192.168.10.5:18080/v1/models | python3 -m json.tool

Then, against whichever serve is up (MTP or DFlash), same invocation -- just swap `--label`:

    python3 vllm/dflash_accept_probe.py \
      --base-url http://192.168.10.5:18080/v1 \
      --model <served-id> \
      --api-key "$B70_API_KEY" \
      --turns 8 --max-tokens 400 --label dflash-spec15

Run it once per serve config (MTP spec=3; DFlash spec={5,7,11,15}) and compare the ALL-row
`acc_len` and tok/s. Crossover decision = which gives higher tok/s at its acc_len on THIS
coding workload (the cold random-text bench that gave 19.06 vs 30.24 is not representative).

Notes:
- Stdlib only (urllib) -- runs on host python, no venv.
- `--api-key` optional (DD enforces a key; pass `$B70_API_KEY`).
- If a serve exposes no spec counters, spec columns print `-` and it still reports tok/s.
- Per-turn deltas isolate each turn; the cumulative row + per-position vector summarize the run.

---

## TASK 2 -- drafter W8A8 quant

### VERDICT: calibrated SmoothQuant+GPTQ is INFEASIBLE. Route = data-free RTN W8A8. int8, valid.

Two independent blockers to the GPTQ pipeline used for our 27B target
(`scripts/49_quantize_27b_w8a8.sh`, llmcompressor `oneshot` + `SmoothQuantModifier` +
`GPTQModifier`, ultrachat calib):

1. **No HF modeling class ships with the drafter.** `models/files/qwen3.6-27b/dflash-draft/`
   contains only `config.json`, `model.safetensors`, README, assets. config.json declares
   `architectures=["DFlashDraftModel"]` and `auto_map: {AutoModel: "dflash.DFlashDraftModel"}`
   but there is **no `dflash.py`** in the dir. So `trust_remote_code=True` has nothing to
   import, and `model_type: "qwen3"` does not match the DFlash state dict (extra `fc`,
   `hidden_norm`, no `lm_head`/`embed_tokens`). llmcompressor/transformers cannot instantiate
   it. (The working DFlash modeling code lives IN-TREE in vLLM:
   `vllm/model_executor/models/qwen3_dflash.py` + `vllm/v1/spec_decode/dflash.py` -- not a HF class.)

2. **The drafter's real input is TARGET hidden states, not text.** Its forward combines hidden
   states tapped from 5 target layers `[1,16,31,46,61]` through `fc: Linear[5120, 25600]`
   (25600 = 5x5120) via `combine_hidden_states`. A calibration forward THROUGH the drafter alone
   is impossible without co-running the full 27B target and plumbing per-layer hidden states --
   which llmcompressor's `SequentialPipeline` does not do. There is no clean calibration signal.

RTN (round-to-nearest) needs neither a modeling class nor a calibration forward: it quantizes
the raw safetensors weights directly. `quant_dflash_drafter.py` does exactly this and emits
compressed-tensors **"int-quantized" W8A8** that is byte-format-identical to our
`w8a8-sqgptq` target (verified against
`models/files/qwen3.6-27b/w8a8-sqgptq`): `.weight` -> I8 `[out,in]`, `.weight_scale` -> BF16
`[out,1]`, per-channel symmetric weights + dynamic per-token int8 activations. vLLM's
compressed-tensors loader + the B70 int8 XMX path consume it unchanged.

Quality note: RTN loses the SmoothQuant outlier smoothing, so accept-length may drop slightly
vs a bf16 drafter. But spec decoding is **lossless w.r.t. the target** -- the W8A8 target
verifies every proposed token, so drafter RTN error can only cost a little accept-length, never
output correctness. The win is halving the 3.3 GB bf16 drafter weight read per draft pass
(DFLASH_XPU.md hypothesis 2). If RTN accept-length proves too low, the only calibrated
alternative is a custom harness that runs the target to dump layer-tap hidden states and feeds
them to a hand-rolled GPTQ over the drafter linears -- a separate build, not a one-shot.

### What is quantized

- 35 int8 linears = {q,k,v,o}_proj + {gate,up,down}_proj across all 5 layers.
- 23 tensors kept bf16 = all norms (input/post_attention layernorms, q_norm, k_norm,
  hidden_norm, final norm) + `fc` (the target-hidden-state adapter, kept bf16 by default as the
  accuracy-critical input projection; set `QUANT_FC=1` to also int8 it).
- No `lm_head`/`embed_tokens` in the checkpoint (drafter shares the target's embedding/LM-head).
- `config.json` gets a `quantization_config` (ignore `re:.*fc.*` when fc stays bf16);
  `dflash_config`, `block_size`, `auto_map`, `architectures` are asserted-preserved.

Validated in the CPU container on the REAL weights: per-channel int8 of a gate_proj gave
mean-rel reconstruction error 0.96%, dtypes/shapes I8`[17408,5120]` + BF16`[17408,1]`
(match the target checkpoint), safetensors save/load round-trips.

### Command for the orchestrator (CPU-only, NO --device, NO gpu lease)

    docker run --rm --entrypoint bash \
      -e CUDA_VISIBLE_DEVICES="" -e ZE_AFFINITY_MASK="" \
      -v /mnt/vm_8tb/github/b70_ai_things:/repo -w /repo \
      vllm-xpu-env:int8g-v0240 \
      -c 'python3 vllm/quant_dflash_drafter.py'

Output: `models/files/qwen3.6-27b/dflash-draft-w8a8-rtn/` (~1.9 GB). Needs torch+safetensors
only (both in the image). A few minutes; loads ~3.3 GB into CPU RAM.

NAMING: output is tagged **`-rtn`**, NOT `-sqgptq`. The task memo said `dflash-draft-w8a8-sqgptq`,
but per CLAUDE.md "Model Identity", output dirs must encode the REAL method, and this is RTN
(no GPTQ, no SmoothQuant -- both infeasible above). Mislabeling RTN as sqgptq is exactly the
method-mixup the repo rule guards against. Override with `OUT=...` if a different path is wanted,
but keep the method tag honest.

### Serving the quantized drafter

The draft `config.json`'s `quantization_config` is auto-detected by vLLM's compressed-tensors
loader (the draft model is a full `ModelConfig`). Confirmed in
`vllm/config/speculative.py`: the `quantization` field ("Quantization method that was used to
quantize the draft model weights ... takes effect when using the draft model-based speculative
method") applies to the draft-model load path -- so an explicit override is available but not
required. Serve exactly as DFLASH_XPU.md, pointing the draft `model` at the new dir:

    SPEC='{"method":"dflash","model":"/models/qwen3.6-27b/dflash-draft-w8a8-rtn","num_speculative_tokens":15}' \
    CAPSIZES="1,2,4,8,16" PREFIXCACHE=0 ./bin/gpu-run bash rdy_to_serve/vllm/qwen36-27b-w8a8/serve.sh start

(Wedge discipline from DFLASH_XPU.md still applies: DFlash TP=2 inits have crashed once; one
attempt, verify `bin/xpu-health` between starts, never chain.)

---

## Files

- `/mnt/vm_8tb/github/b70_ai_things/vllm/dflash_accept_probe.py`
- `/mnt/vm_8tb/github/b70_ai_things/vllm/quant_dflash_drafter.py`
- `/mnt/vm_8tb/github/b70_ai_things/vllm/DFLASH_FOLLOWUP_PREP.md`
