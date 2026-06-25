# Ornith-1.0-35B -- model notes + serve params (to test later)

**Repo:** `deepreinforce-ai/Ornith-1.0-35B` (HF, public/ungated, BF16) -- **Added:** 2026-06-25
**Status:** DOWNLOADING (`scripts/121_download_ornith_35b.sh`, detached) -> `models/deepreinforce-ai_Ornith-1.0-35B`.
NOT yet served or evaluated on our box. Quant plan: QUANTS_TODO.md "Ornith" block (O1 W4A16, O2 W8A8, O3 W4A8).

## Why we want it

A Qwen3.5-based **35B MoE coder**. We historically liked the Qwen 35B MoE's speed but its accuracy/eval was
weak. Ornith is the candidate replacement -- maker-reported gains over Qwen3.5-35B on the evals we care about:

| eval | Ornith-1.0-35B | Qwen3.5-35B |
|---|---|---|
| SWE-bench Verified | 75.6 | 70 |
| Terminal-Bench 2.1 (Terminus-2) | 64.2 | 41.4 |
| NL2Repo | 34.6 | 20.5 |

(Maker's numbers, unverified by us. VALIDATE with our own coding eval before trusting -- read generated text,
do not rely on a token-count bench; see the Q8 false-positive lesson in QUANTS_TODO.)

## Architecture (confirmed from the downloaded config.json, 2026-06-25)

- `architectures: ["Qwen3_5MoeForConditionalGeneration"]`, `model_type: qwen3_5_moe` -- SAME family as our
  Qwen3.6-35B-A3B, so the Q6/Q7 (35B MoE) quant + serve playbook applies.
- **MoE:** `num_experts: 256`, `num_experts_per_tok: 8` (top-8), `moe_intermediate_size: 512`. **Has a SHARED
  expert** (`shared_expert_intermediate_size: 512`) -- a DIFFERENCE from our Qwen 35B MoE (which has none); the
  fused-MoE kernel path with shared experts differs slightly (cf. DeepSeek-V2-Lite note in QUANTS_TODO sec 7).
- **Layers:** `num_hidden_layers: 40`, `hidden_size: 2048`, `head_dim: 256`, heads 16 / kv 2.
- **Hybrid attention (GDN/DeltaNet):** `layer_types` = `linear_attention` x30 + `full_attention` x10 (every 4th,
  `full_attention_interval: 4`). `attn_output_gate: true`. -> the selective-SmoothQuant mapping (QUANTS Q0:
  map only full-attn + MLP layers, SKIP `linear_attn`) is required, same as the Qwen3.5 hybrids.
- **MTP:** `mtp_num_hidden_layers: 1` (`mtp_use_dedicated_embeddings: false`) -- **it HAS an MTP head.** Keep it
  BF16 in every quant (your ask + QUANTS gotcha #3); never quantize `mtp.fc`.
- **VLM:** vision + video (`vision_config`, `image_token_id`, `video_token_id`, video_preprocessor present).
  Vision `intermediate_size: 4304` -- the SAME odd dim that trips the group-128 int4 kernel (docs/kernel/15);
  keeping vision BF16 (ignore list) sidesteps it. Needs `trust-remote-code` + our VLM-config graft to serve.
- `tie_word_embeddings: false`, `vocab_size: 248320`, `max_position_embeddings: 262144`,
  `transformers_version: 5.8.1` (needs a recent transformers).

## Maker's suggested serve command (VERBATIM, from the HF model card)

```bash
vllm serve deepreinforce-ai/Ornith-1.0-35B \
    --served-model-name Ornith-1.0-35B \
    --tensor-parallel-size 8 \
    --host 0.0.0.0 --port 8000 \
    --max-model-len 262144 \
    --gpu-memory-utilization 0.90 \
    --enable-prefix-caching \
    --enable-auto-tool-choice --tool-call-parser qwen3_xml \
    --reasoning-parser qwen3 \
    --trust-remote-code
```

## Our 2-card B70 adaptation (to try once a checkpoint exists)

The maker assumes an 8-GPU host. On our dual-B70 box:
- `--tensor-parallel-size 2` (we have two cards, not eight). The BF16 35B (~70 GB) will NOT fit one 32 GB card
  -> TP=2 minimum for BF16; a W4A16/W4A8 quant may fit differently (measure).
- Route through the GPU lease: `./bin/gpu-run vllm serve ...` (TP=2 touches both cards). NEVER set
  `CCL_TOPO_P2P_ACCESS=1` in a TP>1 serve (wedges the box -- AGENTS.md / P2P_GPU.md).
- Image: it is a `qwen3_5_moe` GDN+MoE VLM -- expect `:v0230` (GDN kernel) for BF16/W4A16, and
  `intel/llm-scaler-vllm` for the int8 MoE serve (docs/kernel/20). Confirm `qwen3_xml` tool parser +
  `qwen3` reasoning parser exist in whichever image we use (they are recent vLLM additions).
- Keep `--max-model-len 262144` only if KV fits; on 2 cards expect to cap it lower (the 27B serves at
  131072). Start smaller (e.g. 32768) for a smoke, then push.
- `--trust-remote-code` is required (custom qwen3_5_moe modelling code).

## Next steps

1. Download finishes -> inspect `model.safetensors.index.json` for exact `mtp.*`, vision, expert, and
   `gate`/`router` module names (the config gives shapes, the index gives the leaf names the IGNORE list needs).
2. BF16 smoke serve on TP=2 (small max-model-len) to confirm the arch loads + the tool/reasoning parsers work.
3. Produce O1/O2/O3 (QUANTS_TODO) -- GPU-gated; queue behind the current GPU agent.
