# qwen36-27b-w4a16-mtp

Qwen3.6-27B **W4A16** (compressed-tensors pack-quantized, 4-bit weight / 16-bit activation, **text-only**) + a
**BF16 MTP graft**, served **TP=2 + MTP spec=3, cudagraph_mode=NONE** on 2x B70. Image `:v0230` (GDN baked in).

```
GRAPH=1 CGMODE=NONE MTPTOK=3 /mnt/vm_8tb/b70/gpu-run bash serve.sh start
bash serve.sh stop
```

## Why this dir exists

It combines two proven recipes that each need a different `sitecustomize.py`, and Python imports only ONE
sitecustomize per interpreter -- so `patches/sitecustomize.py` MERGES three patches (it is the load-bearing piece):

1. **Text-only arch registration** (from `../qwen36-27b-w4a16`): register the exact text `Qwen3_5ForCausalLM`
   so vLLM does not normalize it onto the VL class and build a weightless vision tower / assert on the 4304-dim MLP.
   `qwen35_text_hybrid.py` (the marker subclass: `is_hybrid=True`, GDN state shapes, mrope, `language_model.`->`model.`
   remap) is co-located here, resolved lazily from PYTHONPATH at arch-resolve time.
2. **MTP drafter unquant** (from `../qwen36-27b-w8a8-sqgptq-mtp`): force ONLY `Qwen3_5MultiTokenPredictor` to build
   with `quant_config=None`, else the grafted BF16 `mtp.*` linears load through the W4A16 quant path -> 0% accept.
3. **csag** (capture-safe all_gather): only needed when the spec-verify all_gather is RECORDED into a graph. On
   `cudagraph_mode=NONE` there is no capture, so the base oneCCL all_gather runs eagerly and is correct -> serve.sh
   sets `CSAG_DISABLE=1` on NONE (the merged shim then skips it). A `CGMODE=PIECEWISE` experiment flips it back on.

## Config notes

- **`CGMODE=NONE` is mandatory for sustained MTP** here: PIECEWISE crashes ~20-28k tokens (XPU graph-replay
  command-stream accumulation; the W8A8 analog, campaign 120 / `docs/20260625_w8a8_27b_mtp_graph_campaign.md`).
  MTP_TODO M4's "W4A16 TP=2 MTP-on CRASHES (spec-allgather not graph-capturable)" was the PIECEWISE-capture crash;
  NONE sidesteps it (eager all_gather).
- `IMG=v0230` bakes GDN in -> **no `.so` mount** (unlike the `:int8g` W8A8 recipe). No `NOMM` (text-only class is
  not multimodal). `COMPILESZ` empty (required for spec-decode). `UTIL=0.90` (TP=2 splits the model ~12 GiB/card).
- The W4A16-graft + MTP combo was validated 2026-06-23 (accept 68.75%, accept_len 3.75).

## Verify

`docker logs vllm_qwen36-27b-w4a16-mtp` should show: `registered Qwen3_5ForCausalLM -> text-only hybrid class`,
`Qwen3_5MultiTokenPredictor forced unquantized`, `csag DISABLED (CSAG_DISABLE=1)`, `Detected MTP model`, and
SpecDecoding metrics with a nonzero acceptance length. A degenerate `!!!!` reply = arch-reg/remap not applied
(`b70_gen_probe` flags it GARBAGE and fails `start`).
