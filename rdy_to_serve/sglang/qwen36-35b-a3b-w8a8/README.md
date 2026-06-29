# Qwen3.6-35B-A3B Quark W8A8 INT8 MoE -- sglang-XPU, 2x B70, TP=2

The FIRST sglang serve of the int8 W8A8 MoE (256 experts, top-8, shared expert + GDN hybrid attention),
and the first sglang 35B entry. Route A from `research/w8a8/MOE_UNBLOCK.md`: the 256 routed experts run
**TRUE int8** through sglang's in-tree Triton `fused_moe` (`use_int8_w8a8`) -- NO custom kernel build.

## Run
```bash
/mnt/vm_8tb/b70/gpu-run bash serve.sh start   # serve TP=2, coherence-gated, leave serving
bash serve.sh bench                            # c1+c4 regime bench + single-stream soak
bash serve.sh stop                             # stop + release the GPU
/mnt/vm_8tb/b70/gpu-run bash serve.sh run      # start + bench + stop in one lease
```
Endpoint: `http://192.168.10.5:30000/v1`. Served id: `qwen36-35b-a3b-quark-w8a8-int8` (greedy on XPU).

## Verified (TP=2, eager, IN2048/OUT128 warm)
| metric | c1 | c4 |
|---|---|---|
| TTFT | **272 ms** | 637 ms |
| per-stream decode | 7.94 t/s | 5.55 t/s |
| aggregate out | 7.84 t/s | 23.87 t/s |

Single-stream soak (2000 tok): decode **8.26 t/s STABLE** (windows 8.27/8.25/8.24/8.25/8.29, first/last
1.00x -- no degradation), coherent throughout. KV cache ~1.04M tokens. Box HEALTHY after.

- **TTFT is best-in-class for the 35B** -- the int8-XMX prefill win (probe: int8 prefill 1.43x bf16).
- **Decode is eager-slow** (~8 t/s): memory-bound, no graph capture, no MTP, dense linears dequant->bf16.
  The decode levers (XPUGraph capture, NEXTN MTP, fused int8 dense) are open follow-ups, same as the
  27B W8A8 path got. This entry is the COHERENT correctness+prefill baseline, not yet a decode champ.
- vs vLLM Quark W8A8 (43.1 c1 @ GRAPH=1): sglang trades raw decode for the production scheduler that
  does NOT co-batch prefill+decode the way that risks the vLLM GDN "!!!!" poison (SHORTCOMINGS.md).

## The unblock chain (what this entry mounts -- `research/w8a8/SGLANG_MOE_PLAN.md`)
All mount-not-bake over `sglang-xpu:mtp` (no image rebuild):
- `sglang/patches/int8_actquant_xpu.py` -- XPU-safe per-token int8 activation quant. The stock
  `int8_kernel.py` uses `tl.extra.cuda.libdevice.round`, which does NOT link on triton-xpu
  (`ZE_RESULT_ERROR_INVALID_MODULE_UNLINKED`); replaced by round-half-away `tl.floor`/`tl.ceil`.
- `sglang/patches/quark_moe_int8.py` -- `Int8MoEMethod` (routes 256 experts to the in-tree int8
  Triton fused_moe) + `Int8DequantLinear` (dense int8->bf16 at load) + a `QuarkConfig.get_quant_method`
  monkeypatch (stock sglang Quark only dispatches FP8/MXFP4). Also a
  `FusedMoE._load_per_channel_weight_scale` unsqueeze: Quark stores 1-D `[N]` weight scales but sglang
  wants `[N,1]`. Dense linear scales are likewise registered 1-D (the GDN merged in_proj_* loaders strip
  the ChannelQuantScaleParameter reshape).
- `sglang/images/sglang-xpu-mtp/woq_shim.py` -- the `.pth`-auto-imported hook; `B70_QUARK_MOE_INT8=1`
  runs `quark_moe_int8.install()` in EVERY process (the model builds in the TP workers, so a
  main-process-only wrapper would miss them).

## Recipe details (baked into serve.sh)
- TP=2 (int8 ~35GB -> ~17.9 GiB/card; load 32s/rank). `--device xpu --attention-backend intel_xpu
  --linear-attn-backend triton`. GDN-safe: `--mamba-ssm-dtype float32`, `--skip-server-warmup`,
  `--disable-cuda-graph`, `--disable-overlap-schedule`, `--page-size 64`, `--disable-radix-cache`.
- Health note: sglang's `/health` runs a generative ping with a 20s timeout; the FIRST forward
  JIT-compiles the int8 MoE Triton kernel and exceeds 20s, so `/health` returns 503 for ~3-4 min while
  the kernels compile+cache, then flips to 200. serve.sh waits this out (200-step health loop).
- Text-only serve (the quant excludes `visual.*`); a vision graft is a follow-up (cf. the 27B path).

verified: coherence gate GREEN + regime bench (2026-06-29). Re-verify: `bash serve.sh smoke`.
