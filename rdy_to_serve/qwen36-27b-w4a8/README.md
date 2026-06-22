# Qwen3.6-27B W4A8 (int4 weights / int8 activations, SmoothQuant+GPTQ, prepacked)

Serves `Qwen3.6-27B-W4A8-sqgptq-prepacked` on ONE Intel Arc Pro B70. The int8-activation / int8-XMX path
on the 27B (GDN + lm_head kept bf16 for quality). ~24 GiB, VRAM-tight, **fp16 KV** (see note below).

> [!] SECONDARY pick. The w4a16 int4-AutoRound 27B (`../qwen36-27b-int4/`) decodes FASTER (~30.8 vs
> ~20.9 t/s captured) and is less VRAM-tight. Use THIS only when you specifically want the int8-activation
> path on the 27B (e.g. int8 GEMM/GEMV research on a real 27B workload).

## Run (on the GPU host)
```bash
ssh root@192.168.10.5 && cd /mnt/vm_8tb/b70/rdy_to_serve/qwen36-27b-w4a8
/mnt/vm_8tb/b70/gpu-run bash serve.sh          # start (PIECEWISE capture), wait healthy, gen-probe
bash serve.sh stop
```
Endpoint: `http://<host>:8000/v1`. Served id: `qwen36-27b-w4a8-sqgptq`.

## Image + the three extra pieces beyond the 14B W4A8
- Image `vllm-xpu-env:int8g` (build: `../../images/int8g/`).
- PREPACK: `patches/xpu.py` + `patches/compressed_tensors_w4a8_int.py` + `VLLM_W4A8_PREPACKED=1` (local).
- GDN: Qwen3.6-27B uses gated-delta-net; the `:int8g` baked kernel ships `GDN_KERNELS_ENABLED=OFF`. The
  serve mounts the GDN-enabled `_xpu_C.abi3.so` (+ sibling `libgdn_attn_kernels_xe_2.so`) from
  `vllm-xpu-kernels/` over the baked one (a large compiled binary -> referenced by host path, not copied).
- Text-only VLM (`NOMM=1`). **fp16 KV** -- vLLM 0.23 rejects fp8 KV on this checkpoint ("fp8_e5m2
  kv-cache is not supported with fp8 checkpoints"); override `KVDTYPE=fp8_e5m2` only where accepted.

verified: _common smoke=GREEN (eager, :int8g, fp16 KV, 2026-06-23, 120s): HEALTHY, id ok, prepack +
GDN .so mount ok (loaded past the gated-delta-net decode op), gen "Paris... <think>...". Re-verify:
`bin/serve-sweep --smoke`.
