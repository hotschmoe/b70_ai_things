# Qwen3.6-27B int4-AutoRound (W4A16) -- PRIMARY single-card quality pick

Serves `Lorbus_Qwen3.6-27B-int4-AutoRound` (host `/mnt/vm_8tb/b70/models/...`) on ONE Intel Arc Pro B70.
This is the current **daily-driver** model (run 2x data-parallel via `../../daily_driver_serve.sh`).

## Run (on the GPU host)
```bash
ssh root@192.168.10.5 && cd /mnt/vm_8tb/b70/rdy_to_serve/qwen36-27b-int4
/mnt/vm_8tb/b70/gpu-run bash serve.sh          # start (PIECEWISE capture), wait healthy, gen-probe
bash serve.sh stop                              # stop + release the GPU
GRAPH=1 /mnt/vm_8tb/b70/gpu-run bash serve.sh run   # serve + concurrency sweep + stop, one lease
```
Endpoint: `http://<host>:8000/v1`. Served id: `qwen36-27b-int4`.

## Image: `vllm-xpu-env:v0230` (vLLM 0.23.0)
Plain v0230 serves int4 AutoRound -- **no runtime patch**. The image needs the full build with the
GDN/`gdn_attention` kernel (the gated-delta-net decode op); `:int8`/`:int8g` lack it and crash on the
first token for this checkpoint.

## Recipe (baked into serve.sh)
- GRAPH=1 PIECEWISE capture (the ~4x decode lever; eager is ~7.8 t/s). DTYPE=auto, UTIL=0.92.
- CAPSIZES=1,2,4,8,16,32,64 so batches up to 64 stay captured (else >8 falls back to eager).
- NOMM=1 -- 27B is a `qwen3_5` VLM; text-only serve skips the vision-encoder profiling crash on XPU.
- TOOLCALL=1 / qwen3_coder / REASONPARSER=qwen3 -- Qwen3.6 emits XML tool calls (NOT hermes JSON).

## Verified perf (see ../../FINDINGS.md / docs/SERVING.md)
- Decode ~30.8 t/s captured single-stream. Aggregate ~28 t/s @C1 -> ~217 @C32 -> ~235 @C64.
- Long ctx (fp16 KV, UTIL=0.92): caps ~133k; 256k does NOT fit. Add `KVDTYPE=fp8_e5m2` to ~2x it.
- Per-stream decode drops below single-stream past C8 (GDN batches poorly) -> stay C2-C4 for low latency.
- Other 27B dirs are NOT this recipe: `Qwen3.6-27B-W4A16` (compressed-tensors) will NOT serve.

verified: smoke=PENDING (this pass) -- update via `bin/serve-sweep --smoke`
