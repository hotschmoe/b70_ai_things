# Qwen3.6-35B-A3B MoE int4-AutoRound -- FASTEST single-card decode

Serves `Intel_Qwen3.6-35B-A3B-int4-AutoRound` (host `/mnt/vm_8tb/b70/models/...`) on ONE Intel Arc Pro B70.
35B total / ~3B active (A3B MoE, 256 routed experts) -> more knowledge than the 27B at higher decode speed.

## Run (on the GPU host)
```bash
ssh root@192.168.10.5 && cd /mnt/vm_8tb/b70/rdy_to_serve/qwen36-35b-a3b-int4
/mnt/vm_8tb/b70/gpu-run bash serve.sh          # start (PIECEWISE capture), wait healthy, gen-probe
bash serve.sh stop                              # stop + release the GPU
GRAPH=1 /mnt/vm_8tb/b70/gpu-run bash serve.sh run   # serve + concurrency sweep + stop, one lease
```
Endpoint: `http://<host>:8000/v1`. Served id: `qwen36-35b-a3b-int4`.

## [!] Image: `vllm-xpu-env:v0230moe` (NOT :v0230, NOT llm-scaler 0.14.x)
`:v0230moe` = `:v0230` + the INC-XPU `RoutedExperts -> MoeWNA16` patch BAKED on this leaf tag
(see `../../contrib/vllm_moe_xpu/`). This is the bake-on-leaf side of the patch rule: a compiled/baked
MoE-routing change lives on its OWN image tag so it can never affect a dense model. No runtime mount.
Plain `:v0230` leaves MoE routing unbaked; `intel/llm-scaler-vllm:0.14.x` has no `_moe_C` (int8 MoE dies).

## Recipe (baked into serve.sh)
- GRAPH=1 PIECEWISE capture. DTYPE=auto, UTIL=0.90, MAXLEN=8192, MAXSEQS=64.
- CAPSIZES=1,2,4,8,16,32,64. KVDTYPE=fp8_e5m2 (fp8-storage KV -> ~65 t/s + 2x ctx/batch; B70 has no FP8 ALU).
- TOOLCALL=1 / qwen3_coder / REASONPARSER=qwen3.

## Verified perf (see ../../FINDINGS.md / docs/SERVING.md)
- Decode ~56.8 t/s captured (fp16 KV) / ~65 t/s (fp8 KV) single-stream -- fastest single-card we have.
- Aggregate throughput plateaus ~206 t/s at N>=8 (the routed-expert union approaches all 256 experts).

verified: smoke=PENDING (this pass) -- update via `bin/serve-sweep --smoke`
