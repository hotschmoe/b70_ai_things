# Qwen3-14B W4A8 (int4 weights / int8 activations, GPTQ, prepacked)

Serves `Qwen3-14B-W4A8-gptq-prepacked` on ONE Intel Arc Pro B70. int4 weights (4x smaller) with int8
activations on our INT8 systolic path. ~9.3 GiB packed on disk.

## Run (on the GPU host)
```bash
cd /mnt/vm_8tb/b70/rdy_to_serve/qwen3-14b-w4a8
/mnt/vm_8tb/b70/gpu-run bash serve.sh          # start (PIECEWISE capture), wait healthy, gen-probe
bash serve.sh stop
```
Endpoint: `http://<host>:8000/v1`. Served id: `qwen3-14b-w4a8-gptq`.

## Image + the prepack pieces (all local to this dir)
- Image `vllm-xpu-env:int8g` (build: `../../images/int8g/`).
- `patches/xpu.py` + `patches/compressed_tensors_w4a8_int.py` -- the patched mixed-precision loader + the
  W4A8 scheme, bind-mounted over vLLM; with `VLLM_W4A8_PREPACKED=1` they load the int4-packed weights
  directly (no large unpacked-int8 GPU transient). Dense 14B (Qwen3) -> no GDN, no vision tower.

verified: _common smoke=GREEN (eager, :int8g, 2026-06-23, 83s): HEALTHY, id ok, prepack mount + env ok,
gen "Paris... Berlin...". Re-verify: `bin/serve-sweep --smoke`.
