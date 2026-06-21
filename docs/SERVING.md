# SERVING.md -- canonical copy-paste recipes to serve our models on the B70

The single source of truth for "how do I serve model X on the B70 right now."
If you reconstruct a serve command from JOURNAL/scripts again, you did it wrong --
fix THIS doc instead. Keep recipes verified-and-current; date each change.

## Where everything runs
- GPU host: Unraid @ `192.168.10.5` (`ssh root@192.168.10.5`). NOT mounted on the dev box.
- Repo is synced to the host at `/mnt/vm_8tb/b70/` (FLAT layout: `30_serve_w4a8_graph.sh`,
  `31_decode_probe.sh`, `35_sweep_bench.sh`, `gpu-run` all live at that root -- NOT under `scripts/`).
- Models: `/mnt/vm_8tb/b70/models/<dir>` on the host, bind-mounted into the container at `/models/<dir>`.
  -> `MODEL=` and `TOKPATH=` always use the **container** path `/models/...`, never the host path.
- Serve port is always `18080`. Served container name defaults to `vllm_w4a8`.

## [!] Every GPU touch goes through the flock lease
`/mnt/vm_8tb/b70/gpu-run <cmd>` -- runs `<cmd>` only while holding `/mnt/vm_8tb/b70/gpu.lock`.
`/mnt/vm_8tb/b70/gpu-run --status` -- who holds it (or `free`). A `docker run -d` serve returns
immediately, so wrapping JUST the serve in gpu-run releases the lease right after startup. To hold the
lease for a serve+bench session, wrap the WHOLE session (serve -> wait -> bench -> `docker stop`) in one
`gpu-run` call. Release by `docker stop <name>` when done.

## The serve script: `30_serve_w4a8_graph.sh` (env-driven, all models)
One script serves every quant; knobs are env vars (despite the "w4a8" name it serves w4a16/MoE too):

| env | default | meaning |
|---|---|---|
| `IMG` | `vllm-xpu-env:int8g` | docker image (see per-model below) |
| `MODEL` | `/models/Qwen3-14B-W4A8-gptq` | **container** path to model dir |
| `SERVED` | `qwen3-14b-w4a8-gptq` | served-model-name (the `/v1/models` id) |
| `GRAPH` | `0` | `1` = PIECEWISE XPU graph capture = the big decode lever (~2-4x) |
| `DTYPE` | `float16` | `auto` for the 27B/MoE |
| `UTIL` | `0.90` | gpu-memory-utilization |
| `MAXLEN` | `4096` | --max-model-len |
| `MAXSEQS` | `4` | --max-num-seqs (RAISE for concurrency, e.g. 64) |
| `CAPSIZES` | (none) | cudagraph capture sizes, e.g. `"1,2,4,8,16,32,64"`. Default capture tops at 8 -> batches >8 fall back to EAGER. Set this for a captured concurrency sweep. |
| `CGMODE` | `PIECEWISE` | keep PIECEWISE (FULL is blocked by SYCL-Graph scratch limits) |
| `NOMM` | (none) | `1` = text-only serve of a VLM (skip vision-encoder profiling that crashes on XPU). REQUIRED for the Qwen3.6-27B (it is a `qwen3_5` VLM). |
| `KVDTYPE` | (none) | `fp8_e5m2` = fp8-storage KV (halves KV BW, 2x ctx/batch). Omit = fp16 KV. |
| `NAME` | `vllm_w4a8` | container name |

The script: `docker rm -f` old containers -> `docker run -d` -> waits up to ~14 min for `/health`
-> prints the served `/v1/models` id + capture/kernel log lines -> prints `HEALTHY` or `NOT HEALTHY`.

---

## RECIPE: Qwen3.6-27B (int4 AutoRound = w4a16), captured  [PRIMARY single-card quality pick]
Decode ~30.8 t/s captured (eager is only ~7.8). Model 17.6 GiB + ~7.5 GiB KV. Image MUST be `:v0230`
(the full build with the GDN/`gdn_attention` kernel; `:int8g` lacks it and crashes on the first token).

```bash
ssh root@192.168.10.5
cd /mnt/vm_8tb/b70
./gpu-run env \
  IMG=vllm-xpu-env:v0230 \
  MODEL=/models/Lorbus_Qwen3.6-27B-int4-AutoRound \
  SERVED=qwen36-27b-int4 \
  GRAPH=1 DTYPE=auto UTIL=0.92 \
  MAXLEN=8192 MAXSEQS=64 \
  CAPSIZES=1,2,4,8,16,32,64 \
  NOMM=1 NAME=vllm_w4a8 \
  bash ./30_serve_w4a8_graph.sh
# (gpu-run releases after startup; to bench, wrap serve+bench+stop in ONE gpu-run -- see campaign script.)
```
- Long context (MEASURED 2026-06-21, UTIL=0.92, fp16 KV): **256k does NOT fit** -- a 262144-tok seq needs
  16.2 GiB KV vs ~8.3 GiB free (model 16.69 GiB) -> vLLM caps max len at ~133k. So **MAXLEN<=131072 at fp16
  KV**, or add `KVDTYPE=fp8_e5m2` to roughly double it. Context window is throughput-NEUTRAL: an 8k vs 128k
  serve sweep at 512/128 was near-identical (KV pool = f(util), not max-len).
- Concurrency (captured, fp16 KV, 512/128): aggregate ~28 t/s @C1 -> ~217 @C32 -> ~235 @C64; per-stream
  decode drops below single-stream past C8 (GDN batches poorly). Low-latency serving: stay C2-C4. See FINDINGS.
- Other 27B dirs on the host are NOT this recipe: `Qwen3.6-27B-W4A16` (our compressed-tensors w4a16)
  won't serve (XPUwNa16 needs dims /32; the gated-attn 4304 dim fails). Use the Lorbus AutoRound int4.

## RECIPE: Qwen3.6-27B W4A8 (prepacked, quality GDN+lm_head bf16)  [SECONDARY -- slower than w4a16]
Only if you specifically want the int8-activation/int8-XMX path on the 27B. Decode **20.9 t/s** captured
(< w4a16's 30.8), VRAM-tight (24.35 GiB, OOM-prone), and needs MORE setup, so w4a16 above is the default.
Requires: the offline-prepacked model (`Qwen3.6-27B-W4A8-q-prepacked`, int4-packed on disk), `PREPACK=1`
(mounts the patched loader+kernel from `patches/`), and a **rebuilt GDN `_xpu_C.abi3.so`** (the `:int8g`
build ships `GDN_KERNELS_ENABLED=OFF` -> `_xpu_C has no attribute gdn_attention` at decode) mounted via
`KERNEL_SO` (which also auto-mounts the sibling `libgdn_attn_kernels_xe_2.so`). Served with **fp8 KV**.

```bash
./gpu-run env \
  IMG=vllm-xpu-env:int8g \
  MODEL=/models/Qwen3.6-27B-W4A8-q-prepacked \
  SERVED=qwen36-27b-w4a8 \
  GRAPH=1 PREPACK=1 NOMM=1 KVDTYPE=fp8_e5m2 UTIL=0.90 \
  KERNEL_SO=/mnt/vm_8tb/b70/vllm-xpu-kernels/vllm_xpu_kernels/_xpu_C.abi3.so \
  NAME=vllm_w4a8 \
  bash ./30_serve_w4a8_graph.sh
```

## RECIPE: Qwen3.6-35B-A3B MoE (int4 AutoRound), captured  [FASTEST single-card decode]
Decode ~56.8 t/s captured (fp16 KV) / ~65 t/s with fp8 KV. Needs the MoE-routing image `:v0230moe`
(`:v0230` + the INC-XPU RoutedExperts -> MoeWNA16 patch; see `contrib/vllm_moe_xpu/`).

```bash
./gpu-run env \
  IMG=vllm-xpu-env:v0230moe \
  MODEL=/models/Intel_Qwen3.6-35B-A3B-int4-AutoRound \
  SERVED=qwen36-35b-a3b-int4 \
  GRAPH=1 DTYPE=auto UTIL=0.90 \
  MAXLEN=8192 MAXSEQS=64 \
  CAPSIZES=1,2,4,8,16,32,64 \
  KVDTYPE=fp8_e5m2 \
  NAME=vllm_w4a8 \
  bash ./30_serve_w4a8_graph.sh
```
- Aggregate throughput plateaus ~206 t/s at N>=8 (the routed-expert union approaches all 256 experts).

## RECIPE: Qwen3-14B (FP8 / w8a8-int8 / w4a16-gptq)  [14B-class workhorse]
- FP8 (fastest decode ~32 t/s, lowest TTFT): `IMG=vllm-xpu-env:v0230 MODEL=/models/Qwen_Qwen3.6-...` no
  -- use the 14B FP8 dir; serve plain (no GRAPH needed for fp8, but GRAPH=1 helps TTFT).
- w4a8 / w8a8 int8 (our INT8 kernel, lights the systolic path): `IMG=vllm-xpu-env:int8g`,
  `MODEL=/models/Qwen3-14B-W4A8-gptq` or `...-W8A8-gptq`, `GRAPH=1` for captured decode.
  See `docs/kernel/` + `evals/results/SUMMARY.md` for the full quant ladder.

---

## Concurrency sweep against a running server: `35_sweep_bench.sh`
Needs a server already up (it `docker exec`s into it). Env: `NAME` (container), `MODEL` (the SERVED id),
`LABEL` (csv name), `TOKPATH` (**container** `/models/...` path), `CONC` (levels).

```bash
NAME=vllm_w4a8 MODEL=qwen36-27b-int4 \
  LABEL=qwen36-27b-int4-piecewise \
  TOKPATH=/models/Lorbus_Qwen3.6-27B-int4-AutoRound \
  CONC="1 2 4 8 16 32 64" \
  bash /mnt/vm_8tb/b70/35_sweep_bench.sh
# -> /mnt/vm_8tb/b70/results/sweep_<LABEL>_<stamp>.csv
#    columns: concurrency,req_s,out_tok_s,mean_ttft_ms,mean_tpot_ms,per_stream_decode_tok_s
```
Bench is `vllm bench serve` random 512-in/128-out `--ignore-eos`; `out_tok_s` = aggregate, per-stream
decode = 1000/TPOT. For a CAPTURED sweep, serve with `CAPSIZES` covering the `CONC` levels.

## [!] Always verify the served model (CLAUDE.md rule)
`curl -s http://192.168.10.5:18080/v1/models | python3 -m json.tool` -- the `id` must match `SERVED`
and encode the quant. Cross-check against `evals/configs/models.yaml`.

---
Last verified: 2026-06-21 (27B captured concurrency campaign).
