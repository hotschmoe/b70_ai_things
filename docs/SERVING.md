# SERVING.md -- canonical copy-paste recipes to serve our models on the B70

The single source of truth for "how do I serve model X on the B70 right now."
If you reconstruct a serve command from JOURNAL/scripts again, you did it wrong --
fix THIS doc instead. Keep recipes verified-and-current; date each change.

**Daily driver:** `./daily_driver_serve.sh` (repo root) brings up the current daily-driver model and
keeps the API at `http://192.168.10.5:18080/v1` live for our apps. `start|stop|status|restart|logs`.
Edit its CONFIG block to change which model/recipe we serve. It holds the GPU lease (BOTH cards)
while up, so `stop` it before running GPU experiments/quant. Current daily driver: **Qwen3.6-27B W4A16,
served as 2x DATA-PARALLEL replicas** (one captured replica per B70 -> :18091/:18092) behind an nginx
round-robin proxy on :18080. ~2.1x aggregate throughput, full single-stream latency/replica, zero inter-GPU
comms (measured; see the data-parallel recipe below + `scripts/64_dataparallel_2rep.sh`).

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

## RECIPE: Qwen3.6-35B-A3B Quark **W8A8 INT8** MoE, 2x B70 TP=2 on `:v0230`  [WORKING 2026-06-22]
int8 W8A8 MoE 35B (ckpt `nameistoken/Qwen3.6-35B-A3B-Quark-W8A8-INT8`) **serves + generates coherently**
on **2x B70 TP=2**. Use `scripts/76_quark35b_v0230.sh`. Key: serve on **`vllm-xpu-env:v0230` (vLLM
0.23.0)** -- NOT the llm-scaler `0.14.x` image. 0.23 already ships `QuarkW8A8Int8MoEMethod` + the
dynamic-per-token int8 linear dispatch AND routes the 256 int8 experts through the **Triton
`fused_moe_kernel`** on XPU (the int4 MoE path that works here -- `contrib/vllm_moe_xpu`). The llm-scaler
0.14.1 image had **no XPU MoE op suite** (`vllm._moe_C` unbuilt -> `topk_softmax` missing) -- a dead end
(see kernel/20 sec 8). The ONLY patch needed on v0230 is **one bind-mounted `quark.py`**
(`contrib/llm_scaler_quark_int8_moe/v0230/quark.py`) that reroutes the int8 **linear** layers
(linear_attn.*, mlp.shared_expert.*) to a weight-only int8->bf16 **dequant** GEMM, because XPU has no int8
scaled-mm kernel (`_POSSIBLE_KERNELS` KeyErrors). Experts stay TRUE int8.

```bash
ssh root@192.168.10.5 && cd /mnt/vm_8tb/b70
# patch already at patches/quark_v0230.py; serve+probe+sweep, then auto-stops:
./gpu-run bash 76_quark35b_v0230.sh           # IMG=vllm-xpu-env:v0230 TP=2 MAXLEN=8192 UTIL=0.92
```
Verified: `Model loading took 17.54 GiB`/card, KV 10.2 GiB, concurrency 89x@8192, `backend=xccl world_size=2`,
Triton `fused_moe_kernel` JIT (E=256,N=256, int8), gen: "The capital of France is" -> " Paris, a city
renowned for its rich history, culture, and iconic landmarks." Served **enforce-eager** (graph capture is
the open perf lever; PIECEWISE capture gave +617% on the int4 MoE, so the same is the next step here).
Reusable B70 gotchas (also apply to any llm-scaler TP=2): (a) #41663 env `CCL_TOPO_P2P_ACCESS=0`,
`CCL_ZE_IPC_EXCHANGE=pidfd`, `CCL_ENABLE_SYCL_KERNELS=0` (script 76 sets these for TP>1); (b) do NOT pin
`ONEAPI_DEVICE_SELECTOR`+`ZE_AFFINITY_MASK` (aborts the model-inspect subprocess on a 2-card box);
(c) the Unraid host has no `python3` -- serve scripts must not parse JSON with it (use `--served-model-name`).

## RECIPE: Qwen3-14B (FP8 / w8a8-int8 / w4a16-gptq)  [14B-class workhorse]
- FP8 (fastest decode ~32 t/s, lowest TTFT): `IMG=vllm-xpu-env:v0230 MODEL=/models/Qwen_Qwen3.6-...` no
  -- use the 14B FP8 dir; serve plain (no GRAPH needed for fp8, but GRAPH=1 helps TTFT).
- w4a8 / w8a8 int8 (our INT8 kernel, lights the systolic path): `IMG=vllm-xpu-env:int8g`,
  `MODEL=/models/Qwen3-14B-W4A8-gptq` or `...-W8A8-gptq`, `GRAPH=1` for captured decode.
  See `docs/kernel/` + `evals/results/SUMMARY.md` for the full quant ladder.

---

## RECIPE: dual-B70 DATA-PARALLEL (PREFERRED for any model that fits one card)  [2026-06-21]
**This is the dual-GPU win.** Run one independent replica PER CARD (zero inter-GPU comms) behind a round-robin
proxy. Measured 27B int4: **~2.1x aggregate (C64 ~525 t/s)** AND full single-stream ~30.8 t/s/replica, no
contention -- beats TP=2 (0.53x) and PP=2 (0.78x) on BOTH axes. The ONLY thing DP can't do is serve a model
bigger than one card (then use PP=2 below).
```bash
# Persistent daily driver (2 replicas + nginx proxy on :18080, Open WebUI tied in):
./daily_driver_serve.sh start      # from the dev box; holds the lease for both cards
# Ad-hoc bench (solo baseline + concurrent aggregate):
ssh root@192.168.10.5 'cd /mnt/vm_8tb/b70 && ./gpu-run bash 64_dataparallel_2rep.sh'
# Manual (per replica): the 30_serve script now takes DEVICE (card 0|1) + PORT:
#   DEVICE=0 PORT=18091 NAME=vllm_dp0 ... bash 30_serve_w4a8_graph.sh
#   DEVICE=1 PORT=18092 NAME=vllm_dp1 ... bash 30_serve_w4a8_graph.sh
# then an nginx least_conn proxy (dp_nginx.conf) on :18080 across 127.0.0.1:18091/18092.
```
**[2026-06-22] OPT-IN MTP on the daily driver (~1.79x single-stream):** the daily driver serves the exact 27B int4
W4A16 config the MTP campaign measured. Start it with **`DD_MTP=1 ./daily_driver_serve.sh start`** to enable MTP
spec=4 per replica -> single-stream decode **~30.8 -> ~55 t/s (1.79x)**, accept_len ~3.25, Half-KV-free (MTP_TODO /
JOURNAL 2026-06-22). **TRADEOFF (decide by workload):** MTP is a LOW-CONCURRENCY lever -- the spec-verify runs the
model x(1+spec), so it goes compute-bound as concurrency rises and can REDUCE aggregate throughput past ~C8. Enable
for INTERACTIVE use (1-few users / Open WebUI / a single coding agent); leave OFF (default) for high-concurrency
batch/agent fan-out. Mechanism: `DD_MTP=1` appends `MTPTOK=4 COMPILESZ=` to the serve env; the host `30_serve` builds
the spec-config JSON from the integer (passing the JSON directly through the nested ssh/bash -c strips its quotes).
FULL graph capture would lift this further but is blocked on v0230 (gdn_attention spec op can't run in a captured graph
-- MTP_TODO). MTP is a DENSE-model lever: do NOT enable it for the 35B-A3B MoE daily driver (+3% flat, MoE headline is capture).

## RECIPE: dual-B70 tensor-parallel (TP=2) / pipeline (PP=2)  [for models too big for one card]
Two B70s can shard a model across both cards. Multi-GPU model-parallel is a CAPACITY play, not a single-stream
speed play (no GPU P2P; collectives round-trip GPU->host RAM->GPU over PCIe). NOTE: the PCIe link is healthy
**Gen3 x16** (H2D 12.82 GB/s measured) -- the "Gen1 x1" lspci reading is an Intel Arc artifact (KB 000094587),
NOT a real bottleneck. The multi-GPU cap is no-P2P host-staged collectives. Use model-parallel ONLY to serve a
model that does NOT fit one 32 GB card (else use DATA-PARALLEL above). For fit-one-card models, DP wins outright.

**[!] Prefer PP=2 over TP=2 when you must go model-parallel.** Measured eager 27B int4 single-stream:
PP=2 = 6.11 t/s (0.78x single-card) vs TP=2 = 4.18 (0.53x) -- PP is +46% because it does ONE hidden-state handoff
per token instead of TP's ~128 host-staged all-reduces, so it barely touches the interconnect. PP=2 also gives a much larger
KV pool (~19 GiB/stage vs TP's tight per-layer split). Set PP via `43_serve_multi.sh TP=1 PP=2` (see
`62_pp2_27b.sh`). Use TP only if a single layer can't fit one card (not the case for our 27B/35B). Captured (graph)
TP=2 is BLOCKED (oneCCL can't sycl_graph-record an all-reduce on the stable config) -> multi-GPU is eager-only.

Two serve paths, both carry the **Battlemage multi-GPU stability env** (vLLM #41663): `CCL_ENABLE_SYCL_KERNELS=0`
(the load-bearing GP-fault fix), `CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0`, `SYCL_UR_USE_LEVEL_ZERO_V2=0`,
`CCL_ATL_TRANSPORT=ofi`, `CCL_ZE_IPC_EXCHANGE=pidfd`, `CCL_TOPO_P2P_ACCESS=0`, `VLLM_WORKER_MULTIPROC_METHOD=spawn`,
`--distributed-executor-backend mp`. NEVER set `CCL_ALLREDUCE=ring` (drops to ~0.5 tok/s).

- **Captured (preferred):** `30_serve_w4a8_graph.sh` now has a `TP=` knob (default 1, backward-compatible). TP>1
  exposes both cards (no ZE_AFFINITY pin) + the env above. Example (27B int4 across both cards):
  ```bash
  ./gpu-run env IMG=vllm-xpu-env:v0230 MODEL=/models/Lorbus_Qwen3.6-27B-int4-AutoRound \
    SERVED=qwen36-27b-int4-tp2 GRAPH=1 DTYPE=auto UTIL=0.92 MAXLEN=8192 MAXSEQS=64 \
    CAPSIZES=1,2,4,8,16,32,64 NOMM=1 TP=2 NAME=vllm_w4a8 bash ./30_serve_w4a8_graph.sh
  ```
- **Eager (simplest, for big models / bring-up):** `43_serve_multi.sh` (env: `TP` default 2, `PP`, `MODEL`,
  `SERVED`, `QUANT` none|fp8|<ckpt>, `KVDTYPE`, `MAXLEN`, `MAXSEQS`, `UTIL`, `EXTRA`). Serves the full BF16 27B
  (too big for one card) at TP=2 with `EXTRA='--limit-mm-per-prompt {"image":0,"video":0}'`.
- Campaign harness: `58_tp2_campaign.sh` (serve TP=2 + concurrency sweep + PCIe link-state probes).

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

## Tool calling + reasoning (coding agents: pi, etc.)
Agents send `tool_choice:"auto"`; vLLM **400s** unless a tool parser is configured, and it emits tool calls
as model-specific TEXT that the parser lifts into the API `tool_calls` field. Enable via the serve knob
`TOOLCALL=1` (-> `--enable-auto-tool-choice --tool-call-parser ${TOOLPARSER:-qwen3_coder}`), plus
`REASONPARSER=qwen3` (-> `--reasoning-parser qwen3`, splits `<think>` into `reasoning_content`).
- **[!] Qwen3.6 uses Qwen3-Coder XML** (`<function=name><parameter=..>`), NOT Hermes JSON. `hermes` returns
  HTTP 200 but **EMPTY `tool_calls`** -- its `json.loads()` chokes on the XML (`JSONDecodeError`) -- so the
  agent silently gets nothing to run. **Use `qwen3_coder`** (this build also has `qwen3_xml`). Check the build's
  parsers: `docker exec <ctr> bash -lc 'grep -niE qwen /opt/venv/lib/python3.12/site-packages/vllm/tool_parsers/__init__.py'`.
- The daily driver (`daily_driver_serve.sh`) sets `TOOLCALL=1 TOOLPARSER=qwen3_coder REASONPARSER=qwen3`.

## [!] Always verify the served model (CLAUDE.md rule)
`curl -s http://192.168.10.5:18080/v1/models | python3 -m json.tool` -- the `id` must match `SERVED`
and encode the quant. Cross-check against `evals/configs/models.yaml`.

---
Last verified: 2026-06-21 (dual-B70: DATA-PARALLEL preferred -- ~2.1x, zero comms; PP=2 for oversized models;
the "Gen1 x1" was an Intel Arc lspci artifact -- real link is healthy Gen3 x16, H2D 12.82 GB/s).
