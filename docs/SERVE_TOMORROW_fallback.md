# SERVE-TOMORROW fallback runbook (2026-07-07)

**CURRENT LIVE DD (2026-07-07): NVFP4 B4 (Option 3 below), served as `hotschmoe-dd` on :18080.**
Stable by construction: MTP off -> only the target decode graph replays 1/step and its all-reduces
go through PUSH_AR posted-write (no oneCCL in the captured decode) -> no per-replay command re-append
-> no linear_stream overflow. Warm decode ~20-25 t/s, coherent, prefix cache + vision + tool/reason parsers.

ROOT CAUSE (2026-07-07, docs/20260707_dd_mtp_piecewise_neo_abort.md): the abort is a oneCCL collective
recorded into the captured SYCL graph that re-appends commands per replay (our torch 2.12+xpu oneCCL
predates SYCL-graph Record&Replay; torch-xpu-ops#2992). The real full-speed fix = upgrade oneCCL
>=2021.17.2/2022.0 + backport #2992 (RESEARCH_TODO 11h). Until that lands, serve one of the STABLE
configs below (all avoid the crash by keeping oneCCL collectives out of a high-frequency replayed graph).

## Ranked stable options (all TP=2, both cards, coherent + no NEO abort)

| # | config | serve | decode | MTP | notes |
|---|--------|-------|--------|-----|-------|
| 1 | **W8A8 enforce-eager + MTP** | GRAPH=0 (below) | ~15-16 t/s | yes | PROVEN: 40min soak + full agentic eval, 0 crashes. Rock-solid, slowest. |
| 2 | **NVFP4 drafter-eager (MTP kept)** | B70_XPU_DRAFTER_EAGER=1 (below) | ~22-26 t/s | yes | VALIDATED 2026-07-07: 44k-token soak clean. Keeps MTP + target capture. |
| 3 | **NVFP4 B4 (graph + MTP-off)** | MTP off + PUSH_AR (below) | 25-31 t/s | no | The pre-existing stable NVFP4 DD; fastest stable, but drops MTP. |

DO NOT serve captured+MTP (the old "DD": NVFP4 fused GRAPH=1 MTP5, or W8A8 GRAPH=1 MTP3
WITHOUT drafter-eager) unattended -- it crashes (linear_stream.h:84) after ~8-12k tokens
(NVFP4 TP=2) / ~3h (W8A8 TP=2).

## Exact serve commands (run from the repo root, /mnt/vm_8tb/github/b70_ai_things)

### Option 1 -- W8A8 enforce-eager + MTP (bulletproof)
```
NAME=b70_daily_0 PORT=18080 TP=2 MAXLEN=131072 GRAPH=0 SERVED=hotschmoe-dd \
  API_KEY=<key> ./bin/gpu-run bash rdy_to_serve/vllm/qwen36-27b-w8a8/serve.sh start
# GRAPH=0 = enforce-eager => no graph capture => no leak. Keeps MTP3 (~16 t/s).
# stop: bash rdy_to_serve/vllm/qwen36-27b-w8a8/serve.sh stop
```

### Option 2 -- NVFP4 drafter-eager (keeps MTP, faster than opt 1)
```
NAME=b70_daily_0 PORT=8078 TP=2 MODE=fused GRAPH=1 MTPTOK=5 KV_FP8=0 \
  CAPSIZES=1,2,4,8 MAXLEN=131072 UTIL=0.85 PUSH_AR=1 \
  B70_EXTRA_ENV="B70_XPU_DRAFTER_EAGER=1" \
  ./bin/gpu-run bash vllm/nvfp4/serve_nvfp4_27b.sh start
# drafter runs eager (no leak), target decode captured. KV_FP8=0 avoids the fp8-KV repetition fault.
# stop: bash vllm/nvfp4/serve_nvfp4_27b.sh stop   (or docker rm -f b70_daily_0)
```

### Option 3 -- NVFP4 B4 (graph + MTP-off + decode push), the pre-existing stable DD
```
NAME=b70_daily_0 PORT=8078 TP=2 MODE=fused GRAPH=1 KV_FP8=0 PUSH_AR=1 PUSH_AR_GRAPH=1 \
  CAPSIZES=1,2,4,8 MAXLEN=131072 UTIL=0.85 \
  ./bin/gpu-run bash vllm/nvfp4/serve_nvfp4_27b.sh start
# MTPTOK unset => MTP OFF => no drafter => no leak. PUSH_AR_GRAPH=1 recovers captured-decode speed.
```

## If a crash happens anyway
- It is a clean SIGABRT (worker dies, container exits, NO GPU wedge / NO reboot needed).
- Recover: `docker rm -f b70_daily_0` then re-run the serve command. Health in ~2-3 min.
- Verify it is THIS bug: `docker logs b70_daily_0 2>&1 | grep -c linear_stream` (>0 = the abort).

## Verify which config is live
```
curl -s http://192.168.10.5:${PORT}/v1/models | python3 -m json.tool   # served id
docker logs b70_daily_0 2>&1 | grep -iE 'drafter-eager|enforce.eager|speculative'
```
