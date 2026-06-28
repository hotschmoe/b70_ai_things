# qwen36-27b-int4-graph -- the FASTEST single-stream driver (int4 + XPUGraph decode capture)

Qwen3.6-27B **int4** (Lorbus AutoRound W4A16, woqgemm) with **`torch.xpu.XPUGraph` decode capture** -- the
first sglang-XPU cuda-graph. Single card, **SAMPLING-capable** (honors temperature/top_p, unlike the greedy
MTP driver), **VISION retained**. Single-stream **c1 ~23.5 t/s = 2.5x the 9.4 eager ceiling and +53% over the
int4+MTP driver (15.3)**, coherent, soak-stable, and GDN-correct under mixed load (0 garbage).

## How it works (the frontier win)
sglang stays eager on XPU out of the box ("cuda graph: False"). `xpu_cudagraph.py` (opt-in `B70_XPU_CUDAGRAPH=1`,
baked into `sglang-xpu:mtp`) fixes the two blockers: (1) adds `"xpu"` to `model_runner.init_cuda_graphs`'s device
allow-list; (2) implements the missing decode cuda-graph hooks on `XPUAttentionBackend` (static token-level
`page_table` + `cache_seqlens`, refilled in-place each replay). The GDN/mamba side was already capture-safe.
`torch.xpu.XPUGraph` (SYCL-Graph over Level-Zero, torch 2.12) is proven non-degrading on B70 (4000 replays stable).
**`--attention-backend triton` is required** -- the XPU FlashAttn kernel hits the SYCL-Graph
`work_group_scratch_memory` wall at capture; pure-triton attention clears it (== vLLM's `TRITON_ATTN`).

## Run (on the GPU host)
```bash
cd /mnt/vm_8tb/github/b70_ai_things/rdy_to_serve/qwen36-27b-int4-graph
/mnt/vm_8tb/b70/gpu-run --card 0 bash serve.sh start   # serve, capture at startup, coherence-gated probe
bash serve.sh bench                                     # warm c1 (pp/ttft/tg @ ctx2048) + soak
bash serve.sh stop
/mnt/vm_8tb/b70/gpu-run --card 0 bash serve.sh run      # start + bench + stop in one lease
```
Endpoint: `http://<host>:30000/v1`. Served id: `qwen36-27b-int4-graph`. Image: `sglang-xpu:mtp` (no mounts).

## [!] Pin card 0 -- the cards are asymmetric
On this box the `xe` driver drives the console/display off one Arc card; `ZE_AFFINITY_MASK=1` (card 1) is the
display-attached card and runs DOWNCLOCKED. Measured: **card 0 = 23.5 t/s, card 1 = 15.3 t/s** (same config,
same image). serve.sh defaults `DEVICE=0` (the fast compute card). For DP=2 this is asymmetric (card 0 ~23.5,
card 1 ~15.3) -- still a net win, but not symmetric.

## [!] Single-stream driver -- use DP=2 for concurrency
`--max-running-requests 1` + a single captured `bs=1` graph. Multi-bucket capture (`bs>1`) currently HALVES
single-stream (a single decode pads up to the `bs=N` graph; `--disable-cuda-graph-padding` breaks capture
entirely). So for >1 user, run **DP=2** (`../../sglang/serve_dp2_graph.sh` -> 2 users @ ~23.5 each, beating
MTP-DP2's 2x15), NOT a higher `max-running-requests` here. Per-card multi-stream-at-speed is open work.

## Driver matrix (all CORRECT + VISION on sglang-XPU)
| driver | c1 t/s | sampling | cards | use |
|---|---|---|---|---|
| **int4 + XPUGraph (this)** | **~23.5** | **yes** | 1 | FASTEST single-stream; interactive single user |
| int4 + XPUGraph DP=2 (`../../sglang/serve_dp2_graph.sh`) | ~23.5 x2 slots | yes | 2 | 2 full-speed users (wedge-proof) |
| int4 + NEXTN MTP steps=7 (`../qwen36-27b-int4-mtp`) | ~15.3 | greedy only | 1 | latency (superseded by graph for single-user) |
| int4 woq DP=2 (`../../sglang/serve_dp2.sh`) | ~9.4/replica | yes | 2 | high-concurrency (>2 users), wedge-proof |

verified: 2026-06-28 -- baked sglang-xpu:mtp image, ZERO mounts. CARD 0 (scripts reproduce, graph_reproduce_card0.log):
WARM c1 23.51 t/s (TTFT 1208ms @ctx2048), soak 23.03 (stable 1.11x), server gen-throughput ~22, coherent +
sampling-varies + 95 "cuda graph: True". CARD 1 (scripts/142 acceptance, verify_graph_image.log): c1 15.28
(display-attached, downclocked) but soak 23.18 -- same capture/coherence/sampling. Capture takes ~51s at startup.
