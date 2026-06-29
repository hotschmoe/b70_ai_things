# qwen36-27b-int4-mtp -- the LATENCY daily driver (int4 + NEXTN MTP, single card)

Qwen3.6-27B **int4** (Lorbus AutoRound W4A16, served via the `woqgemm` XPU int4 GEMM) + a grafted **BF16
NEXTN MTP head**, served **single-card with chain-MTP (num-steps=7), GREEDY, VISION retained** on one
Intel Arc Pro B70. This is the **first config to STABLY beat the ~9.4 t/s sglang-XPU eager ceiling** on the
dual-B70 box -- single-stream **c1 ~15.3 t/s = 1.62x** baseline, mean accept_len ~4.1-4.4 -- and it stays
**correct under sustained mixed load** (the agentic prefill+decode pattern that makes vLLM emit `!!!!`).

## Run (on the GPU host)
```bash
cd /mnt/vm_8tb/github/b70_ai_things/rdy_to_serve/qwen36-27b-int4-mtp
/mnt/vm_8tb/b70/gpu-run --card 0 bash serve.sh start   # serve, wait healthy, coherence-gated probe, stay up
bash serve.sh bench                                     # warm c1/c4 (pp/ttft/tg @ ctx2048) + soak
bash serve.sh accept                                    # mean accept length from the decode log
bash serve.sh stop                                      # stop + release the card
/mnt/vm_8tb/b70/gpu-run --card 0 bash serve.sh run      # start + bench + stop in one lease
```
Endpoint: `http://<host>:30000/v1`. Served id: `qwen36-27b-int4-mtp-nextn`.

## Image: `sglang-xpu:mtp` (self-contained -- NO runtime patch mounts)
Build context: `../../sglang/images/sglang-xpu-mtp/` (`FROM sglang-xpu:woq` + two baked files):
- `mtp_tree_xpu.py` -- the pure-torch XPU fallback for sglang's NEXTN/EAGLE tree kernels (chain, topk=1),
  with all **4 XPU gates** fixed: (1) `assign_extend_cache_locs` draft-slot gather; (2) mamba-scatter
  `is_cuda`-guard stripper; (3) `top_p_renorm_probs` torch nucleus fallback; (4) `eagle_sample` greedy-verify
  branch extended to XPU. Opt-in via `B70_XPU_MTP=1` (the inherited `woq_shim.py` `.pth` installs it).
- `memory_pool.py` -- 3-line `device="cuda"` -> `device=device` fix for the spec-decode mamba state cache.

The `woqgemm` int4 path + `woq_shim.py` are inherited unchanged from `:woq`. To rebuild:
`docker build -t sglang-xpu:mtp sglang/images/sglang-xpu-mtp/`.

## Recipe (baked into serve.sh)
- `--speculative-algorithm NEXTN --speculative-num-steps 7 --speculative-eagle-topk 1
  --speculative-num-draft-tokens 8` -- chain depth-7 = near-peak (15.31 t/s; plateaus ~steps=9). topk=1
  (chain) is the only tree shape the XPU torch fallback supports.
- `--attention-backend intel_xpu --linear-attn-backend triton --mamba-ssm-dtype float32` -- the proven
  GDN-correct sglang-XPU stack. `--disable-cuda-graph` (eager: XPU has no working cudagraph for this model,
  so there is NO L0 graph-replay degradation). `--disable-overlap-schedule --disable-radix-cache --page-size 64`.
- `--skip-server-warmup` -- avoids the startup warmup forward (which poisons GDN state on some quant paths).
- `--max-running-requests 4` -- the spec mamba intermediate-state cache scales with this; `8` OOMs the KV at
  ctx 4096. Requests beyond 4 QUEUE and complete fine. `--mem-fraction-static 0.92`, `--context-length 4096`.

## [!] Two standing caveats
- **GREEDY-ONLY.** MTP verify runs greedily on XPU (gate 4) -> output is the target model's argmax (correct
  greedy decoding) but `temperature`/`top_p`/`top_k` are **IGNORED** (exactly like the NPU/HIP spec path).
  Great for coding/agentic (often greedy anyway); for sampling diversity use the non-MTP int4 woq DP=2 driver
  (`../../sglang/serve_dp2.sh`). Restoring sampling under MTP = open task #14 (a pure-torch chain rejection-sampler).
- **Single-stream / low-concurrency LATENCY lever.** MTP amortizes per-token launch overhead for ONE stream;
  for high concurrency the non-MTP int4 woq DP=2 driver (both cards, ~9.4/replica, sampling OK, wedge-proof)
  is the better aggregate-throughput choice.

## Driver matrix (all CORRECT + VISION on sglang-XPU)
| driver | c1 t/s | aggregate | sampling | cards | use |
|---|---|---|---|---|---|
| **int4 + NEXTN MTP steps=7 (this)** | **~15.3 (1.62x)** | ~26 MC4/card | greedy only | 1 | LATENCY / interactive single stream |
| **int4 + MTP DP=2 (`../../sglang/serve_dp2_mtp.sh`)** | ~15 x2 slots | **~50.7 MC8** | greedy only | 2 | latency AND multi-user (2x full-speed slots, wedge-proof) |
| int4 woq DP=2 (`../../sglang/serve_dp2.sh`) | ~9.4/replica | — | yes | 2 | high-concurrency / unattended (sampling, wedge-proof) |
| bf16 TP=2 (`../../sglang/serve_sglang.sh`) | ~9.2 (c4 agg ~23) | — | yes | 2 | best c4 aggregate, attended |

DP=2 MTP (2026-06-28): two single-card replicas behind nginx :18080. proxy MC2 = 2 users each at full ~15 t/s;
proxy MC8 = ~50.7 tok/s aggregate (~1.9x one card), accept_len holds 4.31 under load. No cross-card collective
-> wedge-proof. This is the throughput/multi-user companion to the single-card latency driver above.

verified: 2026-06-27 (scripts/131, baked sglang-xpu:mtp image, ZERO mounts): healthy + coherent; WARM c1
decode 15.31 t/s (TTFT 911ms @ ctx2048), c4 4.62/stream (agg 17.72); mean accept_len 4.48 (29 batches);
sustained mixed load 21/21 OK (0 garbage/degen, container survived, post-load coherent). Log:
../../sglang/verify_mtp_image.log. (The soak-probe t/s under-counts spec-streamed tokens -- known artifact;
the warm bench_serving number is authoritative.)
