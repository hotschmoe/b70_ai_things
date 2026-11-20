# TensorFold GPU probe journal (2026-11-20)

Format CONFIG -> COMMAND -> RESULT -> VERDICT. GPU yield seen: "GPU YIELDED 2026-11-20 05:26:03 UTC" in ORCHESTRATOR.md (my first poll loop matched the line pattern too narrowly and noticed it at 05:35).
## a. xpu-health both cards (05:35 UTC)
CONFIG: both cards free after GPU YIELDED; DD inactive. COMMAND: bin/gpu-run bin/xpu-health. RESULT: card 0 OK, card 1 OK, HEALTHY. VERDICT: pass.

## b. P2P precheck and device tests (05:36 UTC)
CONFIG: both cards visible (no ZE_AFFINITY_MASK), kernel 7.1.0-070100, binaries from main 9e2a9bd7 (sha256 in README).
COMMAND: bin/gpu-run python3 -I ze_p2p_precheck.py
RESULT: driver 0: 2 devices, both deviceId 0xe223 Intel(R) Arc(TM) Pro B70 Graphics; zeDeviceCanAccessPeer(dev0->dev1)=1, (dev1->dev0)=1.
VERDICT: Level Zero reports peer access both ways on this host (this is the precheck TensorFold's openPeers needs; it says nothing about P2P bandwidth or stability).
COMMAND: gpu-run --card 0 env ZE_AFFINITY_MASK=0 tf-xpu-qwen27-means-test (main and pr3603 builds)
RESULT: "group means: worst error / sum|terms|: prompt 5.12e-5, decode 1.86e-5 (bound 2e-6)" -> test returns error.GroupMeansLost, then the process segfaults (exit 139) in error unwinding. Identical on main and pr3603 builds, deterministic.
VERDICT: FAIL of TensorFold's own synthetic GEMM group-means gate by ~25x on this host (their receipts claim pass). Possible causes not investigated: host ocloc 26.22.38646.4 vs theirs, driver/runtime, our zig c++ wrapper for the nGEN generator. The model-level checks below pass, so treat as an accuracy flag to report upstream, not a blocker.
COMMAND: gpu-run --card 0 env ZE_AFFINITY_MASK=0 tf-xpu-qwen27-close-test GPTQ_DIR
RESULT: ok (3 open/close rounds, 30 s). Round 2 line: "32614 MiB free before the first runner, 30006 MiB after this one closed" (about 2.6 GiB not returned at that point; test still passes).
VERDICT: pass; note the 2.6 GiB residual.
COMMAND: gpu-run --card 0 env ZE_AFFINITY_MASK=0 tf-xpu-qwen27-seam-test GPTQ_DIR
RESULT: cut at 64 and at 1024: 0 of 248320 logits differ from the whole prompt; ok. VERDICT: pass.
COMMAND: gpu-run --card 0 env ZE_AFFINITY_MASK=0 XPU_VERIFY_TEST=1 xpu-qwen27 GPTQ_DIR ids_1024.txt 32
RESULT: perfect drafts equal to serial true; wrong drafts equal to serial true; load 6.7 s (NVMe copy); prompt 1024 tokens 2071 tok/s; serial decode 34.1 tok/s (31 tokens, includes warmup). VERDICT: pass.
## c. Single card base GPTQ (card 0, 05:37-05:50 UTC)
CONFIG: main 9e2a9bd7 tensorfold-native (sha256 e6f10e1d...), SergiioB GPTQ g128 (NVMe copy) + DFlash2 50307d4c, --parallel 8 --context 16384 --temperature 0 --name tf-qwen38-27b-gptq-int4-g128-dflash2 --no-update-check, port 18144, gpu-run --card 0 env ZE_AFFINITY_MASK=0. Device confirmed via /proc/PID/fdinfo: drm-pdev 0000:0b:00.0 (renderD128), 21.08 GB VRAM used. Card 1 = 0000:44:00.0. Idle sysfs link state of both cards reads 2.5 GT/s x1 (idle down-training), so it says nothing about the loaded Gen3 link.
RESULT startup: load 11.7 s; "window ms by rows: 1 29.3, 8 32.0, 16 33.9"; slotsThatFit = 7 streams of 16384 (not 8; 2 GiB prompt cache is reserved first; --prompt-cache-gib 0 gives 8). /v1/models id = served name. Greedy "17 times 23" -> '391' twice (deterministic).
COMMAND: cpu-run --quiet run_bench_c.sh (bench_openai, spread_client, bench_concurrent --alone --serial, bench_cache).
RESULT decode (greedy, bench_openai, 256 tok, median of 3): code 142.3 tok/s, chat 101.8; T=1: 137.9 / 93.1. TTFT 0.09-0.10 s (short prompts).
RESULT concurrency (spread_client, 256 tok, distinct prompts, greedy, chat template, single run each; chunk-gap = gap between SSE chunks, a chunk carries tok/chunk tokens):
 n=1: agg 79.7 (this prompt is a low-acceptance chat prompt), TTFT p50/p95 0.11/0.11 s, chunk gap p50/p95 40.6/41.6 ms, 3.2 tok/chunk
 n=2: agg 175.1, per-stream med/min/max 115/92/138, TTFT 0.12/0.27, gap 40.2/41.8 ms, 4.7 tok/chunk
 n=4: agg 236.8, per-stream 78/62/90, TTFT 0.30/0.45, gap 43.2/45.9 ms, 3.5 tok/chunk
 n=8 (only 7 slots, the 8th queues): agg 224.7, per-stream 45/39/58, TTFT 0.53/4.75, gap 44.6/45.9 ms, 2.2 tok/chunk
RESULT bench_concurrent (code/chat prompts, same text for every stream, greedy aggregate / per-stream tok/s): n=1 142/142 code, 102/102 chat; n=2 228/119, 170/86; n=4 304/79, 235/62; n=8 265/44, 226/40 (TTFT p95 5.2-6.2 s from the queued 8th stream). Sampled cells slightly lower.
RESULT concurrent == solo: --alone and --serial both zero unequal in every cell at n=1,2,4,8 (alone equal 2/4/8/16, serial equal as listed in results/). 
RESULT cold prefill (bench_cache, real-text token ids, cache_state cold): 1024 -> 1.93k tok/s (0.53 s); 4096 -> 1.90k (2.15 s); 8192 -> 1.85k (4.42 s); 12288 -> 1.81k (6.78 s). Two simultaneous prompts serialize: second TTFT = 2x (e.g. 8192: 4.40 / 8.84 s).
RESULT ctx 65536 server (same settings): slotsThatFit = 2 streams of 65536; cold 32768-token prefill 1.62k tok/s (20.2 s); 2 simultaneous 32k: 20.3 / 40.7 s.
RESULT 4 simultaneous cold 20000-token prompts (--parallel 4 --context 24576, slots 4): TTFT 11.6 / 23.2 / 34.8 / 46.5 s (strictly FIFO prefill, aggregate 1.72k tok/s); a single 20k prompt: 11.55 s. Strata comparison point 17/39/61/84 s: TensorFold is 1.5-1.8x lower on the last stream.
VERDICT: base 27B GPTQ on one B70 with DFlash2: decode 142/102 tok/s single (code/chat), 4 streams 235-304 agg, 8 streams ~225-265 agg; prefill ~1.8-1.9k tok/s default; correctness gates (concurrent==solo, drafted==serial) pass. Server stopped with SIGINT: "draining ... stopped", exit 0.

## c2. W4A8 prompt GEMMs (XPU_PREFILL_INT8=1, GPTQ base, ctx 16384)
RESULT cold prefill 1024/4096/8192/12288 = 2.52k / 2.50k / 2.42k / 2.35k tok/s; decode unchanged (greedy code 142.1, chat 91.4 on 2 reps). 4 simultaneous 20k (ctx 24576 par 4): TTFT 9.1 / 18.2 / 27.4 / 36.5 s (aggregate 2.19k tok/s).
VERDICT: meets the 2000+ prefill target; local TensorFold option, their doc claims +0.04..0.13 percent perplexity; quality not checked here.

## d. pi-adapted RTN variants (speed and acceptance only; no quality judgement)
CONFIG: same server shape (ctx 16384, parallel 8, DFlash2) with qwen3.8-27b-pi-adapted-rot-rtn4 and -norot-rtn4 (rot has hadamard128 marker).
RESULT: both load and serve with no refusal (7 slots, load ~12 s). Identity ok; greedy 391 for both. Acceptance (DFlash2 accepted/proposed over 25 identical 256-token requests: 12 bench_openai + 1 + 8 spread): base GPTQ 0.287 (3.49 tok/round), pi rot 0.290 (3.43), pi norot 0.295 (3.55). bench_openai greedy code/chat tok/s: base 141.6/101.6, pi rot 159.1/99.7, pi norot 145.1/95.6 (generated text differs, so tok/s differences mostly reflect text). spread n=8: base agg 246 (per-stream med 45.8, TTFT p50/p95 0.66/4.69), rot 240 (43.6, 0.67/4.96), norot 259 (48.1, 0.67/5.14). 4096 cold prefill (default path): base 1.90k, rot 1.86k, norot 1.90k tok/s.
W4A4 arm (rot + XPU_PREFILL_INT8=1 XPU_PREFILL_INT4=1, pi-adapted-rot-rtn4): starts, greedy 391; cold prefill 1024/4096/8192/12288 = 2.72k / 2.69k / 2.60k / 2.51k tok/s; decode greedy code 141.8, chat 100.1.
VERDICT: DFlash2 acceptance is essentially unchanged by the LoRA (0.29 vs 0.29), so the pi model should serve at base speed. Not evaluated: any pi quality.

## e. Two replicas (card 0 port 18144, card 1 port 18145, 06:01 UTC)
CONFIG: two independent tensorfold-native processes, each its own gpu-run --card N + ZE_AFFINITY_MASK=N; fdinfo: 18144 -> 0000:0b:00.0, 18145 -> 0000:44:00.0, 21.08 GB each. Client spread_client round-robin over both URLs, 256-token greedy, distinct prompts.
RESULT default (7 slots per replica): n=2 (1+1) agg 160; n=8 (4+4) agg 399, per-stream med 73, TTFT p50/p95 0.35/0.52 s, chunk gap p50/p95 43.5/45.6 ms; n=14 (7+7) agg 477, per-stream med 48 (min 35, max 71), TTFT 0.39/0.84, gap 44.5/45.5; n=16 (3 runs) agg 547 / 474 / 483, per-stream med 47-48 (min 33-35), TTFT p50/p95 0.46-0.48 / 3.95-4.14 s (the 2 streams beyond the 14 slots queue), chunk gap p50/p95 44.5/45.8 ms, 2.3 tok/chunk.
RESULT --prompt-cache-gib 0 (8 slots per replica): n=16 x3: agg 498 / 508 / 516, per-stream med 42-46 (min 32-34, max 59-66), TTFT p50/p95 0.47-0.68 / 0.73-1.14 s, chunk gap p50/p95 44.8/45.9 ms, 2.0-2.15 tok/chunk, zero errors.
VERDICT: 16 streams at ~500 tok/s aggregate and ~45 tok/s per stream with sub-1.2 s TTFT p95 when the prompt cache is disabled; with the cache on, two streams queue and TTFT p95 is ~4 s. Per-stream decode at 16 is ~45 tok/s, far from the 150 tok/s single-stream goal (concurrency trades per-stream speed: ~45 ms per round regardless of streams).

## f. pr3603 tf-xpu-tp-bench on both cards (06:07 UTC)
CONFIG: pr3603 207aaae1 binary (sha256 475d95e6...), gpu-run both cards, no ZE_AFFINITY_MASK, timeout 600.
COMMAND: bin/gpu-run pr3603/zig-out/bin/tf-xpu-tp-bench
RESULT: opened peers (precheck passed) but produced NO output; at 06:07:38 UTC the kernel logged on card 1 (0000:44:00.0): "Check job timeout ... guc_id=2", "Timedout job ... in tf-xpu-tp-bench", devcoredump created, "GT0: reset queued/started/done". Process spun at ~90 percent CPU for 2 min until I killed it (SIGTERM then SIGKILL, exit 143). xpu-health after: card 0 OK, card 1 OK, HEALTHY. No xe-reset was needed.
VERDICT: FAIL. The first cross-card collective submission hangs on this host (PCIe Gen3 pair, kernel 7.1) and the GuC resets the card-1 engine; same failure class as the repo's recorded "loaded push graph stalls at first submission". No P2P collective timings obtained. zeDeviceCanAccessPeer=1 both ways is therefore NOT sufficient evidence that TensorFold's TP path works here. Devcoredump left at /sys/class/drm/card1/device/devcoredump/data (not saved).

## g. Final health (06:10 UTC)
RESULT: gpu-run bin/xpu-health: card 0 OK, card 1 OK, HEALTHY. No server left running (pgrep clean, ports 18144/18145 closed). GPU RESUME line appended to ORCHESTRATOR.md at 06:10:20 UTC. Total GPU time ~35 min.

