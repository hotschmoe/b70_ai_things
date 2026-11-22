# GPU claims (append-only: time UTC, lead, cards, expected minutes, purpose; then release line)

2026-11-20 07:11 UTC [B-worker sonnet-means] claim cards 1 ~10 min: health, means-test baseline vs fix (short runs)
2026-11-20 07:12:04 UTC [B-worker codex-lat] claim cards 1 ~25 min: single-card health and latency suite
2026-11-20 07:14:44 UTC [B-worker codex-lat] release cards 1
2026-11-20 07:15 UTC [B-worker sonnet-means] claim cards 1 ~12 min: xpu-qwen27 + seam test before/after Alo fix (27B GPTQ, ids_1024)
2026-11-20 07:16 UTC [A-K] card 0, ~2 min, initial fn-gguf health check
2026-11-20 07:16 UTC [A-K] released card 0
2026-11-20 07:17 UTC [A-K] card 0, ~3 min, fn-gguf initial IQ4_XS parity
2026-11-20 07:17 UTC [A-K] released card 0
2026-11-20 07:17 UTC [A-K] card 0, ~3 min, fn-gguf dense parity formats
2026-11-20 07:17 UTC [A-K] released card 0
2026-11-20 07:23:55 UTC [B-worker codex-lat] claim cards 1 ~5 min: clock mapping verification and health
2026-11-20 07:24 UTC [A-K] card 0, ~3 min, fn-gguf MTP Q8 and column-slice parity
2026-11-20 07:24:29 UTC [B-worker codex-lat] release cards 1
2026-11-20 07:24 UTC [A-K] released card 0
2026-11-20 07:24 UTC [A-K] card 0, ~3 min, fn-gguf repeated-expert IQ2_XS parity
2026-11-20 07:24 UTC [A-K] released card 0
2026-11-20 07:25 UTC [A-K] card 0, ~3 min, fn-gguf repeated-expert parity formats
2026-11-20 07:25 UTC [B-worker sonnet-means] release cards 1
2026-11-20 07:26 UTC [A-K] card 0, ~3 min, fn-gguf diagnostic timings
2026-11-20 07:27 UTC [A-lead] claim cards 0+1 ~25 min: Strata 0216 oracle (plain mode, stdin protocol, 8 prompts, no HTTP port), health before/after
2026-11-20 07:28 UTC [A-lead] released cards 0+1 (oracle)
2026-11-20 07:28 UTC [A-lead] claim cards 0+1 ~25 min: Strata oracle retry (shard-2 symlink mount fix)
2026-11-20 07:29 UTC [A-lead] released cards 0+1 (oracle retry)
2026-11-20 07:30 UTC [A-lead] claim cards 0+1 ~25 min: Strata oracle retry 2 (--spec 2 MTP minimum)
2026-11-20 07:32 UTC [A-K] released card 0
2026-11-20 07:32 UTC [B-worker sonnet-means] claim cards 1 ~3 min: unwinding check + post-health
2026-11-20 07:33 UTC [A-lead] released cards 0+1 (oracle retry 2)
2026-11-20 07:33 UTC [A-K] released card 0
2026-11-20 07:33 UTC [A-K] card 0, ~3 min, fn-gguf isolated and full expert parity
2026-11-20 07:33 UTC [B-worker sonnet-means] release cards 1
2026-11-20 07:39 UTC [A-K] released card 0
2026-11-20 07:40 UTC [A-K] card 0, ~3 min, fn-gguf final expert parity and timing
2026-11-20 07:43:49 UTC [B-worker codex-lat] claim cards 1 ~8 min: decode GEMV and host-USM reads
2026-11-20 07:44:20 UTC [B-worker codex-lat] release cards 1
2026-11-20 07:44:20 UTC [B-worker codex-trace] claim cards 1 ~15 min: runner-parity
2026-11-20 07:44 UTC [B-worker sonnet-ab] claim cards 1 ~2 min: xpu-health before A/B
2026-11-20 07:44:54 UTC [B-worker codex-trace] release cards 1
2026-11-20 07:44 UTC [A-K] released card 0
2026-11-20 07:45 UTC [B-worker sonnet-ab] release cards 1
2026-11-20 07:45 UTC [A-K] card 0, ~2 min, final fn-gguf health check
2026-11-20 07:45 UTC [A-K] released card 0
2026-11-20 07:45:33 UTC [B-worker codex-lat] claim cards 0,1 ~25 min: pair health and fresh-event push exchange
2026-11-20 07:45 UTC [B-worker sonnet-ab] claim cards 1 ~9 min: A/B arm old1 (old) serve + benches on port 18152
2026-11-20 07:47:12 UTC [B-worker codex-lat] release cards 0,1
2026-11-20 07:48:07 UTC [B-worker codex-trace] claim cards 1 ~15 min: runner-verification
2026-11-20 07:48:45 UTC [B-worker codex-trace] release cards 1
2026-11-20 07:49:08 UTC [B-worker codex-trace] release cards 1 (verification wait cancelled; GPU lease was never acquired)
2026-11-20 07:52 UTC [B-worker sonnet-ab] release cards 1
2026-11-20 07:52 UTC [A-K] card 0, ~1 min, fn-gguf pre-check health
2026-11-20 07:52 UTC [A-K] released card 0
2026-11-20 07:52 UTC [A-K] card 0, ~2 min, fn-gguf strict expert verdict verification
2026-11-20 07:52 UTC [A-K] released card 0
2026-11-20 07:52 UTC [B-worker sonnet-ab] claim cards 1 ~10 min: A/B arm old1 (old) serve + benches on port 18152
2026-11-20 07:52 UTC [A-K] card 0, ~1 min, fn-gguf post-check health
2026-11-20 07:53 UTC [A-K] released card 0
2026-11-20 07:55 UTC [L] card 0, ~20 min, GGUF rank loads and BF16 HC validation
2026-11-20 07:55:48 UTC [A-KF] claim card 0 ~1 min: pre-run health
2026-11-20 07:56:04 UTC [A-KF] released card 0
2026-11-20 07:56:31 UTC [A-KF] claim card 0 ~2 min: leased health and IQ4_XS parity smoke
2026-11-20 07:56:46 UTC [A-KF] released card 0
2026-11-20 08:01 UTC [B-worker sonnet-ab] release cards 1
2026-11-20 08:02:22 UTC [B-worker codex-trace] claim cards 1 ~15 min: runner-verification
2026-11-20 08:02 UTC [B-worker sonnet-ab] claim cards 1 ~10 min: A/B arm old1 (old) serve + benches on port 18152
2026-11-20 08:02:34 UTC [B-worker codex-trace] release cards 1
2026-11-20 08:03:30 UTC [A-KF] claim card 0 ~5 min: dense parity formats 1-5
2026-11-20 08:04 UTC [L] released card 0
2026-11-20 08:04:50 UTC [A-KF] released card 0
2026-11-20 08:04:54 UTC [A-KF] claim card 0 ~5 min: dense parity formats 6-10
2026-11-20 08:05:58 UTC [A-KF] released card 0
2026-11-20 08:06:02 UTC [A-KF] claim card 0 ~5 min: remaining dense formats and column slices
2026-11-20 08:09:06 UTC [A-KF] released card 0
2026-11-20 08:09:11 UTC [A-KF] claim card 0 ~5 min: expert parity and reference comparisons
2026-11-20 08:09 UTC [B-worker sonnet-ab] release cards 1
2026-11-20 08:09 UTC [B-worker sonnet-ab] claim cards 1 ~10 min: A/B arm new1 (hilofix) serve + benches on port 18152
2026-11-20 08:10:16 UTC [B-worker codex-trace] claim cards 1 ~15 min: serve-acceptance
2026-11-20 08:10:17 UTC [B-worker codex-trace] release cards 1
2026-11-20 08:10:22 UTC [A-KF] released card 0
2026-11-20 08:10:52 UTC [A-KF] claim card 0 ~5 min: device timestamp benchmarks specialized path
2026-11-20 08:12:08 UTC [A-KF] released card 0
2026-11-20 08:12:14 UTC [A-KF] claim card 0 ~5 min: device timestamp benchmarks reference path
2026-11-20 08:12:53 UTC [A-KF] released card 0
2026-11-20 08:16 UTC [B-worker sonnet-ab] release cards 1
2026-11-20 08:16 UTC [B-worker sonnet-ab] claim cards 1 ~10 min: A/B arm old2 (old) serve + benches on port 18152
2026-11-20 08:16:31 UTC [B-worker codex-trace] claim cards 1 ~15 min: serve-acceptance
2026-11-20 08:16:32 UTC [B-worker codex-trace] release cards 1
2026-11-20 08:18:55 UTC [A-KF] claim card 0 ~5 min: final dense parity after BF16 vector and wide fallback changes
2026-11-20 08:20:11 UTC [A-KF] released card 0
2026-11-20 08:20:21 UTC [A-KF] claim card 0 ~5 min: final slices and expert chain parity
2026-11-20 08:23:31 UTC [A-KF] released card 0
2026-11-20 08:23:31 UTC [A-MT] claim card 0 ~5 min: MTP rank load and upload verification with health checks
2026-11-20 08:23 UTC [B-worker sonnet-ab] release cards 1
2026-11-20 08:23:34 UTC [A-KF] claim card 0 ~5 min: final device timestamp benchmarks
2026-11-20 08:23 UTC [B-worker sonnet-ab] claim cards 1 ~2 min: xpu-health after A/B
2026-11-20 08:24 UTC [B-worker sonnet-ab] release cards 1
2026-11-20 08:24:09 UTC [A-MT] released card 0
2026-11-20 08:24:54 UTC [B-worker codex-trace] claim cards 1 ~15 min: serve-acceptance
2026-11-20 08:25:30 UTC [A-I] claim cards 0+1 ~2 min: pre-GGUF-forward xpu-health (queued behind leases)
2026-11-20 08:25:35 UTC [B-worker codex-trace] release cards 1
2026-11-20 08:26:03 UTC [A-I] released cards 0+1
2026-11-20 08:26:09 UTC [A-KF] released card 0
2026-11-20 08:26:09 UTC [B-worker codex-lat] claim cards 1 ~5 min: host USM flag wake-up and health
2026-11-20 08:26:13 UTC [A-KF] claim card 0 ~5 min: final reference device timestamp benchmarks
2026-11-20 08:26:20 UTC [A-I] claim cards 0+1 ~3 min: GGUF TP2 host-AR p0 first-token comparison
2026-11-20 08:26:38 UTC [B-worker codex-lat] release cards 1
2026-11-20 08:26:47 UTC [A-KF] released card 0
2026-11-20 08:27:04 UTC [A-I] released cards 0+1
2026-11-20 08:27:26 UTC [B-worker codex-lat] claim cards 0,1 ~25 min: pair transport comparison
2026-11-20 08:27:32 UTC [A-KF] claim card 0 ~5 min: final parity matrix after wide-row selection
2026-11-20 08:29:37 UTC [B-worker codex-lat] release cards 0,1
2026-11-20 08:31:41 UTC [A-KF] released card 0
2026-11-20 08:31:49 UTC [A-KF] claim card 0 ~5 min: IQ3_XXS wide-row final timestamps
2026-11-20 08:32:14 UTC [B-worker codex-lat] claim cards 0,1 ~10 min: recovery after pull engine timeout
2026-11-20 08:32:40 UTC [A-I] claim cards 0+1 ~8 min: GGUF TP2 host-AR teacher eight prompts run 1 (p7 last), timeout 600s
2026-11-20 08:33:12 UTC [B-worker codex-lat] release cards 0,1
2026-11-20 08:33:12 UTC [B-worker codex-trace] claim cards 1 ~15 min: serve-acceptance
2026-11-20 08:33:46 UTC [B-worker codex-trace] release cards 1
2026-11-20 08:33:46 UTC [A-KF] released card 0
2026-11-20 08:33:51 UTC [A-KF] claim card 0 ~5 min: post-run card 0 health
2026-11-20 08:37:42 UTC [A-I] released cards 0+1
2026-11-20 08:37:57 UTC [A-KF] released card 0
2026-11-20 08:37:57 UTC [B-worker codex-lat] claim cards 0,1 ~5 min: bounded repro-tp subdevice default
2026-11-20 08:39:20 UTC [B-worker codex-lat] release cards 0,1
2026-11-20 08:39:47 UTC [A-I] claim cards 0+1 ~5 min: GGUF TP2 host-AR teacher eight prompts run 2 determinism, timeout 600s
2026-11-20 08:41:26 UTC [B-worker codex-pushar] claim cards 0,1 ~4 min: required pre-run health checks
2026-11-20 08:42:26 UTC [B-worker codex-pushar] release cards 0,1
2026-11-20 08:42:51 UTC [B-worker codex-pushar] claim cards 0,1 ~8 min: pre-health and bounded push correctness
2026-11-20 08:43:43 UTC [A-I] claim cards 0+1 ~1 min: post-GGUF-forward xpu-health (queued after run 2)
2026-11-20 08:46:17 UTC [A-KF] claim card 0 ~5 min: round2 pre-run health
2026-11-20 08:46:25 UTC [A-I] released cards 0+1
2026-11-20 08:46:25 UTC [B-worker codex-pushar] release cards 0,1
2026-11-20 08:46:57 UTC [A-I] released cards 0+1
2026-11-20 08:46:58 UTC [A-I] claim cards 0+1 ~1 min: GGUF TP2 p0 serial/verify rollback check plus final health, timeout 300s
2026-11-20 08:47:13 UTC [A-KF] released card 0
2026-11-20 08:47:27 UTC [B-worker codex-pushar] claim cards 0,1 ~9 min: push soak and health
2026-11-20 08:47:43 UTC [A-I] released cards 0+1
2026-11-20 08:48:17 UTC [B-worker codex-lat] claim cards 0,1 ~5 min: bounded repro-reuse host default
2026-11-20 08:49:24 UTC [B-worker codex-lat] release cards 0,1
2026-11-20 08:49:53 UTC [B-worker codex-pushar] release cards 0,1
2026-11-20 08:50:11 UTC [B-worker codex-pushar] claim cards 0,1 ~8 min: recovery after preceding host-event device loss
2026-11-20 08:51:10 UTC [A-I] claim cards 0+1 ~1 min: corrected xpu-flashnext GGUF serial/verify rollback plus final health, timeout 300s
2026-11-20 08:51:22 UTC [B-worker codex-pushar] release cards 0,1
2026-11-20 08:51:32 UTC [B-worker codex-pushar] claim cards 0,1 ~9 min: push soak and health
2026-11-20 08:53:45 UTC [B-worker codex-park] claim cards 1 ~5 min: pre-health, bounded host-tier copy soak, post-health
2026-11-20 08:54:21 UTC [B-worker codex-park] release cards 1
2026-11-20 08:56:15 UTC [A-I] released cards 0+1
2026-11-20 08:56:38 UTC [A-KF] claim card 0 ~5 min: Strata stock parity and kernel benches
2026-11-20 08:56:45 UTC [B-worker codex-park] claim cards 1 ~5 min: card 1 copy-engine 1 MiB-2 GiB matrix and health
2026-11-20 08:56:45 UTC [B-worker codex-pushar] release cards 0,1
2026-11-20 08:57:05 UTC [B-worker codex-pushar] claim cards 0,1 ~9 min: push timing and health
2026-11-20 08:57:11 UTC [A-KF] released card 0
2026-11-20 08:57:31 UTC [B-worker codex-park] release cards 1
2026-11-20 08:57:48 UTC [B-worker codex-park] claim cards 0,1 ~8 min: both-card engine transfer matrix and health
2026-11-20 08:58:02 UTC [B-worker codex-pushar] release cards 0,1
2026-11-20 08:58:04 UTC [A-KF] claim card 0 ~5 min: Strata original parity and bench controls
2026-11-20 08:58:20 UTC [A-KF] claim card 0 ~5 min: Strata matched real-tensor device reference table
2026-11-20 08:59:21 UTC [B-worker codex-park] release cards 0,1
2026-11-20 08:59:34 UTC [A-KF] released card 0
2026-11-20 09:00:09 UTC [A-KF] claim card 0 ~5 min: Strata remaining parity siblings and original benches
2026-11-20 09:00:12 UTC [A-KF] released card 0
2026-11-20 09:01:16 UTC [B-worker codex-pushar] claim cards 0,1 ~9 min: push timing-160 and health
2026-11-20 09:01:28 UTC [B-worker codex-park] claim cards 0,1 ~8 min: quiet transfer matrix and duplex probe, pre/post health
2026-11-20 09:03:11 UTC [A-KF] released card 0
2026-11-20 09:03:11 UTC [B-worker codex-lat] claim cards 0,1 ~5 min: bounded repro-reuse host workaround
2026-11-20 09:04:30 UTC [A-I] claim cards 0+1 ~3 min: M-A4 health-a4-pre.log
2026-11-20 09:04:31 UTC [B-worker codex-lat] release cards 0,1
2026-11-20 09:05:02 UTC [B-worker codex-pushar] release cards 0,1
2026-11-20 09:05:23 UTC [A-KF] claim card 0 ~5 min: Strata IQ2 row-reuse parity control without lane override
2026-11-20 09:06:29 UTC [B-worker codex-park] release cards 0,1
2026-11-20 09:08:59 UTC [A-KF] claim card 0 ~5 min: Strata complete expert and real rank-slice device reference
2026-11-20 09:09:24 UTC [B-worker codex-park] claim cards 1 ~6 min: EU host-store compile and verified transfers, pre/post health
2026-11-20 09:10:07 UTC [A-I] released cards 0+1
2026-11-20 09:10:07 UTC [B-worker codex-lat] claim cards 0,1 ~25 min: pair transport comparison
2026-11-20 09:10:10 UTC [A-I] claim cards 0+1 ~10 min: M-A4 a4-p0.log
2026-11-20 09:11:02 UTC [B-worker codex-lat] release cards 0,1
2026-11-20 09:11:15 UTC [A-KF] released card 0
2026-11-20 09:11:41 UTC [B-worker codex-lat] claim cards 0,1 ~3 min: post-health after event clock error
2026-11-20 09:11:47 UTC [B-worker codex-pushar] claim cards 0,1 ~9 min: push timing-160 and health
2026-11-20 09:12:12 UTC [B-worker codex-lat] release cards 0,1
2026-11-20 09:13:25 UTC [A-KF] released card 0
2026-11-20 09:14:16 UTC [B-worker codex-park] release cards 1
2026-11-20 09:16:16 UTC [A-I] released cards 0+1
2026-11-20 09:16:46 UTC [B-worker codex-pushar] release cards 0,1
2026-11-20 09:19:17 UTC [B-worker codex-lat] claim cards 0,1 ~25 min: pair transport comparison
2026-11-20 09:19:28 UTC [B-worker codex-pushar] claim cards 0,1 ~9 min: push timing-160 and health
2026-11-20 09:19:35 UTC [B-worker codex-park] claim cards 1 ~3 min: pre-27B health
2026-11-20 09:20:22 UTC [B-worker codex-lat] release cards 0,1
2026-11-20 09:20:37 UTC [B-worker codex-park] release cards 1
2026-11-20 09:20:38 UTC [B-worker codex-park] claim cards 1 ~12 min: 27B qwen-smoke-int8-same
2026-11-20 09:21:04 UTC [B-worker codex-lat] claim cards 0,1 ~3 min: post-health after EU clock error
2026-11-20 09:21:34 UTC [B-worker codex-lat] release cards 0,1
2026-11-20 09:21:37 UTC [A-I] claim cards 0+1 ~10 min: M-A4 a4-p1.log
2026-11-20 09:24:49 UTC [B-worker codex-pushar] release cards 0,1 (queued attempt cancelled before acquisition)
2026-11-20 09:24:49 UTC [B-worker codex-pushar] claim cards 0,1 ~3 min: receiver-barrier correctness only
2026-11-20 09:25:18 UTC [B-worker codex-pushar] release cards 0,1
2026-11-20 09:25:30 UTC [B-worker codex-pushar] claim cards 0,1 ~4 min: receiver-barrier 21000-hop correctness stress
2026-11-20 09:25:34 UTC [A-KF] claim card 0 ~5 min: Strata final qualified dense and rank slices with NIGHT IQ3_S
2026-11-20 09:26:03 UTC [B-worker codex-pushar] release cards 0,1
2026-11-20 09:26:11 UTC [B-worker codex-pushar] claim cards 0,1 ~8 min: pre-health and bounded push correctness
2026-11-20 09:26:11 UTC [B-worker codex-pushar] release cards 0,1
2026-11-20 09:26:15 UTC [B-worker codex-pushar] claim cards 0,1 ~9 min: push timing and health
2026-11-20 09:26:27 UTC [B-worker codex-park] release cards 1
2026-11-20 09:27:55 UTC [A-I] released cards 0+1
2026-11-20 09:27:55 UTC [A-I] claim cards 0+1 ~10 min: M-A4 a4-p2.log
2026-11-20 09:29:22 UTC [A-KF] released card 0
2026-11-20 09:29:51 UTC [B-worker codex-pushar] release cards 0,1
2026-11-20 09:30:07 UTC [B-worker codex-pushar] claim cards 0,1 ~9 min: push latency and health
2026-11-20 09:30:42 UTC [B-worker codex-park] claim cards 1 ~12 min: 27B qwen-base-int8-16
2026-11-20 09:31:50 UTC [A-I] released cards 0+1
2026-11-20 09:31:50 UTC [B-worker codex-trace] claim cards 1 ~15 min: async-smoke-serve
2026-11-20 09:31:50 UTC [A-I] claim cards 0+1 ~10 min: M-A4 a4-p3.log
2026-11-20 09:33:12 UTC [B-worker codex-trace] release cards 1
2026-11-20 09:33:44 UTC [B-worker codex-pushar] release cards 0,1
2026-11-20 09:34:29 UTC [B-worker codex-park] release cards 1
2026-11-20 09:34:29 UTC [B-worker codex-park] claim cards 1 ~12 min: 27B qwen-park-int8-other-k1-16
2026-11-20 09:36:03 UTC [A-I] released cards 0+1
2026-11-20 09:36:04 UTC [A-I] claim cards 0+1 ~10 min: M-A4 a4-p4.log
2026-11-20 09:36:09 UTC [B-worker codex-trace] claim cards 1 ~15 min: async-full-serve
2026-11-20 09:36:55 UTC [B-worker codex-trace] release cards 1
2026-11-20 09:37:26 UTC [B-worker codex-park] release cards 1
2026-11-20 09:37:31 UTC [B-worker codex-pushar] claim cards 0,1 ~9 min: push latency and health
2026-11-20 09:38:58 UTC [A-I] released cards 0+1
2026-11-20 09:38:58 UTC [A-I] claim cards 0+1 ~10 min: M-A4 a4-p5.log
2026-11-20 09:39:32 UTC [B-worker codex-trace] claim cards 1 ~15 min: async-markers-serve
2026-11-20 09:40:20 UTC [B-worker codex-trace] release cards 1
2026-11-20 09:40:51 UTC [B-worker codex-pushar] release cards 0,1
2026-11-20 09:41:01 UTC [B-worker codex-pushar] claim cards 0,1 ~9 min: push timing and health
2026-11-20 09:41:48 UTC [B-worker codex-park] claim cards 1 ~12 min: 27B qwen-draft-base-int8-32
2026-11-20 09:43:54 UTC [A-I] released cards 0+1
2026-11-20 09:43:54 UTC [A-I] claim cards 0+1 ~10 min: M-A4 a4-p6.log
2026-11-20 09:44:24 UTC [B-worker codex-pushar] release cards 0,1
2026-11-20 09:49:46 UTC [B-worker codex-park] release cards 1
2026-11-20 09:49:46 UTC [B-worker codex-park] claim cards 1 ~12 min: 27B qwen-draft-park-int8-other-k1-16
2026-11-20 09:50:52 UTC [A-I] released cards 0+1
2026-11-20 09:50:52 UTC [A-I] claim cards 0+1 ~10 min: M-A4 a4-p7.log
2026-11-20 09:52:03 UTC [B-worker codex-park] release cards 1
2026-11-20 09:52:03 UTC [B-worker codex-trace] claim cards 1 ~15 min: async-smoke-serve
2026-11-20 09:52:18 UTC [B-worker codex-pushar] claim cards 0,1 ~8 min: bounded counter-push transport probe
2026-11-20 09:53:27 UTC [B-worker codex-trace] release cards 1
2026-11-20 09:53:28 UTC [B-worker codex-lat] claim cards 0,1 ~25 min: chained-hop and 48-layer skeleton
2026-11-20 10:02:18 UTC [B-worker codex-pushar] release cards 0,1
2026-11-20 10:02:22 UTC [B-worker codex-pushar] claim cards 0,1 ~8 min: bounded counter-push transport probe
2026-11-20 10:02:35 UTC [B-worker codex-lat] release cards 0,1
2026-11-20 10:03:33 UTC [B-worker codex-pushar] claim cards 0,1 ~8 min: recovery after bounded peer-spin timeouts
2026-11-20 10:03:43 UTC [A-KF] claim card 0 ~5 min: q8-pre-health
2026-11-20 10:05:55 UTC [A-I] released cards 0+1
2026-11-20 10:06:34 UTC [A-I] claim cards 0+1 ~10 min: M-A4 a4-C2-T2.log
2026-11-20 10:07:01 UTC [B-worker codex-pushar] release cards 0,1
2026-11-20 10:07:33 UTC [B-worker codex-pushar] release cards 0,1
2026-11-20 10:07:42 UTC [B-worker codex-pushar] claim cards 0,1 ~8 min: bounded raw-push transport probe
2026-11-20 10:08:14 UTC [B-worker codex-park] claim cards 1 ~12 min: 27B qwen-fault-int8-same
2026-11-20 10:08:21 UTC [A-KF] released card 0
2026-11-20 10:09:27 UTC [B-worker codex-pushar] release cards 0,1 (queued attempt cancelled before acquisition)
2026-11-20 10:09:27 UTC [B-worker codex-pushar] claim cards 0,1 ~9 min: push soak and health
2026-11-20 10:09:50 UTC [A-KF] claim card 0 ~5 min: q8-smoke-dense
2026-11-20 10:11:56 UTC [B-worker codex-pushar] release cards 0,1 (queued attempt cancelled before acquisition)
2026-11-20 10:11:56 UTC [B-worker codex-pushar] claim cards 0,1 ~4 min: single-process push soak correctness
2026-11-20 10:12:24 UTC [B-worker codex-pushar] release cards 0,1
2026-11-20 10:12:37 UTC [B-worker codex-pushar] claim cards 0,1 ~8 min: bounded raw-push transport probe
2026-11-20 10:14:04 UTC [A-I] released cards 0+1
2026-11-20 10:14:04 UTC [A-I] claim cards 0+1 ~10 min: M-A4 a4-C2-T4.log
2026-11-20 10:14:12 UTC [B-worker codex-park] release cards 1
2026-11-20 10:14:12 UTC [B-worker codex-park] claim cards 1 ~3 min: health after cancelled restore
2026-11-20 10:14:13 UTC [A-KF] released card 0
2026-11-20 10:14:27 UTC [B-worker codex-park] release cards 1
2026-11-20 10:14:27 UTC [B-worker codex-park] claim cards 1 ~12 min: 27B qwen-interference-int8
2026-11-20 10:14:28 UTC [A-KF] claim card 0 ~5 min: q8-expert-parity
2026-11-20 10:15:16 UTC [B-worker codex-pushar] release cards 0,1
2026-11-20 10:15:25 UTC [B-worker codex-pushar] claim cards 0,1 ~8 min: bounded raw-push-compute transport probe
2026-11-20 10:16:02 UTC [A-I] released cards 0+1
2026-11-20 10:16:02 UTC [A-I] claim cards 0+1 ~10 min: M-A4 a4-C4-T2.log
2026-11-20 10:16:37 UTC [B-worker codex-park] release cards 1
2026-11-20 10:16:37 UTC [B-worker codex-park] claim cards 1 ~12 min: 27B qwen-lat-int8-8k
2026-11-20 10:16:42 UTC [A-KF] released card 0
2026-11-20 10:17:32 UTC [A-KF] claim card 0 ~5 min: full-parity-q8-and-bf16
2026-11-20 10:18:07 UTC [B-worker codex-pushar] release cards 0,1
2026-11-20 10:18:27 UTC [B-worker codex-pushar] claim cards 0,1 ~8 min: bounded eu-push transport probe
2026-11-20 10:19:14 UTC [A-I] released cards 0+1
2026-11-20 10:19:14 UTC [A-I] claim cards 0+1 ~10 min: M-A4 a4-C4-T4.log
2026-11-20 10:19:27 UTC [B-worker codex-park] release cards 1
2026-11-20 10:19:27 UTC [B-worker codex-trace] claim cards 1 ~15 min: async-smoke-acceptance
2026-11-20 10:19:27 UTC [B-worker codex-park] claim cards 1 ~12 min: 27B qwen-lat-int8-16k
2026-11-20 10:20:50 UTC [B-worker codex-trace] release cards 1
2026-11-20 10:21:08 UTC [A-KF] released card 0
2026-11-20 10:21:55 UTC [B-worker codex-pushar] release cards 0,1 (EU probe exited; release receipt repaired)
2026-11-20 10:22:47 UTC [A-I] released cards 0+1
2026-11-20 10:22:47 UTC [A-I] claim cards 0+1 ~10 min: M-A4 a4-C4qsa-T2.log
2026-11-20 10:23:05 UTC [B-worker codex-park] release cards 1
2026-11-20 10:23:05 UTC [B-worker codex-trace] claim cards 1 ~15 min: async-full-acceptance
2026-11-20 10:23:47 UTC [B-worker codex-park] claim cards 1 ~12 min: 27B qwen-lat-int8-1k
2026-11-20 10:23:52 UTC [B-worker codex-trace] release cards 1
2026-11-20 10:24:41 UTC [A-KF] claim card 0 ~5 min: qualified-default
2026-11-20 10:26:59 UTC [A-I] released cards 0+1
2026-11-20 10:26:59 UTC [A-I] claim cards 0+1 ~10 min: M-A4 a4-C4qsa-T4.log
2026-11-20 10:27:27 UTC [B-worker codex-park] release cards 1
2026-11-20 10:27:27 UTC [B-worker codex-trace] claim cards 1 ~15 min: async-markers-acceptance
2026-11-20 10:27:59 UTC [B-worker codex-pushar] claim cards 0,1 ~8 min: pre-health and bounded push correctness
2026-11-20 10:28:05 UTC [B-worker codex-pushar] release cards 0,1
2026-11-20 10:28:13 UTC [B-worker codex-trace] release cards 1
2026-11-20 10:28:13 UTC [B-worker codex-pushar] claim cards 0,1 ~8 min: bounded push correctness
2026-11-20 10:28:55 UTC [A-KF] released card 0
2026-11-20 10:29:36 UTC [B-worker codex-park] claim cards 1 ~12 min: 27B fast-base-int8
2026-11-20 10:32:13 UTC [A-I] released cards 0+1
2026-11-20 10:32:13 UTC [A-I] claim cards 0+1 ~10 min: M-A4 a4-C4stagger-T2.log
2026-11-20 10:32:14 UTC [B-worker codex-pushar] release cards 0,1
2026-11-20 10:33:16 UTC [B-worker codex-pushar] claim cards 0,1 ~4 min: signed-zero kernel single-process correctness soak
2026-11-20 10:33:48 UTC [B-worker codex-pushar] release cards 0,1
2026-11-20 10:33:57 UTC [B-worker codex-park] release cards 1
2026-11-20 10:33:57 UTC [B-worker codex-park] claim cards 1 ~12 min: 27B fast-park-int8-same-k1
2026-11-20 10:34:14 UTC [B-worker codex-pushar] claim cards 0,1 ~4 min: signed-zero kernel 21000-hop stress
2026-11-20 10:34:45 UTC [B-worker codex-pushar] release cards 0,1
2026-11-20 10:35:03 UTC [B-worker codex-pushar] claim cards 0,1 ~9 min: push timing and health
2026-11-20 10:35:50 UTC [A-I] released cards 0+1
2026-11-20 10:35:50 UTC [A-I] claim cards 0+1 ~10 min: M-A4 a4-C4stagger-T4.log
2026-11-20 10:36:01 UTC [B-worker codex-park] release cards 1
2026-11-20 10:36:01 UTC [B-worker codex-park] claim cards 1 ~12 min: 27B fast-park-int8-other-k1
2026-11-20 10:36:30 UTC [B-worker codex-pushar] release cards 0,1
2026-11-20 10:36:41 UTC [B-worker codex-pushar] claim cards 0,1 ~9 min: push latency and health
2026-11-20 10:36:45 UTC [A-KF] claim card 0 ~5 min: qualified-native-half
2026-11-20 10:37:40 UTC [A-I] released cards 0+1
2026-11-20 10:37:46 UTC [A-I] claim cards 0+1 ~3 min: M-A4 health-a4-post.log
2026-11-20 10:37:51 UTC [B-worker codex-park] release cards 1
2026-11-20 10:37:51 UTC [B-worker codex-park] claim cards 1 ~12 min: 27B fast-draft-base-int8
2026-11-20 10:38:23 UTC [B-worker codex-pushar] release cards 0,1
2026-11-20 10:39:13 UTC [A-KF] released card 0
2026-11-20 10:39:42 UTC [A-I] released cards 0+1
2026-11-20 10:39:54 UTC [B-worker codex-park] release cards 1
2026-11-20 10:39:54 UTC [B-worker codex-trace] claim cards 1 ~15 min: async-full-acceptance
2026-11-20 10:39:54 UTC [B-worker codex-park] claim cards 1 ~12 min: 27B fast-draft-int8-same-k1
2026-11-20 10:40:41 UTC [B-worker codex-trace] release cards 1
2026-11-20 10:40:55 UTC [B-worker codex-park] release cards 1
2026-11-20 10:40:56 UTC [B-worker codex-park] claim cards 1 ~12 min: 27B fast-draft-int8-other-k1
2026-11-20 10:41:25 UTC [B-worker codex-trace] claim cards 1 ~15 min: async-markers-acceptance
2026-11-20 10:42:12 UTC [B-worker codex-trace] release cards 1
2026-11-20 10:42:27 UTC [B-worker codex-park] release cards 1
2026-11-20 10:42:27 UTC [B-worker codex-park] claim cards 1 ~12 min: 27B fast-interference-int8
2026-11-20 10:43:05 UTC [A-P] claim cards 0+1 ~3 min: preflight xpu-health
2026-11-20 10:43:39 UTC [A-P] released cards 0+1
2026-11-20 10:44:13 UTC [B-worker codex-park] release cards 1
2026-11-20 10:44:16 UTC [A-P] claim cards 0+1 ~10 min: Level Zero PLE residency probe
2026-11-20 10:44:28 UTC [A-P] released cards 0+1
2026-11-20 10:45:46 UTC [B-worker codex-park] claim cards 1 ~12 min: 27B suite-base-kv1
2026-11-20 10:46:05 UTC [A-KF] claim card 0 ~5 min: qualified-row-reuse
2026-11-20 10:46:19 UTC [B-worker codex-park] release cards 1
2026-11-20 10:46:19 UTC [B-worker codex-park] claim cards 1 ~12 min: 27B suite-kv1-same-k1
2026-11-20 10:46:42 UTC [B-worker codex-park] claim cards 1 ~12 min: 27B batch-base-int8
2026-11-20 10:46:57 UTC [B-worker codex-park] claim cards 1 ~12 min: 27B suite-draft-base-int8
2026-11-20 10:47:17 UTC [B-worker codex-park] release cards 1 (queued batch and draft cases cancelled before acquisition)
2026-11-20 10:47:34 UTC [A-KF] released card 0
2026-11-20 10:47:45 UTC [B-worker codex-park] release cards 1
2026-11-20 10:47:45 UTC [B-worker codex-park] claim cards 1 ~12 min: 27B suite-kv1-same-k7
2026-11-20 10:48:32 UTC [B-worker codex-park] release cards 1
2026-11-20 10:48:32 UTC [B-worker codex-park] claim cards 1 ~12 min: 27B suite-kv1-same-k64
2026-11-20 10:48:32 UTC [A-P] claim cards 0+1 ~10 min: p6-check.log
2026-11-20 10:49:22 UTC [B-worker codex-park] release cards 1
2026-11-20 10:49:22 UTC [B-worker codex-park] claim cards 1 ~12 min: 27B suite-kv1-other-k1
2026-11-20 10:49:24 UTC [A-P] released cards 0+1
2026-11-20 10:49:29 UTC [B-worker codex-pushar] claim cards 0,1 ~4 min: replay pre-health
2026-11-20 10:49:36 UTC [B-worker codex-park] release cards 1
2026-11-20 10:49:36 UTC [B-worker codex-park] claim cards 1 ~12 min: 27B suite-kv1-other-k7
2026-11-20 10:49:46 UTC [B-worker codex-park] release cards 1
2026-11-20 10:49:46 UTC [B-worker codex-park] claim cards 1 ~12 min: 27B suite-kv1-other-k64
2026-11-20 10:49:57 UTC [B-worker codex-park] release cards 1
2026-11-20 10:49:57 UTC [B-worker codex-park] claim cards 1 ~12 min: 27B suite-base-kv0
2026-11-20 10:50:39 UTC [A-P] claim cards 0+1 ~10 min: p6-check.log
2026-11-20 10:50:45 UTC [B-worker codex-park] release cards 1
2026-11-20 10:50:46 UTC [B-worker codex-park] claim cards 1 ~12 min: 27B suite-kv0-same-k1
2026-11-20 10:52:16 UTC [A-I] claim cards 0+1 ~3 min: M-A5 transport health-a5t-pre.log
2026-11-20 10:55:00 UTC [A-P] released cards 0+1
2026-11-20 10:55:11 UTC [B-worker codex-park] release cards 1
2026-11-20 10:55:11 UTC [B-worker codex-park] claim cards 1 ~12 min: 27B suite-kv0-same-k7
2026-11-20 10:55:14 UTC [A-P] claim cards 0+1 ~10 min: solo-0-device-check.log
2026-11-20 10:56:31 UTC [B-worker codex-pushar] release cards 0,1
2026-11-20 10:56:37 UTC [B-worker codex-pushar] claim cards 0,1 ~9 min: replay counter-non-immediate 10240 bytes 96 hops depth 1
2026-11-20 10:57:00 UTC [A-I] released cards 0+1
2026-11-20 10:58:52 UTC [B-worker codex-kv8] claim cards 1 ~3 min: pre-test health
2026-11-20 10:59:07 UTC [B-worker codex-kv8] release cards 1
2026-11-20 11:02:33 UTC [B-worker codex-park] release cards 1
2026-11-20 11:02:34 UTC [B-worker codex-park] claim cards 1 ~12 min: 27B suite-kv0-same-k64
2026-11-20 11:03:42 UTC [A-KF] claim card 0 ~5 min: qualified-packed-fields
2026-11-20 11:05:24 UTC [B-worker codex-pushar] release cards 0,1 (queued attempt cancelled before acquisition)
2026-11-20 11:07:36 UTC [A-P] released cards 0+1
2026-11-20 11:07:36 UTC [A-P] claim cards 0+1 ~10 min: solo-1-device-check.log
2026-11-20 11:08:50 UTC [A-I] claim cards 0+1 ~10 min: M-A5 transport a5t-teacher.log
2026-11-20 11:09:37 UTC [B-worker codex-park] release cards 1
2026-11-20 11:09:37 UTC [B-worker codex-park] claim cards 1 ~12 min: 27B suite-kv0-other-k1
2026-11-20 11:10:28 UTC [A-KF] released card 0
2026-11-20 11:12:05 UTC [B-worker codex-pushar] claim cards 0,1 ~9 min: replay counter-non-immediate 10240 bytes 96 hops depth 1
2026-11-20 11:14:05 UTC [A-P] released cards 0+1
2026-11-20 11:14:05 UTC [A-P] claim cards 0+1 ~10 min: solo-2-device-check.log
2026-11-20 11:14:41 UTC [B-worker codex-kv8] claim cards 1 ~3 min: paged KV kernel parity and chunk seams
2026-11-20 11:14:42 UTC [B-worker codex-kv8] release cards 1
2026-11-20 11:15:34 UTC [B-worker codex-kv8] claim cards 0+1 ~13 min: pre-health and bf16page serial versus MTP smoke
2026-11-20 11:16:25 UTC [B-worker codex-kv8] release cards 0+1
2026-11-20 11:17:40 UTC [B-worker codex-kv8] claim cards 0+1 ~12 min: int8 p0 256-token serial versus MTP
2026-11-20 11:19:18 UTC [B-worker codex-kv8] release cards 0+1
2026-11-20 11:19:18 UTC [A-I] released cards 0+1
2026-11-20 11:19:30 UTC [B-worker codex-park] release cards 1
2026-11-20 11:19:30 UTC [B-worker codex-park] claim cards 1 ~12 min: 27B suite-kv0-other-k7
2026-11-20 11:19:49 UTC [B-worker codex-kv8] claim cards 0+1 ~12 min: bf16page p0 256-token serial versus MTP
2026-11-20 11:21:22 UTC [B-worker codex-kv8] release cards 0+1
2026-11-20 11:21:43 UTC [B-worker codex-kv8] claim cards 0+1 ~12 min: int8 a4-p1 M-A4 correctness
2026-11-20 11:23:11 UTC [B-worker codex-kv8] release cards 0+1
2026-11-20 11:23:11 UTC [B-worker codex-kv8] claim cards 0+1 ~12 min: int8 a4-p2 M-A4 correctness
2026-11-20 11:24:39 UTC [B-worker codex-kv8] release cards 0+1
2026-11-20 11:24:39 UTC [B-worker codex-kv8] claim cards 0+1 ~12 min: int8 a4-p3 M-A4 correctness
2026-11-20 11:27:03 UTC [B-worker codex-kv8] release cards 0+1
2026-11-20 11:27:07 UTC [B-worker codex-kv8] claim cards 0+1 ~12 min: int8 a4-p4 M-A4 correctness
2026-11-20 11:27:12 UTC [B-worker codex-kv8] release cards 0+1
2026-11-20 11:27:16 UTC [B-worker codex-pushar] release cards 0,1
2026-11-20 11:27:34 UTC [B-worker codex-pushar] claim cards 0,1 ~9 min: replay counter-immediate 10240 bytes 96 hops depth 1
2026-11-20 11:28:11 UTC [B-worker codex-pushar] release cards 0,1 (queued attempt cancelled before acquisition)
2026-11-20 11:29:12 UTC [B-worker codex-kv8] claim cards 0+1 ~12 min: int8 a4-p4 M-A4 correctness
2026-11-20 11:29:17 UTC [B-worker codex-kv8] release cards 0+1
2026-11-20 11:29:30 UTC [B-worker codex-park] release cards 1
2026-11-20 11:30:28 UTC [A-P] released cards 0+1
2026-11-20 11:30:28 UTC [A-P] claim cards 0+1 ~10 min: solo-3-device-check.log
2026-11-20 11:30:30 UTC [B-worker codex-park] claim cards 1 ~12 min: 27B suite-kv0-other-k7
2026-11-20 11:30:57 UTC [A-KF] claim card 0 ~5 min: qualified-q2-vector
2026-11-20 11:31:17 UTC [B-worker codex-kv8] claim cards 0+1 ~12 min: int8 a4-p4 M-A4 correctness
2026-11-20 11:32:45 UTC [B-worker codex-kv8] release cards 0+1
2026-11-20 11:32:55 UTC [B-worker codex-kv8] claim cards 0+1 ~12 min: int8 a4-p5 M-A4 correctness
2026-11-20 11:34:54 UTC [B-worker codex-kv8] release cards 0+1
2026-11-20 11:35:08 UTC [B-worker codex-kv8] claim cards 0+1 ~25 min: local Strata extra oracle positions, no network
2026-11-20 11:35:52 UTC [B-worker codex-kv8] release cards 0+1
2026-11-20 11:35:56 UTC [A-I] claim cards 0+1 ~10 min: M-A5 transport a5t-teacher.log
2026-11-20 11:36:13 UTC [B-worker codex-kv8] claim cards 0+1 ~3 min: health after stopping local oracle
2026-11-20 11:36:43 UTC [B-worker codex-kv8] release cards 0+1
2026-11-20 11:37:13 UTC [B-worker codex-kv8] claim cards 0+1 ~25 min: local Strata extra oracle with event workaround, no network
2026-11-20 11:38:00 UTC [B-worker codex-kv8] release cards 0+1
2026-11-20 11:38:36 UTC [B-worker codex-pushar] claim cards 0,1 ~9 min: replay counter-immediate 10240 bytes 96 hops depth 1
2026-11-20 11:40:30 UTC [B-worker codex-park] release cards 1
2026-11-20 11:40:36 UTC [A-P] released cards 0+1
2026-11-20 11:40:36 UTC [A-P] claim cards 0+1 ~10 min: solo-4-device-check.log
2026-11-20 11:41:02 UTC [A-P] claim cards 0+1 ~10 min: C4-T2-device-check.log
2026-11-20 11:41:26 UTC [B-worker codex-kv8] claim cards 1 ~3 min: descriptor window and CPU kernel parity
2026-11-20 11:41:26 UTC [B-worker codex-kv8] release cards 1
2026-11-20 11:41:28 UTC [A-KF] released card 0
2026-11-20 11:41:30 UTC [B-worker codex-park] claim cards 1 ~12 min: 27B suite-kv0-other-k7
2026-11-20 11:41:51 UTC [B-worker codex-kv8] claim cards 0+1 ~22 min: legacy model chunk invariance and 512-token seam
2026-11-20 11:42:59 UTC [A-P] claim cards 0+1 ~10 min: final-solo-0-check.log
2026-11-20 11:44:25 UTC [B-worker codex-kv8] release cards 0+1
2026-11-20 11:44:35 UTC [B-worker codex-kv8] claim cards 0+1 ~22 min: bf16page model chunk invariance and 512-token seam
2026-11-20 11:44:40 UTC [B-worker codex-kv8] release cards 0+1
2026-11-20 11:46:40 UTC [B-worker codex-kv8] claim cards 0+1 ~22 min: bf16page model chunk invariance and 512-token seam
2026-11-20 11:46:45 UTC [B-worker codex-kv8] release cards 0+1
2026-11-20 11:47:21 UTC [B-worker codex-kv8] claim cards 0+1 ~22 min: bf16page model chunk invariance and 512-token seam
2026-11-20 11:47:26 UTC [B-worker codex-kv8] release cards 0+1
2026-11-20 11:47:45 UTC [A-I] released cards 0+1
2026-11-20 11:47:45 UTC [B-worker codex-trace2] claim cards 0,1 ~20 min: ping
2026-11-20 11:48:36 UTC [B-worker codex-pushar] release cards 0,1
2026-11-20 11:48:42 UTC [B-worker codex-pushar] claim cards 0,1 ~9 min: replay counter-immediate 10240 bytes 96 hops depth 1
2026-11-20 11:48:43 UTC [B-worker codex-trace2] release cards 0,1
2026-11-20 11:49:26 UTC [B-worker codex-kv8] claim cards 0+1 ~22 min: bf16page model chunk invariance and 512-token seam
2026-11-20 11:49:31 UTC [B-worker codex-kv8] release cards 0+1
2026-11-20 11:51:19 UTC [A-P] released cards 0+1
2026-11-20 11:51:19 UTC [A-P] claim cards 0+1 ~10 min: solo-5-device-check.log
2026-11-20 11:51:30 UTC [B-worker codex-park] release cards 1
2026-11-20 11:51:31 UTC [B-worker codex-kv8] claim cards 0+1 ~22 min: bf16page model chunk invariance and 512-token seam
2026-11-20 11:51:36 UTC [B-worker codex-kv8] release cards 0+1
2026-11-20 11:52:30 UTC [B-worker codex-park] claim cards 1 ~12 min: 27B suite-kv0-other-k7
2026-11-20 11:53:32 UTC [A-P] released cards 0+1
2026-11-20 11:53:36 UTC [B-worker codex-kv8] claim cards 0+1 ~22 min: bf16page model chunk invariance and 512-token seam
2026-11-20 11:56:10 UTC [B-worker codex-kv8] release cards 0+1
2026-11-20 11:56:20 UTC [B-worker codex-kv8] claim cards 0+1 ~22 min: int8 model chunk invariance and 512-token seam
2026-11-20 11:56:25 UTC [B-worker codex-kv8] release cards 0+1
2026-11-20 11:58:25 UTC [B-worker codex-kv8] claim cards 0+1 ~22 min: int8 model chunk invariance and 512-token seam
2026-11-20 11:58:30 UTC [B-worker codex-kv8] release cards 0+1
2026-11-20 11:58:42 UTC [B-worker codex-pushar] release cards 0,1
2026-11-20 11:58:44 UTC [A-P] released cards 0+1
2026-11-20 11:58:44 UTC [A-P] claim cards 0+1 ~10 min: final-solo-1-check.log
2026-11-20 11:58:48 UTC [B-worker codex-pushar] claim cards 0,1 ~9 min: replay counter-immediate 10240 bytes 96 hops depth 1
2026-11-20 11:58:53 UTC [B-worker codex-kv8] claim cards 1 ~3 min: descriptor window and CPU kernel parity
2026-11-20 11:58:53 UTC [B-worker codex-kv8] release cards 1
2026-11-20 12:00:30 UTC [B-worker codex-kv8] claim cards 0+1 ~22 min: int8 model chunk invariance and 512-token seam
2026-11-20 12:00:35 UTC [B-worker codex-kv8] release cards 0+1
2026-11-20 12:02:25 UTC [A-P] released cards 0+1
2026-11-20 12:02:25 UTC [A-P] claim cards 0+1 ~10 min: solo-7-device-check.log
2026-11-20 12:02:34 UTC [B-worker codex-park] release cards 1
2026-11-20 12:02:34 UTC [B-worker codex-park] claim cards 1 ~12 min: 27B suite-kv0-other-k64
2026-11-20 12:02:35 UTC [B-worker codex-kv8] claim cards 0+1 ~22 min: int8 model chunk invariance and 512-token seam
2026-11-20 12:02:54 UTC [A-P] claim cards 0+1 ~10 min: perf-C1-0-first.log
2026-11-20 12:03:15 UTC [A-I] claim cards 0+1 ~10 min: M-A5 transport a5t-group0.log
2026-11-20 12:04:49 UTC [A-KF] claim card 0 ~5 min: health-before
2026-11-20 12:05:10 UTC [B-worker codex-kv8] release cards 0+1
2026-11-20 12:07:42 UTC [A-P] released cards 0+1
2026-11-20 12:07:42 UTC [A-P] claim cards 0+1 ~10 min: final-solo-6-check.log
2026-11-20 12:08:39 UTC [B-worker codex-pushar] release cards 0,1
2026-11-20 12:09:08 UTC [B-worker codex-pushar] claim cards 0,1 ~9 min: replay device-reset 10240 bytes 96 hops depth 1
2026-11-20 12:12:29 UTC [A-P] released cards 0+1
2026-11-20 12:12:40 UTC [B-worker codex-park] release cards 1
2026-11-20 12:13:35 UTC [B-worker codex-park] claim cards 1 ~12 min: 27B batch-base-int8
2026-11-20 12:14:11 UTC [A-P] released cards 0+1
2026-11-20 12:14:11 UTC [A-P] claim cards 0+1 ~10 min: perf-C4-T2-0-first.log
2026-11-20 12:19:08 UTC [B-worker codex-pushar] release cards 0,1
2026-11-20 12:19:25 UTC [A-I] released cards 0+1
2026-11-20 12:19:25 UTC [A-I] claim cards 0+1 ~10 min: M-A5 transport a5t-group1.log
2026-11-20 12:19:26 UTC [B-worker codex-pushar] claim cards 0,1 ~9 min: replay device-reset 10240 bytes 96 hops depth 1
2026-11-20 12:19:38 UTC [A-KF] released card 0
2026-11-20 12:20:07 UTC [B-worker codex-kv8] claim cards 0+1 ~12 min: int8 a4-p0 M-A4 correctness
2026-11-20 12:20:34 UTC [B-worker codex-kv8] release cards 0+1
2026-11-20 12:22:01 UTC [A-P] released cards 0+1
2026-11-20 12:22:01 UTC [B-worker codex-trace2] claim cards 0,1 ~20 min: host-c1-markers-pre
2026-11-20 12:23:13 UTC [A-KF] claim card 0 ~5 min: baseline-dense
2026-11-20 12:23:13 UTC [A-KF] claim card 0 ~5 min: baseline-timing
2026-11-20 12:23:13 UTC [A-KF] claim card 0 ~5 min: baseline-quant
2026-11-20 12:23:13 UTC [A-KF] claim card 0 ~5 min: baseline-experts
2026-11-20 12:23:25 UTC [B-worker codex-trace2] release cards 0,1
2026-11-20 12:23:34 UTC [B-worker codex-park] release cards 1
2026-11-20 12:23:34 UTC [B-worker codex-park] claim cards 1 ~12 min: 27B batch-park-int8-other-k1
2026-11-20 12:24:41 UTC [A-P] released cards 0+1
2026-11-20 12:24:41 UTC [A-P] claim cards 0+1 ~10 min: perf-C4-T4-0-first.log
2026-11-20 12:29:26 UTC [B-worker codex-pushar] release cards 0,1
2026-11-20 12:29:34 UTC [B-worker codex-pushar] claim cards 0,1 ~9 min: replay device-reset 10240 bytes 96 hops depth 1
2026-11-20 12:29:39 UTC [A-I] released cards 0+1
2026-11-20 12:29:39 UTC [A-I] claim cards 0+1 ~10 min: M-A5 transport a5t-C4-T2.log
2026-11-20 12:30:48 UTC [A-P] claim cards 0+1 ~10 min: default-p6-check.log
2026-11-20 12:31:01 UTC [A-KF] claim card 0 ~5 min: baseline-shapes
2026-11-20 12:31:06 UTC [B-worker codex-kv8] claim cards 1 ~3 min: descriptor window and CPU kernel parity
2026-11-20 12:31:06 UTC [B-worker codex-kv8] release cards 1
2026-11-20 12:31:28 UTC [A-KF] released card 0
2026-11-20 12:32:02 UTC [A-KF] released card 0
2026-11-20 12:32:16 UTC [A-KF] released card 0
2026-11-20 12:32:42 UTC [A-KF] released card 0
2026-11-20 12:32:42 UTC [B-worker codex-trace2] claim cards 0,1 ~20 min: host-c1-full
2026-11-20 12:33:34 UTC [B-worker codex-park] release cards 1
2026-11-20 12:34:09 UTC [B-worker codex-trace2] release cards 0,1
2026-11-20 12:34:34 UTC [B-worker codex-park] claim cards 1 ~12 min: 27B batch-park-int8-other-k1
2026-11-20 12:34:49 UTC [A-P] claim cards 0+1 ~10 min: fallback-p6-check.log
2026-11-20 12:35:15 UTC [A-P] released cards 0+1
2026-11-20 12:35:15 UTC [A-P] claim cards 0+1 ~10 min: perf-C1-0-second.log
2026-11-20 12:35:48 UTC [A-KF] claim card 0 ~5 min: final-gates-timing
2026-11-20 12:37:07 UTC [B-worker codex-pushar] release cards 0,1
2026-11-20 12:37:15 UTC [B-worker codex-pushar] claim cards 0,1 ~9 min: replay device-reset 163840 bytes 96 hops depth 1
2026-11-20 12:38:12 UTC [A-I] released cards 0+1
2026-11-20 12:38:12 UTC [A-I] claim cards 0+1 ~10 min: M-A5 transport a5t-C4qsa-T4.log
2026-11-20 12:39:25 UTC [A-P] released cards 0+1
2026-11-20 12:39:43 UTC [A-KF] released card 0
2026-11-20 12:39:43 UTC [B-worker codex-trace2] claim cards 0,1 ~20 min: host-c1-markers-post
2026-11-20 12:41:08 UTC [B-worker codex-trace2] release cards 0,1
2026-11-20 12:41:20 UTC [B-worker codex-park] release cards 1
2026-11-20 12:41:20 UTC [B-worker codex-park] claim cards 1 ~12 min: 27B batch-draft-base-int8
2026-11-20 12:42:45 UTC [B-worker codex-kv8] claim cards 0+1 ~12 min: int8 a4-p1 M-A4 correctness
2026-11-20 12:43:11 UTC [B-worker codex-kv8] release cards 0+1
2026-11-20 12:43:22 UTC [B-worker codex-kv8] claim cards 0+1 ~12 min: bf16page a4-p0 M-A4 correctness
2026-11-20 12:43:50 UTC [B-worker codex-kv8] release cards 0+1
2026-11-20 12:44:08 UTC [A-P] released cards 0+1
2026-11-20 12:45:39 UTC [A-P] released cards 0+1
2026-11-20 12:45:39 UTC [A-P] claim cards 0+1 ~10 min: perf-C4-T2-0-second.log
2026-11-20 12:47:15 UTC [B-worker codex-pushar] release cards 0,1
2026-11-20 12:47:22 UTC [A-KF] released card 0
2026-11-20 12:47:38 UTC [B-worker codex-pushar] claim cards 0,1 ~9 min: replay device-reset 163840 bytes 96 hops depth 1
2026-11-20 12:50:05 UTC [A-I] released cards 0+1
2026-11-20 12:50:05 UTC [B-worker codex-trace2] claim cards 0,1 ~20 min: host-c4-markers-pre
2026-11-20 12:50:05 UTC [A-I] claim cards 0+1 ~10 min: M-A5 transport a5t-C4stagger-T2.log
2026-11-20 12:51:20 UTC [B-worker codex-park] release cards 1
2026-11-20 12:51:28 UTC [B-worker codex-trace2] release cards 0,1
2026-11-20 12:52:20 UTC [B-worker codex-park] claim cards 1 ~12 min: 27B batch-draft-base-int8
2026-11-20 12:53:04 UTC [A-P] released cards 0+1
2026-11-20 12:53:04 UTC [A-P] claim cards 0+1 ~10 min: perf-C4-T4-0-second.log
2026-11-20 12:53:14 UTC [A-KF] claim card 0 ~5 min: baseline-tight
2026-11-20 12:54:32 UTC [B-worker codex-pushar] release cards 0,1
2026-11-20 12:54:35 UTC [B-worker codex-pushar] claim cards 0,1 ~9 min: replay device-reset 163840 bytes 192 hops depth 1
2026-11-20 12:55:37 UTC [A-I] released cards 0+1
2026-11-20 12:55:37 UTC [B-worker codex-trace2] claim cards 0,1 ~20 min: host-c4-full
2026-11-20 12:55:37 UTC [A-I] claim cards 0+1 ~10 min: M-A5 transport a5t-C4soak-T4.log
2026-11-20 12:57:04 UTC [B-worker codex-trace2] release cards 0,1
2026-11-20 12:57:18 UTC [B-worker codex-park] release cards 1
2026-11-20 12:57:19 UTC [B-worker codex-park] claim cards 1 ~12 min: 27B batch-draft-park-int8-other-k1
2026-11-20 12:58:24 UTC [A-P] released cards 0+1
2026-11-20 12:58:24 UTC [A-P] claim cards 0+1 ~10 min: perf-C1-1-first.log
2026-11-20 12:59:14 UTC [A-KF] released card 0
2026-11-20 13:01:11 UTC [B-worker codex-pushar] release cards 0,1
2026-11-20 13:01:17 UTC [B-worker codex-pushar] claim cards 0,1 ~9 min: replay device-reset 163840 bytes 192 hops depth 2
2026-11-20 13:03:51 UTC [A-I] released cards 0+1
2026-11-20 13:03:51 UTC [B-worker codex-trace2] claim cards 0,1 ~20 min: host-c4-markers-post
2026-11-20 13:03:51 UTC [A-I] claim cards 0+1 ~10 min: M-A5 transport a5t-fallback.log
2026-11-20 13:05:16 UTC [B-worker codex-trace2] release cards 0,1
2026-11-20 13:05:31 UTC [B-worker codex-park] release cards 1
2026-11-20 13:05:31 UTC [B-worker codex-park] claim cards 1 ~12 min: 27B batch-interference-int8
2026-11-20 13:06:00 UTC [B-worker codex-kv8] claim cards 0+1 ~12 min: bf16page a4-p1 M-A4 correctness
2026-11-20 13:06:26 UTC [B-worker codex-kv8] release cards 0+1
2026-11-20 13:06:36 UTC [B-worker codex-kv8] claim cards 0+1 ~12 min: int8 a4-C2-T2 M-A4 correctness
2026-11-20 13:06:59 UTC [B-worker codex-kv8] release cards 0+1
2026-11-20 13:07:09 UTC [B-worker codex-kv8] claim cards 0+1 ~12 min: int8 a4-C2-T4 M-A4 correctness
2026-11-20 13:07:31 UTC [B-worker codex-kv8] release cards 0+1
2026-11-20 13:09:56 UTC [A-P] released cards 0+1
2026-11-20 13:09:56 UTC [A-P] claim cards 0+1 ~10 min: perf-C4-T2-1-first.log
2026-11-20 13:12:16 UTC [B-worker codex-pushar] release cards 0,1
2026-11-20 13:12:36 UTC [A-I] released cards 0+1
2026-11-20 13:12:36 UTC [A-KF] claim card 0 ~5 min: final-tight-gates-timing
2026-11-20 13:12:36 UTC [A-I] claim cards 0+1 ~3 min: M-A5 transport health-a5t-post.log
2026-11-20 13:12:39 UTC [B-worker codex-pushar] claim cards 0,1 ~9 min: replay device-reset 10240 bytes 192 hops depth 1
2026-11-20 13:13:09 UTC [B-worker codex-park] release cards 1
2026-11-20 13:13:45 UTC [B-worker codex-park] claim cards 1 ~12 min: 27B batch4-interference-int8
2026-11-20 13:15:27 UTC [A-P] released cards 0+1
2026-11-20 13:15:27 UTC [A-P] claim cards 0+1 ~10 min: perf-C4-T4-1-first.log
2026-11-20 13:17:10 UTC [A-KF] released card 0
2026-11-20 13:17:38 UTC [A-I] released cards 0+1
2026-11-20 13:19:03 UTC [B-worker codex-pushar] release cards 0,1
2026-11-20 13:19:36 UTC [B-worker codex-park] release cards 1
2026-11-20 13:21:07 UTC [B-worker codex-park] claim cards 1 ~12 min: 27B batch2-interference-int8
2026-11-20 13:21:40 UTC [A-P] released cards 0+1
2026-11-20 13:21:40 UTC [A-P] claim cards 0+1 ~10 min: perf-C1-1-second.log
2026-11-20 13:21:41 UTC [B-worker codex-kv8] claim cards 0+1 ~12 min: bf16page a4-C2-T2 M-A4 correctness
2026-11-20 13:22:04 UTC [B-worker codex-kv8] release cards 0+1
2026-11-20 13:22:14 UTC [B-worker codex-kv8] claim cards 0+1 ~12 min: bf16page a4-C2-T4 M-A4 correctness
2026-11-20 13:22:37 UTC [B-worker codex-kv8] release cards 0+1
2026-11-20 13:23:16 UTC [B-worker codex-kv8] claim cards 0+1 ~12 min: int8 a4-p0 M-A4 correctness
2026-11-20 13:24:54 UTC [B-worker codex-kv8] release cards 0+1
2026-11-20 13:25:04 UTC [B-worker codex-kv8] claim cards 0+1 ~12 min: int8 a4-p1 M-A4 correctness
2026-11-20 13:26:33 UTC [B-worker codex-kv8] release cards 0+1
2026-11-20 13:27:28 UTC [B-worker codex-park] release cards 1
2026-11-20 13:28:16 UTC [B-worker codex-park] claim cards 1 ~12 min: 27B suite-draft-base-int8
2026-11-20 13:30:04 UTC [A-P] released cards 0+1
2026-11-20 13:30:04 UTC [A-P] claim cards 0+1 ~10 min: perf-C4-T2-1-second.log
2026-11-20 13:30:43 UTC [B-worker codex-kv8] claim cards 0+1 ~12 min: int8 a4-p2 M-A4 correctness
2026-11-20 13:32:12 UTC [B-worker codex-kv8] release cards 0+1
2026-11-20 13:32:44 UTC [B-worker codex-park] release cards 1
2026-11-20 13:32:44 UTC [B-worker codex-park] claim cards 1 ~12 min: 27B suite-draft-int8-same-k1
2026-11-20 13:34:53 UTC [A-P] released cards 0+1
2026-11-20 13:34:53 UTC [B-worker codex-trace2] claim cards 0,1 ~28 min: prefill pilot-full
2026-11-20 13:34:53 UTC [A-P] claim cards 0+1 ~10 min: perf-C4-T4-1-second.log
2026-11-20 13:36:15 UTC [B-worker codex-trace2] release cards 0,1
2026-11-20 13:36:30 UTC [B-worker codex-park] release cards 1
2026-11-20 13:36:30 UTC [B-worker codex-park] claim cards 1 ~12 min: 27B suite-draft-int8-same-k7
2026-11-20 13:38:35 UTC [A-P] released cards 0+1
2026-11-20 13:38:52 UTC [B-worker codex-park] release cards 1
2026-11-20 13:38:52 UTC [B-worker codex-park] claim cards 1 ~12 min: 27B suite-draft-int8-same-k64
2026-11-20 13:38:53 UTC [A-P] claim cards 0+1 ~10 min: health-post.log
2026-11-20 13:39:37 UTC [B-worker codex-park] release cards 1
2026-11-20 13:39:37 UTC [B-worker codex-park] claim cards 1 ~12 min: 27B suite-draft-int8-other-k1
2026-11-20 13:40:06 UTC [A-P] released cards 0+1
2026-11-20 13:40:21 UTC [B-worker codex-park] release cards 1
2026-11-20 13:40:21 UTC [B-worker codex-park] claim cards 1 ~12 min: 27B suite-draft-int8-other-k7
2026-11-20 13:40:37 UTC [B-worker codex-park] release cards 1
2026-11-20 13:40:37 UTC [B-worker codex-park] claim cards 1 ~12 min: 27B suite-draft-int8-other-k64
2026-11-20 13:41:11 UTC [B-worker codex-park] release cards 1
2026-11-20 13:41:21 UTC [B-worker codex-trace2] claim cards 0,1 ~28 min: prefill single-256-markers
2026-11-20 13:42:05 UTC [B-worker codex-park] claim cards 1 ~3 min: final post-test health
2026-11-20 13:48:49 UTC [B-worker codex-trace2] release cards 0,1
2026-11-20 13:49:04 UTC [B-worker codex-park] release cards 1
2026-11-20 13:50:03 UTC [B-worker codex-trace2] claim cards 0,1 ~28 min: prefill single-512-markers
2026-11-20 13:57:22 UTC [B-worker codex-trace2] release cards 0,1
2026-11-20 13:58:22 UTC [B-worker codex-kv8] claim cards 0+1 ~12 min: int8 a4-p3 M-A4 correctness
2026-11-20 14:00:00 UTC [B-worker codex-kv8] release cards 0+1
2026-11-20 14:00:10 UTC [B-worker codex-kv8] claim cards 0+1 ~12 min: int8 a4-p4 M-A4 correctness
2026-11-20 14:01:37 UTC [B-worker codex-kv8] release cards 0+1
2026-11-20 14:01:47 UTC [B-worker codex-kv8] claim cards 0+1 ~12 min: int8 a4-p5 M-A4 correctness
2026-11-20 14:03:38 UTC [B-worker codex-kv8] release cards 0+1
2026-11-20 14:03:48 UTC [B-worker codex-kv8] claim cards 0+1 ~12 min: int8 a4-p6 M-A4 correctness
2026-11-20 14:04:07 UTC [B-worker codex-kv8] release cards 0+1
2026-11-20 14:04:17 UTC [B-worker codex-kv8] claim cards 0+1 ~12 min: int8 a4-p7 M-A4 correctness
2026-11-20 14:05:21 UTC [A-I] claim cards 0+1 ~3 min: M-A5 fast kernels health-a5k-pre.log
2026-11-20 14:06:29 UTC [B-worker codex-kv8] release cards 0+1
2026-11-20 14:07:00 UTC [A-I] released cards 0+1
2026-11-20 14:07:04 UTC [A-I] claim cards 0+1 ~10 min: M-A5 fast kernels a5k-teacher-q8_1-1.log
2026-11-20 14:09:08 UTC [A-I] released cards 0+1
2026-11-20 14:09:08 UTC [A-I] claim cards 0+1 ~10 min: M-A5 fast kernels a5k-teacher-q8_1-2.log
2026-11-20 14:12:13 UTC [A-I] released cards 0+1
2026-11-20 14:12:13 UTC [B-worker codex-trace2] claim cards 0,1 ~28 min: prefill single-1024-markers
2026-11-20 14:13:14 UTC [A-KF] claim card 0 ~5 min: health-before
2026-11-20 14:13:45 UTC [B-worker codex-trace2] release cards 0,1
2026-11-20 14:14:39 UTC [B-worker codex-kv8] claim cards 0+1 ~12 min: bf16page a4-p0 M-A4 correctness
2026-11-20 14:15:21 UTC [A-R] claim cards 0+1 ~3 min: health-r0-pre.log
2026-11-20 14:15:42 UTC [A-I] claim cards 0+1 ~10 min: M-A5 fast kernels a5k-teacher-bf16-1.log
2026-11-20 14:16:12 UTC [B-worker codex-kv8] release cards 0+1
2026-11-20 14:16:26 UTC [A-KF] released card 0
2026-11-20 14:16:29 UTC [B-worker codex-pushar] claim cards 1 ~8 min: PLE host probe imports
2026-11-20 14:16:51 UTC [B-worker codex-park] claim cards 0+1 ~3 min: pre-test health
2026-11-20 14:17:22 UTC [B-worker codex-park] release cards 0+1
2026-11-20 14:18:22 UTC [B-worker codex-kv8] claim cards 0+1 ~12 min: bf16page a4-p1 M-A4 correctness
2026-11-20 14:19:50 UTC [B-worker codex-kv8] release cards 0+1
2026-11-20 14:20:00 UTC [B-worker codex-kv8] claim cards 0+1 ~12 min: bf16page a4-p2 M-A4 correctness
2026-11-20 14:21:28 UTC [B-worker codex-kv8] release cards 0+1
2026-11-20 14:21:58 UTC [A-R] released cards 0+1 rc=0
2026-11-20 14:21:58 UTC [A-R] claim cards 0+1 ~10 min: r0-p0.log
2026-11-20 14:23:37 UTC [A-R] claim cards 0+1 ~10 min: r0-bf16-p0.log
2026-11-20 14:25:30 UTC [A-I] released cards 0+1
2026-11-20 14:25:30 UTC [A-I] claim cards 0+1 ~10 min: M-A5 fast kernels a5k-teacher-bf16-2.log
2026-11-20 14:25:58 UTC [B-worker codex-pushar] release cards 1
2026-11-20 14:26:38 UTC [B-worker codex-pushar] claim cards 1 ~8 min: PLE host probe container
2026-11-20 14:27:51 UTC [A-KF] claim card 0 ~5 min: strata-reference
2026-11-20 14:28:41 UTC [A-R] released cards 0+1 rc=0
2026-11-20 14:30:51 UTC [A-R] released cards 0+1 rc=0
2026-11-20 14:32:28 UTC [A-R] claim cards 0+1 ~10 min: r0-ref-p0.log
2026-11-20 14:34:29 UTC [A-I] released cards 0+1
2026-11-20 14:34:36 UTC [B-worker codex-park] claim cards 0+1 ~12 min: control-int8
2026-11-20 14:34:54 UTC [B-worker codex-trace2] claim cards 0,1 ~28 min: prefill single-1024-markers-retry
2026-11-20 14:35:09 UTC [A-KF] claim card 0 ~5 min: baseline
2026-11-20 14:36:38 UTC [B-worker codex-pushar] release cards 1
2026-11-20 14:36:54 UTC [B-worker codex-pushar] claim cards 1 ~8 min: PLE host probe container
2026-11-20 14:42:11 UTC [B-worker codex-trace2] release cards 0,1
2026-11-20 14:44:05 UTC [A-KF] released card 0
2026-11-20 14:44:36 UTC [B-worker codex-park] release cards 0+1
2026-11-20 14:46:38 UTC [A-R] released cards 0+1 rc=0
2026-11-20 14:46:38 UTC [A-R] claim cards 0+1 ~10 min: r0-ref-C4-T2.log
2026-11-20 14:46:38 UTC [A-I] claim cards 0+1 ~10 min: M-A5 fast kernels a5k-group0.log
2026-11-20 14:46:54 UTC [B-worker codex-pushar] release cards 1
2026-11-20 14:47:11 UTC [B-worker codex-pushar] claim cards 1 ~8 min: PLE host probe container
2026-11-20 14:48:35 UTC [A-KF] released card 0
2026-11-20 14:48:36 UTC [B-worker codex-park] claim cards 0+1 ~12 min: control-int8
2026-11-20 14:49:15 UTC [B-worker codex-trace2] claim cards 0,1 ~28 min: prefill single-256-full
2026-11-20 14:57:11 UTC [B-worker codex-pushar] release cards 1
2026-11-20 14:57:37 UTC [B-worker codex-trace2] release cards 0,1
2026-11-20 14:57:37 UTC [B-worker codex-trace2] claim cards 0,1 ~28 min: prefill single-2048-markers
2026-11-20 14:57:47 UTC [B-worker codex-pushar] claim cards 1 ~8 min: PLE host probe container
2026-11-20 14:58:36 UTC [B-worker codex-park] release cards 0+1
2026-11-20 14:58:46 UTC [A-R] claim cards 0+1 ~10 min: spike.log
2026-11-20 15:04:52 UTC [B-worker codex-trace2] release cards 0,1
2026-11-20 15:05:20 UTC [A-KF] claim card 0 ~5 min: strata-layers16
2026-11-20 15:05:27 UTC [A-R] claim note cards 0+1: R0 C4 T2 has been queued on cpu-run since 14:46:38; no card lease held yet; expected GPU time <=10 min. C1 recorded spike also queued since 14:58.
2026-11-20 15:07:05 UTC [B] codex-trace2 stopped by Lead B at 15:06 (queued single-512-full lease_inner killed before it took cards; no card held). Resumed narrowed: CPU analysis + one <=5 min both-card run later.
2026-11-20 15:07:33 UTC [A-R] released cards 0+1 rc=0
2026-11-20 15:07:33 UTC [A-R] claim cards 0+1 ~10 min: r0-ref-fallback.log
2026-11-20 15:07:47 UTC [B-worker codex-pushar] release cards 1
2026-11-20 15:08:14 UTC [B-worker codex-pushar] claim cards 1 ~8 min: PLE host probe container
2026-11-20 15:10:14 UTC [A-I] released cards 0+1
2026-11-20 15:10:14 UTC [A-I] claim cards 0+1 ~10 min: M-A5 fast kernels a5k-group1.log
2026-11-20 15:10:36 UTC [B-worker codex-park] claim cards 0+1 ~12 min: control-int8
2026-11-20 15:10:45 UTC [A-KF] claim card 0 ~5 min: q6
2026-11-20 15:11:17 UTC [A-KF] released card 0
2026-11-20 15:15:52 UTC [A-R] claim note cards 0+1: first C1 spike returned NOT_READY and hung in error exit; terminating only spike PID 3111780, xe-reset + health recovery queued before the single retry.
2026-11-20 15:15:52 UTC [A-R] claim cards 0+1 ~4 min: reset-spike-1.log
2026-11-20 15:15:54 UTC [A-R] released cards 0+1 rc=137
2026-11-20 15:17:14 UTC [A-R] released cards 0+1 rc=0
2026-11-20 15:17:49 UTC [B-worker codex-pushar] release cards 1
2026-11-20 15:18:24 UTC [B-worker codex-pushar] claim cards 1 ~8 min: PLE host probe buffered
2026-11-20 15:20:36 UTC [B-worker codex-park] release cards 0+1
2026-11-20 15:21:18 UTC [A-I] released cards 0+1
2026-11-20 15:21:18 UTC [A-I] claim cards 0+1 ~10 min: M-A5 fast kernels a5k-C4-T2.log
2026-11-20 15:23:16 UTC [A-KF] released card 0
2026-11-20 15:23:36 UTC [B-worker codex-park] claim cards 0+1 ~12 min: control-int8
2026-11-20 15:23:39 UTC [B-worker codex-kv8] claim cards 0+1 ~12 min: bf16page a4-p3 M-A4 correctness
2026-11-20 15:25:15 UTC [B-worker codex-kv8] release cards 0+1
2026-11-20 15:26:21 UTC [A-R] released cards 0+1 rc=0
2026-11-20 15:26:21 UTC [A-R] claim cards 0+1 ~3 min: health-spike-1.log
2026-11-20 15:27:25 UTC [B-worker codex-pushar] release cards 1
2026-11-20 15:27:59 UTC [B-worker codex-pushar] claim cards 1 ~8 min: PLE host probe direct
2026-11-20 15:29:45 UTC [A-I] released cards 0+1
2026-11-20 15:29:45 UTC [B-worker codex-trace2] claim cards 0,1 ~5 min: 2048-token integration comparison
2026-11-20 15:29:45 UTC [A-I] claim cards 0+1 ~10 min: M-A5 fast kernels a5k-C4qsa-T4.log
2026-11-20 15:31:42 UTC [A-R] claim cards 0+1 ~10 min: spike-retry.log
2026-11-20 15:32:59 UTC [B-worker codex-trace2] release cards 0,1
2026-11-20 15:33:22 UTC [B-worker codex-park] release cards 0+1
2026-11-20 15:33:25 UTC [B-worker codex-kv8] claim cards 0+1 ~12 min: bf16page a4-p4 M-A4 correctness
2026-11-20 15:34:12 UTC [A-KF] claim card 0 ~5 min: q6rp4
2026-11-20 15:34:53 UTC [B-worker codex-kv8] release cards 0+1
2026-11-20 15:35:02 UTC [B-worker codex-park] claim cards 0+1 ~12 min: smoke-int8-other
2026-11-20 15:35:03 UTC [B-worker codex-kv8] claim cards 0+1 ~12 min: bf16page a4-p5 M-A4 correctness
2026-11-20 15:36:56 UTC [B-worker codex-kv8] release cards 0+1
2026-11-20 15:37:26 UTC [A-R] released cards 0+1 rc=0
2026-11-20 15:37:53 UTC [B-worker codex-pushar] release cards 1
2026-11-20 15:40:18 UTC [A-I] released cards 0+1
2026-11-20 15:40:18 UTC [A-I] claim cards 0+1 ~10 min: M-A5 fast kernels a5k-C4stagger-T2.log
2026-11-20 15:41:46 UTC [A-R] released cards 0+1 rc=0
2026-11-20 15:43:04 UTC [A-KF] released card 0
2026-11-20 15:43:34 UTC [B-worker codex-park] release cards 0+1
2026-11-20 15:44:16 UTC [B-worker codex-park] claim cards 0+1 ~12 min: interference-solo
2026-11-20 15:45:06 UTC [B-worker codex-kv8] claim cards 0+1 ~12 min: bf16page a4-p6 M-A4 correctness
2026-11-20 15:45:25 UTC [B-worker codex-kv8] release cards 0+1
2026-11-20 15:47:03 UTC [A-I] released cards 0+1
2026-11-20 15:47:03 UTC [A-I] claim cards 0+1 ~10 min: M-A5 fast kernels a5k-C4soak-T4.log
2026-11-20 15:47:36 UTC [B-worker codex-kv8] claim cards 0+1 ~12 min: bf16page a4-p7 M-A4 correctness
2026-11-20 15:47:44 UTC [A-KF] claim card 0 ~5 min: iq4
2026-11-20 15:47:49 UTC [B-worker codex-park] release cards 0+1
2026-11-20 15:49:46 UTC [B-worker codex-kv8] release cards 0+1
2026-11-20 15:51:38 UTC [A-I] released cards 0+1
2026-11-20 15:51:38 UTC [A-I] claim cards 0+1 ~10 min: M-A5 fast kernels a5k-ple-fallback.log
2026-11-20 15:53:35 UTC [A-KF] released card 0
2026-11-20 15:53:54 UTC [A-I] released cards 0+1
2026-11-20 15:53:57 UTC [B-worker codex-kv8] claim cards 0+1 ~12 min: int8 a4-C2-T2 M-A4 correctness
2026-11-20 15:54:49 UTC [B-worker codex-kv8] release cards 0+1
2026-11-20 15:54:58 UTC [A-R] claim cards 0+1 ~3 min: health-r0-post.log
2026-11-20 15:54:59 UTC [B-worker codex-kv8] claim cards 0+1 ~12 min: int8 a4-C2-T4 M-A4 correctness
2026-11-20 15:55:50 UTC [B-worker codex-kv8] release cards 0+1
2026-11-20 15:56:00 UTC [B-worker codex-kv8] claim cards 0+1 ~12 min: int8 a4-C4-T2 M-A4 correctness
2026-11-20 15:56:27 UTC [A-KF] claim card 0 ~5 min: iq4x4
2026-11-20 15:57:10 UTC [B-worker codex-kv8] release cards 0+1
2026-11-20 15:57:39 UTC [A-R] released cards 0+1 rc=0
2026-11-20 15:58:51 UTC [A-KF] released card 0
2026-11-20 15:59:20 UTC [B-worker codex-kv8] claim cards 0+1 ~12 min: int8 a4-C4-T4 M-A4 correctness
2026-11-20 15:59:41 UTC [B-worker codex-trace2] claim cards 1 ~8 min: Q6_K DPAS parity invariance timing and health
2026-11-20 15:59:42 UTC [B-worker codex-trace2] release cards 1
2026-11-20 16:00:31 UTC [B-worker codex-kv8] release cards 0+1
2026-11-20 16:00:41 UTC [B-worker codex-kv8] claim cards 0+1 ~12 min: int8 a4-C4qsa-T2 M-A4 correctness
2026-11-20 16:02:45 UTC [A-I] claim cards 0+1 ~10 min: M-A5 fast kernels a5k-prof-p2.log
2026-11-20 16:03:24 UTC [B-worker codex-kv8] release cards 0+1
2026-11-20 16:05:09 UTC [A-I] released cards 0+1
2026-11-20 16:05:09 UTC [A-I] claim cards 0+1 ~10 min: M-A5 fast kernels a5k-prof-C4-T2.log
2026-11-20 16:06:39 UTC [A-I] released cards 0+1
2026-11-20 16:06:39 UTC [A-I] claim cards 0+1 ~10 min: M-A5 fast kernels a5k-prof-C4-T4.log
2026-11-20 16:06:49 UTC [B-worker codex-park] claim cards 0+1 ~12 min: interference-solo
2026-11-20 16:07:34 UTC [B-worker codex-kv8] claim cards 0+1 ~12 min: int8 a4-C4qsa-T4 M-A4 correctness
2026-11-20 16:08:34 UTC [A-KF] claim card 0 ~5 min: iq4-persistent-sweep
2026-11-20 16:10:20 UTC [B-worker codex-kv8] release cards 0+1
2026-11-20 16:11:56 UTC [A-I] released cards 0+1
2026-11-20 16:11:56 UTC [A-I] claim cards 0+1 ~10 min: M-A5 fast kernels a5k-prof-p7.log
2026-11-20 16:12:47 UTC [B-worker codex-park] release cards 0+1
2026-11-20 16:15:55 UTC [A-KF] released card 0
2026-11-20 16:17:38 UTC [A-I] released cards 0+1
2026-11-20 16:17:38 UTC [A-I] claim cards 0+1 ~3 min: M-A5 fast kernels health-a5k-post.log
2026-11-20 16:18:23 UTC [A-I] released cards 0+1
2026-11-20 16:18:23 UTC [B-worker codex-trace2] claim cards 1 ~10 min: Q6_K DPAS parity invariance timing and health
2026-11-20 16:18:53 UTC [B-worker codex-trace2] release cards 1
2026-11-20 16:19:07 UTC [B-worker codex-park] claim cards 0+1 ~12 min: interference-mixed
2026-11-20 16:19:33 UTC [A-KF] claim card 0 ~5 min: iq4load
2026-11-20 16:21:06 UTC [B-worker codex-park] release cards 0+1
2026-11-20 16:22:26 UTC [A-KF] released card 0
2026-11-20 16:22:30 UTC [B-worker codex-kv8] claim cards 0+1 ~12 min: int8 a4-C4stagger-T2 M-A4 correctness
2026-11-20 16:23:37 UTC [B-worker codex-kv8] release cards 0+1
2026-11-20 16:23:47 UTC [B-worker codex-kv8] claim cards 0+1 ~12 min: int8 a4-C4stagger-T4 M-A4 correctness
2026-11-20 16:23:58 UTC [A-R] claim cards 0+1 ~15 min: r1-row-invariance.log
2026-11-20 16:24:03 UTC [A-KF] claim card 0 ~5 min: final
2026-11-20 16:24:03 UTC [B-worker codex-trace2] claim cards 1 ~10 min: Q6_K DPAS full-output determinism and health
2026-11-20 16:24:04 UTC [B-worker codex-trace2] release cards 1
2026-11-20 16:24:59 UTC [B-worker codex-kv8] release cards 0+1
2026-11-20 16:26:38 UTC [A-R] released cards 0+1 rc=0
2026-11-20 16:29:59 UTC [A-KF] released card 0
2026-11-20 16:30:02 UTC [B-worker codex-park] claim cards 0+1 ~12 min: matrix-int8-n16-control
2026-11-20 16:30:47 UTC [B-worker codex-trace2] claim cards 1 ~10 min: Q6_K DPAS full-output determinism and health
2026-11-20 16:31:17 UTC [B-worker codex-trace2] release cards 1
2026-11-20 16:32:20 UTC [A-H] claim cards 0+1 ~3 min: health-before.log
2026-11-20 16:32:37 UTC [B-worker codex-park] release cards 0+1
2026-11-20 16:32:37 UTC [B-worker codex-park] claim cards 0+1 ~12 min: matrix-int8-n16-same-k1
2026-11-20 16:33:09 UTC [B-worker codex-kv8] claim cards 0+1 ~12 min: bf16page a4-C2-T2 M-A4 correctness
2026-11-20 16:33:59 UTC [B-worker codex-kv8] release cards 0+1
2026-11-20 16:34:27 UTC [A-H] released cards 0+1 (rc=0)
2026-11-20 16:34:27 UTC [A-H] claim card 0 ~5 min: baseline.log
2026-11-20 16:34:58 UTC [B-worker codex-park] release cards 0+1
2026-11-20 16:34:58 UTC [B-worker codex-park] claim cards 0+1 ~12 min: matrix-int8-n16-other-k1
2026-11-20 16:35:45 UTC [A-H] released card 0 (rc=0)
2026-11-20 16:36:15 UTC [B-worker codex-park] release cards 0+1
2026-11-20 16:36:15 UTC [B-worker codex-trace2] claim cards 1 ~10 min: Q6_K DPAS subgroup Q6 unpack parity timing and health
2026-11-20 16:36:45 UTC [B-worker codex-trace2] release cards 1
2026-11-20 16:37:16 UTC [B-worker codex-park] claim cards 0+1 ~12 min: matrix-int8-n70-control
2026-11-20 16:38:04 UTC [A-H] claim card 0 ~5 min: candidate-check.log
2026-11-20 16:38:09 UTC [B-worker codex-kv8] claim cards 0+1 ~12 min: bf16page a4-C2-T4 M-A4 correctness
2026-11-20 16:38:47 UTC [B-worker codex-park] release cards 0+1
2026-11-20 16:38:56 UTC [B-worker codex-kv8] release cards 0+1
2026-11-20 16:39:06 UTC [B-worker codex-kv8] claim cards 0+1 ~12 min: bf16page a4-C4-T2 M-A4 correctness
2026-11-20 16:40:13 UTC [B-worker codex-kv8] release cards 0+1
2026-11-20 16:40:17 UTC [A-H] released card 0 (rc=0)
2026-11-20 16:40:23 UTC [B-worker codex-kv8] claim cards 0+1 ~12 min: bf16page a4-C4-T4 M-A4 correctness
2026-11-20 16:40:41 UTC [B-worker codex-trace2] claim cards 1 ~10 min: Q6_K DPAS packed Q6 and wide DPAS parity timing and health
2026-11-20 16:40:42 UTC [B-worker codex-trace2] release cards 1
2026-11-20 16:41:31 UTC [B-worker codex-kv8] release cards 0+1
2026-11-20 16:41:41 UTC [B-worker codex-kv8] claim cards 0+1 ~12 min: bf16page a4-C4qsa-T2 M-A4 correctness
2026-11-20 16:42:00 UTC [B-worker codex-trace2] claim cards 1 ~10 min: Q6_K DPAS packed Q6 and wide DPAS parity timing and health
2026-11-20 16:42:01 UTC [B-worker codex-trace2] release cards 1
2026-11-20 16:44:10 UTC [A-H] claim card 0 ~5 min: candidate2-check.log
2026-11-20 16:44:24 UTC [B-worker codex-kv8] release cards 0+1
2026-11-20 16:44:34 UTC [B-worker codex-kv8] claim cards 0+1 ~12 min: bf16page a4-C4qsa-T4 M-A4 correctness
2026-11-20 16:47:03 UTC [A-R] claim cards 0+1 ~15 min: r1-desc-compare.log
2026-11-20 16:47:21 UTC [B-worker codex-kv8] release cards 0+1
2026-11-20 16:47:24 UTC [A-H] released card 0 (rc=0)
2026-11-20 16:47:31 UTC [B-worker codex-kv8] claim cards 0+1 ~12 min: bf16page a4-C4stagger-T2 M-A4 correctness
2026-11-20 16:48:36 UTC [B-worker codex-kv8] release cards 0+1
2026-11-20 16:49:53 UTC [A-R] released cards 0+1 rc=134
2026-11-20 16:50:42 UTC [A-R] claim cards 0+1 ~10 min: reset-r1-desc-1.log
2026-11-20 16:50:46 UTC [B-worker codex-kv8] claim cards 0+1 ~12 min: bf16page a4-C4stagger-T4 M-A4 correctness
2026-11-20 16:51:18 UTC [B-worker codex-trace2] claim cards 1 ~10 min: Q6_K DPAS grouped IQ expert parity timing parity timing and health
2026-11-20 16:51:19 UTC [B-worker codex-trace2] release cards 1
2026-11-20 16:51:19 UTC [B-worker codex-trace2] claim cards 1 ~10 min: Q6_K DPAS packed Q6 and wide DPAS parity timing and health
2026-11-20 16:51:21 UTC [B-worker codex-trace2] release cards 1
2026-11-20 16:51:21 UTC [B-worker codex-trace2] claim cards 1 ~10 min: Q6_K DPAS shared FP16 input parity timing parity timing and health
2026-11-20 16:51:22 UTC [B-worker codex-trace2] release cards 1
2026-11-20 16:51:24 UTC [A-H] claim card 0 ~5 min: candidate3-check.log
2026-11-20 16:51:58 UTC [B-worker codex-kv8] release cards 0+1
2026-11-20 16:52:59 UTC [A-R] released cards 0+1 rc=0
2026-11-20 16:53:00 UTC [B-worker codex-kv8] claim cards 0+1 ~27 min: bf16page 2420-position teacher oracle
2026-11-20 16:53:02 UTC [B-worker codex-kv8] release cards 0+1
2026-11-20 16:53:47 UTC [B-worker codex-park] claim cards 0+1 ~12 min: matrix-int8-n70-control
2026-11-20 16:54:08 UTC [A-H] released card 0 (rc=0)
2026-11-20 16:54:47 UTC [B-worker codex-trace2] claim cards 1 ~10 min: Q6_K DPAS packed Q6 and wide DPAS parity timing and health
2026-11-20 16:55:17 UTC [B-worker codex-trace2] release cards 1
2026-11-20 16:55:57 UTC [B-worker codex-park] release cards 0+1
2026-11-20 16:55:57 UTC [B-worker codex-park] claim cards 0+1 ~12 min: matrix-int8-n70-same-k7
2026-11-20 16:57:38 UTC [B-worker codex-park] release cards 0+1
2026-11-20 16:57:38 UTC [B-worker codex-trace2] claim cards 1 ~10 min: Q6_K DPAS grouped IQ expert chain parity timing parity timing and health
2026-11-20 16:58:08 UTC [B-worker codex-trace2] release cards 1
2026-11-20 16:58:38 UTC [B-worker codex-park] claim cards 0+1 ~12 min: matrix-int8-n70-other-k7
2026-11-20 16:59:16 UTC [B-worker codex-kv8] claim cards 0+1 ~27 min: bf16page 2420-position teacher oracle
2026-11-20 17:00:03 UTC [A-H] claim card 0 ~5 min: candidate4-check.log
2026-11-20 17:01:26 UTC [B-worker codex-park] release cards 0+1
2026-11-20 17:01:36 UTC [A-H] claim card 0 ~5 min: strata-reference.log
2026-11-20 17:03:11 UTC [B-worker codex-kv8] release cards 0+1
2026-11-20 17:03:14 UTC [A-H] released card 0 (rc=0)
2026-11-20 17:03:21 UTC [B-worker codex-kv8] claim cards 0+1 ~27 min: int8 2420-position teacher oracle
2026-11-20 17:07:15 UTC [B-worker codex-kv8] release cards 0+1
2026-11-20 17:07:30 UTC [A-KF] claim card 0 ~5 min: round5-initial-health
2026-11-20 17:07:34 UTC [A-H] released card 0 (rc=0)
2026-11-20 17:07:49 UTC [B-worker codex-kv8] claim cards 0+1 ~27 min: Strata 32768-token oracle, network disabled
2026-11-20 17:08:40 UTC [A-KF] claim card 0 ~5 min: round5-baseline-checks-and-times
2026-11-20 17:08:46 UTC [B-worker codex-kv8] release cards 0+1
2026-11-20 17:08:56 UTC [B-worker codex-kv8] claim cards 0+1 ~27 min: Strata 122880-token oracle, network disabled
2026-11-20 17:09:50 UTC [A-KF] claim card 0 ~5 min: round5-strata-decode-reference
2026-11-20 17:10:03 UTC [A-H] claim card 0 ~5 min: candidate5-check.log
2026-11-20 17:10:44 UTC [A-R] claim cards 0+1 ~15 min: r1-desc-compare-retry.log
2026-11-20 17:10:57 UTC [B-worker codex-kv8] release cards 0+1
2026-11-20 17:11:10 UTC [A-KF] released card 0
2026-11-20 17:11:40 UTC [A-KF] released card 0
2026-11-20 17:12:27 UTC [B-worker codex-park] claim cards 0+1 ~12 min: matrix-int8-n70-other-k7
2026-11-20 17:12:38 UTC [A-KF] claim card 0 ~5 min: round5-expert-gates-then-times
2026-11-20 17:13:20 UTC [A-KF] released card 0
2026-11-20 17:13:23 UTC [A-H] released card 0 (rc=0)
2026-11-20 17:13:35 UTC [B-worker codex-kv8] claim cards 0+1 ~27 min: bf16page 32768-token compact teacher
2026-11-20 17:22:27 UTC [B-worker codex-park] release cards 0+1
2026-11-20 17:25:13 UTC [A-R] claim note cards 0+1: descriptor retry still queued behind Lead B long-context teacher; no retry GPU touch yet; original 17:10 claim remains pending.
2026-11-20 17:26:04 UTC [B-worker codex-kv8] release cards 0+1
2026-11-20 17:27:45 UTC [A-R] released cards 0+1 rc=0
2026-11-20 17:28:49 UTC [A-KF] released card 0
2026-11-20 17:29:27 UTC [B-worker codex-park] claim cards 0+1 ~12 min: matrix-int8-n70-other-k7
2026-11-20 17:29:52 UTC [B-worker codex-kv8] claim cards 0+1 ~27 min: int8 32768-token compact teacher
2026-11-20 17:29:54 UTC [A-KF] claim card 0 ~5 min: round5-expert-resume-prepared-gates-then-times
2026-11-20 17:31:08 UTC [A-KF] claim card 0 ~5 min: round5-decode-rank-gates-and-times
2026-11-20 17:31:15 UTC [A-H] claim card 0 ~5 min: selected-check.log
2026-11-20 17:33:44 UTC [B-worker codex-park] release cards 0+1
2026-11-20 17:36:03 UTC [A-KF] released card 0
2026-11-20 17:36:03 UTC [A-KF] released card 0
2026-11-20 17:38:42 UTC [A-R] claim cards 0+1 ~10 min: r1-mtp-p0-smoke.log
2026-11-20 17:42:29 UTC [B-worker codex-kv8] release cards 0+1
2026-11-20 17:42:32 UTC [A-H] released card 0 (rc=0)
2026-11-20 17:42:32 UTC [A-KF] claim card 0 ~5 min: round5-decode-rank-gates-and-times
2026-11-20 17:42:32 UTC [A-KF] claim card 0 ~5 min: round5-expert-resume-prepared-gates-then-times
2026-11-20 17:42:35 UTC [A-H] claim cards 0+1 ~10 min: a5k-teacher-q8_1-1.log
2026-11-20 17:42:45 UTC [B-worker codex-park] claim cards 0+1 ~12 min: matrix-int8-n70-other-k7
2026-11-20 17:43:40 UTC [B-worker codex-kv8] claim cards 0+1 ~12 min: bf16page 512-row teacher verification
2026-11-20 17:45:26 UTC [B-worker codex-kv8] release cards 0+1
2026-11-20 17:46:52 UTC [A-R] released cards 0+1 rc=0
2026-11-20 17:47:36 UTC [A-KF] released card 0
2026-11-20 17:48:42 UTC [A-KF] released card 0
2026-11-20 17:50:48 UTC [A-H] released cards 0+1 (rc=0)
2026-11-20 17:50:48 UTC [A-H] claim cards 0+1 ~10 min: a5k-teacher-q8_1-2.log
2026-11-20 17:51:32 UTC [B-worker codex-park] release cards 0+1
2026-11-20 17:51:32 UTC [B-worker codex-park] claim cards 0+1 ~12 min: matrix-int8-n256-control
2026-11-20 17:51:37 UTC [B-worker codex-kv8] claim cards 0+1 ~12 min: int8 512-row teacher verification
2026-11-20 17:53:25 UTC [B-worker codex-kv8] release cards 0+1
2026-11-20 17:53:25 UTC [A-KF] claim card 0 ~5 min: round5-decode-qualification-and-times
2026-11-20 17:53:25 UTC [A-KF] claim card 0 ~5 min: round5-strata-rank-reference
2026-11-20 17:56:48 UTC [A-H] released cards 0+1 (rc=0)
2026-11-20 18:00:50 UTC [A-KF] released card 0
2026-11-20 18:01:46 UTC [A-KF] released card 0
2026-11-20 18:06:54 UTC [A-H] claim cards 0+1 ~10 min: a5k-group0.log
2026-11-20 18:06:57 UTC [A-S] claim cards 0+1 ~4 min: pre-serve health
2026-11-20 18:07:28 UTC [A-S] released cards 0+1: pre-serve health rc=0
2026-11-20 18:07:28 UTC [A-S] claim cards 0+1 ~50 min: Flash Next serve benchmark
2026-11-20 18:08:58 UTC [B-worker codex-kv8] claim cards 0+1 ~4 min: merge smoke off greedy
2026-11-20 18:09:04 UTC [A-I] claim cards 0+1 ~3 min: smoke smoke-46afffb
2026-11-20 18:10:25 UTC [B-worker codex-kv8] release cards 0+1
2026-11-20 18:10:27 UTC [B-worker codex-kv8] claim cards 0+1 ~4 min: merge smoke bf16page greedy
2026-11-20 18:11:57 UTC [orchestrator] claim card 0 ~3 min: latbench single (copy H2D/D2H) pinned to cpus 0-7 for NUMA probe
2026-11-20 18:12:31 UTC [B-worker codex-kv8] release cards 0+1
2026-11-20 18:14:49 UTC [orchestrator] release card 0
2026-11-20 18:14:49 UTC [orchestrator] claim card 0 ~3 min: latbench single (copy H2D/D2H) pinned to cpus 8-15 for NUMA probe
2026-11-20 18:16:55 UTC [A-H] released cards 0+1 (rc=0)
2026-11-20 18:16:55 UTC [A-KF] claim card 0 ~5 min: round5-final-fp64-bitwise-and-recorded-gates
2026-11-20 18:16:55 UTC [A-H] claim cards 0+1 ~10 min: a5k-group1.log
2026-11-20 18:17:56 UTC [A-R] claim cards 0+1 ~20 min: r1-state.log
2026-11-20 18:19:13 UTC [orchestrator] release card 0
2026-11-20 18:19:13 UTC [orchestrator] claim card 1 ~3 min: latbench single (copy H2D/D2H) pinned to cpus 0-7 for NUMA probe
2026-11-20 18:25:22 UTC [A-S] released cards 0+1: benchmark session complete; outer script parse error
2026-11-20 18:25:22 UTC [A-S] claim cards 0+1 ~4 min: post-serve health
2026-11-20 18:25:53 UTC [orchestrator] release card 1
2026-11-20 18:25:53 UTC [orchestrator] claim card 1 ~3 min: latbench single (copy H2D/D2H) pinned to cpus 8-15 for NUMA probe
2026-11-20 18:28:05 UTC [A-S] released cards 0+1: post-serve health rc=0
2026-11-20 18:30:22 UTC [orchestrator] release card 1
2026-11-20 18:33:05 UTC [A-I] released cards 0+1
2026-11-20 18:33:47 UTC [B-worker codex-park] claim cards 0+1 ~3 min: two-prompt K=7 T2 smoke
2026-11-20 18:36:38 UTC [A-KF] released card 0
2026-11-20 18:36:38 UTC [A-KF] claim card 0 ~5 min: R5 final Strata rank profiling
2026-11-20 18:38:37 UTC [A-H] released cards 0+1 (rc=0)
2026-11-20 18:38:37 UTC [A-KF] claim card 0 ~5 min: R5 attention64 before-after device timing
2026-11-20 18:38:37 UTC [A-H] claim cards 0+1 ~10 min: a5k-C4-T2.log
2026-11-20 18:42:06 UTC [A-R] released cards 0+1 rc=0
2026-11-20 18:42:06 UTC [A-R] claim cards 0+1 ~20 min: r1-teacher.log
2026-11-20 18:43:37 UTC [B-worker codex-kv8] claim cards 0+1 ~3 min: final GPU health
2026-11-20 18:43:47 UTC [B-worker codex-park] release cards 0+1
2026-11-20 18:44:05 UTC [B-worker codex-kv8] release cards 0+1
2026-11-20 18:44:47 UTC [B-worker codex-park] claim cards 0+1 ~3 min: two-prompt K=7 T2 smoke
2026-11-20 18:47:20 UTC [B-worker codex-park] release cards 0+1 (CPU wait canceled; no GPU claim acquired)
2026-11-20 18:47:23 UTC [B-worker codex-park] claim cards 0+1 ~3 min: two-prompt K=7 T2 smoke
2026-11-20 18:48:10 UTC [A-I] claim cards 0+1 ~3 min: smoke smoke-round4
2026-11-20 18:48:55 UTC [A-KF] released card 0
2026-11-20 18:48:56 UTC [A-KF] released card 0
2026-11-20 18:49:12 UTC [A-S] claim cards 0+1 ~20 min: round2-policy policy serve, no-MTP, pre/post health
2026-11-20 18:50:17 UTC [B-worker codex-park] release cards 0+1
2026-11-20 18:50:17 UTC [A-KF] claim card 0 ~5 min: R5 final Strata rank profiling
2026-11-20 18:50:17 UTC [A-KF] claim card 0 ~5 min: R5 attention64 before-after device timing
2026-11-20 18:51:44 UTC [A-H] released cards 0+1 (rc=0)
2026-11-20 18:51:44 UTC [A-H] claim cards 0+1 ~10 min: a5k-C4-T4.log
2026-11-20 18:53:58 UTC [A-R] released cards 0+1 rc=127
2026-11-20 18:54:48 UTC [B-worker codex-kv8] claim cards 0+1 ~4 min: merge smoke int8 greedy
2026-11-20 18:54:49 UTC [B-worker codex-park] claim cards 0+1 ~3 min: post-smoke GPU health
2026-11-20 18:54:54 UTC [B-worker codex-park] release cards 0+1 (post-smoke health rc=124)
2026-11-20 18:55:46 UTC [B-worker codex-park] claim cards 0+1 ~3 min: retry post-smoke GPU health
2026-11-20 18:55:59 UTC [A-R] claim cards 0+1 ~40 min: r1m11fecc6-suite.log
2026-11-20 18:56:13 UTC [B-worker codex-kv8] release cards 0+1
2026-11-20 18:56:13 UTC [A-KF] claim card 0 ~5 min: R5 final SIMD16 and decode qualification
2026-11-20 18:56:40 UTC [B-worker codex-park] release cards 0+1 (post-smoke health retry rc=0)
2026-11-20 18:58:15 UTC [B-worker codex-kv8] claim cards 0+1 ~4 min: merge smoke int8 concurrent
2026-11-20 18:59:39 UTC [B-worker codex-kv8] release cards 0+1
2026-11-20 19:02:40 UTC [A-I] released cards 0+1
2026-11-20 19:03:29 UTC [A-I] claim cards 0+1 ~5 min: M-A5 fast kernels smoke-reset-health.log
2026-11-20 19:06:58 UTC [A-S] released cards 0+1: round2-policy rc=0
2026-11-20 19:08:15 UTC [A-KF] released card 0
2026-11-20 19:09:01 UTC [A-H] released cards 0+1 (rc=124)
2026-11-20 19:10:14 UTC [A-KF] released card 0
2026-11-20 19:10:42 UTC [A-R] claim cards 0+1 (inherited batch lease): r1m11fecc6/state
2026-11-20 19:13:24 UTC [A-KF] released card 0
2026-11-20 19:14:20 UTC [A-R] release child run r1m11fecc6/state rc=0; outer cards 0+1 batch lease retained
2026-11-20 19:14:20 UTC [A-R] claim cards 0+1 (inherited batch lease): r1m11fecc6/teacher-desc0
2026-11-20 19:16:33 UTC [A-R] release child run r1m11fecc6/teacher-desc0 rc=0; outer cards 0+1 batch lease retained
2026-11-20 19:16:34 UTC [A-R] claim cards 0+1 (inherited batch lease): r1m11fecc6/teacher-desc1
2026-11-20 19:18:20 UTC [A-R] claim-note cards 0+1: r1m11fecc6 bounded batch active since19:10:14; both health and140 state cases PASS, immediate teacher clean; descriptor teacher running, then remaining token/concurrency gates. Initial claim18:55:59 included queue wait.
2026-11-20 19:18:50 UTC [A-R] release child run r1m11fecc6/teacher-desc1 rc=0; outer cards 0+1 batch lease retained
2026-11-20 19:18:50 UTC [A-R] claim cards 0+1 (inherited batch lease): r1m11fecc6/group0-desc0
2026-11-20 19:21:25 UTC [A-R] release child run r1m11fecc6/group0-desc0 rc=0; outer cards 0+1 batch lease retained
2026-11-20 19:21:25 UTC [A-R] claim cards 0+1 (inherited batch lease): r1m11fecc6/group1-desc0
2026-11-20 19:23:46 UTC [A-R] release child run r1m11fecc6/group1-desc0 rc=0; outer cards 0+1 batch lease retained
2026-11-20 19:23:47 UTC [A-R] claim cards 0+1 (inherited batch lease): r1m11fecc6/group0-desc1
2026-11-20 19:25:12 UTC [A] NOTE to all Lead A workers (user: faster loop): correctness/gate runs use gpu-run only, NOT cpu-run --quiet (quiet blocks every build on the host); quiet only around timed benchmark windows. Split GPU batches into <=15 min leases.
2026-11-20 19:26:19 UTC [A-R] release child run r1m11fecc6/group0-desc1 rc=0; outer cards 0+1 batch lease retained
2026-11-20 19:26:19 UTC [A-R] claim-note cards 0+1: r1m11fecc6 batch still running bounded gates; elapsed 16.1 min
2026-11-20 19:26:19 UTC [A-R] claim cards 0+1 (inherited batch lease): r1m11fecc6/group1-desc1
2026-11-20 19:29:10 UTC [A-R] released cards 0+1 rc=143
2026-11-20 19:29:29 UTC [A-R] release child r1m11fecc6/group1-desc1 rc=0, native teardown verified via reaped child/timeout exit0; outer quiet batch intentionally split for new host rules.
2026-11-20 19:29:29 UTC [A-R] claim cards 0+1 ~15 min: r1m11fecc6-concurrency.log
2026-11-20 19:30:44 UTC [A-I] released cards 0+1
2026-11-20 19:30:44 UTC [B-worker codex-trace2] claim cards 0+1 ~3 min: DPAS prefill smoke
2026-11-20 19:30:45 UTC [B-worker codex-trace2] release cards 0+1
2026-11-20 19:30:48 UTC [A-H] claim cards 0+1 ~10 min: a5k-C4-T4.log
2026-11-20 19:31:03 UTC [A-I] claim cards 0+1 ~3 min: smoke smoke-round4-retry
2026-11-20 19:31:16 UTC [A-R] claim cards 0+1 (inherited batch lease): r1m11fecc6/C4-T2
2026-11-20 19:31:44 UTC [B-worker codex-trace2] claim cards 0+1 ~3 min: DPAS prefill smoke
2026-11-20 19:31:45 UTC [B-worker codex-trace2] release cards 0+1
2026-11-20 19:32:49 UTC [A-R] release child run r1m11fecc6/C4-T2 rc=0; outer cards 0+1 batch lease retained
2026-11-20 19:32:49 UTC [A-R] claim cards 0+1 (inherited batch lease): r1m11fecc6/C4qsa-T4
2026-11-20 19:34:33 UTC [A-R] release child run r1m11fecc6/C4qsa-T4 rc=0; outer cards 0+1 batch lease retained
2026-11-20 19:34:33 UTC [A-R] claim cards 0+1 (inherited batch lease): r1m11fecc6/C4stagger-T2
2026-11-20 19:35:59 UTC [A-R] release child run r1m11fecc6/C4stagger-T2 rc=0; outer cards 0+1 batch lease retained
2026-11-20 19:35:59 UTC [A-R] claim cards 0+1 (inherited batch lease): r1m11fecc6/C4soak-T4
2026-11-20 19:37:47 UTC [A-R] release child run r1m11fecc6/C4soak-T4 rc=0; outer cards 0+1 batch lease retained
2026-11-20 19:38:17 UTC [A-R] released cards 0+1 rc=0
2026-11-20 19:39:44 UTC [A-H] released cards 0+1 (rc=0)
2026-11-20 19:39:44 UTC [A-H] claim cards 0+1 ~10 min: a5k-prof-p2.log
2026-11-20 19:42:09 UTC [A-I] released cards 0+1
2026-11-20 19:42:09 UTC [A-KF] claim card 0 ~5 min: R5 hybrid decode full qualification
2026-11-20 19:42:09 UTC [B-worker codex-trace2] claim cards 0+1 ~3 min: DPAS prefill smoke
2026-11-20 19:45:15 UTC [B-worker codex-trace2] release cards 0+1
2026-11-20 19:47:44 UTC [A-H] released cards 0+1 (rc=0)
2026-11-20 19:47:44 UTC [A-H] claim cards 0+1 ~10 min: a5k-prof-C4-T4.log
2026-11-20 19:50:12 UTC [A-KF] released card 0
2026-11-20 19:50:33 UTC [A-KF] claim card 0 ~5 min: R5 qualified hybrid timings Strata experts and final health
2026-11-20 19:52:35 UTC [B-worker codex-park] claim cards 0+1 ~28 min: push smoke and 1K/8K/32K park latency
2026-11-20 19:54:47 UTC [B-worker codex-trace2] claim cards 0+1 ~15 min: DPAS prefill perf
2026-11-20 19:54:48 UTC [B-worker codex-trace2] release cards 0+1
2026-11-20 20:09:50 UTC [A] REQUEST to all (Lead B workers too): cpu-run --quiet blocks every build host-wide; use it only around timed windows, not whole sessions or correctness runs.
2026-11-20 20:10:08 UTC [B-worker codex-park] release cards 0+1
2026-11-20 20:11:23 UTC [B-worker codex-park] claim cards 0+1 ~3 min: post-latency GPU health
2026-11-20 20:11:48 UTC [A-H] released cards 0+1 (rc=0)
2026-11-20 20:11:48 UTC [A-H] claim cards 0+1 ~3 min: health-final.log
2026-11-20 20:12:18 UTC [B-worker codex-park] release cards 0+1 (post-latency health rc=0)
2026-11-20 20:15:13 UTC [A-KF] released card 0
2026-11-20 20:15:13 UTC [B-worker codex-trace2] claim cards 0+1 ~15 min: DPAS prefill perf
2026-11-20 20:16:49 UTC [A-H] released cards 0+1 (rc=124)
2026-11-20 20:17:05 UTC [A-H] claim cards 0+1 ~3 min: health-final-retry.log
2026-11-20 20:21:31 UTC [B-worker codex-trace2] release cards 0+1
2026-11-20 20:22:02 UTC [A-H] released cards 0+1 (rc=0)
2026-11-20 20:25:53 UTC [A-H] claim cards 0+1 ~3 min: hc2-health-pre.log
2026-11-20 20:26:21 UTC [A-H] released cards 0+1 (rc=0)
2026-11-20 20:29:06 UTC [A-R] claim cards 0+1 ~15 min: r2-light1.log
2026-11-20 20:33:09 UTC [A-I] claim cards 0+1 ~3 min: smoke smoke-hc
2026-11-20 20:33:38 UTC [A-R] released cards 0+1 rc=1
2026-11-20 20:35:21 UTC [A-KF] claim card 0 ~5 min: R5 scalar staging pair prefetch and expert pipeline full qualification
2026-11-20 20:35:38 UTC [A-H] claim card 0 ~5 min: hc2/check1.log
2026-11-20 20:35:39 UTC [A-H] claim card 0 ~5 min: hc2/time1.log
2026-11-20 20:35:39 UTC [A-H] released card 0 (rc=0)
2026-11-20 20:36:01 UTC [A-R] claim cards 0+1 ~15 min: r2-ref-session.log
2026-11-20 20:40:35 UTC [A-R] released cards 0+1 rc=0
2026-11-20 20:41:49 UTC [A-R] claim cards 0+1 ~15 min: r2 policy serve, no-MTP, pre/post health
2026-11-20 20:42:57 UTC [A-I] released cards 0+1
2026-11-20 20:42:57 UTC [B-worker codex-trace2] claim cards 0+1 ~6 min: DPAS prefill profile
2026-11-20 20:42:58 UTC [B-worker codex-trace2] release cards 0+1
2026-11-20 20:43:05 UTC [A-S] claim cards 0+1 ~20 min: int-hc policy serve, no-MTP, pre/post health
2026-11-20 20:43:07 UTC [A-KF] released card 0
2026-11-20 20:45:53 UTC [A-R] released cards 0+1: r2 rc=75
2026-11-20 20:45:53 UTC [A-KF] claim card 0 ~5 min: R5 scalar staging pair prefetch and expert pipeline full qualification
2026-11-20 20:45:55 UTC [A-H] released card 0 (rc=0)
2026-11-20 20:50:12 UTC [B-worker codex-park] claim cards 0+1 ~9 min: serve eviction correctness smoke and health
2026-11-20 20:52:39 UTC [A-S] released cards 0+1: int-hc rc=1
2026-11-20 20:52:39 UTC [B-worker codex-trace2] claim cards 0+1 ~6 min: DPAS prefill profile
2026-11-20 20:52:40 UTC [B-worker codex-trace2] release cards 0+1
2026-11-20 20:52:41 UTC [A-KF] released card 0
2026-11-20 20:52:55 UTC [A-S] claim cards 0+1 ~20 min: int-hc2 policy serve, no-MTP, pre/post health
2026-11-20 20:53:54 UTC [A-H] claim cards 0+1 ~3 min: smoke hc2-h
2026-11-20 20:57:08 UTC [B-worker codex-park] release cards 0+1
2026-11-20 20:57:08 UTC [A-KF] claim card 0 ~5 min: R5 scalar staging pair prefetch and expert pipeline full qualification
2026-11-20 20:57:36 UTC [B-worker codex-park] claim cards 0+1 ~9 min: serve eviction correctness smoke and health
2026-11-20 20:57:43 UTC [B-worker codex-trace2] claim cards 0+1 ~6 min: DPAS prefill profile
2026-11-20 20:57:44 UTC [B-worker codex-trace2] release cards 0+1
2026-11-20 20:59:31 UTC [A-H] released cards 0+1
2026-11-20 21:02:08 UTC [B-worker codex-park] release cards 0+1
2026-11-20 21:02:18 UTC [B-worker codex-park] claim cards 0+1 ~9 min: serve eviction correctness smoke and health
2026-11-20 21:03:56 UTC [A-R] claim cards 0+1 ~15 min: r2 policy serve, no-MTP, pre/post health; GPU-only startup then ~2min timed client CPU quiet (avoid quiet waits behind this lease)
2026-11-20 21:04:56 UTC [A-S] released cards 0+1: int-hc2 rc=1
2026-11-20 21:05:32 UTC [A-S] claim cards 0+1 ~20 min: int-hc3 policy serve, no-MTP, pre/post health
2026-11-20 21:05:55 UTC [A-KF] released card 0
2026-11-20 21:05:55 UTC [B-worker codex-trace2] claim cards 0+1 ~6 min: DPAS prefill profile
2026-11-20 21:05:56 UTC [B-worker codex-trace2] release cards 0+1
2026-11-20 21:07:36 UTC [B-worker codex-park] release cards 0+1
2026-11-20 21:07:47 UTC [B-worker codex-park] claim cards 0+1 ~9 min: serve eviction correctness smoke and health
2026-11-20 21:10:41 UTC [A-R] released cards 0+1: r2 rc=75
2026-11-20 21:10:41 UTC [A-KF] claim card 0 ~5 min: R5 scalar staging pair prefetch and expert pipeline full qualification
2026-11-20 21:11:01 UTC [A-R] claim cards 0+1 ~15 min: r2-off policy serve, no-MTP, pre/post health; GPU-only startup then ~2min timed client CPU quiet (avoid quiet waits behind this lease)
2026-11-20 21:12:39 UTC [A-KF] released card 0 (queued probe canceled; no GPU workload started)
2026-11-20 21:12:39 UTC [A-KF] claim card 0 ~5 min: R5 final pair-prefetch full parity fixed queue
2026-11-20 21:13:13 UTC [A-R] released cards 0+1: r2-off rc=126
2026-11-20 21:15:09 UTC [B-worker codex-park] release cards 0+1
2026-11-20 21:19:00 UTC [A-S] released cards 0+1: int-hc3 rc=0
2026-11-20 21:19:00 UTC [B-worker codex-trace2] claim cards 0+1 ~6 min: DPAS prefill profile
2026-11-20 21:21:19 UTC [B-worker codex-trace2] release cards 0+1
2026-11-20 21:21:20 UTC [A-H] claim cards 0+1 ~15 min: hc2 policy serve, no-MTP, pre/post health
2026-11-20 21:23:49 UTC [A-KF] released card 0
2026-11-20 21:23:50 UTC [A-R] claim cards 0+1 ~15 min: r2 bounded policy serve/no-MTP/pre-post health; CPU quiet acquired before GPU
2026-11-20 21:25:48 UTC [A-KF] claim card 0 ~5 min: R5 qualified final pair-prefetch device times and closing health
2026-11-20 21:26:25 UTC [A-R] released cards 0+1: r2 rc=1
2026-11-20 21:31:48 UTC [B-worker codex-park] claim cards 0+1 ~9 min: serve eviction correctness smoke and health
2026-11-20 21:34:25 UTC [B-worker codex-park] release cards 0+1
2026-11-20 21:34:57 UTC [B-worker codex-park] claim cards 0+1 ~17 min: eviction C2/C4 benchmark and health
2026-11-20 21:35:50 UTC [A-I] claim cards 0+1 ~3 min: smoke smoke-kv8off
2026-11-20 21:37:26 UTC [B-worker codex-park] release cards 0+1
2026-11-20 21:37:51 UTC [B-worker codex-park] claim cards 0+1 ~17 min: eviction C2/C4 benchmark and health
2026-11-20 21:41:07 UTC [A-H] released cards 0+1: hc2 rc=0
2026-11-20 21:45:36 UTC [B-worker codex-park] release cards 0+1
2026-11-20 21:45:48 UTC [A-H] claim cards 0+1 ~3 min: hc3/health-pre.log
2026-11-20 21:47:50 UTC [A-KF] released card 0
2026-11-20 21:47:50 UTC [B-worker codex-trace2] claim cards 1 ~4 min: grouped expert adapter parity
2026-11-20 21:47:51 UTC [B-worker codex-trace2] release cards 1
2026-11-20 21:48:20 UTC [A-H] released cards 0+1 (rc=0)
2026-11-20 21:54:14 UTC [A-R] claim cards 0+1 ~15 min: r2-light2-session.log
2026-11-20 21:56:20 UTC [A-I] released cards 0+1
2026-11-20 22:00:17 UTC [A-R] released cards 0+1 rc=0
2026-11-20 22:01:47 UTC [B-worker codex-park] claim cards 0+1 ~9 min: serve eviction correctness smoke and health
2026-11-20 22:02:10 UTC [A-H] claim cards 0+1 ~5 min: hc3/profile-baseline.log
2026-11-20 22:04:24 UTC [B-worker codex-park] release cards 0+1
2026-11-20 22:04:41 UTC [B-worker codex-park] claim cards 0+1 ~17 min: eviction C2/C4 benchmark and health
2026-11-20 22:06:31 UTC [A-R] claim cards 0+1 ~15 min: r2 bounded policy serve/no-MTP/pre-post health; CPU quiet acquired before GPU
2026-11-20 22:07:33 UTC [B-worker codex-park] release cards 0+1
2026-11-20 22:07:52 UTC [B-worker codex-park] claim cards 0+1 ~17 min: eviction C2/C4 benchmark and health
2026-11-20 22:12:35 UTC [A-R] released cards 0+1: r2 rc=1
2026-11-20 22:15:36 UTC [B-worker codex-park] release cards 0+1
2026-11-20 22:17:01 UTC [A-H] released cards 0+1 (rc=0)
2026-11-20 22:21:44 UTC [A-I] claim cards 0+1 ~3 min: smoke smoke-comb
2026-11-20 22:31:20 UTC [B-worker codex-park] claim cards 0+1 ~9 min: serve eviction correctness smoke and health
2026-11-20 22:34:10 UTC [B-worker codex-park] release cards 0+1
2026-11-20 22:34:46 UTC [B-worker codex-park] claim cards 0+1 ~17 min: eviction C2/C4 benchmark and health
2026-11-20 22:37:41 UTC [B-worker codex-park] release cards 0+1
2026-11-20 22:40:07 UTC [A-I] released cards 0+1
2026-11-20 22:41:25 UTC [A-S] claim cards 0+1 ~15 min: comb-r2 bounded policy serve/no-MTP/pre-post health; CPU quiet acquired before GPU
2026-11-20 22:41:52 UTC [A-KF] claim cards 0+1 ~3 min: smoke prep-r6-imm
2026-11-20 22:43:21 UTC [A-H] claim card 0 ~5 min: hc3/check-candidate1.log
2026-11-20 22:44:50 UTC [A-S] released cards 0+1: comb-r2 rc=0
2026-11-20 22:47:22 UTC [A-KF] released cards 0+1
2026-11-20 22:47:26 UTC [A-H] released card 0 (rc=134)
2026-11-20 22:48:13 UTC [A-KF] claim cards 0+1 ~15 min: prepared forward light record gate and pre/post health
2026-11-20 22:49:54 UTC [A-S] claim cards 0+1 ~15 min: comb-imm bounded policy serve/no-MTP/pre-post health; CPU quiet acquired before GPU
2026-11-20 22:51:29 UTC [B-worker codex-park] claim cards 0+1 ~9 min: merged integration cold-on serve correctness and health
2026-11-20 22:52:10 UTC [A-KF] released cards 0+1: prepared forward light record gate rc=0
2026-11-20 22:55:45 UTC [A-S] released cards 0+1: comb-imm rc=0
2026-11-20 22:58:24 UTC [B-worker codex-park] release cards 0+1
2026-11-20 23:01:21 UTC [B-worker codex-ledger] claim cards 0+1 ~10 min: HC ledger hc2 off
2026-11-20 23:01:35 UTC [A-R] claim cards 0+1 ~15 min: r2-light3-session.log
2026-11-20 23:02:06 UTC [A-KF] claim cards 0+1 ~15 min: prep-rec bounded policy serve/no-MTP/pre-post health; CPU quiet acquired before GPU
2026-11-20 23:02:30 UTC [A-H] claim card 0 ~5 min: hc3/check-candidate2.log
2026-11-20 23:03:28 UTC [B-worker codex-ledger] release cards 0+1 hc2 off rc=1
2026-11-20 23:07:15 UTC [A-R] released cards 0+1 rc=0
2026-11-20 23:08:03 UTC [B] REQUEST: codex-ledger capture (user priority 21:50, orchestrator ask) needs ONE both-card slot ~10 min next; please let it in after the current holder (no new long claims until it lands).
2026-11-20 23:11:11 UTC [A-KF] released cards 0+1: prep-rec rc=1
2026-11-20 23:11:11 UTC [A-KF] claim cards 0+1 ~15 min: prep-imm bounded policy serve/no-MTP/pre-post health; CPU quiet acquired before GPU
2026-11-20 23:11:14 UTC [A-H] released card 0 (rc=134)
2026-11-20 23:14:55 UTC [A-KF] released cards 0+1: prep-imm rc=0
2026-11-20 23:18:38 UTC [B-worker codex-ledger] claim cards 0+1 ~25 min: HC ledger hc2 off markers full
2026-11-20 23:19:09 UTC [A-R] claim cards 0+1 ~15 min: r2-diag-retry bounded policy serve/no-MTP/pre-post health; CPU quiet acquired before GPU
2026-11-20 23:19:25 UTC [B-worker codex-park] claim cards 0+1 ~9 min: merged integration mirror-on serve correctness and health
2026-11-20 23:26:52 UTC [B] NOTE to Lead A: A-R (pid 3447395) holds cpu-run --quiet since 23:19 while queued for the GPUs (held by codex-ledger until ~23:43) - this blocks every build host-wide; please acquire quiet only after the GPU lease.
2026-11-20 23:29:12 UTC [A] NOTE (re B 23:26): fn-a3/fn-int da583ba tools/flashnext_serve_bench.sh now waits until both cards are free (gpu-run --status) BEFORE taking cpu-run --quiet, so a queued bench no longer blocks builds. Lead A workers: merge b70/fn-a3 tools/ before the next bench; until then do not queue a bench while the GPUs are held.
2026-11-20 23:30:44 UTC [B-worker codex-ledger] release cards 0+1 hc2 rc=0
2026-11-20 23:35:04 UTC [A-R] released cards 0+1: r2-diag-retry rc=1
2026-11-20 23:37:07 UTC [B-worker codex-park] release cards 0+1
2026-11-20 23:37:15 UTC [B-worker codex-park] claim cards 0+1 ~28 min: mirror on park latency contexts 1024
2026-11-20 23:39:32 UTC [B-worker codex-park] release cards 0+1
2026-11-20 23:39:43 UTC [B-worker codex-park] claim cards 0+1 ~28 min: mirror off park latency contexts 1024
2026-11-20 23:40:36 UTC [A-H] claim card 0 ~5 min: hc3/check-debug.log
2026-11-20 23:42:13 UTC [B-worker codex-park] release cards 0+1
2026-11-20 23:42:16 UTC [A-H] released card 0 (rc=134)
2026-11-20 23:42:37 UTC [B-worker codex-park] claim cards 0+1 ~12 min: C4 cap-2 preallocation and mirror off/on bench
2026-11-20 23:46:57 UTC [B-worker codex-park] release cards 0+1
2026-11-20 23:47:21 UTC [B-worker codex-park] claim cards 0+1 ~9 min: merged integration cache-on serve correctness and health
2026-11-20 23:49:40 UTC [B-worker codex-park] release cards 0+1
2026-11-20 23:50:00 UTC [B-worker codex-park] claim cards 0+1 ~12 min: C4 1K prompt mirror off/on and health
2026-11-20 23:52:10 UTC [A-H] claim card 0 ~5 min: hc3/check-rebase.log
2026-11-20 23:54:01 UTC [A-KF] claim cards 0+1 ~3 min: smoke prep-r6-final-imm
2026-11-20 23:54:55 UTC [B-worker codex-park] release cards 0+1
2026-11-20 23:55:00 UTC [A-H] released card 0 (rc=0)
2026-11-20 23:55:23 UTC [B-worker codex-park] claim cards 0+1 ~10 min: mirror on park latency contexts 8192
2026-11-20 23:57:25 UTC [A-KF] released cards 0+1
2026-11-20 23:57:26 UTC [B-worker codex-ledger] claim cards 0+1 ~25 min: HC ledger int-da583ba off markers full
2026-11-21 00:01:54 UTC [B-worker codex-park] release cards 0+1
2026-11-21 00:02:15 UTC [B-worker codex-park] claim cards 0+1 ~10 min: mirror off park latency contexts 8192
2026-11-21 00:02:46 UTC [B-worker codex-park] claim cards 0+1 ~9 min: merged integration cache-on serve correctness and health
2026-11-21 00:07:44 UTC [A-KF] claim cards 0+1 ~15 min: prep-imm bounded policy serve/no-MTP/pre-post health; CPU quiet acquired before GPU
2026-11-21 00:12:21 UTC [B-worker codex-ledger] release cards 0+1 int-da583ba rc=0
2026-11-21 00:16:52 UTC [B-worker codex-park] release cards 0+1
2026-11-21 00:17:31 UTC [B-worker codex-park] claim cards 0+1 ~28 min: mirror on park latency contexts 32768
2026-11-21 00:19:17 UTC [B-worker codex-park] release cards 0+1
2026-11-21 00:22:51 UTC [A-KF] released cards 0+1: prep-imm rc=0
2026-11-21 00:25:48 UTC [B-worker codex-park] claim cards 0+1 ~28 min: mirror off park latency contexts 32768
2026-11-21 00:29:10 UTC [A-H] claim card 0 ~5 min: hc3/check-chain.log
2026-11-21 00:34:20 UTC [B-worker codex-park] release cards 0+1
2026-11-21 00:38:24 UTC [B-worker codex-arfuse] claim cards 0+1 ~12 min: baseline fixed greedy gate and timings
2026-11-21 00:43:32 UTC [B-worker codex-vplan] claim cards 0+1 ~5 min: stamped C1/C4 verify timing and token gate
2026-11-21 00:44:30 UTC [A-R] claim cards 0+1 ~10 min: r5-micro-default-gate.log
2026-11-21 00:45:45 UTC [B-worker codex-park] release cards 0+1
2026-11-21 00:45:53 UTC [A-H] released card 0 (rc=0)
2026-11-21 00:48:59 UTC [B-worker codex-arfuse] release cards 0+1: baseline CPU quiet timeout, session rc=124, post-health healthy
2026-11-21 00:49:01 UTC [B-worker codex-park] claim cards 0+1 ~9 min: merged integration cache-on serve correctness and health
2026-11-21 00:49:07 UTC [B-worker codex-arfuse] claim cards 0+1 ~12 min: baseline fixed greedy gate and timings
2026-11-21 00:51:53 UTC [B-worker codex-vplan] release cards 0+1: stamped C1/C4 rc=1
2026-11-21 00:53:50 UTC [A-I] claim cards 0+1 ~3 min: smoke smoke-int-prep
2026-11-21 00:53:51 UTC [A-R] released cards 0+1 rc=0
2026-11-21 00:54:26 UTC [A-R] claim cards 0+1 ~10 min: r5-fwd-old-c4-gate.log
2026-11-21 00:56:25 UTC [B-worker codex-park] release cards 0+1
2026-11-21 00:58:35 UTC [B-worker codex-hostsplit] claim cards 0+1 ~15 min: host API and markers
2026-11-21 00:59:17 UTC [B-worker codex-arfuse] release cards 0+1: baseline rc=124
2026-11-21 01:02:15 UTC [B-worker codex-vplan] claim cards 0+1 ~5 min: stamped C1/C4 retry with CPU readiness check
2026-11-21 01:03:25 UTC [A-R] released cards 0+1 rc=0
2026-11-21 01:03:38 UTC [A-R] claim cards 0+1 ~10 min: r5-fwd-fixed-c4-gate.log
2026-11-21 01:10:13 UTC [B-worker codex-hostsplit] release cards 0+1 rc=0
2026-11-21 01:10:56 UTC [B-worker codex-vplan] release cards 0+1: stamped C1/C4 retry rc=1
2026-11-21 01:15:49 UTC [A-R] released cards 0+1 rc=0
2026-11-21 01:16:18 UTC [A-R] claim cards 0+1 ~10 min: r5-fwd-fixed-cycle-gate.log
2026-11-21 01:18:04 UTC [A-I] released cards 0+1
2026-11-21 01:18:23 UTC [B-worker codex-vplan] claim cards 0+1 ~5 min: stamped C1/C4 retry2 with CPU readiness check
2026-11-21 01:22:24 UTC [A-R] released cards 0+1 rc=0
2026-11-21 01:22:37 UTC [A-R] claim cards 0+1 ~10 min: r5-fwd-old-c4b-gate.log
2026-11-21 01:22:58 UTC [B-worker codex-park] claim cards 0+1 ~9 min: merged integration cache-on serve correctness and health
2026-11-21 01:23:10 UTC [B-worker codex-vplan] release cards 0+1: stamped C1/C4 retry2 rc=1
2026-11-21 01:25:17 UTC [B-worker codex-park] release cards 0+1
2026-11-21 01:27:20 UTC [A-R] released cards 0+1 rc=0
2026-11-21 01:33:26 UTC [A-H] claim card 0 ~5 min: hc3/timing.log
2026-11-21 01:33:57 UTC [A-H] claim cards 0+1 ~5 min: hc3/profile-selected.log
2026-11-21 01:34:24 UTC [A-KF] claim cards 0+1 ~3 min: smoke prep-r7
2026-11-21 01:36:18 UTC [A-H] claim cards 0+1 ~10 min: hc3/markers.log
2026-11-21 01:36:43 UTC [A-H] released cards 0+1 (rc=0)
2026-11-21 01:38:00 UTC [A-H] claim cards 0+1 ~4 min: smoke+HC/MTP math hc3-h
2026-11-21 01:39:05 UTC [A-KF] released cards 0+1
2026-11-21 01:39:05 UTC [A-H] released cards 0+1
2026-11-21 01:40:25 UTC [A-H] claim cards 0+1 ~4 min: smoke+HC/MTP math hc3-h2
2026-11-21 01:42:26 UTC [B-worker codex-vplan] claim cards 0+1 ~7 min: exclusive verify plan stamps with pinned fallback
2026-11-21 01:43:04 UTC [A-H] released cards 0+1
2026-11-21 01:43:33 UTC [B-worker codex-park] claim cards 0+1 ~9 min: merged integration cache-on serve correctness and health
2026-11-21 01:44:27 UTC [B-worker codex-arfuse] claim cards 0+1 ~12 min: baseline fixed greedy gate and timings
2026-11-21 01:47:26 UTC [B-worker codex-vplan] release cards 0+1: exclusive verify plan stamps rc=0
2026-11-21 01:48:41 UTC [B-worker codex-evpool] claim cards 0+1 ~25 min: event-pool comparison
2026-11-21 01:49:28 UTC [A-H] released queued timing/markers claims: cancelled before GPU acquisition to avoid idle CPU quiet hold
2026-11-21 01:49:29 UTC [A-S] claim cards 0+1 ~15 min: int-prep bounded policy serve/no-MTP/pre-post health; CPU quiet acquired before GPU
2026-11-21 01:49:48 UTC [A-H] claim card 0 ~5 min: hc3/timing.log
2026-11-21 01:50:05 UTC [B-worker codex-park] release cards 0+1
2026-11-21 01:50:59 UTC [A-H] claim cards 0+1 ~10 min: hc3/markers.log
2026-11-21 01:51:20 UTC [B-worker codex-ttft] claim cards 0+1 ~15 min: cold C4 and solo 8K baseline, phase stamps
2026-11-21 01:51:59 UTC [B-worker codex-arfuse] release cards 0+1: baseline rc=143
2026-11-21 01:54:45 UTC [A] REQUEST: integration serve_bench int-prep + int-prep-desc (fn-a3 bf8415b = hc2 + KF round 6 prepared inputs; smoke PASS 01:18) has waited 35 min for a both-card slot. Please let [A-S] in next (~6 min each). Lead B: this is the next ledger build.
2026-11-21 01:56:36 UTC [B-worker codex-ttft] release cards 0+1: cold C4 and solo baseline rc=124
2026-11-21 02:00:40 UTC [B] ACK orchestrator 02:00: [A-S] integration serve_bench goes first, then the B ledger regen on that build, then B workers in short leases. B workers (codex-park, codex-kv8/ttft, codex-pushar/arfuse, codex-trace/evpool): after your current lease, check this file and do not claim while an [A-S] claim is pending.
2026-11-21 02:02:35 UTC [B-worker codex-evpool] release cards 0+1 rc=0
2026-11-21 02:03:08 UTC [B-worker codex-arfuse] claim cards 0+1 ~12 min: baseline fixed greedy gate and timings
2026-11-21 02:06:14 UTC [A-S] released cards 0+1: int-prep rc=0
2026-11-21 02:06:15 UTC [B-worker codex-arfuse] release cards 0+1: baseline rc=75
2026-11-21 02:06:17 UTC [A-I] claim cards 0+1 ~3 min: smoke r5-smoke
2026-11-21 02:10:15 UTC [A] int-prep serve_bench DONE on fn-int bf8415b (C1 71.7 / C4 123.4 / C8 146.2 / prefill 2k 484). Lead B: ledger regen on bf8415b is next. [A-S] runs one more integration bench now (int-prep-desc, XPU_FN_DESC=1, ~6 min, priority).
2026-11-21 02:10:32 UTC [A-KF] claim cards 0+1 ~14 min: R7 paired markers pair, CPU quiet before GPU, record off
2026-11-21 02:11:24 UTC [B-worker codex-park] claim cards 0+1 ~9 min: physical-slot C6 identity and health
2026-11-21 02:13:51 UTC [B-worker codex-park] release cards 0+1
2026-11-21 02:15:10 UTC [A-KF] released cards 0+1: R7 markers pair rc=0
2026-11-21 02:15:10 UTC [A-H] claim cards 0+1 ~15 min: hc3 bounded policy serve/no-MTP/pre-post health; CPU quiet acquired before GPU
2026-11-21 02:15:27 UTC [B-worker codex-arfuse] claim cards 0+1 ~12 min: baseline fixed greedy gate and timings
2026-11-21 02:18:51 UTC [A-H] released cards 0+1: hc3 rc=0
2026-11-21 02:18:54 UTC [A-R] NOTE: smoke r5-smoke (A-I tagged claim 02:06:17, filed by A-R) was cancelled while still queued for cpu-run quiet; never acquired the GPUs. Rebuilding for the immediate-drain commit.
2026-11-21 02:18:55 UTC [A-S] claim cards 0+1 ~15 min: int-prep-desc bounded policy serve/no-MTP/pre-post health; CPU quiet acquired before GPU
2026-11-21 02:19:37 UTC [B-worker codex-arfuse] release cards 0+1: baseline rc=143
2026-11-21 02:23:11 UTC [A-S] released cards 0+1: int-prep-desc rc=0
2026-11-21 02:27:33 UTC [B-worker codex-park] claim cards 0+1 ~9 min: physical-slot C6 identity and health
2026-11-21 02:30:15 UTC [B-worker codex-park] release cards 0+1
2026-11-21 02:30:29 UTC [B-worker codex-park] claim cards 0+1 ~25 min: slots-c4-control C4 P=auto quantum=4
2026-11-21 02:33:47 UTC [A-KF] claim cards 0+1 ~15 min: prep2 bounded policy serve/no-MTP/pre-post health; CPU quiet acquired before GPU
2026-11-21 02:34:08 UTC [B-worker codex-park] release cards 0+1
2026-11-21 02:34:24 UTC [B-worker codex-park] claim cards 0+1 ~25 min: slots-c8-auto C8 P=auto quantum=4
2026-11-21 02:34:26 UTC [A-H] claim card 0 ~1 min: queued hc3 real-weight draft microbench; waiting for CPU quiet, no lease held
2026-11-21 02:34:26 UTC [A-H] claim cards 0+1 ~4 min: queued hc3 matched host.fn.mtp markers and final health; no lease held; smoke/serve already passed
2026-11-21 02:37:42 UTC [A-KF] released cards 0+1: prep2 rc=0
2026-11-21 02:41:09 UTC [B-worker codex-park] release cards 0+1
2026-11-21 02:41:31 UTC [B-worker codex-park] claim cards 0+1 ~25 min: slots-c16-auto C16 P=auto quantum=4
2026-11-21 02:42:51 UTC [A-PF] claim cards 0+1 ~12 min: prefill profile pf0 (64:0 64:1 512:0 512:1)
2026-11-21 02:43:59 UTC [B-worker codex-arfuse] claim cards 0+1 ~12 min: baseline fixed greedy gate and timings
2026-11-21 02:45:46 UTC [B-worker codex-park] release cards 0+1
2026-11-21 02:46:04 UTC [B-worker codex-park] claim cards 0+1 ~25 min: slots-c8-p2-q1 C8 P=2 quantum=1
2026-11-21 02:46:56 UTC [A-I] claim cards 0+1 ~3 min: smoke smoke-int-desc
2026-11-21 02:53:22 UTC [A-PF] released cards 0+1: pf0 rc=1
2026-11-21 02:55:35 UTC [B-worker codex-arfuse] release cards 0+1: baseline rc=0
2026-11-21 02:55:43 UTC [B-worker codex-arfuse] claim cards 0+1 ~12 min: candidate-gate fixed greedy gate and timings
2026-11-21 02:59:43 UTC [A-KF] claim card 0 ~5 min: projection packing bitwise row-position/canary gate
2026-11-21 03:00:10 UTC [B-worker codex-park] release cards 0+1
2026-11-21 03:00:15 UTC [B-worker codex-arfuse] release cards 0+1: candidate-gate rc=75
2026-11-21 03:00:24 UTC [B-worker codex-park] claim cards 0+1 ~25 min: slots-c16-p2-q1 C16 P=2 quantum=1
2026-11-21 03:00:29 UTC [A-H] released card 0 (rc=0)
2026-11-21 03:00:58 UTC [A-KF] released card 0: projection packing gate rc=0
2026-11-21 03:00:58 UTC [A-KF] claim cards 0+1 ~3 min: smoke prep-r7-pack
2026-11-21 03:05:02 UTC [B-worker codex-arfuse] claim cards 0+1 ~12 min: candidate-gate fixed greedy gate and timings
2026-11-21 03:07:29 UTC [B-worker codex-park] release cards 0+1
2026-11-21 03:07:48 UTC [B-worker codex-park] claim cards 0+1 ~25 min: slots-c16-p2-q4 C16 P=2 quantum=4
2026-11-21 03:09:59 UTC [B-worker codex-park] claim cards 0+1 ~25 min: slots-c8-p2-q4 C8 P=2 quantum=4
2026-11-21 03:10:13 UTC [A-KF] claim card 0 ~14 min: all-type fp64 parity and bitwise invariance, both activation paths plus reference
2026-11-21 03:11:37 UTC [A-H] released cards 0+1 (rc=0)
2026-11-21 03:13:58 UTC [A-KF] released cards 0+1
2026-11-21 03:14:03 UTC [B-worker codex-arfuse] release cards 0+1: candidate-gate rc=75
2026-11-21 03:14:38 UTC [A] PRIORITY (orchestrator 02:00): integration smoke-int-desc ([A-I] tag, ~3 min) + serve_bench int-desc ([A-S], ~6 min) on fn-a3 6d92e0f have queued since 02:51. Workers and B-workers: hold new claims until [A-S] releases.
2026-11-21 03:18:42 UTC [B-worker codex-park] release cards 0+1
2026-11-21 03:21:59 UTC [A-I] claim cards 0+1 ~3 min: smoke r5-smoke
2026-11-21 03:22:15 UTC [B-worker codex-park] release cards 0+1
2026-11-21 03:23:56 UTC [A-KF] released card 0: full parity/invariance rc=0
2026-11-21 03:24:15 UTC [B-worker codex-arfuse] claim cards 0+1 ~12 min: candidate-gate fixed greedy gate and timings
2026-11-21 03:26:27 UTC [A-I] released cards 0+1
2026-11-21 03:27:54 UTC [A] PRIORITY.hold created (/mnt/vm_8tb/b70/results/tf-flashnext/PRIORITY.hold) for int-desc smoke+bench. ALL workers: no new gpu-run while it exists (while [ -e PRIORITY.hold ]; do sleep 10; done). Auto-removed when [A-S] releases int-desc.
2026-11-21 03:28:29 UTC [B] PRIORITY.hold SET ('/mnt/vm_8tb/b70/results/tf-flashnext/PRIORITY.hold', allows A-S A-I). bin/gpu-run and bin/cpu-run now wait on it for every other B70_AGENT before taking a NEW lease; mid-lease runs finish. Auto-removed when [A-S] logs released (watcher), stale after 60 min. Lead A: future integration benches: echo 'A-S A-I' > PRIORITY.hold while queued, rm after.
2026-11-21 03:28:48 UTC [B] NOTE: at 03:27 Lead B killed a QUEUED cpu-run waiter pid 3634660 (zig build xpu-native -j4, cwd unknown) believing it was its own; if a worker's build failed around 03:27 with a signal, just rerun it. Lead B also cancelled its own evpool compile check (cpu lease freed for the integration smoke).
2026-11-21 03:28:59 UTC [B-worker codex-arfuse] release cards 0+1: candidate-gate rc=0
2026-11-21 03:29:13 UTC [B-worker codex-arfuse] claim cards 0+1 ~12 min: baseline fixed greedy gate and timings
2026-11-21 03:30:17 UTC [B] Freed the CPU lock: an orphan zig child of Lead B's cancelled compile kept fd 7 (killed). Also cancelled two queued B-worker builds (codex-park fn-rt-evict, codex-trace fn-rt-ledger) so they re-queue behind PRIORITY.hold.
2026-11-21 03:31:18 UTC [A-R] NOTE: smoke r5-smoke (A-I tagged claim 02:51-ish, filed by A-R) cancelled while queued for cpu-run quiet, never held GPUs; requeued behind PRIORITY.hold.
2026-11-21 03:34:54 UTC [A-S] claim cards 0+1 ~15 min: int-desc bounded policy serve/no-MTP/pre-post health; CPU quiet acquired before GPU
2026-11-21 03:38:31 UTC [A-S] released cards 0+1: int-desc rc=0
2026-11-21 03:38:31 UTC [B] PRIORITY.hold removed ([A-S] released)
2026-11-21 03:38:39 UTC [A-PF] claim cards 0+1 ~13 min: prefill profile pf1 (64:1 512:0 512:1 2048:1)
2026-11-21 03:41:17 UTC [B-worker codex-arfuse] release cards 0+1: baseline rc=124
2026-11-21 03:42:44 UTC [B-worker codex-arfuse] claim cards 0+1 ~12 min: baseline fixed greedy gate and timings
2026-11-21 03:43:31 UTC [B-worker codex-ledger] claim cards 0+1 ~20 min: descriptor ledger evpool off/on, full on only
2026-11-21 03:44:28 UTC [A-KF] claim cards 0+1 ~9 min (bound 14.7): R7 three-arm markers, record off
2026-11-21 03:44:58 UTC [B] USER DIRECTIVE 03:45 applied: all four B workers (codex-kv8/ttft, codex-pushar/arfuse, codex-park/c16, codex-trace/regen) interrupted and resumed with: leases <= 5 min (most 1-3), quick gate (2 prompts x 32 tok + C2==solo), short bench, health once per session.
2026-11-21 03:45:20 UTC [B-worker codex-arfuse] release cards 0+1: interrupted queued baseline request, no lease acquired
2026-11-21 03:45:39 UTC [B-worker codex-ledger] release cards 0+1: queued claim interrupted; no lease acquired
2026-11-21 03:45:45 UTC [B-worker codex-ttft] claim cards 0+1 ~5 min: cold 8K C4 and C1 baseline
2026-11-21 03:46:35 UTC [B-worker codex-ledger] claim cards 0+1 ~5 min: int-bf8415b markers
2026-11-21 03:46:58 UTC [B-worker codex-arfuse] claim cards 0+1 ~5 min: c2-gate gate
2026-11-21 03:47:42 UTC [A-PF] released cards 0+1: pf1 rc=0
2026-11-21 03:52:01 UTC [B-worker codex-ttft] release cards 0+1: cold baseline rc=0
2026-11-21 03:54:30 UTC [B-worker codex-park] claim cards 0+1 ~5 min: post-merge two-prompt 32-token C2 gate
2026-11-21 03:56:03 UTC [B-worker codex-ledger] release cards 0+1 int-bf8415b markers rc=0
2026-11-21 03:56:03 UTC [B-worker codex-ledger] claim cards 0+1 ~5 min: int-bf8415b-evpool markers
2026-11-21 03:58:35 UTC [B-worker codex-arfuse] release cards 0+1: c2-gate rc=0
2026-11-21 03:58:46 UTC [B-worker codex-arfuse] claim cards 0+1 ~5 min: perf-off bench
2026-11-21 04:01:07 UTC [B-worker codex-park] release cards 0+1
2026-11-21 04:03:24 UTC [B-worker codex-arfuse] release cards 0+1: perf-off rc=124
2026-11-21 04:04:09 UTC [B-worker codex-ledger] release cards 0+1 int-bf8415b-evpool markers rc=0
2026-11-21 04:04:09 UTC [B-worker codex-ledger] claim cards 0+1 ~5 min: int-bf8415b-evpool full
2026-11-21 04:06:57 UTC [B] 04:10 value check: codex-ttft (=codex-kv8) and codex-ledger PAUSED, codex-park idle; only codex-arfuse may claim (<=5 min, yields to PRIORITY.hold).
2026-11-21 04:07:57 UTC [A] PRIORITY.hold ACTIVE since 04:03 for merge train 1 (fn-a3 3edbc4d) quick benches train1-imm then train1-rec (~3-4 min each). All workers: no new claims until it is removed.
2026-11-21 04:12:17 UTC [A-S] claim cards 0+1 ~15 min: train1-imm bounded policy serve/no-MTP/pre-post health; CPU quiet acquired before GPU
2026-11-21 04:15:02 UTC [A-S] released cards 0+1: train1-imm rc=0
2026-11-21 04:15:03 UTC [A-S] claim cards 0+1 ~15 min: train1-rec bounded policy serve/no-MTP/pre-post health; CPU quiet acquired before GPU
2026-11-21 04:15:07 UTC [B-worker codex-arfuse] claim cards 0+1 ~5 min: perf-off bench
2026-11-21 04:17:25 UTC [A-S] released cards 0+1: train1-rec rc=1
2026-11-21 04:17:26 UTC [A-KF] claim cards 0+1 ~3 min (max 4m50s): R7 short markers zero
2026-11-21 04:17:32 UTC [A-H] claim cards 0+1 ~5 min: hc4/core-gate.log
2026-11-21 04:17:34 UTC [A-PF] claim cards 0+1 ~5 min: prefill profile pf2 (64:0 64:1 512:0 512:1 2048:0 2048:1)
2026-11-21 04:20:27 UTC [A-KF] released cards 0+1: R7 short markers zero rc=143
2026-11-21 04:21:36 UTC [A-H] released queued core gate claim before GPU acquisition: new PRIORITY.hold; requeue guarded
2026-11-21 04:23:11 UTC [B-worker codex-arfuse] release cards 0+1: perf-off rc=75
2026-11-21 04:23:57 UTC [B-worker codex-arfuse] claim cards 0+1 ~5 min: perf-off bench
2026-11-21 04:26:11 UTC [A-PF] released cards 0+1: pf2 rc=0
2026-11-21 04:26:12 UTC [A-S] claim cards 0+1 ~15 min: train1-rec-noevpool bounded policy serve/no-MTP/pre-post health; CPU quiet acquired before GPU
2026-11-21 04:29:02 UTC [A-S] released cards 0+1: train1-rec-noevpool rc=0
2026-11-21T04:29:06Z [A-PF] claim cards 0+1 ~3 min: smoke gate d8baa8d
2026-11-21 04:29:06 UTC [A-I] claim cards 0+1 ~3 min: smoke pf-d8baa8d
2026-11-21 04:31:45 UTC [B-worker codex-arfuse] release cards 0+1: perf-off rc=75
2026-11-21 04:34:02 UTC [A-I] released cards 0+1
2026-11-21T04:34:02Z [A-PF] release cards 0+1: smoke gate d8baa8d
2026-11-21 04:34:11 UTC [A-H] claim cards 0+1 ~5 min: hc4/core-d3-a1.log
2026-11-21 04:34:14 UTC [B-worker codex-ledger] release cards 0+1: interrupted full capture while queued; no lease acquired
2026-11-21 04:37:08 UTC [A-H] released cards 0+1 (rc=75)
2026-11-21 04:39:15 UTC [B-worker codex-arfuse] claim cards 0+1 ~5 min: ext-gate gate
2026-11-21 04:41:44 UTC [B-worker codex-arfuse] release cards 0+1: ext-gate rc=0
2026-11-21 04:48:37 UTC [B-worker codex-ledger] claim cards 0+1 ~5 min: train1-imm C1/C4 markers plus full
2026-11-21 04:49:23 UTC [A-R] claim cards 0+1 ~15 min: mtp-q2-t256 bounded policy serve/no-MTP/pre-post health; CPU quiet acquired before GPU
2026-11-21 04:52:43 UTC [B-worker codex-ledger] release cards 0+1 train1-imm rc=0
2026-11-21 04:53:05 UTC [B-worker codex-arfuse] claim cards 0+1 ~5 min: perf-off bench
2026-11-21 04:57:57 UTC [B-worker codex-arfuse] release cards 0+1: perf-off rc=75
2026-11-21 04:58:03 UTC [A-PF] claim card 1 ~120s: gemm1
2026-11-21 05:00:58 UTC [A-R] released cards 0+1: mtp-q2-t256 rc=0
2026-11-21 05:01:00 UTC [A-PF] release card 1: gemm1 rc=0
2026-11-21 05:03:55 UTC [A-H] claim cards 0+1 ~5 min: hc4/depth3-a1.log
2026-11-21 05:05:13 UTC [A-S] claim cards 0+1 ~15 min: int-pf-512 bounded policy serve/no-MTP/pre-post health; CPU quiet acquired before GPU
2026-11-21 05:06:25 UTC [A-H] released cards 0+1 (rc=75)
2026-11-21 05:09:04 UTC [A-S] released cards 0+1: int-pf-512 rc=0
2026-11-21 05:09:04 UTC [A-R] claim cards 0+1 ~15 min: mtp-q2-d4 bounded policy serve/no-MTP/pre-post health; CPU quiet acquired before GPU
2026-11-21 05:10:57 UTC [B-worker codex-arfuse] claim cards 0+1 ~5 min: perf-off bench
2026-11-21 05:15:42 UTC [A-H] released-session note: core156e010 quick gate PASS (tipca25bbe); H owns no leases. HC4 timing blocked: GPU owner files empty, CPU quiet A-R mtp-q2-d4 pending gpu-run behind PRIORITY.hold A-S/A-I.
2026-11-21 05:16:46 UTC [A-R] released queued mtp-q2-d4 claim before GPU acquisition (it held cpu-run --quiet behind PRIORITY.hold); no lease acquired
2026-11-21 05:21:39 UTC [B-worker codex-keep] claim cards 0+1 ~5 min: prepared keep gate and C1/C4 off/on
2026-11-21 05:26:10 UTC [A-S] claim cards 0+1 ~15 min: int-pf-2048 bounded policy serve/no-MTP/pre-post health; CPU quiet acquired before GPU
2026-11-21 05:29:11 UTC [A-S] released cards 0+1: int-pf-2048 rc=0
2026-11-21 05:29:14 UTC [O-HW] claim card 1 ~3 min: hwprobe smoke
2026-11-21 05:29:16 UTC [A-PF] claim card 1 ~120s: gemm2
2026-11-21 05:30:51 UTC [B-worker codex-arfuse] claim cards 0+1 ~5 min: perf-off bench
2026-11-21 05:32:49 UTC [B-worker codex-keep] release cards 0+1 rc=0
2026-11-21 05:32:50 UTC [O-HW] release card 1: hwprobe smoke rc=0
2026-11-21 05:32:53 UTC [A-PF] release card 1: gemm2 rc=0
2026-11-21 05:33:03 UTC [A-PF] claim cards 0+1 ~5 min: prefill profile pf3 (512:1 2048:1)
2026-11-21 05:34:18 UTC [O-HW] claim card 1 ~3 min: hwprobe main
2026-11-21 05:35:20 UTC [A-H] claim cards 0+1 ~5 min: hc4/depth3q-a1.log
2026-11-21 05:35:30 UTC [B-worker codex-arfuse] release cards 0+1: perf-off rc=75
2026-11-21 05:36:17 UTC [B-worker codex-lane] claim cards 0+1 ~5 min: single-server MTP lane depth sweep
2026-11-21 05:38:12 UTC [B-worker codex-arfuse] release cards 0+1: perf-off rc=75
2026-11-21 05:40:05 UTC [A-H] queued depth3 claim resource note: only ~20s CPU quiet is needed after model ready; no CPU quiet during load/waits. Depth1/depth2 follow as separate <=5m leases. Core quick gate already PASS; no repeats.
2026-11-21 05:40:30 UTC [A-PF] released cards 0+1: pf3 rc=0
2026-11-21 05:40:53 UTC [A-KF] claim cards 0+1 ~3 min (max 4m50s): kf8-merge
2026-11-21 05:40:54 UTC [A-KF] released cards 0+1: kf8-merge rc=124
2026-11-21 05:40:56 UTC [A-KF] claim cards 0+1 ~3 min (max 4m50s): kf8-merge
2026-11-21 05:40:57 UTC [A-KF] released cards 0+1: kf8-merge rc=124
2026-11-21 05:40:59 UTC [A-KF] claim cards 0+1 ~3 min (max 4m50s): kf8-merge
2026-11-21 05:41:00 UTC [A-KF] released cards 0+1: kf8-merge rc=124
2026-11-21 05:41:02 UTC [A-KF] claim cards 0+1 ~3 min (max 4m50s): kf8-merge
2026-11-21 05:41:03 UTC [A-KF] released cards 0+1: kf8-merge rc=124
2026-11-21T05:41:05Z [A-PF] claim cards 0+1 ~3 min: smoke gate 7973882
2026-11-21 05:41:05 UTC [A-KF] claim cards 0+1 ~3 min (max 4m50s): kf8-merge
2026-11-21 05:41:05 UTC [A-I] claim cards 0+1 ~3 min: smoke pf-7973882
2026-11-21 05:41:06 UTC [A-KF] released cards 0+1: kf8-merge rc=124
2026-11-21 05:41:08 UTC [A-KF] claim cards 0+1 ~3 min (max 4m50s): kf8-merge
2026-11-21 05:41:09 UTC [A-KF] released cards 0+1: kf8-merge rc=124
2026-11-21 05:41:11 UTC [A-KF] claim cards 0+1 ~3 min (max 4m50s): kf8-merge
2026-11-21 05:41:12 UTC [A-KF] released cards 0+1: kf8-merge rc=124
2026-11-21 05:41:14 UTC [A-KF] claim cards 0+1 ~3 min (max 4m50s): kf8-merge
2026-11-21 05:41:15 UTC [A-KF] released cards 0+1: kf8-merge rc=124
2026-11-21 05:41:17 UTC [A-KF] claim cards 0+1 ~3 min (max 4m50s): kf8-merge
2026-11-21 05:41:18 UTC [A-KF] released cards 0+1: kf8-merge rc=124
2026-11-21 05:41:20 UTC [A-KF] claim cards 0+1 ~3 min (max 4m50s): kf8-merge
2026-11-21 05:41:21 UTC [A-KF] released cards 0+1: kf8-merge rc=124
2026-11-21 05:41:23 UTC [A-KF] claim cards 0+1 ~3 min (max 4m50s): kf8-merge
2026-11-21 05:41:24 UTC [A-KF] released cards 0+1: kf8-merge rc=124
2026-11-21 05:41:26 UTC [A-KF] claim cards 0+1 ~3 min (max 4m50s): kf8-merge
2026-11-21 05:41:27 UTC [A-KF] released cards 0+1: kf8-merge rc=124
2026-11-21 05:41:29 UTC [A-KF] claim cards 0+1 ~3 min (max 4m50s): kf8-merge
2026-11-21 05:41:30 UTC [A-KF] released cards 0+1: kf8-merge rc=124
2026-11-21 05:41:32 UTC [A-KF] claim cards 0+1 ~3 min (max 4m50s): kf8-merge
2026-11-21 05:41:33 UTC [A-KF] released cards 0+1: kf8-merge rc=124
2026-11-21 05:41:35 UTC [A-KF] claim cards 0+1 ~3 min (max 4m50s): kf8-merge
2026-11-21 05:41:36 UTC [A-KF] released cards 0+1: kf8-merge rc=124
2026-11-21 05:41:38 UTC [A-KF] claim cards 0+1 ~3 min (max 4m50s): kf8-merge
2026-11-21 05:41:40 UTC [A-KF] released cards 0+1: kf8-merge rc=124
2026-11-21 05:41:42 UTC [A-KF] claim cards 0+1 ~3 min (max 4m50s): kf8-merge
2026-11-21 05:41:43 UTC [A-KF] released cards 0+1: kf8-merge rc=124
2026-11-21 05:41:45 UTC [A-KF] claim cards 0+1 ~3 min (max 4m50s): kf8-merge
2026-11-21 05:41:46 UTC [A-KF] released cards 0+1: kf8-merge rc=124
2026-11-21 05:41:48 UTC [A-KF] claim cards 0+1 ~3 min (max 4m50s): kf8-merge
2026-11-21 05:41:49 UTC [A-KF] released cards 0+1: kf8-merge rc=124
2026-11-21 05:41:51 UTC [A-KF] claim cards 0+1 ~3 min (max 4m50s): kf8-merge
2026-11-21 05:41:52 UTC [A-KF] released cards 0+1: kf8-merge rc=124
2026-11-21 05:41:54 UTC [A-KF] claim cards 0+1 ~3 min (max 4m50s): kf8-merge
2026-11-21 05:41:55 UTC [A-KF] released cards 0+1: kf8-merge rc=124
2026-11-21 05:41:57 UTC [A-KF] claim cards 0+1 ~3 min (max 4m50s): kf8-merge
2026-11-21 05:41:58 UTC [A-KF] released cards 0+1: kf8-merge rc=124
2026-11-21 05:42:00 UTC [A-KF] claim cards 0+1 ~3 min (max 4m50s): kf8-merge
2026-11-21 05:42:01 UTC [A-KF] released cards 0+1: kf8-merge rc=124
2026-11-21 05:42:03 UTC [A-KF] claim cards 0+1 ~3 min (max 4m50s): kf8-merge
2026-11-21 05:42:04 UTC [A-KF] released cards 0+1: kf8-merge rc=124
2026-11-21 05:42:06 UTC [A-KF] claim cards 0+1 ~3 min (max 4m50s): kf8-merge
2026-11-21 05:42:07 UTC [A-KF] released cards 0+1: kf8-merge rc=124
2026-11-21 05:42:09 UTC [A-KF] claim cards 0+1 ~3 min (max 4m50s): kf8-merge
2026-11-21 05:42:10 UTC [A-KF] released cards 0+1: kf8-merge rc=124
2026-11-21 05:42:12 UTC [A-KF] claim cards 0+1 ~3 min (max 4m50s): kf8-merge
2026-11-21 05:42:13 UTC [A-KF] released cards 0+1: kf8-merge rc=124
2026-11-21 05:42:15 UTC [A-KF] claim cards 0+1 ~3 min (max 4m50s): kf8-merge
2026-11-21 05:42:16 UTC [A-KF] released cards 0+1: kf8-merge rc=124
2026-11-21 05:42:18 UTC [A-KF] claim cards 0+1 ~3 min (max 4m50s): kf8-merge
2026-11-21 05:42:19 UTC [A-KF] released cards 0+1: kf8-merge rc=124
2026-11-21 05:42:21 UTC [A-KF] claim cards 0+1 ~3 min (max 4m50s): kf8-merge
2026-11-21 05:42:22 UTC [A-KF] released cards 0+1: kf8-merge rc=124
2026-11-21 05:42:24 UTC [A-KF] claim cards 0+1 ~3 min (max 4m50s): kf8-merge
2026-11-21 05:42:25 UTC [A-KF] released cards 0+1: kf8-merge rc=124
2026-11-21 05:42:27 UTC [A-KF] claim cards 0+1 ~3 min (max 4m50s): kf8-merge
2026-11-21 05:42:28 UTC [A-KF] released cards 0+1: kf8-merge rc=124
2026-11-21 05:42:30 UTC [A-KF] claim cards 0+1 ~3 min (max 4m50s): kf8-merge
2026-11-21 05:42:31 UTC [A-KF] released cards 0+1: kf8-merge rc=124
2026-11-21 05:42:33 UTC [A-KF] claim cards 0+1 ~3 min (max 4m50s): kf8-merge
2026-11-21 05:42:35 UTC [A-KF] released cards 0+1: kf8-merge rc=124
2026-11-21 05:42:37 UTC [A-KF] claim cards 0+1 ~3 min (max 4m50s): kf8-merge
2026-11-21 05:42:38 UTC [A-KF] released cards 0+1: kf8-merge rc=124
2026-11-21 05:42:40 UTC [A-KF] claim cards 0+1 ~3 min (max 4m50s): kf8-merge
2026-11-21 05:42:41 UTC [A-KF] released cards 0+1: kf8-merge rc=124
2026-11-21 05:42:43 UTC [A-KF] claim cards 0+1 ~3 min (max 4m50s): kf8-merge
2026-11-21 05:42:44 UTC [A-KF] released cards 0+1: kf8-merge rc=124
2026-11-21 05:42:46 UTC [A-KF] claim cards 0+1 ~3 min (max 4m50s): kf8-merge
2026-11-21 05:42:47 UTC [A-KF] released cards 0+1: kf8-merge rc=124
2026-11-21 05:42:49 UTC [A-KF] claim cards 0+1 ~3 min (max 4m50s): kf8-merge
2026-11-21 05:42:50 UTC [A-KF] released cards 0+1: kf8-merge rc=124
2026-11-21 05:42:52 UTC [A-KF] claim cards 0+1 ~3 min (max 4m50s): kf8-merge
2026-11-21 05:42:53 UTC [A-KF] released cards 0+1: kf8-merge rc=124
2026-11-21 05:42:55 UTC [A-H] released cards 0+1 (rc=75)
2026-11-21 05:42:55 UTC [A-KF] claim cards 0+1 ~3 min (max 4m50s): kf8-merge
2026-11-21 05:42:56 UTC [A-KF] released cards 0+1: kf8-merge rc=124
2026-11-21 05:42:58 UTC [A-KF] claim cards 0+1 ~3 min (max 4m50s): kf8-merge
2026-11-21 05:42:59 UTC [A-KF] released cards 0+1: kf8-merge rc=124
2026-11-21 05:43:01 UTC [A-KF] claim cards 0+1 ~3 min (max 4m50s): kf8-merge
2026-11-21 05:43:02 UTC [A-KF] released cards 0+1: kf8-merge rc=124
2026-11-21 05:43:04 UTC [A-KF] claim cards 0+1 ~3 min (max 4m50s): kf8-merge
2026-11-21 05:43:05 UTC [A-H] claim cards 0+1 ~5 min: hc4/depth3q-a2.log
2026-11-21 05:43:05 UTC [A-KF] released cards 0+1: kf8-merge rc=124
2026-11-21 05:43:07 UTC [A-KF] claim cards 0+1 ~3 min (max 4m50s): kf8-merge
2026-11-21 05:43:08 UTC [A-KF] released cards 0+1: kf8-merge rc=124
2026-11-21 05:43:10 UTC [A-KF] claim cards 0+1 ~3 min (max 4m50s): kf8-merge
2026-11-21 05:43:11 UTC [A-KF] released cards 0+1: kf8-merge rc=124
2026-11-21 05:43:13 UTC [A-KF] claim cards 0+1 ~3 min (max 4m50s): kf8-merge
2026-11-21 05:43:14 UTC [A-KF] released cards 0+1: kf8-merge rc=124
2026-11-21 05:43:16 UTC [A-KF] claim cards 0+1 ~3 min (max 4m50s): kf8-merge
2026-11-21 05:43:17 UTC [A-KF] released cards 0+1: kf8-merge rc=124
2026-11-21 05:43:19 UTC [A-KF] claim cards 0+1 ~3 min (max 4m50s): kf8-merge
2026-11-21 05:43:20 UTC [A-KF] released cards 0+1: kf8-merge rc=124
2026-11-21 05:43:22 UTC [A-KF] claim cards 0+1 ~3 min (max 4m50s): kf8-merge
2026-11-21 05:43:23 UTC [A-KF] released cards 0+1: kf8-merge rc=124
2026-11-21 05:43:25 UTC [A-KF] claim cards 0+1 ~3 min (max 4m50s): kf8-merge
2026-11-21 05:43:26 UTC [A-KF] released cards 0+1: kf8-merge rc=124
2026-11-21 05:43:28 UTC [A-KF] claim cards 0+1 ~3 min (max 4m50s): kf8-merge
2026-11-21 05:43:29 UTC [A-KF] released cards 0+1: kf8-merge rc=124
2026-11-21 05:43:32 UTC [A-KF] claim cards 0+1 ~3 min (max 4m50s): kf8-merge
2026-11-21 05:43:33 UTC [A-KF] released cards 0+1: kf8-merge rc=124
2026-11-21 05:43:35 UTC [A-KF] claim cards 0+1 ~3 min (max 4m50s): kf8-merge
2026-11-21 05:43:36 UTC [A-KF] released cards 0+1: kf8-merge rc=124
2026-11-21 05:43:38 UTC [A-KF] claim cards 0+1 ~3 min (max 4m50s): kf8-merge
2026-11-21 05:43:39 UTC [A-KF] released cards 0+1: kf8-merge rc=124
2026-11-21 05:43:40 UTC [A-KF] R8 merge gate polling correction: prior attempts were lock-acquisition timeouts only; no GPU lease or client acquired. Waiting on owner files outside all locks.
2026-11-21 05:43:57 UTC [A-H] released queued depth3q-a2 before GPU acquisition: dispatcher replaced with CPU-readiness guard; no workload started.
2026-11-21 05:45:36 UTC [A-HS] claim card 0 ~3 min: cold1.log
2026-11-21 05:45:59 UTC [B-worker codex-lane] release cards 0+1
2026-11-21 05:46:00 UTC [A-HS] released card 0 (rc=0): cold1.log
2026-11-21 05:46:15 UTC [O-HW] release card 1: hwprobe main rc=0
2026-11-21 05:46:15 UTC [A-KF] claim cards 0+1 ~3 min (max 4m50s): kf8-merge
2026-11-21 05:46:29 UTC [B-worker codex-arfuse] claim cards 0+1 ~5 min: perf-off bench
2026-11-21 05:46:41 UTC [B-worker codex-arfuse] release cards 0+1: perf-off rc=124
2026-11-21 05:46:51 UTC [B-worker codex-arfuse] claim cards 0+1 ~5 min: perf-off bench
2026-11-21 05:47:13 UTC [B-worker codex-lane] claim cards 0+1 ~5 min: existing C2 policy control
2026-11-21 05:48:37 UTC [A-KF] released cards 0+1: kf8-merge rc=0
2026-11-21 05:49:05 UTC [O-HW] claim card 1 ~3 min: hwprobe main-v2
2026-11-21 05:49:37 UTC [A-R] claim cards 0+1 ~3 min: Strata oracle draftdbg strata-s2
2026-11-21 05:50:42 UTC [A-H] timing dependency: H4 core156e010 merge-ready. Need one ~3m GPU session with CPU build-dispatch gap through model load and ~20s timed client (CPU quiet only during client). Current queued builders have prevented every timed attempt; H owns no leases.
2026-11-21 05:50:51 UTC [B-worker codex-route] claim cards 0+1 ~5 min: C1 C4 C8 routing ids
2026-11-21 05:51:11 UTC [B-worker codex-arfuse] release cards 0+1: perf-off rc=75
2026-11-21 05:51:20 UTC [B-worker codex-arfuse] claim cards 0+1 ~5 min: perf-off bench
2026-11-21 05:53:33 UTC [B-worker codex-lane] release cards 0+1
2026-11-21 05:55:05 UTC [B-worker codex-lane] claim cards 0+1 ~5 min: reset and health after timed-out card-1 health probe
2026-11-21 05:56:19 UTC [A-I] released cards 0+1
2026-11-21T05:56:19Z [A-PF] release cards 0+1: smoke gate 7973882
2026-11-21 05:58:43 UTC [B-worker codex-route] release cards 0+1 rc=0
2026-11-21 06:00:10 UTC [B-worker codex-lane] release cards 0+1
2026-11-21 06:00:36 UTC [A-R] released cards 0+1: Strata oracle draftdbg strata-s2 rc=0
2026-11-21 06:01:13 UTC [A-HS] claim card 1 ~3 min: cold2.log
2026-11-21 06:01:27 UTC [O-HW] release card 1: hwprobe main-v2 rc=0
2026-11-21 06:01:27 UTC [A-HS] released card 1 (rc=0): cold2.log
2026-11-21 06:01:40 UTC [A-I] claim cards 0+1 ~3 min: smoke hs-once-cbb488c
2026-11-21 06:01:58 UTC [O-HW] claim card 0 ~3 min: hwprobe main-v2
2026-11-21 06:04:38 UTC [B-worker codex-route] claim cards 0+1 ~3 min: distinct-prompt C8 routing
2026-11-21 06:06:14 UTC [B-worker codex-arfuse] claim cards 0+1 ~5 min: pinned fusion off/on A/B
2026-11-21 06:06:56 UTC [B-worker codex-route] release cards 0+1 distinct C8 rc=0
2026-11-21 06:10:31 UTC [A-PF] claim card 1 ~120s: gemm3
2026-11-21 06:11:25 UTC [B-worker codex-arfuse] release cards 0+1: pinned fusion off/on A/B rc=0
2026-11-21 06:11:26 UTC [A-PF] release card 1: gemm3 rc=0
2026-11-21 06:11:26 UTC [A-PF] claim cards 0+1 ~5 min: prefill profile pf4 (512:1 2048:1)
2026-11-21 06:13:44 UTC [A-PF] released cards 0+1: pf4 rc=0
2026-11-21 06:22:45 UTC [A-I] released cards 0+1
2026-11-21 06:23:12 UTC [A-H] released-session scheduling note: GPU0/GPU1 owner files empty; CPU quiet O-HW pid3781195 (hwprobe/in-main.sh 0), child gpu-run3796143 waits behind PRIORITY.hold A-S/A-I mtime06:15. Integration may be blocked by this cycle. H owns no resources; core156e010 gatePASS, timing still pending.
2026-11-21 06:27:58 UTC [A-H] released/waiting dispatchers stopped BEFORE acquisition: H4 core156e010 merge-ready (tipca25bbe), quick gatePASS. Depth tuning/hc4 BLOCKED on CPU quiet/hold scheduling; no H resources remain. See workers/H_report4.md.
2026-11-21 06:28:22 UTC [B-worker codex-lane] claim cards 0+1 ~5 min: new C2 policy quick gate
2026-11-21 06:33:09 UTC [O-HW] released: stopped hwprobe main-v2 card 0 session (held cpu-run --quiet while gpu-run waited behind PRIORITY.hold; never acquired GPU). Sorry for the deadlock. Future O-HW runs: hold-clear -> gpu-run, no --quiet (device-timed).
2026-11-21 06:41:26 UTC [B] bin/gpu-run + bin/cpu-run: non-priority waiters now POLL (flock -n every 2 s, re-checking PRIORITY.hold), priority agents keep a blocking flock, so the integration run wins the next release instead of losing the non-FIFO flock race (A-S int-pf2 waited 25 min behind builds). Tested on fake locks. Applies to new invocations only.
2026-11-21 06:42:11 UTC [A-S] claim cards 0+1 ~15 min: int-pf2 bounded policy serve/no-MTP/pre-post health; CPU quiet acquired before GPU
2026-11-21 06:44:57 UTC [A-S] released cards 0+1: int-pf2 rc=0
2026-11-21 06:44:59 UTC [O-HW] claim card 0 ~3 min: hwprobe main-v2
2026-11-21 06:45:04 UTC [A-PF] claim cards 0+1 ~5 min: prefill profile pf5 (2048:1:256:2 512:1:256:3 2048:1:256:3)
2026-11-21 06:45:21 UTC [O-HW] release card 0: hwprobe main-v2 rc=0
2026-11-21 06:45:40 UTC [B-worker codex-arfuse] claim cards 0+1 ~3 min: peer-store hop microbench
2026-11-21 06:45:48 UTC [O-HW] claim card 1 ~3 min: hwprobe alu-bw+sustain
2026-11-21 06:47:58 UTC [B-worker codex-lane] release cards 0+1
2026-11-21 06:50:26 UTC [A-PF] released cards 0+1: pf5 rc=0
2026-11-21T06:50:43Z [A-PF] claim cards 0+1 ~3 min: smoke gate 4974075
2026-11-21 06:50:43 UTC [A-I] claim cards 0+1 ~3 min: smoke pf-4974075
2026-11-21 06:52:23 UTC [B-worker codex-arfuse] release cards 0+1: peer-store hop microbench rc=137
2026-11-21 06:52:51 UTC [O-HW] release card 1: hwprobe alu-bw+sustain rc=0
2026-11-21 06:53:05 UTC [O-HW] claim card 0 ~3 min: hwprobe alu-bw+sustain
2026-11-21 06:55:09 UTC [A-R] claim cards 0+1 ~15 min: d1-bf16 bounded policy serve/no-MTP/pre-post health; CPU quiet acquired before GPU
2026-11-21 06:55:29 UTC [O-HW] release card 0: hwprobe alu-bw+sustain rc=0
2026-11-21 06:55:29 UTC [O-HW] claim cards 0+1 ~1 min: hwprobe p2p properties query
2026-11-21 06:55:29 UTC [O-HW] release cards 0+1: hwprobe p2p rc=0
2026-11-21 06:55:35 UTC [B-worker codex-arfuse] claim cards 0+1 ~3 min: peer-store hop microbench
2026-11-21 06:55:47 UTC [O-HW] claim card 1 ~3 min: hwprobe hostusm-atomic
2026-11-21 06:58:14 UTC [A-R] released cards 0+1: d1-bf16 rc=0
2026-11-21 06:58:15 UTC [O-HW] release card 1: hwprobe hostusm-atomic rc=0
2026-11-21 06:59:17 UTC [B-worker codex-arfuse] release cards 0+1: peer-store hop microbench rc=0
2026-11-21 06:59:28 UTC [A-H] planned H5 controls cards 0+1 ~3m each: depth2/depth3, 128 tokens, AFTER A-S int-mtp2 release. Baseline has not reached GPU yet: int-mtp2-build.log still queued since06:45, current CPU owner A-KF since06:58. H owns no resources; exact integration binary will be captured after release.
2026-11-21 07:04:19 UTC [A-H] H5 precondition note: int-mtp2 build still waiting since06:45; PRIORITY.hold currently ABSENT. Shared wrapper is_priority only applies with an active hold, so new CPU build pollers can win before A-S. H controls remain idle until int-mtp2 release; hold untouched.
2026-11-21 07:06:07 UTC [A-H] H5 baseline-wait detail: integration native build PID3812160 is alive, but B70_AGENT is UNSET in its environment (verified only that non-secret field). Therefore it is not an A-S priority waiter; creating an A-S/A-I-only hold would not prioritize this existing build. No H GPU work started; hold/processes untouched.
2026-11-21 07:06:32 UTC [A-I] released cards 0+1
2026-11-21T07:06:33Z [A-PF] release cards 0+1: smoke gate
2026-11-21 07:06:51 UTC [A-PF] claim cards 0+1 ~5 min: prefill profile pf6 (64:0 512:1 2048:1)
2026-11-21 07:10:15 UTC [A-PF] released cards 0+1: pf6 rc=0
2026-11-21 07:11:25 UTC [A-R] claim cards 0+1 ~15 min: d1-f32 bounded policy serve/no-MTP/pre-post health; CPU quiet acquired before GPU
2026-11-21 07:14:49 UTC [A-R] released cards 0+1: d1-f32 rc=0
2026-11-21 07:14:52 UTC [A-S] claim cards 0+1 ~15 min: int-mtp2 bounded policy serve/no-MTP/pre-post health; CPU quiet acquired before GPU
2026-11-21 07:17:35 UTC [A-S] released cards 0+1: int-mtp2 rc=0
2026-11-21 07:17:36 UTC [A-PF] claim cards 0+1 ~4 min: prefill profile pf7 (1024:1)
2026-11-21 07:17:37 UTC [A-KF] claim card 0 ~15 sec: R8 project health-pre
2026-11-21 07:18:58 UTC [A-KF] claim cards 0+1 ~3 min (max 4m50s): kf8-fusion-gate
2026-11-21 07:19:10 UTC [A-R] claim cards 0+1 ~15 min: d1-f32 bounded policy serve/no-MTP/pre-post health; CPU quiet acquired before GPU
2026-11-21 07:19:58 UTC [A-PF] released cards 0+1: pf7 rc=0
2026-11-21 07:19:58 UTC [A-PF] claim cards 0+1 ~4 min: prefill profile pf8 (1024:1)
2026-11-21 07:21:04 UTC [A-KF] R8 requests are QUEUED ONLY: pre-health and GPU-only quick gate wait in gpu-run; current owner A-PF, no KF CPU/GPU resources. Prior claim lines record request time, not acquired time. Owned-command bounds: health 90s, kernel timing 180s, quick gate 285+5s.
2026-11-21 07:22:24 UTC [A-PF] released cards 0+1: pf8 rc=0
2026-11-21 07:25:23 UTC [A-KF] released cards 0+1: kf8-fusion-gate rc=0
2026-11-21 07:25:42 UTC [A-KF] released card 0: R8 project health-pre rc=0
2026-11-21 07:28:38 UTC [A-R] released cards 0+1: d1-f32 rc=0
2026-11-21 07:29:47 UTC [A-R] note: d1-f32 two-card bench (acquired 07:25:43, released 07:28:38) was already queued before the 07:25 card-0-only directive reached me; no further two-card claims from A-R until [INK] releases card 1
2026-11-21 07:37:23 UTC [A-LY] claim card 0 ~3 min: health+base expert timing
2026-11-21 07:37:45 UTC [A-LY] release card 0: health+base expert timing rc=0
2026-11-21 07:37:52 UTC [A-LY] claim card 0 ~3 min: s1 expert timing
2026-11-21 07:37:59 UTC [A-LY] release card 0: s1 expert timing rc=0
2026-11-21 07:38:03 UTC [B-worker codex-ooo] claim cards 0 ~3 min: real-GGUF sibling dispatch
2026-11-21 07:38:11 UTC [A-LY] claim card 0 ~3 min: s1 Q8_EXPERT tests
2026-11-21 07:38:32 UTC [B-worker codex-ooo] release cards 0 rc=0
2026-11-21 07:38:39 UTC [A-LY] release card 0: s1 Q8_EXPERT tests rc=0
2026-11-21 07:47:31 UTC [A-HS] claim card 0 ~3 min: cold3.log
2026-11-21 07:47:33 UTC [A-HS] released card 0 (rc=0): cold3.log
2026-11-21 08:01:08 UTC [INK] claim card 1 (Inkling eval, hours): llama.cpp SYCL inkling-2-mini-iq4nl serve :18160 + abfast sysadmin/langs/base, user-approved; card 0 untouched
2026-11-21 08:01:59 UTC [A-PF] claim card 0 ~120s: gdn1
2026-11-21 08:02:01 UTC [A-PF] release card 0: gdn1 rc=0
2026-11-21T08:03:07Z [A-H] claim card 0 ~3 min: single-card descriptor slice bitwise + real GGUF R1/4/8/16 timestamps; card1 untouched
2026-11-21T08:03:41Z [A-H] released card 0: single-card slice exit=0
2026-11-21 08:05:00 UTC [A-KF] queued card0 only ~2min (owned max285+5s): fused ROW_BITS, no CPU quiet
2026-11-21 08:05:47 UTC [A-KF] released card0: fused ROW_BITS rc=0
2026-11-21 08:08:13 UTC [A-R] claim card 0 ~1 min: res-f32 op check (tf-xpu-fn-gguf-test res-f32)
2026-11-21 08:08:13 UTC [A-R] release card 0: res-f32 op check rc=0
2026-11-21 08:08:48 UTC [A-HS] claim card 0 ~3 min: cold4.log
2026-11-21 08:08:49 UTC [A-HS] released card 0 (rc=0): cold4.log
2026-11-21 08:10:02 UTC [A-PF] claim card 0 ~150s: attn1
2026-11-21 08:10:03 UTC [A-PF] release card 0: attn1 rc=0
2026-11-21 08:10:03 UTC [A-PF] claim card 0 ~120s: gdn2
2026-11-21 08:10:04 UTC [A-PF] release card 0: gdn2 rc=0
2026-11-21 08:11:11 UTC [A-LY] claim card 0 ~3 min: s2 split expert timing+Q8_EXPERT
2026-11-21 08:11:39 UTC [A-LY] release card 0: s2 split expert timing+Q8_EXPERT rc=0
2026-11-21 08:11:47 UTC [A-KF] queued card0 only ~20sec owned: cold fixture health-pre
2026-11-21 08:12:03 UTC [A-KF] released card0: cold health-pre rc=0
2026-11-21T08:12:15Z [A-H] claim card 0 ~2 min: four-lane page-boundary slice unit check, R1/4/8/16, card1 untouched
2026-11-21T08:12:46Z [A-H] released card 0: four-lane page-boundary slice exit=0
2026-11-21 08:21:14 UTC [A-PF] claim card 0 ~120s: gdn3
2026-11-21 08:21:15 UTC [A-PF] release card 0: gdn3 rc=0
2026-11-21 08:42:10 UTC [A-KF] queued card0 only ~20sec owned: cold fixture health-post
2026-11-21 08:42:24 UTC [A-KF] released card0: cold health-post rc=0
2026-11-21 08:43:21 UTC [A-PF] claim card 0 ~150s: attn2
2026-11-21 08:43:22 UTC [A-PF] release card 0: attn2 rc=0
2026-11-21 08:43:22 UTC [A-PF] claim card 0 ~120s: gdn4
2026-11-21 08:43:23 UTC [A-PF] release card 0: gdn4 rc=0
2026-11-21 08:46:24 UTC [A-KF] queued card0 only ~20sec owned: cold fixture health-pre
2026-11-21 08:46:39 UTC [A-KF] released card0: cold health-pre rc=0
2026-11-21 08:50:08 UTC [A-HS] claim card 0 ~3 min: slice hc1
2026-11-21 08:50:41 UTC [A-HS] released card 0 (rc=0): slice hc1
2026-11-21 08:51:09 UTC [A-PF] claim card 0 ~170s: attns1
2026-11-21 08:51:09 UTC [A-PF] release card 0: attns1 rc=0
2026-11-21 08:51:44 UTC [A-HS] claim card 0 ~3 min: slice ln1
2026-11-21 08:52:21 UTC [A-HS] released card 0 (rc=0): slice ln1
2026-11-21 08:53:04 UTC [A-PF] claim card 0 ~170s: attns2
2026-11-21 08:53:06 UTC [A-PF] release card 0: attns2 rc=0
2026-11-21 08:57:04 UTC [A-KF] queued card0 only ~2min (owned max285+5s): DPAS integer/down ROW_BITS, no CPU quiet
2026-11-21 08:57:33 UTC [A-KF] released card0: DPAS integer/down ROW_BITS rc=1
2026-11-21 08:58:23 UTC [A-S] claim card 0 ~1 min: slice gate fn-a3 988effc (default hashes vs H reference)
2026-11-21 08:58:55 UTC [A-S] released card 0: slice gate rc=0
2026-11-21 08:58:59 UTC [A-PF] claim card 0 ~170s: attns3
2026-11-21 08:59:01 UTC [A-PF] release card 0: attns3 rc=0
2026-11-21 08:59:35 UTC [A-LY] claim card 0 ~3 min: rank0 load test split layout
2026-11-21 08:59:36 UTC [A-LY] release card 0: rank0 load test split layout rc=0
2026-11-21 08:59:48 UTC [A-PF] claim card 0 ~175s: attns4
2026-11-21 08:59:51 UTC [A-PF] release card 0: attns4 rc=0
2026-11-21T09:02:09Z [A-H] claim card 0 ~2 min: H6 serving OOO schedule slice hash/timestamps and private replay event batches; card1 untouched
2026-11-21T09:02:42Z [A-H] released card 0: H6 OOO/private-event slice exit=134
2026-11-21 09:03:55 UTC [A-HS] claim card 0 ~3 min: slice hg1
2026-11-21 09:04:15 UTC [A-LY] claim card 0 ~3 min: rank0 load test split layout
2026-11-21 09:04:34 UTC [A-HS] released card 0 (rc=0): slice hg1
2026-11-21 09:04:53 UTC [A-LY] release card 0: rank0 load test split layout rc=0
2026-11-21 09:05:07 UTC [A-PF] claim card 0 ~150s: attns5
2026-11-21 09:05:07 UTC [A-PF] release card 0: attns5 rc=0
2026-11-21 09:05:09 UTC [A-LY] claim card 0 ~3 min: rank0 load test raw layout
2026-11-21 09:05:17 UTC [A-LY] release card 0: rank0 load test raw layout rc=0
2026-11-21 09:06:55 UTC [A-KF] claim card0 only ~3min (max250+5s): fused real-weight cold timestamp bundle
2026-11-21 09:09:56 UTC [A-KF] released card0: cold timestamp bundle rc=0
2026-11-21 09:09:56 UTC [A-KF] queued card0 only ~20sec owned: cold fixture health-post
2026-11-21 09:10:15 UTC [A-KF] released card0: cold health-post rc=0
2026-11-21 09:14:52 UTC [A-PF] claim card 0 ~150s: attns6
2026-11-21 09:14:52 UTC [A-PF] release card 0: attns6 rc=0
2026-11-21T09:19:29Z [A-H] claim card 0 ~2 min: final H6 OOO/private replay event slice gate; own build paused, card1 untouched
2026-11-21T09:19:49Z [A-H] released card 0: final H6 slice exit=1
2026-11-21T09:21:38Z [A-H] claim card 0 ~3 min: H6 slice + health with default60s cap (prior Docker delayed exit0); build paused, card1 untouched
2026-11-21T09:22:10Z [A-H] released card 0: H6 final slice/health60 exit=0
2026-11-21 09:23:54 UTC [A-LY] claim card 0 ~3 min: merged 988effc: expert fixture + Q8_EXPERT
2026-11-21 09:24:21 UTC [A-LY] release card 0: merged 988effc: expert fixture + Q8_EXPERT rc=0
2026-11-21 09:24:40 UTC [A-LY] claim card 0 ~3 min: slice gate split (fn-slice 3 256 1)
2026-11-21 09:24:58 UTC [A-LY] release card 0: slice gate split (fn-slice 3 256 1) rc=0
2026-11-21 09:25:18 UTC [A-LY] claim card 0 ~3 min: slice gate raw A/B
2026-11-21 09:25:20 UTC [A-LY] release card 0: slice gate raw A/B rc=0
2026-11-21 09:30:49 UTC [A-HS] claim card 0 ~3 min: slice hg3
2026-11-21 09:31:09 UTC [A-HS] released card 0 (rc=1): slice hg3
2026-11-21 09:31:14 UTC [A-LY] claim card 0 ~3 min: slice base 988effc
2026-11-21 09:31:15 UTC [A-LY] release card 0: slice base 988effc rc=0
2026-11-21 09:31:43 UTC [A-HS] claim card 0 ~3 min: slice hg4
2026-11-21 09:32:19 UTC [A-HS] released card 0 (rc=0): slice hg4
2026-11-21 09:37:46 UTC [A-KF] queued card0 only ~2min (owned max285+5s): corrected DPAS down and primary expert up row bits
2026-11-21 09:38:24 UTC [A-KF] released card0: corrected DPAS row bits rc=0
2026-11-21 09:41:01 UTC [A-KF] queued card0 only ~2min (owned max285+5s): remaining expert XMX and tiled-router row bits
2026-11-21 09:41:10 UTC [A-LY] claim card 0 ~3 min: slice gate split + raw A/B
2026-11-21 09:41:48 UTC [A-KF] released card0: corrected DPAS row bits rc=0
2026-11-21 09:42:06 UTC [A-LY] release card 0: slice gate split + raw A/B rc=0
2026-11-21 09:42:29 UTC [A-HS] claim card 0 ~3 min: slice pp1
2026-11-21 09:45:00 UTC [A-HS] released card 0 (rc=1): slice pp1
2026-11-21 09:46:11 UTC [A-KF] queued card0 only ~20sec owned: cold fixture health-pre
2026-11-21 09:46:23 UTC [A-KF] released card0: cold health-pre rc=0
2026-11-21 09:48:28 UTC [A-S] claim card 0 ~1 min: slice gate fn-a3 233608f
2026-11-21 09:49:00 UTC [A-S] released card 0: slice gate rc=0
2026-11-21 09:51:59 UTC [A-LY] claim card 0 ~3 min: step3 MTP q2_0 expert timing + invariance
2026-11-21 09:52:14 UTC [A-LY] release card 0: step3 MTP q2_0 expert timing + invariance rc=0
2026-11-21 09:52:57 UTC [A-PF] claim card 0 ~170s: dpas1
2026-11-21 09:52:57 UTC [A-PF] release card 0: dpas1 rc=0
2026-11-21 09:57:15 UTC [A-S] claim card 0 ~1 min: slice gate fn-a3 1c15dcd
2026-11-21 09:57:48 UTC [A-S] released card 0: slice gate rc=0
2026-11-21 09:58:34 UTC [A-PF] claim card 0 ~170s: dpas2
2026-11-21 09:58:34 UTC [A-PF] release card 0: dpas2 rc=0
2026-11-21 10:04:23 UTC [A-HS] claim card 0 ~3 min: slice pp2
2026-11-21 10:05:03 UTC [A-HS] released card 0 (rc=0): slice pp2
2026-11-21 10:05:04 UTC [A-KF] claim card0 only ~3min (max250+5s): fused real-weight cold timestamp bundle
2026-11-21 10:05:16 UTC [A-PF] claim card 0 ~170s: fused1
2026-11-21 10:07:30 UTC [A-KF] released card0: cold timestamp bundle rc=0
2026-11-21 10:07:30 UTC [A-KF] queued card0 only ~20sec owned: cold fixture health-post
2026-11-21 10:07:45 UTC [A-KF] released card0: cold health-post rc=0
2026-11-21 10:07:53 UTC [A-PF] release card 0: fused1 rc=0
2026-11-21 10:08:22 UTC [A-LY] claim card 0 ~3 min: R2 down split + MTP split: fixtures
2026-11-21 10:08:48 UTC [A-LY] release card 0: R2 down split + MTP split: fixtures rc=0
2026-11-21 10:10:12 UTC [A-PF] claim card 0 ~175s: slice1
2026-11-21 10:10:15 UTC [A-PF] release card 0: slice1 rc=0
2026-11-21T10:10:27Z [A-H] claim card 0 ~3 min: H7 fused/packed recording hash+equal launches+binding reuse; card1 untouched
2026-11-21T10:11:00Z [A-H] released card 0: H7 fused/packed slice gate exit=0
2026-11-21 10:11:12 UTC [A-LY] claim card 0 ~3 min: R2: MTP invariance, Q8_EXPERT, slice gate
2026-11-21 10:11:45 UTC [A-LY] release card 0: R2: MTP invariance, Q8_EXPERT, slice gate rc=0
2026-11-21 10:16:28 UTC [A-PF] claim card 0 ~175s: slice2
2026-11-21 10:16:34 UTC [A-PF] release card 0: slice2 rc=0
2026-11-21 10:17:08 UTC [A-LY] claim card 0 ~3 min: load time split/downraw/raw (rank 0 load test)
2026-11-21 10:17:39 UTC [A-LY] release card 0: load time split/downraw/raw (rank 0 load test) rc=0
2026-11-21 10:18:44 UTC [A-HS] claim card 0 ~3 min: slice pp3
2026-11-21 10:19:28 UTC [A-HS] released card 0 (rc=0): slice pp3
2026-11-21 10:23:45 UTC [A-S] claim card 0 ~1 min: slice gate fn-a3 3299501d
2026-11-21 10:24:20 UTC [A-S] released card 0: slice gate rc=0
2026-11-21 10:25:44 UTC [A-HS] claim card 0 ~3 min: slice pp4
2026-11-21 10:26:27 UTC [A-HS] released card 0 (rc=0): slice pp4
2026-11-21 10:26:55 UTC [A-KF] queued card0 only ~2min (owned max285+5s): hybrid expert full row bits and prepared/unprepared API bits
2026-11-21 10:27:35 UTC [A-KF] released card0: corrected DPAS row bits rc=0
2026-11-21 10:31:05 UTC [A-S] claim card 0 ~1 min: slice gate fn-a3 d97096cb
2026-11-21 10:31:38 UTC [A-S] released card 0: slice gate rc=0
2026-11-21 10:32:51 UTC [A-PF] claim card 0 ~170s: dpas3
2026-11-21 10:32:52 UTC [A-PF] release card 0: dpas3 rc=0
2026-11-21 10:34:35 UTC [A-LY] claim card 0 ~3 min: C1: two-parts-per-lane expert up A/B
2026-11-21 10:34:59 UTC [A-LY] release card 0: C1: two-parts-per-lane expert up A/B rc=0
2026-11-21 10:35:48 UTC [A-HS] claim card 0 ~3 min: slice dg1
2026-11-21 10:36:34 UTC [A-HS] released card 0 (rc=0): slice dg1
2026-11-21 10:37:18 UTC [A-PF] claim card 0 ~170s: dpas4
2026-11-21 10:37:20 UTC [A-PF] release card 0: dpas4 rc=0
2026-11-21 10:37:33 UTC [A-PF] claim card 0 ~170s: dpas5
2026-11-21 10:37:36 UTC [A-PF] release card 0: dpas5 rc=0
2026-11-21 10:39:11 UTC [A-LY] claim card 0 ~3 min: C1: upw bit compare
2026-11-21 10:39:14 UTC [A-LY] release card 0: C1: upw bit compare rc=0
2026-11-21 10:39:47 UTC [A-LY] claim card 0 ~3 min: C1: upw bit compare repeat
2026-11-21 10:39:52 UTC [A-LY] release card 0: C1: upw bit compare repeat rc=0
2026-11-21 10:40:02 UTC [A-PF] claim card 0 ~170s: dpas6
2026-11-21 10:40:03 UTC [A-PF] release card 0: dpas6 rc=0
2026-11-21 10:41:48 UTC [A-HS] claim card 0 ~3 min: slice gd1
2026-11-21 10:42:28 UTC [A-LY] claim card 0 ~3 min: C1: upw debug compare
2026-11-21 10:42:30 UTC [A-HS] released card 0 (rc=0): slice gd1
2026-11-21 10:42:35 UTC [A-LY] release card 0: C1: upw debug compare rc=0
2026-11-21 10:44:55 UTC [A-PF] claim card 0 ~175s: dpas7
2026-11-21 10:44:59 UTC [A-PF] release card 0: dpas7 rc=0
2026-11-21 10:46:48 UTC [A-LY] claim card 0 ~3 min: C1: two-rows-per-lane expert up A/B
2026-11-21 10:47:05 UTC [A-LY] release card 0: C1: two-rows-per-lane expert up A/B rc=0
2026-11-21 10:48:36 UTC [A-PF] claim card 0 ~178s: dpas8
2026-11-21 10:48:41 UTC [A-PF] release card 0: dpas8 rc=0
2026-11-21 10:49:21 UTC [A-LY] claim card 0 ~3 min: C1: two-rows up R1-3 sweep
2026-11-21 10:49:51 UTC [A-LY] release card 0: C1: two-rows up R1-3 sweep rc=0
2026-11-21 10:50:48 UTC [A-H] claim card 0 ~3 min: H8 real-GGUF slice tail/cache A-B and serial keep budget; card1 untouched
2026-11-21 10:51:14 UTC [A-PF] claim card 0 ~178s: dpas9
2026-11-21 10:51:40 UTC [A-H] released card 0: H8 tail/cache/keep rc=0
2026-11-21 10:51:45 UTC [A-PF] release card 0: dpas9 rc=0
2026-11-21 10:53:22 UTC [A-LY] claim card 0 ~3 min: C1: MTP q2_0 rows-per-subgroup 4 vs 8
2026-11-21 10:53:32 UTC [A-LY] release card 0: C1: MTP q2_0 rows-per-subgroup 4 vs 8 rc=0
2026-11-21 10:54:11 UTC [A-PF] claim card 0 ~120s: dpas10
2026-11-21 10:54:12 UTC [A-PF] release card 0: dpas10 rc=0
2026-11-21 10:55:26 UTC [A-HS] claim card 0 ~3 min: slice gd2
2026-11-21 10:56:14 UTC [A-HS] released card 0 (rc=0): slice gd2
2026-11-21 10:57:07 UTC [A-PF] claim card 0 ~175s: slice3
2026-11-21 10:57:13 UTC [A-PF] release card 0: slice3 rc=0
2026-11-21 10:59:09 UTC [A-LY] claim card 0 ~3 min: C1: MTP q2_0 WG size sweep
2026-11-21 10:59:14 UTC [A-LY] release card 0: C1: MTP q2_0 WG size sweep rc=0
2026-11-21 11:01:14 UTC [A-KF] queued card0 only ~2min (owned max285+5s): hybrid expert full row bits and prepared/unprepared API bits
2026-11-21 11:01:23 UTC [A-H] claim card 0 ~3 min: H8 final canonical sampler tail hashes/cache A-B; card1 untouched
2026-11-21 11:01:33 UTC [A-PF] claim card 0 ~175s: slice4
2026-11-21 11:01:54 UTC [A-KF] queued card0 only ~20sec owned: cold fixture health-pre
2026-11-21 11:01:55 UTC [A-KF] released card0: corrected DPAS row bits rc=0
2026-11-21 11:02:10 UTC [A-KF] released card0: cold health-pre rc=0
2026-11-21 11:03:01 UTC [A-H] released card 0: H8 final tail/cache rc=0
2026-11-21 11:03:07 UTC [A-PF] release card 0: slice4 rc=0
2026-11-21 11:03:50 UTC [A-LY] claim card 0 ~3 min: C1: four-row down sweep
2026-11-21 11:04:10 UTC [A-LY] release card 0: C1: four-row down sweep rc=0
2026-11-21 11:04:40 UTC [A-PF] claim card 0 ~120s: dpas11
2026-11-21 11:04:41 UTC [A-PF] release card 0: dpas11 rc=0
2026-11-21 11:10:10 UTC [A-LY] claim card 0 ~3 min: C1 gate: upw default R=1 (fixture, Q8_EXPERT, slice)
2026-11-21 11:11:07 UTC [A-LY] release card 0: C1 gate: upw default R=1 (fixture, Q8_EXPERT, slice) rc=0
2026-11-21 11:11:25 UTC [A-HS] claim card 0 ~3 min: slice dp1
2026-11-21 11:12:06 UTC [A-HS] released card 0 (rc=0): slice dp1
2026-11-21 11:13:41 UTC [A-PF] claim card 0 ~175s: slice5
2026-11-21 11:13:50 UTC [A-PF] release card 0: slice5 rc=0
2026-11-21 11:14:03 UTC [A-HS] claim card 0 ~3 min: slice io1
2026-11-21 11:14:47 UTC [A-LY] claim card 0 ~3 min: C1: DRAM-realistic (rotated experts) fixture raw/ups/upw
2026-11-21 11:15:05 UTC [A-HS] released card 0 (rc=0): slice io1
2026-11-21 11:15:34 UTC [A-LY] release card 0: C1: DRAM-realistic (rotated experts) fixture raw/ups/upw rc=0
2026-11-21 11:20:51 UTC [A-S] claim card 0 ~1 min: slice gate fn-a3 fc200f43
2026-11-21 11:21:24 UTC [A-S] released card 0: slice gate rc=0
2026-11-21 11:23:15 UTC [A-LY] claim card 0 ~3 min: C1: unrolled-chunk up (rotated)
2026-11-21 11:23:35 UTC [A-LY] release card 0: C1: unrolled-chunk up (rotated) rc=0
2026-11-21 11:24:29 UTC [A-HS] claim card 0 ~3 min: slice db1
2026-11-21 11:25:07 UTC [A-HS] released card 0 (rc=0): slice db1
2026-11-21 11:25:14 UTC [A-PF] claim card 0 ~175s: tail1
2026-11-21 11:25:55 UTC [A-PF] release card 0: tail1 rc=0
2026-11-21 11:31:45 UTC [A-H] claim card 0 ~3 min: H9 batched keep bitwise/canaries, slice hashes, MTP timestamps; card1 untouched
2026-11-21 11:32:16 UTC [A-H] released card 0: H9 keep gate rc=134
2026-11-21 11:32:22 UTC [A-LY] claim card 0 ~3 min: C1: preloaded-chunk up (rotated)
2026-11-21 11:32:43 UTC [A-LY] release card 0: C1: preloaded-chunk up (rotated) rc=0
2026-11-21 11:33:43 UTC [A-HS] claim card 0 ~3 min: attn-op o1
2026-11-21 11:33:47 UTC [A-HS] released card 0 (rc=1): attn-op o1
2026-11-21 11:33:51 UTC [A-KF] claim card0 only ~3min (max250+5s): expert union real-weight cold timestamp bundle
2026-11-21 11:34:02 UTC [A-KF] released card0: cold timestamp bundle rc=0
2026-11-21 11:34:02 UTC [A-KF] queued card0 only ~20sec owned: cold fixture health-post
2026-11-21 11:34:18 UTC [A-KF] released card0: cold health-post rc=0
2026-11-21 11:36:46 UTC [A-HS] claim card 0 ~3 min: attn-op o2
2026-11-21 11:37:01 UTC [A-HS] released card 0 (rc=0): attn-op o2
2026-11-21 11:37:44 UTC [A-PF] claim card 0 ~170s: bf1
2026-11-21 11:37:45 UTC [A-PF] release card 0: bf1 rc=0
2026-11-21 11:39:23 UTC [A-HS] claim card 0 ~3 min: attn-op o3
2026-11-21 11:39:32 UTC [A-HS] released card 0 (rc=0): attn-op o3
2026-11-21 11:40:16 UTC [A-HS] claim card 0 ~3 min: attn-op o4
2026-11-21 11:40:30 UTC [A-HS] released card 0 (rc=0): attn-op o4
2026-11-21 11:40:42 UTC [A-KF] queued card0 only ~2min (owned max285+5s): hybrid expert full row bits and prepared/unprepared API bits
2026-11-21 11:40:44 UTC [A-HS] claim card 0 ~3 min: attn-op o5
2026-11-21 11:41:24 UTC [A-KF] released card0: corrected DPAS row bits rc=0
2026-11-21 11:41:24 UTC [A-KF] queued card0 only ~20sec owned: cold fixture health-pre
2026-11-21 11:41:38 UTC [A-KF] released card0: cold health-pre rc=0
2026-11-21 11:41:47 UTC [A-HS] released card 0 (rc=0): attn-op o5
2026-11-21 11:41:53 UTC [A-H] claim card 0 ~3 min: H9 keep gate after strict flag registration, bitwise/canaries and MTP timestamps; card1 untouched
2026-11-21 11:42:45 UTC [A-PF] claim card 0 ~170s: bf2
2026-11-21 11:44:14 UTC [A-LY] claim card 0 ~3 min: MTP q8_1-act experts: timing + agreement
2026-11-21 11:44:24 UTC [A-H] released card 0: H9 registered keep gate rc=0
2026-11-21 11:44:29 UTC [A-LY] release card 0: MTP q8_1-act experts: timing + agreement rc=0
2026-11-21 11:44:31 UTC [A-PF] release card 0: bf2 rc=0
2026-11-21 11:44:49 UTC [A-KF] claim card0 only ~3min (max250+5s): expert union real-weight cold timestamp bundle
2026-11-21 11:44:49 UTC [A-HS] claim card 0 ~3 min: attn-op o6
2026-11-21 11:45:01 UTC [A-KF] released card0: cold timestamp bundle rc=0
2026-11-21 11:45:01 UTC [A-KF] queued card0 only ~20sec owned: cold fixture health-post
2026-11-21 11:45:18 UTC [A-KF] released card0: cold health-post rc=0
2026-11-21 11:45:29 UTC [A-HS] released card 0 (rc=0): attn-op o6
2026-11-21 11:51:14 UTC [A-PF] claim card 0 ~120s: bf3
2026-11-21 11:51:16 UTC [A-PF] release card 0: bf3 rc=0
2026-11-21 11:51:16 UTC [A-PF] claim card 0 ~175s: tail2
2026-11-21 11:51:52 UTC [A-PF] release card 0: tail2 rc=0
2026-11-21 11:52:09 UTC [A-PF] claim card 0 ~120s: tail3
2026-11-21 11:52:26 UTC [A-PF] release card 0: tail3 rc=0
2026-11-21 11:55:00 UTC [A-LY] claim card 0 ~3 min: MTP q8 act modes 1/2 agreement
2026-11-21 11:55:04 UTC [A-LY] release card 0: MTP q8 act modes 1/2 agreement rc=0
2026-11-21 12:00:51 UTC [A-S] claim card 0 ~1 min: slice gate fn-a3 d96bd26c
2026-11-21 12:01:23 UTC [A-S] released card 0: slice gate rc=0
2026-11-21 12:02:14 UTC [A-KF] queued card0 only ~2min (owned max285+5s): hybrid expert full row bits and prepared/unprepared API bits
2026-11-21 12:02:54 UTC [A-KF] released card0: corrected DPAS row bits rc=0
2026-11-21 12:02:54 UTC [A-KF] queued card0 only ~20sec owned: cold fixture health-pre
2026-11-21 12:03:08 UTC [A-KF] released card0: cold health-pre rc=0
2026-11-21 12:03:52 UTC [A-KF] claim card0 only ~3min (max250+5s): expert union real-weight cold timestamp bundle
2026-11-21 12:03:57 UTC [A-PF] claim card 0 ~150s: bf4
2026-11-21 12:04:04 UTC [A-KF] released card0: cold timestamp bundle rc=0
2026-11-21 12:04:04 UTC [A-KF] queued card0 only ~20sec owned: cold fixture health-post
2026-11-21 12:04:21 UTC [A-KF] released card0: cold health-post rc=0
2026-11-21 12:04:25 UTC [A-PF] release card 0: bf4 rc=0
2026-11-21 12:09:35 UTC [A-S] claim card 0 ~2 min: slice + tail/keep gate fn-a3 e8d534c0
2026-11-21 12:12:11 UTC [A-S] released card 0: slice gate rc=0
2026-11-21 12:12:43 UTC [A-HS] claim card 0 ~3 min: slice fin1
2026-11-21 12:13:21 UTC [A-HS] released card 0 (rc=0): slice fin1
2026-11-21 12:13:22 UTC [A-HS] claim card 0 ~3 min: attn-op fin2
2026-11-21 12:13:25 UTC [A-HS] released card 0 (rc=0): attn-op fin2
2026-11-21 12:14:43 UTC [A-PF] claim card 0 ~100s: bf5
2026-11-21 12:14:43 UTC [A-PF] release card 0: bf5 rc=0
2026-11-21 12:14:43 UTC [A-PF] claim card 0 ~100s: bf6
2026-11-21 12:14:44 UTC [A-PF] release card 0: bf6 rc=0
2026-11-21 12:15:58 UTC [A-LY] claim card 0 ~3 min: MTP q8 act modes 1/2 agreement
2026-11-21 12:16:02 UTC [A-LY] release card 0: MTP q8 act modes 1/2 agreement rc=0
2026-11-21 12:16:02 UTC [A-LY] claim card 0 ~3 min: dense iq3_s row-split fused project
2026-11-21 12:17:23 UTC [A-LY] release card 0: dense iq3_s row-split fused project rc=0
2026-11-21 12:17:53 UTC [A-LY] claim card 0 ~3 min: dense iq3_s row-split fused shared
2026-11-21 12:20:33 UTC [A-LY] release card 0: dense iq3_s row-split fused shared rc=0
2026-11-21 12:21:36 UTC [A-S] claim card 0 ~2 min: slice + tail gate fn-a3 78d34e79
2026-11-21 12:23:41 UTC [A-PF] claim card 0 ~175s: tail4
2026-11-21 12:23:48 UTC [A-KF] queued card0 only ~2min (owned max285+5s): hybrid expert full row bits and prepared/unprepared API bits
2026-11-21 12:24:09 UTC [A-S] released card 0: slice gate rc=0
2026-11-21 12:24:47 UTC [A-PF] release card 0: tail4 rc=0
2026-11-21 12:25:29 UTC [A-KF] released card0: corrected DPAS row bits rc=0
2026-11-21 12:25:29 UTC [A-KF] queued card0 only ~20sec owned: cold fixture health-pre
2026-11-21 12:25:43 UTC [A-KF] released card0: cold health-pre rc=0
2026-11-21 12:26:15 UTC [A-LY] claim card 0 ~3 min: dense iq3_s row-split KREG check
2026-11-21 12:26:15 UTC [A-LY] release card 0: dense iq3_s row-split KREG check rc=0
2026-11-21T12:27:57Z [A-H] claim card 0 ~3 min: H10 drafter Q8 HC and private-buffer MTP overlap; timeout170, card1 untouched.
2026-11-21T12:28:39Z [A-H] released card 0: H10 session rc=0; no card1 claim.
2026-11-21 12:33:01 UTC [A-KF] claim card0 only ~3min (max250+5s): expert union real-weight cold timestamp bundle
2026-11-21 12:33:01 UTC [A-S] claim card 0 ~2 min: slice + tail gate fn-a3 b4dce2a1
2026-11-21 12:33:13 UTC [A-KF] released card0: cold timestamp bundle rc=0
2026-11-21 12:33:13 UTC [A-KF] queued card0 only ~20sec owned: cold fixture health-post
2026-11-21 12:33:29 UTC [A-KF] released card0: cold health-post rc=0
2026-11-21 12:35:04 UTC [A-PF] claim card 0 ~120s: tail5
2026-11-21 12:36:03 UTC [A-S] released card 0: slice gate rc=0
2026-11-21 12:36:38 UTC [A-PF] release card 0: tail5 (lease script interrupted after slice rc=0; health rerun next)
2026-11-21 12:41:30 UTC [A-HS] claim card 0 ~3 min: attn-op mw1
2026-11-21 12:41:35 UTC [A-HS] released card 0 (rc=0): attn-op mw1
2026-11-21 12:42:12 UTC [A-HS] claim card 0 ~3 min: attn-op g1
2026-11-21 12:42:17 UTC [A-HS] released card 0 (rc=0): attn-op g1
2026-11-21 12:46:24 UTC [A-KF] queued card0 only ~2min (owned max285+5s): selected expert API bindings; identical C12 kernel bytes
2026-11-21 12:46:58 UTC [A-KF] released card0: corrected DPAS row bits rc=0
2026-11-21 12:50:54 UTC [A-LY] claim card 0 ~3 min: gate: dense iq3_s row-split + MTP q8 flag (fixture, Q8_EXPERT, slice)
2026-11-21 12:51:50 UTC [A-LY] release card 0: gate: dense iq3_s row-split + MTP q8 flag (fixture, Q8_EXPERT, slice) rc=0
2026-11-21 12:52:40 UTC [A-PF] claim card 0 ~150s: tail6
2026-11-21 12:53:05 UTC [A-PF] release card 0: tail6 rc=0
2026-11-21 12:54:47 UTC [A-HS] claim card 0 ~3 min: slice mw2
2026-11-21 12:55:36 UTC [A-HS] released card 0 (rc=0): slice mw2
2026-11-21 12:59:29 UTC [A-S] claim card 0 ~2 min: slice + tail gate fn-a3 7eb760e9
2026-11-21 13:01:55 UTC [A-PF] claim card 0 ~170s: inv1
2026-11-21 13:02:04 UTC [A-S] released card 0: slice gate rc=0
2026-11-21 13:02:17 UTC [A-PF] release card 0: inv1 rc=0
2026-11-21 13:02:22 UTC [A-PF] claim card 0 ~150s: tail7
2026-11-21 13:02:47 UTC [A-PF] release card 0: tail7 rc=0
2026-11-21 13:04:12 UTC [A-HS] claim card 0 ~3 min: attn-op v2a
2026-11-21 13:04:16 UTC [A-HS] released card 0 (rc=0): attn-op v2a
2026-11-21 13:06:09 UTC [A-HS] claim card 0 ~3 min: slice v2s
2026-11-21 13:07:08 UTC [A-HS] released card 0 (rc=0): slice v2s
2026-11-21 13:07:23 UTC [A-LY] claim card 0 ~3 min: load-time: staged split repack vs heap vs raw
2026-11-21 13:08:16 UTC [A-LY] release card 0: load-time: staged split repack vs heap vs raw rc=0
2026-11-21 13:08:25 UTC [A-LY] claim card 0 ~3 min: gate: staged split load (slice hashes)
2026-11-21 13:08:55 UTC [A-LY] release card 0: gate: staged split load (slice hashes) rc=0
2026-11-21 13:09:16 UTC [A-PF] claim card 0 ~170s: combo1
2026-11-21 13:10:21 UTC [A-PF] release card 0: combo1 rc=0
2026-11-21 13:14:57 UTC [A-S] claim card 0 ~2 min: slice + tail gate fn-a3 cb11ab12
2026-11-21 13:16:11 UTC [A-LY] claim card 0 ~3 min: dense KREG types R=1..8 rotated timing
2026-11-21 13:16:17 UTC [INK] note card 1: inkling server crashed (SYCL GET_ROWS int range at ~65k ctx); card 1 lease still held by INK for rebuild + rerun
2026-11-21 13:17:31 UTC [A-S] released card 0: slice gate rc=0
2026-11-21 13:17:33 UTC [A-LY] release card 0: dense KREG types R=1..8 rotated timing rc=0
2026-11-21 13:18:05 UTC [A-PF] claim card 0 ~170s: combo2
2026-11-21 13:19:11 UTC [A-PF] release card 0: combo2 rc=0
2026-11-21 13:22:43 UTC [A-S] claim card 0 ~2 min: slice + tail gate fn-a3 c2ae97b8
2026-11-21 13:25:16 UTC [A-S] released card 0: slice gate rc=0
2026-11-21 13:25:20 UTC [A-HS] claim card 0 ~3 min: attn-op v2b
2026-11-21 13:25:24 UTC [A-HS] released card 0 (rc=0): attn-op v2b
2026-11-21 13:27:02 UTC [A-PF] claim card 0 ~170s: combo3
2026-11-21 13:27:58 UTC [A-PF] release card 0: combo3 rc=0
2026-11-21 13:28:06 UTC [A-HS] claim card 0 ~3 min: slice v2c
2026-11-21 13:28:53 UTC [A-HS] released card 0 (rc=0): slice v2c
2026-11-21 13:30:03 UTC [A-PF] claim card 0 ~178s: sweep1
2026-11-21 13:30:28 UTC [A-KF] claim card 0 ~3 min: round9 slice; maximum 285s
2026-11-21 13:32:01 UTC [A-PF] release card 0: sweep1 rc=0
2026-11-21 13:32:02 UTC [A-KF] acquired card 0 ~3 min: round9 slice gate
2026-11-21 13:34:36 UTC [A-KF] released card 0: round9 slice rc=0
2026-11-21 13:34:36 UTC [A-KF] claim card 0 ~3 min: round9 dense-q8; maximum 285s
2026-11-21 13:34:36 UTC [A-KF] acquired card 0 ~3 min: round9 dense-q8 gate
2026-11-21 13:35:08 UTC [A-KF] released card 0: round9 dense-q8 rc=0
2026-11-21 13:35:08 UTC [A-KF] claim card 0 ~3 min: round9 dense-bf16; maximum 285s
2026-11-21 13:35:08 UTC [A-KF] acquired card 0 ~3 min: round9 dense-bf16 gate
2026-11-21 13:35:47 UTC [A-KF] released card 0: round9 dense-bf16 rc=0
2026-11-21 13:35:47 UTC [A-KF] claim card 0 ~3 min: round9 experts-default; maximum 285s
2026-11-21 13:35:47 UTC [A-KF] acquired card 0 ~3 min: round9 experts-default gate
2026-11-21 13:36:10 UTC [A-S] claim card 0 ~2 min: slice + tail gate fn-a3 ddc07c72
2026-11-21 13:36:22 UTC [A-KF] released card 0: round9 experts-default rc=0
2026-11-21 13:36:22 UTC [A-KF] claim card 0 ~3 min: round9 experts-opt; maximum 285s
2026-11-21 13:36:22 UTC [A-KF] acquired card 0 ~3 min: round9 experts-opt gate
2026-11-21 13:36:57 UTC [A-KF] released card 0: round9 experts-opt rc=0
2026-11-21 13:36:57 UTC [A-KF] claim card 0 ~3 min: round9 union; maximum 285s
2026-11-21 13:36:57 UTC [A-KF] acquired card 0 ~3 min: round9 union gate
2026-11-21 13:37:34 UTC [A-KF] released card 0: round9 union rc=0
2026-11-21 13:38:10 UTC [A-PF] claim card 0 ~120s: head1
2026-11-21 13:38:25 UTC [A-KF] claim card 0 ~20s: round9 timing health-pre
2026-11-21 13:39:57 UTC [A-LY] claim card 0 ~3 min: q6_k row-split dense + fused project
2026-11-21 13:40:06 UTC [A-S] released card 0: slice gate rc=0
2026-11-21 13:41:30 UTC [A-LY] release card 0: q6_k row-split dense + fused project rc=0
2026-11-21 13:41:34 UTC [A-PF] release card 0: head1 rc=0
2026-11-21 13:41:50 UTC [A-KF] released card 0: round9 timing health-pre rc=0
2026-11-21 13:46:42 UTC [A-KF] claim card 0 ~1 min: round9 rotating real expert bank device timestamps; max250s
2026-11-21 13:46:49 UTC [A-KF] released card 0: round9 rotating expert timestamps rc=0
2026-11-21 13:46:49 UTC [A-KF] claim card 0 ~20s: round9 timing health-post
2026-11-21 13:47:09 UTC [A-KF] released card 0: round9 timing health-post rc=1
2026-11-21 13:48:03 UTC [A-KF] claim card 0 ~1 min: failed 20s post-health recheck at normal60s; no reset (tool resets both cards, INK reserved card1)
2026-11-21 13:48:19 UTC [A-KF] released card 0: normal60s post-health recheck rc=0 reset=0
2026-11-21 13:52:50 UTC [A-S] claim card 0 ~2 min: slice + tail gate fn-a3 a4da5a1f
2026-11-21 13:54:56 UTC [A-PF] claim card 0 ~120s: head2
2026-11-21 13:55:23 UTC [A-S] released card 0: slice gate rc=0
2026-11-21 13:55:31 UTC [A-PF] release card 0: head2 rc=0
2026-11-21 13:58:48 UTC [A-LY] claim card 0 ~3 min: q6_k rs DPAS + dense-rs (R1..16, prefill)
2026-11-21 13:59:07 UTC [A-LY] release card 0: q6_k rs DPAS + dense-rs (R1..16, prefill) rc=0
2026-11-21 13:59:44 UTC [A-LY] claim card 0 ~3 min: q6_k rs-only gate: fixtures, Q8_EXPERT, slice
2026-11-21 14:00:37 UTC [A-LY] release card 0: q6_k rs-only gate: fixtures, Q8_EXPERT, slice rc=0
2026-11-21 14:01:08 UTC [A-LY] claim card 0 ~3 min: q6_k rs-only VRAM delta + fused project hashes
2026-11-21 14:03:01 UTC [A-LY] release card 0: q6_k rs-only VRAM delta + fused project hashes rc=0
2026-11-21 14:03:24 UTC [A-S] claim card 0 ~2 min: slice + tail gate fn-a3 1144114c
2026-11-21 14:05:50 UTC [A-KF] claim card 0 ~3 min: round10 iq2s; maximum 285s
2026-11-21 14:05:55 UTC [A-S] released card 0: slice gate rc=0
2026-11-21 14:05:56 UTC [A-KF] acquired card 0 ~3 min: round10 iq2s gate
2026-11-21 14:06:31 UTC [A-KF] released card 0: round10 iq2s rc=0
2026-11-21 14:07:00 UTC [A-PF] claim card 0 ~120s: head3
2026-11-21 14:07:01 UTC [A-PF] release card 0: head3 rc=0
2026-11-21 14:09:26 UTC [A-LY] claim card 0 ~3 min: q6_k rs stride-32 DPAS + dense-rs + load
2026-11-21 14:09:34 UTC [A-LY] release card 0: q6_k rs stride-32 DPAS + dense-rs + load rc=0
2026-11-21 14:09:51 UTC [A-LY] claim card 0 ~3 min: q6_k rs-only gate w32
2026-11-21 14:11:09 UTC [A-KF] claim card 0 ~3 min: round10 iq2s; maximum 285s
2026-11-21 14:12:07 UTC [A-PF] claim card 0 ~120s: head4
2026-11-21 14:12:07 UTC [A-LY] release card 0: q6_k rs-only gate w32 rc=0
2026-11-21 14:12:08 UTC [A-KF] acquired card 0 ~3 min: round10 iq2s gate
2026-11-21 14:12:41 UTC [A-KF] released card 0: round10 iq2s rc=0
2026-11-21 14:12:42 UTC [A-KF] claim card 0 ~60s: round10 timing health-pre
2026-11-21 14:12:54 UTC [A-KF] released card 0: round10 timing health-pre rc=0
2026-11-21 14:12:59 UTC [A-PF] release card 0: head4 rc=0
2026-11-21 14:18:43 UTC [A-S] claim card 0 ~2 min: slice + tail gate fn-a3 8f2258aa
2026-11-21 14:21:16 UTC [A-S] released card 0: slice gate rc=0
2026-11-21 14:21:32 UTC [A-KF] claim card 0 ~1 min: round10 rotating real expert bank device timestamps; max250s
2026-11-21 14:21:34 UTC [A-PF] claim card 0 ~150s: head5
2026-11-21 14:21:35 UTC [A-KF] released card 0: round10 rotating expert timestamps rc=0
2026-11-21 14:21:35 UTC [A-KF] claim card 0 ~60s: round10 timing health-post
2026-11-21 14:21:50 UTC [A-KF] released card 0: round10 timing health-post rc=0
2026-11-21 14:22:00 UTC [A-LY] claim card 0 ~3 min: iq3_s rs-only DPAS/PF + dense-rs + load
2026-11-21 14:22:03 UTC [A-PF] release card 0: head5 rc=0
2026-11-21 14:22:18 UTC [A-LY] release card 0: iq3_s rs-only DPAS/PF + dense-rs + load rc=0
2026-11-21 14:22:39 UTC [A-LY] claim card 0 ~3 min: iq3_s rs-only gate w33
2026-11-21 14:23:17 UTC [A-PF] claim card 0 ~120s: sweep2
2026-11-21 14:23:35 UTC [A-LY] release card 0: iq3_s rs-only gate w33 rc=0
2026-11-21 14:23:35 UTC [A-LY] claim card 0 ~3 min: iq3_s rs-only fused project hashes
2026-11-21 14:24:56 UTC [A-LY] release card 0: iq3_s rs-only fused project hashes rc=0
2026-11-21 14:25:53 UTC [A-PF] release card 0: sweep2 rc=0
2026-11-21 14:31:09 UTC [A-S] claim card 0 ~2 min: slice + tail gate fn-a3 06dfcb1d
2026-11-21 14:33:43 UTC [A-S] released card 0: slice gate rc=0
2026-11-21 14:33:47 UTC [A-LY] claim card 0 ~3 min: iq4_xs arithmetic decode A/B
2026-11-21 14:34:31 UTC [A-LY] release card 0: iq4_xs arithmetic decode A/B rc=0
2026-11-21 14:34:49 UTC [A-LY] claim card 0 ~3 min: iq4_xs arith decode fused A/B
2026-11-21 14:36:06 UTC [A-PF] claim card 0 ~120s: c1rank
2026-11-21 14:37:30 UTC [A-LY] release card 0: iq4_xs arith decode fused A/B rc=0
2026-11-21 14:39:31 UTC [A-PF] release card 0: c1rank rc=124
2026-11-21 14:40:39 UTC [A-S] claim card 0 ~2 min: slice + tail gate fn-a3 3fe36cf3
2026-11-21 14:43:09 UTC [A-S] released card 0: slice gate rc=0
2026-11-21 14:43:23 UTC [A-LY] claim card 0 ~3 min: iq4_xs row-split KREG probe
2026-11-21 14:43:25 UTC [A-LY] release card 0: iq4_xs row-split KREG probe rc=0
2026-11-21 14:44:28 UTC [A-PF] claim card 0 ~150s: c1rankb
2026-11-21 14:46:10 UTC [A-LY] claim card 0 ~3 min: KREG rows-per-WG at R=2 probe
2026-11-21 14:46:42 UTC [A-PF] release card 0: c1rankb rc=0
2026-11-21 14:48:08 UTC [A-LY] release card 0: KREG rows-per-WG at R=2 probe rc=0
2026-11-21 14:50:30 UTC [A-HS] claim card 0 ~3 min: cold-b1.log
2026-11-21 14:50:32 UTC [A-HS] released card 0 (rc=0): cold-b1.log
2026-11-21 14:53:31 UTC [A-LY] claim card 0 ~3 min: iq4_xs R2 wide KREG gate
2026-11-21 14:54:23 UTC [A-HS] claim card 0 ~3 min: slice gate hb1
2026-11-21 14:54:29 UTC [A-LY] release card 0: iq4_xs R2 wide KREG gate rc=0
2026-11-21 14:57:06 UTC [A-PF] claim card 0 ~150s: lay1
2026-11-21 14:57:25 UTC [A-HS] released card 0 (rc=124): slice gate hb1
2026-11-21 14:57:48 UTC [A-PF] release card 0: lay1 rc=0
2026-11-21 14:57:57 UTC [A-HS] claim card 0 ~3 min: slice gate hb2
2026-11-21 14:59:12 UTC [A-LY] claim card 0 ~3 min: ups_iq2_xs decode-ALU probe
2026-11-21 14:59:38 UTC [A-PF] claim card 0 ~150s: lay2
2026-11-21 15:00:26 UTC [A-HS] released card 0 (rc=0): slice gate hb2
2026-11-21 15:00:26 UTC [A-HS] claim card 0 ~3 min: slice gate hb3
2026-11-21 15:01:00 UTC [A-HS] released card 0 (rc=0): slice gate hb3
2026-11-21 15:01:09 UTC [A-LY] release card 0: ups_iq2_xs decode-ALU probe rc=0
2026-11-21 15:01:34 UTC [A-PF] release card 0: lay2 rc=0
2026-11-21 15:05:09 UTC [A-S] claim card 0 ~2 min: slice + tail gate fn-a3 32ca6f77
2026-11-21 15:07:44 UTC [A-S] released card 0: slice gate rc=0
2026-11-21 15:08:58 UTC [A-KF] claim card 0 ~20s: round11 health-pre, cap60s
2026-11-21 15:09:13 UTC [A-KF] released card 0: round11 health-pre rc=0
2026-11-21 15:09:13 UTC [A-KF] claim card 0 ~2 min: round11 new, cap285s
2026-11-21 15:09:13 UTC [A-KF] acquired card 0 ~2 min: round11 new
2026-11-21 15:09:14 UTC [A-KF] released card 0: round11 new rc=0
2026-11-21 15:09:14 UTC [A-KF] claim card 0 ~20s: round11 health-post, cap60s
2026-11-21 15:09:29 UTC [A-KF] released card 0: round11 health-post rc=0
2026-11-21 15:11:25 UTC [A-LY] claim card 0 ~3 min: ups_iq2_xs staging probe
2026-11-21 15:11:29 UTC [A-LY] release card 0: ups_iq2_xs staging probe rc=0
2026-11-21 15:13:53 UTC [A-KF] claim card 0 ~20s: round11 health-pre, cap60s
2026-11-21 15:14:06 UTC [A-KF] released card 0: round11 health-pre rc=0
2026-11-21 15:14:06 UTC [A-KF] claim card 0 ~2 min: round11 matrix, cap285s
2026-11-21 15:14:06 UTC [A-KF] acquired card 0 ~2 min: round11 matrix
2026-11-21 15:14:26 UTC [A-KF] released card 0: round11 matrix rc=0
2026-11-21 15:14:26 UTC [A-KF] claim card 0 ~20s: round11 health-post, cap60s
2026-11-21 15:14:41 UTC [A-KF] released card 0: round11 health-post rc=0
2026-11-21 15:14:43 UTC [A-S] claim card 0 ~2 min: slice + tail gate fn-a3 72d50dd0
2026-11-21 15:17:16 UTC [A-S] released card 0: slice gate rc=0
2026-11-21 15:17:40 UTC [A-LY] claim card 0 ~3 min: ups_iq2_xs staging probe (real)
2026-11-21 15:17:44 UTC [A-LY] release card 0: ups_iq2_xs staging probe (real) rc=0
2026-11-21 15:18:48 UTC [A-KF] claim card 0 ~20s: round11 health-pre, cap60s
2026-11-21 15:19:03 UTC [A-KF] released card 0: round11 health-pre rc=0
2026-11-21 15:19:03 UTC [A-KF] claim card 0 ~2 min: round11 hc, cap285s
2026-11-21 15:19:03 UTC [A-KF] acquired card 0 ~2 min: round11 hc
2026-11-21 15:19:04 UTC [A-KF] released card 0: round11 hc rc=0
2026-11-21 15:19:04 UTC [A-KF] claim card 0 ~20s: round11 health-post, cap60s
2026-11-21 15:19:18 UTC [A-KF] released card 0: round11 health-post rc=0
2026-11-21 15:19:37 UTC [A-KF] claim card 0 ~20s: round11 health-pre, cap60s
2026-11-21 15:19:41 UTC [A-PF] claim card 0 ~175s: lay3
2026-11-21 15:19:56 UTC [A-KF] released card 0: round11 health-pre rc=0
2026-11-21 15:21:19 UTC [A-LY] claim card 0 ~3 min: ups_iq2_xs sign-mask table A/B
2026-11-21 15:22:03 UTC [A-PF] release card 0: lay3 rc=0
2026-11-21 15:22:13 UTC [A-LY] release card 0: ups_iq2_xs sign-mask table A/B rc=0
2026-11-21 15:24:01 UTC [A-KF] claim card 0 ~1 min: round11 slice all, max250s
2026-11-21 15:24:18 UTC [A-HS] claim card 0 ~3 min: cold-f1.log
2026-11-21 15:24:18 UTC [A-KF] released card 0: round11 slice all rc=0
2026-11-21 15:24:18 UTC [A-KF] claim card 0 ~20s: round11 health-post, cap60s
2026-11-21 15:24:22 UTC [A-HS] released card 0 (rc=0): cold-f1.log
2026-11-21 15:24:33 UTC [A-KF] released card 0: round11 health-post rc=0
2026-11-21 15:26:10 UTC [A-LY] claim card 0 ~3 min: ups_iq2_xs split SLM/global grid A/B
2026-11-21 15:26:15 UTC [A-LY] release card 0: ups_iq2_xs split SLM/global grid A/B rc=0
2026-11-21 15:31:50 UTC [A-S] claim card 0 ~2 min: slice + tail gate fn-a3 82d548e8
2026-11-21 15:33:50 UTC [A-LY] claim card 0 ~3 min: ups_iq2_xs packed 2-bit grid A/B
2026-11-21 15:34:24 UTC [A-S] released card 0: slice gate rc=0
2026-11-21 15:34:33 UTC [A-LY] release card 0: ups_iq2_xs packed 2-bit grid A/B rc=0
2026-11-21 15:36:09 UTC [A-HS] claim card 0 ~3 min: hc-cold d1
2026-11-21 15:36:14 UTC [A-HS] released card 0 (rc=0): hc-cold d1
2026-11-21 15:39:49 UTC [A-KF] claim card 0 ~20s: round11 health-pre, cap60s
2026-11-21 15:40:06 UTC [A-KF] released card 0: round11 health-pre rc=0
2026-11-21 15:40:06 UTC [A-KF] claim card 0 ~2 min: round11 final, cap285s
2026-11-21 15:40:06 UTC [A-KF] acquired card 0 ~2 min: round11 final
2026-11-21 15:40:27 UTC [A-KF] released card 0: round11 final rc=0
2026-11-21 15:40:27 UTC [A-KF] claim card 0 ~20s: round11 health-post, cap60s
2026-11-21 15:40:41 UTC [A-KF] released card 0: round11 health-post rc=0
2026-11-21 15:40:58 UTC [A-KF] claim card 0 ~20s: round11 health-pre, cap60s
2026-11-21 15:41:07 UTC [A-LY] claim card 0 ~3 min: q_apply mul-free A/B
2026-11-21 15:41:13 UTC [A-KF] released card 0: round11 health-pre rc=0
2026-11-21 15:41:28 UTC [A-LY] release card 0: q_apply mul-free A/B rc=0
2026-11-21 15:43:16 UTC [A-PF] claim card 0 ~150s: lay4
2026-11-21 15:43:41 UTC [A-PF] release card 0: lay4 rc=0
2026-11-21 15:45:21 UTC [A-HS] claim card 0 ~3 min: hc-cold d2
2026-11-21 15:45:25 UTC [A-HS] released card 0 (rc=0): hc-cold d2
2026-11-21 15:45:54 UTC [A-HS] claim card 0 ~3 min: slice gate fz1
2026-11-21 15:46:29 UTC [A-HS] released card 0 (rc=0): slice gate fz1
2026-11-21 15:47:39 UTC [A-LY] claim card 0 ~3 min: hardware half conversion A/B
2026-11-21 15:48:00 UTC [A-LY] release card 0: hardware half conversion A/B rc=0
2026-11-21 15:49:45 UTC [A-KF] claim card 0 ~1 min: round11 slice all, max250s
2026-11-21 15:49:53 UTC [A-HS] claim card 0 ~3 min: hc-cold hot
2026-11-21 15:50:02 UTC [A-KF] released card 0: round11 slice all rc=0
2026-11-21 15:50:02 UTC [A-KF] claim card 0 ~20s: round11 health-post, cap60s
2026-11-21 15:50:06 UTC [A-HS] released card 0 (rc=0): hc-cold hot
2026-11-21 15:50:18 UTC [A-KF] released card 0: round11 health-post rc=0
2026-11-21 15:52:10 UTC [A-PF] claim card 0 ~170s: lay5
2026-11-21 15:52:37 UTC [A-PF] release card 0: lay5 rc=0
2026-11-21 15:53:55 UTC [A-LY] claim card 0 ~3 min: iq2-only hardware half A/B
2026-11-21 15:54:15 UTC [A-LY] release card 0: iq2-only hardware half A/B rc=0
2026-11-21 15:54:44 UTC [A-LY] claim card 0 ~3 min: iq2 hardware-half gate
2026-11-21 15:55:45 UTC [A-LY] release card 0: iq2 hardware-half gate rc=0
2026-11-21 15:56:51 UTC [A-HS] claim card 0 ~3 min: slice gate pf1
2026-11-21 15:57:24 UTC [A-HS] released card 0 (rc=0): slice gate pf1
2026-11-21 15:57:54 UTC [A-PF] claim card 0 ~170s: lay6
2026-11-21 15:59:40 UTC [A-LY] claim card 0 ~3 min: ups 2 tiles per WG A/B
2026-11-21 16:00:11 UTC [A-PF] release card 0: lay6 rc=0
2026-11-21 16:00:32 UTC [A-LY] release card 0: ups 2 tiles per WG A/B rc=0
2026-11-21 16:02:13 UTC [A-HS] claim card 0 ~3 min: slice gate pf2
2026-11-21 16:02:46 UTC [A-HS] released card 0 (rc=0): slice gate pf2
2026-11-21 16:03:43 UTC [A-PF] claim card 0 ~170s: lay7
2026-11-21 16:04:10 UTC [A-PF] release card 0: lay7 rc=0
2026-11-21 16:11:56 UTC [A-S] claim card 0 ~2 min: slice + tail gate fn-a3 c78c82b7
2026-11-21 16:12:01 UTC [A-HS] claim card 0 ~3 min: slice gate ar1
2026-11-21 16:14:32 UTC [A-S] released card 0: slice gate rc=0
2026-11-21 16:15:06 UTC [A-HS] released card 0 (rc=0): slice gate ar1
2026-11-21 16:16:33 UTC [A-KF] claim card 0 ~20s: round11 health-pre, cap60s
2026-11-21 16:16:49 UTC [A-KF] released card 0: round11 health-pre rc=0
2026-11-21 16:16:49 UTC [A-KF] claim card 0 ~2 min: round11 final, cap285s
2026-11-21 16:16:49 UTC [A-KF] acquired card 0 ~2 min: round11 final
2026-11-21 16:17:10 UTC [A-KF] released card 0: round11 final rc=0
2026-11-21 16:17:10 UTC [A-KF] claim card 0 ~20s: round11 health-post, cap60s
2026-11-21 16:17:24 UTC [A-KF] released card 0: round11 health-post rc=0
2026-11-21 16:18:47 UTC [A-KF] claim card 0 ~20s: round11 health-pre, cap60s
2026-11-21 16:19:02 UTC [A-KF] released card 0: round11 health-pre rc=0
2026-11-21 16:19:35 UTC [A-S] claim card 0 ~2 min: slice + tail gate fn-a3 5e92440d
2026-11-21 16:21:51 UTC [A-HS] claim card 0 ~3 min: slice gate gs1
2026-11-21 16:22:09 UTC [A-S] released card 0: slice gate rc=0
2026-11-21 16:22:42 UTC [A-HS] released card 0 (rc=0): slice gate gs1
2026-11-21 16:24:27 UTC [A-KF] claim card 0 ~1 min: round11 slice all, max250s
2026-11-21 16:24:38 UTC [A-LY] claim card 0 ~3 min: ups 16-lane non-bitwise A/B
2026-11-21 16:24:45 UTC [A-KF] released card 0: round11 slice all rc=0
2026-11-21 16:24:45 UTC [A-KF] claim card 0 ~20s: round11 health-post, cap60s
2026-11-21 16:25:02 UTC [A-KF] released card 0: round11 health-post rc=0
2026-11-21 16:25:10 UTC [A-LY] release card 0: ups 16-lane non-bitwise A/B rc=0
2026-11-21 16:27:09 UTC [A-HS] claim card 0 ~3 min: slice gate gs2
2026-11-21 16:27:41 UTC [A-HS] released card 0 (rc=0): slice gate gs2
2026-11-21 16:29:26 UTC [A-LY] claim card 0 ~3 min: ups 4-lane non-bitwise A/B
2026-11-21 16:29:47 UTC [A-LY] release card 0: ups 4-lane non-bitwise A/B rc=0
2026-11-21 16:31:43 UTC [A-HS] claim card 0 ~3 min: slice gate gj1
2026-11-21 16:32:27 UTC [A-HS] released card 0 (rc=0): slice gate gj1
2026-11-21 16:32:54 UTC [A-HS] claim card 0 ~3 min: slice gate gjt1
2026-11-21 16:33:34 UTC [A-PF] claim card 0 ~180s: sw1
2026-11-21 16:35:23 UTC [A-HS] released card 0 (rc=0): slice gate gjt1
2026-11-21 16:35:23 UTC [A-HS] claim card 0 ~3 min: slice gate gjt2
2026-11-21 16:36:43 UTC [A-LY] claim card 0 ~3 min: MTP tail slice R1 profile
2026-11-21 16:37:52 UTC [A-HS] released card 0 (rc=0): slice gate gjt2
2026-11-21 16:38:23 UTC [A-PF] release card 0: sw1 rc=0
2026-11-21 16:38:38 UTC [A-LY] release card 0: MTP tail slice R1 profile rc=0
2026-11-21 16:38:50 UTC [A-LY] claim card 0 ~3 min: MTP tail slice R1 profile
2026-11-21 16:38:53 UTC [A-PF] claim card 0 ~180s: sw2
2026-11-21 16:39:03 UTC [A-KF] claim card 0 ~20s: round11 health-pre, cap60s
2026-11-21 16:39:18 UTC [A-KF] released card 0: round11 health-pre rc=0
2026-11-21 16:39:18 UTC [A-KF] claim card 0 ~2 min: round11 final, cap285s
2026-11-21 16:41:07 UTC [A-LY] release card 0: MTP tail slice R1 profile rc=0
2026-11-21 16:44:08 UTC [A-PF] release card 0: sw2 rc=124
2026-11-21 16:44:09 UTC [A-KF] acquired card 0 ~2 min: round11 final
2026-11-21 16:44:32 UTC [A-KF] released card 0: round11 final rc=0
2026-11-21 16:44:32 UTC [A-KF] claim card 0 ~20s: round11 health-post, cap60s
2026-11-21 16:44:55 UTC [A-KF] released card 0: round11 health-post rc=0
2026-11-21 16:45:44 UTC [A-PF] claim card 0 ~180s: sw3
2026-11-21 16:46:15 UTC [A-PF] release card 0: sw3 rc=0
2026-11-21 16:46:24 UTC [A-PF] claim card 0 ~180s: sw4
2026-11-21 16:47:03 UTC [A-PF] release card 0: sw4 rc=0
2026-11-21 16:48:28 UTC [A-S] claim card 0 ~2 min: slice + tail gate fn-a3 e8e21a4f
2026-11-21 16:50:53 UTC [A-LY] claim card 0 ~3 min: MTP tail rmsnorm+chainhead
2026-11-21 16:51:02 UTC [A-S] released card 0: slice gate rc=0
2026-11-21 16:52:31 UTC [A-KF] claim card 0 ~20s: round11 health-pre, cap60s
2026-11-21 16:52:46 UTC [A-KF] released card 0: round11 health-pre rc=0
2026-11-21 16:52:46 UTC [A-KF] claim card 0 ~2 min: round11 final, cap285s
2026-11-21 16:53:05 UTC [A-LY] release card 0: MTP tail rmsnorm+chainhead rc=0
2026-11-21 16:53:06 UTC [A-KF] acquired card 0 ~2 min: round11 final
2026-11-21 16:53:18 UTC [A-PF] claim card 0 ~180s: ab1
2026-11-21 16:53:28 UTC [A-KF] released card 0: round11 final rc=0
2026-11-21 16:53:28 UTC [A-KF] claim card 0 ~20s: round11 health-post, cap60s
2026-11-21 16:53:44 UTC [A-KF] released card 0: round11 health-post rc=0
2026-11-21 16:54:13 UTC [A-PF] release card 0: ab1 rc=0
2026-11-21 16:55:36 UTC [A-KF] claim card 0 ~20s: round11 health-pre, cap60s
2026-11-21 16:55:50 UTC [A-KF] released card 0: round11 health-pre rc=0
2026-11-21 16:57:20 UTC [A-HS] claim card 0 ~3 min: slice gate ff1
2026-11-21 16:57:51 UTC [A-HS] released card 0 (rc=0): slice gate ff1
2026-11-21 16:58:25 UTC [A-KF] claim card 0 ~1 min: round11 slice default, max250s
2026-11-21 16:58:34 UTC [A-KF] released card 0: round11 slice default rc=0
2026-11-21 16:58:34 UTC [A-KF] claim card 0 ~20s: round11 health-post, cap60s
2026-11-21 16:58:50 UTC [A-KF] released card 0: round11 health-post rc=0
2026-11-21 16:59:27 UTC [A-PF] claim card 0 ~120s: ab2
2026-11-21 16:59:39 UTC [A-KF] claim card 0 ~20s: round11 health-pre, cap60s
2026-11-21 16:59:50 UTC [A-LY] claim card 0 ~3 min: invariance-mtp + decode-check
2026-11-21 16:59:52 UTC [A-PF] release card 0: ab2 rc=0
2026-11-21 16:59:58 UTC [A-KF] released card 0: round11 health-pre rc=0
2026-11-21 16:59:58 UTC [A-KF] claim card 0 ~2 min: round11 final, cap285s
2026-11-21 17:00:02 UTC [A-LY] release card 0: invariance-mtp + decode-check rc=0
2026-11-21 17:00:04 UTC [A-KF] acquired card 0 ~2 min: round11 final
2026-11-21 17:00:26 UTC [A-KF] released card 0: round11 final rc=0
2026-11-21 17:00:26 UTC [A-KF] claim card 0 ~20s: round11 health-post, cap60s
2026-11-21 17:00:50 UTC [A-KF] released card 0: round11 health-post rc=0
2026-11-21 17:02:47 UTC [A-LY] claim card 0 ~3 min: draft head RP/groups sweep
2026-11-21 17:02:55 UTC [A-LY] release card 0: draft head RP/groups sweep rc=0
2026-11-21 17:05:37 UTC [A-HS] claim card 0 ~3 min: slice gate ff2
2026-11-21 17:06:10 UTC [A-HS] released card 0 (rc=0): slice gate ff2
2026-11-21 17:06:23 UTC [A-PF] claim card 0 ~120s: ab3
2026-11-21 17:06:43 UTC [A-PF] release card 0: ab3 rc=0
2026-11-21 17:07:28 UTC [A-LY] claim card 0 ~3 min: draft head groups sweep all R
2026-11-21 17:07:34 UTC [A-LY] release card 0: draft head groups sweep all R rc=0
2026-11-21 17:12:12 UTC [A-S] claim card 0 ~2 min: slice + tail gate fn-a3 5cfa39f7
2026-11-21 17:13:51 UTC [A-LY] claim card 0 ~3 min: MTP norm/argmax gate
2026-11-21 17:14:43 UTC [A-S] released card 0: slice gate rc=0
2026-11-21 17:15:40 UTC [A-LY] release card 0: MTP norm/argmax gate rc=0
2026-11-21 17:21:39 UTC [A] note: ORCHESTRATOR owner decision relayed: INK stopped, card 1 returns to TensorFold after the [INK] release + health line. First two-card user = A-S quick serve bench of the fn-a3 tip under PRIORITY.hold (C1/C2/C4/C8 + prefill). Lead B Codex workers (arfuse/park/lane/kv8/ttft): no two-card claims until that bench releases.
2026-11-21 17:22:13 UTC [A-S] claim card 0 ~2 min: slice + tail gate fn-a3 2f022b5f
2026-11-21 17:23:27 UTC [INK] release card 1: Inkling eval stopped by owner; server down; xpu-health --card 1 HEALTHY
2026-11-21 17:24:48 UTC [A-S] released card 0: slice gate rc=0
2026-11-21 17:25:08 UTC [A-S] queued: two-card quick serve bench fn-a3 2f022b5f (int-rf) under PRIORITY.hold, ~25-40 min, cards 0+1
2026-11-21 17:26:13 UTC [A-HS] claim card 0 ~3 min: slice gate ff4
2026-11-21 17:26:45 UTC [A-HS] released card 0 (rc=0): slice gate ff4
2026-11-21 17:29:02 UTC [A-I] claim cards 0+1 ~3 min: smoke smoke-int-rf
2026-11-21 17:32:41 UTC [A-I] released cards 0+1
2026-11-21 17:32:41 UTC [A-S] claim cards 0+1 ~15 min: int-rf bounded policy serve/no-MTP/pre-post health; CPU quiet acquired before GPU
2026-11-21 17:35:14 UTC [A-S] released cards 0+1: int-rf rc=0
2026-11-21 17:35:18 UTC [A-HS] claim card 0 ~3 min: slice gate ff5
2026-11-21 17:35:19 UTC [A-PF] claim card 0 ~170s: pf1
2026-11-21 17:35:50 UTC [A-HS] released card 0 (rc=0): slice gate ff5
2026-11-21 17:36:10 UTC [A-PF] release card 0: pf1 rc=0
2026-11-21 17:37:44 UTC [A-LY] claim card 0 ~3 min: MTP tail profile on 2f022b5f
2026-11-21 17:39:45 UTC [A-LY] release card 0: MTP tail profile on 2f022b5f rc=0
2026-11-21 17:40:20 UTC [A-I] claim cards 0+1 ~3 min: smoke smoke-int-rf2
2026-11-21 17:43:23 UTC [A-I] released cards 0+1
2026-11-21 17:43:24 UTC [A-S] claim cards 0+1 ~15 min: int-rf2 bounded policy serve/no-MTP/pre-post health; CPU quiet acquired before GPU
2026-11-21 17:45:57 UTC [A-S] released cards 0+1: int-rf2 rc=0
2026-11-21 17:46:02 UTC [A-PF] claim card 0 ~120s: pf2
2026-11-21 17:46:04 UTC [A-HS] claim card 0 ~3 min: slice gate ff6
2026-11-21 17:46:23 UTC [A-HS] released card 0 (script stopped during wait; gate ff6 aborted, no slice run)
2026-11-21 17:46:30 UTC [A-PF] release card 0: pf2 rc=0
2026-11-21 17:46:33 UTC [A-HS] claim card 0 ~3 min: slice gate ff6b
2026-11-21 17:47:05 UTC [A-HS] released card 0 (rc=0): slice gate ff6b
2026-11-21 17:47:23 UTC [A-HS] claim card 0 ~3 min: slice gate fft0
2026-11-21 17:48:37 UTC [A-LY] claim card 0 ~3 min: MTP router+shared_down
2026-11-21 17:49:53 UTC [A-HS] released card 0 (rc=0): slice gate fft0
2026-11-21 17:49:53 UTC [A-HS] claim card 0 ~3 min: slice gate fft1
2026-11-21 17:51:21 UTC [A-LY] release card 0: MTP router+shared_down rc=cancelled (killed while waiting for the lock; never acquired)
2026-11-21 17:51:21 UTC [A-LY] claim card 0 ~3 min: MTP router+shared_down
2026-11-21 17:51:24 UTC [A] note: ORCHESTRATOR 17:55 owner decision relayed to all workers (incl. Lead B Codex workers via this line): switch to SERVING; land/park work with HANDOFF; ALL GPU work stops by 18:55 UTC; after that no claims on either card (production serve + soak); no CPU-heavy work while the serve is live. The A-S card1-queue is cancelled (PRIORITY.hold held by A-S for the wrap-up).
2026-11-21 17:52:13 UTC [A-HS] released card 0 (rc=1): slice gate fft1
2026-11-21 17:54:12 UTC [A-LY] release card 0: MTP router+shared_down rc=0
2026-11-21 17:54:38 UTC [A-S] claim card 0 ~2 min: slice + tail gate fn-a3 e87b5074
2026-11-21 17:55:19 UTC [A] removed PRIORITY.hold (orchestrator 18:00 wrap-up); will not recreate. card1-queue cancelled and parked as H/r11/card1-queue.sh.parked-wrapup. In flight: A-S card-0 slice gate (a8b678ec), then A cutover capability probes on cards 0+1 (<=3 x 10 min, port 18151), all done before 18:55.
2026-11-21 17:55:38 UTC [A-PF] claim card 0 ~170s: pf3
2026-11-21 17:57:07 UTC [A-PF] release card 0: pf3 never acquired (waited on A-S), cancelled at serve switch
2026-11-21 17:57:11 UTC [A-S] released card 0: slice gate rc=0
2026-11-21 17:57:11 UTC [A-S] queued: H/r11 card1-queue (flag A/Bs + KL teacher + HS prefetch-AR + MTP arms + R dump) on fn-a3 e87b5074, cards 0+1, ~45-60 min, leases <=280s, PRIORITY.hold
2026-11-21 17:57:30 UTC [A] claim cards 0+1 ~10 min: cutover probe c200-desc (fn-a3 e87b5074 binary, ctx 200000 par 4, port 18151)
2026-11-21 19:03:34 UTC [INT] claim cards 0+1 ~15 min: a3-e87b5074 bounded policy serve/no-MTP/pre-post health; CPU quiet acquired before GPU
2026-11-21 19:06:09 UTC [INT] released cards 0+1: a3-e87b5074 rc=0
2026-11-21 19:12:59 UTC [API] claim cards 0+1 <=5 min: API quick gate on port 18154 (dd-api 9b4c7691 xpu-native, TF_FIT_MAX_TOKENS/TF_STRUCTURED_FALLBACK), api_gate.py all
2026-11-21 19:13:03 UTC [VIS] claim cards 0+1 ~5 min: vision quick gate a2e63f48 (port 18153, classic KV)
2026-11-21 19:15:46 UTC [VIS] released cards 0+1: vision quick gate classic rc=0
2026-11-21 19:16:08 UTC [VIS] claim cards 0+1 ~5 min: vision quick gate a2e63f48 retry ctx 16384 (port 18153)
2026-11-21 19:17:58 UTC [SAMP] claim cards 0+1 ~5 min: fn-samp 28030ff0 sampling quick gate (port 18152, samp_session.sh)
2026-11-21 19:18:07 UTC [API] released cards 0+1: API quick gate 4/5 PASS (reasoning, tools, forced, budget); edge FAIL on max_tokens clamp (500); xpu-health HEALTHY
2026-11-21 19:18:07 UTC [KV] claim cards 0+1 ~5 min: KV quick 1: load 200000x4 paged int8 desc, budget, p0 C4==solo, 40K (port 18151, bef2f1a8)
2026-11-21 19:20:08 UTC [SAMP] released cards 0+1: fn-samp 28030ff0 sampling quick gate rc=0
2026-11-21 19:22:22 UTC [VIS] released cards 0+1: vision quick gate ctx16384 rc=0
2026-11-21 19:22:30 UTC [VIS] claim cards 0+1 ~5 min: vision quick gate a2e63f48 XPU_FN_KV=int8 ctx 65536 (port 18153)
2026-11-21 19:23:27 UTC [API] claim cards 0+1 <=5 min: API quick gate rerun on port 18154 (dd-api 32b27d57, clamp fix)
2026-11-21 19:24:38 UTC [KV] released cards 0+1 (rc=1): KV quick 1: load 200000x4 paged int8 desc, budget, p0 C4==solo, 40K
2026-11-21 19:27:15 UTC [VIS] released cards 0+1: vision quick gate int8 rc=0
2026-11-21 19:29:19 UTC [API] released cards 0+1: API quick gate PASS 5/5 on dd-api 32b27d57 (port 18154); xpu-health HEALTHY
2026-11-21 19:39:47 UTC [VIS] claim cards 0+1 ~5 min: vision gate 2b552f0c desc+paged int8 ctx 65536 + extras (port 18153)
2026-11-21 19:42:43 UTC [VIS] released cards 0+1: vision gate desc+paged rc=0
2026-11-21 19:43:26 UTC [VIS] claim cards 0+1 ~4 min: long-prompt TTFT text vs image, 2b552f0c desc+paged (port 18153)
2026-11-21 19:44:37 UTC [KV] claim cards 0+1 ~5 min: KV quick 2: 200000x4 paged int8 desc + RAM cache: budget, p0 C4==solo, prefill 8K/32K, cache repeat (port 18151, bef2f1a8)
2026-11-21 19:46:35 UTC [VIS] released cards 0+1: long-prompt TTFT rc=0
2026-11-21 19:49:00 UTC [KV] released cards 0+1 (rc=1): KV quick 2: 200000x4 paged int8 desc + RAM cache: budget, p0 C4==solo, prefill 8K/32K, cache repeat
2026-11-21 19:49:35 UTC [KV] claim cards 0+1 ~5 min: KV quick 3 A/B control: legacy bf16 desc ctx 16384, 2K/8K prefill + needle (port 18151, bef2f1a8)
2026-11-21 19:51:08 UTC [SAMP] claim cards 0+1 ~4 min: SAMP profile baseline (host merge, argmax drafts) (port 18152)
2026-11-21 19:52:09 UTC [KV] released cards 0+1 (rc=1): KV quick 3 A/B control: legacy bf16 desc ctx 16384, 2K/8K prefill + needle
2026-11-21 19:52:42 UTC [KV] claim cards 0+1 ~5 min: KV quick 4: paged int8 desc 200000x4 + cache: prefill 2K/8K/32K, cache repeat (port 18151, bef2f1a8)
2026-11-21 19:55:15 UTC [SAMP] released cards 0+1: SAMP profile baseline (host merge, argmax drafts) rc=0
2026-11-21 19:58:26 UTC [KV] released cards 0+1 (rc=0): KV quick 4: paged int8 desc 200000x4 + cache: prefill 2K/8K/32K, cache repeat
2026-11-21 19:58:35 UTC [KV] claim cards 0+1 ~5 min: KV quick 5: paged int8 desc 200000x4: 90K prompt + 256 decode, 20K solo (port 18151, bef2f1a8)
2026-11-21 19:59:03 UTC [hotschmoe-dd-tf] claim cards 0+1 <= 300 s: gpu_window prod fb0e6006-a (port 18150, steps P0-identity P1-bench S1-samp VIS-T0 VIS-I VIS-T1)
2026-11-21 20:03:19 UTC [KV] released cards 0+1 (rc=1): KV quick 5: paged int8 desc 200000x4: 90K prompt + 256 decode, 20K solo
2026-11-21 20:04:26 UTC [KV] claim cards 0+1 ~5 min: KV quick 6: paged int8 desc: 48K and 70K needle (90K returned EOS) (port 18151, bef2f1a8)
2026-11-21 20:06:06 UTC [hotschmoe-dd-tf] released cards 0+1: gpu_window prod fb0e6006-a rc=0
2026-11-21 20:06:16 UTC [hotschmoe-dd-tf] claim cards 0+1 <= 300 s: gpu_window prod fb0e6006-b (port 18150, steps P0-identity API1-reasoning API2-tools API3-forced API4-budget API5-edge P9-functional)
2026-11-21 20:11:01 UTC [KV] released cards 0+1 (rc=124): KV quick 6: paged int8 desc: 48K and 70K needle (90K returned EOS)
2026-11-21 20:11:08 UTC [KV] claim cards 0+1 ~1 min: post-timeout xpu-health (q6 lease hit its 295 s timeout during teardown)
2026-11-21 20:11 UTC [VIS] claim cards 0+1 ~4 min: vision gate 097ff79c desc+paged int8 ctx 65536 + JPEG-EXIF/WebP (port 18153)
2026-11-21 20:11:50 UTC [KV] released cards 0+1: post-timeout xpu-health
2026-11-21 20:12:40 UTC [KV] claim cards 0+1 ~5 min: KV quick 7: paged int8 desc: 70K filler with the note at the end (EOS at >=70K diagnosis) (port 18151, bef2f1a8)
2026-11-21 20:14 UTC [VIS] released cards 0+1: vision gate formats rc=0
2026-11-21 20:17:35 UTC [SAMP] claim cards 0+1 ~4 min: SAMP profile: device merge + coupled drafts + penalty gather, then gate (port 18152)
2026-11-21 20:18:34 UTC [KV] released cards 0+1 (rc=0): KV quick 7: paged int8 desc: 70K filler with the note at the end (EOS at >=70K diagnosis)
2026-11-21 20:18:41 UTC [KV] claim cards 0+1 ~5 min: KV quick 8: bf16page desc control: 70K needle near start (port 18151, bef2f1a8)
2026-11-21 20:21:43 UTC [SAMP] released cards 0+1: SAMP profile: device merge + coupled drafts + penalty gather, then gate rc=0
2026-11-21 20:22:00 UTC [SAMP] claim cards 0+1 ~4 min: SAMP device-merge exactness check (gate with host-merge cross-check) (port 18152)
2026-11-21 20:24:16 UTC [hotschmoe-dd-tf] released cards 0+1: gpu_window prod fb0e6006-b rc=1
2026-11-21 20:24:50 UTC [hotschmoe-dd-tf] claim cards 0+1 <= 300 s: gpu_window prod fb0e6006-c (port 18150, steps P0-identity P9-functional)
2026-11-21 20:26:48 UTC [SAMP] released cards 0+1: SAMP device-merge exactness check (gate with host-merge cross-check) rc=0
2026-11-21 20:30:04 UTC [KV] released cards 0+1 (rc=1): KV quick 8: bf16page desc control: 70K needle near start
2026-11-21 20:30:16 UTC [KV] claim cards 0+1 ~5 min: KV quick 9: int8 paged desc WITHOUT eviction/mirror: prefill 8K/32K A/B (port 18151, bef2f1a8)
2026-11-21 20:32:16 UTC [SAMP] claim cards 0+1 ~4 min: SAMP final build: gate + merge/gather exactness check (port 18152)
2026-11-21 20:33:21 UTC [KV] released cards 0+1 (rc=0): KV quick 9: int8 paged desc WITHOUT eviction/mirror: prefill 8K/32K A/B
2026-11-21 20:34:25 UTC [KV] claim cards 0+1 ~5 min: KV quick 10: 20K solo then 4 x 20K concurrent on 3 physical slots (park/restore bitwise) (port 18151, bef2f1a8)
2026-11-21 20:36:23 UTC [hotschmoe-dd-tf] released cards 0+1: gpu_window prod fb0e6006-c rc=0
2026-11-21 20:40:11 UTC [SAMP] released cards 0+1: SAMP final build: gate + merge/gather exactness check rc=0
2026-11-21 20:40:23 UTC [SAMP] claim cards 0+1 ~4 min: SAMP final build timing (C1/C4 x greedy/topk1/t0.6/instruct) (port 18152)
2026-11-21 20:44:10 UTC [KV] released cards 0+1 (rc=1): KV quick 10: 20K solo then 4 x 20K concurrent on 3 physical slots (park/restore bitwise)
2026-11-21 20:44:23 UTC [KV] claim cards 0+1 ~5 min: KV quick 11: solo 20K seeds 2 and 3 (concurrent seed 2 returned EOS after 2 parks during prefill) (port 18151, bef2f1a8)
2026-11-21 20:47:09 UTC [SAMP] released cards 0+1: SAMP final build timing (C1/C4 x greedy/topk1/t0.6/instruct) rc=0
2026-11-21 20:50:16 UTC [KV] released cards 0+1 (rc=1): KV quick 11: solo 20K seeds 2 and 3 (concurrent seed 2 returned EOS after 2 parks during prefill)
2026-11-21 20:55:23 UTC [hotschmoe-dd-tf] claim cards 0+1 <= 300 s: gpu_window prod 7d08051e (port 18150, steps P0-identity S1-samp VIS-T0 VIS-I VIS-T1 P9-functional)
2026-11-21 20:58:16 UTC [hotschmoe-dd-tf] released cards 0+1: gpu_window prod 7d08051e rc=0
2026-11-21 21:05:01 UTC [KV] claim cards 0+1 ~6 min: Strata 0.2.16 reference (container strata-ref, port 18151): needle/EOS + 32K/90K TTFT cold+repeat
2026-11-21 21:09:21 UTC [KV] released cards 0+1: Strata reference (SESSION rc=0 health_post=0 gpu-run: [KV] done in 32s (exit 0))
2026-11-21 21:09:36 UTC [KV] claim cards 0+1 ~5 min: KV quick 12: f70795f7 (sha 4ddf7dac): p0 C4 vs bef2f1a8 ref; chat-mode seed-2 20K and needle-early 70K (Strata-matched) (port 18151, bef2f1a8)
2026-11-21 21:13:29 UTC [KV] released cards 0+1 (rc=0): KV quick 12: f70795f7 (sha 4ddf7dac): p0 C4 vs bef2f1a8 ref; chat-mode seed-2 20K and needle-early 70K (Strata-matched)
2026-11-21 21:13:41 UTC [KV] claim cards 0+1 ~5 min: KV quick 13: f70795f7: 90K chat needle + RAM prompt cache repeat (port 18151, bef2f1a8)
2026-11-21 21:17:07 UTC [KV] released cards 0+1 (rc=0): KV quick 13: f70795f7: 90K chat needle + RAM prompt cache repeat
2026-11-21 21:17:07 UTC [KV] claim cards 0+1 ~5 min: KV quick 14: f70795f7: 20K solo + 4 x 20K concurrent on 3 physical slots (park/restore) + parks gate (port 18151, bef2f1a8)
2026-11-21 21:20:14 UTC [KV] released cards 0+1 (rc=1): KV quick 14: f70795f7: 20K solo + 4 x 20K concurrent on 3 physical slots (park/restore) + parks gate
2026-11-21 21:20:40 UTC [KV] claim cards 0+1 ~5 min: KV quick 15: f70795f7: long-answer 20K solo + 4 x 20K concurrent on 3 physical slots (park/restore bitwise) + parks gate (port 18151, bef2f1a8)
2026-11-21 21:23:38 UTC [KV] released cards 0+1 (rc=1): KV quick 15: f70795f7: long-answer 20K solo + 4 x 20K concurrent on 3 physical slots (park/restore bitwise) + parks gate
2026-11-21 21:23:54 UTC [KV] claim cards 0+1 ~5 min: KV quick 16: f70795f7: 20K solo + 4 x 20K concurrent, ~1200-token copy replies (park/restore bitwise) + parks gate (port 18151, bef2f1a8)
2026-11-21 21:27:13 UTC [KV] released cards 0+1 (rc=1): KV quick 16: f70795f7: 20K solo + 4 x 20K concurrent, ~1200-token copy replies (park/restore bitwise) + parks gate
2026-11-21 21:27:27 UTC [KV] claim cards 0+1 ~5 min: KV quick 17: f70795f7: 20K solo + 5 x 20K concurrent (4 cold > 3 physical slots), 1200-token replies, park/restore bitwise (port 18151, bef2f1a8)
2026-11-21 21:31:14 UTC [KV] released cards 0+1 (rc=0): KV quick 17: f70795f7: 20K solo + 5 x 20K concurrent (4 cold > 3 physical slots), 1200-token replies, park/restore bitwise
2026-11-21 21:31:40 UTC [KV] claim cards 0+1 ~5 min: KV quick 18: solo refs seeds 1-4 at 20K, 1200-token replies (for park bitwise) (port 18151, bef2f1a8)
2026-11-21 21:35:06 UTC [KV] released cards 0+1 (rc=0): KV quick 18: solo refs seeds 1-4 at 20K, 1200-token replies (for park bitwise)
2026-11-21 21:35:06 UTC [KV] claim cards 0+1 ~5 min: KV quick 19: 5 x 20K concurrent, 1200-token replies, every stream vs its solo ref (park/restore bitwise) (port 18151, bef2f1a8)
2026-11-21 21:39:08 UTC [KV] released cards 0+1 (rc=0): KV quick 19: 5 x 20K concurrent, 1200-token replies, every stream vs its solo ref (park/restore bitwise)
2026-11-21 21:39:18 UTC [KV] claim cards 0+1 ~5 min: KV quick 20: f70795f7 prefill rates 8K / 32K (prod KV mode) (port 18151, bef2f1a8)
2026-11-21 21:42:01 UTC [KV] released cards 0+1 (rc=0): KV quick 20: f70795f7 prefill rates 8K / 32K (prod KV mode)
2026-11-21 21:44:33 UTC [KV] claim cards 0+1 ~5 min: KV quick 21: XPU_FN_EVICT_SLOTS=4 at 200000: 5 x 20K 1200-token replies vs q18 solo refs (expect ~0 parks) (port 18151, bin 4ddf7dac)
2026-11-21 21:49:17 UTC [KV] released cards 0+1 (rc=0): KV quick 21: XPU_FN_EVICT_SLOTS=4 at 200000: 5 x 20K 1200-token replies vs q18 solo refs (expect ~0 parks)
2026-11-21 21:49:29 UTC [KV] claim cards 0+1 ~5 min: KV quick 22: EVICT_SLOTS=4 under page pressure (KV_PAGES=150, ctx 65536): 5 x 20K 1200-token vs q18 refs (port 18151, bin 4ddf7dac)
2026-11-21 21:54:07 UTC [KV] released cards 0+1 (rc=0): KV quick 22: EVICT_SLOTS=4 under page pressure (KV_PAGES=150, ctx 65536): 5 x 20K 1200-token vs q18 refs
2026-11-21 21:57:33 UTC [hotschmoe-dd-tf] claim cards 0+1 <= 300 s: gpu_window prod 2cb54319 (port 18150, steps P0-identity KV0-budget KV1-short-c4 S1-samp VIS-I VIS-F P9-functional)
2026-11-21 22:00:32 UTC [hotschmoe-dd-tf] released cards 0+1: gpu_window prod 2cb54319 rc=0
2026-11-21 22:09:25 UTC [hotschmoe-dd-tf] claim cards 0+1 ~15 min: regress bounded policy serve/no-MTP/pre-post health; CPU quiet acquired before GPU
2026-11-21 22:11:52 UTC [hotschmoe-dd-tf] released cards 0+1: regress rc=0
2026-11-21 22:11:52 UTC [hotschmoe-dd-tf] claim cards 0+1 <= 2400 s: gpu_window prod final-2cb54319 (port 18150)
2026-11-21 22:18:44 UTC [hotschmoe-dd-tf] released cards 0+1: gpu_window prod final-2cb54319 rc=0
2026-11-21 22:58:23 UTC [KV] claim cards 0+1 ~5 min: KVP1: paged int8 prod KV + prefill kernel profile, 8K + 32K (port 18151, bin 65d17dbb)
2026-11-21 22:59:40 UTC [SCHED] claim cards 0+1 ~5 min: a1 prod config quick bench, 4 lanes, EVICT_SLOTS=4 (port 18153, frozen 2cb54319)
2026-11-21 23:00:47 UTC [KV] released cards 0+1 (rc=0): KVP1: paged int8 prod KV + prefill kernel profile, 8K + 32K
2026-11-21 23:00:47 UTC [KV] claim cards 0+1 ~5 min: KVP2: legacy bf16 desc (KV off, ctx 16384) + prefill kernel profile, 8K (port 18151, bin 65d17dbb)
2026-11-21 23:03:01 UTC [KV] released cards 0+1 (rc=0): KVP2: legacy bf16 desc (KV off, ctx 16384) + prefill kernel profile, 8K
2026-11-21 23:03:51 UTC [hotschmoe-dd-tf] claim cards 0+1 <= 300 s: gpu_window prod abl-prod (port 18150)
2026-11-21 23:06:51 UTC [SCHED] released cards 0+1: a1-prod4 rc=0
2026-11-21 23:06:51 UTC [SAMP] claim cards 0+1 ~5 min: perf row c: prod config 2cb54319, p0-p7 C1/C2/C4/C8 x greedy/t0.6/gen + code C1, sample profile (port 18152)
2026-11-21 23:10:54 UTC [SAMP] released cards 0+1: perf row c: prod config 2cb54319, p0-p7 C1/C2/C4/C8 x greedy/t0.6/gen + code C1, sample profile rc=0
2026-11-21 23:13:14 UTC [hotschmoe-dd-tf] released cards 0+1: gpu_window prod abl-prod rc=0
2026-11-21 23:13:14 UTC [hotschmoe-dd-tf] claim cards 0+1 <= 300 s: gpu_window prod abl-bench (port 18150)
2026-11-21 23:14:10 UTC [hotschmoe-dd-tf] released cards 0+1: gpu_window prod abl-bench rc=1
2026-11-21 23:14:10 UTC [hotschmoe-dd-tf] claim cards 0+1 <= 300 s: gpu_window prod abl-desc0 (port 18150)
2026-11-21 23:16:32 UTC [hotschmoe-dd-tf] released cards 0+1: gpu_window prod abl-desc0 rc=0
2026-11-21 23:16:32 UTC [hotschmoe-dd-tf] claim cards 0+1 <= 300 s: gpu_window prod abl-pc0 (port 18150)
2026-11-21 23:18:57 UTC [hotschmoe-dd-tf] released cards 0+1: gpu_window prod abl-pc0 rc=0
2026-11-21 23:18:57 UTC [hotschmoe-dd-tf] claim cards 0+1 <= 300 s: gpu_window prod abl-evict0 (port 18150)
2026-11-21 23:21:24 UTC [hotschmoe-dd-tf] released cards 0+1: gpu_window prod abl-evict0 rc=0
2026-11-21 23:21:24 UTC [hotschmoe-dd-tf] claim cards 0+1 <= 300 s: gpu_window prod abl-ctx16k (port 18150)
2026-11-21 23:27:56 UTC [SAMP] claim cards 0+1 ~5 min: perf row c: bench config 2cb54319, p0-p7 C1/C2/C4/C8 x greedy/t0.6/gen + code C1, sample profile (port 18152)
2026-11-21 23:28:06 UTC [SCHED] claim cards 0+1 ~5 min: a3 prod config + XPU_FN_EVICT_DRAIN=1 (dd-sched), quick bench C1-C8 + lane trace (port 18153, frozen 2cb54319)
2026-11-21 23:32:28 UTC [SAMP] released cards 0+1: perf row c: bench config 2cb54319, p0-p7 C1/C2/C4/C8 x greedy/t0.6/gen + code C1, sample profile rc=0
2026-11-21 23:35:03 UTC [hotschmoe-dd-tf] released cards 0+1: gpu_window prod abl-ctx16k rc=0
2026-11-21 23:35:03 UTC [hotschmoe-dd-tf] claim cards 0+1 <= 300 s: gpu_window prod abl-slotsauto (port 18150)
2026-11-21 23:40:03 UTC [hotschmoe-dd-tf] released cards 0+1: gpu_window prod abl-slotsauto rc=124
2026-11-21 23:40:11 UTC [SAMP] claim cards 0+1 ~5 min: row 6 depth sweep d3c0/d4c0 prod 2cb54319 (port 18152)
2026-11-21 23:41:59 UTC [O] PRIORITY.hold 'KV KRN': CPU gap for KV ReleaseFast build (starved 25 min behind quiet GPU windows) + KRN host tests; GPU/CPU leases of others wait; removed when both finish (<= 30 min)
2026-11-21 23:45:01 UTC [SAMP] released cards 0+1: row 6 depth sweep d3c0/d4c0 prod 2cb54319 rc=124
2026-11-21 23:46:48 UTC [SCHED] released cards 0+1: a3-drain4 not run (withdrawn while queued; requeued gpu-run first)
2026-11-21 23:46:48 UTC [SCHED] claim cards 0+1 ~5 min: a3 prod config + XPU_FN_EVICT_DRAIN=1 (dd-sched 5c?), quick bench C1-C8 + lane trace (port 18153)
2026-11-21 23:51:54 UTC [O] PRIORITY.hold removed (KV build + KRN host tests done or 30 min cap)
2026-11-21 23:51:55 UTC [O] PRIORITY.hold 'SAMP' for SAMP's queued xpu-native build (waiting since 23:07); removed when it finishes (<= 20 min)
2026-11-21 23:52:03 UTC [hotschmoe-dd-tf] claim cards 0+1 <= 300 s: gpu_window prod abl-par8 (port 18150)
2026-11-21 23:58:37 UTC [KRN] claim cards 0+1 <= 5 min: B3 a1 bench config XPU_FN_GDN_DEFER=0 quick bench (port 18155, krn f4ed779e)
2026-11-22 00:02:00 UTC [SCHED] released cards 0+1: a3-drain4 rc=0
2026-11-22 00:02:11 UTC [SCHED] released cards 0+1: a3-drain4 not run (cpu-run --quiet blocked by PRIORITY.hold SAMP while holding GPU; retry with short CPU wait)
2026-11-22 00:04:51 UTC [O] PRIORITY.hold removed (SAMP build done or cap)
2026-11-22 00:05:11 UTC [SCHED] claim cards 0+1 ~5 min: a3 prod config + XPU_FN_EVICT_DRAIN=1 (dd-sched), quick bench C1-C8 + lane trace (port 18153)
2026-11-22 00:07:25 UTC [hotschmoe-dd-tf] released cards 0+1: gpu_window prod abl-par8 rc=0
2026-11-22 00:09:05 UTC [hotschmoe-dd-tf] claim cards 0+1 <= 300 s: gpu_window prod abl-bench (port 18150)
2026-11-22 00:09:51 UTC [SAMP] claim cards 0+1 ~5 min: row c shared top-K pass ON, new bin d6df2818 prod (port 18152)
2026-11-22 00:10:00 UTC [KRN] released cards 0+1 (arm rc=0): B3 a1 bench config defer=0
2026-11-22 00:10:00 UTC [KRN] claim cards 0+1 <= 5 min: B3 a2 bench config XPU_FN_GDN_DEFER=1 quick bench (port 18155, krn f4ed779e)
2026-11-22 00:11:34 UTC [O] lock-order deadlock broken: killed SAMP lease pid 375073 (gpu-run outside cpu-run --quiet). Rule: cpu-run --quiet OUTSIDE, gpu-run INSIDE.
2026-11-22 00:14:15 UTC [SAMP] claim cards 0+1 ~5 min: row c shared top-K pass OFF, new bin d6df2818 prod (port 18152)
2026-11-22 00:14:30 UTC [KRN] released cards 0+1 (arm rc=0): B3 a2 bench config defer=1
2026-11-22 00:16:22 UTC [SCHED] released cards 0+1: a3-drain4 not run (withdrawn: lock order fixed to cpu-run --quiet outside)
2026-11-22 00:17:40 UTC [SAMP] released cards 0+1: row c shared top-K pass OFF, new bin d6df2818 prod rc=1
2026-11-22 00:20:06 UTC [hotschmoe-dd-tf] released cards 0+1: gpu_window prod abl-bench rc=0
2026-11-22 00:20:07 UTC [KV] claim cards 0+1 ~5 min: KV 38ccad07 on prod 2cb54319: p0 C4 vs prod ref, 8K/32K prefill, 32K cache repeat (cache+evict timing) (port 18151, bin 52accb06)
2026-11-22 00:21:36 UTC [hotschmoe-dd-tf] claim cards 0+1 <= 300 s: gpu_window prod abl-chunk64 (port 18150)
2026-11-22 00:23:00 UTC [KV] released cards 0+1 (rc=0): KV 38ccad07 on prod 2cb54319: p0 C4 vs prod ref, 8K/32K prefill, 32K cache repeat (cache+evict timing)
2026-11-22 00:23:01 UTC [SAMP] claim cards 0+1 ~5 min: row c shared top-K pass ON (re-run of killed arm), new bin d6df2818 prod (port 18152)
2026-11-22 00:26:35 UTC [SAMP] released cards 0+1: row c shared top-K pass ON (re-run of killed arm), new bin d6df2818 prod rc=0
2026-11-22 00:26:42 UTC [O] PRIORITY.hold 'SCHED' for 2 x ~4 min admission-fix arms (a3 prod+drain 4 lanes, a4 --parallel 8); auto-removed when SCHED releases its 2nd lease or 20 min
2026-11-22 00:26:54 UTC [SCHED] claim cards 0+1 ~5 min: a3 prod config + XPU_FN_EVICT_DRAIN=1 (dd-sched), quick bench C1-C8 + lane trace (port 18153)
2026-11-22 00:27:55 UTC [SCHED] released cards 0+1: a3-drain4 not run (lock busy), retry
2026-11-22 00:28:40 UTC [SCHED] claim cards 0+1 ~5 min: a3 prod config + XPU_FN_EVICT_DRAIN=1 (dd-sched), quick bench C1-C8 + lane trace (port 18153)
2026-11-22 00:29:08 UTC [KRN] claim cards 0+1 <= 5 min: B3 a3 prod config defer=0 quick bench + p0 C4 vs KV ref + 4K cache repeat (port 18155, krn f4ed779e)
2026-11-22 00:29:40 UTC [SCHED] released cards 0+1: a3-drain4 not run (lock busy), retry
2026-11-22 00:30:25 UTC [SCHED] claim cards 0+1 ~5 min: a3 prod config + XPU_FN_EVICT_DRAIN=1 (dd-sched), quick bench C1-C8 + lane trace (port 18153)
2026-11-22 00:33:51 UTC [SCHED] released cards 0+1: a3-drain4 rc=0
2026-11-22 00:34:06 UTC [SCHED] claim cards 0+1 ~5 min: a4 prod config + XPU_FN_EVICT_DRAIN=1, --parallel 8 slots=state fit (dd-sched) (port 18153)
2026-11-22 00:36:42 UTC [SCHED] released cards 0+1: a4-drain8 rc=0
2026-11-22 00:36:45 UTC [O] PRIORITY.hold removed (SCHED leases done=2 or 20 min cap)
2026-11-22 00:39:45 UTC [SCHED] claim cards 0+1 ~5 min: row 11 mixed C2: A decoding + B 16K prompt, chunk 512, prod+drain (port 18153)
2026-11-22 00:39:52 UTC [KRN] released cards 0+1 (arm rc=0): B3 a3 prod config defer=0
2026-11-22 00:40:12 UTC [KRN] claim cards 0+1 <= 5 min: B3 a4 prod config defer=1 quick bench + p0 C4 vs KV ref + 4K cache repeat (port 18155, krn f4ed779e)
2026-11-22 00:40:45 UTC [SCHED] released cards 0+1: m1-c512 not run (lock busy), retry
2026-11-22 00:41:17 UTC [KV] claim cards 0+1 ~5 min: KV cache submit-now + timing; prefill profile of 38ccad07 kernel (8K) (port 18151, bin 1d7751d0)
2026-11-22 00:43:45 UTC [KV] released cards 0+1 (rc=0): KV cache submit-now + timing; prefill profile of 38ccad07 kernel (8K)
2026-11-22 00:43:52 UTC [SCHED] claim cards 0+1 ~5 min: row 11 mixed C2: A decoding + B 16K prompt, chunk 512, prod+drain (port 18153)
2026-11-22 00:44:52 UTC [SCHED] released cards 0+1: m1-c512 not run (lock busy), retry
2026-11-22 00:45:08 UTC [KV] claim cards 0+1 ~5 min: KV for SCHED d30457b8: --parallel 8 solo refs seeds 0-7 at 6K, 600-token replies (port 18151, bin b908b338)
2026-11-22 00:48:15 UTC [KV] released cards 0+1 (rc=0): KV for SCHED d30457b8: --parallel 8 solo refs seeds 0-7 at 6K, 600-token replies
2026-11-22 00:51:04 UTC [hotschmoe-dd-tf] released cards 0+1: gpu_window prod abl-chunk64 rc=0
2026-11-22 00:51:19 UTC [SCHED] claim cards 0+1 ~5 min: row 11 mixed C2: A decoding + B 16K prompt, chunk 512, prod+drain (port 18153)
2026-11-22 00:52:17 UTC [KV] claim cards 0+1 ~5 min: KV for SCHED d30457b8: --parallel 8, pool forced 72 pages, 8 x 6K concurrent vs solo refs (page pressure) (port 18151, bin b908b338)
2026-11-22 00:52:19 UTC [SCHED] released cards 0+1: m1-c512 not run (lock busy), retry
2026-11-22 00:52:35 UTC [hotschmoe-dd-tf] claim cards 0+1 <= 300 s: gpu_window prod tr-prod-markers (port 18150)
2026-11-22 00:55:28 UTC [KV] released cards 0+1 (rc=0): KV for SCHED d30457b8: --parallel 8, pool forced 72 pages, 8 x 6K concurrent vs solo refs (page pressure)
2026-11-22 00:58:09 UTC [SCHED] claim cards 0+1 ~5 min: row 11 mixed C2: A decoding + B 16K prompt, chunk 512, prod+drain (port 18153)
2026-11-22 00:58:24 UTC [KRN] released cards 0+1 (arm rc=0): B3 a4 prod config defer=1
2026-11-22 01:00:10 UTC [SCHED] released cards 0+1: m1-c512 not run (lock busy), retry
2026-11-22 01:00:30 UTC [SCHED] claim cards 0+1 ~5 min: row 11 mixed C2: A decoding + B 16K prompt, chunk 512, prod+drain (port 18153)
2026-11-22 01:02:30 UTC [SCHED] released cards 0+1: m1-c512 not run (lock busy), retry
2026-11-22 01:02:50 UTC [SCHED] claim cards 0+1 ~5 min: row 11 mixed C2: A decoding + B 16K prompt, chunk 512, prod+drain (port 18153)
2026-11-22 01:04:32 UTC [KRN] claim cards 0+1 <= 5 min: B3 a5 prod config defer=1 EVICT_SLOTS=2 park/restore bitwise (p0 C4) + cache repeat + dd_functional image (port 18155, krn f4ed779e)
2026-11-22 01:04:51 UTC [SCHED] released cards 0+1: m1-c512 not run (lock busy), retry
2026-11-22 01:05:11 UTC [SCHED] claim cards 0+1 ~5 min: row 11 mixed C2: A decoding + B 16K prompt, chunk 512, prod+drain (port 18153)
2026-11-22 01:07:12 UTC [SCHED] released cards 0+1: m1-c512 not run (lock busy), retry
2026-11-22 01:07:32 UTC [SCHED] claim cards 0+1 ~5 min: row 11 mixed C2: A decoding + B 16K prompt, chunk 512, prod+drain (port 18153)
2026-11-22 01:09:19 UTC [SCHED] released cards 0+1: m1-c512 not run (withdrawn: binary rebuilt for the park fix)
2026-11-22 01:11:54 UTC [SCHED] claim cards 0+1 ~5 min: row 11 mixed C2: A decoding + B 16K prompt, chunk 512, prod+drain baa16cbc (port 18153)
2026-11-22 01:12:41 UTC [KRN] released cards 0+1 (arm rc=0): B3 a5 prod defer=1 EVICT_SLOTS=2 gates + dd_functional image
2026-11-22 01:12:56 UTC [KRN] claim cards 0+1 <= 5 min: B3 a6 prod defer=1 EVICT_SLOTS=2, 4 x 2K/300-token streams forced to rotate (park/restore bitwise vs solo) (port 18155, krn f4ed779e)
2026-11-22 01:15:12 UTC [SCHED] released cards 0+1: m1-c512 rc=0
2026-11-22 01:18:11 UTC [KRN] released cards 0+1 (arm rc=1): B3 a6 forced park defer=1 (FAIL: NoHeldDrafts after restore)
2026-11-22 01:18:26 UTC [KRN] claim cards 0+1 <= 5 min: B3 a7 control: same forced park arm with defer=0 (port 18155, krn f4ed779e)
2026-11-22 01:20:13 UTC [hotschmoe-dd-tf] released cards 0+1: gpu_window prod tr-prod-markers rc=0
2026-11-22 01:20:14 UTC [SAMP] claim cards 0+1 ~5 min: row c shared top-K pass OFF (full client), new bin d6df2818 prod (port 18152)
2026-11-22 01:21:43 UTC [hotschmoe-dd-tf] claim cards 0+1 <= 300 s: gpu_window prod tr-prod-full (port 18150)
2026-11-22 01:23:50 UTC [SAMP] released cards 0+1: row c shared top-K pass OFF (full client), new bin d6df2818 prod rc=0
2026-11-22 01:23:50 UTC [SCHED] claim cards 0+1 ~5 min: row 11 mixed C2: A + B ~30K prompt, chunk 256/256, prod+drain baa16cbc (port 18153)
2026-11-22 01:27:11 UTC [SCHED] released cards 0+1: m3-c256 rc=0
2026-11-22 01:27:12 UTC [SAMP] claim cards 0+1 ~5 min: row c shared top-K pass OFF (full client), new bin d6df2818 prod (port 18152)
2026-11-22 01:30:49 UTC [SAMP] released cards 0+1: row c shared top-K pass OFF (full client), new bin d6df2818 prod rc=0
2026-11-22 01:33:27 UTC [SAMP] claim cards 0+1 ~5 min: row 8 drafter dump d1 prod config d6df2818 (port 18152)
2026-11-22 01:33:50 UTC [KRN] released cards 0+1 (arm rc=1): B3 a7 control forced-park defer=0 (conc4 PASS, but 0 parks happened)
2026-11-22 01:36:02 UTC [SAMP] released cards 0+1: row 8 drafter dump d1 prod config d6df2818 rc=0
2026-11-22 01:38:27 UTC [hotschmoe-dd-tf] released cards 0+1: gpu_window prod tr-prod-full rc=0
2026-11-22 01:38:30 UTC [SCHED] claim cards 0+1 ~5 min: row 11 mixed C2: A + B ~30K prompt, chunk 512, PREFILL_SHARE 0.5, prod+drain baa16cbc (port 18153)
2026-11-22 01:39:57 UTC [hotschmoe-dd-tf] claim cards 0+1 <= 300 s: gpu_window prod tr-bench-markers (port 18150)
2026-11-22 01:42:16 UTC [SCHED] released cards 0+1: m2-c512s rc=0
2026-11-22 01:43:02 UTC [SCHED] claim cards 0+1 ~5 min: B8 drafts on/off at T0.6 + card defaults C1-C8, prod+drain par8 baa16cbc (port 18153)
2026-11-22 01:46:51 UTC [SCHED] released cards 0+1: b8-prod8 rc=0
2026-11-22 01:51:02 UTC [O] PRIORITY.hold 'KV' ~25 min: s8fix, s4fix (drain-fix NoHeldDrafts gate on SCHED baa16cbc) + tokens/round sizing; auto-removed after KV's 3rd lease or 30 min
2026-11-22 01:51:36 UTC [KV] claim cards 0+1 ~5 min: KV for SCHED baa16cbc (park held-draft fix): --parallel 8, DRAIN, pool 72 pages, 8 x 6K vs solo refs (port 18151, bin 6319b6a5)
2026-11-22 01:51:52 UTC [KRN] claim cards 0+1 <= 5 min: B3 a8 forced park defer=1 rerun on drainPages fix (bin cb1020c5) (port 18155, krn f4ed779e)
2026-11-22 01:54:51 UTC [KV] released cards 0+1 (rc=0): KV for SCHED baa16cbc (park held-draft fix): --parallel 8, DRAIN, pool 72 pages, 8 x 6K vs solo refs
2026-11-22 01:55:03 UTC [KV] claim cards 0+1 ~5 min: KV for SCHED baa16cbc: --parallel 4, DRAIN, pool 40 pages, 4 x 6K vs solo refs (port 18151, bin 6319b6a5)
2026-11-22 01:57:42 UTC [KV] released cards 0+1 (rc=0): KV for SCHED baa16cbc: --parallel 4, DRAIN, pool 40 pages, 4 x 6K vs solo refs
2026-11-22 01:57:43 UTC [KV] claim cards 0+1 ~5 min: KV sizing: tokens/round int8 KV, prod 2cb54319, ctx 16384, 4 oracle prompts x 256 tok, temp 0 and 0.6 (port 18151, bin 6ce1f656)
2026-11-22 02:00:10 UTC [KV] released cards 0+1 (rc=0): KV sizing: tokens/round int8 KV, prod 2cb54319, ctx 16384, 4 oracle prompts x 256 tok, temp 0 and 0.6
2026-11-22 02:00:10 UTC [KV] claim cards 0+1 ~5 min: KV sizing: tokens/round bf16 KV, prod 2cb54319, ctx 16384, 4 oracle prompts x 256 tok, temp 0 and 0.6 (port 18151, bin 6ce1f656)
2026-11-22 02:01:07 UTC [KV] released cards 0+1 (rc=1): KV sizing: tokens/round bf16 KV, prod 2cb54319, ctx 16384, 4 oracle prompts x 256 tok, temp 0 and 0.6
2026-11-22 02:01:08 UTC [KV] claim cards 0+1 ~5 min: KV sizing: tokens/round off KV, prod 2cb54319, ctx 16384, 4 oracle prompts x 256 tok, temp 0 and 0.6 (port 18151, bin 6ce1f656)
2026-11-22 02:03:52 UTC [KV] released cards 0+1 (rc=0): KV sizing: tokens/round off KV, prod 2cb54319, ctx 16384, 4 oracle prompts x 256 tok, temp 0 and 0.6
2026-11-22 02:04:08 UTC [KV] claim cards 0+1 ~5 min: KV sizing: tokens/round bf16 paged KV (evict off), prod 2cb54319, ctx 16384, 4 oracle prompts x 256 tok, temp 0 and 0.6 (port 18151, bin 6ce1f656)
2026-11-22 02:06:25 UTC [KV] released cards 0+1 (rc=1): KV sizing: tokens/round bf16 paged KV (evict off), prod 2cb54319, ctx 16384, 4 oracle prompts x 256 tok, temp 0 and 0.6
2026-11-22 02:06:28 UTC [O] PRIORITY.hold removed (KV leases done=3 or 30 min cap)
2026-11-22 02:06:28 UTC [O] PRIORITY.hold 'INT hotschmoe-dd-tf' ~10 min: two bench traced arms; auto-removed after INT's 2nd lease or 15 min
2026-11-22 02:09:08 UTC [hotschmoe-dd-tf] released cards 0+1: gpu_window prod tr-bench-markers rc=0
2026-11-22 02:10:38 UTC [hotschmoe-dd-tf] claim cards 0+1 <= 300 s: gpu_window prod tr-bench-full (port 18150)
2026-11-22 02:12:59 UTC [hotschmoe-dd-tf] released cards 0+1: gpu_window prod tr-bench-full rc=0
2026-11-22 02:13:00 UTC [O] PRIORITY.hold removed (INT leases=2 or cap)
2026-11-22 02:13:02 UTC [SAMP] claim cards 0+1 ~5 min: row 8 drafter dump d1 bench config d6df2818 (port 18152)
2026-11-22 02:15:21 UTC [KRN] released cards 0+1 (not run): B3 a8 withdrawn while queued; re-gate on top of the park draft-snapshot fix (SCHED baa16cbc)
2026-11-22 02:15:39 UTC [SAMP] released cards 0+1: row 8 drafter dump d1 bench config d6df2818 rc=0
2026-11-22 02:15:40 UTC [KRN] claim cards 0+1 <= 5 min: B3 a9 per-op GDN profile tool, defer 0 then 1 (2 loads, bench env) (port 18155, krn f4ed779e)
2026-11-22 02:20:20 UTC [KV] claim cards 0+1 ~5 min: KV sizing: tokens/round bf16page paged KV (evict off), prod 2cb54319, ctx 16384, 4 oracle prompts x 256 tok, temp 0 and 0.6 (port 18151, bin 6ce1f656)
2026-11-22 02:22:53 UTC [KV] released cards 0+1 (rc=0): KV sizing: tokens/round bf16page paged KV (evict off), prod 2cb54319, ctx 16384, 4 oracle prompts x 256 tok, temp 0 and 0.6
2026-11-22 02:28:28 UTC [KV] claim cards 0+1 ~5 min: KV control: prod 2cb54319, --parallel 8, NO drain, pool 72 pages, 8 x 6K vs solo refs (latent NoHeldDrafts?) (port 18151, bin 6ce1f656)
2026-11-22 02:31:38 UTC [KV] released cards 0+1 (rc=0): KV control: prod 2cb54319, --parallel 8, NO drain, pool 72 pages, 8 x 6K vs solo refs (latent NoHeldDrafts?)
2026-11-22 02:31:39 UTC [SAMP] claim cards 0+1 ~5 min: row 6 d4c0 prod 2cb54319 (port 18152)
2026-11-22 02:34:34 UTC [SAMP] released cards 0+1: row 6 d4c0 prod 2cb54319 rc=0
2026-11-22 02:37:56 UTC [SAMP] claim cards 0+1 ~5 min: row a coupled drafts on T1.0/T0.6 prod d6df2818 (port 18152)
2026-11-22 02:38:12 UTC [KRN] released cards 0+1 (arm rc=0): B3 a9 per-op GDN profile defer 0/1
2026-11-22 02:38:22 UTC [KRN] claim cards 0+1 <= 5 min: B3 a8 forced park defer=1 on e081c554 (GDN defer + drainPages + park drafts fix), bin 69c227c2 (port 18155, krn f4ed779e)
2026-11-22 02:40:56 UTC [SAMP] released cards 0+1: row a coupled drafts on T1.0/T0.6 prod d6df2818 rc=0
2026-11-22 02:43:48 UTC [SCHED] claim cards 0+1 ~5 min: B8 gate XPU_FN_PLAIN_STREAMS=4, T0.6 + card, C1-C8, par8+drain 8ae40250 (port 18153)
2026-11-22 02:43:51 UTC [KRN] released cards 0+1 (arm rc=0): B3 a8 forced park defer=1 on e081c554
2026-11-22 02:44:15 UTC [O] PRIORITY.hold 'INT hotschmoe-dd-tf' for the a52ef3f2 freeze chain (host tests, build, G1, prod subset, forced-park, unit + dd_functional); INT removes it when done; cap 50 min
2026-11-22 02:46:17 UTC [SCHED] released cards 0+1: b8-plain4 not run (lock busy), retry
2026-11-22 02:58:04 UTC [hotschmoe-dd-tf] claim cards 0+1 ~15 min: regress bounded policy serve/no-MTP/pre-post health; CPU quiet acquired before GPU
2026-11-22 03:00:32 UTC [hotschmoe-dd-tf] released cards 0+1: regress rc=0
2026-11-22 03:01:32 UTC [hotschmoe-dd-tf] claim cards 0+1 <= 720 s: gpu_window prod stack1 (port 18150, steps P0-identity P1-bench KV0-budget KV1-short-c4 KV2-long90k KV3-cache90k S1-samp VIS-I API2-tools API5-edge ABL P9-functional)
2026-11-22 03:06:24 UTC [hotschmoe-dd-tf] released cards 0+1: gpu_window prod stack1 rc=0
2026-11-22 03:07:24 UTC [hotschmoe-dd-tf] claim cards 0+1 <= 420 s: gpu_window prod stack1-park (port 18150, steps P0-identity PK-solo6k PK-conc4 PK-parks)
2026-11-22 03:10:06 UTC [hotschmoe-dd-tf] released cards 0+1: gpu_window prod stack1-park rc=1
2026-11-22 03:10:47 UTC [hotschmoe-dd-tf] claim cards 0+1 <= 420 s: gpu_window prod stack1-park2 (port 18150, steps P0-identity PK-solo6k PK-conc4 PK-parks)
2026-11-22 03:13:34 UTC [hotschmoe-dd-tf] released cards 0+1: gpu_window prod stack1-park2 rc=0
2026-11-22 03:14:23 UTC [INT] claim cards 0+1 ~10 min: production unit hotschmoe-dd-tensorfold up on 18124 with frozen a52ef3f2 for dd_functional via :18080 (then stopped)
2026-11-22 03:18:57 UTC [INT] released cards 0+1: production unit stopped after dd_functional via :18080 PASS (frozen a52ef3f2); PRIORITY.hold 'INT hotschmoe-dd-tf' removed (freeze chain done)
2026-11-22 03:19:00 UTC [SAMP] claim cards 0+1 ~5 min: row 6 d4c-0.4 prod 2cb54319 (port 18152)
2026-11-22 03:21:53 UTC [SAMP] released cards 0+1: row 6 d4c-0.4 prod 2cb54319 rc=0
2026-11-22 03:21:58 UTC [SCHED] claim cards 0+1 ~5 min: B8 gate XPU_FN_PLAIN_STREAMS=4, T0.6 + card, C1-C8, par8+drain 8ae40250 (port 18153)
2026-11-22 03:23:58 UTC [SCHED] released cards 0+1: b8-plain4 not run (lock busy), retry
2026-11-22 03:24:18 UTC [SCHED] claim cards 0+1 ~5 min: B8 gate XPU_FN_PLAIN_STREAMS=4, T0.6 + card, C1-C8, par8+drain 8ae40250 (port 18153)
2026-11-22 03:26:19 UTC [SCHED] released cards 0+1: b8-plain4 not run (lock busy), retry
2026-11-22 03:26:39 UTC [SCHED] claim cards 0+1 ~5 min: B8 gate XPU_FN_PLAIN_STREAMS=4, T0.6 + card, C1-C8, par8+drain 8ae40250 (port 18153)
2026-11-22 03:28:29 UTC [KV] claim cards 0+1 ~5 min: KV occupancy 4 slices/64-lane WG (fn_page_attn_sparse + fn_dpage_sparse): p0 C4 vs prod ref, 8K prefill profile (port 18151, bin 4ee1724a)
2026-11-22 03:28:39 UTC [SCHED] released cards 0+1: b8-plain4 not run (lock busy), retry
2026-11-22 03:29:16 UTC [O] PRIORITY.hold 'INT hotschmoe-dd-tf' for batch 2 chain (63eb6fa1: 2 CPU steps + 6 GPU arms + unit/dd_functional); INT removes it when done; cap 75 min
2026-11-22 03:31:13 UTC [KV] released cards 0+1 (rc=0): KV occupancy 4 slices/64-lane WG (fn_page_attn_sparse + fn_dpage_sparse): p0 C4 vs prod ref, 8K prefill profile
2026-11-22 03:38:56 UTC [hotschmoe-dd-tf] claim cards 0+1 ~15 min: regress bounded policy serve/no-MTP/pre-post health; CPU quiet acquired before GPU
2026-11-22 03:41:10 UTC [hotschmoe-dd-tf] released cards 0+1: regress rc=0
2026-11-22 03:41:55 UTC [hotschmoe-dd-tf] claim cards 0+1 <= 720 s: gpu_window prod stack2-prod (port 18150, steps P0-identity P1-bench KV0-budget KV1-short-c4 KV2-long90k KV3-cache90k S1-samp VIS-I API2-tools API5-edge ABL P9-functional)
2026-11-22 03:46:31 UTC [hotschmoe-dd-tf] released cards 0+1: gpu_window prod stack2-prod rc=0
2026-11-22 03:47:16 UTC [hotschmoe-dd-tf] claim cards 0+1 <= 300 s: gpu_window prod stack2-benchflags (port 18150, steps ABL)
2026-11-22 03:49:48 UTC [hotschmoe-dd-tf] released cards 0+1: gpu_window prod stack2-benchflags rc=0
2026-11-22 03:50:33 UTC [hotschmoe-dd-tf] claim cards 0+1 <= 300 s: gpu_window prod stack2-pm (port 18150, steps ABL)
2026-11-22 03:52:27 UTC [hotschmoe-dd-tf] released cards 0+1: gpu_window prod stack2-pm rc=0
2026-11-22 03:53:12 UTC [hotschmoe-dd-tf] claim cards 0+1 <= 420 s: gpu_window prod stack2-parks2 (port 18150, steps P0-identity PK-solo6k PK-conc4 PK-parks)
2026-11-22 03:55:56 UTC [hotschmoe-dd-tf] released cards 0+1: gpu_window prod stack2-parks2 rc=0
2026-11-22 03:56:41 UTC [hotschmoe-dd-tf] claim cards 0+1 <= 420 s: gpu_window prod stack2-pages40 (port 18150, steps P0-identity PK-solo6k PK-conc4 PK-parks)
2026-11-22 03:58:51 UTC [hotschmoe-dd-tf] released cards 0+1: gpu_window prod stack2-pages40 rc=0
2026-11-22 03:59:32 UTC [INT] claim cards 0+1 ~10 min: production unit up on 18124 with frozen 63eb6fa1 (+GDN_DEFER, EVICT_DRAIN) for dd_functional via :18080 (then stopped)
2026-11-22 04:03:51 UTC [INT] released cards 0+1: production unit stopped after dd_functional via :18080 PASS (frozen 63eb6fa1); PRIORITY.hold 'INT hotschmoe-dd-tf' removed (batch 2 chain done)
2026-11-22 04:03:51 UTC [KV] claim cards 0+1 ~5 min: KV: paged bf16 (bf16page) prefill profile at 8K, to split paging vs int8 cost in fn_page_attn_sparse (port 18151, bin 1d7751d0)
2026-11-22 04:05:39 UTC [KV] released cards 0+1 (rc=0): KV: paged bf16 (bf16page) prefill profile at 8K, to split paging vs int8 cost in fn_page_attn_sparse
2026-11-22 04:05:42 UTC [SCHED] claim cards 0+1 ~5 min: B8 gate XPU_FN_PLAIN_STREAMS=4, T0.6 + card, C1-C8, par8+drain 8ae40250 (port 18153)
2026-11-22 04:06:12 UTC [O] PRIORITY.hold 'INT hotschmoe-dd-tf' for batch 3 chain (c6346c82: KV paged-attn block reads + cache copies + slots; 8-lane prod + 8 x 60K forced park); INT removes it; cap 90 min
2026-11-22 04:09:22 UTC [SCHED] released cards 0+1: b8-plain4 rc=0
2026-11-22 04:17:07 UTC [hotschmoe-dd-tf] claim cards 0+1 ~15 min: regress bounded policy serve/no-MTP/pre-post health; CPU quiet acquired before GPU
2026-11-22 04:19:33 UTC [hotschmoe-dd-tf] released cards 0+1: regress rc=0
2026-11-22 04:20:19 UTC [hotschmoe-dd-tf] claim cards 0+1 <= 720 s: gpu_window prod stack3-prod4 (port 18150, steps P0-identity P1-bench KV0-budget KV1-short-c4 KV2-long90k KV3-cache90k S1-samp VIS-I API2-tools API5-edge ABL P9-functional)
2026-11-22 04:24:54 UTC [hotschmoe-dd-tf] released cards 0+1: gpu_window prod stack3-prod4 rc=0
2026-11-22 04:25:39 UTC [hotschmoe-dd-tf] claim cards 0+1 <= 720 s: gpu_window prod stack3-prod8 (port 18150, steps P0-identity P1-bench KV0-budget KV1-short-c4 KV2-long90k KV3-cache90k S1-samp VIS-I API2-tools API5-edge ABL P9-functional)
2026-11-22 04:30:09 UTC [hotschmoe-dd-tf] released cards 0+1: gpu_window prod stack3-prod8 rc=0
2026-11-22 04:30:54 UTC [hotschmoe-dd-tf] claim cards 0+1 <= 1900 s: gpu_window prod stack3-long8 (port 18150, steps P0-identity LK-solo60k LK-conc8 LK-parks)
2026-11-22 04:40:36 UTC [hotschmoe-dd-tf] released cards 0+1: gpu_window prod stack3-long8 rc=1
2026-11-22 04:41:43 UTC [hotschmoe-dd-tf] claim cards 0+1 <= 1900 s: gpu_window prod stack3-long8s4 (port 18150, steps P0-identity LK-solo60k LK-conc8 LK-parks)
2026-11-22 04:51:32 UTC [hotschmoe-dd-tf] released cards 0+1: gpu_window prod stack3-long8s4 rc=0
2026-11-22 04:56:40 UTC [INT] claim cards 0+1 ~6 min: production unit up on 18124 with frozen c6346c82 (4 lanes) for dd_functional via :18080 (then stopped)
2026-11-22 05:00:26 UTC [INT] released cards 0+1: production unit stopped after dd_functional via :18080 PASS (frozen c6346c82); PRIORITY.hold 'INT hotschmoe-dd-tf' removed (batch 3 chain done)
2026-11-22 05:00:33 UTC [SCHED] claim cards 0+1 ~5 min: chunk gate L512: teacher-forced prompt_logprobs, prod env chunk 512 (port 18153)
2026-11-22 05:02:34 UTC [SCHED] released cards 0+1: cg-L512 not run (lock busy), retry
2026-11-22 05:02:54 UTC [SCHED] claim cards 0+1 ~5 min: chunk gate L512: teacher-forced prompt_logprobs, prod env chunk 512 (port 18153)
2026-11-22 05:04:54 UTC [SCHED] released cards 0+1: cg-L512 not run (lock busy), retry
2026-11-22 05:05:14 UTC [SCHED] claim cards 0+1 ~5 min: chunk gate L512: teacher-forced prompt_logprobs, prod env chunk 512 (port 18153)
2026-11-22 05:07:15 UTC [SCHED] released cards 0+1: cg-L512 not run (lock busy), retry
2026-11-22 05:07:35 UTC [SCHED] claim cards 0+1 ~5 min: chunk gate L512: teacher-forced prompt_logprobs, prod env chunk 512 (port 18153)
2026-11-22 05:09:36 UTC [SCHED] released cards 0+1: cg-L512 not run (lock busy), retry
2026-11-22 05:09:56 UTC [SCHED] claim cards 0+1 ~5 min: chunk gate L512: teacher-forced prompt_logprobs, prod env chunk 512 (port 18153)
2026-11-22 05:11:57 UTC [SCHED] released cards 0+1: cg-L512 not run (lock busy), retry
2026-11-22 05:12:17 UTC [SCHED] claim cards 0+1 ~5 min: chunk gate L512: teacher-forced prompt_logprobs, prod env chunk 512 (port 18153)
2026-11-22 05:13:42 UTC [KRN] claim cards 0+1 ~5 min: host-cost split profile legacy vs paged KV defer=1 on krn e081c554+tool (no port)
2026-11-22 05:14:17 UTC [SCHED] released cards 0+1: cg-L512 not run (lock busy), retry
2026-11-22 05:16:26 UTC [O] PRIORITY.hold 'KV' ~30 min: p8ref, p8p (8-lane park gate on c6346c82) + c256ref/c256p8/c256p4 (chunk-256 park gates); auto-removed after 5 KV leases or 35 min
2026-11-22 05:17:08 UTC [KRN] released cards 0+1: host-cost split profile legacy vs paged KV defer=1 on krn e081c554+tool rc=0
2026-11-22 05:17:09 UTC [KV] claim cards 0+1 ~5 min: KV for INT: c6346c82 prod env --parallel 8, solo refs 12K seeds 0-7 x 300 (port 18151, bin 253c67f8)
2026-11-22 05:20:22 UTC [KV] released cards 0+1 (rc=0): KV for INT: c6346c82 prod env --parallel 8, solo refs 12K seeds 0-7 x 300
2026-11-22 05:20:27 UTC [KV] claim cards 0+1 ~5 min: KV for INT: c6346c82 prod env --parallel 8, KV_PAGES=100, 8 x 12K x 300 vs solo (forced parks) (port 18151, bin 253c67f8)
2026-11-22 05:23:44 UTC [KV] released cards 0+1 (rc=0): KV for INT: c6346c82 prod env --parallel 8, KV_PAGES=100, 8 x 12K x 300 vs solo (forced parks)
2026-11-22 05:23:50 UTC [KV] claim cards 0+1 ~5 min: KV for SCHED: 63eb6fa1 prod env chunk 256 --parallel 8, solo refs 6K seeds 0-7 x 600 (port 18151, bin 9fb022a8)
2026-11-22 05:26:30 UTC [KV] released cards 0+1 (rc=0): KV for SCHED: 63eb6fa1 prod env chunk 256 --parallel 8, solo refs 6K seeds 0-7 x 600
2026-11-22 05:26:35 UTC [KV] claim cards 0+1 ~5 min: KV for SCHED: 63eb6fa1 chunk 256 --parallel 8, 72 pages, 8 x 6K x 600 vs solo (port 18151, bin 9fb022a8)
2026-11-22 05:29:14 UTC [KV] released cards 0+1 (rc=0): KV for SCHED: 63eb6fa1 chunk 256 --parallel 8, 72 pages, 8 x 6K x 600 vs solo
2026-11-22 05:29:20 UTC [KV] claim cards 0+1 ~5 min: KV for SCHED: 63eb6fa1 chunk 256 --parallel 4, 40 pages, 4 x 6K x 600 vs solo (port 18151, bin 9fb022a8)
2026-11-22 05:31:27 UTC [KV] released cards 0+1 (rc=0): KV for SCHED: 63eb6fa1 chunk 256 --parallel 4, 40 pages, 4 x 6K x 600 vs solo
2026-11-22 05:31:31 UTC [O] PRIORITY.hold removed (KV leases=5 or cap)
2026-11-22 05:31:32 UTC [SAMP] claim cards 0+1 ~5 min: row a coupled C1/C4/C8 bitwise on prod c6346c82 (port 18152)
2026-11-22 05:34:21 UTC [SAMP] released cards 0+1: row a coupled C1/C4/C8 bitwise on prod c6346c82 rc=0
2026-11-22 05:34:32 UTC [INT] claim cards 0+1 ~8 min: production unit up on 18124, c6346c82 at 8 lanes (TF_PARALLEL=8, EVICT_SLOTS unset): /v1/models + dd_functional via :18080 + ABL C1/C4/C8 (then stopped)
2026-11-22 05:44:18 UTC [INT] released cards 0+1: production unit stopped (c6346c82, 8 lanes + XPU_FN_COUPLED=1; dd_functional via :18080 PASS twice)
