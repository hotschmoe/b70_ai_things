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
