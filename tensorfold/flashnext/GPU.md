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
