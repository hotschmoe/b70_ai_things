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
