# Release notes: hotschmoe-dd on TensorFold Flash-Next, 2026-11-22 

Status: RELEASED (validated 13:00-14:17 UTC; go-live 16:00 UTC). Git tag: tf-dd-20261122.

## Build

| Item | Value |
|---|---|
| Model | Qwen3.8 Flash-Next ISTA GSQ-RCO IQ3_XXS GGUF, TP2 on two Arc Pro B70 |
| Served name | hotschmoe-dd (aliases: qwen3.8-flash-next-ISTA-GSQ-RCO-IQ3_XXS-tensorfold, ...-strata-0216) |
| Engine | TensorFold Flash-Next (Zig / Level Zero), local release branch b70/dd-rel |
| Commit | ebdc73dd532604c280a0acd10a400d7737ffbbbe (= 983cc068 + KRN 03719727) |
| Binary | /mnt/vm_8tb/b70/build/tensorfold-dd/ebdc73dd/zig-out/xpu/bin/tensorfold-native |
| Binary sha256 | aeb0bd2b8fde4eb6c57a342829e15d07271bba572597a4b7456ec9bce29d215b |
| Env | tensorfold/flashnext/dd/tf-dd-prod.env (+ tf-dd-common.env) |
| Unit | hotschmoe-dd-tensorfold.service (systemd --user), backend 127.0.0.1:18124, frontdoor :18080 |

## Configuration and flags

- 8 parallel streams, 200000 context per stream, max output 32768, reasoning on (effort medium, thinking budget 16384).
- KV: paged int8 (XPU_FN_KV=int8, XPU_FN_DESC=1, XPU_FN_KV_PAGES=auto), RAM eviction/parking with write-behind mirror
  (XPU_FN_EVICT=1, XPU_FN_MIRROR=1, XPU_FN_EVICT_DRAIN=1), prompt cache 6 GiB per card in pinned host RAM.
- Prefill chunk 512 (XPU_CHUNK = XPU_CHUNK_SOLO = 512).
- Decode: MTP speculative decoding, deferred GDN commit (XPU_FN_GDN_DEFER=1), coupled draft draws (XPU_FN_COUPLED=1),
  recorded windows over paged KV (XPU_FN_RECORD=1 XPU_FN_GATE=1; any refused window falls back to the normal path).
- Sampling by request (temperature, top_p, top_k, min_p, penalties, seed); defaults from generation_config.json;
  thinking-off requests use the model card's instruct sampling.
- Vision: CPU strata-vision sidecar + mmproj (JPEG with EXIF rotation, PNG, WebP).
- API: OpenAI-compatible chat completions with streaming, qwen3-coder tool calls (tool_choice auto/none/required/named),
  structured output by prompt fallback, usage with cached_tokens, /health, /v1/models, /metrics.

## Measured (production env, greedy bench unless noted)

| Row | C1 | C2 | C4 | C8 | Prefill 2K | 90K TTFT cold / cached |
|---|---|---|---|---|---|---|
| Release ebdc73dd | 106.3 | 166.7 | 218.4 | 280.3 | 1548 tok/s | 75.0 s / 0.105 s |
| Cutover build 2cb54319 (2026-11-21) | 100.0 | 144.1 | 123.0 | 123.2 | 1390 tok/s | 88.7 s |
| Strata 0.2.16 (previous production) | 133.5 | - | - | - | - | - |

Sampled T0.6 (p0-p3, 256 tokens): C1 128.3, C4 217.7, C8 263.2 tok/s aggregate.
Long-context decode C1 (greedy): 1K 96.4, 16K 91.3, 64K 90.0, 128K 84.9 tok/s.
Owner targets: C1 100 met; C2 150 met; prefill 1500 met at 2K (about 1200 tok/s at 90K); C4 300 and C8 500 not met.

## Validation (2026-11-22, 13:00-14:17 UTC, exact production env; evidence workers/INT/rel, workers/REL/rel-rc-20261122T1311Z)

- Full GPU window (relA, 25 steps): PASS. Bench regression C1 109.4 / C2 141.2 / C4 173.6 / C8 209.6 / PF2k 1557 with
  greedy output bitwise equal to the reference; production C1 106.9 / C2 167.2 / C4 219.1 / C8 278.3 / PF2k 1545,
  90K TTFT 75.4 s cold / 0.111 s cached; vision x4, API1-5, functional, 60K multi-turn cache check, 0 recorded-window
  fallbacks; seeded sampled output after the full list equal to the stable reference, concurrent == single stream.
  Earlier gates on the same build: forced parking at 8 slots (72 pages: 72 parks / 69 restores, 0 errors, == solo),
  fallback drill (65 forced fallbacks, 0 HTTP 5xx), unit bounce.
- 60 min realistic mixed traffic through :18080 (release_traffic.py): PASS on every hard check. 236/236 requests ok in
  55 conversations (agent 139, chat 57, long-document 23, image 17), 1-8 concurrent, prompts up to 86K; finishes
  stop 88 / tool_calls 134 / length 14; 229 tool calls, 0 malformed; images 17/17 correct; 168/181 follow-ups served
  from the prompt cache; 0 fatal log lines; swap flat; 0 fallbacks.
  - Device memory rose 0.94 GiB per card in 8 steps of about 118 MiB during the first 34 minutes, then stayed flat for
    26 minutes: one per-stream state slab per resident stream (slab_est 124.7 MB, 8 streams), inside the 4.38 GB reserve
    the KV sizing keeps. Free VRAM at the end: about 1.1 GiB per card.
  - Host memory rose 2.7 GiB as the pinned host tier filled (prompt cache 11.9 of 12 GiB), then stayed flat.
  - Latency: TTFT p50 6.9 s overall; cached follow-ups 1.0 s (chat) and 7.6 s (agent) p50; agent follow-up p99 139 s
    and long-document follow-up p99 198 s when 8 agents with 45-90K contexts were active at once (prefill queueing).
- Rollback rehearsal: PASS. Env switched to 983cc068 with recording off: ready in 70 s, dd_functional with image through
  :18080 PASS; switched back to the release env: ready in 55 s, dd_functional PASS, 0 fallbacks; clean stops with the
  post-stop health check rc=0.

## Known limits

- Only about 3 streams fit at a full 200K context at once; more long streams park to host RAM and resume without
  recompute. A park costs about 0.3 s on average (up to about 2.3 s).
- Long prompts that arrive together are prefilled in turn: 7 x 60K at once means the last first token after about 6.5
  minutes (prefill throughput, about 1050-1200 tok/s).
- While a new long prompt prefills, other decoding streams stall about 0.5 s per round (up to about 1.3 s).
- Seeded sampled output is reproducible run to run, and concurrent == single stream, in every controlled test. But seed
  reproducibility across server sessions is not guaranteed: after prompt-cache resumes, seeded samples differed between
  sessions on 2 of 4 short prompts at T0.6 / T1.0 (greedy unaffected in all gates). Cause open (NEXT-QUEUE NR-1); not
  caused by recorded windows; a host-side audit found no uninitialised read.
- With recorded windows on, the on-card MTP draft chain is not used, so single-stream sampled decode is about 3% slower
  than without recording; streams 2-8 are 9-19% faster.
- With many long agent conversations at once, follow-up TTFT p99 reaches about 2-3 minutes (prefill is served in turn).
- One tool call to a tool that was not offered was kept in 60 minutes (metric kept_tool_calls_unoffered_total = 1).
- No logprobs; TIFF/AVIF images are not accepted (JPEG/PNG/WebP are).
- After 16:00 the unit holds both GPUs; research GPU work needs the owner's OK and a planned window.

## Rollback

1. Turn off recorded windows (no rebuild): delete the "export XPU_FN_RECORD=1 XPU_FN_GATE=1" line in tf-dd-prod.env,
   then systemctl --user restart hotschmoe-dd-tensorfold and run dd_functional.py with an image through :18080.
2. Previous builds: ROLLBACK A (983cc068), B (ed106eaf, prompt cache 2), 0/1/2 in the header of tf-dd-prod.env: set
   TF_COMMIT, TF_BIN and TF_BIN_SHA256 from the header, restart, dd_functional.
3. Back to Strata 0.2.16: tensorfold/flashnext/dd/gpu_window.sh rollback (stops the TensorFold unit, its post-stop
   health check runs, re-enables and starts hotschmoe-dd = Strata, moves the passive record back, verifies :18080).
   Manual equivalent: systemctl --user disable --now hotschmoe-dd-tensorfold; systemctl --user enable --now hotschmoe-dd
   (strata/flash-next/dd/README.md); the frontdoor stays up; run dd_functional.py with an image through :18080.
