# TensorFold hotschmoe-dd serve readiness (owner directive 2026-11-21 18:15 UTC)

Strata 0.2.16 serves production on both cards. Users are the soak; we only observe (passive metrics record:
hotschmoe-dd-soak.service, alerts only). All TensorFold work below is CPU-only. GPU checks are batched in
GPU-TESTS.md, to run in ONE window when the owner brings Strata down. Cutover = run GPU-TESTS.md, run
dd_functional.py through 18080 with an image, then switch the unit (stop hotschmoe-dd, start hotschmoe-dd-tensorfold).

Base: b70/dd-serve from 372bf1b2 (int-rf2: C1 109.5 / C2 139.5 / C4 168 / C8 209 / prefill 1560; single stream is
82% of Strata's 133.5; C4, C8 and prefill are ahead of Strata).

| Item | Need (parity with hotschmoe-dd-0216.json) | Owner | Status |
|---|---|---|---|
| KV | Long context in the FAST path: paged int8 KV working WITH descriptor windows (today they are exclusive, and desc mode preallocates 16 rows x context, so ~20K max). 200000 context, >= 4 parallel; users send 58K-90K token prompts. Plus parking/eviction to RAM (merge b70/fn-rt-evict 2c5d885 + mirror + RAM prompt cache) | worker KV (Opus) | started |
| SAMP | Real sampling: temperature, top_p, top_k, min_p, presence/frequency/repetition penalties, seed, with MTP speculative decoding kept lossless (rejection-sampling acceptance); greedy stays bitwise | worker SAMP (Opus) | started |
| VIS | Vision in the serving path: image inputs through a CPU vision encoder (Strata's strata-vision sidecar + mmproj, or llama.cpp mtmd) to embeddings, injected into the prompt with Qwen3.8 vision tokens and positions | worker VIS (Opus) | started |
| API | OpenAI/Anthropic parity: model id hotschmoe-dd + aliases, reasoning on by default with effort medium + 16384 thinking budget, reasoning_content, qwen3-coder tool calls incl. streaming tool-call deltas, tool_choice auto/none/required + structured output (libtfgrammar missing), /v1/messages if Strata clients use it, usage incl. cached tokens, /health, context-length errors like Strata | worker API (Sonnet) | started |
| INT | Integration branch b70/dd-serve, merges, the frozen build, the unit/launcher (reuse tensorfold/flashnext/dd/), GPU-TESTS.md ordering and gates, cutover + rollback runbook | worker INT (Opus) | FROZEN b70/dd-serve 2cb54319 (sha 6ce1f656) = SAMP a00ab802 + API 32b27d57 + VIS 097ff79c + KV f70795f7; host suite 31/31; full window 22:18: 26 gates, 0 hard failures (gpu-window/final-2cb54319); cutover waits for orchestrator go; quick checks PASS incl. dd_functional with image (16K); waiting for KV handoff, then dd-vis 097ff79c, freeze, window, cutover |

## Parity checklist vs strata/flash-next/dd/hotschmoe-dd-0216.json + dd_functional.py (live; INT and orchestrator update)

Strata DOWN since 18:55 UTC 2026-11-21 (owner: no users today); both cards to TensorFold; xpu-health HEALTHY 18:56.
Rollback = systemctl --user start hotschmoe-dd (Strata 0.2.16; dd/README.md).

| Feature | Strata value | Owner | Status | GPU gate |
|---|---|---|---|---|
| Model id hotschmoe-dd + alias qwen3.8-flash-next-ISTA-GSQ-RCO-IQ3_XXS-strata-0216 | yes | API | PASS on dd-serve fb0e6006 (INT quick 20:05: hotschmoe-dd first + tensorfold + strata-0216 aliases) | P0 identity |
| 4+ slots (queue, not reject) | parallel 4 | API/KV | PASS at 16K on fb0e6006 (8 requests on 4 lanes all 200; dd_functional 4 concurrent exact); 200K x 4 waits for KV | P0 + 4 concurrent == solo |
| Long context 200K, paged KV in fast (desc) path; 58-90K prompts | 200000, int8 KV | KV | in progress (b70/dd-kv bef2f1a8 desc over paged arena) | 90K prompt + 4 x long admission |
| Parking / KV eviction to RAM, restore w/o recompute | kv-resident 65536 + RAM | KV | not started (fn-rt-evict 2c5d885 to merge) | park/restore bitwise |
| Real sampling (temp, top_p, top_k, min_p, penalties, seed) | yes | SAMP | MERGED into b70/dd-serve 19:22 (fn-samp 28030ff0, ff); SAMP gate 12/12 PASS on :18152; prod env: TF_TEMPERATURE empty (generation_config) + --instruct-sampling card; clamp proxy retired | S1-samp |
| MTP under sampling (lossless) + greedy bitwise | spec 4 MTP | SAMP | MERGED: greedy p0 bitwise == expected-current, drafts on == off (keyed draws, exact-match accept), MTP accept 0.419 @ T 0.6; perf 2026-11-22: shared top-K d6df2818 (bitwise) + XPU_FN_COUPLED=1 (lossless, +5.5% T1.0) handed to INT; drafter vs FP64 0.771/0.781 | S1-samp + G1 regress |
| Reasoning on, effort medium, budget 16384, reasoning_content | yes | API | MERGED 19:41 (dd-api 32b27d57 -> dd-serve dc8370e5; openai.zig conflict = union with SAMP fields); API gate 5/5 on :18154; INT re-gate running | API1 + P9 |
| Tool calls (qwen3-coder), streaming deltas, tool_choice | yes | API | MERGED (TF_STRUCTURED_FALLBACK=1, TF_FIT_MAX_TOKENS=1 in prod env); /slots + /props added | API2/API3/API5 + P9 |
| Streaming SSE + usage/cached tokens | yes | API | MERGED; cached_tokens stays 0 until a prompt cache is configured (KV) | API1/API5 |
| Vision (image_url via CPU strata-vision + mmproj) | images on | VIS | MERGED 19:42 (dd-vis a2e63f48 -> dd-serve fb0e6006; 2 conflicts with SAMP resolved); gated by VIS in classic + int8 non-desc; must re-gate in desc+paged after KV | VIS-T0/I/T1 + P9 |
| Speed at production config | C1 133.5 (Strata) | INT | int-rf2 C1 109.5. fb0e6006 in legacy-window int8 paged mode (pre-KV): C1 87.7 / C2 115.8 / C4 136.2 / C8 140.7 (4 lanes) / PF2k 1119; dd_functional long stream 123.8 tok/s. Production number needs KV desc+paged | P1 bench >= 0.90 x int-rf2 |
| Perf B3: deferred GDN commit (XPU_FN_GDN_DEFER=1) | - | KRN | PASS on b70/krn-gdn e081c554, handed to INT. Prod C1 100->103, C2 144->153, C4 150->179, C8 127->165, PF same; bench C8 210->229. p0 + C4 == KV ref, cache repeat equal, dd_functional image PASS, forced park 14/14 bitwise | p0 C4 + forced park (EVICT_SLOTS=2) + dd_functional |
| Candidate fold-in: fn-a3 e87b5074 (one-card gated) | - | INT | gated 19:06: oracle/solo/health PASS; C1 110.8 C2 135.9 C4 172.9 C8 212.3 PF 1551 (C2, PF below int-rf2, noise) -> not merged, base stays 372bf1b2 | done |
| Unit hotschmoe-dd-tensorfold + frontdoor :18080 + dd_functional with image | - | INT | dd_functional.py with image PASS direct to :18150 on fb0e6006 (20:36, 16K); unit written, not installed; cutover after KV + freeze + window | C1-C3 cutover |

## Order after parity (owner 2026-11-21 19:45 UTC)

Engine strategy on hold (owner designs our own engine separately): no fork restructuring, no GitHub fork, no repo
reorganisation. (1) Parity until TF serves hotschmoe-dd and passes dd_functional.py through :18080 with an image.
(2) Then performance, single stream and concurrency, measured with ALL features on: the bench gains long-context rows
(32K/90K prompt prefill + decode at depth) and sampling rows (temperature 0.6, card defaults), not only greedy 16K.
Lessons for a future engine go in ENGINE-LESSONS.md.

## Backlog (owner decisions)

- TIFF/AVIF image input: DECLINED for now (owner 2026-11-22 00:30; no client sends them). Backlog item: a
  TENSORFOLD_IMAGE_CONVERT=<cmd> hook (bytes in, PNG out) applied only to TIFF/AVIF, as Strata converts via Pillow.
