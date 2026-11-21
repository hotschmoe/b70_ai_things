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
| INT | Integration branch b70/dd-serve, merges, the frozen build, the unit/launcher (reuse tensorfold/flashnext/dd/), GPU-TESTS.md ordering and gates, cutover + rollback runbook | worker INT (Opus) | b70/dd-serve created (372bf1b2 + 1a5a57d0 bench port/model knobs); host suite 20/20 green at base + xpu-native builds (356 s); GPU-TESTS.md ordered (runner dd/gpu_window.sh, CPU self-test PASS); launcher/unit/prod env + runbook dd/README.md ready; frozen build pending item merges |

## Parity checklist vs strata/flash-next/dd/hotschmoe-dd-0216.json + dd_functional.py (live; INT and orchestrator update)

Strata DOWN since 18:55 UTC 2026-11-21 (owner: no users today); both cards to TensorFold; xpu-health HEALTHY 18:56.
Rollback = systemctl --user start hotschmoe-dd (Strata 0.2.16; dd/README.md).

| Feature | Strata value | Owner | Status | GPU gate |
|---|---|---|---|---|
| Model id hotschmoe-dd + alias qwen3.8-flash-next-ISTA-GSQ-RCO-IQ3_XXS-strata-0216 | yes | API | in progress | P0 identity |
| 4+ slots (queue, not reject) | parallel 4 | API/KV | in progress | P0 + 4 concurrent == solo |
| Long context 200K, paged KV in fast (desc) path; 58-90K prompts | 200000, int8 KV | KV | in progress (b70/dd-kv bef2f1a8 desc over paged arena) | 90K prompt + 4 x long admission |
| Parking / KV eviction to RAM, restore w/o recompute | kv-resident 65536 + RAM | KV | not started (fn-rt-evict 2c5d885 to merge) | park/restore bitwise |
| Real sampling (temp, top_p, top_k, min_p, penalties, seed) | yes | SAMP | in progress (b70/fn-samp) | distribution sanity + seed repeat |
| MTP under sampling (lossless) + greedy bitwise | spec 4 MTP | SAMP | in progress | greedy hash f8caa42cc5b4; accept @0.6 |
| Reasoning on, effort medium, budget 16384, reasoning_content | yes | API | in progress (b70/dd-api) | dd_functional thinking |
| Tool calls (qwen3-coder), streaming deltas, tool_choice | yes | API | in progress | dd_functional tool_call + roundtrip |
| Streaming SSE + usage/cached tokens | yes | API | in progress | stream check |
| Vision (image_url via CPU strata-vision + mmproj) | images on | VIS | in progress (b70/dd-vis) | dd_functional vision "blue" |
| Speed at production config | C1 133.5 (Strata) | INT | int-rf2 C1 109.5 = 82% of Strata; C4/C8/PF ahead | P1 bench >= 0.90 x int-rf2 |
| Candidate fold-in: fn-a3 e87b5074 (one-card gated) | - | INT | two-card gate queued | quick bench gate |
| Unit hotschmoe-dd-tensorfold + frontdoor :18080 + dd_functional with image | - | INT | unit written, not installed | C1-C3 cutover |
