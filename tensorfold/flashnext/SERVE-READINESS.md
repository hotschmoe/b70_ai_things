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
| INT | Integration branch b70/dd-serve, merges, the frozen build, the unit/launcher (reuse tensorfold/flashnext/dd/), GPU-TESTS.md ordering and gates, cutover + rollback runbook | worker INT (Opus) | started |
