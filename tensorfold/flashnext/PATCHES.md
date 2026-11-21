# TensorFold patches for hotschmoe-dd serving (keep small, documented, upstreamable)

Engine strategy is undecided (fork Strata / fork TensorFold / own Zig engine near an upstream). Every patch: one concern,
flag or clean API, a test. One line each.

| Branch | Commit | What | Flag / default | Upstreamable | Test |
|---|---|---|---|---|---|
| b70/dd-serve | 1a5a57d0 | serve bench: FN_BENCH_PORT / FN_BENCH_MODEL pick the server (tools/flashnext_serve_bench.py) | env; defaults 18150 / tf-flashnext-iq3xxs-tp2-mtp unchanged | yes (local tool) | python ast parse; used by INT gates |
| b70/dd-kv | bef2f1a8 | Descriptor windows over the paged int8 KV arena (fn_window_pages.cl), XPU_FN_KV_PAGES=N\|auto arena sizing, MTP device chain with paged desc | XPU_FN_KV=int8 + XPU_FN_DESC=1 (+ XPU_FN_KV_PAGES=auto); default off | yes (new kernels + flag; no change to default paths) | test-fn-dpage-host (paged vs dense CPU reference, page budget); GPU: workers/KV kv-quick |
| b70/dd-kv | 02f235db | Merge b70/fn-rt-evict 2c5d885: park to RAM, restore without recompute, mirror, RAM prompt cache | XPU_FN_EVICT=1, XPU_FN_EVICT_SLOTS, XPU_FN_MIRROR=1; default off | yes (existing worker branch) | test-fn-evict-host, test-fn-park-host, test-fn-evict-own-host, test-fn-resident-slots-host, test-fn-evict-admit-host |
| b70/fn-samp | da4716cc | lanes: Strata 0.2.16 sampling chain (penalties, top_k, top_p, min_p, temperature) + keyed Gumbel draw, shard top-64 merge, rejection-sampling reference; Penalties.repeat/last_n | new module; defaults unchanged (repeat 1, last_n 0) | yes | test-fn-sampling-host (numpy fixture, chi-square, masks, merge, spec p/q) |
| b70/fn-samp | 874eac43 | server: repetition_penalty/repeat_penalty + penalty_last_n/repeat_last_n mapping (window 64) behind Info.repetition_penalty | capability; off engines refuse != 1 | yes | test-penalty-request-host (28 pass) |
| b70/fn-samp | 0fc39fb4 | flashnext xpu: rowRuns (fn_topk64_part per rank shard -> host) and rowShards (whole bf16 shard rows) | lazy buffers; unused when greedy | yes | test-flashnext-xpu-host compile; GPU s0b |
| b70/fn-samp | 28030ff0 | xpu lane: Flash Next sampled/penalized draws on the host chain; greedy keeps device argmax; greedy_only off, penalties + repetition_penalty on | by request (temperature > 0 or penalties) | yes | GPU quick gate samp_gate.py 12/12 PASS, p0 bitwise |
| b70/dd-vis | a2e63f48 | --vision for Flash Next XPU GGUF TP2: strata-vision CPU sidecar rows spliced over image pads, Strata M-RoPE positions (shifted frame) via additive fn_mrope.cl kernels, vision=true cap | --vision + TF_VISION_EXE/MMPROJ/MODEL; off by default | yes (one family, new files + additive kernels; text kernels untouched) | test-fn-mrope-host, test-fn-vision-host; GPU vis_gate classic + int8 KV PASS |
| b70/dd-api | 9b4c7691 | thinking budget also read from reasoning_budget_tokens / thinking_budget_tokens (Strata, llama.cpp names; 0 = none) | none (additive request fields) | yes | test-dd-api (requestBudget), parity echo_* |
| b70/dd-api | 9b4c7691 | TF_FIT_MAX_TOKENS=1 clamps max_tokens to the room the prompt leaves instead of a context-length 400 | env, default off | yes | parity ctx_overflow_max_tokens; GPU api-edge clamp |
| b70/dd-api | 32b27d57 | the clamp keeps one token back (lane engine refuses prompt + max_tokens + 1 > window) | with TF_FIT_MAX_TOKENS | yes | GPU api-edge |
| b70/dd-api | 9b4c7691 | TF_STRUCTURED_FALLBACK=1: tool_choice required / named written as a forced call opening in the prompt (thinking off that turn); response_format json_object / json_schema as a system instruction (no validation); no 400 on engines with no token grammar | env, default off | yes (engine-agnostic) | parity tool_choice_* and format_json_*; GPU api-forced |
| b70/dd-api | 9b4c7691 | GET /slots and GET /props (llama.cpp subset) | none (new routes) | yes | parity slots, props |
| b70/dd-api | 9b4c7691 | fake_serve: TF_FAKE_PROFILE=dd (production Info, lane cap + queue), echo_request script; build steps dd-fake-serve, test-dd-api | test tooling | yes | run_parity.sh |
| b70/dd-serve | dc8370e5 | merge b70/dd-api 32b27d57 (openai.zig: union of SAMP penalty fields and API budget aliases/response_format fallback) | - | n/a (integration) | INT host suite2 + quick check |
| b70/dd-serve | fb0e6006 | merge b70/dd-vis a2e63f48 (xpu_forward.zig fields side by side; xpu_flashnext.zig SAMP caps then openVision, greedy_only stays removed) | - | n/a (integration) | INT host suite2 + quick check |
