# GPU test list for the cutover window (run in one window, in order; each <= 3 min unless noted)

Owner of ordering and gates: worker INT. Each item: command, expected time, PASS gate, rollback.
Sections per item: KV, SAMP, VIS, API, INT. Workers write their section below; INT transcribes each step into
tensorfold/flashnext/dd/gpu_window_steps.tsv (the file the runner executes) and orders it here.

## How the window runs (owner)

Before the window (CPU only, any time, Strata keeps serving):

    tensorfold/flashnext/dd/gpu_window.sh preflight     # frozen sha, clean worktree, steps file, fixture, port 18150 free

The window (both cards; Strata down for about 30 min; GPU total about 25 min, budget 45):

    CONFIRM=yes tensorfold/flashnext/dd/gpu_window.sh window
    # = write PRIORITY.hold (hotschmoe-dd-tf), stop hotschmoe-dd, G1 regress, G2 prod (all steps below), remove hold.
    # Any HARD gate FAIL -> Strata is restarted automatically and the window ends with no cutover.
    CONFIRM=yes WIN_DIR=<window dir printed above> tensorfold/flashnext/dd/gpu_window.sh cutover
    # = install + start hotschmoe-dd-tensorfold, dd_functional.py through :18080 with the image fixture,
    #   then enable TF / disable Strata. FAIL -> automatic rollback.
    tensorfold/flashnext/dd/gpu_window.sh rollback      # any time later: TF unit down (+xpu-health), Strata up, functional

Results: /mnt/vm_8tb/b70/results/hotschmoe-dd/gpu-window/<UTC tag>/gates.tsv (one line per gate), window.log, per-step dirs.
Every GPU touch is under bin/gpu-run (both cards); timed parts hold bin/cpu-run --quiet outside the lease. Port 18150
only; 18080/18124 are touched only by cutover and rollback. No prompt or response text is logged beyond gate summaries.

## Order and budget

| # | Step | Owner | Kind | GPU time | PASS gate | Rollback if it fails |
|---|---|---|---|---|---|---|
| 0 | preflight (CPU) | INT | hard | 0 | frozen binary sha == env, worktree clean at TF_COMMIT, no @FREEZE, steps file valid, fixture, :18150 free | fix and rerun; Strata untouched |
| 1 | stop Strata | owner/INT | - | 0.3 min | both card leases free within 120 s | systemctl --user start hotschmoe-dd |
| G1 | regress: int-rf2 config quick bench on the frozen binary | INT | hard + soft | 3 min | HARD: health pre/post HEALTHY (cards 0 1), p0 128-token greedy hash f8caa42cc5b4 bitwise == int-rf2, every C-level stream == solo p0; SOFT: C1/C2/C4/C8/PF2k >= 0.95 x int-rf2 (109.5/139.5/168.0/209.0/1560) | runner restarts Strata; regression bisect on CPU |
| G2.0 | health pre + production server load on :18150 (tf-dd-prod.env: 200000 x 4) | INT | hard | 2-3 min | xpu-health HEALTHY; /v1/models within 600 s; load line recorded (GiB/GPU, load s) | session teardown + post-health; Strata restarted |
| P0 | identity | INT | hard | 0.1 min | first id hotschmoe-dd, aliases listed, context_length 200000, WIN_EXPECT_CAPS (vision, greedy_only=false, penalties) | as G2.0 |
| P1 | quick serve bench at the production config | INT | soft | 1 min | completes (oracle recorded or == TF_PROD_EXPECTED, concurrency == solo); C1/C2/C4/C8/PF2k >= 0.90 x int-rf2 (C8 queues on 4 lanes) | owner decision; no automatic rollback |
| KV* | 90K prompt + decode, 200K x 4 admission, park/restore bitwise | KV | hard | <= 8 min | KV section | as G2.0 |
| SAMP* | sampling sanity + seeds + MTP acceptance at temp 0.6 | SAMP | hard/soft | <= 3 min | SAMP section | as G2.0 |
| VIS* | image request | VIS | hard | <= 2 min | VIS section | as G2.0 |
| API* | tool call + streaming, reasoning, tool_choice, errors | API | hard | <= 3 min | API section | as G2.0 |
| P9 | dd_functional.py direct to :18150 with image (no key) | INT | hard | 2 min | thinking ok + reasoning chars, tool call get_weather + round trip, 4 concurrent exact, vision blue, long stream >= 512 tokens with code | as G2.0 |
| G2.9 | server log scan + stop + health post | INT | hard | 0.5 min | no DEVICE_LOST/panic/error.X/OOM since launch; xpu-health HEALTHY | xe-reset in teardown; Strata restarted |
| C1 | cutover: start hotschmoe-dd-tensorfold (stops Strata via Conflicts) | INT | hard | 2-3 min | :18080 /v1/models answers with the key within 900 s | gpu_window.sh rollback (automatic) |
| C2 | dd_functional.py through :18080 with the key and image fixture | INT | hard | 2 min | same as P9 through the frontdoor | gpu_window.sh rollback (automatic) |
| C3 | enable TF unit, disable Strata unit, soak monitor -> --backend tf (drop-in backend-tf.conf; rollback removes it) | INT | - | 0 | units as stated | gpu_window.sh rollback |

Estimated GPU time: G1 3 + G2 load 3 + P0/P1 1.5 + KV 8 + SAMP 3 + VIS 2 + API 3 + P9 2 + teardown 1 + cutover 5 = about 31 min.

## INT

Frozen build (CPU, before the window): `tensorfold/flashnext/dd/build_dd.sh <b70/dd-serve head>`, then put TF_COMMIT,
TF_BIN and TF_BIN_SHA256 into tf-dd-prod.env and run preflight. The window tests exactly the binary the unit serves.

- G1 regress -- `gpu_window.sh regress` (inside window). Runs the unchanged tools/flashnext_serve_bench.sh from the
  frozen worktree with S_SKIP_BUILD=1, FN_BENCH_EXPECTED = workers/S/expected-current.json, in a clean environment
  (no XPU_FN_KV etc.), so it is the int-rf2 config: ctx 16384, --parallel auto, greedy, thinking off. 153 s at int-rf2.
  PASS: rc 0 (bench asserts oracle_equal and stream == solo; outer asserts HEALTHY pre/post); speed soft gate via
  gpu_window_checks.py compare (>= 0.95 x int-rf2).
- G2 prod -- `gpu_window.sh prod` (inside window): cpu-run --quiet + gpu-run (both cards) + timeout 2400 around one
  session: xpu-health, dd_launch_tf.sh with TF_PORT=18150 and tf-dd-prod.env, then every line of gpu_window_steps.tsv
  in order (each under its own timeout <= 180 s), log scan, stop, xpu-health (xe-reset first if the log shows a device
  loss). A HARD step failure ends the session at once.
- P0 identity, P1 bench, P9 functional: lines in gpu_window_steps.tsv. TF_PROD_EXPECTED (optional) is the KV worker's
  expected p0 token file for the production KV mode; without it P1 records p0 and checks concurrency == solo only.
- CPU self-test of the runner (mock server, no GPU): /mnt/vm_8tb/b70/results/tf-flashnext/workers/INT/selftest/run.sh
  -> SELFTEST PASS (hard failure stops the loop, soft failure recorded, teardown + post-health always run).

## KV

Owner: worker KV (branch b70/dd-kv on top of b70/dd-serve; merges b70/fn-rt-evict 2c5d885). Production flags for
tf-dd-prod.env (all default-off in the engine):

    export XPU_FN_KV=int8 XPU_FN_DESC=1 XPU_FN_KV_PAGES=auto XPU_FN_EVICT=1 XPU_FN_EVICT_SLOTS=auto XPU_FN_MIRROR=1
    export XPU_CHUNK=512 XPU_CHUNK_SOLO=512   # chunk route-around (below); 64/64 is the fallback pair

- Descriptor windows (<= 16 rows) now run over the shared int8 page arena (fn_window_pages.cl); there is no per-slot
  bf16 context plane, so context is bounded by the arena, not by 16 x context x 17 KB. Prompt chunks > 16 rows use
  the legacy paged pass. XPU_FN_KV_PAGES=auto sizes the arena from measured free VRAM after weights and activations
  minus a reserve for 24 slabs + tables + 1.25 GiB margin; the load prints one 'paged KV budget' line. CPU model at
  the int-rf2 free figure: about 946 pages = 484K tokens = 5 x 90K or 2 x 200K resident; streams beyond that park
  to host RAM (EVICT_SLOTS=auto = full-context fits, so 4 logical streams on 2 physical slots at 200000).
- Chunk route-around: int8 512-row vs 64-row prefill is not bitwise (B/kv8: max logit diff 0.0625, argmax equal;
  bf16page is bitwise). Root cause unproven. Equal XPU_CHUNK and XPU_CHUNK_SOLO make solo and busy prompts take
  identical pass shapes, so every concurrency == solo gate stays bitwise; 512/512 keeps the prefill rate.
- TF_PROD_EXPECTED: cannot be made on the CPU. In the first window P1 runs in record mode and writes
  expected-prod.json in its step dir; set TF_PROD_EXPECTED to that file for every later window.

Steps (TAB-separated lines for gpu_window_steps.tsv, after P1-bench; KV_OUT=$TF_WIN shares files across steps):

| id | kind | timeout | command | PASS gate |
|---|---|---|---|---|
| KV0-budget | hard | 10 | python3 $TF_HERE/kv_gate.py budget $TF_SERVER_LOG | source auto, resident_90k >= 4, resident_200k >= 2 |
| KV1-long90k | hard | 180 | KV_OUT=$TF_WIN python3 $TF_HERE/kv_gate.py long $TF_URL 90000 256 kv90k | HTTP 200, prompt_tokens >= 81000, needle recalled, 256 tokens or stop; TTFT/prefill/decode recorded (soft: TTFT <= 90 s) |
| KV2-solo40k | hard | 120 | KV_OUT=$TF_WIN python3 $TF_HERE/kv_gate.py long $TF_URL 40000 128 kv40k | as KV1 at 40000; ids saved to $TF_WIN/kv40k.json |
| KV3-conc4 | hard | 180 | python3 $TF_HERE/kv_gate.py conc $TF_URL 40000 4 128 $TF_WIN/kv40k.json | 4 distinct 40K prompts admitted and complete, all needles recalled, stream 0 ids == KV2 bitwise (parked/restored on 2 physical slots) |
| KV4-parks | hard | 10 | python3 $TF_HERE/kv_gate.py parks $TF_SERVER_LOG 1 | >= 1 'eviction park' and >= 1 'eviction restore' since launch, no eviction errors (park/restore bitwise is KV3 + P1 C4/C8 == solo) |
| KV5-long180k | soft | 180 | KV_OUT=$TF_WIN python3 $TF_HERE/kv_gate.py long $TF_URL 180000 32 kv180k | prompt_tokens >= 162000, needle recalled, 32 tokens; TTFT recorded |

GPU time about 9 min (KV1 ~70 s, KV2 ~35 s, KV3 ~120 s, KV5 ~150 s). Fail of KV0..KV4 blocks cutover (rollback as
G2.0). Fallbacks without a rebuild: XPU_CHUNK=64 XPU_CHUNK_SOLO=64 if P1 C4 != solo; XPU_FN_KV_PAGES=<n> if KV0
reports too few pages; unset XPU_FN_EVICT* (then 4 x 200K cannot all be resident) if KV4 shows park errors.
Not in this window (needs a second server config): 512 vs 64 int8 first-divergence diagnostic (per-layer router,
pre-append K/V, codes, attention at the 4,800-row boundary; B/kv8 NOTES).

## SAMP

Branch b70/fn-samp (worker SAMP), head 28030ff0 on b70/dd-serve 1a5a57d0. Sampling is on by request: temperature
> 0 (or nonzero penalties) takes Strata 0.2.16's chain on the host over each rank's head-shard top 64 (whole shard
rows when penalties apply); temperature 0 without penalties keeps the device argmax, bitwise unchanged. MTP stays
lossless: draws are keyed per position and drafts are accepted by exact match (Strata's own rule with
STRATA_SPEC_PROB off, its production default). The engine publishes greedy_only=false, penalties=true and
repetition_penalty=true. Requests map temperature/top_p/top_k/min_p/seed/presence/frequency plus
repetition_penalty (alias repeat_penalty) and penalty_last_n (alias repeat_last_n, default 64) like Strata.

Production env (INT, tf-dd-prod.env): keep TF_TEMPERATURE empty (the server then applies generation_config.json:
temperature 1.0, top_p 0.95, top_k 20, Strata's thinking set) and add the instruct card set for thinking-off chats
(temperature 0.7, top_p 0.8, top_k 20, presence_penalty 1.5, Strata's MODEL_CARD_SAMPLING):

    TF_EXTRA_ARGS="--instruct-sampling card"     # append to whatever KV/VIS/API put there

The clamp proxy (tf_clamp_proxy.py, TF_LISTEN=unix) is no longer needed for sampled clients.

Step lines for gpu_window_steps.tsv (INT; TAB-separated, same server as the other prod steps):

    S1-samp	SAMP	hard	150	SAMP_P0_EXPECTED=${TF_SAMP_P0:-} python3 $TF_HERE/samp_gate.py $TF_URL $TF_MODEL > samp.log 2>&1; rc=$?; grep '^GATE' samp.log; exit $rc

Gates printed by dd/samp_gate.py (no prompt/response text; hashes and counts only):

| Gate | Hard | Pass |
|---|---|---|
| samp-caps | yes | /v1/models capabilities: greedy_only false, penalties true, repetition_penalty true |
| samp-s0-greedy-p0-bitwise | yes (only with SAMP_P0_EXPECTED, int-rf2 config) | p0 greedy 128 tokens == workers/S/expected-current.json |
| samp-s0b-topk1-p0-equals-greedy | yes (same) | p0 at T 0.6 top_k 1 == p0 greedy (shard top-64 merge exact over 128 rows) |
| samp-s1-greedy-repeat | yes | two greedy chats identical |
| samp-s2-topk1-equals-greedy | yes | T 0.6 top_k 1 chat == greedy chat |
| samp-s3-seed-repeat | yes | T 0.6 top_p 0.95 top_k 20 seed 1234 twice identical |
| samp-s4-t0.6-coherent-mtp | yes | that reply coherent (>= 20 words, ASCII, on topic, no 4-gram > 3x) and MTP drafted > 0; accept rate logged |
| samp-s4b-mtp-accept-floor | soft | MTP accept rate at T 0.6 >= 0.25 |
| samp-s5-penalties-served | yes | presence 1.5 and repetition 1.1 + frequency 0.2 replies coherent |
| samp-s6-seeds-differ | soft | seed 4321 reply differs from seed 1234 |
| samp-s7-drafts-lossless | soft | seed 1234 with "draft": false == with drafts |

Quick gate run by SAMP (one lease <= 5 min, port 18152): /mnt/vm_8tb/b70/results/tf-flashnext/workers/SAMP/samp_session.sh
(the int-rf2 serve of tools/flashnext_serve_bench.sh plus --instruct-sampling card, SAMP_P0_EXPECTED set). Result:
2026-11-21 19:18 UTC, 124 s lease, binary sha ddefd8fc (ReleaseFast): all 12 gates PASS (s0 p0 greedy 128 tokens
bitwise == expected-current; s0b top_k 1 == greedy; T 0.6 seed repeat; drafts on == off; MTP accept 0.419 at T 0.6,
63 tokens at 60.3 tok/s incl. TTFT vs greedy 71 tokens in 0.75 s), server log clean, xpu-health HEALTHY pre/post.
Evidence: /mnt/vm_8tb/b70/results/tf-flashnext/workers/SAMP/g1-28030ff0/.

## VIS

Branch b70/dd-vis head 2b552f0c (worker VIS; includes b70/dd-kv bef2f1a8). Quick gate PASSED 19:42 UTC in desc+paged int8 ctx 65536 (also classic, int8). Behaviour is off unless the server gets `--vision` and the TF_VISION_* env below
(sidecar = Strata's own strata-vision CPU binary + mmproj, the same one hotschmoe-dd-0216.json runs, 8 threads, 300
tokens an image). Production env lines for tf-dd-prod.env (INT):

    TF_EXTRA_ARGS="--vision"     # append to whatever API/KV put there
    export TF_VISION_EXE=/mnt/vm_8tb/b70/build/strata-ab/build-vision-cpu/bin/strata-vision
    export TF_VISION_MMPROJ=/mnt/vm_8tb/github/b70_ai_things/models/files/qwen3.8-flash-next/ista-gsq-rco-iq3-xxs/mmproj-Qwen3.8-Flash-Next-BF16.gguf
    export TF_VISION_MODEL=$FN_GGUF
    export TF_VISION_LIB=/mnt/vm_8tb/b70/build/tensorfold/vision-lib   # libgomp.so.1 (from strata-sycl-dev:020; the host has none)
    export TF_VISION_THREADS=8 TF_VISION_MAX_TOKENS=300
    WIN_EXPECT_CAPS: add vision=true

Startup log line to expect: `flashnext: --vision on: strata-vision ready in N s`. A `--vision off` line means text only
(P0 then fails on vision=true). Per image: `flashnext vision: image N tokens (WxH) encoded in M ms`.

Steps (gate script tensorfold/flashnext/dd/vis_gate.py; INT transcribes into gpu_window_steps.tsv after P1, before P9):

    VIS-T0	VIS	hard	60	python3 $TF_HERE/vis_gate.py text $TF_URL vis_t0.json
    VIS-I	VIS	hard	120	python3 $TF_HERE/vis_gate.py image $TF_URL $TF_FIXTURE $TF_SERVER_LOG vis_img.json
    VIS-T1	VIS	hard	60	python3 $TF_HERE/vis_gate.py text $TF_URL vis_t1.json && python3 $TF_HERE/vis_gate.py same vis_t0.json vis_t1.json

| Step | Kind | GPU time | PASS gate |
|---|---|---|---|
| VIS-T0 | hard | 0.2 min | fixed text prompt, greedy, thinking off, 128 tokens: completes; hash recorded |
| VIS-I | hard | 0.5 min | the fixture answers "blue" twice (second = encoder cache hit), same word; two encoder log lines with tokens == WxH; prompt_tokens > image tokens; TTFT and encoder ms recorded (soft: TTFT < 3 s on the 224x224 fixture) |
| VIS-T1 | hard | 0.2 min | the same text prompt after the image requests: hash and token count == VIS-T0 (image handling leaves text bit-identical) |

Rollback: as G2.0 (a VIS failure ends the session; cutover can go text-only by dropping --vision).

## API

### API worker result (branch b70/dd-api, head 32b27d57 on b70/dd-serve 372bf1b2)

Quick GPU gate PASSED 5/5 on port 18154 (136 s lease incl. 74 s load; xpu-health HEALTHY): tools/api results in workers/API/gpu2.
Needs two env exports in tf-dd-prod.env (INT): `export TF_FIT_MAX_TOKENS=1 TF_STRUCTURED_FALLBACK=1` (both default off in the binary).
Also drop `--temperature 0` once SAMP lands (Strata applies the model-card defaults T=1.0 top_p=0.95 top_k=20 when a client sends none).
libtfgrammar: not needed (the XPU lane backend has no masked first-token path, so Info.structures stays false; the fallback above
replaces it). To build it anyway: git clone --branch v0.2.8 --recurse-submodules https://github.com/mlc-ai/xgrammar DIR; zig build grammar -Dxgrammar=DIR.

Steps (TAB-separated lines for gpu_window_steps.tsv; each prints one GATE line; GPU time about 100 s total):

| id | kind | timeout | command | PASS gate |
|---|---|---|---|---|
| API1-reasoning | hard | 100 | python3 $TF_HERE/api_gate.py reasoning $TF_URL | thinking on by default: reasoning_content non-stream and stream (before content), answer 205, usage chunk, [DONE], finish stop, cached_tokens key |
| API2-tools | hard | 120 | python3 $TF_HERE/api_gate.py tools $TF_URL | streamed write_file call: name first, >= 3 argument deltas, 30/30 lines, finish tool_calls, max SSE gap < 20 s; non-stream get_weather call id call_*; tool_choice none -> text, no call |
| API3-forced | hard | 100 | python3 $TF_HERE/api_gate.py forced $TF_URL | tool_choice required and named produce tool_calls (also streamed); response_format json_schema valid JSON object in >= 2 of 3 |
| API4-budget | soft | 100 | python3 $TF_HERE/api_gate.py budget $TF_URL | reasoning_budget_tokens / thinking_budget_tokens / thinking_budget = 48 each give < 1200 reasoning chars and an answer; reasoning_effort none has no reasoning_content |
| API5-edge | hard | 120 | python3 $TF_HERE/api_gate.py edge $TF_URL | models first id hotschmoe-dd + strata-0216 alias + context_length >= 100000, /health 200, over-window prompt 400 naming the context window, max_tokens = window clamps (200, stop), finish length/stop, usage keys + cached_tokens, 8 concurrent requests on 4 lanes all 200 and exact |

Observed on the real engine: reasoning 342 chars at 205; write_file 36 argument deltas, max gap 0.05 s; budget 48 -> 86 reasoning chars;
forced and named calls ok; response_format 3/3; queue8 max 1.5 s. Not covered here: cached_tokens > 0 needs a prompt cache (KV item;
TF_PROMPT_CACHE_GIB=0 in the DD env today); Strata prompt-cache parity for 58-90K prompts is KV's.
CPU parity (no GPU): tensorfold/flashnext/api_parity/run_parity.sh -- 77 cases against Strata's real server.py (mock engine), 0 gaps
with the two flags (9 without them), 2 SAMP cases pending (temperature, penalties); accepted differences listed in parity.py KNOWN.
