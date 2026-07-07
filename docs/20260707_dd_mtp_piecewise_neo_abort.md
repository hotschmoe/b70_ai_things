# DD crash 2026-07-07: NEO linear_stream.h:84 abort (MTP-verify x piecewise cudagraph)

Status doc. config -> command -> result -> verdict. Newest evidence appended.

## Summary of the incident

- Container `b70_daily_0` (vllm/qwen36-27b-w8a8 TP=2, v0.24.0, PIECEWISE capture,
  MTP spec=3, prefix-cache on, vision on) died `07-07 00:39:26`.
- Both TP workers aborted simultaneously with:
  ```
  Abort was called at 84 line in file:
  ../../neo/shared/source/command_stream/linear_stream.h
  ```
  -> `Worker proc VllmWorker-0 died unexpectedly` -> EngineCore fatal ->
  clean vLLM shutdown (container Exited(0); the `RuntimeError: cancelled` /
  `EngineDeadError` traces below the abort are downstream shm-broadcast fallout,
  NOT the cause).
- dmesg CLEAN (no xe/GuC/DEVICE_LOST/reset). This is a pure host-side software
  abort in the Intel Compute Runtime (NEO) command-stream builder, NOT the TP=2
  hardware wedge and NOT a DEVICE_LOST.

## State at the moment of the abort (from dump_input)

- Single request, decode step. `num_computed_tokens=50512` (~50K ctx),
  `num_output_tokens=1202`.
- `num_scheduled_tokens=4` = 1 + 3 MTP spec tokens.
  `scheduled_spec_decode_tokens=[-1,-1,-1]` (MTP verify path active).
- `num_common_prefix_blocks=[0,0,0,61]`.
- `max_seq_len=200000`, `kv_cache_dtype=auto` (bf16 KV), `cudagraph_mode=PIECEWISE`,
  `cudagraph_capture_sizes=[1,2,4,6,8]`, `enforce_eager=False`, spec method=mtp
  num_spec_tokens=3, TP=2, prefix caching on.

## Q1: was the session over the 128K ctx limit?

NO. The crashing request was at ~50K computed tokens, well under 128K. Served
max_seq_len was 200000 (raised from the historical 131072). Context LENGTH of the
request was not the direct trigger. Open hypothesis: raising MAXLEN 131072->200000
changes KV-block/command-buffer sizing even for shorter requests -> must test.

## Confirmed cross-quant

- commit b638297 recorded the SAME crash class on the NVFP4 config ("crashed ~1h
  in ... MTP-in-piecewise-graph"). The revert to W8A8 did NOT escape it. So the
  abort is structural to **MTP-verify x piecewise XPU cudagraph**, not quant-specific.
- memory [[nvfp4-dd-repetition-crash-rootcause]] independently flagged a
  "MTP-verify-in-piecewise-XPU-cudagraph NEO linear_stream.h:84 abort (needs both
  MTP+capture)".

## PRIOR ART IN THIS REPO (this abort was already root-caused once)

The 2026-06-25 w8a8 MTP campaign (JOURNAL ~L4270-4438) already hit and localized
this exact abort. Established then:

- faulthandler (B70_DEBUG=1) caught the ground-truth traceback:
  ```
  torch/xpu/graphs.py:108 in replay
  vllm/compilation/cuda_graph.py:360 in __call__
  vllm/model_executor/models/qwen3_5_mtp.py:418 in forward   (MTP DRAFTER forward)
  vllm/v1/spec_decode/llm_base_proposer.py:642 in propose
  ```
- Root cause (then): NEO `LinearStream::getSpace` overflow during torch.xpu graph
  REPLAY of the MTP-drafter captured forward, ACCUMULATING over ~5000 spec steps
  (~20k cumulative gen tokens / ~20 min) until the command stream can't fit the
  next command -> abort. Silent worker death -> shm_broadcast cancelled -> EngineDead.
- Bisect (v0.23): MTP-on + capture -> CRASH; MTP-off + capture -> STABLE;
  MTP-on + enforce-eager -> STABLE (survived 40m / 37k tokens + full agentic eval).
  PUSH_AR EXONERATED (crashes without it too, just slower). => crash needs BOTH
  MTP AND graph capture; drop either and it is stable.
- Fix picked THEN: enforce-eager + MTP (stable, ~16 t/s) OR MTP-off + capture.

## THE CORRECTION (2026-07-07)

RESEARCH_TODO 11h and the 2026-07-06 NVFP4 bisection concluded the abort was
"specific to the NVFP4 fused-kernel graph" and that "W8A8 runs MTP-verify inside
the captured graph fine." TODAY REFUTES THAT: W8A8 + MTP3 + PIECEWISE hit the
IDENTICAL abort. So:

- The abort is NOT quant-specific. It is the generic MTP-drafter-graph-replay
  command-stream overflow, present on BOTH NVFP4 and W8A8.
- The v0.24.0 "PIECEWISE + MTP restarts=0" validation (2026-07-03) and the
  "W8A8 keeps MTP+graph" claim were UNDER-SOAKED. v0.24.0 raised the crash
  threshold (crash at ~3h of real use instead of ~20 min) but did not remove it.
- "Stable W8A8 keeps MTP at 36-43 t/s" (used to justify the 2026-07-06 revert)
  described the CRASHING config. The real stable W8A8 options are enforce-eager+MTP
  (~16 t/s) or capture+MTP-off. The DD decision rests on a false premise and must
  be revisited.

## Two candidate mechanisms on v0.24.0 (to disambiguate on GPU)

The old rate (~20k tokens -> crash) does NOT fit today (server up ~3h). Either the
leak rate dropped a lot, or a second mechanism is in play:

- M1 ACCUMULATION: per-replay L0 command/event handle not reclaimed -> LinearStream
  grows monotonically across spec steps -> overflow. (2026-06-25 mechanism.)
- M2 CONTEXT-GATED SINGLE STEP: a single captured decode step at long context
  (~50K KV) submits a command buffer proportional to KV blocks; past a size it
  overflows the linear stream on ITS OWN, no accumulation needed. Fits today's
  crash (single request, 50K ctx, num_common_prefix_blocks=[..,61]) AND the
  operator's Q1 intuition that 128K-maxlen serving did not crash.

SHARP FIRST EXPERIMENT: fresh serve (DD config, B70_DEBUG=2 leak checker), fire ONE
~50K-token forced-decode (ignore_eos) MTP request immediately.
 - crashes on first long request        -> M2 (context-gated single-step overflow)
 - survives, only dies after long soak   -> M1 (accumulation leak)
The leak-checker (ZEL_ENABLE_BASIC_LEAK_CHECKER) names the accumulating handle for M1.

## What linear_stream.h:84 actually is (NEO source, researched 2026-07-07)

`LinearStream::getSpace(size)` in intel/compute-runtime aborts via
`UNRECOVERABLE_IF(sizeUsed + size > maxAvailableSpace)` (and a sibling
`+ batchBufferEndSize` check in `ensureContinuousSpace`). It is a FIXED-CAP
command-buffer overflow: it fires when a single indivisible encoded command-stream
write cannot fit the current command buffer AND the path cannot chain to a fresh
one (immediate command list / CSR ring, or a single command list larger than one
whole command-buffer allocation). maxAvailableSpace is the per-buffer byte cap of
the command-buffer graphics allocation -- NOT heap/KV memory. So this is about how
many BYTES of GPU commands one submission encodes (kernel launches, walkers,
barriers, semaphore/pipe-control waits), not model memory.

Consequence: a captured MTP-verify decode graph whose encoded command list exceeds
the (~2MB-class) default command-buffer cap, on a path that can't chain, aborts.
Longer context => more KV-block attention commands => more bytes in the SAME
captured-size graph => tips over the cap. This unifies M1 and M2: the real variable
is encoded command-stream BYTES per submission.

Novelty: the agent found NO public issue tying linear_stream.h:84 to graph
capture / spec-decode. A characterized repro + fix would be a genuine upstream
contribution (compute-runtime / torch-xpu-ops / vLLM-XPU).

### TOP CANDIDATE FIX (keeps MTP + capture = fast): enlarge the command buffer
NEO debug key, no rebuild, env-only:
```
NEOReadDebugKeys=1 OverrideCmdListCmdBufferSizeInKb=4096   # then 8192 / 16384
```
If the captured MTP-verify command list merely exceeds the default cap, a bigger cap
holds it -> stable WITH capture AND MTP (the best case RESEARCH_TODO 11h wanted).
Only helps the growable-container path; if it's a non-chainable CSR-ring stream it
won't, and we fall back to bounding command bytes (smaller CAPSIZES / fewer spec
tokens / shorter captured decode) or the proven eager+MTP.

Ranked fix ladder:
1. OverrideCmdListCmdBufferSizeInKb=big (env, keeps MTP+capture) -- TRY FIRST.
2. Reduce encoded bytes per graph: fewer spec tokens, smaller CAPSIZES, cap
   captured decode ctx.
3. enforce-eager + MTP (PROVEN stable, ~16 t/s) -- reliable fallback.
4. capture + MTP-off (fast decode, loses MTP acceptance).

## MECHANISM LOCALIZED (binary disasm, 2026-07-07)

Disassembled `at::xpu::XPUGraphImpl::replay()` in libtorch_xpu.so (torch 2.12+xpu):
replay submits the executable SYCL `command_graph` via `submit_with_event` onto the
stream's in-order queue and NEVER synchronizes (no queue.wait / synchronize / event
destroy in the function body). Each replay therefore (a) appends a graph-exec command
into the queue's NEO immediate command list and (b) allocates a sycl::event / L0
event-pool handle that is dropped un-waited. NEO reclaims both only on a full queue
synchronize.

- Normal single-token decode reclaims implicitly: the sampled-token D2H copy forces a
  per-step queue sync.
- MTP breaks it: LLMBaseProposer.propose (llm_base_proposer.py:613-687) fires
  (num_spec_tokens-1) x (num piecewise sub-graphs) replays per outer decode step,
  keeps all draft tokens on-GPU (torch.stack at the end), and does NOT host-sync
  between draft steps. Across thousands of steps the un-reclaimed commands/events pile
  into the queue's NEO command list until LinearStream::getSpace can't fit the next
  command -> linear_stream.h:84 abort. Needs BOTH MTP (the sync-free replay loop) AND
  capture (the graph-exec commands). Matches every prior observation.
- Call path: qwen3_5_mtp.py forward (@support_torch_compile -> N piecewise
  CUDAGraphWrapper sub-graphs) -> cuda_graph.py:360 replay (capture-once, replay-by-
  descriptor; NO per-call recapture) -> torch/xpu/graphs.py:105 replay -> C++
  XPUGraphImpl::replay submit_with_event. CUDAGraph is torch.xpu.XPUGraph (rebound in
  xpu_model_runner.py:52-54).

### Candidate fixes (all rebuild-free; ranked)
1. FIX-SYNC (principled, keeps MTP+capture): one torch.xpu.synchronize() per propose()
   call -> restores the per-step reclaim cadence normal decode already has. Negligible
   cost (one sync per decode step). Patch point: wrap LLMBaseProposer.propose via
   sitecustomize. UPSTREAM-WORTHY (torch XPUGraphImpl::replay should use
   submit_without_event or drain per replay; the vLLM-side sync is the portable fix).
2. FIX-EVT (cheapest if it works): UR/L0 event-cleanup env so discarded events are
   reclaimed without a full sync (e.g. UR_L0_REUSE_DISCARDED_EVENTS=1, or the immediate-
   command-list event-cleanup threshold). Env-only.
3. FIX-BUF (symptom only): NEOReadDebugKeys=1 OverrideCmdListCmdBufferSizeInKb=large ->
   bigger command buffer just DELAYS an unbounded accumulation; not a true fix, but a
   useful mechanism corroborator (if it only postpones the crash, accumulation confirmed).
4. Fallbacks: enforce-eager+MTP (proven) or capture+MTP-off.

## Experiment log

### E1 (2026-07-07): baseline repro at MAXLEN=131072 -- did NOT crash
config -> DD config exactly (MTP3 + PIECEWISE + PUSH_AR + prefix cache + vision),
  MAXLEN=131072, B70_DEBUG=1 (faulthandler). Probe: 6x forced-decode (ignore_eos)
  of ~55K-token prompts, 3000 tokens each = 18k cumulative decode tokens.
command -> vllm/repro_campaign.sh (CFG_LABEL=e1_baseline_ml131k).
result -> 6/6 CLEAN, prompt_tok=55148, comp_tok=3000 each, decode ~10-14 t/s at
  55K ctx. NO abort. Container healthy after.
verdict -> v0.24.0 leak RATE is far lower than v0.23 (old crash ~20k tokens / ~20min;
  here 18k tokens clean). ALSO: a 55K-context request did NOT crash on run0 -> refutes
  a pure "single long-context submission too big" (M2) trigger. It IS accumulation, but
  slow -- the real DD crash took ~3h of production traffic. Two suspects for the gap:
  (a) MAXLEN=200000 (real) vs 131072 (E1); (b) single-stream 18k tokens vs hours of
  CONCURRENT traffic exercising capture sizes 6/8. Q1 (128K maxlen) partly answered:
  128K maxlen with a 55K request is NOT sufficient to crash on its own; the operator's
  "didn't crash at 128K" memory is consistent with lower cumulative/concurrent load then,
  not a hard maxlen threshold. NEXT: concurrent soak (E1c) to force it fast.

### E1c (2026-07-07): concurrent soak on the live MAXLEN=131072 container -- RUNNING
config -> reuse E1's healthy container. 6 concurrent forced-decode streams, ~8K ctx
  (fast decode = max replay rate), 4000 tok/req, ceiling 900k tok / 45 min.
command -> vllm/soak_concurrent.py e1c_concurrent_ml131k.
result -> *** CRASH (HTTP500) at tok=96000 t=1453s (~24min) reqs=24. faulthandler:
  "Abort was called at 84 line ... linear_stream.h" + "torch/xpu/graphs.py line 107 in
  replay" + EngineDeadError. REPRODUCED with the exact signature.
verdict -> RELIABLE REPRO. Threshold on v0.24.0 ~= 96k decode tokens under 6-way
  concurrency (~24min). Single-stream 18k (E1) was clean -> CONCURRENCY is the accelerant
  (saturates GPU = max replays/sec, and exercises capture sizes 6/8). MAXLEN=131072 ->
  Q1 CLOSED: maxlen is NOT the trigger; the operator's "no crash at 128K" was lower load.
  Baseline crash threshold for fix A/B = survive >> 96k tokens (target ~220k = 2.3x).
